import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_RETRY_WAIT = 5  # seconds, multiplied by the attempt number


def download_file(
    url: str,
    destination,
    timeout: int = 60,
    retries: int = 3,
) -> Path:
    """Download ``url`` to ``destination``, reusing a cached copy if present.

    The body streams into a temporary ``.part`` file that is moved into
    place only once the transfer is complete, so an interrupted download
    never leaves a corrupt file in the cache.  A dropped connection is
    retried, and each retry resumes from the bytes already on disk when
    the server supports range requests.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        logger.debug("Using cached file: %s", destination)
        return destination

    logger.info("Downloading dataset to: %s", destination)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            _stream_to(url, tmp_path, timeout)
            tmp_path.replace(destination)
            logger.info("Download completed.")
            return destination
        except (requests.RequestException, OSError) as exc:
            status = (exc.response.status_code
                      if isinstance(exc, requests.HTTPError) and exc.response is not None else None)
            if status in {400, 401, 403, 404}:
                tmp_path.unlink(missing_ok=True)
                message = ("The dataset file is not available yet; its upload may still be pending"
                           if status == 404 else "The dataset download is not accessible")
                raise requests.HTTPError(f"{message} (HTTP {status}): {url}", response=exc.response) from exc
            if attempt == retries:
                tmp_path.unlink(missing_ok=True)
                raise
            wait = _RETRY_WAIT * attempt
            logger.warning(
                "Download failed (%s: %s) — retrying in %ds (attempt %d/%d).",
                type(exc).__name__, exc, wait, attempt + 1, retries,
            )
            time.sleep(wait)


def _stream_to(url: str, tmp_path: Path, timeout: int) -> None:
    """Stream *url* into *tmp_path*, resuming from any partial data."""
    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()

    # 206 means the range was honoured; anything else restarts from byte 0.
    if resume_from and response.status_code != 206:
        resume_from = 0

    total = resume_from + int(response.headers.get("content-length", 0))

    with open(tmp_path, "ab" if resume_from else "wb") as f, tqdm(
        total=total or None,
        initial=resume_from,
        unit="B",
        unit_scale=True,
        desc="Downloading",
    ) as progress:
        for chunk in response.iter_content(65536):
            f.write(chunk)
            progress.update(len(chunk))

    written = tmp_path.stat().st_size
    if total and written != total:
        raise OSError(f"incomplete download: got {written} of {total} bytes")
