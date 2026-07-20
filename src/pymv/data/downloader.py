import logging
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url: str, destination, timeout: int = 30) -> Path:
    """Download ``url`` to ``destination``, reusing a cached copy if present.

    The file is written to a temporary ``.part`` path and moved into
    place only after the download completes, so an interrupted or
    failed download never leaves a corrupt file in the cache.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        logger.debug("Using cached file: %s", destination)
        return destination

    logger.info("Downloading dataset to: %s", destination)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
        ) as progress:
            for chunk in response.iter_content(8192):
                f.write(chunk)
                progress.update(len(chunk))

        tmp_path.replace(destination)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Download completed.")
    return destination
