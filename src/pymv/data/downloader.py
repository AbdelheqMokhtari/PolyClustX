import logging
import os

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url, destination):

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if os.path.exists(destination):

        logger.debug("Dataset already downloaded.")
        logger.debug("Using cached file: %s", destination)

        return destination

    logger.info("Downloading dataset...")
    logger.info("Saving to: %s", destination)

    response = requests.get(url, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    with open(destination, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc="Downloading",
    ) as progress:

        for data in response.iter_content(block_size):

            f.write(data)
            progress.update(len(data))

    logger.info("Download completed.")

    return destination
