#!/usr/bin/env python3
"""
CLI tool to generate a video using FAL AI based on a prompt and a base image.

- Loads environment variables from .env
- Accepts either an --image-url or a local --base-image path (uploads to GCS)
- Saves the generated MP4 to --output

Requirements:
- FAL_API_KEY in environment (or .env)
- If using --base-image, set GCS bucket env used by upload_image_to_gcs

Usage examples:
  python -m src.cli.generate_video \
    --prompt "Camera slowly pans across the beach at sunset" \
    --image-url "https://example.com/base.jpg" \
    --duration 6 \
    --output output.mp4

  python -m src.cli.generate_video \
    --prompt "Gentle parallax move through a forest" \
    --base-image ./base.jpg \
    --output forest.mp4
"""

import os
import sys
import argparse
import asyncio
import tempfile
import uuid
import logging
from typing import Optional

from dotenv import load_dotenv

# Import core generators and helpers
try:
    from src.core.facebook_image_post import (
        generate_video_with_fal,
        upload_image_to_gcs,
    )
except Exception as e:
    print("Failed to import core modules. Ensure you're running from project root and PYTHONPATH is set.")
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a video from a prompt and base image using FAL AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt describing desired video motion/animation",
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--image-url",
        help="Publicly accessible URL of the base image",
    )
    src_group.add_argument(
        "--base-image",
        help="Local path to a base image to upload to GCS before generation",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=8,
        help="Video duration in seconds (1-10)",
    )
    parser.add_argument(
        "--output",
        default="output_video.mp4",
        help="Path to save the generated video",
    )
    return parser.parse_args()


async def _maybe_upload_base_image(path: str) -> Optional[str]:
    """Upload local image file to GCS and return a public URL.

    Uses the project's upload_image_to_gcs helper, which reads bucket from env.
    """
    if not os.path.isfile(path):
        logging.error(f"Base image not found: {path}")
        return None

    # Destination name in GCS
    dest_name = f"video_base/{uuid.uuid4().hex}.jpg"

    try:
        with open(path, "rb") as f:
            # upload_image_to_gcs(file_like, bucket_name, dest_blob)
            bucket = os.getenv("GCS_BUCKET_NAME")
            if not bucket:
                logging.error("GCS_BUCKET_NAME is not set in environment.")
                return None
            url = upload_image_to_gcs(f, bucket, dest_name)
            if not url:
                logging.error("Failed to upload base image to GCS")
                return None
            logging.info(f"Uploaded base image to GCS: {url}")
            return url
    except Exception as e:
        logging.exception(f"Error uploading base image: {e}")
        return None


async def run() -> int:
    # Load .env early
    load_dotenv()

    # Minimal logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = parse_args()

    # Resolve image_url
    image_url: Optional[str] = args.image_url
    if not image_url and args.base_image:
        image_url = await _maybe_upload_base_image(args.base_image)
        if not image_url:
            return 1

    # Call generator
    logging.info(f"Generating video (duration={args.duration}s)…")
    video_bytes = await generate_video_with_fal(args.prompt, image_url, args.duration)
    if not video_bytes:
        logging.error("Video generation failed")
        return 1

    # Save output
    out_path = args.output
    try:
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        logging.info(f"Saved video to {out_path}")
    except Exception as e:
        logging.exception(f"Failed to save video: {e}")
        return 1

    return 0


def main() -> None:
    try:
        rc = asyncio.run(run())
        sys.exit(rc)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
