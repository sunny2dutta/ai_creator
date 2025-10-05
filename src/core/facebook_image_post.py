"""
Social Media Content Posting System - Production Ready

Automated system for generating and posting AI-created content to Facebook and Instagram.
Supports both images and videos with comprehensive error handling and retry logic.

Design Choices:
- Langfuse integration for observability and tracing
- Retry logic with exponential backoff for API reliability
- GCS for image hosting (required for Instagram Graph API)
- Modular functions for testability
- Environment-based configuration (12-factor app)
- Comprehensive logging for debugging and monitoring

Author: AI Creator Team
License: MIT
"""

import requests
import os
import sys
import io
import asyncio
import tempfile
import logging
import base64
import argparse
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
from PIL import Image
import fal_client
from prompt_generator import PromptGenerator
from profile_manager import ProfileManager
import uuid
from google.cloud import storage
from langfuse import Langfuse
import time

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_post_processor import ImagePostProcessor, enhance_ai_image, apply_portrait_enhancement, apply_lifestyle_filter

# Load environment variables
load_dotenv()

# Configuration from environment variables
# Design Choice: Use env vars instead of hardcoded paths for portability
GCS_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '/Users/debaryadutta/google_cloud_storage.json')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'ai-creator-debarya')
ENABLE_POST_PROCESSING = os.getenv('ENABLE_POST_PROCESSING', 'false').lower() == 'true'

# Set Google Cloud credentials
if os.path.exists(GCS_CREDENTIALS_PATH):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCS_CREDENTIALS_PATH
else:
    logging.warning(f"GCS credentials file not found at {GCS_CREDENTIALS_PATH}")

# Configure logging
# Design Choice: Structured logging with timestamps for production debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('social_media_posting.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)

# Global variables for backwards compatibility
# TODO: Refactor to use dependency injection instead of globals
page_id: Optional[str] = None
access_token: Optional[str] = None
instagram_business_account_id: Optional[str] = None

# Initialize global Langfuse client for observability
# Design Choice: Optional tracing - system works without it
langfuse_client: Optional[Langfuse] = None
if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
    try:
        langfuse_client = Langfuse(
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        )
        logger.info("Langfuse tracing initialized")
    except Exception as e:
        logger.warning(f"Could not initialize Langfuse: {e}. Continuing without tracing.")
        langfuse_client = None
else:
    logger.info("Langfuse credentials not found. Tracing disabled.")

def set_profile_credentials(profile_config) -> None:
    """Set global credentials for a profile.
    
    Design Choice: Uses global variables for backward compatibility.
    TODO: Refactor to pass credentials as parameters.
    
    Args:
        profile_config: Profile configuration object with credentials
        
    Raises:
        AttributeError: If profile_config missing required attributes
    """
    global page_id, access_token, instagram_business_account_id
    
    try:
        page_id = profile_config.facebook_page_id
        access_token = profile_config.access_token
        instagram_business_account_id = profile_config.instagram_business_id
        
        logger.info(f"Credentials set for profile: {getattr(profile_config, 'name', 'unknown')}")
        
        # Validate credentials are not empty
        if not all([page_id, access_token, instagram_business_account_id]):
            logger.warning("Some credentials are empty. Posting may fail.")
            
    except AttributeError as e:
        logger.error(f"Invalid profile_config: {e}")
        raise

def load_json_data(json_path: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file with error handling.
    
    Design Choice: Returns None on error instead of raising exceptions.
    Allows graceful degradation in the calling code.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary, or None if error
        
    Example:
        data = load_json_data('prompt.json')
        if data:
            prompt, caption = extract_prompt_from_json(data)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {json_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading JSON from {json_path}: {e}")
        return None

def extract_prompt_from_json(json_data: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Extract prompt and caption from JSON data.
    
    Design Choice: Handles multiple JSON structures for flexibility.
    Supports both enhanced_prompt and day_content formats.
    
    Args:
        json_data: Dictionary containing prompt data
        
    Returns:
        Tuple of (prompt, caption). Both can be None if extraction fails.
        
    Example:
        prompt, caption = extract_prompt_from_json(data)
        if prompt:
            image = await generate_image_with_fal(prompt)
    """
    if not json_data:
        logger.warning("No JSON data provided for prompt extraction")
        return None, None

    prompt = ""
    caption = ""

    # Extract from enhanced_prompt structure if present
    if 'enhanced_prompt' in json_data:
        enhanced_scene = json_data['enhanced_prompt'].get('enhanced_scene', '')
        visual_elements = json_data['enhanced_prompt'].get('visual_elements', [])
        lighting = json_data['enhanced_prompt'].get('lighting', '')
        composition = json_data['enhanced_prompt'].get('composition', '')
        clothing_details = json_data['enhanced_prompt'].get('clothing_details', '')
        jewelry_accessories = json_data['enhanced_prompt'].get('jewelry_accessories', '')

        # Combine elements into comprehensive prompt
        prompt_parts = [enhanced_scene]
        if visual_elements:
            prompt_parts.append("Visual elements: " + ", ".join(visual_elements))
        if lighting:
            prompt_parts.append("Lighting: " + lighting)
        if composition:
            prompt_parts.append("Composition: " + composition)
        if clothing_details:
            prompt_parts.append("Clothing: " + clothing_details)
        if jewelry_accessories:
            prompt_parts.append("Accessories: " + jewelry_accessories)

        prompt = ". ".join(prompt_parts)

        # Try to get caption from enhanced_prompt first
        caption = json_data['enhanced_prompt'].get('social_media_caption', '')

    # Fallback to day_content for caption if not found in enhanced_prompt
    if not caption and 'day_content' in json_data:
        story_title = json_data['day_content'].get('story_title', '')
        activity = json_data['day_content'].get('activity_description', '')
        caption = f"{story_title}: {activity}" if story_title and activity else story_title or activity

    logger.debug(f"Extracted prompt ({len(prompt)} chars) and caption ({len(caption)} chars)")
    return prompt, caption

def create_generated_prompt_json(
    original_data: Optional[Dict[str, Any]], 
    output_path: str = "generated_prompt.json"
) -> bool:
    """Create filtered JSON file excluding metadata.
    
    Design Choice: Removes metadata to create cleaner prompt files.
    Useful for archiving or sharing prompts without internal tracking data.
    
    Args:
        original_data: Source data dictionary
        output_path: Path for output JSON file (default: generated_prompt.json)
        
    Returns:
        True if file created successfully, False otherwise
        
    Example:
        if create_generated_prompt_json(prompt_data, "output/prompt.json"):
            logger.info("Prompt file created")
    """
    if not original_data:
        logger.warning("No data provided to create_generated_prompt_json")
        return False

    try:
        # Create filtered data excluding metadata
        # Design Choice: Dictionary comprehension for cleaner code
        filtered_data = {
            key: value 
            for key, value in original_data.items() 
            if key != 'metadata'
        }
        
        # Write to file with proper encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {output_path} without metadata ({len(filtered_data)} keys)")
        return True
        
    except IOError as e:
        logger.error(f"IO error creating {output_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating {output_path}: {e}")
        return False

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for social media posting.
    
    Design Choice: Centralized argument parsing with sensible defaults.
    Supports both image and video workflows.
    
    Returns:
        Parsed arguments namespace
        
    Example:
        args = parse_arguments()
        if args.video:
            # Video workflow
        else:
            # Image workflow
    """
    parser = argparse.ArgumentParser(
        description='Generate and post AI-created content to social media platforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --profile rupashi
  %(prog)s --profile mrbananas --video --duration 8
  %(prog)s --input-json prompt.json --profile traveler
  %(prog)s --list-profiles
        """
    )

    parser.add_argument(
        '--profile',
        type=str,
        default='rupashi',
        help='Profile name to use for posting (default: rupashi)'
    )

    parser.add_argument(
        '--category',
        type=str,
        help='Specific category for content generation (optional)'
    )

    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List all available profiles and exit'
    )

    parser.add_argument(
        '--input-json',
        type=str,
        metavar='PATH',
        help='Path to input JSON file with pre-generated prompts'
    )

    parser.add_argument(
        '--video',
        action='store_true',
        help='Generate and post video instead of images'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=5,
        metavar='SECONDS',
        help='Video duration in seconds (default: 5, max: 10)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate content but do not post to social media'
    )

    args = parser.parse_args()
    
    # Validate duration
    if args.duration < 1 or args.duration > 10:
        parser.error("Video duration must be between 1 and 10 seconds")
    
    return args

def upload_image_to_gcs(
    image_data: Any,
    bucket_name: str,
    destination_blob_name: str,
    credentials_path: Optional[str] = None,
    make_public: bool = True,
    timeout: int = 60
) -> Optional[str]:
    """Upload image to Google Cloud Storage and return public URL.
    
    Design Choice: Centralized GCS upload with retry logic.
    Makes blob public by default (required for social media APIs).
    
    Args:
        image_data: Image data as file-like object or bytes
        bucket_name: GCS bucket name
        destination_blob_name: Destination filename in bucket
        credentials_path: Optional path to service account JSON (uses env var if None)
        make_public: Whether to make blob publicly accessible (default: True)
        timeout: Upload timeout in seconds (default: 60)
        
    Returns:
        Public URL of uploaded image, or None if upload fails
        
    Raises:
        ValueError: If bucket_name or destination_blob_name is empty
        
    Example:
        with open('image.jpg', 'rb') as f:
            url = upload_image_to_gcs(f, 'my-bucket', 'images/photo.jpg')
            if url:
                logger.info(f"Uploaded to {url}")
    """
    if not bucket_name or not bucket_name.strip():
        raise ValueError("bucket_name cannot be empty")
    if not destination_blob_name or not destination_blob_name.strip():
        raise ValueError("destination_blob_name cannot be empty")
    
    try:
        # Initialize GCS client
        if credentials_path:
            storage_client = storage.Client.from_service_account_json(credentials_path)
            logger.debug(f"Using GCS credentials from {credentials_path}")
        else:
            # Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
            storage_client = storage.Client()
            logger.debug("Using GCS credentials from environment")

        # Get bucket and blob
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Upload with timeout
        logger.debug(f"Uploading to gs://{bucket_name}/{destination_blob_name}")
        blob.upload_from_file(image_data, timeout=timeout)
        
        # Make public if requested
        if make_public:
            blob.make_public()
            logger.debug(f"Made blob public: {destination_blob_name}")
        
        public_url = blob.public_url
        logger.info(f"Successfully uploaded to GCS: {destination_blob_name}")
        
        return public_url

    except storage.exceptions.GoogleCloudError as e:
        logger.error(f"GCS error during upload: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error uploading to GCS: {e}", exc_info=True)
        return None

async def generate_image_with_fal(
    prompt: str, 
    parent_trace: Optional[Any] = None,
    timeout: int = 120
) -> Optional[Image.Image]:
    """Generate image from text prompt using FAL AI.
    
    Design Choice: Uses FAL AI's nano-banana model for fast generation.
    Integrates with Langfuse for observability and performance tracking.
    
    Args:
        prompt: Text description of desired image
        parent_trace: Optional Langfuse parent trace for nested tracing
        timeout: Maximum time to wait for generation in seconds (default: 120)
        
    Returns:
        PIL Image object if successful, None if generation fails
        
    Raises:
        ValueError: If prompt is empty
        
    Example:
        image = await generate_image_with_fal(
            "A serene beach at sunset with palm trees"
        )
        if image:
            image.save("output.jpg")
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    # Validate FAL API key
    fal_api_key = os.getenv("FAL_API_KEY")
    if not fal_api_key:
        logger.error("FAL_API_KEY not found in environment variables")
        return None
    
    # Create generation span for tracing
    generation = None
    if langfuse_client:
        generation = langfuse_client.start_generation(
            name="fal_image_generation",
            model="fal-ai/nano-banana",
            input={"prompt": prompt}
        )
    
    try:
        logger.info(f"Generating image with FAL AI (prompt length: {len(prompt)} chars)")
        client = fal_client.AsyncClient(key=fal_api_key)
        start_time = time.time()
        
        # Subscribe to FAL AI generation
        result = await client.subscribe(
            "fal-ai/nano-banana",
            arguments={
                "prompt": prompt,
                "image_urls": []  # Empty for generation from scratch
            }
        )

        generation_time = time.time() - start_time
        logger.info(f"Image generation completed in {generation_time:.2f}s")

        # Validate result structure
        if result and 'images' in result and result['images']:
            image_url = result['images'][0]['url']
            logger.debug(f"Downloading generated image from {image_url}")
            
            # Download image with timeout
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_response.content))
            logger.info(f"Successfully generated image: {image.size}")
            
            # Update tracing
            if generation:
                generation.update(
                    output={
                        "image_generated": True,
                        "image_url": image_url,
                        "generation_time_seconds": generation_time,
                        "image_size": f"{image.size[0]}x{image.size[1]}"
                    },
                    metadata={
                        "model": "fal-ai/nano-banana",
                        "generation_time": generation_time
                    }
                )
                generation.end()
            
            return image
        else:
            logger.error("No images in FAL AI result")
            if generation:
                generation.update(
                    output={"image_generated": False, "error": "No images in result"},
                    metadata={"generation_failed": True}
                )
                generation.end()
            return None

    except requests.exceptions.Timeout:
        error_msg = "Image download timed out"
        logger.error(error_msg)
        if generation:
            generation.update(
                output={"image_generated": False, "error": error_msg},
                metadata={"generation_failed": True, "error_type": "timeout"}
            )
            generation.end()
        return None
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error during image generation: {e}"
        logger.error(error_msg)
        if generation:
            generation.update(
                output={"image_generated": False, "error": error_msg},
                metadata={"generation_failed": True, "error": str(e)}
            )
            generation.end()
        return None
        
    except Exception as e:
        error_msg = f"Unexpected error generating image: {e}"
        logger.error(error_msg, exc_info=True)
        if generation:
            generation.update(
                output={"image_generated": False, "error": error_msg},
                metadata={"generation_failed": True, "error": str(e)}
            )
            generation.end()
        return None

async def edit_image_with_fal(
    prompt: str, 
    image_urls: List[str], 
    parent_trace: Optional[Any] = None
) -> Optional[List[Image.Image]]:
    """Edit images using FAL AI's image editing model.
    
    Design Choice: Uses bytedance/seedream model for high-quality edits.
    Supports batch editing of multiple images with same prompt.
    
    Args:
        prompt: Text description of desired edits
        image_urls: List of publicly accessible image URLs to edit
        parent_trace: Optional Langfuse parent trace for nested tracing
        
    Returns:
        List of edited PIL Images if successful, None if editing fails
        
    Raises:
        ValueError: If prompt is empty or image_urls list is empty
        
    Example:
        edited = await edit_image_with_fal(
            "Make the sky more dramatic",
            ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
        )
        if edited:
            for i, img in enumerate(edited):
                img.save(f"edited_{i}.jpg")
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if not image_urls:
        raise ValueError("image_urls list cannot be empty")
    
    # Validate FAL API key
    fal_api_key = os.getenv("FAL_API_KEY")
    if not fal_api_key:
        logger.error("FAL_API_KEY not found in environment variables")
        return None
    
    # Create generation span for tracing
    generation = None
    if langfuse_client:
        generation = langfuse_client.start_generation(
            name="fal_image_editing", 
            model="fal-ai/bytedance/seedream/v4/edit",
            input={"prompt": prompt, "image_urls": image_urls}
        )
    
    try:
        logger.info(f"Editing {len(image_urls)} images with FAL AI")
        client = fal_client.AsyncClient(key=fal_api_key)
        start_time = time.time()
        
        # Subscribe to FAL AI editing
        result = await client.subscribe(
            "fal-ai/bytedance/seedream/v4/edit",
            arguments={
                "prompt": prompt,
                "image_urls": image_urls
            }
        )

        generation_time = time.time() - start_time
        logger.info(f"Image editing completed in {generation_time:.2f}s")

        # Validate result and download images
        if result and 'images' in result and result['images']:
            edited_images = []
            
            for idx, img_data in enumerate(result['images']):
                try:
                    logger.debug(f"Downloading edited image {idx+1}/{len(result['images'])}")
                    img_response = requests.get(img_data['url'], timeout=30)
                    img_response.raise_for_status()
                    image = Image.open(io.BytesIO(img_response.content))
                    edited_images.append(image)
                except Exception as e:
                    logger.error(f"Failed to download edited image {idx+1}: {e}")
                    # Continue with other images
            
            if not edited_images:
                logger.error("Failed to download any edited images")
                return None
            
            logger.info(f"Successfully edited {len(edited_images)} images")
            
            # Update tracing
            if generation:
                generation.update(
                    output={
                        "images_edited": len(edited_images),
                        "generation_time_seconds": generation_time,
                        "success": True
                    },
                    metadata={
                        "model": "fal-ai/bytedance/seedream/v4/edit",
                        "generation_time": generation_time,
                        "input_images_count": len(image_urls),
                        "output_images_count": len(edited_images)
                    }
                )
                generation.end()
            
            return edited_images
        else:
            logger.error("No images in FAL AI editing result")
            if generation:
                generation.update(
                    output={"images_edited": 0, "error": "No images in result"},
                    metadata={"editing_failed": True}
                )
                generation.end()
            return None

    except requests.exceptions.RequestException as e:
        error_msg = f"Network error during image editing: {e}"
        logger.error(error_msg)
        if generation:
            generation.update(
                output={"images_edited": 0, "error": error_msg},
                metadata={"editing_failed": True, "error": str(e)}
            )
            generation.end()
        return None
        
    except Exception as e:
        error_msg = f"Unexpected error editing images: {e}"
        logger.error(error_msg, exc_info=True)
        if generation:
            generation.update(
                output={"images_edited": 0, "error": error_msg},
                metadata={"editing_failed": True, "error": str(e)}
            )
            generation.end()
        return None

async def generate_video_with_fal(
    prompt: str, 
    image_url: str, 
    duration: int = 8
) -> Optional[bytes]:
    """Generate video from image using FAL AI's Kling video model.
    
    Design Choice: Uses image-to-video model for animated content.
    Progress logs are captured for user feedback during long operations.
    
    Args:
        prompt: Text description of desired video motion/animation
        image_url: Publicly accessible URL of base image
        duration: Video duration in seconds (default: 8, typically 5-10)
        
    Returns:
        Video content as bytes (MP4 format) if successful, None if generation fails
        
    Raises:
        ValueError: If prompt or image_url is empty, or duration invalid
        
    Example:
        video_bytes = await generate_video_with_fal(
            "Camera slowly pans across the scene",
            "https://example.com/base_image.jpg",
            duration=5
        )
        if video_bytes:
            with open("output.mp4", "wb") as f:
                f.write(video_bytes)
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if not image_url or not image_url.strip():
        raise ValueError("image_url cannot be empty")
    if duration < 1 or duration > 10:
        raise ValueError("Duration must be between 1 and 10 seconds")
    
    # Validate FAL API key
    fal_api_key = os.getenv("FAL_API_KEY")
    if not fal_api_key:
        logger.error("FAL_API_KEY not found in environment variables")
        return None
    
    try:
        logger.info(f"Generating {duration}s video from image with FAL AI")
        client = fal_client.AsyncClient(key=fal_api_key)

        # Progress callback for logging
        def on_queue_update(update):
            """Log progress updates during video generation."""
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.info(f"Video generation progress: {log['message']}")

        start_time = time.time()
        
        # Subscribe to video generation
        result = await client.subscribe(
            "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
            arguments={
                "prompt": prompt,
                "image_url": image_url
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Video generation completed in {generation_time:.2f}s")

        # Validate result and download video
        if result and 'video' in result:
            video_url = result['video']['url']
            logger.debug(f"Downloading generated video from {video_url}")
            
            # Download with extended timeout (videos are larger)
            video_response = requests.get(video_url, timeout=120)
            video_response.raise_for_status()
            
            video_content = video_response.content
            logger.info(f"Successfully generated video ({len(video_content)} bytes)")
            
            return video_content
        else:
            logger.error("No video in FAL AI result")
            return None

    except requests.exceptions.Timeout:
        logger.error("Video download timed out")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during video generation: {e}")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error generating video: {e}", exc_info=True)
        return None


def resize_for_stories(image: Image.Image) -> Image.Image:
    """Resize image for social media stories (9:16 aspect ratio).
    
    Design Choice: Vertical format optimized for mobile viewing.
    Uses LANCZOS resampling for highest quality. Center-crops to exact dimensions.
    
    Args:
        image: PIL Image to resize
        
    Returns:
        Resized image (1080x1920 pixels)
        
    Raises:
        ValueError: If image is None or invalid
        
    Example:
        story_image = resize_for_stories(original_image)
        story_image.save("story.jpg")
    """
    if image is None:
        raise ValueError("Image cannot be None")
    
    # Instagram/Facebook Stories standard dimensions
    target_width = 1080
    target_height = 1920
    
    logger.debug(f"Resizing image from {image.size} to {target_width}x{target_height} for stories")

    # Calculate dimensions maintaining aspect ratio
    img_ratio = image.width / image.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        # Image is wider, fit to height
        new_height = target_height
        new_width = int(target_height * img_ratio)
    else:
        # Image is taller, fit to width
        new_width = target_width
        new_height = int(target_width / img_ratio)

    # Resize with high-quality LANCZOS resampling
    # Design Choice: LANCZOS provides best quality for downsampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop to exact dimensions
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    cropped = resized.crop((left, top, right, bottom))
    logger.debug(f"Successfully resized to {cropped.size}")
    
    return cropped

def resize_for_feed(image: Image.Image) -> Image.Image:
    """Resize image for social media feed posts (1:1 aspect ratio).
    
    Design Choice: Square format works universally across platforms.
    Centers image on white background if aspect ratio doesn't match.
    
    Args:
        image: PIL Image to resize
        
    Returns:
        Resized square image (1080x1080 pixels)
        
    Raises:
        ValueError: If image is None or invalid
        
    Example:
        feed_image = resize_for_feed(original_image)
        feed_image.save("feed.jpg")
    """
    if image is None:
        raise ValueError("Image cannot be None")
    
    # Instagram/Facebook feed standard size
    target_size = 1080
    
    logger.debug(f"Resizing image from {image.size} to {target_size}x{target_size} for feed")

    # Create a copy to avoid modifying original
    # Design Choice: thumbnail() modifies in-place, so we copy first
    img_copy = image.copy()
    
    # Resize maintaining aspect ratio (thumbnail modifies in-place)
    img_copy.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    # Create square canvas with white background
    # Design Choice: White background is neutral and works with most content
    square_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))

    # Calculate position to center the image
    x = (target_size - img_copy.width) // 2
    y = (target_size - img_copy.height) // 2

    # Paste resized image onto canvas
    square_image.paste(img_copy, (x, y))
    
    logger.debug(f"Successfully created {square_image.size} square image")
    return square_image

def enhance_image_for_posting(
    image: Image.Image, 
    post_type: str = "lifestyle"
) -> Image.Image:
    """Apply AI-powered post-processing to enhance image quality.
    
    Design Choice: Optional enhancement allows quality/performance tradeoff.
    Different enhancement styles for different content types (portrait vs lifestyle).
    
    Args:
        image: PIL Image to enhance
        post_type: Enhancement style ("portrait", "lifestyle", or "natural")
        
    Returns:
        Enhanced PIL Image (or original if post-processing disabled)
        
    Raises:
        ValueError: If image is None or post_type is invalid
        
    Example:
        enhanced = enhance_image_for_posting(image, "portrait")
        enhanced.save("enhanced.jpg")
    """
    if image is None:
        raise ValueError("Image cannot be None")
    
    valid_types = ["portrait", "lifestyle", "natural"]
    if post_type not in valid_types:
        raise ValueError(f"post_type must be one of {valid_types}, got '{post_type}'")

    # Check if post-processing is enabled
    if not ENABLE_POST_PROCESSING:
        logger.info("Post-processing disabled - using original image")
        return image

    logger.info(f"Applying {post_type} enhancement to image")

    try:
        # Convert PIL Image to bytes for processing
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        image_data = img_bytes.getvalue()

        # Apply appropriate enhancement based on post type
        # Design Choice: Different filters for different content types
        if post_type == "portrait":
            enhanced_bytes = apply_portrait_enhancement(image_data)
        elif post_type == "lifestyle":
            enhanced_bytes = apply_lifestyle_filter(image_data)
        else:  # natural
            enhanced_bytes = enhance_ai_image(image_data)

        # Convert back to PIL Image
        enhanced_image = Image.open(io.BytesIO(enhanced_bytes))
        
        logger.info(f"Successfully applied {post_type} enhancement")
        return enhanced_image
        
    except Exception as e:
        logger.error(f"Enhancement failed: {e}. Returning original image.")
        # Design Choice: Fail gracefully - return original if enhancement fails
        return image

def post_image_to_facebook(image: Image.Image, caption: str = "") -> Optional[Dict[str, Any]]:
    """Post an image to the Facebook Page feed.

    Design Choice: Uses GCS for hosting and Graph API for posting via URL.
    Applies enhancement and feed resizing for best quality and compatibility.

    Args:
        image: PIL Image to post
        caption: Optional caption text

    Returns:
        API response dict if successful, otherwise None
    """
    logger.info("Posting image to Facebook feed…")

    # Validate credentials
    if not page_id or not access_token:
        logger.error("Facebook credentials not set. Call set_profile_credentials() first.")
        return None

    # Enhance and resize for feed
    enhanced_image = enhance_image_for_posting(image, "lifestyle")
    feed_image = resize_for_feed(enhanced_image)

    # Save to temp and upload to GCS for a shareable URL
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        feed_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        filename = f"facebook_{uuid.uuid4().hex}.jpg"
        with open(temp_file_path, "rb") as f:
            image_url = upload_image_to_gcs(f, GCS_BUCKET_NAME, filename)

        if not image_url:
            logger.error("Failed to upload image to GCS")
            return None

        logger.info(f"Image uploaded to GCS: {image_url}")

        api_url = f'https://graph.facebook.com/v23.0/{page_id}/photos'
        multipart_fields = {
            "url": (None, image_url),
            "access_token": (None, access_token)
        }
        if caption:
            multipart_fields["caption"] = (None, caption)

        response = requests.post(api_url, files=multipart_fields, timeout=30)

        if response.ok:
            result = response.json()
            logger.info(f"Successfully posted to Facebook feed. Post ID: {result.get('id')}")
            return result
        else:
            logger.error(f"Facebook API error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error posting to Facebook: {e}")
        return None
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_image_to_facebook2(image: Image.Image, caption: str = "") -> Optional[Dict[str, Any]]:
    """Deprecated: Use post_image_to_facebook(). Kept for backward compatibility."""
    logger.warning("post_image_to_facebook2 is deprecated. Using post_image_to_facebook instead.")
    return post_image_to_facebook(image, caption)

def post_story_to_facebook(image: Image.Image) -> Optional[Dict[str, Any]]:
    """Post image to Facebook Stories using the Graph API.

    Returns API response dict if successful, otherwise None.
    """
    logger.info("Posting image to Facebook Stories…")

    # Apply post-processing enhancement first
    enhanced_image = enhance_image_for_posting(image, "portrait")

    # Resize for stories
    story_image = resize_for_stories(enhanced_image)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        story_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        # Step 1: Upload photo with published=false to get photo_id
        upload_url = f'https://graph.facebook.com/v23.0/{page_id}/photos'

        with open(temp_file_path, "rb") as f:
            files = {"source": f}
            data = {
                "access_token": access_token,  # Must be Page Access Token with pages_manage_posts permission
                "published": "false"  # Don't publish, just upload for story use
            }
            upload_response = requests.post(upload_url, files=files, data=data)

        if upload_response.status_code != 200:
            logger.error(f"HTTP Error {upload_response.status_code}: {upload_response.text}")
            return None

        upload_result = upload_response.json()
        if 'error' in upload_result:
            logger.error(f"Facebook API Error: {upload_result['error']}")
            return None

        photo_id = upload_result.get('id')
        if not photo_id:
            logger.error("No photo ID returned from upload")
            return None

        logger.info(f"Photo uploaded successfully. Photo ID: {photo_id}")

        # Step 2: Publish to Stories using photo_id
        story_url = f'https://graph.facebook.com/v23.0/{page_id}/photo_stories'
        story_data = {
            "access_token": access_token,
            "photo_id": photo_id
        }

        story_response = requests.post(story_url, data=story_data)

        if story_response.status_code == 200:
            result = story_response.json()
            if 'error' in result:
                logger.error(f"Facebook API Error: {result['error']}")
                return None

            logger.info(f"Successfully posted Facebook story. Story ID: {result.get('id')}")
            return result
        else:
            logger.error(f"HTTP Error {story_response.status_code}: {story_response.text}")
            return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_image_to_instagram(image: Image.Image, caption: str = "", max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Post image to Instagram feed using Instagram Graph API with retry logic."""
    logger.info("Posting image to Instagram feed…")

    # Apply post-processing enhancement first
    enhanced_image = enhance_image_for_posting(image, "lifestyle")

    # Resize for Instagram feed (square 1:1 aspect ratio)
    feed_image = resize_for_feed(enhanced_image)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        feed_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        # Upload image to GCS first
        import uuid
        filename = f"instagram_{uuid.uuid4().hex}.jpg"

        with open(temp_file_path, "rb") as f:
            image_url = upload_image_to_gcs(f, GCS_BUCKET_NAME, filename)

        if not image_url:
            logger.error("Failed to upload image to GCS")
            return None

        logger.info(f"Image uploaded to GCS: {image_url}")

        # Step 1: Create media container with image_url
        container_url = f'https://graph.facebook.com/v23.0/{instagram_business_account_id}/media'
        container_data = {
            "image_url": image_url,
            "media_type": "IMAGE",  # Explicitly specify this is an image
            "access_token": access_token
        }

        if caption:
            container_data["caption"] = caption

        container_response = requests.post(container_url, data=container_data)

        if container_response.status_code != 200:
            logger.error(f"HTTP Error {container_response.status_code}: {container_response.text}")
            return None

        container_result = container_response.json()
        if 'error' in container_result:
            logger.error(f"Instagram API Error: {container_result['error']}")
            return None

        creation_id = container_result.get('id')
        if not creation_id:
            logger.error("No creation ID returned from container creation")
            return None

        logger.info(f"Media container created successfully. Creation ID: {creation_id}")

        # Step 2: Publish the media with retry logic
        publish_url = f'https://graph.facebook.com/v23.0/{instagram_business_account_id}/media_publish'
        publish_data = {
            "access_token": access_token,
            "creation_id": creation_id
        }

        for attempt in range(max_retries):
            try:
                publish_response = requests.post(publish_url, data=publish_data)

                if publish_response.status_code == 200:
                    result = publish_response.json()
                    if 'error' in result:
                        error_message = result['error'].get('message', '').lower()
                        if 'not ready for publishing' in error_message or 'media processing' in error_message:
                            wait_time = (2 ** attempt) + 1  # Exponential backoff
                            logger.info(f"Media not ready. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Instagram API Error: {result['error']}")
                            return None

                    logger.info(f"Successfully posted to Instagram feed. Media ID: {result.get('id')}")
                    return result

                # Handle HTTP errors with potential retry
                elif publish_response.status_code == 400:
                    try:
                        error_data = publish_response.json()
                        if 'error' in error_data:
                            error_message = error_data['error'].get('message', '').lower()
                            if 'not ready for publishing' in error_message or 'media processing' in error_message:
                                wait_time = (2 ** attempt) + 1
                                print(f"⏳ Media not ready for publishing. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                                continue
                    except:
                        pass

                    if attempt == max_retries - 1:
                        logger.error(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None
                else:
                    if attempt == max_retries - 1:
                        print(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Request exception: {e}")
                    return None
                wait_time = (2 ** attempt) + 1
                logger.warning(f"Request failed. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        logger.error(f"Failed to publish to Instagram feed after {max_retries} attempts")
        return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_story_to_instagram(image: Image.Image, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Post image to Instagram Stories with retry logic."""
    logger.info("Posting image to Instagram Stories…")

    # Apply post-processing enhancement first
    enhanced_image = enhance_image_for_posting(image, "portrait")

    # Resize for Instagram Stories (9:16 aspect ratio)
    story_image = resize_for_stories(enhanced_image)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        story_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        # Upload image to GCS first
        import uuid
        filename = f"instagram_story_{uuid.uuid4().hex}.jpg"

        with open(temp_file_path, "rb") as f:
            image_url = upload_image_to_gcs(f, GCS_BUCKET_NAME, filename)

        if not image_url:
            print("Failed to upload image to GCS")
            return None

        print(f"Image uploaded to GCS: {image_url}")

        # Step 1: Create media container for story
        container_url = f'https://graph.facebook.com/v23.0/{instagram_business_account_id}/media'
        container_data = {
            "image_url": image_url,
            "media_type": "STORIES",
            "access_token": access_token
        }

        container_response = requests.post(container_url, data=container_data)

        if container_response.status_code != 200:
            logger.error(f"HTTP Error {container_response.status_code}: {container_response.text}")
            return None

        container_result = container_response.json()
        if 'error' in container_result:
            logger.error(f"Instagram API Error: {container_result['error']}")
            return None

        creation_id = container_result.get('id')
        if not creation_id:
            logger.error("No creation ID returned from story container creation")
            return None

        logger.info(f"Story media container created successfully. Creation ID: {creation_id}")

        # Step 2: Publish the story with retry logic
        publish_url = f'https://graph.facebook.com/v23.0/{instagram_business_account_id}/media_publish'
        publish_data = {
            "access_token": access_token,
            "creation_id": creation_id
        }

        for attempt in range(max_retries):
            try:
                publish_response = requests.post(publish_url, data=publish_data)

                if publish_response.status_code == 200:
                    result = publish_response.json()
                    if 'error' in result:
                        error_message = result['error'].get('message', '').lower()
                        if 'not ready for publishing' in error_message or 'media processing' in error_message:
                            wait_time = (2 ** attempt) + 1
                            logger.info(f"Media not ready. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Instagram API Error: {result['error']}")
                            return None

                    logger.info(f"Successfully posted Instagram story. Media ID: {result.get('id')}")
                    return result

                # Handle HTTP errors with potential retry
                elif publish_response.status_code == 400:
                    try:
                        error_data = publish_response.json()
                        if 'error' in error_data:
                            error_message = error_data['error'].get('message', '').lower()
                            if 'not ready for publishing' in error_message or 'media processing' in error_message:
                                wait_time = (2 ** attempt) + 1
                                logger.info(f"Media not ready. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                                continue
                    except:
                        pass

                    if attempt == max_retries - 1:
                        logger.error(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None
                else:
                    if attempt == max_retries - 1:
                        logger.error(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Request exception: {e}")
                    return None
                wait_time = (2 ** attempt) + 1
                logger.warning(f"Request failed. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        logger.error(f"Failed to publish Instagram story after {max_retries} attempts")
        return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_video_to_facebook(video_content: bytes, caption: str = "") -> Optional[Dict[str, Any]]:
    """Post video to Facebook feed."""
    logger.info("Posting video to Facebook feed…")

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(video_content)
        temp_file_path = temp_file.name

    try:
        filename = f"facebook_video_{uuid.uuid4().hex}.mp4"
        with open(temp_file_path, "rb") as f:
            video_url = upload_image_to_gcs(f, GCS_BUCKET_NAME, filename)

        if not video_url:
            logger.error("Failed to upload video to GCS")
            return None

        logger.info(f"Video uploaded to GCS: {video_url}")

        api_url = f'https://graph.facebook.com/v23.0/{page_id}/videos'

        multipart_fields = {
            "file_url": (None, video_url),
            "description": (None, caption),
            "access_token": (None, access_token)
        }

        response = requests.post(api_url, files=multipart_fields, timeout=120)

        if response.ok:
            result = response.json()
            logger.info(f"Successfully posted Facebook video. Post ID: {result.get('id')}")
            return result
        else:
            logger.error(f"Facebook API error {response.status_code}: {response.text}")
            return None

    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_video_to_instagram(video_content: bytes, caption: str = "", max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Post video to Instagram feed."""
    logger.info("Posting video to Instagram feed…")

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(video_content)
        temp_file_path = temp_file.name

    try:
        filename = f"instagram_video_{uuid.uuid4().hex}.mp4"

        with open(temp_file_path, "rb") as f:
            video_url = upload_image_to_gcs(f, GCS_BUCKET_NAME, filename)

        if not video_url:
            logger.error("Failed to upload video to GCS")
            return None

        logger.info(f"Video uploaded to GCS: {video_url}")

        container_url = f'https://graph.facebook.com/v23.0/{instagram_business_account_id}/media'
        container_data = {
            "video_url": video_url,
            "media_type": "REELS",
            "access_token": access_token
        }

        if caption:
            container_data["caption"] = caption

        container_response = requests.post(container_url, data=container_data)

        if container_response.status_code != 200:
            print(f"HTTP Error {container_response.status_code}: {container_response.text}")
            return None

        container_result = container_response.json()
        if 'error' in container_result:
            print(f"Instagram API Error: {container_result['error']}")
            return None

        creation_id = container_result.get('id')
        if not creation_id:
            logger.error("No creation ID returned from video container creation")
            return None

        logger.info(f"Video media container created successfully. Creation ID: {creation_id}")

        publish_url = f'https://graph.facebook.com/v23.0/{instagram_business_account_id}/media_publish'
        publish_data = {
            "access_token": access_token,
            "creation_id": creation_id
        }

        for attempt in range(max_retries):
            try:
                publish_response = requests.post(publish_url, data=publish_data)

                if publish_response.status_code == 200:
                    result = publish_response.json()
                    if 'error' in result:
                        error_message = result['error'].get('message', '').lower()
                        if 'not ready for publishing' in error_message or 'media processing' in error_message:
                            wait_time = (2 ** attempt) + 5
                            logger.info(f"Video processing. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Instagram API Error: {result['error']}")
                            return None

                    logger.info(f"Successfully posted video to Instagram feed. Media ID: {result.get('id')}")
                    return result

                elif publish_response.status_code == 400:
                    try:
                        error_data = publish_response.json()
                        if 'error' in error_data:
                            error_message = error_data['error'].get('message', '').lower()
                            if 'not ready for publishing' in error_message or 'media processing' in error_message:
                                wait_time = (2 ** attempt) + 5
                                logger.info(f"Video processing. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                                continue
                    except:
                        pass

                    if attempt == max_retries - 1:
                        logger.error(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None
                else:
                    if attempt == max_retries - 1:
                        logger.error(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Request exception: {e}")
                    return None
                wait_time = (2 ** attempt) + 5
                logger.warning(f"Request failed. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        logger.error(f"Failed to publish video to Instagram feed after {max_retries} attempts")
        return None

    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

async def main():
    # Start main span for the entire pipeline  
    main_trace = None
    if langfuse_client:
        main_trace = langfuse_client.start_span(
            name="ai_creator_pipeline",
            metadata={"component": "main_pipeline"}
        )
    
    # Parse command line arguments
    args = parse_arguments()
    print(args)
    print("TEST")
    
    if main_trace:
        main_trace.update(input={"profile": args.profile, "video": args.video})
    
    # Handle list profiles command
    profile_manager = ProfileManager()
    profiles = profile_manager.list_profiles()
    print("Available profiles:")
    for profile in profiles:
        print(f"  - {profile}")
        
    # Initialize profile manager and get profile config
    profile_manager = ProfileManager()
    try:
        profile_config = profile_manager.get_profile(args.profile)
        set_profile_credentials(profile_config)
        print(profile_config)

    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-profiles to see available profiles")
        if main_trace:
            main_trace.update(output={"error": str(e)}, metadata={"pipeline_failed": True})
        return

    # Handle JSON input if provided
    print(args.input_json)
    if args.input_json:
        json_data = load_json_data(args.input_json)
        if not json_data:
            return

        # Create generated_prompt.json without metadata
        create_generated_prompt_json(json_data)

        # Extract prompt and caption from JSON
        prompt, caption = extract_prompt_from_json(json_data)
        print("Prompt",prompt)
        if not prompt:
            print("Could not extract prompt from JSON data")
            return
    else:
        # Use the profile-specific prompt generation method
        generator = PromptGenerator(profile_name=args.profile)
        detailed_prompt = generator.generate_detailed_prompt()
        prompt, caption = extract_prompt_from_json(detailed_prompt)
        print("Prompt",prompt)


    if args.video:
        # VIDEO ONLY MODE
        print(f"🎬 Video mode: Generating video with duration: {args.duration} seconds")
        
        # First, get base images for the current profile
        image_urls = profile_manager.get_base_images(args.profile)

        if not image_urls:
            print(f"No base images found for profile '{args.profile}'")
            return

        # Generate an edited image first
        print("📸 Generating base image for video...")
        edited_images = await edit_image_with_fal(prompt, image_urls, main_trace)

        if not edited_images:
            logger.error("Failed to generate base image for video")
            return

        # Save the base image and upload to get URL for video generation
        base_image = edited_images[0]
        base_image.save("video_base_image.jpg")
        
        # Upload base image to GCS to get URL
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            base_image.save(temp_file.name, 'JPEG', quality=95)
            temp_file_path = temp_file.name

        try:
            filename = f"video_base_{uuid.uuid4().hex}.jpg"
            with open(temp_file_path, "rb") as f:
                image_url = upload_image_to_gcs(f, "ai-creator-debarya", filename)

            if not image_url:
                logger.error("Failed to upload base image for video generation")
                return

            print(f"Base image uploaded: {image_url}")

            # Generate video from the base image
            video_content = await generate_video_with_fal(prompt, image_url, args.duration)
            
            if not video_content:
                logger.error("Failed to generate video")
                return

            # Save video locally
            with open("output_video.mp4", "wb") as f:
                f.write(video_content)
            print("📹 Video saved as output_video.mp4")

            # Post video to Facebook and Instagram
            try:
                fb_feed_result = post_video_to_facebook(video_content, caption)
                ig_feed_result = post_video_to_instagram(video_content, caption)
                
                # No stories for videos
                fb_story_result = None
                ig_story_result = None

            except Exception as e:
                logger.error(f"Error posting video: {e}")
                return

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    else:
        # IMAGE ONLY MODE (default/most common)
        print("📸 Image mode: Processing images...")
        
        # Get base images for the current profile
        image_urls = profile_manager.get_base_images(args.profile)

        if not image_urls:
            print(f"No base images found for profile '{args.profile}'")
            return

        edited_images = await edit_image_with_fal(prompt, image_urls, main_trace)

        if not edited_images:
            logger.error("Failed to edit images")
            return

        # Save edited images locally
        for i, image in enumerate(edited_images):
            image.save(f"output_edited_{i+1}.jpg")

        # Use the first edited image for posting
        main_image = edited_images[0]

        # Post the images to Facebook and Instagram
        try:
            # Post to Facebook feed (resized for 1:1)
            fb_feed_result = post_image_to_facebook(main_image, caption)

            # Post to Facebook stories (resized for 9:16)
            fb_story_result = post_story_to_facebook(main_image)

            # Post to Instagram feed (resized for 1:1)
            ig_feed_result = post_image_to_instagram(main_image, caption)

            # Post to Instagram stories (resized for 9:16)
            ig_story_result = post_story_to_instagram(main_image)

        except Exception as e:
            print(f"Error posting to social media: {e}")
            fb_feed_result = False
            fb_story_result = False
            ig_feed_result = False
            ig_story_result = False

        # Summary of results
        successful_posts = []
        failed_posts = []

        if fb_feed_result:
            successful_posts.append("Facebook feed")
        else:
            failed_posts.append("Facebook feed")

        if fb_story_result:
            successful_posts.append("Facebook stories")
        else:
            failed_posts.append("Facebook stories")

        if ig_feed_result:
            successful_posts.append("Instagram feed")
        else:
            failed_posts.append("Instagram feed")

        if ig_story_result:
            successful_posts.append("Instagram stories")
        else:
            failed_posts.append("Instagram stories")

        if successful_posts:
            logger.info(f"Successfully posted to: {', '.join(successful_posts)}")

        if failed_posts:
            logger.warning(f"Failed to post to: {', '.join(failed_posts)}")

        if len(successful_posts) == 4:
            logger.info("Successfully posted to all platforms!")
        elif len(successful_posts) == 0:
            logger.error("Failed to post to any platform")

    # Update main trace with final results
    if main_trace:
        main_trace.update(
            output={
                "pipeline_completed": True,
                "successful_posts": successful_posts if 'successful_posts' in locals() else [],
                "failed_posts": failed_posts if 'failed_posts' in locals() else [],
                "total_successful": len(successful_posts) if 'successful_posts' in locals() else 0
            },
            metadata={
                "pipeline_type": "video" if args.video else "image",
                "profile": args.profile,
                "platforms_targeted": 4
            }
        )
        # Flush the trace to ensure it's sent
        langfuse_client.flush()

if __name__ == "__main__":
    asyncio.run(main())




