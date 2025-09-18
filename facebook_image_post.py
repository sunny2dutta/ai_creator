import requests
import os
import sys
import io
import asyncio
import tempfile
import logging
import base64
import argparse
from dotenv import load_dotenv
from PIL import Image
import fal_client
from prompt_generator import FacebookImagePromptGenerator
from profile_manager import ProfileManager
import uuid
from google.cloud import storage

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/debaryadutta/google_cloud_storage.json'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from urllib3/requests
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

# Global variables for backwards compatibility
page_id = None
access_token = None
instagram_business_account_id = None

def set_profile_credentials(profile_config):
    """Set global credentials for a profile"""
    global page_id, access_token, instagram_business_account_id
    page_id = profile_config.facebook_page_id
    access_token = profile_config.access_token
    instagram_business_account_id = profile_config.instagram_business_id

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate and post social media content')

    parser.add_argument(
        '--profile',
        type=str,
        default='rupashi',
        help='Profile name to use for posting (default: rupashi)'
    )

    parser.add_argument(
        '--category',
        type=str,
        help='Specific category for content generation'
    )

    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List all available profiles'
    )

    return parser.parse_args()

def upload_image_to_gcs(image_data, bucket_name, destination_blob_name, credentials_path=None):
    """
    Uploads an image from in-memory data to a GCS bucket and returns the public URL.

    Args:
        image_data: The image data as a file-like object
        bucket_name (str): The name of the GCS bucket
        destination_blob_name (str): The desired name of the object in the bucket
        credentials_path (str, optional): Path to the service account JSON file

    Returns:
        str: The public URL of the uploaded image if successful, otherwise None.
    """
    try:
        if credentials_path:
            storage_client = storage.Client.from_service_account_json(credentials_path)
        else:
            storage_client = storage.Client()  # Uses GOOGLE_APPLICATION_CREDENTIALS env var

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_file(image_data)

        return blob.public_url

    except Exception as e:
        print(f"An error occurred during upload: {e}")
        return None

async def generate_image_with_fal(prompt):
    try:
        client = fal_client.AsyncClient(key=os.getenv("FAL_API_KEY"))
        result = await client.subscribe(
            "fal-ai/nano-banana",
            arguments={
                "prompt": prompt,
                "image_urls": []  # Empty for generation from scratch
            }
        )

        if result and 'images' in result and result['images']:
            image_url = result['images'][0]['url']
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            image = Image.open(io.BytesIO(img_response.content))
            return image  # Return URL for Facebook posting
        else:
            return None

    except Exception as e:
        print(f"Error generating image: {e}")
        return None

async def edit_image_with_fal(prompt, image_urls):
    """Edit images using fal AI's bytedance/seedream/v4/edit model"""
    try:
        client = fal_client.AsyncClient(key=os.getenv("FAL_API_KEY"))
        result = await client.subscribe(
            "fal-ai/bytedance/seedream/v4/edit",
            arguments={
                "prompt": prompt,
                "image_urls": image_urls
            }
        )

        if result and 'images' in result and result['images']:
            edited_images = []
            for img_data in result['images']:
                img_response = requests.get(img_data['url'])
                img_response.raise_for_status()
                image = Image.open(io.BytesIO(img_response.content))
                edited_images.append(image)
            return edited_images
        else:
            return None

    except Exception as e:
        print(f"Error editing images: {e}")
        return None


def resize_for_stories(image):
    """Resize image for Facebook Stories (9:16 aspect ratio)"""
    target_width = 1080
    target_height = 1920

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

    # Resize and crop to exact dimensions
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))

def resize_for_feed(image):
    """Resize image for Facebook feed posts (1:1 aspect ratio)"""
    target_size = 1080

    # Resize maintaining aspect ratio
    image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    # Create square canvas and paste image centered
    square_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))

    # Calculate position to center the image
    x = (target_size - image.width) // 2
    y = (target_size - image.height) // 2

    square_image.paste(image, (x, y))
    return square_image

def post_image_to_facebook(image, caption=""):
    """Post image to Facebook feed"""
    print("Posting to Facebook feed...")

    # Resize for feed
    feed_image = resize_for_feed(image)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        feed_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        api_url = f'https://graph.facebook.com/v23.0/{page_id}/photos'

        # Upload the file directly
        with open(temp_file_path, "rb") as f:
            files = {"source": f}
            data = {
                "access_token": access_token
            }

            if caption:
                data["caption"] = caption

            response = requests.post(api_url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            if 'error' in result:
                print(f"Facebook API Error: {result['error']}")
                return None

            print(f"Successfully posted to feed. Post ID: {result.get('id')}")
            return result
        else:
            print(f"HTTP Error {response.status_code}: {response.text}")
            response.raise_for_status()

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_story_to_facebook(image):
    """Post image to Facebook Stories using two-step process"""
    print("Posting to Facebook Stories...")

    # Resize for stories
    story_image = resize_for_stories(image)

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
                "access_token": access_token,
                "published": "false"  # Don't publish, just upload for story use
            }
            upload_response = requests.post(upload_url, files=files, data=data)

        if upload_response.status_code != 200:
            print(f"HTTP Error {upload_response.status_code}: {upload_response.text}")
            return None

        upload_result = upload_response.json()
        if 'error' in upload_result:
            print(f"Facebook API Error: {upload_result['error']}")
            return None

        photo_id = upload_result.get('id')
        if not photo_id:
            print("No photo ID returned from upload")
            return None

        print(f"Photo uploaded successfully. Photo ID: {photo_id}")

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
                print(f"Facebook API Error: {result['error']}")
                return None

            print(f"Successfully posted story. Story ID: {result.get('id')}")
            return result
        else:
            print(f"HTTP Error {story_response.status_code}: {story_response.text}")
            return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_image_to_instagram(image, caption=""):
    """Post image to Instagram feed using Instagram Graph API"""
    print("Posting to Instagram feed...")

    # Resize for Instagram feed (square 1:1 aspect ratio)
    feed_image = resize_for_feed(image)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        feed_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        # Upload image to GCS first
        import uuid
        filename = f"instagram_{uuid.uuid4().hex}.jpg"

        with open(temp_file_path, "rb") as f:
            image_url = upload_image_to_gcs(f, "ai-creator-debarya", filename)

        if not image_url:
            print("Failed to upload image to GCS")
            return None

        print(f"Image uploaded to GCS: {image_url}")

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
            print(f"HTTP Error {container_response.status_code}: {container_response.text}")
            return None

        container_result = container_response.json()
        if 'error' in container_result:
            print(f"Instagram API Error: {container_result['error']}")
            return None

        creation_id = container_result.get('id')
        if not creation_id:
            print("No creation ID returned from container creation")
            return None

        print(f"Media container created successfully. Creation ID: {creation_id}")

        # Step 2: Publish the media
        publish_url = f'https://graph.facebook.com/v23.0/{instagram_business_account_id}/media_publish'
        publish_data = {
            "access_token": access_token,
            "creation_id": creation_id
        }

        publish_response = requests.post(publish_url, data=publish_data)

        if publish_response.status_code == 200:
            result = publish_response.json()
            if 'error' in result:
                print(f"Instagram API Error: {result['error']}")
                return None

            print(f"Successfully posted to Instagram feed. Media ID: {result.get('id')}")
            return result
        else:
            print(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
            return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_story_to_instagram(image):
    """Post image to Instagram Stories"""
    print("Posting to Instagram Stories...")

    # Resize for Instagram Stories (9:16 aspect ratio)
    story_image = resize_for_stories(image)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        story_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        # Upload image to GCS first
        import uuid
        filename = f"instagram_story_{uuid.uuid4().hex}.jpg"

        with open(temp_file_path, "rb") as f:
            image_url = upload_image_to_gcs(f, "ai-creator-debarya", filename)

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
            print(f"HTTP Error {container_response.status_code}: {container_response.text}")
            return None

        container_result = container_response.json()
        if 'error' in container_result:
            print(f"Instagram API Error: {container_result['error']}")
            return None

        creation_id = container_result.get('id')
        if not creation_id:
            print("No creation ID returned from story container creation")
            return None

        print(f"Story media container created successfully. Creation ID: {creation_id}")

        # Step 2: Publish the story
        publish_url = f'https://graph.facebook.com/v23.0/{instagram_business_account_id}/media_publish'
        publish_data = {
            "access_token": access_token,
            "creation_id": creation_id
        }

        publish_response = requests.post(publish_url, data=publish_data)

        if publish_response.status_code == 200:
            result = publish_response.json()
            if 'error' in result:
                print(f"Instagram API Error: {result['error']}")
                return None

            print(f"Successfully posted Instagram story. Media ID: {result.get('id')}")
            return result
        else:
            print(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
            return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

async def main():
    # Parse command line arguments
    args = parse_arguments()
    print(args)

    # Handle list profiles command
    if args.list_profiles:
        profile_manager = ProfileManager()
        profiles = profile_manager.list_profiles()
        print("Available profiles:")
        for profile in profiles:
            print(f"  - {profile}")
        return

    # Initialize profile manager and get profile config
    profile_manager = ProfileManager()
    try:
        profile_config = profile_manager.get_profile(args.profile)
        set_profile_credentials(profile_config)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-profiles to see available profiles")
        return

    # Choose mode: 'generate' for new image generation or 'edit' for editing existing images
    mode = "generate"  # Change to "edit" if you want to edit existing images

    generator = FacebookImagePromptGenerator()
    prompt, caption = generator.generate_prompt(args.profile)

    # Get base images for the current profile
    image_urls = profile_manager.get_base_images(args.profile)

    if not image_urls:
        print(f"No base images found for profile '{args.profile}'")
        return

    edited_images = await edit_image_with_fal(prompt, image_urls)

    if edited_images:

        # Save edited images locally
        for i, image in enumerate(edited_images):
            image.save(f"output_edited_{i+1}.jpg")

        # Use the first edited image for posting
        main_image = edited_images[0]
    else:
        logger.error("Failed to edit images")
        return

    # Post the image (whether generated or edited) to Facebook and Instagram
    try:
        # Post to Facebook feed (resized for 1:1)
        fb_feed_result = post_image_to_facebook(main_image, caption)

        # Post to Facebook stories (resized for 9:16)
        fb_story_result = post_story_to_facebook(main_image)

        # Post to Instagram feed (resized for 1:1)
        ig_feed_result = post_image_to_instagram(main_image, caption)

        # Post to Instagram stories (resized for 9:16)
        ig_story_result = post_story_to_instagram(main_image)

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

    except Exception as e:
        logger.error(f"Error posting: {e}")

if __name__ == "__main__":
    asyncio.run(main())




