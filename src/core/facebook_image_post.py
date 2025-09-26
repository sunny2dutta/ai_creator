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
from dotenv import load_dotenv
from PIL import Image
import fal_client
from prompt_generator import PromptGenerator
from profile_manager import ProfileManager
import uuid
from google.cloud import storage

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_post_processor import ImagePostProcessor, enhance_ai_image, apply_portrait_enhancement, apply_lifestyle_filter

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

def load_json_data(json_path):
    """Load data from JSON file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return None

def extract_prompt_from_json(json_data):
    """Extract prompt and caption from JSON data"""
    if not json_data:
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

    return prompt, caption

def create_generated_prompt_json(original_data, output_path="generated_prompt.json"):
    """Create generated_prompt.json with everything except metadata"""
    if not original_data:
        return False

    # Create filtered data excluding metadata
    filtered_data = {}
    for key, value in original_data.items():
        if key != 'metadata':
            filtered_data[key] = value

    try:
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        print(f"Created {output_path} without metadata")
        return True
    except Exception as e:
        print(f"Error creating {output_path}: {e}")
        return False

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

    parser.add_argument(
        '--input-json',
        type=str,
        help='Path to input JSON file for consuming generated prompts'
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

def enhance_image_for_posting(image, post_type="lifestyle"):
    """Apply post-processing to make AI images look more realistic before posting"""
    print(f"Applying post-processing enhancement for {post_type}...")

    # Convert PIL Image to bytes for processing
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    image_data = img_bytes.getvalue()

    # Apply appropriate enhancement based on post type
    if post_type == "portrait":
        enhanced_bytes = apply_portrait_enhancement(image_data)
    elif post_type == "lifestyle":
        enhanced_bytes = apply_lifestyle_filter(image_data)
    else:
        enhanced_bytes = enhance_ai_image(image_data)

    # Convert back to PIL Image
    enhanced_image = Image.open(io.BytesIO(enhanced_bytes))

    print("✅ Post-processing enhancement applied")
    return enhanced_image

def post_image_to_facebook(image, caption=""):
    print("Posting to Facebook feed...")

    # Apply post-processing enhancement before uploading
    enhanced_image = enhance_image_for_posting(image, "lifestyle")

    # Save enhanced image to temp file + upload to GCS
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        enhanced_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        filename = f"facebook_{uuid.uuid4().hex}.jpg"
        with open(temp_file_path, "rb") as f:
            image_url = upload_image_to_gcs(f, "ai-creator-debarya", filename)

        if not image_url:
            print("Failed to upload image to GCS")
            return None

        print(f"Image uploaded to GCS: {image_url}")

        api_url = f'https://graph.facebook.com/v23.0/{page_id}/photos'

        # 👇 Force multipart/form-data (same as curl -F)
        multipart_fields = {
            "url": (None, image_url),
            "caption": (None, caption),
            "access_token": (None, access_token)
        }

        response = requests.post(api_url, files=multipart_fields)

        if response.ok:
            result = response.json()
            print(f"✅ Successfully posted. Post ID: {result.get('id')}")
            return result
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None

    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_image_to_facebook2(image, caption=""):
    """Post image to Facebook feed using Page Access Token and pages_manage_posts permission"""
    print("Posting to Facebook feed...")

    # Apply post-processing enhancement first
    enhanced_image = enhance_image_for_posting(image, "lifestyle")

    # Resize for feed
    feed_image = resize_for_feed(enhanced_image)

    # Save to temporary file and upload to GCS to get URL
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        feed_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        # Upload image to GCS first to get a URL
        import uuid
        filename = f"facebook_{uuid.uuid4().hex}.jpg"

        with open(temp_file_path, "rb") as f:
            image_url = upload_image_to_gcs(f, "ai-creator-debarya", filename)

        if not image_url:
            print("Failed to upload image to GCS")
            return None

        print(f"Image uploaded to GCS: {image_url}")
        print("page_id",page_id)

        api_url = f'https://graph.facebook.com/v23.0/{page_id}/photos'

        data = {
            "url": image_url,
            "access_token": access_token
        }

        if caption:
            data["caption"] = caption

                # Use updated endpoint for pages with URL instead of file upload
        print("api_url:", api_url)
        print("page_id:", page_id, type(page_id))
        print("access_token repr (first 10 chars):", repr(access_token)[:60])
        print("data dict BEFORE request:", data)

        # 2) Try sending as multipart/form-data (exactly like curl -F)
        multipart_fields = {k: (None, str(v)) for k, v in data.items()}
        resp_multipart = requests.post(api_url, files=multipart_fields)   # forces multipart/form-data
        print("multipart/form resp:", resp_multipart.status_code, resp_multipart.text)

        # 3) Show exactly what requests will send (prepared request) and then send it
        req = requests.Request('POST', api_url, data=data)
        prepped = req.prepare()
        print("Prepared request headers:", prepped.headers)
        if hasattr(prepped, "body"):
            try:
                print("Prepared request body (first 1000 bytes):", prepped.body[:1000])
            except Exception:
                print("Prepared request body not printable (binary or large). len:", len(prepped.body) if prepped.body else None)

        s = requests.Session()
        resp = s.send(prepped)
        print("response.status_code:", resp.status_code)
        print("response.text:", resp.text)



        response = requests.post(api_url, files=data)

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
    """Post image to Facebook Stories using Page Access Token and pages_manage_posts permission"""
    print("Posting to Facebook Stories...")

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

def post_image_to_instagram(image, caption="", max_retries=3):
    """Post image to Instagram feed using Instagram Graph API with retry logic"""
    print("Posting to Instagram feed...")

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
                            wait_time = (2 ** attempt) + 1  # Exponential backoff: 3, 5, 9 seconds
                            print(f"⏳ Media not ready for publishing. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"Instagram API Error: {result['error']}")
                            return None

                    print(f"Successfully posted to Instagram feed. Media ID: {result.get('id')}")
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
                        print(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None
                else:
                    if attempt == max_retries - 1:
                        print(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Request exception: {e}")
                    return None
                wait_time = (2 ** attempt) + 1
                print(f"⚠️  Request failed. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        print(f"Failed to publish to Instagram feed after {max_retries} attempts")
        return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def post_story_to_instagram(image, max_retries=3):
    """Post image to Instagram Stories with retry logic"""
    print("Posting to Instagram Stories...")

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
                            wait_time = (2 ** attempt) + 1  # Exponential backoff: 3, 5, 9 seconds
                            print(f"⏳ Media not ready for publishing. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"Instagram API Error: {result['error']}")
                            return None

                    print(f"Successfully posted Instagram story. Media ID: {result.get('id')}")
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
                        print(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None
                else:
                    if attempt == max_retries - 1:
                        print(f"HTTP Error {publish_response.status_code}: {publish_response.text}")
                        return None

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Request exception: {e}")
                    return None
                wait_time = (2 ** attempt) + 1
                print(f"⚠️  Request failed. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        print(f"Failed to publish Instagram story after {max_retries} attempts")
        return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

async def main():
    # Parse command line arguments
    args = parse_arguments()
    print(args)
    print("TEST")
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
        # Use the original prompt generation method
        generator = PromptGenerator("/Users/debaryadutta/ai_creator/src/core/7day_arc.json")
        detailed_prompt = generator.generate_detailed_prompt()
        prompt, caption = extract_prompt_from_json(detailed_prompt)
        print("Prompt",prompt)


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




