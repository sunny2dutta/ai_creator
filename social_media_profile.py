import requests
import os
import tempfile
import logging
import uuid
import asyncio
from typing import Optional, List
from PIL import Image
from profile_manager import ProfileManager, ProfileConfig
from prompt_generator import FacebookImagePromptGenerator
import fal_client
from google.cloud import storage

logger = logging.getLogger(__name__)

class SocialMediaProfile:
    """Handles social media posting for a specific profile"""

    def __init__(self, profile_name: str, profile_manager: Optional[ProfileManager] = None):
        self.profile_manager = profile_manager or ProfileManager()
        self.config = self.profile_manager.get_profile(profile_name)
        self.prompt_generator = FacebookImagePromptGenerator(self.profile_manager.get_storage())

        # Set Google Cloud credentials (should be moved to config)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/debaryadutta/google_cloud_storage.json'

    def upload_image_to_gcs(self, image_data, bucket_name: str, destination_blob_name: str, credentials_path: Optional[str] = None) -> Optional[str]:
        """Upload image to Google Cloud Storage and return public URL"""
        try:
            if credentials_path:
                storage_client = storage.Client.from_service_account_json(credentials_path)
            else:
                storage_client = storage.Client()

            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_file(image_data)
            return blob.public_url

        except Exception as e:
            logger.error(f"GCS upload error: {e}")
            return None

    async def generate_image_with_fal(self, prompt: str) -> Optional[Image.Image]:
        """Generate image using fal AI"""
        try:
            client = fal_client.AsyncClient(key=os.getenv("FAL_API_KEY"))
            result = await client.subscribe(
                "fal-ai/nano-banana",
                arguments={
                    "prompt": prompt,
                    "image_urls": []
                }
            )

            if result and 'images' in result and result['images']:
                image_url = result['images'][0]['url']
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                return Image.open(io.BytesIO(img_response.content))

        except Exception as e:
            logger.error(f"Image generation error: {e}")
        return None

    async def edit_image_with_fal(self, prompt: str, image_urls: List[str]) -> Optional[List[Image.Image]]:
        """Edit images using fal AI"""
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

        except Exception as e:
            logger.error(f"Image editing error: {e}")
        return None

    def resize_for_stories(self, image: Image.Image) -> Image.Image:
        """Resize image for Stories (9:16 aspect ratio)"""
        target_width, target_height = 1080, 1920
        img_ratio = image.width / image.height
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            new_height = target_height
            new_width = int(target_height * img_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / img_ratio)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return image.crop((left, top, right, bottom))

    def resize_for_feed(self, image: Image.Image) -> Image.Image:
        """Resize image for feed posts (1:1 aspect ratio)"""
        target_size = 1080
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        square_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        x = (target_size - image.width) // 2
        y = (target_size - image.height) // 2
        square_image.paste(image, (x, y))
        return square_image

    def post_image_to_facebook(self, image: Image.Image, caption: str = "") -> Optional[dict]:
        """Post image to Facebook feed"""
        logger.info("Posting to Facebook feed...")
        feed_image = self.resize_for_feed(image)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            feed_image.save(temp_file.name, 'JPEG', quality=95)
            temp_file_path = temp_file.name

        try:
            api_url = f'https://graph.facebook.com/v23.0/{self.config.facebook_page_id}/photos'

            with open(temp_file_path, "rb") as f:
                files = {"source": f}
                data = {"access_token": self.config.access_token}
                if caption:
                    data["caption"] = caption

                response = requests.post(api_url, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    logger.error(f"Facebook API Error: {result['error']}")
                    return None
                logger.info(f"Successfully posted to Facebook feed. Post ID: {result.get('id')}")
                return result
            else:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                return None

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def post_story_to_facebook(self, image: Image.Image) -> Optional[dict]:
        """Post image to Facebook Stories"""
        logger.info("Posting to Facebook Stories...")
        story_image = self.resize_for_stories(image)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            story_image.save(temp_file.name, 'JPEG', quality=95)
            temp_file_path = temp_file.name

        try:
            # Step 1: Upload photo
            upload_url = f'https://graph.facebook.com/v23.0/{self.config.facebook_page_id}/photos'

            with open(temp_file_path, "rb") as f:
                files = {"source": f}
                data = {
                    "access_token": self.config.access_token,
                    "published": "false"
                }
                upload_response = requests.post(upload_url, files=files, data=data)

            if upload_response.status_code != 200:
                logger.error(f"Upload error: {upload_response.text}")
                return None

            upload_result = upload_response.json()
            if 'error' in upload_result:
                logger.error(f"Facebook API Error: {upload_result['error']}")
                return None

            photo_id = upload_result.get('id')
            if not photo_id:
                logger.error("No photo ID returned from upload")
                return None

            # Step 2: Publish to Stories
            story_url = f'https://graph.facebook.com/v23.0/{self.config.facebook_page_id}/photo_stories'
            story_data = {
                "access_token": self.config.access_token,
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
                logger.error(f"Story post error: {story_response.text}")
                return None

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def post_image_to_instagram(self, image: Image.Image, caption: str = "") -> Optional[dict]:
        """Post image to Instagram feed"""
        logger.info("Posting to Instagram feed...")
        feed_image = self.resize_for_feed(image)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            feed_image.save(temp_file.name, 'JPEG', quality=95)
            temp_file_path = temp_file.name

        try:
            filename = f"instagram_{uuid.uuid4().hex}.jpg"

            with open(temp_file_path, "rb") as f:
                image_url = self.upload_image_to_gcs(f, "ai-creator-debarya", filename)

            if not image_url:
                logger.error("Failed to upload image to GCS")
                return None

            # Step 1: Create media container
            container_url = f'https://graph.facebook.com/v23.0/{self.config.instagram_business_id}/media'
            container_data = {
                "image_url": image_url,
                "media_type": "IMAGE",
                "access_token": self.config.access_token
            }

            if caption:
                container_data["caption"] = caption

            container_response = requests.post(container_url, data=container_data)

            if container_response.status_code != 200:
                logger.error(f"Container creation error: {container_response.text}")
                return None

            container_result = container_response.json()
            if 'error' in container_result:
                logger.error(f"Instagram API Error: {container_result['error']}")
                return None

            creation_id = container_result.get('id')
            if not creation_id:
                logger.error("No creation ID returned")
                return None

            # Step 2: Publish media
            publish_url = f'https://graph.facebook.com/v23.0/{self.config.instagram_business_id}/media_publish'
            publish_data = {
                "access_token": self.config.access_token,
                "creation_id": creation_id
            }

            publish_response = requests.post(publish_url, data=publish_data)

            if publish_response.status_code == 200:
                result = publish_response.json()
                if 'error' in result:
                    logger.error(f"Instagram API Error: {result['error']}")
                    return None
                logger.info(f"Successfully posted to Instagram feed. Media ID: {result.get('id')}")
                return result
            else:
                logger.error(f"Publish error: {publish_response.text}")
                return None

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def post_story_to_instagram(self, image: Image.Image) -> Optional[dict]:
        """Post image to Instagram Stories"""
        logger.info("Posting to Instagram Stories...")
        story_image = self.resize_for_stories(image)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            story_image.save(temp_file.name, 'JPEG', quality=95)
            temp_file_path = temp_file.name

        try:
            filename = f"instagram_story_{uuid.uuid4().hex}.jpg"

            with open(temp_file_path, "rb") as f:
                image_url = self.upload_image_to_gcs(f, "ai-creator-debarya", filename)

            if not image_url:
                logger.error("Failed to upload image to GCS")
                return None

            # Step 1: Create story media container
            container_url = f'https://graph.facebook.com/v23.0/{self.config.instagram_business_id}/media'
            container_data = {
                "image_url": image_url,
                "media_type": "STORIES",
                "access_token": self.config.access_token
            }

            container_response = requests.post(container_url, data=container_data)

            if container_response.status_code != 200:
                logger.error(f"Story container error: {container_response.text}")
                return None

            container_result = container_response.json()
            if 'error' in container_result:
                logger.error(f"Instagram API Error: {container_result['error']}")
                return None

            creation_id = container_result.get('id')
            if not creation_id:
                logger.error("No creation ID returned for story")
                return None

            # Step 2: Publish story
            publish_url = f'https://graph.facebook.com/v23.0/{self.config.instagram_business_id}/media_publish'
            publish_data = {
                "access_token": self.config.access_token,
                "creation_id": creation_id
            }

            publish_response = requests.post(publish_url, data=publish_data)

            if publish_response.status_code == 200:
                result = publish_response.json()
                if 'error' in result:
                    logger.error(f"Instagram API Error: {result['error']}")
                    return None
                logger.info(f"Successfully posted Instagram story. Media ID: {result.get('id')}")
                return result
            else:
                logger.error(f"Story publish error: {publish_response.text}")
                return None

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    async def generate_and_post(self, mode: str = "edit") -> dict:
        """Main method to generate content and post to all platforms"""
        logger.info(f"Starting content generation for profile: {self.config.profile_name}")

        # Generate prompt and caption
        prompt, caption = self.prompt_generator.generate_prompt(self.config.profile_name)
        logger.info("Generated prompt with randomly selected category")

        # Generate or edit image
        if mode == "generate":
            main_image = await self.generate_image_with_fal(prompt)
        else:  # edit mode
            base_images = self.profile_manager.get_base_images(self.config.profile_name)
            if not base_images:
                logger.error("No base images found for editing")
                return {"success": False, "error": "No base images available"}

            edited_images = await self.edit_image_with_fal(prompt, base_images)
            main_image = edited_images[0] if edited_images else None

        if not main_image:
            logger.error("Failed to generate/edit image")
            return {"success": False, "error": "Image generation failed"}

        # Save image locally for debugging
        main_image.save(f"output_{self.config.profile_name}_{category or 'random'}.jpg")

        # Post to all platforms
        results = {
            "facebook_feed": self.post_image_to_facebook(main_image, caption),
            "facebook_story": self.post_story_to_facebook(main_image),
            "instagram_feed": self.post_image_to_instagram(main_image, caption),
            "instagram_story": self.post_story_to_instagram(main_image)
        }

        # Summarize results
        successful_posts = [platform for platform, result in results.items() if result]
        failed_posts = [platform for platform, result in results.items() if not result]

        if successful_posts:
            logger.info(f"Successfully posted to: {', '.join(successful_posts)}")
        if failed_posts:
            logger.warning(f"Failed to post to: {', '.join(failed_posts)}")

        return {
            "success": len(successful_posts) > 0,
            "successful_posts": successful_posts,
            "failed_posts": failed_posts,
            "results": results
        }