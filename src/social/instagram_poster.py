"""
Instagram Posting System using Instagram Graph API
Supports both feed posts and stories posting
"""

import requests
import os
import time
import tempfile
import io
from typing import Optional, Dict, Any, List
import json
from dataclasses import asdict
from ai_celebrity_config import AIInstagramCelebrity, PostType
from google.cloud import storage
import logging

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/debaryadutta/google_cloud_storage.json'

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

class InstagramGraphAPI:
    """Instagram Graph API client for posting content"""
    
    def __init__(self, access_token: str, instagram_business_id: str):
        self.access_token = access_token
        self.instagram_business_id = instagram_business_id
        self.base_url = "https://graph.facebook.com/v18.0"
        
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None, files: Dict = None) -> Dict:
        """Make HTTP request to Instagram Graph API"""
        url = f"{self.base_url}/{endpoint}"
        
        if params is None:
            params = {}
        params['access_token'] = self.access_token
        
        if method.upper() == 'GET':
            response = requests.get(url, params=params)
        elif method.upper() == 'POST':
            if files:
                response = requests.post(url, params=params, data=data, files=files)
            else:
                response = requests.post(url, params=params, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
            
        response.raise_for_status()
        return response.json()
    
    def upload_image(self, image_bytes: bytes, caption: str = "", is_story: bool = False) -> str:
        """Upload image and get media container ID"""
        
        # Create temporary file for upload
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Upload image to Facebook first
            files = {'source': open(temp_file_path, 'rb')}
            params = {
                'access_token': self.access_token
            }
            
            # Upload to Facebook
            upload_response = requests.post(
                f"{self.base_url}/me/photos",
                params=params,
                files=files
            )
            upload_response.raise_for_status()
            upload_data = upload_response.json()
            image_id = upload_data['id']
            
            files['source'].close()
            
            # Create media container
            media_params = {
                'image_url': f"https://graph.facebook.com/{image_id}?access_token={self.access_token}",
                'caption': caption if not is_story else "",  # Stories don't support captions
                'access_token': self.access_token
            }
            
            if is_story:
                media_params['media_type'] = 'STORIES'
            
            endpoint = f"{self.instagram_business_id}/media"
            media_response = requests.post(f"{self.base_url}/{endpoint}", params=media_params)
            media_response.raise_for_status()
            
            return media_response.json()['id']
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def publish_media(self, creation_id: str, max_retries: int = 3) -> str:
        """Publish uploaded media with retry logic for processing delays"""
        endpoint = f"{self.instagram_business_id}/media_publish"
        params = {
            'creation_id': creation_id,
            'access_token': self.access_token
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.base_url}/{endpoint}", params=params)

                if response.status_code == 200:
                    return response.json()['id']

                # Check if it's a media processing delay error
                if response.status_code == 400:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_message = error_data['error'].get('message', '').lower()
                        if 'not ready for publishing' in error_message or 'media processing' in error_message:
                            wait_time = (2 ** attempt) + 1  # Exponential backoff: 3, 5, 9 seconds
                            print(f"⏳ Media not ready for publishing. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue

                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = (2 ** attempt) + 1
                print(f"⚠️  Request failed. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        raise Exception(f"Failed to publish media after {max_retries} attempts")
    
    def post_feed_image(self, image_bytes: bytes, caption: str = "") -> str:
        """Post image to Instagram feed"""
        print(f"Uploading image to Instagram feed...")
        
        # Upload and create media container
        creation_id = self.upload_image(image_bytes, caption, is_story=False)
        print(f"Media container created: {creation_id}")
        
        # Publish the media
        media_id = self.publish_media(creation_id)
        print(f"Successfully posted to feed. Media ID: {media_id}")
        
        return media_id
    
    def post_story_image(self, image_bytes: bytes) -> str:
        """Post image to Instagram Stories"""
        print(f"Uploading image to Instagram Stories...")
        
        # Upload and create story media container
        creation_id = self.upload_image(image_bytes, "", is_story=True)
        print(f"Story media container created: {creation_id}")
        
        # Publish the story
        media_id = self.publish_media(creation_id)
        print(f"Successfully posted to stories. Media ID: {media_id}")
        
        return media_id
    
    def get_account_info(self) -> Dict:
        """Get Instagram account information"""
        endpoint = f"{self.instagram_business_id}"
        params = {
            'fields': 'id,username,name,profile_picture_url,followers_count,follows_count,media_count',
            'access_token': self.access_token
        }
        
        response = requests.get(f"{self.base_url}/{endpoint}", params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_media_insights(self, media_id: str) -> Dict:
        """Get insights for a specific media post"""
        endpoint = f"{media_id}/insights"
        params = {
            'metric': 'impressions,reach,likes,comments,shares,saves',
            'access_token': self.access_token
        }
        
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"Could not fetch insights: {e}")
            return {}

class InstagramPoster:
    """High-level Instagram posting interface"""
    
    def __init__(self, celebrity_config: AIInstagramCelebrity):
        self.celebrity_config = celebrity_config
        self.api = InstagramGraphAPI(
            celebrity_config.instagram_config.access_token,
            celebrity_config.instagram_config.instagram_business_id
        )
        
    def post_to_feed(self, image_bytes: bytes, custom_caption: Optional[str] = None) -> str:
        """Post image to Instagram feed with auto-generated or custom caption"""
        
        if custom_caption:
            caption = custom_caption
        else:
            caption = self._generate_caption()
        
        return self.api.post_feed_image(image_bytes, caption)
    
    def post_to_story(self, image_bytes: bytes) -> str:
        """Post image to Instagram Stories"""
        return self.api.post_story_image(image_bytes)
    
    def _generate_caption(self) -> str:
        """Generate caption based on celebrity profile"""
        celebrity = self.celebrity_config.celebrity
        
        # Simple caption generation based on personality and interests
        captions = [
            f"Living my best life! ✨ #{' #'.join(celebrity.interests[:2])}",
            f"Another day, another adventure! 💫 #{' #'.join(celebrity.personality_traits[:2])}",
            f"Sharing some {celebrity.interests[0]} vibes with you all! 💕",
            f"Feeling {celebrity.personality_traits[0]} today! ✨ What's inspiring you?",
            f"Just me being me! 💫 #{celebrity.occupation.replace(' ', '')}"
        ]
        
        import random
        return random.choice(captions)
    
    def get_posting_limits_status(self) -> Dict:
        """Check current posting limits (25 posts per 24 hours)"""
        # This would require tracking posts in a database or file
        # For now, return a placeholder
        return {
            "daily_limit": 25,
            "posts_today": 0,  # Would need to track this
            "remaining": 25,
            "reset_time": "24 hours from first post"
        }
    
    def validate_connection(self) -> bool:
        """Validate Instagram API connection and credentials"""
        try:
            account_info = self.api.get_account_info()
            print(f"✅ Connected to Instagram account: @{account_info.get('username', 'unknown')}")
            print(f"   Followers: {account_info.get('followers_count', 'N/A')}")
            print(f"   Posts: {account_info.get('media_count', 'N/A')}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Instagram API: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize celebrity config
    celebrity = AIInstagramCelebrity()
    
    # Example: Update with real Instagram API credentials
    celebrity.instagram_config.access_token = "YOUR_REAL_ACCESS_TOKEN"
    celebrity.instagram_config.instagram_business_id = "YOUR_REAL_BUSINESS_ID"
    celebrity.instagram_config.facebook_page_id = "YOUR_REAL_PAGE_ID"
    
    # Initialize poster
    poster = InstagramPoster(celebrity)
    
    # Test connection (will fail without real credentials)
    print("Testing Instagram API connection...")
    is_connected = poster.validate_connection()
    
    if is_connected:
        print("\n✅ Instagram API connection successful!")
        
        # Example posting (requires actual image bytes)
        # This is just for demonstration - you'd get image_bytes from the image generator
        example_image_bytes = b"fake_image_data"  # Replace with real image bytes
        
        try:
            # Post to feed
            # feed_media_id = poster.post_to_feed(example_image_bytes, "Custom caption for my post! #AI #lifestyle")
            
            # Post to story  
            # story_media_id = poster.post_to_story(example_image_bytes)
            
            print("Posts would be published with real image data and credentials!")
            
        except Exception as e:
            print(f"Error posting: {e}")
    else:
        print("\n❌ Please configure your Instagram API credentials in the celebrity config.")
        print("You need:")
        print("1. Instagram Business Account")
        print("2. Facebook Page connected to the Instagram account")
        print("3. Facebook App with Instagram Basic Display and Instagram Graph API permissions")
        print("4. Valid access token with instagram_basic, instagram_content_publish permissions")