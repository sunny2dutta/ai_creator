#!/usr/bin/env python3
import os
import sys
import requests
import tempfile
from PIL import Image
from dotenv import load_dotenv

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))
from profile_manager import ProfileManager

load_dotenv()

def test_facebook_post():
    """Test Facebook posting with current credentials"""

    # Initialize profile manager
    profile_manager = ProfileManager()
    try:
        profile_config = profile_manager.get_profile('rupashi')
        print(f"✅ Profile loaded: {profile_config.profile_name}")
        print(f"📄 Page ID: {profile_config.facebook_page_id}")
        print(f"🔑 Access Token: {profile_config.access_token[:20]}...")
    except ValueError as e:
        print(f"❌ Error loading profile: {e}")
        return

    # Create a simple test image
    test_image = Image.new('RGB', (1080, 1080), color='red')

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        test_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        # Test Facebook posting
        api_url = f'https://graph.facebook.com/v23.0/{profile_config.facebook_page_id}/photos'

        print(f"🚀 Testing Facebook API call to: {api_url}")

        with open(temp_file_path, "rb") as f:
            files = {"source": f}
            data = {
                "access_token": profile_config.access_token,
                "caption": "Test post from debugging session"
            }

            response = requests.post(api_url, files=files, data=data)

        print(f"📊 Response Status: {response.status_code}")
        print(f"📋 Response Body: {response.text}")

        if response.status_code == 200:
            result = response.json()
            if 'error' in result:
                print(f"❌ Facebook API Error: {result['error']}")
            else:
                print(f"✅ Successfully posted! Post ID: {result.get('id')}")
        else:
            print(f"❌ HTTP Error {response.status_code}")

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    test_facebook_post()