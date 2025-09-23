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

def test_different_posting_methods():
    """Test different ways of posting to Facebook to find what works"""

    # Initialize profile manager
    profile_manager = ProfileManager()
    profile_config = profile_manager.get_profile('rupashi')

    page_id = profile_config.facebook_page_id
    access_token = profile_config.access_token

    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='blue')

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        test_image.save(temp_file.name, 'JPEG', quality=95)
        temp_file_path = temp_file.name

    try:
        api_url = f'https://graph.facebook.com/v23.0/{page_id}/photos'

        print("="*60)
        print("METHOD 1: Files + Data (current method)")
        print("="*60)

        with open(temp_file_path, "rb") as f:
            files = {"source": f}
            data = {
                "access_token": access_token,
                "caption": "Test post method 1"
            }
            response = requests.post(api_url, files=files, data=data)

        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")

        print("\n" + "="*60)
        print("METHOD 2: Access token in URL params")
        print("="*60)

        with open(temp_file_path, "rb") as f:
            files = {"source": f}
            params = {"access_token": access_token}
            data = {"caption": "Test post method 2"}
            response = requests.post(api_url, files=files, data=data, params=params)

        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")

        print("\n" + "="*60)
        print("METHOD 3: Try without caption")
        print("="*60)

        with open(temp_file_path, "rb") as f:
            files = {"source": f}
            data = {"access_token": access_token}
            response = requests.post(api_url, files=files, data=data)

        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")

        print("\n" + "="*60)
        print("METHOD 4: Try with published=true explicitly")
        print("="*60)

        with open(temp_file_path, "rb") as f:
            files = {"source": f}
            data = {
                "access_token": access_token,
                "caption": "Test post method 4",
                "published": "true"
            }
            response = requests.post(api_url, files=files, data=data)

        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")

    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    test_different_posting_methods()