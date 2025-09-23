#!/usr/bin/env python3
import os
import sys
import requests
from dotenv import load_dotenv

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))
from profile_manager import ProfileManager

load_dotenv()

def get_page_access_token():
    """Convert User Access Token to Page Access Token"""

    # Initialize profile manager
    profile_manager = ProfileManager()
    try:
        profile_config = profile_manager.get_profile('rupashi')
        user_access_token = profile_config.access_token
        target_page_id = profile_config.facebook_page_id

        print(f"🔄 Converting User Access Token to Page Access Token...")
        print(f"📄 Target Page ID: {target_page_id}")

        # Get all pages managed by this user
        pages_url = f"https://graph.facebook.com/me/accounts?access_token={user_access_token}"
        response = requests.get(pages_url)

        if response.status_code != 200:
            print(f"❌ Error getting pages: {response.text}")
            return

        pages_data = response.json()
        pages = pages_data.get('data', [])

        print(f"\n✅ Found {len(pages)} managed pages:")

        page_token = None
        for page in pages:
            page_id = page.get('id')
            page_name = page.get('name')
            page_access_token = page.get('access_token')

            print(f"  📄 {page_name} (ID: {page_id})")

            if page_id == target_page_id:
                page_token = page_access_token
                print(f"    🎯 MATCH! This is our target page")
                print(f"    🔑 Page Token: {page_token[:20]}...")

        if page_token:
            print(f"\n🎉 SUCCESS! Found Page Access Token for page {target_page_id}")
            print(f"📋 Full Page Token: {page_token}")

            # Test the page token
            print(f"\n🧪 Testing the page token...")
            test_url = f"https://graph.facebook.com/{target_page_id}?access_token={page_token}"
            response = requests.get(test_url)

            if response.status_code == 200:
                page_info = response.json()
                print(f"✅ Page token works! Page: {page_info.get('name')}")

                print(f"\n📝 UPDATE YOUR .ENV FILE:")
                print(f"FACEBOOK_LONG_ACCESS_TOKEN_RUPASHI='{page_token}'")

            else:
                print(f"❌ Page token test failed: {response.text}")

        else:
            print(f"\n❌ Could not find page access token for page {target_page_id}")
            print(f"   Make sure you have admin access to this page.")

    except ValueError as e:
        print(f"❌ Error loading profile: {e}")

if __name__ == "__main__":
    get_page_access_token()