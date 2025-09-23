#!/usr/bin/env python3
import os
import sys
import requests
from dotenv import load_dotenv

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))
from profile_manager import ProfileManager

load_dotenv()

def debug_token_permissions():
    """Debug what permissions the current token has"""

    # Initialize profile manager
    profile_manager = ProfileManager()
    try:
        profile_config = profile_manager.get_profile('rupashi')
        access_token = profile_config.access_token
        page_id = profile_config.facebook_page_id

        print(f"🔍 Debugging token permissions...")
        print(f"📄 Page ID: {page_id}")
        print(f"🔑 Access Token: {access_token[:20]}...")

        # Test 1: Check what permissions this token has
        print("\n" + "="*60)
        print("TEST 1: Token Permissions")
        print("="*60)

        permissions_url = f"https://graph.facebook.com/me/permissions?access_token={access_token}"
        response = requests.get(permissions_url)

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            permissions = response.json()
            print("✅ Current permissions:")
            for perm in permissions.get('data', []):
                status = perm.get('status', 'unknown')
                permission = perm.get('permission', 'unknown')
                print(f"  - {permission}: {status}")
        else:
            print(f"❌ Error getting permissions: {response.text}")

        # Test 2: Check token info
        print("\n" + "="*60)
        print("TEST 2: Token Info")
        print("="*60)

        token_info_url = f"https://graph.facebook.com/debug_token?input_token={access_token}&access_token={access_token}"
        response = requests.get(token_info_url)

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            token_info = response.json()
            data = token_info.get('data', {})
            print(f"✅ Token type: {data.get('type', 'unknown')}")
            print(f"✅ App ID: {data.get('app_id', 'unknown')}")
            print(f"✅ Valid: {data.get('is_valid', 'unknown')}")
            print(f"✅ Scopes: {data.get('scopes', [])}")
        else:
            print(f"❌ Error getting token info: {response.text}")

        # Test 3: Try different API version
        print("\n" + "="*60)
        print("TEST 3: Try API v21.0 instead of v23.0")
        print("="*60)

        api_url_v21 = f'https://graph.facebook.com/v21.0/{page_id}/photos'
        print(f"🚀 Testing v21.0: {api_url_v21}")

        # Just test with a HEAD request to check if endpoint is accessible
        response = requests.head(api_url_v21, params={"access_token": access_token})
        print(f"HEAD request status: {response.status_code}")

        # Test 4: Check if this is a User token vs Page token
        print("\n" + "="*60)
        print("TEST 4: Check Token Type")
        print("="*60)

        me_url = f"https://graph.facebook.com/me?access_token={access_token}"
        response = requests.get(me_url)

        if response.status_code == 200:
            me_data = response.json()
            print(f"✅ Token belongs to: {me_data.get('name', 'unknown')}")
            print(f"✅ ID: {me_data.get('id', 'unknown')}")

            # Check if this ID matches the page ID - if so, it's a page token
            if me_data.get('id') == page_id:
                print("✅ This is a PAGE ACCESS TOKEN")
            else:
                print("⚠️  This is a USER ACCESS TOKEN")
                print("   You may need a Page Access Token for posting")
        else:
            print(f"❌ Error checking token owner: {response.text}")

    except ValueError as e:
        print(f"❌ Error loading profile: {e}")

if __name__ == "__main__":
    debug_token_permissions()