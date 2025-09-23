#!/usr/bin/env python3
import requests

def test_with_working_token():
    """Test with the working token from your curl command"""

    working_token = "EAAdDeYTb4BMBPUqHkL5PVroKNQMoI3tZAVOg5a2dyuZA7IdDyqL1QsQZA1P8uT1AZCS4yYYwhGQpmQGk81wAeEPwtOWvqPxYtol4Q371QCl0pbgAlbRNP963DZAJHXtxZCDRdBZC24lamGiHaFZCcLhkwsC6s4mCSzFkZB9rQ41G6sG6ZAWvrjZA9lBQiySroNjVN1VPNCKowZDZD"
    page_id = "768063046392563"

    api_url = f'https://graph.facebook.com/v23.0/{page_id}/photos'

    print("="*60)
    print("TEST 1: Using working token with URL method (like curl)")
    print("="*60)

    data = {
        "url": "https://picsum.photos/600/400",
        "caption": "Test from Python with working token 🚀",
        "access_token": working_token
    }

    response = requests.post(api_url, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code == 200:
        result = response.json()
        print(f"✅ SUCCESS! Post ID: {result.get('id')}")
    else:
        print(f"❌ Failed with status {response.status_code}")

if __name__ == "__main__":
    test_with_working_token()