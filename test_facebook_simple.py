import requests
import os
from dotenv import load_dotenv

load_dotenv()

page_id = os.environ["FACEBOOK_PAGE_ID"]
# Use the working token from your curl command
access_token = "EAAdDeYTb4BMBPUqHkL5PVroKNQMoI3tZAVOg5a2dyuZA7IdDyqL1QsQZA1P8uT1AZCS4yYYwhGQpmQGk81wAeEPwtOWvqPxYtol4Q371QCl0pbgAlbRNP963DZAJHXtxZCDRdBZC24lamGiHaFZCcLhkwsC6s4mCSzFkZB9rQ41G6sG6ZAWvrjZA9lBQiySroNjVN1VPNCKowZDZD"

# Replicate your exact working curl command
url = f"https://graph.facebook.com/v23.0/{page_id}/photos"

data = {
    'url': 'https://picsum.photos/600/400',
    'caption': 'Hello from Python Graph API 🚀',
    'access_token': access_token
}

data = {
    'url': 'https://picsum.photos/600/400',
    'caption': 'Hello from Python Graph API 🚀',
    'access_token': access_token
}

response = requests.post(url, data=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")