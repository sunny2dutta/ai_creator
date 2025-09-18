import requests
import base64
import os
import asyncio
import fal_client
from PIL import Image
import io
from dotenv import load_dotenv
from prompt_generator import FacebookImagePromptGenerator

load_dotenv()

def test_stability_api():
    """Test Stability AI API directly"""
    print("Testing Stability AI API...")

    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key or api_key == "your_stability_ai_api_key_here":
        print("Stability API key not configured")
        return None

    # Generate prompt using prompt generator
    generator = FacebookImagePromptGenerator()
    prompt, caption = generator.generate_prompt()
    print(f"Using generated prompt: {prompt}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "accept": "image/*"
    }

    # Use files parameter for multipart/form-data
    files = {
        "prompt": (None, prompt),
        "output_format": (None, "jpeg")
    }

    try:
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/core",
            headers=headers,
            files=files
        )

        if response.status_code == 200:
            # v2beta API returns image directly as binary data
            image = Image.open(io.BytesIO(response.content))
            image.save("stability_test.jpg")
            print("Stability AI image saved as stability_test.jpg")
            return image
        else:
            print(f"L Stability API error: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"L Error: {e}")
        return None

async def test_seedream_api():
    """Test seedream API directly"""
    print("Testing seedream API...")

    api_key = os.getenv("FAL_API_KEY")
    if not api_key:
        print("L FAL API key not configured")
        return None

    # Generate prompt using prompt generator
    generator = FacebookImagePromptGenerator()
    prompt, caption = generator.generate_prompt()
    print(f"Using generated prompt: {prompt}")

    try:
        client = fal_client.AsyncClient(key=api_key)
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
            image = Image.open(io.BytesIO(img_response.content))
            image.save("seedream_test.jpg")
            print(" Seedream image saved as seedream_test.jpg")
            return image
        else:
            print("L No images returned from seedream")
            return None

    except Exception as e:
        print(f"L Error: {e}")
        return None

async def main():
    print("Testing APIs...")

    # Test Stability AI
    stability_image = test_stability_api()

    # Test seedream
    seedream_image = await test_seedream_api()

    if stability_image:
        print(" Stability AI working")
    if seedream_image:
        print(" Seedream working")

    if not stability_image and not seedream_image:
        print("L Both APIs failed")

if __name__ == "__main__":
    asyncio.run(main())