"""
AI Image Generation System for Instagram Celebrity
Supports multiple AI image generation services with configurable parameters
"""

import requests
import base64
import io
from PIL import Image
from typing import Optional, Dict, Any
import os
import time
from abc import ABC, abstractmethod
from ai_celebrity_config import ImageConditions, AIInstagramCelebrity

class ImageGeneratorInterface(ABC):
    """Abstract interface for image generation services"""
    
    @abstractmethod
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image from prompt and return image bytes"""
        pass

class OpenAIImageGenerator(ImageGeneratorInterface):
    """OpenAI DALL-E image generator"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/images/generations"
        
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image using OpenAI DALL-E"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "n": 1,
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "standard"),
            "response_format": "url"
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        image_url = result["data"][0]["url"]
        
        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        return image_response.content

class StabilityAIGenerator(ImageGeneratorInterface):
    """Stability AI image generator"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image using Stability AI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        data = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": kwargs.get("cfg_scale", 7),
            "height": kwargs.get("height", 1024),
            "width": kwargs.get("width", 1024),
            "samples": 1,
            "steps": kwargs.get("steps", 30),
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        image_data = base64.b64decode(result["artifacts"][0]["base64"])
        
        return image_data

class ReplicateGenerator(ImageGeneratorInterface):
    """Replicate AI image generator (supports various models)"""
    
    def __init__(self, api_token: str, model: str = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"):
        self.api_token = api_token
        self.model = model
        self.base_url = "https://api.replicate.com/v1/predictions"
        
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image using Replicate"""
        headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "version": self.model,
            "input": {
                "prompt": prompt,
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
                "num_inference_steps": kwargs.get("steps", 30),
                "guidance_scale": kwargs.get("guidance_scale", 7.5)
            }
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        prediction = response.json()
        prediction_id = prediction["id"]
        
        # Poll for completion
        while True:
            status_response = requests.get(
                f"{self.base_url}/{prediction_id}",
                headers=headers
            )
            status_response.raise_for_status()
            status_data = status_response.json()
            
            if status_data["status"] == "succeeded":
                image_url = status_data["output"][0]
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                return image_response.content
            elif status_data["status"] == "failed":
                raise Exception(f"Image generation failed: {status_data.get('error', 'Unknown error')}")
            
            time.sleep(2)

class AIImageGenerator:
    """Main image generator class that manages different AI services"""
    
    def __init__(self, service: str = "openai", **service_kwargs):
        self.service_name = service
        
        if service == "openai":
            api_key = service_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            self.generator = OpenAIImageGenerator(api_key)
        elif service == "stability":
            api_key = service_kwargs.get("api_key") or os.getenv("STABILITY_API_KEY")
            self.generator = StabilityAIGenerator(api_key)
        elif service == "replicate":
            api_token = service_kwargs.get("api_token") or os.getenv("REPLICATE_API_TOKEN")
            model = service_kwargs.get("model", "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b")
            self.generator = ReplicateGenerator(api_token, model)
        else:
            raise ValueError(f"Unsupported service: {service}")
    
    def generate_celebrity_image(self, celebrity_config: AIInstagramCelebrity, 
                               conditions: Optional[ImageConditions] = None,
                               save_path: Optional[str] = None) -> bytes:
        """Generate image for AI celebrity with specific conditions"""
        
        prompt = celebrity_config.get_prompt_for_image_generation(conditions)
        
        # Service-specific parameters based on conditions
        generation_params = {}
        
        if conditions:
            if conditions.aspect_ratio == "9:16":  # Instagram Stories
                if self.service_name == "openai":
                    generation_params["size"] = "1024x1792"
                else:
                    generation_params.update({"width": 1024, "height": 1792})
            elif conditions.aspect_ratio == "1:1":  # Instagram Feed
                if self.service_name == "openai":
                    generation_params["size"] = "1024x1024"
                else:
                    generation_params.update({"width": 1024, "height": 1024})
            
            if conditions.quality == "high":
                if self.service_name == "openai":
                    generation_params["quality"] = "hd"
                else:
                    generation_params["steps"] = 50
        
        # Generate the image
        image_bytes = self.generator.generate_image(prompt, **generation_params)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(image_bytes)
        
        return image_bytes
    
    def resize_for_instagram(self, image_bytes: bytes, post_type: str = "feed") -> bytes:
        """Resize image for Instagram requirements"""
        image = Image.open(io.BytesIO(image_bytes))
        
        if post_type == "feed":
            # Instagram feed: 1080x1080 (1:1 ratio)
            target_size = (1080, 1080)
        elif post_type == "story":
            # Instagram stories: 1080x1920 (9:16 ratio)
            target_size = (1080, 1920)
        else:
            return image_bytes
        
        # Resize maintaining aspect ratio, then crop if needed
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image centered
        new_image = Image.new("RGB", target_size, (255, 255, 255))
        
        # Center the image
        x = (target_size[0] - image.width) // 2
        y = (target_size[1] - image.height) // 2
        new_image.paste(image, (x, y))
        
        # Convert back to bytes
        output = io.BytesIO()
        new_image.save(output, format="JPEG", quality=95)
        return output.getvalue()

# Example usage
if __name__ == "__main__":
    # Initialize AI celebrity config
    celebrity = AIInstagramCelebrity()
    
    # Initialize image generator (replace with your API key)
    # For testing, you can use any of these services:
    # generator = AIImageGenerator("openai", api_key="your-openai-key")
    # generator = AIImageGenerator("stability", api_key="your-stability-key")  
    # generator = AIImageGenerator("replicate", api_token="your-replicate-token")
    
    # Example without actual API keys (will fail without real keys)
    try:
        generator = AIImageGenerator("openai")  # Will use OPENAI_API_KEY env var
        
        # Generate feed post image
        feed_conditions = celebrity.create_custom_image_conditions(
            style=celebrity.default_image_conditions.style,
            aspect_ratio="1:1"
        )
        
        feed_image = generator.generate_celebrity_image(
            celebrity, 
            feed_conditions,
            save_path="celebrity_feed_post.jpg"
        )
        
        # Generate story image  
        story_conditions = celebrity.create_custom_image_conditions(
            style=celebrity.default_image_conditions.style,
            aspect_ratio="9:16"
        )
        
        story_image = generator.generate_celebrity_image(
            celebrity,
            story_conditions, 
            save_path="celebrity_story.jpg"
        )
        
        print("Images generated successfully!")
        
    except Exception as e:
        print(f"Error generating images: {e}")
        print("Make sure to set your API keys as environment variables or pass them directly.")
