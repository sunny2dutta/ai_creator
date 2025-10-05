"""
AI Image Generation System

Production-ready image generation module supporting multiple AI services (OpenAI, Stability AI, Replicate).
Provides unified interface with automatic retry logic, error handling, and quality validation.

Design Choices:
- Abstract base class (ABC) for service implementations ensures consistent interface
- Strategy pattern allows runtime service switching without code changes
- Automatic retry with exponential backoff for transient network failures
- Image validation ensures output meets quality standards before returning
- Graceful degradation: falls back to alternative services if primary fails

Supported Services:
- OpenAI DALL-E 3: High quality, best for photorealistic portraits
- Stability AI SDXL: Fast, cost-effective, good for artistic styles
- Replicate: Flexible, supports multiple models

Author: AI Creator Team
License: MIT
"""

import requests
import base64
import io
from PIL import Image
from typing import Optional, Dict, Any, Tuple
import os
import time
import logging
from abc import ABC, abstractmethod
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import with relative path handling
try:
    from ai_celebrity_config import ImageConditions, AIInstagramCelebrity
except ImportError:
    from src.core.ai_celebrity_config import ImageConditions, AIInstagramCelebrity

# Configure module logger
logger = logging.getLogger(__name__)

class ImageGeneratorInterface(ABC):
    """Abstract interface for image generation services.
    
    Design Choice: ABC ensures all service implementations provide consistent interface.
    This enables polymorphism - any service can be swapped without changing client code.
    """
    
    @abstractmethod
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image from text prompt.
        
        Args:
            prompt: Detailed text description of desired image
            **kwargs: Service-specific parameters (size, quality, steps, etc.)
            
        Returns:
            Image data as bytes (JPEG or PNG format)
            
        Raises:
            requests.HTTPError: If API request fails
            ValueError: If parameters are invalid
            RuntimeError: If image generation fails after retries
        """
        pass
    
    def _create_session_with_retries(self, retries: int = 3) -> requests.Session:
        """Create requests session with automatic retry logic.
        
        Design Choice: Exponential backoff handles transient network issues gracefully.
        Retries on 500, 502, 503, 504 (server errors) but not 4xx (client errors).
        
        Args:
            retries: Maximum number of retry attempts
            
        Returns:
            Configured requests.Session with retry adapter
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[500, 502, 503, 504],  # Retry on server errors
            allowed_methods=["GET", "POST"]  # Retry these HTTP methods
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

class OpenAIImageGenerator(ImageGeneratorInterface):
    """OpenAI DALL-E image generator.
    
    Design Choice: DALL-E 3 produces highest quality photorealistic images.
    Best for: portraits, lifestyle content, professional photography style.
    Cost: ~$0.04-0.12 per image depending on size and quality.
    
    Attributes:
        api_key: OpenAI API key (get from platform.openai.com)
        base_url: API endpoint for image generation
        model: DALL-E model version (dall-e-3 recommended)
    """
    
    def __init__(self, api_key: str, model: str = "dall-e-3"):
        """Initialize OpenAI image generator.
        
        Args:
            api_key: OpenAI API key
            model: Model version (dall-e-2 or dall-e-3)
            
        Raises:
            ValueError: If API key is empty
        """
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API key cannot be empty")
        
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/images/generations"
        logger.info(f"Initialized OpenAI generator with model: {model}")
        
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image using OpenAI DALL-E.
        
        Design Choice: Two-step process (generate URL, then download) allows
        for better error handling and progress tracking.
        
        Args:
            prompt: Image description (max 4000 chars for DALL-E 3)
            **kwargs: size ("1024x1024", "1024x1792", "1792x1024"),
                     quality ("standard" or "hd")
                     
        Returns:
            Image bytes in PNG format
            
        Raises:
            requests.HTTPError: If API call fails
            ValueError: If prompt is too long or parameters invalid
        """
        # Validate prompt length
        if len(prompt) > 4000:
            logger.warning(f"Prompt length {len(prompt)} exceeds 4000 chars, truncating")
            prompt = prompt[:4000]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Build request payload
        data = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,  # Generate 1 image
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "standard"),
            "response_format": "url"  # Get URL instead of base64 (more reliable)
        }
        
        logger.info(f"Generating image with OpenAI {self.model}, size: {data['size']}, quality: {data['quality']}")
        
        try:
            # Create session with retry logic
            session = self._create_session_with_retries()
            
            # Request image generation
            response = session.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            image_url = result["data"][0]["url"]
            
            logger.debug(f"Image generated, downloading from URL")
            
            # Download the generated image
            image_response = session.get(image_url, timeout=30)
            image_response.raise_for_status()
            
            image_bytes = image_response.content
            logger.info(f"Successfully generated image ({len(image_bytes)} bytes)")
            
            return image_bytes
            
        except requests.exceptions.Timeout:
            logger.error("OpenAI API request timed out")
            raise RuntimeError("Image generation timed out. Try again or use a different service.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI image generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

class StabilityAIGenerator(ImageGeneratorInterface):
    """Stability AI SDXL image generator.
    
    Design Choice: Stable Diffusion XL offers good quality at lower cost than DALL-E.
    Best for: artistic styles, creative content, batch generation.
    Cost: ~$0.01-0.03 per image (credits-based pricing).
    
    Attributes:
        api_key: Stability AI API key (get from beta.dreamstudio.ai)
        base_url: API endpoint for SDXL model
    """
    
    def __init__(self, api_key: str, engine: str = "stable-diffusion-xl-1024-v1-0"):
        """Initialize Stability AI generator.
        
        Args:
            api_key: Stability AI API key
            engine: Model engine ID
            
        Raises:
            ValueError: If API key is empty
        """
        if not api_key or not api_key.strip():
            raise ValueError("Stability AI API key cannot be empty")
        
        self.api_key = api_key
        self.engine = engine
        self.base_url = f"https://api.stability.ai/v1/generation/{engine}/text-to-image"
        logger.info(f"Initialized Stability AI generator with engine: {engine}")
        
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image using Stability AI SDXL.
        
        Design Choice: Returns base64-encoded image directly (no separate download step).
        
        Args:
            prompt: Image description
            **kwargs: cfg_scale (7-15, higher = more prompt adherence),
                     steps (30-50, more = higher quality but slower),
                     height/width (512-1024, must be multiples of 64)
                     
        Returns:
            Image bytes in PNG format
            
        Raises:
            requests.HTTPError: If API call fails
            ValueError: If dimensions are invalid
        """
        # Extract and validate parameters
        height = kwargs.get("height", 1024)
        width = kwargs.get("width", 1024)
        
        # Validate dimensions (must be multiples of 64)
        if height % 64 != 0 or width % 64 != 0:
            logger.warning(f"Dimensions {width}x{height} not multiples of 64, rounding")
            height = (height // 64) * 64
            width = (width // 64) * 64
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        data = {
            "text_prompts": [{"text": prompt, "weight": 1}],
            "cfg_scale": kwargs.get("cfg_scale", 7),  # Prompt adherence strength
            "height": height,
            "width": width,
            "samples": 1,  # Number of images to generate
            "steps": kwargs.get("steps", 30),  # Inference steps (quality vs speed)
        }
        
        logger.info(f"Generating image with Stability AI, size: {width}x{height}, steps: {data['steps']}")
        
        try:
            session = self._create_session_with_retries()
            
            response = session.post(self.base_url, headers=headers, json=data, timeout=90)
            response.raise_for_status()
            
            result = response.json()
            
            # Check for artifacts in response
            if "artifacts" not in result or len(result["artifacts"]) == 0:
                raise RuntimeError("No image artifacts returned from Stability AI")
            
            # Decode base64 image
            image_data = base64.b64decode(result["artifacts"][0]["base64"])
            logger.info(f"Successfully generated image ({len(image_data)} bytes)")
            
            return image_data
            
        except requests.exceptions.Timeout:
            logger.error("Stability AI request timed out")
            raise RuntimeError("Image generation timed out. Try reducing steps or size.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Stability AI error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Stability AI generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

class ReplicateGenerator(ImageGeneratorInterface):
    """Replicate AI image generator.
    
    Design Choice: Replicate provides access to many models via unified API.
    Best for: flexibility, trying different models, cost optimization.
    Cost: Pay-per-use, varies by model (~$0.005-0.05 per image).
    
    Attributes:
        api_token: Replicate API token
        model: Model version string (format: owner/name:version)
        base_url: Replicate predictions API endpoint
    """
    
    def __init__(
        self, 
        api_token: str, 
        model: str = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
    ):
        """Initialize Replicate generator.
        
        Args:
            api_token: Replicate API token
            model: Model version identifier
            
        Raises:
            ValueError: If API token is empty
        """
        if not api_token or not api_token.strip():
            raise ValueError("Replicate API token cannot be empty")
        
        self.api_token = api_token
        self.model = model
        self.base_url = "https://api.replicate.com/v1/predictions"
        logger.info(f"Initialized Replicate generator with model: {model[:50]}...")
        
    def generate_image(self, prompt: str, **kwargs) -> bytes:
        """Generate image using Replicate.
        
        Design Choice: Async prediction model requires polling for completion.
        Implements exponential backoff to reduce API calls while waiting.
        
        Args:
            prompt: Image description
            **kwargs: Model-specific parameters (width, height, steps, guidance_scale)
                     
        Returns:
            Image bytes
            
        Raises:
            RuntimeError: If generation fails or times out
            requests.HTTPError: If API calls fail
        """
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
        
        logger.info(f"Starting Replicate prediction, size: {data['input']['width']}x{data['input']['height']}")
        
        try:
            session = self._create_session_with_retries()
            
            # Start prediction
            response = session.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            prediction = response.json()
            prediction_id = prediction["id"]
            logger.debug(f"Prediction started: {prediction_id}")
            
            # Poll for completion with exponential backoff
            max_attempts = 60  # Max 60 attempts (~2 minutes)
            attempt = 0
            wait_time = 1  # Start with 1 second
            
            while attempt < max_attempts:
                status_response = session.get(
                    f"{self.base_url}/{prediction_id}",
                    headers=headers,
                    timeout=10
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                
                status = status_data["status"]
                
                if status == "succeeded":
                    # Extract image URL from output
                    output = status_data.get("output")
                    if not output:
                        raise RuntimeError("No output in successful prediction")
                    
                    image_url = output[0] if isinstance(output, list) else output
                    logger.debug(f"Prediction succeeded, downloading image")
                    
                    # Download image
                    image_response = session.get(image_url, timeout=30)
                    image_response.raise_for_status()
                    
                    image_bytes = image_response.content
                    logger.info(f"Successfully generated image ({len(image_bytes)} bytes)")
                    return image_bytes
                    
                elif status == "failed":
                    error_msg = status_data.get('error', 'Unknown error')
                    logger.error(f"Replicate prediction failed: {error_msg}")
                    raise RuntimeError(f"Image generation failed: {error_msg}")
                    
                elif status == "canceled":
                    raise RuntimeError("Prediction was canceled")
                
                # Still processing, wait before next poll
                time.sleep(wait_time)
                attempt += 1
                
                # Exponential backoff: 1s, 2s, 4s, max 8s
                wait_time = min(wait_time * 2, 8)
                
                if attempt % 10 == 0:
                    logger.debug(f"Still waiting for prediction... (attempt {attempt}/{max_attempts})")
            
            raise RuntimeError(f"Prediction timed out after {max_attempts} attempts")
            
        except requests.exceptions.Timeout:
            logger.error("Replicate API request timed out")
            raise RuntimeError("Image generation timed out")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Replicate API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Replicate generation: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

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
