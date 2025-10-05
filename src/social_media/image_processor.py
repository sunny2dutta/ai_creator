"""
Image Processor

Handles image resizing, enhancement, and format conversion for social media platforms.
Integrates with the image post-processor for AI image enhancement.

Design Choices:
- Separate class for image operations (Single Responsibility)
- Platform-specific resize methods (feed vs story dimensions)
- Optional post-processing for flexibility
- Maintains aspect ratios and quality
- Defensive programming with validation

Author: AI Creator Team
License: MIT
"""

import io
import logging
from typing import Tuple, Optional
from PIL import Image
import sys
import os

# Import post-processor with fallback
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.image_post_processor import (
        ImagePostProcessor,
        enhance_ai_image,
        apply_portrait_enhancement,
        apply_lifestyle_filter
    )
    POST_PROCESSOR_AVAILABLE = True
except ImportError:
    POST_PROCESSOR_AVAILABLE = False
    logging.warning("Image post-processor not available. Enhancement disabled.")

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Processes images for social media posting.
    
    Design Choice: Centralized image processing eliminates code duplication.
    Optional post-processing allows performance tuning.
    
    Attributes:
        enable_post_processing: Whether to apply AI enhancement
        post_processor: ImagePostProcessor instance (if enabled)
    """
    
    # Platform-specific dimensions
    FEED_DIMENSIONS = (1080, 1080)  # 1:1 aspect ratio
    STORY_DIMENSIONS = (1080, 1920)  # 9:16 aspect ratio
    
    def __init__(self, enable_post_processing: bool = True):
        """Initialize image processor.
        
        Args:
            enable_post_processing: Enable AI-powered image enhancement
        """
        self.enable_post_processing = enable_post_processing and POST_PROCESSOR_AVAILABLE
        
        if self.enable_post_processing:
            self.post_processor = ImagePostProcessor()
            logger.info("Image processor initialized with post-processing enabled")
        else:
            self.post_processor = None
            if enable_post_processing and not POST_PROCESSOR_AVAILABLE:
                logger.warning("Post-processing requested but not available")
            else:
                logger.info("Image processor initialized without post-processing")
    
    def resize_for_feed(self, image: Image.Image) -> Image.Image:
        """Resize image for social media feed (1:1 aspect ratio).
        
        Design Choice: Square format works universally across platforms.
        Centers image on white background if aspect ratio doesn't match.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized image (1080x1080)
            
        Example:
            processor = ImageProcessor()
            feed_image = processor.resize_for_feed(original_image)
        """
        return self._resize_to_dimensions(
            image, 
            *self.FEED_DIMENSIONS,
            description="feed (1:1)"
        )
    
    def resize_for_story(self, image: Image.Image) -> Image.Image:
        """Resize image for social media stories (9:16 aspect ratio).
        
        Design Choice: Vertical format optimized for mobile viewing.
        Crops or pads to exact dimensions.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized image (1080x1920)
        """
        return self._resize_to_dimensions(
            image, 
            *self.STORY_DIMENSIONS,
            description="story (9:16)"
        )
    
    def _resize_to_dimensions(
        self, 
        image: Image.Image, 
        target_width: int, 
        target_height: int,
        description: str = ""
    ) -> Image.Image:
        """Resize image to exact dimensions while maintaining quality.
        
        Design Choice: Two-step process (resize + crop/pad) preserves quality.
        Uses LANCZOS resampling for best quality.
        
        Args:
            image: Source image
            target_width: Target width in pixels
            target_height: Target height in pixels
            description: Description for logging
            
        Returns:
            Resized and cropped/padded image
        """
        logger.debug(
            f"Resizing image from {image.size} to {target_width}x{target_height} "
            f"for {description}"
        )
        
        # Calculate aspect ratios
        img_ratio = image.width / image.height
        target_ratio = target_width / target_height
        
        # Resize maintaining aspect ratio
        if img_ratio > target_ratio:
            # Image is wider - fit to height
            new_height = target_height
            new_width = int(target_height * img_ratio)
        else:
            # Image is taller - fit to width
            new_width = target_width
            new_height = int(target_width / img_ratio)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create canvas with target dimensions
        canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        
        # Center the resized image on canvas
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Crop if image is larger than target
        if new_width > target_width or new_height > target_height:
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            resized = resized.crop((left, top, right, bottom))
            canvas.paste(resized, (0, 0))
        else:
            # Paste centered if image is smaller
            canvas.paste(resized, (x_offset, y_offset))
        
        logger.debug(f"Image resized successfully to {target_width}x{target_height}")
        return canvas
    
    def enhance(
        self, 
        image: Image.Image, 
        style: str = "lifestyle"
    ) -> Image.Image:
        """Apply AI-powered enhancement to image.
        
        Design Choice: Optional enhancement allows quality/performance tradeoff.
        Different styles for different content types.
        
        Args:
            image: PIL Image to enhance
            style: Enhancement style ("lifestyle", "portrait", or "natural")
            
        Returns:
            Enhanced image (or original if post-processing disabled)
            
        Raises:
            ValueError: If style is invalid
        """
        if not self.enable_post_processing:
            logger.debug("Post-processing disabled, returning original image")
            return image
        
        valid_styles = ["lifestyle", "portrait", "natural"]
        if style not in valid_styles:
            raise ValueError(
                f"Invalid style '{style}'. Must be one of: {', '.join(valid_styles)}"
            )
        
        logger.info(f"Applying {style} enhancement to image")
        
        try:
            # Convert PIL Image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG', quality=95)
            img_bytes.seek(0)
            image_data = img_bytes.getvalue()
            
            # Apply appropriate enhancement
            if style == "portrait":
                enhanced_bytes = apply_portrait_enhancement(image_data)
            elif style == "lifestyle":
                enhanced_bytes = apply_lifestyle_filter(image_data)
            else:  # natural
                enhanced_bytes = enhance_ai_image(image_data)
            
            # Convert back to PIL Image
            enhanced_image = Image.open(io.BytesIO(enhanced_bytes))
            
            logger.info(f"Successfully applied {style} enhancement")
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}. Returning original image.")
            return image
    
    def prepare_for_feed(
        self, 
        image: Image.Image, 
        enhance: bool = True
    ) -> Image.Image:
        """Complete preparation pipeline for feed posting.
        
        Design Choice: One-stop method combines enhancement and resizing.
        Convenience method for common workflow.
        
        Args:
            image: Source image
            enhance: Whether to apply enhancement
            
        Returns:
            Feed-ready image (enhanced and resized to 1080x1080)
        """
        logger.debug("Preparing image for feed posting")
        
        # Enhance first (works better on full-resolution image)
        if enhance and self.enable_post_processing:
            image = self.enhance(image, style="lifestyle")
        
        # Then resize
        return self.resize_for_feed(image)
    
    def prepare_for_story(
        self, 
        image: Image.Image, 
        enhance: bool = True
    ) -> Image.Image:
        """Complete preparation pipeline for story posting.
        
        Design Choice: Portrait-oriented enhancement for vertical format.
        
        Args:
            image: Source image
            enhance: Whether to apply enhancement
            
        Returns:
            Story-ready image (enhanced and resized to 1080x1920)
        """
        logger.debug("Preparing image for story posting")
        
        # Enhance with portrait style for stories
        if enhance and self.enable_post_processing:
            image = self.enhance(image, style="portrait")
        
        # Then resize
        return self.resize_for_story(image)
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, Optional[str]]:
        """Validate image meets minimum requirements.
        
        Design Choice: Pre-upload validation prevents API errors.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if image exists
        if image is None:
            return False, "Image is None"
        
        # Check minimum dimensions
        min_width, min_height = 320, 320
        if image.width < min_width or image.height < min_height:
            return False, f"Image too small: {image.size}. Minimum: {min_width}x{min_height}"
        
        # Check maximum dimensions (Instagram limit)
        max_dimension = 8192
        if image.width > max_dimension or image.height > max_dimension:
            return False, f"Image too large: {image.size}. Maximum: {max_dimension}x{max_dimension}"
        
        # Check format
        if image.mode not in ('RGB', 'RGBA', 'L'):
            return False, f"Unsupported image mode: {image.mode}"
        
        return True, None
