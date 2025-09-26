"""
AI Image Post-Processing System
Adds realistic enhancements to AI-generated images to make them look more authentic
"""

import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional, Tuple
import random

class ImagePostProcessor:
    """Post-processes AI images to make them look more realistic"""

    def __init__(self):
        pass

    def process_image(self, image_bytes: bytes,
                     apply_color_grading: bool = True,
                     add_noise: bool = True,
                     sharpen_edges: bool = True,
                     apply_lightroom_filter: bool = True) -> bytes:
        """Apply complete post-processing pipeline to image"""

        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply processing steps in order
        if apply_color_grading:
            image = self._apply_color_grading(image)

        if apply_lightroom_filter:
            image = self._apply_lightroom_style_filter(image)

        if sharpen_edges:
            image = self._sharpen_edges_selectively(image)

        if add_noise:
            image = self._add_realistic_noise(image)

        # Convert back to bytes
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        return output.getvalue()

    def _apply_color_grading(self, image: Image.Image) -> Image.Image:
        """Apply color grading to match realistic skin tones and lighting"""

        # Convert to numpy array for manipulation
        img_array = np.array(image)

        # Apply subtle color corrections channel by channel
        # Warm up shadows slightly (common in portrait photography)
        shadows_mask = img_array < 128
        img_array[:, :, 0][shadows_mask[:, :, 0]] = np.clip(img_array[:, :, 0][shadows_mask[:, :, 0]] * 1.05, 0, 255)
        img_array[:, :, 1][shadows_mask[:, :, 1]] = np.clip(img_array[:, :, 1][shadows_mask[:, :, 1]] * 1.02, 0, 255)
        img_array[:, :, 2][shadows_mask[:, :, 2]] = np.clip(img_array[:, :, 2][shadows_mask[:, :, 2]] * 0.98, 0, 255)

        # Cool down highlights slightly
        highlights_mask = img_array > 180
        img_array[:, :, 0][highlights_mask[:, :, 0]] = np.clip(img_array[:, :, 0][highlights_mask[:, :, 0]] * 0.98, 0, 255)
        img_array[:, :, 1][highlights_mask[:, :, 1]] = np.clip(img_array[:, :, 1][highlights_mask[:, :, 1]] * 0.99, 0, 255)
        img_array[:, :, 2][highlights_mask[:, :, 2]] = np.clip(img_array[:, :, 2][highlights_mask[:, :, 2]] * 1.02, 0, 255)

        # Enhance skin tone regions (orange/yellow tones)
        # Create mask for skin-like colors
        skin_mask = (
            (img_array[:, :, 0] > img_array[:, :, 2]) &  # More red than blue
            (img_array[:, :, 1] > img_array[:, :, 2] * 0.8) &  # Some yellow
            (np.sum(img_array, axis=2) > 300)  # Not too dark
        )

        # Slightly enhance skin tones
        img_array[:, :, 0][skin_mask] = np.clip(img_array[:, :, 0][skin_mask] * 1.02, 0, 255)
        img_array[:, :, 1][skin_mask] = np.clip(img_array[:, :, 1][skin_mask] * 1.01, 0, 255)
        img_array[:, :, 2][skin_mask] = np.clip(img_array[:, :, 2][skin_mask] * 0.99, 0, 255)

        processed_image = Image.fromarray(img_array)

        # Fine-tune overall color balance
        enhancer = ImageEnhance.Color(processed_image)
        processed_image = enhancer.enhance(1.05)  # Slight saturation boost

        return processed_image

    def _add_realistic_noise(self, image: Image.Image) -> Image.Image:
        """Add subtle film grain or sensor noise for authenticity"""

        img_array = np.array(image).astype(np.float32)

        # Create noise pattern
        height, width = img_array.shape[:2]

        # Generate different types of noise
        # 1. Fine grain (like film grain)
        fine_noise = np.random.normal(0, 2.5, (height, width, 3))

        # 2. Larger grain for texture
        coarse_h, coarse_w = height//4, width//4
        coarse_noise = np.random.normal(0, 1.0, (coarse_h, coarse_w, 3))
        coarse_noise = np.repeat(np.repeat(coarse_noise, 4, axis=0), 4, axis=1)
        # Ensure coarse_noise matches exact dimensions
        coarse_noise = coarse_noise[:height, :width, :3]

        # 3. Color noise (slight color variations)
        color_noise = np.random.normal(0, 0.8, (height, width, 3))
        color_noise[:, :, 0] *= 0.7  # Less red noise
        color_noise[:, :, 2] *= 1.2  # More blue noise (like digital cameras)

        # Combine noise types
        total_noise = fine_noise + coarse_noise * 0.3 + color_noise

        # Apply noise with varying intensity based on brightness
        # More noise in shadows, less in highlights (realistic camera behavior)
        brightness = np.mean(img_array, axis=2, keepdims=True)
        noise_intensity = np.clip((255 - brightness) / 255 * 1.5 + 0.2, 0.2, 1.0)

        # Apply noise
        img_array += total_noise * noise_intensity

        # Clip values
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def _sharpen_edges_selectively(self, image: Image.Image) -> Image.Image:
        """Sharpen edges selectively to fix AI's fuzzy boundaries"""

        # Create edge mask
        # Convert to grayscale for edge detection
        gray = image.convert('L')

        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)

        # Create edge mask - stronger sharpening where edges are detected
        edge_mask = edge_array / 255.0

        # Apply different levels of sharpening
        # Light sharpening for overall image
        light_sharp = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=3))

        # Strong sharpening for edges
        strong_sharp = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=1))

        # Blend based on edge mask
        img_array = np.array(image).astype(np.float32)
        light_array = np.array(light_sharp).astype(np.float32)
        strong_array = np.array(strong_sharp).astype(np.float32)

        # Expand edge mask to 3 channels
        edge_mask_3d = np.stack([edge_mask, edge_mask, edge_mask], axis=2)

        # Blend: more edge sharpening where edges are detected
        result_array = (
            img_array * (1 - edge_mask_3d * 0.3) +
            light_array * (1 - edge_mask_3d) * 0.3 +
            strong_array * edge_mask_3d * 0.3
        )

        result_array = np.clip(result_array, 0, 255).astype(np.uint8)

        return Image.fromarray(result_array)

    def _apply_lightroom_style_filter(self, image: Image.Image) -> Image.Image:
        """Apply Lightroom-style filters to mimic camera output"""

        # Apply a combination of adjustments that mimic popular Lightroom presets

        # 1. Tone curve adjustment (S-curve for contrast)
        img_array = np.array(image).astype(np.float32) / 255.0

        # Apply S-curve: darken shadows, brighten highlights slightly
        # This mimics film response and popular Instagram filters
        tone_curve = lambda x: np.where(
            x < 0.5,
            2 * x * x,  # Darken shadows
            1 - 2 * (1 - x) * (1 - x)  # Brighten highlights
        )

        img_array = tone_curve(img_array)

        # 2. Split toning (warm highlights, cool shadows)
        shadows_mask = img_array < 0.4
        highlights_mask = img_array > 0.7

        # Warm highlights - apply channel by channel
        img_array[:, :, 0][highlights_mask[:, :, 0]] *= 1.02
        img_array[:, :, 1][highlights_mask[:, :, 1]] *= 1.01
        img_array[:, :, 2][highlights_mask[:, :, 2]] *= 0.98

        # Cool shadows slightly - apply channel by channel
        img_array[:, :, 0][shadows_mask[:, :, 0]] *= 0.98
        img_array[:, :, 1][shadows_mask[:, :, 1]] *= 0.99
        img_array[:, :, 2][shadows_mask[:, :, 2]] *= 1.02

        # 3. Vibrance boost (more natural than saturation)
        # Convert to HSV for vibrance adjustment
        img_pil = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))

        # Apply vibrance (selective saturation boost)
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(1.08)

        # 4. Slight vignette effect (darkening corners)
        img_array = np.array(img_pil).astype(np.float32) / 255.0
        height, width = img_array.shape[:2]

        # Create vignette mask
        Y, X = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        # Distance from center, normalized
        max_dist = np.sqrt((height/2)**2 + (width/2)**2)
        distance = np.sqrt((Y - center_y)**2 + (X - center_x)**2) / max_dist

        # Very subtle vignette
        vignette = 1 - distance * 0.15
        vignette = np.clip(vignette, 0.85, 1.0)

        # Apply vignette
        for i in range(3):
            img_array[:, :, i] *= vignette

        # 5. Film emulation - slight color tint
        # Add very subtle warm tint (like Kodak Portra)
        img_array *= np.array([1.005, 1.002, 0.998])

        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def apply_instagram_style_filter(self, image_bytes: bytes, filter_name: str = "natural") -> bytes:
        """Apply Instagram-style filters"""

        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if filter_name == "natural":
            # Natural, subtle enhancement
            image = self._apply_lightroom_style_filter(image)
            image = self._apply_color_grading(image)

        elif filter_name == "warm":
            # Warm, golden hour look
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)

            # Apply warm tint
            img_array = np.array(image).astype(np.float32)
            img_array *= np.array([1.1, 1.05, 0.95])
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)

        elif filter_name == "moody":
            # Darker, more dramatic look
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.9)

            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)

            # Cool tint
            img_array = np.array(image).astype(np.float32)
            img_array *= np.array([0.95, 0.98, 1.05])
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)

        # Convert back to bytes
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        return output.getvalue()


# Convenience functions for easy use
def enhance_ai_image(image_bytes: bytes) -> bytes:
    """One-click enhancement for AI images"""
    processor = ImagePostProcessor()
    return processor.process_image(image_bytes)

def apply_portrait_enhancement(image_bytes: bytes) -> bytes:
    """Specialized enhancement for portrait images"""
    processor = ImagePostProcessor()
    return processor.process_image(
        image_bytes,
        apply_color_grading=True,
        add_noise=True,
        sharpen_edges=True,
        apply_lightroom_filter=True
    )

def apply_lifestyle_filter(image_bytes: bytes) -> bytes:
    """Apply lifestyle/influencer style filter"""
    processor = ImagePostProcessor()
    enhanced = processor.process_image(
        image_bytes,
        apply_color_grading=True,
        add_noise=False,  # Less noise for cleaner look
        sharpen_edges=True,
        apply_lightroom_filter=True
    )

    # Apply warm Instagram-style filter
    return processor.apply_instagram_style_filter(enhanced, "warm")