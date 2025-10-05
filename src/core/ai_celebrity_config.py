"""
AI Celebrity Configuration Module

This module provides a robust, type-safe configuration system for AI-powered social media celebrities.
It defines the core data structures for celebrity profiles, image generation parameters, posting schedules,
and platform-specific configurations.

Design Choices:
- Dataclasses for immutability and automatic __init__, __repr__, __eq__
- Enums for type-safe, constrained choices (prevents invalid values)
- Optional types for nullable fields with explicit None defaults
- Separation of concerns: each dataclass handles one aspect of configuration
- Factory methods for common use cases and validation

Author: AI Creator Team
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)

class PostType(Enum):
    """Social media post types.
    
    Design Choice: Using Enum ensures type safety and prevents invalid post types.
    Instagram has different requirements for feed vs stories (aspect ratio, captions, etc.)
    """
    FEED = "FEED"
    STORY = "STORIES"

class ImageStyle(Enum):
    """Image generation style presets.
    
    Design Choice: Predefined styles ensure consistency across generated images
    and map to specific AI model parameters for optimal results.
    """
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    FASHION = "fashion"
    FITNESS = "fitness"
    LIFESTYLE = "lifestyle"
    TRAVEL = "travel"
    
    @classmethod
    def from_string(cls, style_str: str) -> 'ImageStyle':
        """Convert string to ImageStyle enum with fallback.
        
        Args:
            style_str: Style as string (case-insensitive)
            
        Returns:
            ImageStyle enum value, defaults to LIFESTYLE if invalid
        """
        try:
            return cls(style_str.lower())
        except (ValueError, AttributeError):
            logger.warning(f"Invalid style '{style_str}', defaulting to LIFESTYLE")
            return cls.LIFESTYLE

@dataclass
class CelebrityProfile:
    """Celebrity persona configuration.
    
    Design Choice: Immutable dataclass ensures profile consistency throughout the app lifecycle.
    All fields are required to ensure complete persona definition for AI generation.
    
    Attributes:
        name: Celebrity's display name (used in captions, logs)
        age: Age in years (affects image generation prompts)
        gender: Gender identity (male/female/non-binary, affects pronouns and styling)
        occupation: Professional role (e.g., "fitness influencer", "travel blogger")
        personality_traits: List of adjectives describing personality (min 3 recommended)
        interests: Topics/activities the celebrity focuses on (min 3 recommended)
        style_preferences: Fashion/aesthetic preferences (used in image generation)
        bio_description: One-sentence bio for social media profiles
    """
    name: str
    age: int
    gender: str
    occupation: str
    personality_traits: List[str]
    interests: List[str]
    style_preferences: List[str]
    bio_description: str
    
    def __post_init__(self) -> None:
        """Validate profile data after initialization.
        
        Design Choice: Fail-fast validation prevents runtime errors downstream.
        """
        if not self.name or not self.name.strip():
            raise ValueError("Celebrity name cannot be empty")
        
        if self.age < 18 or self.age > 100:
            raise ValueError(f"Age must be between 18 and 100, got {self.age}")
        
        if self.gender.lower() not in ['male', 'female', 'non-binary']:
            logger.warning(f"Unusual gender value: {self.gender}")
        
        if len(self.personality_traits) < 2:
            raise ValueError("At least 2 personality traits required for authentic persona")
        
        if len(self.interests) < 2:
            raise ValueError("At least 2 interests required for diverse content")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'occupation': self.occupation,
            'personality_traits': self.personality_traits,
            'interests': self.interests,
            'style_preferences': self.style_preferences,
            'bio_description': self.bio_description
        }

@dataclass
class ImageConditions:
    """Configurable parameters for AI image generation.
    
    Design Choice: Comprehensive parameters give fine-grained control over image output.
    Defaults are set for Instagram feed posts (1:1, high quality).
    
    Attributes:
        style: Visual style preset (affects AI model parameters)
        lighting: Lighting description (e.g., "golden hour", "studio lighting")
        location: Scene location (e.g., "modern apartment", "beach")
        outfit_style: Clothing description (e.g., "casual chic", "athletic wear")
        pose_description: Pose/body language (e.g., "confident and relaxed")
        mood: Emotional tone (e.g., "happy and inspiring", "serene")
        background: Background description (e.g., "clean and minimal")
        quality: Image quality level ("high" or "standard")
        aspect_ratio: Image dimensions ("1:1" for feed, "9:16" for stories)
    """
    style: ImageStyle
    lighting: str
    location: str
    outfit_style: str
    pose_description: str
    mood: str
    background: str
    quality: str = "high"  # Default: high quality for professional appearance
    aspect_ratio: str = "1:1"  # Default: square format for Instagram feed
    
    def __post_init__(self) -> None:
        """Validate image conditions.
        
        Design Choice: Validate aspect ratio to prevent API errors with Instagram.
        """
        valid_ratios = ["1:1", "9:16", "4:5", "16:9"]
        if self.aspect_ratio not in valid_ratios:
            logger.warning(
                f"Unusual aspect ratio '{self.aspect_ratio}'. "
                f"Recommended: {', '.join(valid_ratios)}"
            )
        
        if self.quality not in ["high", "standard"]:
            logger.warning(f"Quality should be 'high' or 'standard', got '{self.quality}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conditions to dictionary for serialization."""
        return {
            'style': self.style.value if isinstance(self.style, ImageStyle) else self.style,
            'lighting': self.lighting,
            'location': self.location,
            'outfit_style': self.outfit_style,
            'pose_description': self.pose_description,
            'mood': self.mood,
            'background': self.background,
            'quality': self.quality,
            'aspect_ratio': self.aspect_ratio
        }
    
@dataclass
class PostingSchedule:
    """Posting schedule configuration.
    
    Design Choice: Separate feed and story frequencies to match platform best practices.
    Instagram allows 25 posts/day total; this config helps stay within limits.
    
    Attributes:
        feed_posts_per_day: Number of feed posts per day (max 25 combined with stories)
        story_posts_per_day: Number of story posts per day
        posting_times: List of times in "HH:MM" 24-hour format (e.g., ["09:00", "18:30"])
        active_days: Days of week to post (e.g., ["Monday", "Tuesday", ...])
    """
    feed_posts_per_day: int
    story_posts_per_day: int
    posting_times: List[str]  # Format: "HH:MM" in 24-hour time
    active_days: List[str]    # Full day names: Monday, Tuesday, etc.
    
    def __post_init__(self) -> None:
        """Validate posting schedule.
        
        Design Choice: Enforce Instagram's 25 posts/day limit to prevent API errors.
        """
        total_posts = self.feed_posts_per_day + self.story_posts_per_day
        if total_posts > 25:
            raise ValueError(
                f"Total posts per day ({total_posts}) exceeds Instagram's limit of 25. "
                f"Reduce feed_posts_per_day ({self.feed_posts_per_day}) or "
                f"story_posts_per_day ({self.story_posts_per_day})"
            )
        
        if self.feed_posts_per_day < 0 or self.story_posts_per_day < 0:
            raise ValueError("Post counts cannot be negative")
        
        # Validate time format
        import re
        time_pattern = re.compile(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$')
        for time_str in self.posting_times:
            if not time_pattern.match(time_str):
                raise ValueError(
                    f"Invalid time format '{time_str}'. Use HH:MM in 24-hour format (e.g., '09:00', '18:30')"
                )
        
        # Validate day names
        valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in self.active_days:
            if day not in valid_days:
                raise ValueError(
                    f"Invalid day '{day}'. Use full day names: {', '.join(valid_days)}"
                )
    
@dataclass
class InstagramConfig:
    """Instagram Graph API configuration.
    
    Design Choice: Separate config class for platform credentials enables easy
    credential rotation and multi-platform support in the future.
    
    Security Note: Never hardcode these values. Load from environment variables.
    
    Attributes:
        access_token: Instagram Graph API access token (requires instagram_content_publish permission)
        instagram_business_id: Instagram Business Account ID
        facebook_page_id: Facebook Page ID linked to Instagram Business Account
    """
    access_token: str
    instagram_business_id: str
    facebook_page_id: str
    
    def __post_init__(self) -> None:
        """Validate Instagram configuration.
        
        Design Choice: Warn about placeholder values to prevent accidental production use.
        """
        if 'YOUR_' in self.access_token or not self.access_token.strip():
            logger.warning(
                "Instagram access_token appears to be a placeholder. "
                "Set real credentials before posting."
            )
        
        if 'YOUR_' in self.instagram_business_id or not self.instagram_business_id.strip():
            logger.warning(
                "Instagram business_id appears to be a placeholder. "
                "Set real credentials before posting."
            )
    
    def is_configured(self) -> bool:
        """Check if credentials are properly configured (not placeholders).
        
        Returns:
            True if credentials appear valid, False otherwise
        """
        return (
            self.access_token and 
            'YOUR_' not in self.access_token and
            self.instagram_business_id and
            'YOUR_' not in self.instagram_business_id and
            self.facebook_page_id and
            'YOUR_' not in self.facebook_page_id
        )
    
class AIInstagramCelebrity:
    """Main configuration class for AI Instagram Celebrity system.
    
    Design Choice: Central configuration object that aggregates all settings.
    Provides sensible defaults while allowing full customization.
    
    This class serves as the single source of truth for:
    - Celebrity persona (profile)
    - Image generation defaults (default_image_conditions)
    - Posting schedule (posting_schedule)
    - Platform credentials (instagram_config)
    
    Usage:
        # Use defaults
        celebrity = AIInstagramCelebrity()
        
        # Customize via methods
        celebrity.update_celebrity_profile(name="Alex", age=28)
        
        # Or load from JSON config file
        celebrity = AIInstagramCelebrity.from_config_file("config.json")
    """
    
    def __init__(self) -> None:
        """Initialize with sensible defaults.
        
        Design Choice: Defaults create a working baseline that can be customized.
        Default persona is a generic lifestyle influencer - neutral starting point.
        """
        # Default celebrity profile - easily customizable
        self.celebrity = CelebrityProfile(
            name="Aria Sterling",
            age=25,
            gender="female",
            occupation="lifestyle influencer",
            personality_traits=["confident", "creative", "inspiring", "authentic"],
            interests=["fashion", "travel", "wellness", "photography"],
            style_preferences=["minimalist", "elegant", "trendy"],
            bio_description="25-year-old lifestyle influencer sharing daily inspiration through fashion, travel, and wellness content"
        )
        
        # Default image conditions - easily customizable
        self.default_image_conditions = ImageConditions(
            style=ImageStyle.LIFESTYLE,
            lighting="natural daylight",
            location="modern apartment",
            outfit_style="casual chic",
            pose_description="confident and relaxed",
            mood="happy and inspiring",
            background="clean and minimal",
            quality="high",
            aspect_ratio="1:1"
        )
        
        # Posting schedule - easily customizable
        self.posting_schedule = PostingSchedule(
            feed_posts_per_day=1,
            story_posts_per_day=3,
            posting_times=["09:00", "14:00", "19:00"],
            active_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        
        # Instagram API config - to be filled with actual credentials
        self.instagram_config = InstagramConfig(
            access_token="YOUR_ACCESS_TOKEN_HERE",
            instagram_business_id="YOUR_INSTAGRAM_BUSINESS_ID_HERE",
            facebook_page_id="YOUR_FACEBOOK_PAGE_ID_HERE"
        )
        
    def update_celebrity_profile(self, **kwargs) -> None:
        """Update celebrity profile parameters.
        
        Design Choice: Allows partial updates without recreating entire profile.
        Only updates fields that exist in CelebrityProfile to prevent typos.
        
        Args:
            **kwargs: Field names and values to update (e.g., name="Alex", age=28)
            
        Raises:
            AttributeError: If attempting to set non-existent field
            
        Example:
            celebrity.update_celebrity_profile(
                name="Jordan Smith",
                age=30,
                personality_traits=["creative", "bold"]
            )
        """
        for key, value in kwargs.items():
            if hasattr(self.celebrity, key):
                setattr(self.celebrity, key, value)
                logger.debug(f"Updated celebrity.{key} = {value}")
            else:
                raise AttributeError(
                    f"CelebrityProfile has no attribute '{key}'. "
                    f"Valid attributes: {', '.join(self.celebrity.__dataclass_fields__.keys())}"
                )
                
    def update_image_conditions(self, **kwargs) -> None:
        """Update default image generation conditions.
        
        Design Choice: Allows tweaking image defaults without recreating object.
        
        Args:
            **kwargs: Field names and values to update
            
        Example:
            celebrity.update_image_conditions(
                lighting="golden hour",
                location="beach",
                mood="serene and peaceful"
            )
        """
        for key, value in kwargs.items():
            if hasattr(self.default_image_conditions, key):
                # Special handling for style enum
                if key == 'style' and isinstance(value, str):
                    value = ImageStyle.from_string(value)
                setattr(self.default_image_conditions, key, value)
                logger.debug(f"Updated image_conditions.{key} = {value}")
            else:
                raise AttributeError(
                    f"ImageConditions has no attribute '{key}'. "
                    f"Valid attributes: {', '.join(self.default_image_conditions.__dataclass_fields__.keys())}"
                )
                
    def create_custom_image_conditions(self, **kwargs) -> ImageConditions:
        """Create custom image conditions for specific posts.
        
        Design Choice: Factory method for one-off image variations while keeping defaults intact.
        
        Args:
            **kwargs: All required ImageConditions fields plus optional overrides
            
        Returns:
            New ImageConditions instance
            
        Raises:
            TypeError: If required fields are missing
            
        Example:
            story_conditions = celebrity.create_custom_image_conditions(
                style=ImageStyle.TRAVEL,
                lighting="sunset glow",
                location="beach",
                outfit_style="summer dress",
                pose_description="walking on sand",
                mood="peaceful",
                background="ocean waves",
                aspect_ratio="9:16"  # Story format
            )
        """
        # Handle string style conversion
        if 'style' in kwargs and isinstance(kwargs['style'], str):
            kwargs['style'] = ImageStyle.from_string(kwargs['style'])
        
        try:
            conditions = ImageConditions(**kwargs)
            return conditions
        except TypeError as e:
            logger.error(f"Failed to create ImageConditions: {e}")
            raise
        
    def get_prompt_for_image_generation(
        self, 
        conditions: Optional[ImageConditions] = None, 
        use_llm: bool = True
    ) -> str:
        """Generate AI image prompt based on celebrity profile and conditions.
        
        Design Choice: Two-tier prompt generation strategy:
        1. LLM-enhanced (GPT-4) for rich, contextual prompts (preferred)
        2. Template-based fallback for reliability when LLM unavailable
        
        Args:
            conditions: Image generation parameters (uses defaults if None)
            use_llm: Whether to use LLM for enhancement (requires API key)
            
        Returns:
            Detailed prompt string for AI image generation
            
        Example:
            prompt = celebrity.get_prompt_for_image_generation(
                conditions=story_conditions,
                use_llm=True
            )
        """
        if conditions is None:
            conditions = self.default_image_conditions
            logger.debug("Using default image conditions for prompt generation")

        if use_llm:
            try:
                # Attempt LLM-enhanced prompt generation
                from prompt_generator import LLMPromptGenerator, CelebrityCharacteristics

                # Convert to LLM celebrity format
                llm_celebrity = CelebrityCharacteristics(
                    name=self.celebrity.name,
                    age=self.celebrity.age,
                    gender=self.celebrity.gender,
                    occupation=self.celebrity.occupation,
                    personality_traits=self.celebrity.personality_traits,
                    interests=self.celebrity.interests,
                    style_preferences=self.celebrity.style_preferences,
                    physical_features=["photogenic", "charismatic"],
                    fashion_style=conditions.outfit_style,
                    lifestyle=conditions.style.value
                )

                # Generate LLM prompt
                generator = LLMPromptGenerator()
                prompt = generator.generate_llm_prompt(
                    llm_celebrity, 
                    style=conditions.style.value, 
                    mood=conditions.mood
                )
                logger.info("Generated LLM-enhanced prompt")
                return prompt

            except ImportError as e:
                logger.warning(f"LLM prompt generator not available: {e}. Using fallback.")
                return self._get_fallback_prompt(conditions)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}. Using fallback.")
                return self._get_fallback_prompt(conditions)
        else:
            logger.debug("Using fallback prompt generation (LLM disabled)")
            return self._get_fallback_prompt(conditions)

    def _get_fallback_prompt(self, conditions: ImageConditions) -> str:
        """Fallback prompt generation using template-based approach.
        
        Design Choice: Template ensures consistent, working prompts when LLM unavailable.
        Structured format works well with DALL-E, Stable Diffusion, and similar models.
        
        Args:
            conditions: Image generation parameters
            
        Returns:
            Template-based prompt string
        """
        # Build structured prompt with all key parameters
        prompt_parts = [
            f"High-quality photograph of a {self.celebrity.age}-year-old {self.celebrity.gender} {self.celebrity.occupation}",
            f"personality: {', '.join(self.celebrity.personality_traits)}",
            f"style: {conditions.style.value}",
            f"lighting: {conditions.lighting}",
            f"location: {conditions.location}",
            f"outfit: {conditions.outfit_style}",
            f"pose: {conditions.pose_description}",
            f"mood: {conditions.mood}",
            f"background: {conditions.background}",
            f"quality: {conditions.quality}",
            f"aspect ratio: {conditions.aspect_ratio}"
        ]
        
        prompt = ", ".join(prompt_parts)
        logger.debug(f"Generated fallback prompt: {prompt[:100]}...")
        return prompt
    
    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> 'AIInstagramCelebrity':
        """Create AIInstagramCelebrity from configuration dictionary.
        
        Design Choice: Factory method for loading from JSON/dict configs.
        Enables config file-based initialization.
        
        Args:
            config: Dictionary with 'celebrity', 'image_conditions', 'posting_schedule' keys
            
        Returns:
            Configured AIInstagramCelebrity instance
            
        Example:
            with open('config.json') as f:
                config = json.load(f)
            celebrity = AIInstagramCelebrity.from_config_dict(config)
        """
        instance = cls()
        
        # Update celebrity profile
        if 'celebrity' in config:
            for key, value in config['celebrity'].items():
                if hasattr(instance.celebrity, key):
                    setattr(instance.celebrity, key, value)
        
        # Update image conditions
        if 'image_conditions' in config:
            for key, value in config['image_conditions'].items():
                if hasattr(instance.default_image_conditions, key):
                    if key == 'style' and isinstance(value, str):
                        value = ImageStyle.from_string(value)
                    setattr(instance.default_image_conditions, key, value)
        
        # Update posting schedule
        if 'posting_schedule' in config:
            for key, value in config['posting_schedule'].items():
                if hasattr(instance.posting_schedule, key):
                    setattr(instance.posting_schedule, key, value)
        
        # Update Instagram config
        if 'instagram_config' in config:
            for key, value in config['instagram_config'].items():
                if hasattr(instance.instagram_config, key):
                    setattr(instance.instagram_config, key, value)
        
        logger.info(f"Loaded configuration for celebrity: {instance.celebrity.name}")
        return instance

# Example usage and quick customization
if __name__ == "__main__":
    # Initialize with default settings
    ai_celebrity = AIInstagramCelebrity()
    
    # Easy customization examples:
    
    # Change celebrity persona
    ai_celebrity.update_celebrity_profile(
        name="Luna Martinez",
        age=28,
        occupation="fitness influencer",
        personality_traits=["energetic", "motivational", "dedicated"],
        interests=["fitness", "nutrition", "mental health"]
    )
    
    # Change default image style
    ai_celebrity.update_image_conditions(
        style=ImageStyle.FITNESS,
        lighting="gym lighting",
        location="modern fitness studio",
        outfit_style="athletic wear",
        mood="energetic and focused"
    )
    
    # Create custom conditions for a specific post
    custom_conditions = ai_celebrity.create_custom_image_conditions(
        style=ImageStyle.TRAVEL,
        lighting="golden hour",
        location="beach sunset",
        outfit_style="summer dress",
        pose_description="walking on beach",
        mood="serene and peaceful",
        background="ocean sunset landscape"
    )
    
    print("Default prompt:", ai_celebrity.get_prompt_for_image_generation())
    print("Custom prompt:", ai_celebrity.get_prompt_for_image_generation(custom_conditions))