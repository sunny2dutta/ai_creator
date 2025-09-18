"""
Configuration system for AI Instagram Celebrity
Easily editable parameters for celebrity persona and posting conditions
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class PostType(Enum):
    FEED = "FEED"
    STORY = "STORIES"

class ImageStyle(Enum):
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    FASHION = "fashion"
    FITNESS = "fitness"
    LIFESTYLE = "lifestyle"
    TRAVEL = "travel"

@dataclass
class CelebrityProfile:
    """Celebrity persona configuration"""
    name: str
    age: int
    gender: str
    occupation: str
    personality_traits: List[str]
    interests: List[str]
    style_preferences: List[str]
    bio_description: str

@dataclass
class ImageConditions:
    """Configurable parameters for image generation"""
    style: ImageStyle
    lighting: str
    location: str
    outfit_style: str
    pose_description: str
    mood: str
    background: str
    quality: str = "high"
    aspect_ratio: str = "1:1"
    
@dataclass
class PostingSchedule:
    """Posting schedule configuration"""
    feed_posts_per_day: int
    story_posts_per_day: int
    posting_times: List[str]  # Format: "HH:MM"
    active_days: List[str]    # Days of week
    
@dataclass
class InstagramConfig:
    """Instagram API configuration"""
    access_token: str
    instagram_business_id: str
    facebook_page_id: str
    
class AIInstagramCelebrity:
    """Main configuration class for AI Instagram Celebrity system"""
    
    def __init__(self):
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
        
    def update_celebrity_profile(self, **kwargs):
        """Update celebrity profile parameters"""
        for key, value in kwargs.items():
            if hasattr(self.celebrity, key):
                setattr(self.celebrity, key, value)
                
    def update_image_conditions(self, **kwargs):
        """Update default image generation conditions"""
        for key, value in kwargs.items():
            if hasattr(self.default_image_conditions, key):
                setattr(self.default_image_conditions, key, value)
                
    def create_custom_image_conditions(self, **kwargs) -> ImageConditions:
        """Create custom image conditions for specific posts"""
        conditions = ImageConditions(**kwargs)
        return conditions
        
    def get_prompt_for_image_generation(self, conditions: Optional[ImageConditions] = None, use_llm: bool = True) -> str:
        """Generate AI image prompt based on celebrity profile and conditions"""
        if conditions is None:
            conditions = self.default_image_conditions

        if use_llm:
            try:
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
                return generator.generate_llm_prompt(llm_celebrity, style=conditions.style.value, mood=conditions.mood)

            except Exception as e:
                print(f"LLM generation failed, using fallback: {e}")
                return self._get_fallback_prompt(conditions)
        else:
            return self._get_fallback_prompt(conditions)

    def _get_fallback_prompt(self, conditions: ImageConditions) -> str:
        """Fallback prompt generation method"""
        prompt = f"High-quality photograph of a {self.celebrity.age}-year-old {self.celebrity.gender} {self.celebrity.occupation}, "
        prompt += f"personality: {', '.join(self.celebrity.personality_traits)}, "
        prompt += f"style: {conditions.style.value}, "
        prompt += f"lighting: {conditions.lighting}, "
        prompt += f"location: {conditions.location}, "
        prompt += f"outfit: {conditions.outfit_style}, "
        prompt += f"pose: {conditions.pose_description}, "
        prompt += f"mood: {conditions.mood}, "
        prompt += f"background: {conditions.background}, "
        prompt += f"aspect ratio: {conditions.aspect_ratio}, "
        prompt += f"quality: {conditions.quality}"

        return prompt

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