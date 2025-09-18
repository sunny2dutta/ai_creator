"""
Celebrity Factory - Minimal Friction AI Celebrity Creation
Pre-built celebrity templates and easy instantiation system
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import asdict
from ai_celebrity_config import AIInstagramCelebrity, CelebrityProfile, ImageConditions, ImageStyle, PostingSchedule

class CelebrityTemplate:
    """Pre-built celebrity template for easy instantiation"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
    
    def create_celebrity(self) -> AIInstagramCelebrity:
        """Create AI celebrity instance from template"""
        celebrity = AIInstagramCelebrity()
        
        # Update celebrity profile
        if "celebrity" in self.config:
            for key, value in self.config["celebrity"].items():
                if hasattr(celebrity.celebrity, key):
                    setattr(celebrity.celebrity, key, value)
        
        # Update image conditions
        if "image_conditions" in self.config:
            for key, value in self.config["image_conditions"].items():
                if hasattr(celebrity.default_image_conditions, key):
                    if key == "style" and isinstance(value, str):
                        setattr(celebrity.default_image_conditions, key, ImageStyle(value))
                    else:
                        setattr(celebrity.default_image_conditions, key, value)
        
        # Update posting schedule
        if "posting_schedule" in self.config:
            for key, value in self.config["posting_schedule"].items():
                if hasattr(celebrity.posting_schedule, key):
                    setattr(celebrity.posting_schedule, key, value)
        
        return celebrity

class CelebrityFactory:
    """Factory for creating AI celebrities with minimal friction"""
    
    def __init__(self):
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load pre-built celebrity templates"""
        
        # Fitness Influencer Template
        self.templates["fitness_influencer"] = CelebrityTemplate("Fitness Influencer", {
            "celebrity": {
                "name": "Alex Chen",
                "age": 27,
                "gender": "non-binary",
                "occupation": "fitness coach and motivational speaker",
                "personality_traits": ["energetic", "motivational", "disciplined", "inspiring", "positive"],
                "interests": ["strength training", "nutrition", "mental health", "outdoor activities", "wellness"],
                "style_preferences": ["athletic", "modern", "functional", "bold"],
                "bio_description": "Fitness coach helping people transform their lives through movement, mindset, and sustainable habits"
            },
            "image_conditions": {
                "style": "fitness",
                "lighting": "bright gym lighting",
                "location": "modern fitness studio",
                "outfit_style": "athletic wear",
                "pose_description": "strong and confident workout poses",
                "mood": "energetic and motivational",
                "background": "gym equipment and modern fitness space",
                "quality": "high"
            },
            "posting_schedule": {
                "feed_posts_per_day": 2,
                "story_posts_per_day": 4,
                "posting_times": ["06:30", "12:00", "17:30", "20:00"],
                "active_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            }
        })
        
        # Travel Blogger Template
        self.templates["travel_blogger"] = CelebrityTemplate("Travel Blogger", {
            "celebrity": {
                "name": "Sofia Wanderlust",
                "age": 29,
                "gender": "female",
                "occupation": "travel photographer and lifestyle blogger",
                "personality_traits": ["adventurous", "curious", "cultured", "free-spirited", "inspiring"],
                "interests": ["photography", "cultural exploration", "sustainable travel", "food", "adventure sports"],
                "style_preferences": ["bohemian", "elegant", "comfortable", "worldly"],
                "bio_description": "Travel photographer sharing authentic cultural experiences and sustainable travel inspiration from around the world"
            },
            "image_conditions": {
                "style": "travel",
                "lighting": "golden hour and natural light",
                "location": "exotic destinations and cultural landmarks",
                "outfit_style": "travel chic and cultural appropriate",
                "pose_description": "candid exploration and cultural immersion",
                "mood": "adventurous and wonder-filled",
                "background": "stunning landscapes and cultural sites",
                "quality": "high"
            },
            "posting_schedule": {
                "feed_posts_per_day": 1,
                "story_posts_per_day": 5,
                "posting_times": ["08:00", "14:00", "18:00", "21:00"],
                "active_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            }
        })
        
        # Fashion Influencer Template
        self.templates["fashion_influencer"] = CelebrityTemplate("Fashion Influencer", {
            "celebrity": {
                "name": "Milan Styles",
                "age": 24,
                "gender": "male",
                "occupation": "fashion model and style consultant",
                "personality_traits": ["stylish", "confident", "creative", "trendsetting", "artistic"],
                "interests": ["haute couture", "street fashion", "sustainable fashion", "art", "design"],
                "style_preferences": ["avant-garde", "minimalist", "luxury", "trendy"],
                "bio_description": "Fashion model and style consultant showcasing the intersection of high fashion and sustainable style"
            },
            "image_conditions": {
                "style": "fashion",
                "lighting": "professional studio lighting",
                "location": "fashion studios and urban settings",
                "outfit_style": "haute couture and designer fashion",
                "pose_description": "editorial and runway inspired poses",
                "mood": "sophisticated and artistic",
                "background": "minimalist and fashion-forward",
                "quality": "high"
            },
            "posting_schedule": {
                "feed_posts_per_day": 2,
                "story_posts_per_day": 3,
                "posting_times": ["10:00", "15:00", "19:30"],
                "active_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            }
        })
        
        # Food Blogger Template
        self.templates["food_blogger"] = CelebrityTemplate("Food Blogger", {
            "celebrity": {
                "name": "Chef Isabella",
                "age": 31,
                "gender": "female",
                "occupation": "chef and culinary content creator",
                "personality_traits": ["passionate", "creative", "warm", "knowledgeable", "approachable"],
                "interests": ["cooking", "sustainable ingredients", "food photography", "cultural cuisine", "nutrition"],
                "style_preferences": ["casual chic", "apron fashion", "colorful", "approachable"],
                "bio_description": "Chef and culinary creator sharing delicious recipes, cooking techniques, and the joy of food culture"
            },
            "image_conditions": {
                "style": "lifestyle",
                "lighting": "warm kitchen lighting",
                "location": "modern kitchen and dining spaces",
                "outfit_style": "chef attire and casual cooking wear",
                "pose_description": "cooking and food preparation poses",
                "mood": "warm and inviting",
                "background": "beautiful kitchen and food styling",
                "quality": "high"
            },
            "posting_schedule": {
                "feed_posts_per_day": 2,
                "story_posts_per_day": 4,
                "posting_times": ["08:00", "12:30", "17:00", "20:30"],
                "active_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            }
        })
        
        # Tech Entrepreneur Template
        self.templates["tech_entrepreneur"] = CelebrityTemplate("Tech Entrepreneur", {
            "celebrity": {
                "name": "Jordan Innovation",
                "age": 28,
                "gender": "non-binary",
                "occupation": "tech entrepreneur and startup founder",
                "personality_traits": ["innovative", "driven", "visionary", "analytical", "inspiring"],
                "interests": ["artificial intelligence", "startup culture", "innovation", "productivity", "future tech"],
                "style_preferences": ["minimalist", "modern", "professional", "tech-forward"],
                "bio_description": "Tech entrepreneur building the future through innovative AI solutions and sharing insights on startup culture"
            },
            "image_conditions": {
                "style": "lifestyle",
                "lighting": "modern office lighting",
                "location": "tech offices and innovation spaces",
                "outfit_style": "modern business casual",
                "pose_description": "confident leadership and innovation focused",
                "mood": "focused and visionary",
                "background": "sleek tech environments",
                "quality": "high"
            },
            "posting_schedule": {
                "feed_posts_per_day": 1,
                "story_posts_per_day": 3,
                "posting_times": ["09:00", "13:00", "18:00"],
                "active_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            }
        })
    
    def list_templates(self) -> List[str]:
        """List all available celebrity templates"""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[Dict]:
        """Get information about a specific template"""
        if template_name in self.templates:
            return self.templates[template_name].config
        return None
    
    def create_celebrity(self, template_name: str) -> Optional[AIInstagramCelebrity]:
        """Create celebrity from template"""
        if template_name in self.templates:
            return self.templates[template_name].create_celebrity()
        return None
    
    def create_custom_celebrity(self, name: str, **kwargs) -> AIInstagramCelebrity:
        """Create custom celebrity with optional parameters"""
        celebrity = AIInstagramCelebrity()
        
        # Update name at minimum
        celebrity.celebrity.name = name
        
        # Apply any custom parameters
        for key, value in kwargs.items():
            if hasattr(celebrity.celebrity, key):
                setattr(celebrity.celebrity, key, value)
        
        return celebrity
    
    def save_celebrity_config(self, celebrity: AIInstagramCelebrity, filename: str):
        """Save celebrity configuration to JSON file"""
        config = {
            "celebrity": asdict(celebrity.celebrity),
            "image_conditions": asdict(celebrity.default_image_conditions),
            "posting_schedule": asdict(celebrity.posting_schedule)
        }
        
        # Convert enums to strings
        if hasattr(config["image_conditions"]["style"], "value"):
            config["image_conditions"]["style"] = config["image_conditions"]["style"].value
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_celebrity_from_config(self, config_file: str) -> Optional[AIInstagramCelebrity]:
        """Load celebrity from existing config file"""
        if not os.path.exists(config_file):
            return None
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        template = CelebrityTemplate("Custom", config)
        return template.create_celebrity()

# Quick access functions for minimal friction
def create_fitness_influencer() -> AIInstagramCelebrity:
    """One-line fitness influencer creation"""
    factory = CelebrityFactory()
    return factory.create_celebrity("fitness_influencer")

def create_travel_blogger() -> AIInstagramCelebrity:
    """One-line travel blogger creation"""
    factory = CelebrityFactory()
    return factory.create_celebrity("travel_blogger")

def create_fashion_influencer() -> AIInstagramCelebrity:
    """One-line fashion influencer creation"""
    factory = CelebrityFactory()
    return factory.create_celebrity("fashion_influencer")

def create_food_blogger() -> AIInstagramCelebrity:
    """One-line food blogger creation"""
    factory = CelebrityFactory()
    return factory.create_celebrity("food_blogger")

def create_tech_entrepreneur() -> AIInstagramCelebrity:
    """One-line tech entrepreneur creation"""
    factory = CelebrityFactory()
    return factory.create_celebrity("tech_entrepreneur")

# Example usage
if __name__ == "__main__":
    factory = CelebrityFactory()
    
    print("Available celebrity templates:")
    for template in factory.list_templates():
        print(f"  - {template}")
    
    # Create a fitness influencer with one line
    fitness_celebrity = create_fitness_influencer()
    print(f"\nCreated: {fitness_celebrity.celebrity.name}")
    print(f"Occupation: {fitness_celebrity.celebrity.occupation}")
    
    # Save configuration for later use
    factory.save_celebrity_config(fitness_celebrity, "my_fitness_celebrity.json")
    print("Configuration saved to my_fitness_celebrity.json")