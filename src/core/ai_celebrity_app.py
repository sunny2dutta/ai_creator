"""
Main AI Instagram Celebrity Application
Automated posting system with scheduling and content generation
"""

import schedule
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random
import logging
from pathlib import Path

from ai_celebrity_config import AIInstagramCelebrity, ImageConditions, ImageStyle, PostType
from image_generator import AIImageGenerator
from instagram_poster import InstagramPoster

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_celebrity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContentScheduler:
    """Content scheduling and posting automation"""
    
    def __init__(self, celebrity_config: AIInstagramCelebrity, image_generator: AIImageGenerator):
        self.celebrity_config = celebrity_config
        self.image_generator = image_generator
        self.instagram_poster = InstagramPoster(celebrity_config)
        self.post_history_file = "post_history.json"
        self.load_post_history()
        
    def load_post_history(self):
        """Load posting history from file"""
        if os.path.exists(self.post_history_file):
            with open(self.post_history_file, 'r') as f:
                self.post_history = json.load(f)
        else:
            self.post_history = {
                "daily_posts": {},
                "total_posts": 0,
                "last_reset": str(datetime.now().date())
            }
    
    def save_post_history(self):
        """Save posting history to file"""
        with open(self.post_history_file, 'w') as f:
            json.dump(self.post_history, f, indent=2)
    
    def reset_daily_count_if_needed(self):
        """Reset daily post count if it's a new day"""
        today = str(datetime.now().date())
        if self.post_history["last_reset"] != today:
            self.post_history["daily_posts"] = {}
            self.post_history["last_reset"] = today
            self.save_post_history()
    
    def can_post_today(self) -> bool:
        """Check if we can still post today (25 post limit)"""
        self.reset_daily_count_if_needed()
        today = str(datetime.now().date())
        posts_today = len(self.post_history["daily_posts"].get(today, []))
        return posts_today < 25
    
    def record_post(self, post_type: str, media_id: str, conditions: Dict):
        """Record a successful post"""
        today = str(datetime.now().date())
        
        if today not in self.post_history["daily_posts"]:
            self.post_history["daily_posts"][today] = []
        
        post_record = {
            "timestamp": str(datetime.now()),
            "type": post_type,
            "media_id": media_id,
            "conditions": conditions
        }
        
        self.post_history["daily_posts"][today].append(post_record)
        self.post_history["total_posts"] += 1
        self.save_post_history()
        
        logger.info(f"✅ Posted {post_type} successfully. Media ID: {media_id}")
    
    def generate_varied_conditions(self, post_type: PostType) -> ImageConditions:
        """Generate varied image conditions to keep content diverse"""
        
        # Base conditions from config
        base = self.celebrity_config.default_image_conditions
        
        # Randomize some aspects for variety
        styles = list(ImageStyle)
        locations = [
            "modern apartment", "coffee shop", "city street", "park", "beach",
            "rooftop terrace", "cozy bedroom", "stylish restaurant", "art gallery",
            "fitness studio", "library", "boutique store"
        ]
        
        lighting_options = [
            "natural daylight", "golden hour", "soft studio lighting", 
            "warm indoor lighting", "bright morning light", "sunset glow"
        ]
        
        outfits = [
            "casual chic", "business casual", "athleisure", "elegant dress",
            "street style", "cozy sweater", "summer dress", "designer outfit"
        ]
        
        moods = [
            "happy and confident", "serene and peaceful", "energetic and vibrant",
            "thoughtful and inspiring", "playful and fun", "elegant and poised"
        ]
        
        # Create varied conditions
        conditions = ImageConditions(
            style=random.choice(styles),
            lighting=random.choice(lighting_options),
            location=random.choice(locations),
            outfit_style=random.choice(outfits),
            pose_description="natural and authentic pose",
            mood=random.choice(moods),
            background="aesthetically pleasing",
            quality="high",
            aspect_ratio="9:16" if post_type == PostType.STORY else "1:1"
        )
        
        return conditions
    
    def create_and_post_content(self, post_type: PostType) -> bool:
        """Generate and post content"""
        
        if not self.can_post_today():
            logger.warning("⚠️ Daily posting limit reached (25 posts)")
            return False
        
        try:
            # Generate varied conditions
            conditions = self.generate_varied_conditions(post_type)
            logger.info(f"🎨 Generating {post_type.value.lower()} image with conditions: {conditions.style.value}, {conditions.location}")
            
            # Generate image
            image_bytes = self.image_generator.generate_celebrity_image(
                self.celebrity_config, 
                conditions
            )
            
            # Resize for Instagram
            image_bytes = self.image_generator.resize_for_instagram(
                image_bytes, 
                "story" if post_type == PostType.STORY else "feed"
            )
            
            # Post to Instagram
            if post_type == PostType.FEED:
                media_id = self.instagram_poster.post_to_feed(image_bytes)
            else:
                media_id = self.instagram_poster.post_to_story(image_bytes)
            
            # Record the post
            self.record_post(
                post_type.value,
                media_id,
                {
                    "style": conditions.style.value,
                    "location": conditions.location,
                    "lighting": conditions.lighting,
                    "outfit": conditions.outfit_style,
                    "mood": conditions.mood
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating/posting content: {e}")
            return False
    
    def post_feed_content(self):
        """Scheduled function to post feed content"""
        logger.info("📱 Creating feed post...")
        self.create_and_post_content(PostType.FEED)
    
    def post_story_content(self):
        """Scheduled function to post story content"""
        logger.info("📖 Creating story post...")
        self.create_and_post_content(PostType.STORY)

class AICelebrityApp:
    """Main application class"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Load configuration
        self.celebrity_config = AIInstagramCelebrity()
        
        if config_file and os.path.exists(config_file):
            self.load_custom_config(config_file)
        
        # Initialize components
        self.setup_image_generator()
        self.scheduler = ContentScheduler(self.celebrity_config, self.image_generator)
        
        # Setup scheduled posting
        self.setup_schedule()
        
    def load_custom_config(self, config_file: str):
        """Load custom configuration from JSON file"""
        with open(config_file, 'r') as f:
            custom_config = json.load(f)
        
        # Update celebrity profile
        if "celebrity" in custom_config:
            for key, value in custom_config["celebrity"].items():
                if hasattr(self.celebrity_config.celebrity, key):
                    setattr(self.celebrity_config.celebrity, key, value)
        
        # Update image conditions
        if "image_conditions" in custom_config:
            for key, value in custom_config["image_conditions"].items():
                if hasattr(self.celebrity_config.default_image_conditions, key):
                    setattr(self.celebrity_config.default_image_conditions, key, value)
        
        # Update posting schedule
        if "posting_schedule" in custom_config:
            for key, value in custom_config["posting_schedule"].items():
                if hasattr(self.celebrity_config.posting_schedule, key):
                    setattr(self.celebrity_config.posting_schedule, key, value)
        
        logger.info(f"✅ Loaded custom configuration from {config_file}")
    
    def setup_image_generator(self):
        """Initialize image generator with preferred service"""
        # Try to initialize with available API keys
        try:
            if os.getenv("OPENAI_API_KEY"):
                self.image_generator = AIImageGenerator("openai")
                logger.info("🎨 Using OpenAI DALL-E for image generation")
            elif os.getenv("STABILITY_API_KEY"):
                self.image_generator = AIImageGenerator("stability")
                logger.info("🎨 Using Stability AI for image generation")
            elif os.getenv("REPLICATE_API_TOKEN"):
                self.image_generator = AIImageGenerator("replicate")
                logger.info("🎨 Using Replicate for image generation")
            else:
                logger.error("❌ No AI image generation service configured. Please set API keys.")
                self.image_generator = None
        except Exception as e:
            logger.error(f"❌ Error initializing image generator: {e}")
            self.image_generator = None
    
    def setup_schedule(self):
        """Setup posting schedule based on configuration"""
        schedule_config = self.celebrity_config.posting_schedule
        
        # Clear existing schedule
        schedule.clear()
        
        # Schedule feed posts
        for post_time in schedule_config.posting_times[:schedule_config.feed_posts_per_day]:
            for day in schedule_config.active_days:
                getattr(schedule.every(), day.lower()).at(post_time).do(self.scheduler.post_feed_content)
        
        # Schedule story posts (more frequent)
        story_times = schedule_config.posting_times[:schedule_config.story_posts_per_day]
        for i, post_time in enumerate(story_times):
            for day in schedule_config.active_days:
                # Offset story times slightly to avoid conflicts
                time_parts = post_time.split(":")
                hour = int(time_parts[0])
                minute = int(time_parts[1]) + (i * 10)  # 10 minute offset for each story
                if minute >= 60:
                    hour += minute // 60
                    minute = minute % 60
                
                adjusted_time = f"{hour:02d}:{minute:02d}"
                getattr(schedule.every(), day.lower()).at(adjusted_time).do(self.scheduler.post_story_content)
        
        logger.info(f"📅 Scheduled {len(schedule.jobs)} posting jobs")
    
    def run_once(self, post_type: str = "feed"):
        """Run a single post for testing"""
        if self.image_generator is None:
            logger.error("❌ Image generator not configured")
            return False
        
        post_type_enum = PostType.FEED if post_type.lower() == "feed" else PostType.STORY
        return self.scheduler.create_and_post_content(post_type_enum)
    
    def validate_setup(self) -> bool:
        """Validate that all components are properly configured"""
        issues = []
        
        # Check image generator
        if self.image_generator is None:
            issues.append("Image generator not configured - need API keys")
        
        # Check Instagram credentials
        try:
            if not self.scheduler.instagram_poster.validate_connection():
                issues.append("Instagram API connection failed")
        except Exception:
            issues.append("Instagram credentials not configured")
        
        if issues:
            logger.error("❌ Configuration issues found:")
            for issue in issues:
                logger.error(f"   - {issue}")
            return False
        
        logger.info("✅ All systems configured and ready")
        return True
    
    def run(self):
        """Start the automated posting system"""
        logger.info("🚀 Starting AI Instagram Celebrity App")
        
        if not self.validate_setup():
            logger.error("❌ Setup validation failed. Please fix configuration issues.")
            return
        
        logger.info(f"🎭 Celebrity: {self.celebrity_config.celebrity.name}")
        logger.info(f"📅 Schedule: {len(schedule.jobs)} jobs configured")
        logger.info("⏰ Waiting for scheduled posts... (Press Ctrl+C to stop)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping AI Celebrity App")

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Instagram Celebrity App")
    parser.add_argument("--config", help="Path to custom configuration file")
    parser.add_argument("--test-feed", action="store_true", help="Post a single feed post for testing")
    parser.add_argument("--test-story", action="store_true", help="Post a single story for testing") 
    parser.add_argument("--validate", action="store_true", help="Validate configuration and exit")
    
    args = parser.parse_args()
    
    # Initialize app
    app = AICelebrityApp(args.config)
    
    if args.validate:
        app.validate_setup()
    elif args.test_feed:
        logger.info("🧪 Testing feed post...")
        app.run_once("feed")
    elif args.test_story:
        logger.info("🧪 Testing story post...")
        app.run_once("story")
    else:
        app.run()