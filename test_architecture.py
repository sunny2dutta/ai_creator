#!/usr/bin/env python3
"""
Test script for the new profile-based architecture
"""

import json
import random
from pathlib import Path

class SimpleProfileTest:
    """Simple test of the profile architecture without external dependencies"""

    def __init__(self):
        self.data_dir = Path("data/prompts")
        self.profiles_file = Path("data/profiles.json")

    def load_profiles(self):
        """Load available profiles"""
        if not self.profiles_file.exists():
            return []

        with open(self.profiles_file) as f:
            data = json.load(f)
            return list(data["profiles"].keys())

    def load_profile_data(self, profile_name):
        """Load profile-specific data"""
        profile_file = self.data_dir / f"{profile_name}.json"

        if not profile_file.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_file}")

        with open(profile_file) as f:
            return json.load(f)

    def test_profile(self, profile_name, category=None):
        """Test profile functionality"""
        print(f"🧪 Testing profile: {profile_name}")

        # Load profile data
        profile_data = self.load_profile_data(profile_name)

        # Show profile info
        print(f"   Topic: {profile_data['topic']}")
        print(f"   Categories: {list(profile_data['categories'].keys())}")

        # Select category
        categories = list(profile_data['categories'].keys())
        if category and category in categories:
            selected_category = category
        else:
            selected_category = random.choice(categories)

        print(f"   Selected category: {selected_category}")

        # Get starter prompts
        category_data = profile_data['categories'][selected_category]
        starter_prompts = category_data['starter_prompts']
        selected_prompt = random.choice(starter_prompts)

        print(f"   Selected starter prompt: {selected_prompt}")

        # Get clothing hints
        clothing_hints = category_data.get('clothing_hints', [])
        if clothing_hints:
            print(f"   Clothing hints: {', '.join(clothing_hints[:2])}...")

        # Get base images
        base_images = profile_data.get('base_images', [])
        print(f"   Base images: {len(base_images)} available")

        # Get style preferences
        style_prefs = profile_data.get('style_preferences', {})
        if style_prefs:
            print(f"   Camera: {style_prefs.get('camera', 'Default')}")
            print(f"   Quality: {style_prefs.get('quality', 'Default')}")

        return {
            'profile': profile_name,
            'category': selected_category,
            'prompt': selected_prompt,
            'clothing_hints': clothing_hints,
            'base_images': base_images,
            'style_preferences': style_prefs
        }

def main():
    """Main test function"""
    print("🚀 Testing Profile-Based Architecture")
    print("=" * 50)

    tester = SimpleProfileTest()

    # List available profiles
    profiles = tester.load_profiles()
    print(f"Available profiles: {profiles}")
    print()

    # Test each profile
    for profile in profiles:
        try:
            result = tester.test_profile(profile)
            print("   ✅ Profile test passed!")
        except Exception as e:
            print(f"   ❌ Profile test failed: {e}")
        print()

    # Test specific category
    if profiles:
        print("🎯 Testing specific category selection:")
        result = tester.test_profile(profiles[0], "Festival cultural woman")
        print("   ✅ Category-specific test passed!")
        print()

    print("🎉 Architecture test completed!")

    # Show what would be passed to image generation
    if profiles:
        print("\n📝 Sample data for image generation:")
        result = tester.test_profile(profiles[0])
        print(f"   Profile: {result['profile']}")
        print(f"   Base Images: {len(result['base_images'])} URLs")
        print(f"   Prompt: {result['prompt'][:80]}...")
        if result['clothing_hints']:
            print(f"   Clothing: {result['clothing_hints'][0]}")

if __name__ == "__main__":
    main()