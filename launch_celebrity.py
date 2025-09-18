#!/usr/bin/env python3
"""
One-Command Celebrity Launcher
Launch any AI celebrity with minimal friction
Usage: python launch_celebrity.py [template_name] [options]
"""

import argparse
import sys
import os
from typing import Optional
from celebrity_factory import CelebrityFactory, create_fitness_influencer, create_travel_blogger, create_fashion_influencer, create_food_blogger, create_tech_entrepreneur
from ai_celebrity_app import AICelebrityApp

def launch_celebrity_from_template(template_name: str, save_config: bool = False, test_mode: bool = False) -> bool:
    """Launch celebrity from template with minimal setup"""
    
    factory = CelebrityFactory()
    
    # Quick access functions for common templates
    quick_creators = {
        "fitness": create_fitness_influencer,
        "travel": create_travel_blogger,
        "fashion": create_fashion_influencer,
        "food": create_food_blogger,
        "tech": create_tech_entrepreneur
    }
    
    # Try quick creators first
    if template_name in quick_creators:
        celebrity = quick_creators[template_name]()
    else:
        # Try full template name
        celebrity = factory.create_celebrity(template_name)
        if not celebrity:
            print(f"❌ Template '{template_name}' not found.")
            print(f"Available templates: {', '.join(factory.list_templates())}")
            return False
    
    # Save configuration if requested
    if save_config:
        config_filename = f"{celebrity.celebrity.name.lower().replace(' ', '_')}_config.json"
        factory.save_celebrity_config(celebrity, config_filename)
        print(f"💾 Configuration saved to {config_filename}")
    
    # Create temporary config file for the app
    temp_config_file = "temp_celebrity_config.json"
    factory.save_celebrity_config(celebrity, temp_config_file)
    
    try:
        # Initialize and run the celebrity app
        app = AICelebrityApp(temp_config_file)
        
        print(f"🎭 Launching {celebrity.celebrity.name} - {celebrity.celebrity.occupation}")
        print(f"📊 Personality: {', '.join(celebrity.celebrity.personality_traits)}")
        print(f"💫 Style: {celebrity.default_image_conditions.style.value}")
        
        if test_mode:
            print("🧪 Running in test mode - posting one feed post...")
            success = app.run_once("feed")
            if success:
                print("✅ Test post successful!")
            else:
                print("❌ Test post failed. Check your API configurations.")
        else:
            print("🚀 Starting automated posting system...")
            app.run()
    
    except KeyboardInterrupt:
        print("\n🛑 Celebrity launcher stopped by user")
    except Exception as e:
        print(f"❌ Error launching celebrity: {e}")
        return False
    finally:
        # Clean up temp config file
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)
    
    return True

def interactive_celebrity_creation():
    """Interactive wizard for creating custom celebrities"""
    print("\n🎨 Celebrity Creation Wizard")
    print("=" * 40)
    
    # Basic info
    name = input("Celebrity name: ").strip()
    if not name:
        print("❌ Name is required")
        return None
    
    age = input("Age (default: 25): ").strip()
    age = int(age) if age.isdigit() else 25
    
    gender = input("Gender (male/female/non-binary, default: female): ").strip().lower()
    if gender not in ["male", "female", "non-binary"]:
        gender = "female"
    
    occupation = input("Occupation (default: lifestyle influencer): ").strip()
    if not occupation:
        occupation = "lifestyle influencer"
    
    print("\nSelect personality traits (comma-separated):")
    print("Examples: confident, creative, inspiring, energetic, authentic, positive")
    traits_input = input("Traits: ").strip()
    traits = [t.strip() for t in traits_input.split(",") if t.strip()] if traits_input else ["confident", "creative"]
    
    print("\nSelect interests (comma-separated):")
    print("Examples: fashion, travel, fitness, food, photography, art")
    interests_input = input("Interests: ").strip()
    interests = [i.strip() for i in interests_input.split(",") if i.strip()] if interests_input else ["lifestyle", "fashion"]
    
    # Create custom celebrity
    factory = CelebrityFactory()
    celebrity = factory.create_custom_celebrity(
        name=name,
        age=age,
        gender=gender,
        occupation=occupation,
        personality_traits=traits,
        interests=interests,
        bio_description=f"{age}-year-old {occupation} sharing {', '.join(interests)} content"
    )
    
    print(f"\n✅ Created custom celebrity: {name}")
    
    # Save option
    save = input("Save configuration? (y/n, default: y): ").strip().lower()
    if save != "n":
        config_filename = f"{name.lower().replace(' ', '_')}_config.json"
        factory.save_celebrity_config(celebrity, config_filename)
        print(f"💾 Configuration saved to {config_filename}")
        return config_filename
    
    return celebrity

def main():
    parser = argparse.ArgumentParser(description="Launch AI Celebrity with minimal friction")
    parser.add_argument("template", nargs="?", help="Celebrity template name (fitness, travel, fashion, food, tech)")
    parser.add_argument("--list", "-l", action="store_true", help="List available templates")
    parser.add_argument("--info", "-i", metavar="TEMPLATE", help="Show template information")
    parser.add_argument("--save", "-s", action="store_true", help="Save configuration file")
    parser.add_argument("--test", "-t", action="store_true", help="Test mode - post once and exit")
    parser.add_argument("--custom", "-c", action="store_true", help="Create custom celebrity interactively")
    parser.add_argument("--config", metavar="FILE", help="Use existing config file")
    
    args = parser.parse_args()
    
    factory = CelebrityFactory()
    
    # List templates
    if args.list:
        print("Available celebrity templates:")
        print("=" * 30)
        for template in factory.list_templates():
            print(f"  {template}")
        print("\nQuick access aliases:")
        print("  fitness → fitness_influencer")
        print("  travel → travel_blogger") 
        print("  fashion → fashion_influencer")
        print("  food → food_blogger")
        print("  tech → tech_entrepreneur")
        return
    
    # Show template info
    if args.info:
        info = factory.get_template_info(args.info)
        if info:
            celebrity_info = info.get("celebrity", {})
            print(f"Template: {args.info}")
            print("=" * 30)
            print(f"Name: {celebrity_info.get('name', 'N/A')}")
            print(f"Age: {celebrity_info.get('age', 'N/A')}")
            print(f"Occupation: {celebrity_info.get('occupation', 'N/A')}")
            print(f"Personality: {', '.join(celebrity_info.get('personality_traits', []))}")
            print(f"Interests: {', '.join(celebrity_info.get('interests', []))}")
        else:
            print(f"❌ Template '{args.info}' not found")
        return
    
    # Custom celebrity creation
    if args.custom:
        result = interactive_celebrity_creation()
        if result:
            launch = input("\nLaunch this celebrity now? (y/n, default: y): ").strip().lower()
            if launch != "n":
                if isinstance(result, str):  # It's a config file
                    app = AICelebrityApp(result)
                    app.run()
                # If it's a celebrity object, we'd need to save it first
        return
    
    # Use existing config file
    if args.config:
        if not os.path.exists(args.config):
            print(f"❌ Config file '{args.config}' not found")
            return
        
        app = AICelebrityApp(args.config)
        print(f"🚀 Launching celebrity from {args.config}")
        
        if args.test:
            print("🧪 Running test post...")
            app.run_once("feed")
        else:
            app.run()
        return
    
    # Launch from template
    if args.template:
        success = launch_celebrity_from_template(
            args.template, 
            save_config=args.save, 
            test_mode=args.test
        )
        if not success:
            sys.exit(1)
    else:
        # Show help if no arguments
        print("🎭 AI Celebrity Launcher")
        print("=" * 25)
        print("Quick start examples:")
        print("  python launch_celebrity.py fitness")
        print("  python launch_celebrity.py travel --test")
        print("  python launch_celebrity.py fashion --save")
        print("  python launch_celebrity.py --custom")
        print("  python launch_celebrity.py --list")
        print("\nFor full help: python launch_celebrity.py --help")

if __name__ == "__main__":
    main()