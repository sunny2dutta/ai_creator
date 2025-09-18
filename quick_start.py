#!/usr/bin/env python3
"""
Quick Start - Instant AI Celebrity Deployment
The fastest way to get an AI celebrity running
"""

import os
import sys
import subprocess
from typing import Dict

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    issues = []
    
    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI DALL-E",
        "STABILITY_API_KEY": "Stability AI", 
        "REPLICATE_API_TOKEN": "Replicate"
    }
    
    has_image_api = False
    for key, service in api_keys.items():
        if os.getenv(key):
            print(f"  ✅ {service} API key found")
            has_image_api = True
        else:
            print(f"  ⚪ {service} API key not found")
    
    if not has_image_api:
        issues.append("No image generation API key found. Set one of: OPENAI_API_KEY, STABILITY_API_KEY, or REPLICATE_API_TOKEN")
    
    # Check Instagram API (optional for testing)
    instagram_keys = ["INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ID", "FACEBOOK_PAGE_ID"]
    has_instagram = all(os.getenv(key) for key in instagram_keys)
    
    if has_instagram:
        print("  ✅ Instagram API credentials found")
    else:
        print("  ⚠️  Instagram API credentials not found (you can still test image generation)")
    
    return issues

def quick_deploy(celebrity_type: str = "fitness") -> bool:
    """Deploy celebrity with zero configuration"""
    
    print(f"🚀 Quick deploying {celebrity_type} celebrity...")
    
    try:
        # Run the launcher in test mode first
        result = subprocess.run([
            sys.executable, "launch_celebrity.py", 
            celebrity_type, "--test", "--save"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Test deployment successful!")
            
            # Ask if user wants to run full deployment
            response = input("🔄 Run full automated posting system? (y/n, default: n): ").strip().lower()
            if response == "y":
                # Run without test mode
                subprocess.run([sys.executable, "launch_celebrity.py", celebrity_type])
            return True
        else:
            print(f"❌ Test deployment failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return False

def interactive_quick_start():
    """Interactive quick start with user choice"""
    
    print("🎭 AI Celebrity Quick Start")
    print("=" * 30)
    
    # Check dependencies
    issues = check_dependencies()
    if issues:
        print("\n❌ Setup issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues and try again.")
        return
    
    print("\n✅ Dependencies check passed!")
    
    # Show celebrity options
    celebrities = {
        "1": ("fitness", "🏋️ Fitness Influencer - Alex Chen"),
        "2": ("travel", "✈️ Travel Blogger - Sofia Wanderlust"), 
        "3": ("fashion", "👗 Fashion Influencer - Milan Styles"),
        "4": ("food", "🍳 Food Blogger - Chef Isabella"),
        "5": ("tech", "💻 Tech Entrepreneur - Jordan Innovation")
    }
    
    print("\nAvailable celebrity templates:")
    print("-" * 35)
    for key, (_, description) in celebrities.items():
        print(f"  {key}. {description}")
    
    # Get user choice
    choice = input("\nSelect celebrity (1-5, default: 1): ").strip()
    if choice not in celebrities:
        choice = "1"
    
    celebrity_type, description = celebrities[choice]
    print(f"\n🎯 Selected: {description}")
    
    # Deploy
    success = quick_deploy(celebrity_type)
    
    if success:
        print("\n🎉 Celebrity deployed successfully!")
        print("💡 Pro tips:")
        print("   - Check the saved configuration file for customization")
        print("   - Use 'python launch_celebrity.py --list' to see all options")
        print("   - Use 'python launch_celebrity.py --custom' for custom celebrities")
    else:
        print("\n❌ Deployment failed. Check your API credentials and try again.")

def show_examples():
    """Show usage examples"""
    print("🚀 Quick Start Examples")
    print("=" * 25)
    print()
    print("1. Instant fitness influencer:")
    print("   python quick_start.py fitness")
    print()
    print("2. Interactive selection:")
    print("   python quick_start.py")
    print()
    print("3. Direct celebrity launch:")
    print("   python launch_celebrity.py travel --test")
    print()
    print("4. Custom celebrity creation:")
    print("   python launch_celebrity.py --custom")
    print()
    print("5. List all available templates:")
    print("   python launch_celebrity.py --list")
    print()

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            show_examples()
        elif sys.argv[1] == "--examples":
            show_examples()
        else:
            # Direct celebrity type
            celebrity_type = sys.argv[1]
            quick_deploy(celebrity_type)
    else:
        interactive_quick_start()

if __name__ == "__main__":
    main()