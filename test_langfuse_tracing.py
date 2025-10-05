#!/usr/bin/env python3
"""
Test script to verify Langfuse tracing is working across the AI pipeline
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))

from arc_prompt_creator import ArcPromptCreator
from prompt_generator import PromptGenerator

load_dotenv()

def test_langfuse_setup():
    """Test if Langfuse credentials are properly configured"""
    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    secret_key = os.getenv('LANGFUSE_SECRET_KEY')
    
    print("🔍 Checking Langfuse Configuration...")
    print(f"LANGFUSE_PUBLIC_KEY: {'✅ Set' if public_key and public_key != 'your_langfuse_public_key_here' else '❌ Not set or default'}")
    print(f"LANGFUSE_SECRET_KEY: {'✅ Set' if secret_key and secret_key != 'your_langfuse_secret_key_here' else '❌ Not set or default'}")
    print(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
    
    if not public_key or public_key == 'your_langfuse_public_key_here':
        print("\n⚠️  Please update your .env file with real Langfuse credentials before running the full pipeline")
        return False
    
    return True

def test_arc_prompt_creator():
    """Test the ArcPromptCreator with tracing"""
    print("\n🧪 Testing ArcPromptCreator with Langfuse tracing...")
    
    try:
        creator = ArcPromptCreator()
        print(f"Langfuse client initialized: {'✅' if creator.langfuse_client else '❌'}")
        
        # Test creating a simple 7-day arc
        story_data = creator.create_7_day_arc("Beach vacation", "test_character")
        
        print(f"Arc creation successful: {'✅' if story_data else '❌'}")
        print(f"Data source: {story_data.get('data_source', 'unknown')}")
        print(f"Days created: {len(story_data.get('days', {}))}")
        
        return True
    except Exception as e:
        print(f"❌ Error in ArcPromptCreator: {e}")
        return False

def test_prompt_generator():
    """Test the PromptGenerator with tracing"""
    print("\n🧪 Testing PromptGenerator with Langfuse tracing...")
    
    try:
        # Use an existing profile arc file
        generator = PromptGenerator("rupashi")
        print(f"Langfuse client initialized: {'✅' if generator.langfuse_client else '❌'}")
        
        # Test generating a detailed prompt
        detailed_prompt = generator.generate_detailed_prompt(1)
        
        print(f"Prompt generation successful: {'✅' if detailed_prompt else '❌'}")
        print(f"Enhancement method: {detailed_prompt.get('metadata', {}).get('enhancement_method', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Error in PromptGenerator: {e}")
        return False

def main():
    print("🚀 Langfuse AI Tracing Setup Verification\n")
    
    # Test configuration
    config_ok = test_langfuse_setup()
    
    # Test components
    arc_ok = test_arc_prompt_creator()
    prompt_ok = test_prompt_generator()
    
    print("\n📊 Test Results Summary:")
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Arc Creator: {'✅ PASS' if arc_ok else '❌ FAIL'}")
    print(f"Prompt Generator: {'✅ PASS' if prompt_ok else '❌ FAIL'}")
    
    if config_ok and arc_ok and prompt_ok:
        print("\n🎉 All tests passed! Langfuse tracing is ready.")
        print("\nNext steps:")
        print("1. Update .env with your actual Langfuse credentials")
        print("2. Run your normal pipeline to see traces in Langfuse dashboard")
        print("3. Check trace data at: https://cloud.langfuse.com")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()