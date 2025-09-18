#!/usr/bin/env python3
"""
Test CLI functionality without external dependencies
"""

import argparse
import sys
import json
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate and post social media content')

    parser.add_argument(
        '--profile',
        type=str,
        default='rupashi',
        help='Profile name to use for posting (default: rupashi)'
    )

    parser.add_argument(
        '--category',
        type=str,
        help='Specific category for content generation'
    )

    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List all available profiles'
    )

    return parser.parse_args()

def load_profiles():
    """Load available profiles"""
    profiles_file = Path("data/profiles.json")
    if not profiles_file.exists():
        return []

    with open(profiles_file) as f:
        data = json.load(f)
        return list(data["profiles"].keys())

def main():
    """Test main function"""
    args = parse_arguments()

    if args.list_profiles:
        profiles = load_profiles()
        print("Available profiles:")
        for profile in profiles:
            print(f"  - {profile}")
        return

    print(f"Selected profile: {args.profile}")
    if args.category:
        print(f"Selected category: {args.category}")
    else:
        print("Category: Will be randomly selected")

    # Load profile data
    profile_file = Path(f"data/prompts/{args.profile}.json")
    if profile_file.exists():
        with open(profile_file) as f:
            profile_data = json.load(f)
            categories = list(profile_data['categories'].keys())
            print(f"Available categories: {categories}")

            if args.category and args.category not in categories:
                print(f"⚠️  Warning: Category '{args.category}' not found for profile '{args.profile}'")
                print(f"Available categories: {categories}")
    else:
        print(f"❌ Profile file not found: {profile_file}")

if __name__ == "__main__":
    main()