# AI Celebrity Quick Start Guide 🎭

Launch AI celebrities with **zero friction** - from idea to Instagram in under 60 seconds!

## 🚀 Ultra Quick Start (30 seconds)

```bash
# Launch any celebrity instantly
python quick_start.py fitness      # Fitness influencer  
python quick_start.py travel       # Travel blogger
python quick_start.py fashion      # Fashion influencer
python quick_start.py food         # Food blogger
python quick_start.py tech         # Tech entrepreneur

# Or interactive selection
python quick_start.py
```

## 💫 One-Command Celebrity Launch

```bash
# Launch with one command
python launch_celebrity.py fitness
python launch_celebrity.py travel --test
python launch_celebrity.py fashion --save
```

## 📋 Available Celebrity Templates

| Template | Description | Personality | Content Focus |
|----------|-------------|-------------|---------------|
| `fitness` | Alex Chen - Fitness Coach | Energetic, Motivational | Workouts, Nutrition, Wellness |
| `travel` | Sofia Wanderlust - Travel Blogger | Adventurous, Cultural | Destinations, Culture, Photography |
| `fashion` | Milan Styles - Fashion Model | Stylish, Artistic | Fashion, Style, Design |
| `food` | Chef Isabella - Culinary Creator | Passionate, Creative | Recipes, Cooking, Food Culture |
| `tech` | Jordan Innovation - Tech Entrepreneur | Innovative, Visionary | AI, Startups, Innovation |

## 🎯 Usage Examples

### 1. Instant Launch (Zero Configuration)
```bash
python quick_start.py
# Select from menu, test automatically, deploy immediately
```

### 2. Test Before Launch
```bash
python launch_celebrity.py fitness --test
# Posts one image to verify everything works
```

### 3. Save Configuration
```bash
python launch_celebrity.py travel --save
# Creates reusable config file: sofia_wanderlust_config.json
```

### 4. Custom Celebrity
```bash
python launch_celebrity.py --custom
# Interactive wizard to create personalized celebrity
```

### 5. Use Existing Config
```bash
python launch_celebrity.py --config my_celebrity.json
# Launch from previously saved configuration
```

## ⚡ Minimal Friction Features

### 🏭 Celebrity Factory
Pre-built templates with optimized:
- Personality traits
- Visual styles  
- Posting schedules
- Content themes

### 🎨 One-Line Creation
```python
from celebrity_factory import create_fitness_influencer, create_travel_blogger

# Create any celebrity with one function call
celebrity = create_fitness_influencer()
celebrity = create_travel_blogger()
```

### 🔧 Smart Defaults
- Instagram-optimized image ratios
- Platform-appropriate posting times
- Diverse content generation
- Automatic API service detection

## 📊 Quick Commands Reference

```bash
# List all templates
python launch_celebrity.py --list

# Get template info
python launch_celebrity.py --info fitness

# Create custom celebrity
python launch_celebrity.py --custom

# Test deployment
python launch_celebrity.py fitness --test

# Save configuration  
python launch_celebrity.py travel --save

# Launch from config
python launch_celebrity.py --config my_config.json

# Interactive quick start
python quick_start.py
```

## 🔑 Setup (5 minutes)

### 1. API Keys (Choose One)
```bash
# OpenAI DALL-E
export OPENAI_API_KEY="your-key-here"

# Stability AI  
export STABILITY_API_KEY="your-key-here"

# Replicate
export REPLICATE_API_TOKEN="your-key-here"
```

### 2. Instagram API (Optional for Testing)
```bash
export INSTAGRAM_ACCESS_TOKEN="your-token"
export INSTAGRAM_BUSINESS_ID="your-id"  
export FACEBOOK_PAGE_ID="your-page-id"
```

### 3. Verify Setup
```bash
python quick_start.py
# Automatically checks all dependencies
```

## 🎭 Template Customization

### Quick Personality Change
```python
from celebrity_factory import CelebrityFactory

factory = CelebrityFactory()
celebrity = factory.create_celebrity("fitness")

# Modify personality
celebrity.celebrity.name = "Your Custom Name"
celebrity.celebrity.personality_traits = ["energetic", "inspiring", "fun"]
```

### Save Custom Template
```python
factory.save_celebrity_config(celebrity, "my_custom.json")
```

### Visual Style Adjustment
```python
celebrity.default_image_conditions.style = ImageStyle.FASHION
celebrity.default_image_conditions.location = "modern studio"
celebrity.default_image_conditions.lighting = "professional lighting"
```

## 🚨 Troubleshooting

### ❌ "No image generation API key found"
- Set one of: `OPENAI_API_KEY`, `STABILITY_API_KEY`, or `REPLICATE_API_TOKEN`

### ❌ "Instagram API connection failed"
- For testing: Use `--test` flag to test image generation only
- For full deployment: Set Instagram API credentials

### ❌ "Template not found"
- Use `python launch_celebrity.py --list` to see available templates
- Check spelling: `fitness`, `travel`, `fashion`, `food`, `tech`

### ❌ "Import error"
- Ensure all files are in the same directory
- Install required dependencies: `pip install schedule pillow requests`

## 💡 Pro Tips

1. **Start with Test Mode**: Always use `--test` flag first to verify setup
2. **Save Configurations**: Use `--save` to create reusable celebrity configs  
3. **Quick Templates**: Use short names: `fitness` instead of `fitness_influencer`
4. **Custom Creation**: Use `--custom` for unique celebrities matching your brand
5. **Batch Deployment**: Save multiple configs and switch between them easily

## 🎯 Common Workflows

### For Content Testing
```bash
python launch_celebrity.py fitness --test --save
# Test image generation + save config for later
```

### For Production Launch
```bash
python quick_start.py
# Interactive setup + dependency check + full deployment
```

### For Custom Brands
```bash
python launch_celebrity.py --custom
# Create brand-specific celebrity + save config + launch
```

## 🌟 What Makes This "Minimal Friction"?

- ✅ **Zero Config Required**: Pre-built templates work out of the box
- ✅ **One Command Launch**: `python quick_start.py fitness`
- ✅ **Smart Defaults**: Optimized for Instagram success
- ✅ **Automatic API Detection**: Uses whatever image API you have
- ✅ **Test Mode**: Verify before full deployment
- ✅ **Interactive Wizards**: Guide you through setup
- ✅ **Template Library**: 5 proven celebrity archetypes
- ✅ **Save & Reuse**: Create once, launch anytime

**From zero to AI celebrity in under 60 seconds! 🚀**