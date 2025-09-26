# AI Instagram Celebrity System

Automated AI-powered Instagram celebrity that generates and posts pictures to Instagram feed and stories using Instagram Graph API. It posts long 7 day arc based on personalities of the AI influencers. CUrrently seeded with two:

1. Rupashi: A beautiful Bengali woman who lives in Kolkata and dreams of being a model in Paris.
2. Mr. Bananas: A chimpanzee who is also an intern at a big company wants to be CEO one day.

System gets details of this personas from RAGs created and creates posts on Facebook (feed and stories) and Instagram.

Current flow:

1. Create a 7 day arc based on an initial prompt. Initial prompt is user input.
2. Based on each day. Create a more detailed prompt, extract traits for relation DB and characteristics from RAGs
3. Using the detailed prompt and past images create a new picture to post.

There is a post processing step:

  1. Color Grading - Adjusts skin tones, shadows, and highlights to match realistic lighting conditions
  2. Lightroom-Style Filters - Applies tone curves, split toning, vibrance boosts, and subtle vignette effects
  3. Edge Sharpening - Selectively sharpens edges to fix AI's fuzzy boundaries while maintaining natural appearance
  4. Realistic Noise - Adds film grain and sensor noise patterns to mimic camera artifacts
  5. Instagram-Style Filters - Applies "natural", "warm", or "moody" filter presets


## Features

- **Configurable Celebrity Persona**: Easily customize celebrity profile, personality, interests, and style
- **AI Image Generation**: Supports multiple AI services (OpenAI DALL-E, Stability AI, Replicate)
- **Instagram Posting**: Automated posting to both Instagram feed and stories
- **Flexible Scheduling**: Customizable posting schedule with multiple posts per day
- **Content Variety**: Automatic generation of varied image conditions for diverse content
- **API Compliance**: Respects Instagram's 25 posts per day limit
- **Easy Configuration**: JSON-based configuration for quick parameter changes

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

2. Edit `config.json` to customize your AI celebrity:
   - Celebrity profile (name, age, personality, interests)
   - Image generation conditions (style, lighting, locations)
   - Posting schedule (times, frequency)
   - Instagram API credentials

### 3. API Setup Required

#### Instagram Graph API Setup:
1. Create Facebook App at developers.facebook.com
2. Add Instagram Basic Display product
3. Create Instagram Business Account
4. Connect Facebook Page to Instagram Business Account
5. Get access token with `instagram_basic` and `instagram_content_publish` permissions

#### AI Image Generation (choose one):
- **OpenAI**: Get API key from platform.openai.com
- **Stability AI**: Get API key from beta.dreamstudio.ai
- **Replicate**: Get API token from replicate.com

### 4. Running the System

```bash
# Validate configuration
python ai_celebrity_app.py --validate

# Test with single posts
python ai_celebrity_app.py --test-feed
python ai_celebrity_app.py --test-story

# Run automated posting
python ai_celebrity_app.py

# Use custom config file
python ai_celebrity_app.py --config my_custom_config.json
```

## File Structure

- `ai_celebrity_config.py` - Configuration classes and celebrity profile system
- `image_generator.py` - AI image generation with multiple service support
- `instagram_poster.py` - Instagram Graph API integration
- `ai_celebrity_app.py` - Main application with scheduling
- `config.json` - Editable configuration file
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## Customization Examples

### Change Celebrity Persona
Edit `config.json`:
```json
{
  "celebrity": {
    "name": "Alex Thompson",
    "age": 28,
    "occupation": "travel photographer",
    "personality_traits": ["adventurous", "creative", "inspiring"],
    "interests": ["travel", "photography", "culture", "food"]
  }
}
```

### Modify Image Conditions
```json
{
  "image_conditions": {
    "style": "travel",
    "lighting": "golden hour",
    "location": "scenic outdoor location",
    "outfit_style": "casual travel wear",
    "mood": "adventurous and free"
  }
}
```

### Adjust Posting Schedule
```json
{
  "posting_schedule": {
    "feed_posts_per_day": 2,
    "story_posts_per_day": 4,
    "posting_times": ["09:00", "12:00", "15:00", "18:00"],
    "active_days": ["Monday", "Wednesday", "Friday"]
  }
}
```

## API Limits & Compliance

- **Instagram**: 25 posts per day per account (combined feed + stories)
- **OpenAI**: Rate limits vary by plan
- **Stability AI**: Credits-based system
- **Replicate**: Pay-per-use pricing

The system automatically tracks daily posting limits and respects Instagram's restrictions.

## Content Generation

The system generates varied content by randomizing:
- Image styles (realistic, artistic, fashion, fitness, lifestyle, travel)
- Locations (studio, outdoor, urban, etc.)
- Lighting conditions (natural, golden hour, studio)
- Outfit styles (casual, formal, athletic, etc.)
- Poses and moods

## Logging & Monitoring

- All activities logged to `ai_celebrity.log`
- Post history tracked in `post_history.json`
- Real-time console output for monitoring

## Error Handling

- Automatic retry logic for API failures
- Graceful handling of rate limits
- Comprehensive error logging
- Connection validation before posting

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly rotate access tokens
- Monitor API usage and costs

## Troubleshooting

### Common Issues:

1. **Instagram API Errors**: Verify business account setup and permissions
2. **Image Generation Fails**: Check API keys and credit balance
3. **Posting Limit Reached**: System respects 25 posts/day limit automatically
4. **Connection Issues**: Validate credentials with `--validate` flag

### Support:

For issues with:
- Instagram API: Check Meta's developer documentation
- Image generation: Refer to respective AI service documentation
- System bugs: Check logs in `ai_celebrity.log`
