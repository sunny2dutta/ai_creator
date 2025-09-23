#!/bin/bash
cd /Users/debaryadutta/ai_creator
echo "$(date): Running scheduled post" >> scheduler.log
python -m src.core.facebook_image_post >> scheduler.log 2>&1