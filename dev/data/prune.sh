#!/bin/bash

# Process videos in both directories
for dir in "train-smol" "val-smol"; do
    echo "Processing directory: $dir"
    
    # Find all video files
    find "$dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" \) | while read -r video; do
        echo "Processing: $video"
        
        # Create temporary output filename
        temp_output="${video%.*}_temp.mp4"
        
        # Process the video
        ffmpeg -y -i "$video" -vframes 32 -an "$temp_output" && \
        mv "$temp_output" "$video"
    done
done
