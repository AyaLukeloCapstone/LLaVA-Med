import os
import json

# Directory containing images
image_dir = 'data/images'
# Output JSON file
output_file = 'images.json'

# List all JPEG files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Create a list or dictionary containing the image information
# This example simply lists the file paths
image_data = [{'image_path': os.path.join(image_dir, f)} for f in image_files]

# Write the data to a JSON file
with open(output_file, 'w') as f:
    json.dump(image_data, f, indent=4)

print(f"JSON file created with {len(image_files)} images.")