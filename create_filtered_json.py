import json
import os

# Define the path to the directory containing your images
image_directory = '/scratch/ae2195/LLaVA-Med/data/images'
json_file_path = '/scratch/ae2195/LLaVA-Med/data/alignment/llava_med_alignment_500k.json'

# Load JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Initialize a counter for progress tracking
total_items = len(data)
processed = 0

# Filter out items where the corresponding image file doesn't exist in the directory
filtered_data = []
for item in data:
    if os.path.exists(os.path.join(image_directory, item['image'])):
        filtered_data.append(item)
    processed += 1
    print(f"Processed {processed}/{total_items} items...")

# Save the filtered data back to a new JSON file
with open('/scratch/ae2195/LLaVA-Med/data/alignment/filtered_json_file.json', 'w') as file:
    json.dump(filtered_data, file, indent=4)

print(f"Filtered data saved to 'filtered_json_file.json'")
