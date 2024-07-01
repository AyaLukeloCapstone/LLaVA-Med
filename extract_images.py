import os
import shutil

# Define the source directory and the image file name
source_directory = 'data/Allimages'
image_name = '23094188_fig2.jpg'  # Replace with your specific image file name

# Define the target directory
target_directory = 'demo_images'

# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Construct the full path to the source and target files
source_file = os.path.join(source_directory, image_name)
target_file = os.path.join(target_directory, image_name)

# Check if the source file exists and then copy it to the target directory
if os.path.exists(source_file):
    shutil.copy(source_file, target_file)
    print(f'Image {image_name} copied successfully to {target_directory}')
else:
    print(f'Image {image_name} not found in {source_directory}')