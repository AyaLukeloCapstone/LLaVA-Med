# import os
# import json
# import shutil
# import tarfile
# import argparse
# import logging
# import urllib.request
# from urllib.error import HTTPError, URLError
# from tqdm import tqdm

# def download_file(url, output_path, max_retries=3):
#     retries = 0
#     while retries < max_retries:
#         try:
#             urllib.request.urlretrieve(url, output_path)
#             return True
#         except (HTTPError, URLError) as e:
#             logging.warning(f"Retry {retries+1}: Error downloading {url} - {e}")
#             retries += 1
#     return False

# def main(args):
#     input_data = []
#     with open(args.input_path) as f:
#         for line in f:
#             input_data.append(json.loads(line))

#     # Download all PMC articles
#     logging.info('Downloading PMC articles')
#     for sample in tqdm(input_data):
#         pmc_tar_url = sample['pmc_tar_url']
#         output_path = os.path.join(args.pmc_output_path, os.path.basename(pmc_tar_url))
#         if not download_file(pmc_tar_url, output_path):
#             logging.error(f'Failed to download {pmc_tar_url} after retries')

#     # Untar all PMC articles
#     logging.info('Untarring PMC articles')
#     for sample in tqdm(input_data):
#         fname = os.path.join(args.pmc_output_path, os.path.basename(sample['pmc_tar_url']))
#         if os.path.exists(fname):
#             try:
#                 with tarfile.open(fname, "r:gz") as tar:
#                     tar.extractall(args.pmc_output_path)
#             except tarfile.TarError as e:
#                 logging.error(f"Error extracting {fname}: {e}")

#     # Copy to images directory
#     logging.info('Copying images')
#     for sample in tqdm(input_data):
#         src = os.path.join(args.pmc_output_path, sample['image_file_path'])
#         dst = os.path.join(args.images_output_path, sample['pair_id']+'.jpg')
#         if os.path.exists(src) and os.path.isfile(src):
#             shutil.copyfile(src, dst)
#         else:
#             logging.warning(f"Image file not found or invalid: {src}")

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_path', type=str, default='data/llava_med_image_urls.jsonl')
#     parser.add_argument('--pmc_output_path', type=str, default='data/pmc_articles/')
#     parser.add_argument('--images_output_path', type=str, default='data/images/')
#     args = parser.parse_args()
#     main(args)

# import os
# import json
# import shutil
# from tqdm import tqdm
# import tarfile
# import argparse
# from urllib.error import HTTPError

# import urllib.request
# from concurrent.futures import ThreadPoolExecutor, as_completed

# def download(url, output_path):
#     try:
#         urllib.request.urlretrieve(url, output_path)
#     except (HTTPError) as e:
#         pass
#         #logging.warning(f"Retry {retries+1}: Error downloading {url} - {e}")


# def main(args):
#     input_data = []
#     with open(args.input_path) as f:
#         for line in f:
#             input_data.append(json.loads(line))

#     # Download all PMC articles
#     print('Downloading PMC articles')
#     with ThreadPoolExecutor(max_workers=30) as executor:
#         results = [executor.submit(download, sample['pmc_tar_url'], os.path.join(args.pmc_output_path, os.path.basename(sample['pmc_tar_url']))) for sample in input_data] 
#         for future in tqdm(as_completed(results), total=len(input_data)):
#             future.result()

#     # Untar all PMC articles
#     print('Untarring PMC articles')
#     for sample in tqdm(input_data):
#         fname = os.path.join(args.pmc_output_path, os.path.basename(os.path.join(sample['pmc_tar_url'])))
#         tar = tarfile.open(fname, "r:gz")
#         tar.extractall(args.pmc_output_path)
#         tar.close()
        
#     # Copy to images directory
#     print('Copying images')
#     for sample in tqdm(input_data):
#         src = os.path.join(args.pmc_output_path, sample['image_file_path'])
#         dst = os.path.join(args.images_output_path, sample['pair_id']+'.jpg')
#         shutil.copyfile(src, dst)
      

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_path', type=str, default='data/llava_med_image_urls.jsonl')
#     parser.add_argument('--pmc_output_path', type=str, default='data/pmc_articles/')
#     parser.add_argument('--images_output_path', type=str, default='data/images/')
#     args = parser.parse_args()
#     main(args)


##our modificication: download images properly:
import os
import json
import shutil
import tarfile
import argparse
from urllib.error import HTTPError
import urllib.request
from multiprocessing import Pool
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/llava_med_image_urls.jsonl')
parser.add_argument('--pmc_output_path', type=str, default='data/pmc_articles/')
parser.add_argument('--images_output_path', type=str, default='data/images/')
parser.add_argument('--remove_pmc', action='store_true', default=True, help='remove pmc articles after image extraction')
parser.add_argument('--cpus', type=int, default=-1, help='number of cpus to use in multiprocessing (default: all)')
args = parser.parse_args()

input_data = []
with open(args.input_path) as f:
    for line_number, line in enumerate(f, 1):
        try:
            if line.strip():  # Checks if the line is not empty
                input_data.append(json.loads(line))
            else:
                print(f"Skipping empty line at {line_number}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")

#this function check first if the image exist, if not then it try to download it
def download_func(idx):
    sample = input_data[idx]
    dst = os.path.join(args.images_output_path, sample['pair_id']+'.jpg')
    print("Start Downloading image: ",dst)

    # Check if the image already exists
    if os.path.exists(dst):
        print(f"Image {dst} already exists. Skipping download and extraction.")
        return

    try:
        # Download the tar file
        urllib.request.urlretrieve(sample['pmc_tar_url'], os.path.join(args.pmc_output_path, os.path.basename(sample['pmc_tar_url'])))
        fname = os.path.join(args.pmc_output_path, os.path.basename(os.path.join(sample['pmc_tar_url'])))

        # Extract the tar file
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(args.pmc_output_path)
        tar.close()

        # Copy the specific image file
        src = os.path.join(args.pmc_output_path, sample['image_file_path'])
        shutil.copyfile(src, dst)  

        # Clean up if specified
        if args.remove_pmc:
            os.remove(fname)
            shutil.rmtree(os.path.join(args.pmc_output_path, str(os.path.basename(sample['pmc_tar_url']))).split('.tar.gz')[0]+'/')
    except Exception as e:
        print(e)
        
if args.cpus == -1:
    cpus = mp.cpu_count()
else:
    cpus = args.cpus
    
pool = Pool(cpus)

pool.map(download_func, range(0, len(input_data)))