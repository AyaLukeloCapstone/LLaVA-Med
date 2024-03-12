# ORGINAL CODE THAT WORKED 



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



# NOW THAT the download is done focusing on the untarring, untaring one PMC article one by one and then copying to the images path 

import os
import json
import shutil
from tqdm import tqdm
import tarfile
import argparse
from urllib.error import HTTPError

import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def main(args):
    input_data = []
    with open(args.input_path) as f:
        for line in f:
            input_data.append(json.loads(line))

    print('Processing PMC articles')
    for sample in tqdm(input_data):
        fname = os.path.join(args.pmc_output_path, os.path.basename(os.path.join(sample['pmc_tar_url'])))
        # Check if the tar file exists before attempting to untar
        if os.path.exists(fname):
            try:
                tar = tarfile.open(fname, "r:gz")
                tar.extractall(args.pmc_output_path)
                tar.close()

                # After untarring, immediately attempt to copy the image
                src = os.path.join(args.pmc_output_path, sample['image_file_path'])
                dst = os.path.join(args.images_output_path, sample['pair_id']+'.jpg')
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
                else:
                    print(f"Source file not found, skipping copy: {src}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
        else:
            print(f"File not found, skipping: {fname}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/llava_med_image_urls.jsonl')
    parser.add_argument('--pmc_output_path', type=str, default='data/pmc_articles/')
    parser.add_argument('--images_output_path', type=str, default='data/images/')
    args = parser.parse_args()
    main(args)
