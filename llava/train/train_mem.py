# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn


# # new added
# # Parse command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--input_json', required=True, help='Path to the JSON file containing image paths')
# args = parser.parse_args()

# # Load the JSON file
# with open(args.input_json, 'r') as f:
#     image_data = json.load(f)


# #end

replace_llama_attn_with_flash_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()
