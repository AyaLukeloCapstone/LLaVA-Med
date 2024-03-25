# from tinyllava.model.builder import load_pretrained_model
# from tinyllava.mm_utils import get_model_name_from_path
# from tinyllava.eval.run_tiny_llava import eval_model

# model_path = "/scratch/ltl2113/LLaVA-Med/TinyLLaVABench/checkpoints/tiny-llava-base-pretrain/checkpoint-48000"
# prompt = "What is the organ in the image?"
# image_file = "/scratch/ltl2113/LLaVA-Med/data/qa50_images/18063892_F3.jpg"

# args = type('Args', (), {
#     "model_path": model_path,
#     "model_base": None,
#     "model_name": get_model_name_from_path(model_path),
#     "query": prompt,
#     "conv_mode": "phi",
#     "image_file": image_file,
#     "sep": ",",
#     "temperature": 0,
#     "top_p": None,
#     "num_beams": 1,
#     "max_new_tokens": 512
# })()

# eval_model(args)

from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path
from tinyllava.eval.run_tiny_llava import eval_model

model_path = "bczhou/TinyLLaVA-3.1B"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": "phi",
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)

