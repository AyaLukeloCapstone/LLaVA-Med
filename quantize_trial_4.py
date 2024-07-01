##version 1

# import torch
# from PIL import Image
# import os
# from transformers import AutoTokenizer, AutoConfig
# from llava import LlavaLlamaForCausalLM
# from llava.utils import disable_torch_init
# from llava.conversation import conv_templates
# from transformers import CLIPVisionModel, CLIPImageProcessor
# from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
# from accelerate import Accelerator

# # Configuration for 4-bit quantization
# print("Configuring quantization parameters...")
# bnb_quantization_config = BnbQuantizationConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

# # Loading and quantizing the model
# print("Loading and quantizing the model...")
# model_name = "/scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain"
# model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# quantized_model = load_and_quantize_model(
#     model, 
#     weights_location=model_name,
#     bnb_quantization_config=bnb_quantization_config,
#     device_map='auto'
# ).cuda()


# # Save the quantized model
# print("Saving the quantized model...")
# accelerator = Accelerator()
# save_path = "/scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-quantized"
# accelerator.save_model(quantized_model, save_path)
# print("Model saved successfully at:", save_path)


### version 2:
# import torch
# from PIL import Image
# from transformers import AutoTokenizer, AutoConfig
# from llava import LlavaLlamaForCausalLM
# from llava.utils import disable_torch_init
# from llava.conversation import conv_templates
# from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
# from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
# from accelerate import Accelerator

# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"


# # Configuration for 4-bit quantization
# print("Configuring quantization parameters...")
# bnb_quantization_config = BnbQuantizationConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

# # Loading and quantizing the model
# print("Loading and quantizing the model...")
# model_name = "/scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-SG1_Final"
# model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
# # quantized_model = load_and_quantize_model(
# #     model, 
# #     weights_location=model_name,
# #     bnb_quantization_config=bnb_quantization_config,
# #     device_map='auto'
# # )

# # Load tokenizer and image processor
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
# print("Loaded Vision tower: ",vision_tower)
# mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
# tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
# if mm_use_im_start_end:
#     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

# vision_config = vision_tower.config
# vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
# vision_config.use_im_start_end = mm_use_im_start_end
# if mm_use_im_start_end:
#     vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

# image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

# mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
# mm_projector_weights = torch.load("/scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-SG1_Final/mm_projector.bin", map_location='cpu')
# mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

# model.model.mm_projector = mm_projector.cuda().half()
# model.model.vision_tower = [vision_tower]


# # Define your hardcoded question and associated image
# hardcoded_question = {
#     "question_id": 1,
#     "text": "Provide a detailed description of the given image\n<image>",
#     "image": "_f1-asm-5-401.jpg"  # Update this path to the actual image location
# }

# # Processing the question
# image_file = hardcoded_question["image"]
# qs = hardcoded_question["text"]
# cur_prompt = qs + '\n' + "<image>"
# prompt = tokenizer(cur_prompt, return_tensors="pt")

# # Load image
# image_path = "/scratch/ltl2113/LLaVA-Med/images/" + hardcoded_question["image"]
# image = Image.open(image_path)

# # Process image
# image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"][0]
# # Check the image tensor shape
# # print("Image tensor shape before adjustment:", image_tensor['pixel_values'].shape)

# # # Adjust the image tensor dimension if necessary
# # if image_tensor['pixel_values'].dim() == 5:  # Example condition check for an extra dimension
# #     image_tensor['pixel_values'] = image_tensor['pixel_values'].squeeze(0)

# # print("Image tensor shape after adjustment:", image_tensor['pixel_values'].shape)


# # Generating output
# input_ids = prompt["input_ids"].cuda()
# print("model: ",model)
# print("model device: ",model.device)
# print("Inputs ids: ",input_ids)
# print("model vision tower: ",model.model.vision_tower)
# # print("model vision tower device: ",model.model.vision_tower.device)
# with torch.inference_mode():
#     output_ids = model.generate(
#         input_ids=input_ids,
#         images=image_tensor.unsqueeze(0).cuda(),  # Ensuring the dimension matches expected input
#         max_new_tokens=200,
#         do_sample=True,
#         temperature=0.7
#     )
# print("Output ids: ",output_ids)

# # Process output
# decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print("Output:", decoded_output)

# # Save the quantized model
# print("Saving the quantized model...")
# # accelerator = Accelerator()
# # save_path = "/scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-SG1_Final-quantized-3"
# # # quantized_model.save_pretrained(save_path)
# # # accelerator.save_model(quantized_model, save_path)
# # print("Model saved successfully at:", save_path)

##VERSION 3
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from accelerate import Accelerator

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"




detail_describe_instructions = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]

concise_describe_instructions = [
    "Describe the following image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the following image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo below.",
    "Write a terse but informative summary of the following picture.",
    "Create a compact narrative representing the image presented.",
]

prompt_pool = detail_describe_instructions + concise_describe_instructions

prompt_pool = [ "Describe the following image in detail."]


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        print("Here now 0")
        
        print(model_name)
        if "BiomedCLIP" in model_name or "biomed_clip" in model_name:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, use_cache=True).cuda()
            model = model.to(torch.float16)
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
            openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_config = openai_vision_tower.config
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            setattr(vision_tower, 'config', vision_config)
        else:
            print("Here now 1")
            myoriginal_model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()
            bnb_quantization_config = BnbQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
            model = load_and_quantize_model(
                myoriginal_model, 
                weights_location=model_name,
                bnb_quantization_config=bnb_quantization_config,
                device_map='auto')

            save_path = "/scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-SG1_Final-quantized-test"
            model.save_pretrained(save_path)
            # accelerator.save_model(model, save_path)

            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # import pdb; pdb.set_trace()
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # in case of using a pretrained model with only a MLP projector weights
        print("Here 2:")
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()

        if "BiomedCLIP" in model.config.mm_vision_tower:
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)


        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(answers_file), "images"), exist_ok=True)
    ans_file = open(answers_file, "w")
    save_image_folder = os.path.join(os.path.dirname(os.path.expanduser(args.answers_file)), "images")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        # question = line['conversations'][0]
        # gt_ans = line["conversations"][1]

        try:
            question = line["conversations"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversations"][1] # ['value']        
        except:
            question = line["conversatons"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversatons"][1] # ['value']    

        qs = question['value']

        qs = qs.replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
            else:
                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            cur_prompt = cur_prompt + '\n' + '<image>'
        else:
            images = None

        if args.conv_mode == 'simple_legacy':
            qs += '\n\n### Response:'
        assert gt_ans['from'] == 'gpt'
        # conv = default_conversation.copy()
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        # TODO: new implementation
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        if args.conv_mode == 'simple_legacy':
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break

        try:
            index = outputs.index(conv.sep)
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep)

        outputs = outputs[:index].strip()

        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs
            inputs = tokenizer([prompt + outputs_reasoning + ' ###\nANSWER:'])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            keywords = ['###']
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=64,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs

        # new implementation ends

        # original implementation
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # try:
        #     index = outputs.index(conv.sep, len(prompt))
        # except ValueError:
        #     outputs += conv.sep
        #     index = outputs.index(conv.sep, len(prompt))

        # outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    args = parser.parse_args()

    eval_model(args)
