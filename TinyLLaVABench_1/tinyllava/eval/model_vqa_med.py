import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from tinyllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


#####################
#Added from LLavaMed
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

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

##########################


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    #ORIGINAL CODE FROM TINY LLAVA MODEL_VQA_SCIENCE
    # questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    # for i, line in enumerate(tqdm(questions)):
    #     idx = line["id"]
    #     question = line['conversations'][0]
    #     qs = question['value'].replace('<image>', '').strip()
    #     cur_prompt = qs

    #     if 'image' in line:
    #         image_file = line["image"]
    #         image = Image.open(os.path.join(args.image_folder, image_file))
    #         image_tensor = process_images([image], image_processor, model.config)[0]
    #         images = image_tensor.unsqueeze(0).half().cuda()
            
    #         if getattr(model.config, 'mm_use_im_start_end', False):
    #             qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #         else:
    #             qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    #         cur_prompt = '<image>' + '\n' + cur_prompt
    #     else:
    #         images = None
            

    #     if args.single_pred_prompt:
    #         qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
    #         cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

    #     conv = conv_templates[args.conv_mode].copy()
    #     conv.append_message(conv.roles[0], qs)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()

    #     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    #     with torch.inference_mode():
    #         output_ids = model.generate(
    #             input_ids,
    #             images=images,
    #             do_sample=True if args.temperature > 0 else False,
    #             temperature=args.temperature,
    #             max_new_tokens=1024,
    #             use_cache=True,

    #         )

    #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    #     ans_id = shortuuid.uuid()
    #     ans_file.write(json.dumps({"question_id": idx,
    #                                "prompt": cur_prompt,
    #                                "text": outputs,
    #                                "answer_id": ans_id,
    #                                "model_id": model_name,
    #                                "metadata": {}}) + "\n")
    #     ans_file.flush()

    #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    #     ans_id = shortuuid.uuid()
    #     ans_file.write(json.dumps({"question_id": idx,
    #                                "prompt": cur_prompt,
    #                                "text": outputs,
    #                                "answer_id": ans_id,
    #                                "model_id": model_name,
    #                                "metadata": {}}) + "\n")
    #     ans_file.flush()
    # ans_file.close()

    #added this part as image_token_len is not initialized anywhere
    image_token_len=256
    #CODE FROM MODEL_VQA_MED 
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
        print(f"Input: {cur_prompt}, Output:{output_ids}")
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
        # print("Input token length:", input_token_len)
        # print("Input IDs size:", input_ids.size())
        # print("Output IDs size:", output_ids.size())
        #comment this part as it was commented in tiny llava
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        #added this from tinny llava:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

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
        print(f"Input: {cur_prompt}, Output:{output_ids}")
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
            #comment this part as it was commented on tinny lllava code
            # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            # if n_diff_input_output > 0:
            #     print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')

            #outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            #added this 
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs
            print(f"Input: {cur_prompt}, Output:{output_ids}")

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
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)


