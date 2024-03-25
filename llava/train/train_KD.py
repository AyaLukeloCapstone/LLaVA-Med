# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch

import transformers 
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer 

from llava import conversation as conversation_lib
from llava import LlavaLlamaForCausalLM

from PIL import Image
import torch.nn as nn
import math


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

            if isinstance(sentence["value"], int):
                sentence["value"] = str(sentence["value"])
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.multimodal_cfg['image_folder']
            processor = self.multimodal_cfg['image_processor']
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except Exception as exn:
                print(exn)
                import random
                return random.choice(self)

            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
            elif self.multimodal_cfg['image_aspect_ratio'] == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # import pdb; pdb.set_trace()
            image_token_len = self.multimodal_cfg["image_token_len"]
            patch_size = int(image.shape[1]//math.sqrt(image_token_len))
            cur_token_len = (image.shape[1]//patch_size) * (image.shape[2]//patch_size)   # FIXME: 14 is hardcoded patch size

            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversatons"] for e in sources])

            sources = preprocess_multimodal(
                sources,
                self.multimodal_cfg, cur_token_len)
        else:
            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversatons"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.multimodal_cfg['is_multimodal']:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer, 
                                data_path=data_args.data_path,
                                multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    image_token_len=data_args.image_token_len,
                                    image_folder=data_args.image_folder,
                                    image_aspect_ratio=data_args.image_aspect_ratio,
                                    use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
                                    image_processor=getattr(data_args, 'image_processor', None)))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

## TRAINING CODE HERE 
def train():

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda'

    ##STUDENT MODEL:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    student_model_args, data_args, student_training_args = parser.parse_args_into_dataclasses()


    if student_model_args.vision_tower is not None:
        student_model = LlavaLlamaForCausalLM.from_pretrained(
            student_model_args.model_name_or_path,
            cache_dir=student_training_args.cache_dir
        )
        # pip install accelerate
    else:
        student_model = transformers.LlamaForCausalLM.from_pretrained(
            student_model_args.model_name_or_path,
            cache_dir=student_training_args.cache_dir
        )

    student_device = student_model.device
    student_model.config.use_cache = False

    if student_model_args.freeze_backbone:
        student_model.model.requires_grad_(False)


    student_tokenizer = transformers.AutoTokenizer.from_pretrained(
        student_model_args.model_name_or_path,
        cache_dir=student_training_args.cache_dir,
        model_max_length=student_training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if student_model_args.version == "v0":
        if student_tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=student_tokenizer,
                model=student_model,
            )
        if "llama" in student_model_args.model_name_or_path:
            student_tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        student_tokenizer.pad_token = student_tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]


    if student_model_args.vision_tower is not None:
        model_vision_dict = student_model.model.initialize_vision_modules(
            vision_tower=student_model_args.vision_tower,
            mm_vision_select_layer=student_model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=student_model_args.pretrain_mm_mlp_adapter
        )
        dtype = torch.float32
        if student_training_args.fp16:
            dtype = torch.float16
        if student_training_args.bf16:
            dtype = torch.bfloat16
        # student_model.model.vision_tower[0].to(dtype=dtype, device=student_device)
        vision_config = model_vision_dict['vision_config']


        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.image_processor = model_vision_dict['image_processor']
        data_args.is_multimodal = True


        student_model.config.tune_mm_mlp_adapter = student_training_args.tune_mm_mlp_adapter = student_model_args.tune_mm_mlp_adapter
        if student_model_args.tune_mm_mlp_adapter:
            student_model.requires_grad_(False)
            for p in student_model.model.mm_projector.parameters():
                p.requires_grad = True


        student_model.config.freeze_mm_mlp_adapter = student_training_args.freeze_mm_mlp_adapter
        if student_training_args.freeze_mm_mlp_adapter:
            for p in student_model.model.mm_projector.parameters():
                p.requires_grad = False


        student_model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = student_model_args.mm_use_im_start_end
        vision_config.use_im_start_end = student_training_args.use_im_start_end = student_model_args.mm_use_im_start_end
        student_model.initialize_vision_tokenizer(mm_use_im_start_end=student_model_args.mm_use_im_start_end, tokenizer=student_tokenizer, device=student_device,
                                          tune_mm_mlp_adapter=student_model_args.tune_mm_mlp_adapter, pretrain_mm_mlp_adapter=student_model_args.pretrain_mm_mlp_adapter)


        params_no_grad = [n for n, p in student_model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if student_training_args.fsdp is not None and len(student_training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")


                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func


                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)


    ##TEACHER MODEL:
    ##create data arguments
    teacher_model_args = ModelArguments(
    model_name_or_path="/scratch/ae2195/LLaVA-Med/checkpoints/llava-med-7b-pretrain",
    version="v0",
    freeze_backbone=True,
    tune_mm_mlp_adapter=True,
    vision_tower="openai/clip-vit-large-patch14",
    mm_vision_select_layer=-2,
    pretrain_mm_mlp_adapter=None,
    mm_use_im_start_end=True)


    teacher_training_args = TrainingArguments(
        output_dir="/scratch/ae2195/LLaVA-Med/checkpoints/llava-med-7b-teacher",
        overwrite_output_dir=True,
        model_max_length= 2048,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=1,)


    if teacher_model_args.vision_tower is not None:
        teacher_model = LlavaLlamaForCausalLM.from_pretrained(
            teacher_model_args.model_name_or_path,
            cache_dir=teacher_training_args.cache_dir
        )
    else:
        teacher_model = transformers.LlamaForCausalLM.from_pretrained(
            teacher_model_args.model_name_or_path,
            cache_dir=teacher_training_args.cache_dir
        )
    teacher_model.config.use_cache = False


    if teacher_model_args.freeze_backbone:
        teacher_model.model.requires_grad_(False)


    teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(
        teacher_model_args.model_name_or_path,
        cache_dir=teacher_training_args.cache_dir,
        model_max_length=teacher_training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )


    if teacher_model_args.version == "v0":
        if teacher_tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=teacher_tokenizer,
                model=teacher_model,
            )
        if "llama" in teacher_model_args.model_name_or_path:
            teacher_tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        teacher_tokenizer.pad_token = teacher_tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]


    if teacher_model_args.vision_tower is not None:
        model_vision_dict = teacher_model.model.initialize_vision_modules(
            vision_tower=teacher_model_args.vision_tower,
            mm_vision_select_layer=teacher_model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=teacher_model_args.pretrain_mm_mlp_adapter
        )
        dtype = torch.float32
        if teacher_training_args.fp16:
            dtype = torch.float16
        if teacher_training_args.bf16:
            dtype = torch.bfloat16

        # teacher device
        teacher_device = teacher_model.device
     
        # teacher_model.model.vision_tower[0].to(dtype=dtype, device=teacher_device)
        vision_config = model_vision_dict['vision_config']


        teacher_model.config.tune_mm_mlp_adapter = teacher_training_args.tune_mm_mlp_adapter = teacher_model_args.tune_mm_mlp_adapter
        if teacher_model_args.tune_mm_mlp_adapter:
            teacher_model.requires_grad_(False)
            for p in teacher_model.model.mm_projector.parameters():
                p.requires_grad = True


        teacher_model.config.freeze_mm_mlp_adapter = teacher_training_args.freeze_mm_mlp_adapter
        if teacher_training_args.freeze_mm_mlp_adapter:
            for p in model.model.mm_projector.parameters():
                p.requires_grad = False


        teacher_model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = teacher_model_args.mm_use_im_start_end
        vision_config.use_im_start_end = teacher_training_args.use_im_start_end = teacher_model_args.mm_use_im_start_end
        teacher_model.initialize_vision_tokenizer(mm_use_im_start_end=teacher_model_args.mm_use_im_start_end, tokenizer=student_tokenizer, device=teacher_device,
                                          tune_mm_mlp_adapter=teacher_model_args.tune_mm_mlp_adapter, pretrain_mm_mlp_adapter=teacher_model_args.pretrain_mm_mlp_adapter)


        params_no_grad = [n for n, p in teacher_model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if teacher_training_args.fsdp is not None and len(teacher_training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")


                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func
                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)


    data_module = make_supervised_data_module(tokenizer=student_tokenizer,
                                              data_args=data_args)
    student_trainer = LLaVATrainer(model=student_model,
                    tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    args=student_training_args,
                    **data_module)


    if list(pathlib.Path(student_training_args.output_dir).glob("checkpoint-*")):
        student_trainer.train(resume_from_checkpoint=True)
    else:
        student_trainer.train()
    student_trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=student_trainer,
                                   output_dir=student_training_args.output_dir)




if __name__ == "__main__":
    train()


