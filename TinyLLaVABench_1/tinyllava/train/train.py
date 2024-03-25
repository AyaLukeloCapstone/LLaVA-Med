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
import re
import time
from dataclasses import dataclass, field
import json
import logging
from packaging import version
import pathlib
from typing import Dict, Optional, Sequence, List


import torch
import tokenizers
import transformers 
from torch.utils.data import Dataset
from peft import prepare_model_for_kbit_training


################
import sys
# Add the directory containing the `llava` module to the Python path
sys.path.append('/scratch/ae2195/LLaVA-Med/llava')

# imported from train.py for teacher model 
from llava import LlavaLlamaForCausalLM
################

from tinyllava.train.llava_trainer import LLaVATrainer
from tinyllava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
from tinyllava import conversation as conversation_lib
from tinyllava.model import *
from tinyllava.mm_utils import tokenizer_image_token
from tinyllava.train.train_utils import *
from tinyllava.utils import rank0_print, local_rank
from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.model.model_factory import *
from tinyllava.arguments import *

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def train():
    global local_rank
    # 1. load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # 2. prepare model
    # 2.1 kbit & compute_dtype  ===>  model
    # 2.2 vision_tower.property  and load
    
    # 3. prepare tokenizer
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = get_bnb_model_args(training_args)
    # TODO: vision_tower type check
    if model_args.vision_tower is not None:
        model = ModelSelect(model_args.model_name_or_path).from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
            attn_implementation="flash_attention_2",
            torch_dtype=compute_dtype
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if training_args.lora_enable:
        model = lora_setting(model, training_args)
    
    Tokenizer, init_tokenizer = TokenizerSelect(model_args.model_name_or_path)()
    tokenizer = Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer = init_tokenizer(tokenizer)

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token = tokenizer.pad_token

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    model.config.tokenizer_padding_side = tokenizer.padding_side
    if model_args.vision_tower is not None:
        # model.config.tune_embed_tokens = training_args.tune_embed_tokens = model_args.tune_embed_tokens
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        
        if training_args.gradient_checkpointing:
            vision_tower.vision_tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(vision_tower.vision_tower, "enable_input_require_grads"):
                vision_tower.vision_tower.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                vision_tower.vision_tower.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.tune_vision_tower = training_args.tune_vision_tower = model_args.tune_vision_tower
        model.config.tune_entire_model = training_args.tune_entire_model = model_args.tune_entire_model
        if model_args.tune_entire_model:
            rank0_print(f'Tune entire model!')
            lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
            rank0_print(f'Tune the MLP! The LR of MLP is {lr_of_mlp}')
            if training_args.lora_enable:
                unlock_vit(training_args, model_args, vision_tower)
            else:
                model.requires_grad_(True)
                unlock_vit(training_args, model_args, vision_tower)

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    rank0_print(model.get_model().mm_projector)

    if training_args.bits in [4, 8]:
        lora_kbit_setting(model, training_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    rank0_print("trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    rank0_print("total parameters: ", sum(p.numel() for p in model.parameters()))


    ############### Changing the training code of TinyLlava for including Knowledge distillation ###############

    # initializing the teacher model 

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
     
        teacher_model.model.vision_tower[0].to(dtype=dtype, device=training_args.device)
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
        teacher_model.initialize_vision_tokenizer(mm_use_im_start_end=teacher_model_args.mm_use_im_start_end, tokenizer=teacher_tokenizer, device=training_args.device,
                                          tune_mm_mlp_adapter=teacher_model_args.tune_mm_mlp_adapter, pretrain_mm_mlp_adapter=teacher_model_args.pretrain_mm_mlp_adapter)

    ########################################################

    # trainer = LLaVATrainer(model=model,
    #                        tokenizer=tokenizer,
    #                        args=training_args,
    #                        **data_module)

    ### trainer object - instance of LLaVATrainer class 
    trainer = LLaVATrainer(model=model,
                tokenizer=tokenizer,
                teacher_model=teacher_model,
                args=training_args,
                **data_module)


########################################################


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()

    trainer.save_state()
    # model.config.use_cache = True
    # if training_args.lora_enable:
    #     lora_save_model(model, training_args)
    # else:
    safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
