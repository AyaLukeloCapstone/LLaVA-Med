import os
import torch
import torch.nn as nn


from transformers import Trainer
from typing import Dict, Optional, Sequence
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence


import torch


import transformers
from torch.utils.data import Dataset


from llava import conversation as conversation_lib
from llava import LlavaLlamaForCausalLM


from PIL import Image
import torch.nn as nn
import math




def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).


    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model




class LLaVATrainer(Trainer):
    def __init__(self, model, tokenizer,teacher_model, *args, **kwargs):
        # Initialize the Trainer with training_args
        super().__init__(model=model, tokenizer=tokenizer, *args, **kwargs)
       
        self.teacher_model = teacher_model
   
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     Overrides the default compute_loss to implement distillation loss.
    #     """
    #     # Assuming 'inputs' already contains 'labels', and 'logits' from the student model
    #     if "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
       
    #     # Forward pass through the student model
    #     outputs = model(**inputs)
       
    #     # Compute distillation loss
    #     distil_loss = self.get_distil_loss(model, outputs, labels, inputs) #             
    #     # TRY TO PASS THE INPUTS AS AN ARGUMENT HERE 
       
    #     return (distil_loss, outputs) if return_outputs else distil_loss


    # def get_distil_loss(self, model, outputs, labels, inputs):
    #     with torch.no_grad():
    #         self.teacher_model.eval()
    #         # Assuming 'inputs' here should be compatible with the teacher model
    #         teacher_outputs = self.teacher_model(**inputs, use_cache=False)
    #         teacher_logits = teacher_outputs.logits
       
    #     logits = outputs.logits  # Student logits
       
    #     if self.args.model_parallel:
    #         # Custom implementation for model parallelism
    #         pass  # Placeholder for parallel loss calculation
    #     else:
    #         teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    #         inf_mask = torch.isinf(logits)
    #         logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    #         prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    #         x = torch.sum(prod_probs, dim=-1).view(-1)
    #         mask = (labels != -100).int()  # Assuming -100 is used for ignored indices
    #         distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
       
    #     return distil_loss

    # def compute_loss(self, student_model, inputs, return_outputs=False):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     inputs = inputs.to(device)
    #     outputs_student = student_model(**inputs)


    # def compute_loss(self, student_model, inputs, return_outputs=False):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to device
    #     outputs_student = student_model(**inputs)


    #     # get cross entropy loss and logits from student
    #     loss_cross_entropy = outputs_student.loss
    #     logits_student = outputs_student.logits
    #     # get teacher logits
    #     with torch.no_grad():
    #         outputs_teacher = self.teacher_model(**inputs)
    #         logits_teacher = outputs_teacher.logits
    #     # compute kl diveregnce loss
    #     loss_kd = self.args.temperature ** 2 * nn.KLDivLoss(reduction='batchmean')(
    #         F.log_softmax(logits_student / self.args.temperature, dim=-1),
    #         F.softmax(logits_teacher / self.args.temperature, dim=-1)
    #     )
    #     # compute final student loss
    #     loss = self.args.alpha * loss_cross_entropy + (1. - self.args.alpha) * loss_kd
    #     return (loss, outputs_student) if return_outputs else loss

    

        # Tuesday Mars 12th
    def compute_loss(self, student_model, inputs, return_outputs=False):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print("Device initialized:", device) 
     
        inputs_student = {key: value.to(student_model.device) for key, value in inputs.items()}  # Move tensors to device item by item to free from dict error 
        outputs_student = student_model(**inputs_student)

        inputs_teacher = {key: value.to(self.teacher_model.device) for key, value in inputs.items()}  # Move tensors to device item by item to free from dict error 

        # print("post outputs_student :", device) 
        # get cross entropy loss and logits from student
        loss_cross_entropy = outputs_student.loss
        logits_student = outputs_student.logits

        with torch.no_grad():
            print("entering the loop printing to check device :", device) 
            outputs_teacher = self.teacher_model(**inputs_teacher)
            logits_teacher = outputs_teacher.logits


        # print("exiting the loop printing to check device  :", device) 


        # compute kl divergence loss
        loss_kd = self.args.temperature ** 2 * nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(logits_student / self.args.temperature, dim=-1),
            F.softmax(logits_teacher / self.args.temperature, dim=-1)
        )
        # compute final student loss
        loss = self.args.alpha * loss_cross_entropy + (1. - self.args.alpha) * loss_kd
        return (loss, outputs_student) if return_outputs else loss



    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()


            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v.cpu().clone().detach() # Chunyuan: to solve the saving OOM problem


            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))


        super(LLaVATrainer, self)._save(output_dir, state_dict)


