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
 
   def compute_loss(self, student_model, inputs, return_outputs=False):
        #PRINTING FOR DEBUGGING PURPOSES
        print("inputs: ",inputs)
        print("student model device: ",student_model.device)
 
        print("teacher model device: ",self.teacher_model.device)

        #student_inputs: moving the inputs to same device as student model
        student_inputs = {key: value.to(student_model.device) for key, value in inputs.items()}  # Move tensors to device
        # Pass the inputs to the student model
        outputs_student = student_model(**student_inputs)

        # print("post outputs_student :", device)
        # get cross entropy loss and logits from student
        loss_cross_entropy = outputs_student.loss
        logits_student = outputs_student.logits
        print("Done getting the student output")
        
        with torch.no_grad():
            #teacher_inputs: moving the inputs to same device as teacher model
            # teacher_inputs = {key: value.to(self.teacher_model.device) for key, value in inputs.items()}
            outputs_teacher = self.teacher_model(**student_inputs)
            logits_teacher = outputs_teacher.logits

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







