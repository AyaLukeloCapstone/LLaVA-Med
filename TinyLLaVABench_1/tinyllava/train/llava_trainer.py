import os
import torch
from torch import nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
from typing import List, Optional

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


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(self, model, tokenizer, *args, **kwargs):
        # Initialize the Trainer with only the arguments it expects
        super().__init__(model=model, tokenizer=tokenizer, *args, **kwargs)
        # Then handle the teacher_model separately for the LLaVATrainer
        self.teacher_model = load_teacher_model(model.device)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
                ]

            if getattr(self.args, "moe_enable", False):
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer



    # Compute loss from article https://medium.com/@vmn11/speeding-up-transfomers-knowledge-distillation-f8b67418627f
    def compute_loss(self, student_model, inputs, return_outputs=False):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # inputs = inputs.to(device)
        outputs_student = student_model(**inputs)
        # get cross entropy loss and logits from student
        loss_cross_entropy = outputs_student.loss
        logits_student = outputs_student.logits
        # get teacher logits
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits
        # compute kl diveregnce loss
        loss_kd = self.args.temperature ** 2 * nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(logits_student / self.args.temperature, dim=-1),
            F.softmax(logits_teacher / self.args.temperature, dim=-1)
        )
        # compute final student loss
        loss = self.args.alpha * loss_cross_entropy + (1. - self.args.alpha) * loss_kd
        return (loss, outputs_student) if return_outputs else loss


    def load_teacher_model(device):
            ############### Changing the training code of TinyLlava for including Knowledge distillation ###############

        # initializing the teacher model 

        ##TEACHER MODEL:
        ##create data arguments
        teacher_model_args = ModelArguments(
        model_name_or_path="/scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain",
        version="v0",
        freeze_backbone=True,
        tune_mm_mlp_adapter=True,
        vision_tower="openai/clip-vit-large-patch14",
        mm_vision_select_layer=-2,
        pretrain_mm_mlp_adapter=None,
        mm_use_im_start_end=True)


        teacher_training_args = TrainingArguments(
            output_dir="/scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-teacher",
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
        
            teacher_model.model.vision_tower[0].to(dtype=dtype, device=device)
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
            teacher_model.initialize_vision_tokenizer(mm_use_im_start_end=teacher_model_args.mm_use_im_start_end, tokenizer=teacher_tokenizer, device=device,
                                            tune_mm_mlp_adapter=teacher_model_args.tune_mm_mlp_adapter, pretrain_mm_mlp_adapter=teacher_model_args.pretrain_mm_mlp_adapter)
            return teacher_model


    # # ORIGINAL save function of TinyLLAVA
    # def _save_checkpoint(self, model, trial, metrics=None):
    #     if getattr(self.args, 'tune_mm_mlp_adapter', False):
    #         from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    #         checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    #         run_dir = self._get_output_dir(trial=trial)
    #         output_dir = os.path.join(run_dir, checkpoint_folder)

    #         # Only save Adapter
    #         keys_to_match = ['mm_projector', 'vision_resampler']
    #         if getattr(self.args, "use_im_start_end", False):
    #             keys_to_match.extend(['embed_tokens', 'embed_in'])

    #         weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

    #         if self.args.local_rank == 0 or self.args.local_rank == -1:
    #             self.model.config.save_pretrained(output_dir)
    #             torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
    #     else:
    #         super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    # def _save(self, output_dir: Optional[str] = None, state_dict=None):
    #     if getattr(self.args, 'tune_mm_mlp_adapter', False):
    #         pass
    #     else:
    #         super(LLaVATrainer, self)._save(output_dir, state_dict)



    # llava med save function 
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
                os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        super(LLaVATrainer, self)._save(output_dir, state_dict)
