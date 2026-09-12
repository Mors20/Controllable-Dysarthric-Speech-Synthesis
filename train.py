import copy
import argparse
import gc
import os
import random
import sys
from datetime import datetime
from typing import List, Optional, Tuple
from indextts.utils.typical_sampling import TypicalLogitsWarper
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from indextts.utils.checkpoint import load_checkpoint
from indextts.BigVGAN.models import BigVGAN
from indextts.data_utils_pathology_disentangle_embedding import (
    collate_finetune_fn,
    load_finetune_datasets_pathology,
)
from indextts.gpt.model import UnifiedVoice
from torch import nn
from transformers import LogitsProcessorList
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grl(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)



class MyUnifiedVoice(UnifiedVoice):
    def get_conditioning(self,  speech_conditioning_input, cond_mel_lengths=None,pathology_ids=None,speaker_id=None):
        if speaker_id is None:
            spk_conditioning_input, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
            spk_conds_mask = self.cond_mask_pad(mask.squeeze(1))
            spk_conds = self.perceiver_encoder(spk_conditioning_input, spk_conds_mask)  # (b, 32, d)
        else:
            param_name = f'speaker_{speaker_id}_center'
            spk_conds = getattr(self, param_name)

        device = speech_conditioning_input.device if speech_conditioning_input is not None else next(self.parameters()).device
        # Select the pathology prefix for each sample in the batch.
        conds_list = []
        if pathology_ids is None:
            raise ValueError("pathology_ids is required for pathology conditioning")
        if not torch.is_tensor(pathology_ids) and not isinstance(pathology_ids, (list, tuple)):
            pathology_ids = [pathology_ids]
        for pathology_id in pathology_ids:
            if torch.is_tensor(pathology_id):
                pathology_id = int(pathology_id.item())
            param_name = f'mean_pathology_condition_{pathology_id}'
            if not hasattr(self, param_name):
                raise ValueError(f"Unknown pathology condition {pathology_id}; missing {param_name}")
            pathology_condition = getattr(self, param_name)
            if pathology_condition.device != device:
                pathology_condition = pathology_condition.to(device)
            # Normalize each prefix to shape (1, 32, model_dim).
            if pathology_condition.ndim == 2:
                pathology_condition = pathology_condition.unsqueeze(0)
            elif pathology_condition.ndim == 4:
                pathology_condition = pathology_condition.squeeze(0)
            conds_list.append(pathology_condition)


        # Assemble the selected prefixes into (batch_size, 32, model_dim).
        pathology_conds = torch.cat(conds_list, dim=0)

        conds = spk_conds + pathology_conds
        return conds , spk_conds

    def inference_speech(self, speech_conditioning_mel, text_inputs, cond_mel_lengths=None, input_tokens=None, num_return_sequences=1,
                         max_generate_length=None, typical_sampling=False, typical_mass=.9, speaker_id=None,pathology_ids=None,**hf_generate_kwargs):
        """
        Args:
            speech_conditioning_mel: (b, n_mels, frames) or (n_mels, frames)
            text_inputs: (b, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
            input_tokens: additional tokens for generation in shape (b, s) or (s,)
            max_generate_length: limit the number of generated tokens
            hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`
        """
        if speech_conditioning_mel.ndim == 2: #'mean_condition_F01_indextts'
            speech_conditioning_mel = speech_conditioning_mel.unsqueeze(0)
        if cond_mel_lengths is None:
            cond_mel_lengths = torch.tensor([speech_conditioning_mel.shape[-1]], device=speech_conditioning_mel.device)

        conds_latent, spk_conds = self.get_conditioning( speech_conditioning_input=speech_conditioning_mel,cond_mel_lengths=cond_mel_lengths, pathology_ids=pathology_ids,speaker_id=speaker_id)
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        self.inference_model.store_mel_emb(inputs_embeds)
        if input_tokens is None:
            inputs = input_ids
        else:
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)
            assert num_return_sequences % input_tokens.shape[0] == 0, \
                    "The num_return_sequences must be divisible by the batch number of input_tokens"
            assert num_return_sequences % text_inputs.shape[0] == 0, \
                    "The num_return_sequences must be divisible by the batch number of text_inputs"
            b = num_return_sequences // input_ids.shape[0]
            if b > 1:
                input_ids = input_ids.repeat(b, 1)
                attention_mask = attention_mask.repeat(b, 1)
            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([input_ids, input_tokens], dim=1)
            attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)
        trunc_index = inputs.shape[1]
        logits_processor = LogitsProcessorList()
        if typical_sampling:
            # employ custom typical sampling
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}")
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))
        max_length = (trunc_index + self.max_mel_tokens - 1) if max_generate_length is None else trunc_index + max_generate_length
        output = self.inference_model.generate(inputs,
                                            bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token,
                                            eos_token_id=self.stop_mel_token, attention_mask=attention_mask,
                                            max_length=max_length, logits_processor=logits_processor,
                                            num_return_sequences=num_return_sequences,
                                            **hf_generate_kwargs)
        if isinstance(output, torch.Tensor):
            return output[:, trunc_index:]
        # GenerateOutput
        output.sequences = output.sequences[:, trunc_index:]
        return output


    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths,
                cond_mel_lengths=None, types=None, text_first=True, raw_mels=None, return_attentions=False,
                return_latent=False, clip_inputs=False,pathology_ids=None,speaker_id=None):

        speech_conditioning_latent,spk_conds = self.get_conditioning( speech_conditioning_input=speech_conditioning_latent,cond_mel_lengths=cond_mel_lengths, pathology_ids=pathology_ids,speaker_id=speaker_id)

        # Types are expressed by expanding the text embedding space.
        if types is not None:
            text_inputs = text_inputs * (1 + types).unsqueeze(-1)

        if clip_inputs:
            # This model will receive micro-batches with a ton of padding for both the text and MELs. Ameliorate this by
            # chopping the inputs by the maximum actual length.
            max_text_len = text_lengths.max()
            text_inputs = text_inputs[:, :max_text_len]
            max_mel_len = wav_lengths.max() // self.mel_length_compression
            mel_codes = mel_codes[:, :max_mel_len]
            if raw_mels is not None:
                raw_mels = raw_mels[:, :, :max_mel_len * 4]

        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        mel_codes_lengths = torch.ceil(wav_lengths / self.mel_length_compression).long() + 1
        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        conds = speech_conditioning_latent
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
        if raw_mels is not None:
            mel_inp = F.pad(raw_mels, (0, 8))
        else:
            mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        if text_first:
            text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=return_attentions, return_latent=return_latent)
            if return_latent:
                return mel_logits[:, :-2]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.
        else:
            mel_logits, text_logits = self.get_logits(conds, mel_emb, self.mel_head, text_emb, self.text_head, get_attns=return_attentions, return_latent=return_latent)
            if return_latent:
                return text_logits[:, :-2]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

        if return_attentions:
            return mel_logits

        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits

def load_UnifiedVoice(gpt_config: DictConfig, gpt_checkpoint_path: str, device: torch.device) -> MyUnifiedVoice:
    """Load UnifiedVoice weights from a checkpoint."""

    state_dict = torch.load(gpt_checkpoint_path, map_location=device, weights_only=True)
    state_dict = state_dict["model"] if "model" in state_dict else state_dict
    model = MyUnifiedVoice(**gpt_config)
    ms, up = model.load_state_dict(state_dict, strict=False)
    model.post_init_gpt2_config()
    del state_dict
    return model.to(device)


def clear_torch_cache():
    """Release cached CUDA memory."""
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def forward_gpt2(
    model: MyUnifiedVoice,
    inputs_embeds: torch.FloatTensor,
    text_lengths: torch.LongTensor,
    codes_lengths: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_latent: bool = False,
    output_logits: bool = True,
):
    assert attention_mask is not None, "Attention mask must be provided for UnifiedVoice forward pass."

    """Run the GPT-2 portion of the UnifiedVoice forward pass."""
    b = inputs_embeds.shape[0]
    gpt_out = model.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
    hidden_state = gpt_out.last_hidden_state

    # Vectorized implementation to replace the for loop
    conditioning_len = 32

    # Remove conditioning part from hidden states and attention mask
    h_no_cond = hidden_state[:, conditioning_len:]  # [b, seq_len, hidden_dim]
    attention_no_cond = attention_mask[:, conditioning_len:]  # [b, seq_len]

    # Apply final_norm to all samples at once
    latent = model.final_norm(h_no_cond)  # [b, seq_len, hidden_dim]

    # Get max lengths for efficient processing
    max_text_len = text_lengths.max().item()
    max_mel_len = codes_lengths.max().item()

    # Create batched tensors for text and mel latents
    batch_text_latents = torch.zeros(b, max_text_len, latent.shape[-1], device=latent.device, dtype=latent.dtype)
    batch_mel_latents = torch.zeros(b, max_mel_len, latent.shape[-1], device=latent.device, dtype=latent.dtype)

    # Fill the batched tensors
    for i in range(b):
        text_len = text_lengths[i].item()
        mel_len = codes_lengths[i].item()

        # Extract valid latent for this sample
        sample_valid_mask = attention_no_cond[i] == 1
        sample_latent = latent[i][sample_valid_mask]  # [valid_len, hidden_dim]

        # Verify the expected length
        expected_len = text_len + mel_len
        assert sample_latent.shape[0] == expected_len, \
            f"Expected valid_latent shape {expected_len}, got {sample_latent.shape[0]}, " \
            f"text_len: {text_len}, mel_len: {mel_len}"

        # Split and assign to batched tensors
        batch_text_latents[i, :text_len] = sample_latent[:text_len]
        batch_mel_latents[i, :mel_len] = sample_latent[text_len:text_len + mel_len]

    # Vectorized head processing
    # Process all text latents at once
    batch_text_logits = model.text_head(batch_text_latents)  # [b, max_text_len, vocab_size]
    batch_text_logits = batch_text_logits.permute(0, 2, 1)  # [b, vocab_size, max_text_len]

    # Process all mel latents at once
    batch_mel_logits = model.mel_head(batch_mel_latents)  # [b, max_mel_len, vocab_size]
    batch_mel_logits = batch_mel_logits.permute(0, 2, 1)  # [b, vocab_size, max_mel_len]

    # Return the processed batches directly.
    # The tensors are already padded to the max length in the batch.
    output = {}
    if output_logits:
        output["logits"] = (batch_text_logits, batch_mel_logits)
    if output_latent:
        output["latent"] = (batch_text_latents, batch_mel_latents)
    return output

def forward_UnifiedVoice(
    model: MyUnifiedVoice,
    mel_spec: torch.FloatTensor,
    mel_codes: torch.LongTensor,
    text_ids: torch.LongTensor,
    mel_lengths: torch.LongTensor,
    codes_lengths: torch.LongTensor,
    text_lengths: torch.LongTensor,
    mel_same_spk: torch.FloatTensor,
    spk_mel_lengths: torch.LongTensor,
    add_mel_stop_token: bool = True,
    output_loss: bool = True,
    output_logits: bool = True,
    output_latent: bool = False,
    loss_reduction: str = "mean",
    pathology_ids: torch.LongTensor = None,
    speaker_id: torch.LongTensor = None
):
    """Run the complete UnifiedVoice training forward pass."""

    conditioning_latent, spk_conds = model.get_conditioning( speech_conditioning_input=mel_same_spk,cond_mel_lengths=spk_mel_lengths, pathology_ids=pathology_ids,speaker_id=speaker_id)

    # -------- build text_inputs with start/stop tokens --------
    B, T_pad = text_ids.shape
    max_out_text = T_pad + 2  # +<start> +<stop>
    text_inputs = text_ids.new_zeros((B, max_out_text))
    for i, L in enumerate(text_lengths):
        L = L.item()
        text_inputs[i, 0] = model.start_text_token
        text_inputs[i, 1 : L + 1] = text_ids[i, :L]
        text_inputs[i, L + 1] = model.stop_text_token
    text_targets = text_inputs[:, 1:].clone().contiguous()

    # -------- build mel_inputs with start/stop tokens --------
    B, M_pad = mel_codes.shape
    extra_stop = 1 if add_mel_stop_token else 0
    max_out_mel = M_pad + 1 + extra_stop  # +<start> (+<stop>)
    mel_inputs = mel_codes.new_zeros((B, max_out_mel))
    for i, L in enumerate(codes_lengths):
        L = L.item()
        mel_inputs[i, 0] = model.start_mel_token
        mel_inputs[i, 1 : L + 1] = mel_codes[i, :L]
        if add_mel_stop_token:
            mel_inputs[i, L + 1] = model.stop_mel_token
    mel_targets = mel_inputs[:, 1:].clone().contiguous()

    # Embeddings
    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    # for later use in loss and lengths
    mel_codes = mel_inputs

    inputs_embeds = torch.cat([conditioning_latent, text_emb, mel_emb], dim=1)

    # Create attention mask for the combined sequence
    batch_size, total_seq_len = inputs_embeds.shape[:2]
    attention_mask = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=inputs_embeds.device)

    # Calculate actual sequence lengths for each sample
    conditioning_len = conditioning_latent.shape[1]
    actual_text_lengths = text_lengths + 2  # +2 for start/stop tokens
    actual_mel_lengths = codes_lengths + 1 + int(add_mel_stop_token)  # +1 for start token + optional stop token

    for i in range(batch_size):
        # Set conditioning part (always valid)
        attention_mask[i, :conditioning_len] = 1

        # Set text part (considering actual text length without padding)
        text_start = conditioning_len
        text_end = text_start + actual_text_lengths[i].item()
        attention_mask[i, text_start:text_end] = 1

        # Set mel part (considering actual mel length without padding)
        # mel_start should be based on the actual text_emb length, not text_end
        mel_start = conditioning_len + text_emb.shape[1]
        mel_end = mel_start + actual_mel_lengths[i].item()
        attention_mask[i, mel_start:mel_end] = 1

    gpt2_outputs = forward_gpt2(
        model,
        inputs_embeds,
        text_lengths + 2,
        codes_lengths + 1 + int(add_mel_stop_token),
        attention_mask=attention_mask,
        output_latent=output_latent,
        output_logits=output_logits or output_loss,
    )

    outputs = {}
    if output_logits or output_loss:
        text_logits, mel_logits = gpt2_outputs["logits"]
        text_logits = text_logits[:, :, :-1].contiguous()
        mel_logits = mel_logits[:, :, :-1].contiguous()
        if output_loss:
            # Create masks based on actual sequence lengths
            batch_size = text_targets.size(0)

            # Text mask: consider actual text length + 1 (stop token)
            text_mask = torch.zeros_like(text_targets, dtype=torch.bool)
            for i in range(batch_size):
                actual_text_len = text_lengths[i].item() + 1  # +1 for stop token
                text_mask[i, :actual_text_len] = True

            # Mel mask: consider actual codes length + stop token (if any)
            mel_mask = torch.zeros_like(mel_targets, dtype=torch.bool)
            for i in range(batch_size):
                actual_mel_len = codes_lengths[i].item() + int(add_mel_stop_token)
                mel_mask[i, :actual_mel_len] = True

            loss_text = F.cross_entropy(text_logits, text_targets.long(), reduction='none')
            loss_mel = F.cross_entropy(mel_logits, mel_targets.long(), reduction='none')

            # Apply masking and reduce - only calculate loss on valid positions
            loss_text = (loss_text * text_mask).sum() / text_mask.sum() if text_mask.sum() > 0 else torch.tensor(0.0, device=text_logits.device)
            loss_mel = (loss_mel * mel_mask).sum() / mel_mask.sum() if mel_mask.sum() > 0 else torch.tensor(0.0, device=mel_logits.device)

            outputs["loss"] = (loss_text, loss_mel)

            # Calculate mel prediction accuracy
            with torch.no_grad():
                # Apply mask to get valid positions for accuracy calculation
                mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
                mel_targets_flat = mel_targets.view(-1)
                mel_mask_flat = mel_mask.view(-1)

                # Only calculate accuracy on valid positions
                if mel_mask_flat.sum() > 0:
                    valid_mel_logits = mel_logits_flat[mel_mask_flat]
                    valid_mel_targets = mel_targets_flat[mel_mask_flat]
                    mel_acc_1, mel_acc_10, mel_acc_20 = top_k_accuracy(valid_mel_logits, valid_mel_targets, k=(1, 10, 20))
                    outputs["mel_accuracy"] = {"acc_1": mel_acc_1, "acc_10": mel_acc_10, "acc_20": mel_acc_20}
                else:
                    outputs["mel_accuracy"] = {"acc_1": 0.0, "acc_10": 0.0, "acc_20": 0.0}

        if output_logits:
            outputs["logits"] = (text_logits, mel_logits)
            outputs["targets"] = (text_targets, mel_targets)


    if output_latent:
        outputs["latent"] = gpt2_outputs["latent"]
    outputs["embedding_sum"] = conditioning_latent
    outputs['embedding_spk'] = spk_conds
    return outputs

def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: Tuple[int, ...] = (1, 10, 20)) -> List[float]:
    """Compute top-k classification accuracy."""
    max_k = max(k)
    _, topk_preds = torch.topk(logits, max_k, dim=1)  # (B*L, max_k)

    # Reshape for comparison
    targets_reshaped = targets.view(-1, 1) # (B*L, 1)
    topk_preds_reshaped = topk_preds.view(-1, max_k) # (B*L, max_k)

    res = []
    for ki in k:
        # Check if the target is in the top-ki predictions
        correct_k = (topk_preds_reshaped[:, :ki] == targets_reshaped).any(dim=-1)
        acc = correct_k.float().mean().item() * 100
        res.append(acc)
    return res

class Trainer:
    """Train UnifiedVoice with timbre and pathology conditioning."""
    def __init__(self, config: DictConfig, save_checkpoints: bool = True):
        """
        Initialize the trainer.

        Args:
            config: OmegaConf configuration loaded from YAML.
        """
        self.config = config
        self.save_checkpoints = save_checkpoints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Seed all random number generators.
        self._set_seed(self.config.train.seed)

        # Prepare output directories and logging.
        self.finetune_dir = self.config.train.finetune_model_dir
        self.checkpoint_dir = os.path.join(self.finetune_dir, "checkpoints_dys_spk_grl_exp1")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._setup_logging()

        # Load models and the tokenizer.
        self._load_models()

        # Initialize training state.
        self.best_val_loss = (0, float('inf'), float('inf'))  # (epoch, text_loss, mel_loss)
        self.update_steps = 0

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")

    def _setup_logging(self):
        """Configure loguru output."""
        log_path = os.path.join(self.checkpoint_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        logger.add(log_path, level="INFO", encoding="utf-8")
        logger.info("Logging configured. Logs will be saved to console and file.")
        logger.info("Full configuration:\n" + OmegaConf.to_yaml(self.config))

    def _load_models(self):
        """Load the BPE tokenizer and UnifiedVoice model."""
        logger.info("Loading models...")
        # BPE
        bpe_model_path = os.path.join(self.finetune_dir, self.config.dataset.bpe_model)
        self.bpe_model = spm.SentencePieceProcessor(bpe_model_path)
        logger.info("BPE model loaded.")

        # UnifiedVoice

        gpt_checkpoint_path = os.path.join(self.finetune_dir, self.config.gpt_checkpoint)
        self.model = load_UnifiedVoice(self.config.gpt, gpt_checkpoint_path, self.device)
        logger.info("UnifiedVoice base model loaded.")

        # Apply LoRA and freeze the base parameters.
        self.model = self._apply_lora(self.model)
        logger.info("LoRA applied to the model.")

        for p in self.model.perceiver_encoder.parameters():
            p.requires_grad = True
        if hasattr(self.config, "pathology") :
            self._load_pathology_embedding()




    def _load_pathology_embedding(self):
        self.pathology_mean_conditions = {}

        num_classes = int(getattr(self.config.pathology, "num_classes", 2))
        cond_num = int(getattr(self.config.gpt, "condition_num_latent", 32))
        model_dim = int(getattr(self.config.gpt, "model_dim", 512))

        emb_dir = getattr(self.config.pathology, "embedding_dir", None)
        if emb_dir is None:
            raise ValueError("config.pathology.embedding_dir must be set to load pretrained pathology embeddings")



        for pathology_label in range(num_classes):
            param_name = f"mean_pathology_condition_{pathology_label}"

            # ---------- 1. load saved embedding ----------
            # Prefer the canonical filename and accept the legacy filename.

            emb_path = os.path.join(emb_dir, f"mean_pathology_condition_{pathology_label}.npy")
            if not os.path.exists(emb_path):
                legacy_path = os.path.join(emb_dir, f"mean_Sevcondition_{pathology_label}.npy")
                if os.path.exists(legacy_path):
                    emb_path = legacy_path


            emb = np.load(emb_path)

            # ---------- 2. normalize shape ----------
            # Accept either (32, D) or (1, 32, D).
            if emb.ndim == 2:
                emb = emb[None, ...]   # (1, 32, D)
            elif emb.ndim != 3:
                raise ValueError(f"{emb_path} has invalid shape {emb.shape}")

            if emb.shape[1] != cond_num or emb.shape[2] != model_dim:
                raise ValueError(
                    f"{emb_path} shape mismatch: expected (1,{cond_num},{model_dim}), "
                    f"got {emb.shape}"
                )

            condition = torch.from_numpy(emb).float().to(self.device)
            logger.info(f"[Pathology {pathology_label}] loaded embedding from {emb_path}")

            # ---------- 3. register parameter ----------
            param = torch.nn.Parameter(condition, requires_grad=True)

            if hasattr(self.model, param_name):
                logger.warning(f"Parameter {param_name} already exists; overwriting.")
                try:
                    delattr(self.model, param_name)
                except Exception:
                    pass

            self.model.register_parameter(param_name, param)
            self.pathology_mean_conditions[pathology_label] = param

            logger.debug(f"Registered {param_name} with shape {tuple(param.shape)}")

        # ---------- 4. pathology classifier ----------
        self.model.pathology_classifier_sum = nn.Linear(model_dim, 2).to(self.device)
        for p in self.model.pathology_classifier_sum.parameters():
            p.requires_grad = True
        self.model.pathology_classifier_spk = nn.Linear(model_dim, 2).to(self.device)
        for p in self.model.pathology_classifier_spk.parameters():
            p.requires_grad = True
        logger.info(
            f"Loaded and registered {len(self.pathology_mean_conditions)} pretrained pathology embeddings."
        )


    def _apply_lora(self, model: MyUnifiedVoice) -> MyUnifiedVoice:
        """Configure and apply LoRA to the model."""
        lora_cfg = self.config.train.lora
        gpt_lora_config = LoraConfig(
            r=lora_cfg.r,
            target_modules=lora_cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            bias="none",
        )
        model.requires_grad_(False)
        model.inference_model = get_peft_model(model.inference_model, gpt_lora_config)
        return model
    def _move_params_to_new_group(self, optimizer, names_to_move, new_lr, new_wd):
        """
        Move matching parameters from their existing optimizer group into a new group.
        A parameter matches when its name contains any string in ``names_to_move``.
        """
        id2name = {id(p): n for n, p in self.model.named_parameters()}

        moved = []
        for g in optimizer.param_groups:
            keep = []
            for p in g["params"]:
                n = id2name.get(id(p), "")
                if any(k in n for k in names_to_move):
                    moved.append(p)
                else:
                    keep.append(p)
            g["params"] = keep

        # Remove optimizer groups left empty by the move.
        optimizer.param_groups = [g for g in optimizer.param_groups if len(g["params"]) > 0]

        if len(moved) > 0:
            optimizer.add_param_group({"params": moved, "lr": new_lr, "weight_decay": new_wd})

        return len(moved)

    def _setup_optimizer_and_scheduler(self, num_training_steps: int = 1000):
        opt_cfg = self.config.train.optimizer
        base_lr = opt_cfg.learning_rate

        self.optimizer = create_loraplus_optimizer(
            model=self.model,
            optimizer_cls=AdamW,
            lr=base_lr,
            loraplus_lr_ratio=opt_cfg.loraplus_lr_ratio,
            loraplus_weight_decay=opt_cfg.weight_decay,
        )

        # Give pathology prefixes, classifiers, and the Perceiver dedicated rates.
        pathology_lr = base_lr * 50
        cls_lr = base_lr * 5
        perc_lr = base_lr * 5

        n_pathology = self._move_params_to_new_group(
            self.optimizer,
            names_to_move=["mean_pathology_condition_"],
            new_lr=pathology_lr,
            new_wd=0.0
        )
        n_cls = self._move_params_to_new_group(
            self.optimizer,
            names_to_move=["pathology_classifier"],
            new_lr=cls_lr,
            new_wd=0.0
        )
        n_perc = self._move_params_to_new_group(
            self.optimizer,
            names_to_move=["perceiver_encoder"],
            new_lr=perc_lr,
            new_wd=opt_cfg.weight_decay
        )





        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * opt_cfg.warmup_ratio),
            num_training_steps=num_training_steps,
        )


        logger.info(
            f"Optimizer regrouped: moved pathology={n_pathology} (lr={pathology_lr}, wd=0), "
            f"cls={n_cls} (lr={cls_lr}, wd=0), "
            f"perceiver={n_perc} (lr={perc_lr}, wd={opt_cfg.weight_decay})"
        )
    def _train_step(self, data_batch: tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one training forward pass and compute its loss terms."""
        self.model.train()
        self.model.inference_model.kv_cache = False  # KV caching is only used for autoregressive inference.

        # Unpack data_batch: mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths
        mel_spec, mel_codes, text_ids, conditions, speaker_ids, pathology_ids, mel_same_spk,mel_lengths, codes_lengths, text_lengths,spk_mel_lengths = data_batch
        outputs = forward_UnifiedVoice(
            self.model,
            mel_spec,
            mel_codes,
            text_ids,
            mel_lengths,
            codes_lengths,
            text_lengths,
            mel_same_spk=mel_same_spk,
            output_loss=True,
            output_logits=True,
            spk_mel_lengths=spk_mel_lengths,
            pathology_ids=pathology_ids
        )
        loss_text, loss_mel = outputs["loss"]
        mel_accuracy = outputs.get("mel_accuracy", {"acc_1": 0.0, "acc_10": 0.0, "acc_20": 0.0})

        # ---------- classification loss ----------
        sum_cls_loss = torch.tensor(0.0, device=loss_text.device)
        spk_cls_loss = torch.tensor(0.0, device=loss_text.device)
        if "embedding_sum" in outputs and 'embedding_spk' in outputs:
            grl_lambda_max = float(getattr(self.config.pathology, "grl_lambda_max", 0.01))
            grl_warmup_steps = int(getattr(self.config.pathology, "grl_warmup_steps", 500))

            progress = min(1.0, self.global_step / grl_warmup_steps)
            grl_lambda = grl_lambda_max * progress



            embedding_sum = outputs["embedding_sum"]       # [B, 32, D]
            embedding_spk = outputs["embedding_spk"]        # [B, 32, D]
            # Mean-pool conditioning tokens before classification.
            embedding_sum = embedding_sum.mean(dim=1)            # [B, D]
            embedding_spk = embedding_spk.mean(dim=1)            # [B, D]
            logits_sum = self.model.pathology_classifier_sum(embedding_sum)  # [B, num_classes]
            logits_spk = self.model.pathology_classifier_spk(grl(embedding_spk, grl_lambda))    # [B, num_classes]
            binary_labels = (pathology_ids > 0).long().to(logits_sum.device)
            sum_cls_loss = F.cross_entropy(logits_sum, binary_labels)
            spk_cls_loss = F.cross_entropy(logits_spk, binary_labels)


        return loss_text, loss_mel, sum_cls_loss, spk_cls_loss,mel_accuracy




    @torch.no_grad()
    def _validate_epoch(self, valid_ds: Dataset, epoch: int):
        """Evaluate the model on the validation set."""
        self.model.eval()
        logger.info(f"Validating at epoch {epoch + 1}...")

        total_text_loss, total_mel_loss = 0.0, 0.0
        total_text_tokens, total_mel_tokens = 0, 0
        all_mel_logits, all_mel_targets = [], []
        num_batches = 0

        for batch in tqdm(valid_ds, desc="Validation", dynamic_ncols=True):
            # Move tensors to the training device while retaining string speaker IDs.
            data_batch = []
            for item in batch:
                if torch.is_tensor(item):
                    data_batch.append(item.to(self.device))
                else:
                    data_batch.append(item)

            # Unpack the collated multi-speaker batch.
            mel_spec, mel_codes, text_ids, conditions, speaker_ids, pathology_ids, mel_same_spk,mel_lengths, codes_lengths, text_lengths,spk_mel_lengths = data_batch

            outputs = forward_UnifiedVoice(
                self.model,
                mel_spec,
                mel_codes,
                text_ids,
                mel_lengths,
                codes_lengths,
                text_lengths,
                mel_same_spk=mel_same_spk,
                output_loss=True,
                output_logits=True,
                spk_mel_lengths=spk_mel_lengths,
                pathology_ids=pathology_ids
            )

            loss_text, loss_mel = outputs["loss"]
            batch_text_tokens = text_lengths.sum().item()
            batch_mel_tokens = (codes_lengths + 1).sum().item()  # +1 for stop token

            total_text_loss += loss_text.item() * batch_text_tokens
            total_mel_loss += loss_mel.item() * batch_mel_tokens
            total_text_tokens += batch_text_tokens
            total_mel_tokens += batch_mel_tokens
            num_batches += 1

            # Collect logits and targets for accuracy calculation
            # Accuracy is computed over mel-code predictions.
            current_mel_logits = outputs["logits"][1]  # mel logits [B, V, L]
            current_mel_targets = outputs["targets"][1]  # mel targets [B, L]
            if current_mel_logits.numel() > 0 and current_mel_targets.numel() > 0:
                # Create mask based on actual sequence lengths instead of assuming 0 is padding
                batch_size = current_mel_targets.size(0)
                mel_mask = torch.zeros_like(current_mel_targets, dtype=torch.bool)
                for i in range(batch_size):
                    actual_mel_len = codes_lengths[i].item() + 1  # +1 for stop token if add_mel_stop_token is True
                    mel_mask[i, :actual_mel_len] = True

                valid_mask = mel_mask.view(-1)
                if valid_mask.sum() > 0:
                    mel_logits_filtered = current_mel_logits.permute(0, 2, 1).reshape(-1, current_mel_logits.size(1))[valid_mask]
                    mel_targets_filtered = current_mel_targets.view(-1)[valid_mask]
                    all_mel_logits.append(mel_logits_filtered)
                    all_mel_targets.append(mel_targets_filtered)

            clear_torch_cache()

        avg_text_loss = total_text_loss / total_text_tokens
        avg_mel_loss = total_mel_loss / total_mel_tokens

        # Aggregate accuracy across validation batches.
        all_mel_logits = torch.cat(all_mel_logits, dim=0)
        all_mel_targets = torch.cat(all_mel_targets, dim=0)
        acc_1, acc_10, acc_20 = top_k_accuracy(all_mel_logits, all_mel_targets, k=(1, 10, 20))

        logger.info(f"**Validation results at epoch {epoch + 1}**")
        logger.info(f"Text Loss: {avg_text_loss:.4f}, Mel Loss: {avg_mel_loss:.4f}")
        logger.info(f"Accuracy@1: {acc_1:.2f}%, Accuracy@10: {acc_10:.2f}%, Accuracy@20: {acc_20:.2f}%")

        return avg_text_loss, avg_mel_loss, acc_1, acc_10, acc_20

    def _save_checkpoint(self, file_name: str, merge_lora: bool, unload_after_merge: bool):
        """Save a model checkpoint, optionally with merged LoRA weights."""
        checkpoint_path = os.path.join(self.checkpoint_dir, file_name)

        self.model.eval()

        model_to_save = self.model

        if merge_lora:
            logger.info("Merging LoRA weights into the model for saving...")
            if unload_after_merge:
                # Merge into a copy so training can continue with the original adapters.
                logger.info("Creating a deep copy of the model for a clean merge. This may take a moment...")
                model_to_save = copy.deepcopy(self.model)

                # Merge and unload adapters in the checkpoint copy.
                fused_inference_model = model_to_save.inference_model.merge_and_unload()
                model_to_save.inference_model = fused_inference_model
                logger.info("LoRA weights merged and unloaded in the copied model.")
            else:
                # Merge adapters in place when the caller will unmerge them after saving.
                self.model.inference_model.merge_adapter()

        state_dict = model_to_save.state_dict()
        checkpoint_data = {'model': state_dict}

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        # Release the checkpoint copy after serialization.
        if merge_lora and unload_after_merge:
            del model_to_save
            clear_torch_cache()
            logger.info("Cleaned up the temporary merged model.")

        # Restore adapters after an in-place merge.
        if merge_lora and not unload_after_merge:
            logger.info("Unmerging LoRA weights to continue training...")
            self.model.inference_model.unmerge_adapter()

        self.model.train()
    def print_trainable_parameters(self):
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name}: shape={tuple(param.shape)}")
    def train(self, train_ds: Dataset, valid_ds: Dataset, max_train_batches: Optional[int] = None,
              skip_validation: bool = False):
        """
        Run the complete training loop.

        Args:
            train_ds: Training dataset.
            valid_ds: Validation dataset.
        """

        train_cfg = self.config.train
        batches_per_epoch = len(train_ds)
        if max_train_batches is not None:
            batches_per_epoch = min(batches_per_epoch, max_train_batches)
        total_update_steps = max(1, batches_per_epoch * train_cfg.epochs)
        self._setup_optimizer_and_scheduler(num_training_steps=total_update_steps)

        logger.info(f"Starting training for {train_cfg.epochs} epochs.")
        logger.info(f"Optimizer batches per epoch: {batches_per_epoch}")
        logger.info(f"Total update steps: {total_update_steps}")




        text_weight = train_cfg.text_weight
        self.global_step = 0
        for epoch in range(train_cfg.epochs):
            logger.info(f"EPOCH {epoch + 1}/{train_cfg.epochs} started" + "=" * 30)
            # val_text_loss, val_mel_loss, _, _, _ = self._validate_epoch(valid_ds, epoch)
            for batch_idx, batch in enumerate(train_ds):
                if max_train_batches is not None and batch_idx >= max_train_batches:
                    break
                # Move tensors to the training device while retaining string speaker IDs.
                data_batch = []
                for item in batch:
                    if torch.is_tensor(item):
                        data_batch.append(item.to(self.device))
                    else:
                        data_batch.append(item)

                loss_text, loss_mel, sum_cls_loss, spk_cls_loss,mel_accuracy  = self._train_step(tuple(data_batch))
                acc_1, acc_10, acc_20 = mel_accuracy["acc_1"], mel_accuracy["acc_10"], mel_accuracy["acc_20"]

                alpha = float(getattr(self.config.pathology, "alpha", 1.0))
                beta = float(getattr(self.config.pathology, "beta", 1.0))
                weighted_loss = (
                    text_weight * loss_text
                    + (1.0 - text_weight) * loss_mel
                    + alpha * sum_cls_loss
                    + beta * spk_cls_loss
                )

                if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                    logger.warning(f"NaN or Inf loss at epoch {epoch}, batch {batch_idx}. Skipping.")
                    continue

                # ------------------ Optimisation Step ------------------
                self.optimizer.zero_grad()
                weighted_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.update_steps += 1



                logger.info(
                    f"Epoch {epoch + 1}/{train_cfg.epochs} | Batch {batch_idx + 1}/{len(train_ds)} | "
                    f"text_loss={loss_text.item():.4f}, mel_loss={loss_mel.item():.4f}, "
                    f"sum_cls_loss={sum_cls_loss.item():.4f}, "
                    f"spk_cls_loss={spk_cls_loss.item():.4f}, "
                    f"acc@1={acc_1:.2f}%, acc@10={acc_10:.2f}%, acc@20={acc_20:.2f}%, "
                    f"grad_norm={grad_norm.item():.2f}"
                )
                self.global_step += 1


            # --- Epoch End ---
            if skip_validation:
                val_text_loss, val_mel_loss = float(loss_text.item()), float(loss_mel.item())
                logger.info("Validation skipped by request.")
            else:
                val_text_loss, val_mel_loss, _, _, _ = self._validate_epoch(valid_ds, epoch)

            # Save a checkpoint after each epoch.
            epoch_checkpoint_name = f"gpt_epoch_{epoch + 1}.pth"
            if self.save_checkpoints:
                logger.info(f"Saving model for epoch {epoch + 1}: {epoch_checkpoint_name}")
                self._save_checkpoint(epoch_checkpoint_name, merge_lora=True, unload_after_merge=True)

            if self.save_checkpoints and val_mel_loss < self.best_val_loss[2]:
                logger.info(f"New best validation mel_loss: {val_mel_loss:.4f}. Saving best model.")
                self.best_val_loss = (epoch, val_text_loss, val_mel_loss)
                self._save_checkpoint("gpt_best.pth", merge_lora=True, unload_after_merge=True)

            clear_torch_cache()

        # --- Training End ---
        logger.info("Training finished.")
        if not self.save_checkpoints:
            return
        self._save_checkpoint("gpt_finetuned.pth", merge_lora=True, unload_after_merge=True)

        # Save the configuration used for the final checkpoint.
        final_config_path = os.path.join(self.finetune_dir, "config_finetuned.yaml")
        final_config = self.config.copy()
        final_config.gpt_checkpoint = "checkpoints/gpt_finetuned.pth"
        OmegaConf.save(final_config, final_config_path)
        logger.info(f"Final config saved to {final_config_path}")

        logger.info(f"Best validation loss at epoch {self.best_val_loss[0] + 1}: "
                    f"text_loss: {self.best_val_loss[1]:.4f}, mel_loss: {self.best_val_loss[2]:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndexTTS with timbre/pathology factorization.")
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs", "controllable_dysarthric_speech_synthesis.yaml"),
    )
    parser.add_argument("--model-dir", help="Directory containing gpt.pth, dvae.pth and bpe.model")
    parser.add_argument("--data-dir", help="Prepared dataset directory containing speaker_info.json")
    parser.add_argument("--embedding-dir", help="Directory containing mean_pathology_condition_*.npy")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Do not write large checkpoints (for smoke tests)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}.")

    config = OmegaConf.load(config_path)
    if args.model_dir:
        config.train.finetune_model_dir = os.path.abspath(args.model_dir)
    if args.data_dir:
        config.train.data_path = os.path.abspath(args.data_dir)
    if args.embedding_dir:
        config.pathology.embedding_dir = os.path.abspath(args.embedding_dir)
    if args.epochs is not None:
        config.train.epochs = args.epochs
    bpe_model_path = os.path.join(config.train.finetune_model_dir, config.dataset.bpe_model)

    # Build multi-speaker datasets with the configured tokenizer model.
    train_ds, valid_ds = load_finetune_datasets_pathology(config, bpe_model_path)
    train_ds = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_finetune_fn, num_workers=args.num_workers)
    valid_ds = DataLoader(valid_ds, batch_size=min(args.batch_size, 8), shuffle=False,
                          collate_fn=collate_finetune_fn, num_workers=min(args.num_workers, 2))

    trainer = Trainer(config, save_checkpoints=not args.no_save)
    trainer.train(train_ds, valid_ds, max_train_batches=args.max_train_batches,
                  skip_validation=args.skip_validation)
    logger.info("Finetuning UnifiedVoice completed.")


if __name__ == "__main__":
    main()
