# -*- coding: utf-8 -*-
"""Pathology-conditioned inference for controllable dysarthric TTS.

Implements inference with the trained pathology prefix table, GRL, and LoRA:
upstream IndexTTS v1 infer pipeline + MyUnifiedVoice additive conditioning
(spk_conds from prompt + mean_pathology_condition_k learned prefix), pathology params
random-registered then restored from the finetuned checkpoint (gpt_best.pth,
LoRA already merged at save time).

Usage:
    from indextts.inference import IndexTTS, pathology_map
    tts = IndexTTS(cfg_path=..., model_dir=..., is_fp16=False, use_cuda_kernel=False)
    tts.infer(audio_prompt="FC02.wav", text="...", output_path="out.wav", pathology_ids=1)
"""
import os
import time
import warnings
from typing import Dict, List

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.typical_sampling import TypicalLogitsWarper
from transformers import LogitsProcessorList

# per-patient condition index (k): 0 = healthy control condition
pathology_map = {
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,
    "F03": 0, "F04": 0, "M03": 0,
    "F01": 1, "M01": 2, "M02": 3, "M04": 4, "M05": 5,
}


def _upgrade_legacy_pathology_keys(state_dict):
    """Translate parameter names stored by legacy checkpoints."""
    replacements = (
        ("mean_Sevcondition_", "mean_pathology_condition_"),
        ("severity_classifier_", "pathology_classifier_"),
    )
    upgraded = dict(state_dict)
    for old_key in list(state_dict):
        new_key = old_key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        if new_key != old_key:
            upgraded.setdefault(new_key, state_dict[old_key])
            upgraded.pop(old_key, None)
    return upgraded


class MyUnifiedVoice(UnifiedVoice):
    """Combine prompt-derived timbre conditioning with a pathology prefix."""

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None, pathology_ids=None, speaker_id=None):
        if speaker_id is None:
            spk_conditioning_input, mask = self.conditioning_encoder(
                speech_conditioning_input.transpose(1, 2), cond_mel_lengths)  # (b, s, d), (b, 1, s)
            spk_conds_mask = self.cond_mask_pad(mask.squeeze(1))
            spk_conds = self.perceiver_encoder(spk_conditioning_input, spk_conds_mask)  # (b, 32, d)
        else:
            param_name = f'speaker_{speaker_id}_center'
            spk_conds = getattr(self, param_name)

        device = speech_conditioning_input.device if speech_conditioning_input is not None else next(self.parameters()).device
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
            if pathology_condition.ndim == 2:
                pathology_condition = pathology_condition.unsqueeze(0)
            elif pathology_condition.ndim == 4:
                pathology_condition = pathology_condition.squeeze(0)
            conds_list.append(pathology_condition)

        pathology_conds = torch.cat(conds_list, dim=0)
        conds = spk_conds + pathology_conds.to(spk_conds.dtype)
        return conds, spk_conds

    def inference_speech(self, speech_conditioning_mel, text_inputs, cond_mel_lengths=None, input_tokens=None,
                         num_return_sequences=1, max_generate_length=None, typical_sampling=False, typical_mass=.9,
                         speaker_id=None, pathology_ids=None, **hf_generate_kwargs):
        if speech_conditioning_mel.ndim == 2:
            speech_conditioning_mel = speech_conditioning_mel.unsqueeze(0)
        if cond_mel_lengths is None:
            cond_mel_lengths = torch.tensor([speech_conditioning_mel.shape[-1]], device=speech_conditioning_mel.device)

        conds_latent, spk_conds = self.get_conditioning(
            speech_conditioning_input=speech_conditioning_mel, cond_mel_lengths=cond_mel_lengths,
            pathology_ids=pathology_ids, speaker_id=speaker_id)
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        self.inference_model.store_mel_emb(inputs_embeds)
        if input_tokens is None:
            inputs = input_ids
        else:
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)
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
        output.sequences = output.sequences[:, trunc_index:]
        return output

    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths,
                cond_mel_lengths=None, types=None, text_first=True, raw_mels=None, return_attentions=False,
                return_latent=False, clip_inputs=False, pathology_ids=None, speaker_id=None):

        speech_conditioning_latent, spk_conds = self.get_conditioning(
            speech_conditioning_input=speech_conditioning_latent, cond_mel_lengths=cond_mel_lengths,
            pathology_ids=pathology_ids, speaker_id=speaker_id)

        if types is not None:
            text_inputs = text_inputs * (1 + types).unsqueeze(-1)

        if clip_inputs:
            max_text_len = text_lengths.max()
            text_inputs = text_inputs[:, :max_text_len]
            max_mel_len = wav_lengths.max() // self.mel_length_compression
            mel_codes = mel_codes[:, :max_mel_len]
            if raw_mels is not None:
                raw_mels = raw_mels[:, :, :max_mel_len * 4]

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
            text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head,
                                                      get_attns=return_attentions, return_latent=return_latent)
            if return_latent:
                return mel_logits[:, :-2]
        else:
            mel_logits, text_logits = self.get_logits(conds, mel_emb, self.mel_head, text_emb, self.text_head,
                                                      get_attns=return_attentions, return_latent=return_latent)
            if return_latent:
                return text_logits[:, :-2]

        if return_attentions:
            return mel_logits

        loss_text = F.cross_entropy(text_logits, text_targets.long())
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits


class IndexTTS:
    def __init__(self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, device=None,
                 use_cuda_kernel=None, gpt_checkpoint=None):
        """
        Args:
            gpt_checkpoint: explicit path to the finetuned gpt_best.pth; overrides cfg.gpt_checkpoint.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.gpt = MyUnifiedVoice(**self.cfg.gpt)
        self._load_pathology_embedding()

        gpt_path = gpt_checkpoint or os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        state_dict = torch.load(gpt_path, map_location="cpu", weights_only=True)
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        state_dict = _upgrade_legacy_pathology_keys(state_dict)
        missing, unexpected = self.gpt.load_state_dict(state_dict, strict=False)
        pathology_loaded = [k for k in state_dict if k.startswith("mean_pathology_condition_")]
        logger.info(f"checkpoint pathology params: {pathology_loaded}")
        if missing:
            logger.warning(f"missing keys (first 8): {missing[:8]}")
        if unexpected:
            logger.warning(f"unexpected keys (first 8): {unexpected[:8]}")
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", gpt_path)
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=self.is_fp16)

        if self.use_cuda_kernel:
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load
                anti_alias_activation_cuda = load.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except Exception:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def _load_pathology_embedding(self):
        """Random-init and register the pathology prefix table; real values come from the checkpoint."""
        self.pathology_mean_conditions = {}
        num_classes = int(getattr(self.cfg.pathology, "num_classes", 2))
        cond_num = int(getattr(self.cfg.gpt, "condition_num_latent", 32))
        model_dim = int(getattr(self.cfg.gpt, "model_dim", 512))
        init_std = float(getattr(self.cfg.pathology, "init_std", 0.02))

        for pathology_label in range(num_classes):
            condition = torch.randn(1, cond_num, model_dim) * init_std
            param_name = f"mean_pathology_condition_{pathology_label}"
            param = torch.nn.Parameter(condition, requires_grad=False)
            if hasattr(self.gpt, param_name):
                try:
                    delattr(self.gpt, param_name)
                except Exception:
                    pass
            self.gpt.register_parameter(param_name, param)
            self.pathology_mean_conditions[pathology_label] = param
            logger.debug(f"Registered parameter {param_name} with shape {tuple(param.shape)}")
        logger.info(f"Random-initialized and registered {num_classes} pathology conditions.")

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        code_lens = []
        codes_list = []
        device = codes.device
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
        except Exception:
            pass

    def infer(self, audio_prompt, text, output_path, pathology_ids=0, verbose=False,
              max_text_tokens_per_segment=120, **generation_kwargs):
        print(">> starting inference...")
        start_time = time.perf_counter()

        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
        cond_mel_frame = cond_mel.shape[-1]

        auto_conditioning = cond_mel
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment)
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        wavs = []
        has_warned = False
        for sent in segments:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                      cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                    device=text_tokens.device),
                                                      pathology_ids=pathology_ids,
                                                      do_sample=do_sample,
                                                      top_p=top_p,
                                                      top_k=top_k,
                                                      temperature=temperature,
                                                      num_return_sequences=1,
                                                      length_penalty=length_penalty,
                                                      num_beams=num_beams,
                                                      repetition_penalty=repetition_penalty,
                                                      max_generate_length=max_mel_tokens,
                                                      **generation_kwargs)
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(f"WARN: generation stopped due to exceeding max_mel_tokens ({max_mel_tokens}).",
                                  category=RuntimeWarning)
                    has_warned = True

                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(auto_conditioning, text_tokens,
                                      torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                      code_lens * self.gpt.mel_length_compression,
                                      cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                    device=text_tokens.device),
                                      return_latent=True, clip_inputs=False,
                                      pathology_ids=pathology_ids)
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    wav = wav.squeeze(1)
                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                wavs.append(wav.cpu())
        end_time = time.perf_counter()
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / max(wav_length, 1e-6):.4f}")

        wav = wav.cpu()
        if output_path:
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)


if __name__ == "__main__":
    import argparse
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        default=os.path.join(repo_root, "configs", "controllable_dysarthric_speech_synthesis.yaml"),
    )
    ap.add_argument("--model-dir", required=True,
                    help="directory containing BigVGAN, DVAE and tokenizer files")
    ap.add_argument("--gpt-ckpt", required=True, help="path to the trained gpt_best.pth")
    ap.add_argument("--prompt", required=True, help="timbre prompt wav")
    ap.add_argument("--text", required=True)
    ap.add_argument("--pathology", type=int, default=0, help="condition index k (0=healthy, 1=F01, 2=M01, 3=M02, 4=M04, 5=M05)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tts = IndexTTS(cfg_path=args.cfg, model_dir=args.model_dir, is_fp16=False,
                   use_cuda_kernel=False, gpt_checkpoint=args.gpt_ckpt)
    tts.infer(audio_prompt=args.prompt, text=args.text, output_path=args.out, pathology_ids=args.pathology)
