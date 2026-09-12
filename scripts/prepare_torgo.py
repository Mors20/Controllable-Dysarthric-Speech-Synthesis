# -*- coding: utf-8 -*-
"""Prepare TORGO and counterfactual audio for controllable dysarthric TTS.

Produces under the requested ``--out_dir``:
  <speaker>/feats/<uid>_{mel,codes,condition}.npy
  <speaker>/metadata_train.jsonl / metadata_valid.jsonl
  speaker_info.json
  pathology_embedding/mean_pathology_condition_{0..5}.npy

Speaker dirs covered (70 total, matching the original merged set):
  * 15 real TORGO speakers
  * 5  patient-timbre conversions      X_converted            (pathology 0: healthy content, patient timbre)
  * 50 patient-content conversions     X_converted_from_Y     (pathology = class(Y), X timbre)

pathology_label classes (per-patient, ground truth from the original metadata):
  0 = healthy control condition (controls + mild patients F03/F04/M03 + X_converted)
  1 = F01, 2 = M01, 3 = M02, 4 = M04, 5 = M05

Item fields (as consumed by data_utils_pathology_disentangle_embedding.FinetuneDataset):
  text, codes, mels, condition, duration, pathology_label, patient_id
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
import torchaudio
from loguru import logger
from omegaconf import OmegaConf

from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.vqvae.xtts_dvae import DiscreteVAE
from indextts.gpt.model import UnifiedVoice

# ----------------------------------------------------------------------------
# Per-patient pathology classes defined by the dataset metadata.
PATHOLOGY_CLASS = {"F01": 1, "M01": 2, "M02": 3, "M04": 4, "M05": 5}
REAL_SPEAKERS = ["F01", "F03", "F04", "FC01", "FC02", "FC03",
                 "M01", "M02", "M03", "M04", "M05",
                 "MC01", "MC02", "MC03", "MC04"]
CONTENT_SOURCES = ["F01", "M01", "M02", "M04", "M05"]           # the 5 modeled patients
TIMBRE_TARGETS = ["F03", "F04", "FC01", "FC02", "FC03",
                  "M03", "MC01", "MC02", "MC03", "MC04"]        # VC timbre targets


def pathology_of_dir(speaker_dir: str) -> int:
    """pathology_label for every item inside a given speaker dir."""
    if "_converted_from_" in speaker_dir:
        src = speaker_dir.split("_converted_from_")[1]
        return PATHOLOGY_CLASS[src]
    if speaker_dir.endswith("_converted"):
        return 0                       # healthy content in patient timbre
    return PATHOLOGY_CLASS.get(speaker_dir, 0)


def load_prompts(torgo_root: str, speaker: str):
    """uid -> text from TORGO prompts. uid key: (session, utt_id)."""
    prompts = {}
    spk_dir = os.path.join(torgo_root, speaker)
    if not os.path.isdir(spk_dir):
        return prompts
    for session in sorted(os.listdir(spk_dir)):
        pdir = os.path.join(spk_dir, session, "prompts")
        if not os.path.isdir(pdir):
            continue
        for f in os.listdir(pdir):
            if not f.endswith(".txt"):
                continue
            uid = f[:-4]
            try:
                with open(os.path.join(pdir, f), "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read().strip()
            except Exception:
                continue
            if not text:
                continue
            # skip non-verbal / picture-description prompts (no fixed text)
            low = text.lower()
            if low.endswith(".jpg") or low.startswith("[") or "/" in text:
                continue
            prompts[(session, uid)] = text
    return prompts


def iter_wavs(root_dir: str):
    """Yield (session, mic, uid, wav_path) under a speaker-style dir."""
    for session in sorted(os.listdir(root_dir)):
        sdir = os.path.join(root_dir, session)
        if not os.path.isdir(sdir) or not session.lower().startswith("session"):
            continue
        for mic in ("wav_arrayMic", "wav_headMic"):
            mdir = os.path.join(sdir, mic)
            if not os.path.isdir(mdir):
                continue
            for f in sorted(os.listdir(mdir)):
                if f.endswith(".wav"):
                    yield session, mic, f[:-4], os.path.join(mdir, f)


class FeatureExtractor:
    def __init__(self, finetune_dir: str, config_path: str, device: str = "cuda"):
        self.device = device
        cfg = OmegaConf.load(config_path)
        self.mel_fn = MelSpectrogramFeatures().to(device)

        dvae_path = os.path.join(finetune_dir, cfg.dvae_checkpoint)
        self.dvae = DiscreteVAE(channels=cfg.vqvae.channels,
                                num_tokens=cfg.vqvae.num_tokens,
                                hidden_dim=cfg.vqvae.hidden_dim,
                                num_resnet_blocks=cfg.vqvae.num_resnet_blocks,
                                codebook_dim=cfg.vqvae.codebook_dim,
                                num_layers=cfg.vqvae.num_layers,
                                positional_dims=cfg.vqvae.positional_dims,
                                kernel_size=cfg.vqvae.kernel_size,
                                use_transposed_convs=cfg.vqvae.use_transposed_convs)
        dvae_sd = torch.load(dvae_path, map_location="cpu")
        dvae_sd = dvae_sd.get("model", dvae_sd)
        self.dvae.load_state_dict(dvae_sd, strict=False)
        self.dvae.eval().to(device)

        gpt_path = os.path.join(finetune_dir, "gpt.pth")
        gpt_sd = torch.load(gpt_path, map_location="cpu")
        gpt_sd = gpt_sd.get("model", gpt_sd)
        self.gpt = UnifiedVoice(**cfg.gpt)
        self.gpt.load_state_dict(gpt_sd, strict=False)
        self.gpt.eval().to(device)

    @torch.no_grad()
    def extract(self, wav_path: str):
        audio, sr = torchaudio.load(wav_path)
        audio = audio[:1]  # mono
        if sr != 24000:
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        duration = audio.shape[-1] / 24000.0
        if duration < 0.2 or duration > 20:
            return None
        if not torch.isfinite(audio).all():
            return None                                            # corrupt wav (NaN samples)
        audio = audio.to(self.device)
        mel = self.mel_fn(audio)                                   # (1, 100, T)
        codes = self.dvae.get_codebook_indices(mel)                # (1, T//4)
        cond_len = torch.tensor([mel.shape[-1]], device=self.device)
        cond = self.gpt.get_conditioning(mel, cond_len)            # (1, 32, 1280)
        if not (torch.isfinite(mel).all() and torch.isfinite(cond).all()):
            return None                                            # degenerate features
        return (mel.cpu().numpy().astype(np.float32),
                codes.cpu().numpy().astype(np.int64),
                cond.cpu().numpy().astype(np.float32),
                duration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torgo_root", required=True)
    ap.add_argument("--converted_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--finetune_dir", required=True)
    ap.add_argument(
        "--config",
        default=os.path.join(REPO, "configs", "controllable_dysarthric_speech_synthesis.yaml"),
    )
    ap.add_argument("--valid_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--only_speaker", default=None, help="process a single speaker dir (for parallel/partial runs)")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-items-per-speaker", type=int,
                    help="Limit items for a fast pipeline smoke test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fx = FeatureExtractor(args.finetune_dir, args.config)

    # prompts for all real speakers (converted dirs mirror a real speaker's session layout)
    all_prompts = {spk: load_prompts(args.torgo_root, spk) for spk in REAL_SPEAKERS}

    # enumerate speaker dirs: (dir_name, physical_root, prompt_speaker)
    speaker_dirs = []
    for spk in REAL_SPEAKERS:
        speaker_dirs.append((spk, os.path.join(args.torgo_root, spk), spk))
    for x in CONTENT_SOURCES:                       # X_converted mirrors X's own layout
        d = f"{x}_converted"
        speaker_dirs.append((d, os.path.join(args.converted_root, d), x))
    for x in TIMBRE_TARGETS:                        # X_converted_from_Y mirrors X's (target) layout
        for y in CONTENT_SOURCES:
            d = f"{x}_converted_from_{y}"
            speaker_dirs.append((d, os.path.join(args.converted_root, d), x))

    speaker_info = []
    for dir_name, phys_root, prompt_spk in speaker_dirs:
        if args.only_speaker and dir_name != args.only_speaker:
            continue
        if not os.path.isdir(phys_root):
            logger.warning(f"missing dir, skip: {phys_root}")
            continue
        pathology = pathology_of_dir(dir_name)
        out_spk = os.path.join(args.out_dir, dir_name)
        feats_dir = os.path.join(out_spk, "feats")
        os.makedirs(feats_dir, exist_ok=True)

        items = []
        prompts = all_prompts[prompt_spk]
        for session, mic, uid, wav_path in iter_wavs(phys_root):
            if args.max_items_per_speaker is not None and len(items) >= args.max_items_per_speaker:
                break
            text = prompts.get((session, uid))
            if text is None:
                continue
            key = f"{session}_{mic}_{uid}"
            mel_p = os.path.join(feats_dir, key + "_mel.npy")
            codes_p = os.path.join(feats_dir, key + "_codes.npy")
            cond_p = os.path.join(feats_dir, key + "_condition.npy")
            dur_p = os.path.join(feats_dir, key + "_dur.json")
            if args.skip_existing and all(os.path.exists(p) for p in (mel_p, codes_p, cond_p, dur_p)):
                duration = json.load(open(dur_p))["duration"]
            else:
                try:
                    r = fx.extract(wav_path)
                except Exception as e:
                    logger.warning(f"extract failed {wav_path}: {e}")
                    continue
                if r is None:
                    continue
                mel, codes, cond, duration = r
                np.save(mel_p, mel)
                np.save(codes_p, codes)
                np.save(cond_p, cond)
                json.dump({"duration": duration}, open(dur_p, "w"))
            items.append({
                "text": text,
                "codes": codes_p,
                "mels": mel_p,
                "condition": cond_p,
                "duration": duration,
                "pathology_label": pathology,
                "patient_id": dir_name,
                "wav": wav_path,
            })

        if not items:
            logger.warning(f"no items for {dir_name}")
            continue

        # 2% per-speaker valid split
        idx = np.arange(len(items))
        rng.shuffle(idx)
        # Keep at least one training item; a one-item split is useful for smoke tests.
        n_valid = 0 if len(items) == 1 else min(len(items) - 1, max(1, int(round(len(items) * args.valid_ratio))))
        valid_set = set(idx[:n_valid].tolist())
        with open(os.path.join(out_spk, "metadata_train.jsonl"), "w", encoding="utf-8") as ftr, \
             open(os.path.join(out_spk, "metadata_valid.jsonl"), "w", encoding="utf-8") as fva:
            for i, it in enumerate(items):
                line = json.dumps(it, ensure_ascii=False) + "\n"
                (fva if i in valid_set else ftr).write(line)

        # medoid condition = mean condition of the speaker (kept for speaker_info completeness)
        conds = np.stack([np.load(it["condition"])[0] for it in items[:: max(1, len(items)//200)]])
        medoid_p = os.path.join(out_spk, "medoid_condition.npy")
        np.save(medoid_p, conds.mean(axis=0, keepdims=True).astype(np.float32))

        speaker_info.append({
            "speaker": dir_name,
            "train_jsonl": os.path.join(out_spk, "metadata_train.jsonl"),
            "valid_jsonl": os.path.join(out_spk, "metadata_valid.jsonl"),
            "medoid_condition": medoid_p,
            "pathology_label": pathology,
            "n_items": len(items),
        })
        logger.info(f"[{dir_name}] items={len(items)} pathology={pathology}")

    if args.only_speaker:
        # merge into existing speaker_info.json
        si_path = os.path.join(args.out_dir, "speaker_info.json")
        existing = json.load(open(si_path)) if os.path.exists(si_path) else []
        existing = [s for s in existing if s["speaker"] not in {x["speaker"] for x in speaker_info}]
        speaker_info = existing + speaker_info
    for entry in speaker_info:
        if "pathology_label" not in entry and "severity_label" in entry:
            entry["pathology_label"] = entry.pop("severity_label")
    speaker_info.sort(key=lambda s: s["speaker"])
    with open(os.path.join(args.out_dir, "speaker_info.json"), "w", encoding="utf-8") as f:
        json.dump(speaker_info, f, indent=2, ensure_ascii=False)

    # pathology embedding init: mean condition per class over TRAIN items
    emb_dir = os.path.join(args.out_dir, "pathology_embedding")
    os.makedirs(emb_dir, exist_ok=True)
    sums = {k: None for k in range(6)}
    counts = {k: 0 for k in range(6)}
    for si in speaker_info:
        with open(si["train_jsonl"], encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                label = it.get("pathology_label", it.get("severity_label"))
                k = int(label)
                c = np.load(it["condition"])[0]          # (32, 1280)
                sums[k] = c if sums[k] is None else sums[k] + c
                counts[k] += 1
    for k in range(6):
        if counts[k] == 0:
            logger.warning(f"class {k} empty — skipping embedding")
            continue
        mean_c = (sums[k] / counts[k]).astype(np.float32)[None]   # (1, 32, 1280)
        np.save(os.path.join(emb_dir, f"mean_pathology_condition_{k}.npy"), mean_c)
        logger.info(f"mean_pathology_condition_{k}: n={counts[k]}")

    # summary
    logger.info(f"speakers: {len(speaker_info)}, total items: {sum(s['n_items'] for s in speaker_info)}")


if __name__ == "__main__":
    main()
