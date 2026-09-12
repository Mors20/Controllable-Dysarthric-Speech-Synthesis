# -*- coding: utf-8 -*-
"""Generate Seed-VC counterfactual data for controllable dysarthric TTS.

Reproduces the two conversion sets used in TORGO_merged (evidence-verified):

MODE new  -> X_converted_from_Y  (50 dirs = 10 timbre targets x 5 patient sources)
    For EVERY utterance SLOT of timbre target X, if patient Y (content+pathology
    donor) has >=1 utterance with the SAME prompt text, pick one at random as the
    VC content source; X's own utterance at that slot is the timbre reference.
    Output mirrors the TARGET (X) path (one converted copy of Y's speech per
    matching X slot -> a single Y utterance fans out into many target slots):
        <out>/X_converted_from_Y/<X-session>/<mic>/<uid>.wav

MODE old  -> X_converted  (5 dirs, X in the 5 modeled patients)
    For EVERY utterance of patient X (timbre reference, defines output slot),
    pick one random cross-group (control) utterance with the SAME text as the
    content source and convert. Output mirrors the TARGET (X) path:
        <out>/X_converted/<X-session>/<mic>/<uid>.wav
    (healthy content in patient timbre -> pathology_label 0)

Run from the seed-vc repo root:
    python generate_data.py --mode new --torgo_root /path/to/TORGO \
        --output /path/to/generated
VC params match the originals: diffusion-steps 30, length-adjust 1.0,
inference-cfg-rate 0.7, f0-condition False, seed 1234, fp16 True.
"""
import argparse
import os
import sys
import time

import numpy as np
import librosa
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DYS = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
CONTROL = ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"]
SOURCES = ["F01", "M01", "M02", "M04", "M05"]                 # pathology/content donors
TARGETS_NEW = ["F03", "F04", "FC01", "FC02", "FC03", "M03", "MC01", "MC02", "MC03", "MC04"]


def get_group(spk):
    return "dys" if spk in DYS else "control"


def scan_torgo(torgo_root):
    """rows: dicts(wav, spk, session, mic, uid, text)."""
    rows = []
    for spk in DYS + CONTROL:
        sdir = os.path.join(torgo_root, spk)
        if not os.path.isdir(sdir):
            continue
        for session in sorted(os.listdir(sdir)):
            pdir = os.path.join(sdir, session, "prompts")
            if not os.path.isdir(pdir):
                continue
            prompts = {}
            for f in os.listdir(pdir):
                if f.endswith(".txt"):
                    try:
                        t = open(os.path.join(pdir, f), encoding="utf-8", errors="ignore").read().strip()
                    except Exception:
                        continue
                    low = t.lower()
                    if not t or low.endswith(".jpg") or low.startswith("[") or "/" in t:
                        continue
                    prompts[f[:-4]] = t
            for mic in ("wav_arrayMic", "wav_headMic"):
                mdir = os.path.join(sdir, session, mic)
                if not os.path.isdir(mdir):
                    continue
                for f in sorted(os.listdir(mdir)):
                    if not f.endswith(".wav"):
                        continue
                    uid = f[:-4]
                    text = prompts.get(uid)
                    if text is None:
                        continue
                    rows.append({"wav": os.path.join(mdir, f), "spk": spk, "session": session,
                                 "mic": mic, "uid": uid, "text": text})
    return rows


def build_pairs_new(rows, output_dir, rng):
    """(src_wav, ref_wav, out_path) for X_converted_from_Y — TARGET-slot iterated.

    src = random same-text utterance of patient Y (content+pathology),
    ref = target X's own utterance at the slot (timbre),
    out = X's slot path under X_converted_from_Y.
    """
    by_spk_text = {}
    for r in rows:
        by_spk_text.setdefault(r["spk"], {}).setdefault(r["text"], []).append(r)
    pairs = []
    for x in TARGETS_NEW:
        tgt_rows = [r for r in rows if r["spk"] == x]
        for y in SOURCES:
            ysrc = by_spk_text.get(y, {})
            for tr in tgt_rows:
                cands = ysrc.get(tr["text"])
                if not cands:
                    continue
                src = cands[int(rng.integers(0, len(cands)))]
                out = os.path.join(output_dir, f"{x}_converted_from_{y}",
                                   tr["session"], tr["mic"], tr["uid"] + ".wav")
                pairs.append((src["wav"], tr["wav"], out))
    return pairs


def build_pairs_old(rows, output_dir, rng):
    """(src_wav, ref_wav, out_path) for X_converted (5 patient-timbre dirs)."""
    pairs = []
    by_text = {}
    for r in rows:
        by_text.setdefault(r["text"], []).append(r)
    for x in SOURCES:                      # X = timbre target (patient)
        tgt_rows = [r for r in rows if r["spk"] == x]
        for tr in tgt_rows:
            cands = [r for r in by_text.get(tr["text"], []) if get_group(r["spk"]) != get_group(x)]
            if not cands:
                continue
            src = cands[int(rng.integers(0, len(cands)))]
            out = os.path.join(output_dir, f"{x}_converted",
                               tr["session"], tr["mic"], tr["uid"] + ".wav")
            pairs.append((src["wav"], tr["wav"], out))
    return pairs


@torch.no_grad()
def convert_one(model, semantic_fn, vocoder_fn, campplus_model, mel_fn, device, fp16,
                src_wav, ref_wav, out_path, diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7):
    sr = 22050
    hop_length = 256
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = librosa.load(src_wav, sr=sr)[0]
    ref_audio = librosa.load(ref_wav, sr=sr)[0]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    S_alt = semantic_fn(converted_waves_16k)
    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.float())
    mel2 = mel_fn(ref_audio.float())
    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    cond, _, _, _, _ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)
    prompt_condition, _, _, _, _ = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=None)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(cat_condition,
                                            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                            mel2, style2, None, diffusion_steps,
                                            inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            generated_wave_chunks.append(_crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
            break
        else:
            generated_wave_chunks.append(
                _crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len))
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, vc_wave.cpu(), sr)


def _crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["new", "old", "both"], default="both")
    ap.add_argument("--torgo_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--diffusion-steps", type=int, default=30)
    ap.add_argument("--length-adjust", type=float, default=1.0)
    ap.add_argument("--inference-cfg-rate", type=float, default=0.7)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shard", type=int, default=0, help="shard index for multi-process runs")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max-items", type=int, help="Limit this shard for a smoke test")
    ap.add_argument("--dry-run", action="store_true", help="Build and validate pairs without loading models")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        ap.error("require num_shards >= 1 and 0 <= shard < num_shards")

    rows = scan_torgo(args.torgo_root)
    print(f"TORGO rows with text: {len(rows)}")

    rng = np.random.default_rng(args.seed)
    pairs = []
    if args.mode in ("new", "both"):
        pairs += build_pairs_new(rows, args.output, rng)
    if args.mode in ("old", "both"):
        pairs += build_pairs_old(rows, args.output, rng)
    print(f"Total pairs to convert: {len(pairs)}")

    todo = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard and not os.path.exists(p[2])]
    if args.max_items is not None:
        todo = todo[:args.max_items]
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} to do")
    if args.dry_run:
        for src, ref, out in todo[:5]:
            print(f"PAIR src={src} ref={ref} out={out}")
        print("DRY_RUN_DONE")
        return

    # Load the f0-less Whisper-small Seed-VC model only after pair validation.
    import types as _types
    from inference import load_models
    la = _types.SimpleNamespace(f0_condition=False, checkpoint=None, config=None, fp16=args.fp16)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import inference as _inf
    _inf.device = device
    _inf.fp16 = args.fp16
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = load_models(la)

    t0 = time.time()
    for i, (src, ref, out) in enumerate(todo, 1):
        try:
            convert_one(model, semantic_fn, vocoder_fn, campplus_model, mel_fn, device, args.fp16,
                        src, ref, out, args.diffusion_steps, args.length_adjust, args.inference_cfg_rate)
        except Exception as e:
            print(f"[{i}/{len(todo)}] FAILED src={src} ref={ref} err={repr(e)}")
        if i % 100 == 0:
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] elapsed {el/60:.1f}min eta {el/i*(len(todo)-i)/60:.1f}min", flush=True)
    print("GEN_DONE")


if __name__ == "__main__":
    main()
