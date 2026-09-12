# Controllable Dysarthric Speech Synthesis

This repository contains the data generation, training, and inference code for controllable dysarthric speech synthesis.

**Project page:** [Audio samples and system overview](https://mors20.github.io/Controllable-Dysarthric-Speech-Synthesis/)

## Requirements

Use Linux, Python 3.10, and an NVIDIA GPU. Prepare:

- the TORGO dataset;
- the four IndexTTS-1.5 files shown below;
- enough disk space for generated audio and prepared features.

```text
artifacts/pretrained/
 bpe.model
 dvae.pth
 gpt.pth
 bigvgan_generator.pth
```

Install the environment once:

```bash
conda create -n tts python=3.10 -y
conda activate tts
pip install -e .
pip install -r requirements-seed-vc.txt
```

All commands below are run from the repository root unless stated otherwise.

## Step 1: Generate counterfactual audio with Seed-VC

Input: the original TORGO folder.
Output: voice-converted WAV files.

```bash
cd third_party/seed-vc
python generate_data.py \
  --mode both \
  --torgo_root /path/to/TORGO \
  --output /path/to/converted_audio
cd ../..
```

The script skips output files that already exist, so it is safe to resume. Before a full run, add `--dry-run` to check the paths without loading the models.

## Step 2: Prepare training features

Input: original TORGO audio, Step 1 output, and IndexTTS-1.5 files.
Output: mel features, codec tokens, conditioning features, manifests, and pathology embeddings.

```bash
python scripts/prepare_torgo.py \
  --torgo_root /path/to/TORGO \
  --converted_root /path/to/converted_audio \
  --out_dir /path/to/prepared_data \
  --finetune_dir artifacts/pretrained \
  --config configs/controllable_dysarthric_speech_synthesis.yaml
```

This step also skips completed features and can be resumed.

## Step 3: Train

Input: Step 2 output and the IndexTTS-1.5 files.
Output: checkpoints under `artifacts/pretrained/checkpoints_dys_spk_grl_exp1/`.

```bash
python train.py \
  --config configs/controllable_dysarthric_speech_synthesis.yaml \
  --model-dir artifacts/pretrained \
  --data-dir /path/to/prepared_data \
  --embedding-dir /path/to/prepared_data/pathology_embedding \
  --epochs 20 \
  --batch-size 2 \
  --num-workers 4
```

For a quick training check, append:

```text
--epochs 1 --max-train-batches 1 --skip-validation --no-save
```

## Step 4: Run inference

Choose a prompt WAV for the target voice and a pathology ID:

- `0`: healthy condition
- `1`: F01
- `2`: M01
- `3`: M02
- `4`: M04
- `5`: M05

```bash
python -m indextts.inference \
  --cfg configs/controllable_dysarthric_speech_synthesis.yaml \
  --model-dir artifacts/pretrained \
  --gpt-ckpt /path/to/gpt_best.pth \
  --prompt /path/to/prompt.wav \
  --text "Please call Stella." \
  --pathology 4 \
  --out outputs/example.wav
```

Use the same prompt with different pathology IDs to change the articulation condition while retaining the prompt speaker's timbre.
