import argparse
import json
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Reuse the text normalization and tokenization pipeline used for inference.
from indextts.utils.front import TextNormalizer, TextTokenizer


class FinetuneDataset(Dataset):
    """
    Custom dataset used for UnifiedVoice fine-tuning, supporting multi-speaker data.
    """
    def __init__(self, manifest_files: List[str], bpe_path: str, speaker_ids: List[str], config: DictConfig):
        super().__init__()
        self.config = config
        self.data = []

        # Keep training text processing consistent with inference.
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(bpe_path, self.normalizer)

        print(">> TextNormalizer loaded for training")
        print(f">> BPE model loaded from: {bpe_path}")

        # Indices support condition-aware and speaker-aware sampling.
        self.pathology2indices: Dict[int, List[int]] = {}
        self.speaker2indices: Dict[str, List[int]] = {}

        for manifest_file, speaker_id in zip(manifest_files, speaker_ids):
            logger.info(f"Loading data from manifest: {manifest_file} for speaker: {speaker_id}")
            with open(manifest_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    item = json.loads(line.strip())

                    duration = item.get("duration", 0)
                    if duration > 20 or duration < 0.2:
                        continue

                    item["speaker_id"] = speaker_id
                    # Accept legacy manifests that store the label as severity_label.
                    pathology = item.get("pathology_label")
                    if pathology is None:
                        pathology = item.get("severity_label", -1)
                    pathology = -1 if pathology is None else int(pathology)
                    item["pathology_label"] = pathology

                    self.data.append(item)
                    idx = len(self.data) - 1

                    self.pathology2indices.setdefault(pathology, []).append(idx)
                    self.speaker2indices.setdefault(speaker_id, []).append(idx)

        logger.info(
            f"Loaded {len(self.data)} samples from {len(manifest_files)} speakers. "
            f"pathology buckets: { {k: len(v) for k, v in self.pathology2indices.items()} }"
        )

    def _load_mel(self, item: dict) -> torch.FloatTensor:
        """Load a mel spectrogram for prompt sampling."""
        mels_path = item.get("mels", None)
        mels_npy = np.load(mels_path)
        return torch.FloatTensor(mels_npy)

    def _sample_index_from_bucket(self, bucket: List[int], exclude_idx: int) -> int:
        """
        Sample an index from a bucket while avoiding ``exclude_idx`` when possible.
        """
        if not bucket:
            return exclude_idx

        if len(bucket) == 1:
            return bucket[0]

        for _ in range(10):
            j = int(np.random.choice(bucket))
            if j != exclude_idx:
                return j
        # Use a deterministic fallback if repeated random draws select the source item.
        return bucket[0] if bucket[0] != exclude_idx else bucket[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        text = item["text"]
        codes_path = item.get("codes", None)
        mels_path = item.get("mels", None)
        condition_path = item.get("condition", None)
        speaker_id = item["speaker_id"]
        pathology_label = item.get("pathology_label", None)

        # Apply the same text pipeline used during inference.
        text_tokens_list = self.tokenizer.tokenize(text)
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        text_ids = torch.LongTensor(text_ids).unsqueeze(0)

        # Load precomputed acoustic features.
        codes_npy = np.load(codes_path)
        mels_npy = np.load(mels_path)
        condition_npy = np.load(condition_path) if condition_path else None

        mel_spec = torch.FloatTensor(mels_npy)  # [B, D, T]
        mel_codes = torch.LongTensor(codes_npy)  # [B, T]
        condition = torch.FloatTensor(condition_npy) if condition_npy is not None else None

        # Sample a real recording from the timbre-donor speaker.
        # Converted directory names retain the donor speaker as their prefix:
        #   'FC02_converted_from_F01' -> 'FC02',  'M01_converted' -> 'M01',  'F01' -> 'F01'
        spk_id = speaker_id.split('_converted')[0]
        spk_bucket = self.speaker2indices.get(spk_id, [])

        spk_idx = self._sample_index_from_bucket(spk_bucket, exclude_idx=index)

        mel_same_speaker = self._load_mel(self.data[spk_idx])
        return (
            mel_spec,
            mel_codes,
            text_ids,
            condition,
            speaker_id,
            int(pathology_label),
            mel_same_speaker,      # Random timbre prompt from the same donor speaker.
        )


# Batch collation


def _pad_sequence(seqs, pad_value=0, dim=-1):
    """Pad a list of tensors on the specified dimension to the max length.

    Args:
        seqs (List[torch.Tensor]): list of tensors with shape (..., L_i)
        pad_value (int|float): value to use for padding
        dim (int): dimension to pad on (default: last dimension)

    Returns:
        Tuple[torch.Tensor, torch.LongTensor]:
            - padded tensor of shape (*batch, max_len)
            - lengths tensor (B,)
    """

    assert dim == -1, "Padding dimension must be the last dimension"

    lengths = torch.tensor([s.shape[dim] for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    # Determine output shape
    out_shape = list(seqs[0].shape)
    out_shape[dim] = max_len
    out_shape = [len(seqs)] + out_shape  # prepend batch dim

    padded = seqs[0].new_full(out_shape, pad_value)

    for i, s in enumerate(seqs):
        # Copy each tensor into a batch padded along the final dimension.
        if s.dim() == 1:  # 1D tensor: [L] -> [B, L]
            padded[i, :s.shape[0]] = s
        elif s.dim() == 2:  # 2D tensor: [D, L] -> [B, D, L]
            padded[i, :, :s.shape[1]] = s
        elif s.dim() == 3:  # 3D tensor: [C, D, L] -> [B, C, D, L]
            padded[i, :, :, :s.shape[2]] = s
        else:  # Four or more dimensions.
            padded[i, ..., :s.shape[-1]] = s

    return padded, lengths


def collate_finetune_fn(batch):
    """
    Returns:
        mel_specs            (B, 100, T_max)
        mel_codes            (B, T_max_codes)
        text_ids             (B, T_max_text)
        conditions           (B, 32, 1280)
        speaker_ids          List[str] of length B
        pathology_labels      (B,)  LongTensor
        mel_same_speaker     (B, 100, T_max_spk)     TIMBRE prompt
        mel_lengths          (B,)
        codes_lengths        (B,)
        text_lengths         (B,)
        spk_mel_lengths      (B,)
    """
    (
        mel_specs,
        mel_codes,
        text_ids,
        conditions,
        speaker_ids,
        pathology_labels,
        mel_same_speaker,
    ) = zip(*batch)

    # Remove the extra batch dimension left by data preprocessing for mel_specs and mel_codes (1, ...) -> (...)
    mel_specs = [spec.squeeze(0) if spec.dim() >= 3 and spec.size(0) == 1 else spec for spec in mel_specs]
    mel_codes = [codes.squeeze(0) if codes.dim() >= 2 and codes.size(0) == 1 else codes for codes in mel_codes]

    mel_same_speaker = [
        m.squeeze(0) if m is not None and m.dim() >= 3 and m.size(0) == 1 else m
        for m in mel_same_speaker
    ]

    # Remove the extra batch dimension added to text_ids inside the dataset (1, L) -> (L)
    text_ids = [ids.squeeze(0) if ids.dim() == 2 and ids.size(0) == 1 else ids for ids in text_ids]

    # Conditioning features are required for every sample.
    assert all(cond is not None for cond in conditions), "conditions must not be None"
    conditions = [cond.squeeze(0) if cond.dim() >= 3 and cond.size(0) == 1 else cond for cond in conditions]

    # A timbre prompt is required for every sample.
    assert all(m is not None for m in mel_same_speaker), "mel_same_speaker must not be None"

    # Pad the primary training inputs.
    mel_specs_padded, mel_lengths = _pad_sequence(list(mel_specs), pad_value=0.0, dim=-1)
    mel_codes_padded, codes_lengths = _pad_sequence(list(mel_codes), pad_value=0, dim=-1)
    text_ids_padded, text_lengths = _pad_sequence(list(text_ids), pad_value=0, dim=-1)

    # Pad the timbre prompts.
    mel_same_spk_padded, spk_mel_lengths = _pad_sequence(list(mel_same_speaker), pad_value=0.0, dim=-1)

    # Stack conditions directly since they all have the same shape [32, 1280]
    conditions_padded = torch.stack(conditions, dim=0)  # [B, 32, 1280]

    # Convert integer pathology labels to a tensor of shape [B].
    pathology_labels = torch.tensor(pathology_labels, dtype=torch.long)

    return (
        mel_specs_padded,        # [B, 100, T_max]
        mel_codes_padded,        # [B, T_max_codes]
        text_ids_padded,         # [B, T_max_text]
        conditions_padded,       # [B, 32, 1280]
        list(speaker_ids),       # [B] list[str]
        pathology_labels,         # [B]
        mel_same_spk_padded,     # [B, 100, T_max_spk]   TIMBRE prompt
        mel_lengths,
        codes_lengths,
        text_lengths,
        spk_mel_lengths,
    )


def load_finetune_datasets_pathology(config: DictConfig, bpe_path: str) -> Tuple[Dataset, Dataset]:
    """Utility helper to load the train/validation datasets for multi-speaker training.

    Args:
        config (DictConfig): Global configuration.
        bpe_path (str): Path to the BPE model file.

    Returns:
        Tuple[Dataset, Dataset]: ``(train_dataset, validation_dataset)``.
    """
    # Read per-speaker manifest locations.
    speaker_info_path = os.path.join(config.train.data_path, "speaker_info.json")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)

    train_manifest_files = []
    valid_manifest_files = []
    speaker_ids = []

    # Collect training and validation manifests for every speaker.
    for speaker_info in speaker_info_list:
        speaker_id = speaker_info['speaker']

        train_file = speaker_info['train_jsonl']
        valid_file = speaker_info['valid_jsonl']

        if os.path.exists(train_file) and os.path.exists(valid_file):
            train_manifest_files.append(train_file)
            valid_manifest_files.append(valid_file)
            speaker_ids.append(speaker_id)
            logger.info(f"Added speaker {speaker_id} with data from {train_file} and {valid_file}")
        else:
            logger.warning(f"Missing metadata files for speaker {speaker_id}")

    # Construct the training and validation datasets.
    train_dataset = FinetuneDataset(train_manifest_files, bpe_path, speaker_ids, config)
    valid_dataset = FinetuneDataset(valid_manifest_files, bpe_path, speaker_ids, config)

    return train_dataset, valid_dataset


def load_speaker_conditions(config: DictConfig) -> dict:
    """Load the mean conditioning tensor for every speaker.

    Args:
        config (DictConfig): Global configuration.

    Returns:
        dict: Dictionary mapping speaker_id to mean_condition tensor.
    """
    speaker_info_path = os.path.join(config.train.data_path, "speaker_info.json")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)

    speaker_conditions = {}
    for speaker_info in speaker_info_list:
        speaker_id = speaker_info['speaker']
        medoid_path = speaker_info['medoid_condition']

        if os.path.exists(medoid_path):
            condition = np.load(medoid_path)
            speaker_conditions[speaker_id] = torch.from_numpy(condition).float()
            logger.info(f"Loaded mean condition for speaker {speaker_id}: shape {condition.shape}")
        else:
            raise ValueError(f"Missing medoid_condition.npy for speaker {speaker_id}")

    return speaker_conditions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load finetune datasets.")
    parser.add_argument("--config", type=str, default="finetune_models/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--bpe_model", type=str, default="finetune_models/bpe.model", help="Path to the SentencePiece model.")

    args = parser.parse_args()

    # Load config file
    config = OmegaConf.load(args.config)

    # Load datasets with the configured BPE model.
    train_dataset, valid_dataset = load_finetune_datasets_pathology(config, args.bpe_model)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")

    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_finetune_fn)
    sample_batch = next(iter(loader))
    logger.info(f"Batch has {len(sample_batch)} fields")
