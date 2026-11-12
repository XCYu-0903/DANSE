import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn


def get_clap_audio_embed(clap_model, extractor, waveform):
    """Extract normalized CLAP audio embeddings for a batch of waveforms."""
    is_longer_eval = torch.zeros(waveform.size(0), 1, dtype=torch.bool, device=waveform.device)
    input_data = {
        "input_features": extractor(waveform),
        "is_longer": is_longer_eval,
        "use_specAug": False,
        "infer_mode": True,
    }
    return clap_model.get_audio_features(**input_data)


def get_clap_text_embed(clap_model, auto_tokenizer, input_list, device=None):
    """Extract normalized CLAP text embeddings for a list of prompts."""
    if device is None:
        device = next(clap_model.parameters()).device
    inputs = auto_tokenizer(input_list, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return clap_model.get_text_features(**inputs)


def get_text_embed_bank(embedding_path, device):
    embedding_files = glob(os.path.join(embedding_path, "*.npy"))
    if not embedding_files:
        raise FileNotFoundError(f"No text embeddings found in: {embedding_path}")

    entries = []
    for embedding_file in embedding_files:
        embedding_id = int(os.path.basename(embedding_file).split("_", 1)[0])
        embedding = torch.from_numpy(np.load(embedding_file)).squeeze().to(device)
        entries.append((embedding_id, embedding))

    entries.sort(key=lambda item: item[0])
    return torch.stack([embedding for _, embedding in entries], dim=0)


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


def load_checkpoint(filepath, device):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    print(f"Loading '{filepath}'")
    return torch.load(filepath, map_location=device)


def save_checkpoint(filepath, obj):
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)


def scan_checkpoint(checkpoint_dir, prefix):
    checkpoint_paths = []
    for path in glob(os.path.join(checkpoint_dir, f"{prefix}*")):
        step = os.path.basename(path).rsplit("_", 1)[-1]
        if os.path.isfile(path) and step.isdigit():
            checkpoint_paths.append(path)
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=lambda path: int(os.path.basename(path).rsplit("_", 1)[-1]))


def resolve_checkpoint_pair(checkpoint_dir, resume):
    if resume == "best":
        generator_path = os.path.join(checkpoint_dir, "g_best")
        discriminator_path = os.path.join(checkpoint_dir, "do_best")
        if not (os.path.isfile(generator_path) and os.path.isfile(discriminator_path)):
            raise FileNotFoundError(f"Best checkpoint pair not found in: {checkpoint_dir}")
        return generator_path, discriminator_path

    generator_paths = {}
    discriminator_paths = {}
    for prefix, paths in (("g_", generator_paths), ("do_", discriminator_paths)):
        for path in glob(os.path.join(checkpoint_dir, f"{prefix}*")):
            step = os.path.basename(path).rsplit("_", 1)[-1]
            if os.path.isfile(path) and step.isdigit():
                paths[int(step)] = path

    available_steps = sorted(generator_paths.keys() & discriminator_paths.keys())
    if resume == "latest":
        if not available_steps:
            return None, None
        step = available_steps[-1]
    else:
        try:
            step = int(resume)
        except ValueError as error:
            raise ValueError("--resume must be 'latest', 'best', or a numeric step.") from error
        if step not in available_steps:
            raise FileNotFoundError(f"Checkpoint pair for step {step} not found in: {checkpoint_dir}")

    return generator_paths[step], discriminator_paths[step]
