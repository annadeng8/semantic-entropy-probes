import os
import pickle
from argparse import ArgumentParser
from uncertainty.models.huggingface_models import HuggingfaceModel

def init_model(args):
    """Initialize the Huggingface model."""
    return HuggingfaceModel(args.model_name, max_new_tokens=args.model_max_new_tokens)

def save(obj, file, save_dir="/workspace/saved"):
    """Save object to file in specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file)
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Saved] {file} → {save_path}")

def get_parser():
    """Return a minimal argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2b", help="Model name")
    parser.add_argument("--model_max_new_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    return parser
