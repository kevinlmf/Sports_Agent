#!/usr/bin/env python3
"""
Pre-download HuggingFace models to avoid deadlocks during training

Run this ONCE before training to download all required models.
"""

import os
import sys
from pathlib import Path
import logging

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Show progress for downloads

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_name, model_type='tokenizer'):
    """Download a single model"""
    try:
        logger.info(f"Downloading {model_type}: {model_name}")

        if model_type == 'tokenizer':
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(model_name)
        elif model_type == 'image_processor':
            from transformers import AutoImageProcessor
            AutoImageProcessor.from_pretrained(model_name)
        elif model_type == 'clip':
            from transformers import CLIPProcessor
            CLIPProcessor.from_pretrained(model_name)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return False

        logger.info(f"✓ Successfully downloaded: {model_name}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False


def main():
    """Download all required models"""
    print("\n" + "="*70)
    print("HUGGINGFACE MODEL PRE-DOWNLOAD")
    print("="*70)
    print("\nThis will download the following models:")
    print("  1. bert-base-uncased (~420MB)")
    print("  2. dmis-lab/biobert-v1.1 (~420MB)")
    print("  3. openai/clip-vit-base-patch32 (~600MB)")
    print("  4. google/vit-base-patch16-224 (~350MB)")
    print("\nTotal: ~1.8GB")
    print("\nThis is a ONE-TIME download. Models will be cached locally.")
    print("="*70 + "\n")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    models_to_download = [
        ('bert-base-uncased', 'tokenizer'),
        ('dmis-lab/biobert-v1.1', 'tokenizer'),
        ('openai/clip-vit-base-patch32', 'clip'),
        ('google/vit-base-patch16-224', 'image_processor'),
    ]

    results = []
    for model_name, model_type in models_to_download:
        success = download_model(model_name, model_type)
        results.append((model_name, success))
        print()  # Blank line between downloads

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for model_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {model_name}")

    print(f"\nTotal: {success_count}/{total_count} successful")

    if success_count == total_count:
        print("\n✓ All models downloaded successfully!")
        print("You can now run training without deadlocks.")
    else:
        print("\n⚠ Some models failed to download.")
        print("Check your internet connection and try again.")
        sys.exit(1)


if __name__ == '__main__':
    main()
