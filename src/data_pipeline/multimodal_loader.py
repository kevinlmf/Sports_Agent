"""
Multimodal Data Loader for Sports Injury Risk Prediction
Supports: Vision (images/video frames) + Text (reports/notes) + Tabular (numerical features)

Designed for modern deep learning pipelines with:
- Vision: Training videos, posture images, biomechanical captures
- Text: Athlete logs, medical reports, coach notes
- Tabular: Load metrics, workload, acute:chronic ratios
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoImageProcessor, CLIPProcessor
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class MultiModalSportsDataset(Dataset):
    """
    Multimodal dataset for sports injury risk prediction

    Supports three modalities:
    1. Vision: Training videos, posture analysis, biomechanical images
    2. Text: Medical reports, athlete notes, coach assessments
    3. Tabular: Numerical features (workload, vitals, performance metrics)

    Args:
        tabular_path: Path to CSV with numerical features
        text_path: Optional path to CSV with text data (columns: 'id', 'text')
        image_dir: Optional directory with images (filenames should match IDs)
        vision_model: Type of vision encoder ('clip', 'vit', 'resnet')
        text_model: Type of text encoder ('bert', 'biobert', 'clip')
        max_text_length: Maximum text sequence length
        image_size: Target image size (height, width)
    """

    def __init__(
        self,
        tabular_path: str,
        text_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        vision_model: str = 'clip',
        text_model: str = 'bert',
        max_text_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        label_column: str = 'injury',
        id_column: str = 'athlete_id'
    ):
        super().__init__()

        self.vision_model = vision_model
        self.text_model = text_model
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.label_column = label_column
        self.id_column = id_column

        # Load tabular data
        self.tabular_df = pd.read_csv(tabular_path)
        logger.info(f"Loaded {len(self.tabular_df)} samples from {tabular_path}")

        # Load text data if provided
        self.text_df = None
        if text_path is not None:
            self.text_df = pd.read_csv(text_path)
            logger.info(f"Loaded text data from {text_path}")

        # Set up image directory
        self.image_dir = Path(image_dir) if image_dir else None
        if self.image_dir:
            logger.info(f"Image directory: {self.image_dir}")

        # Initialize processors
        self._init_processors()

        # Extract features and labels
        self.label_col = label_column
        if label_column in self.tabular_df.columns:
            self.labels = self.tabular_df[label_column].values
            self.feature_columns = [col for col in self.tabular_df.columns
                                   if col not in [label_column, id_column]]
        else:
            self.labels = None
            self.feature_columns = [col for col in self.tabular_df.columns
                                   if col != id_column]

        self.tabular_features = self.tabular_df[self.feature_columns].values.astype(np.float32)

        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Modalities: Tabular={True}, Text={text_path is not None}, Vision={image_dir is not None}")

    def _init_processors(self):
        """Initialize text tokenizers and image processors"""
        import os
        # Disable parallelism to avoid mutex lock issues
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Prevent HuggingFace from spawning multiple processes during model loading
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

        # Text processor - try local first, then download if needed
        try:
            if self.text_model == 'bert':
                try:
                    self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
                    logger.info("Loaded BERT tokenizer from cache")
                except:
                    logger.warning("BERT not in cache, downloading (this may take a while)...")
                    self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
            elif self.text_model == 'biobert':
                try:
                    self.text_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1', local_files_only=True)
                    logger.info("Loaded BioBERT tokenizer from cache")
                except:
                    logger.warning("BioBERT not in cache, downloading (this may take a while)...")
                    self.text_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1', local_files_only=False)
            elif self.text_model == 'clip':
                try:
                    self.text_tokenizer = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
                    logger.info("Loaded CLIP processor from cache")
                except:
                    logger.warning("CLIP not in cache, downloading (this may take a while)...")
                    self.text_tokenizer = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=False)
            else:
                logger.warning(f"Unknown text model {self.text_model}, using BERT")
                try:
                    self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
                except:
                    self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
        except Exception as e:
            logger.error(f"Error loading text tokenizer: {e}")
            logger.error("TIP: Run 'python scripts/download_models.py' to pre-download models")
            raise

        # Image processor
        try:
            if self.vision_model == 'clip':
                try:
                    self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
                    logger.info("Loaded CLIP image processor from cache")
                except:
                    logger.warning("CLIP not in cache, downloading (this may take a while)...")
                    self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=False)
                self.image_transform = None  # CLIP processor handles transforms
            elif self.vision_model == 'vit':
                try:
                    self.image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', local_files_only=True)
                    logger.info("Loaded ViT image processor from cache")
                except:
                    logger.warning("ViT not in cache, downloading (this may take a while)...")
                    self.image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', local_files_only=False)
                self.image_transform = None
            else:  # Default transforms for ResNet, custom models
                self.image_processor = None
                self.image_transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        except Exception as e:
            logger.error(f"Error loading image processor: {e}")
            logger.error("TIP: Run 'python scripts/download_models.py' to pre-download models")
            raise

    def __len__(self) -> int:
        return len(self.tabular_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get multimodal sample

        Returns:
            Dictionary with keys:
            - 'tabular': Numerical features (n_features,)
            - 'text_input_ids': Tokenized text (max_length,) [if text available]
            - 'text_attention_mask': Text attention mask (max_length,) [if text available]
            - 'image': Image tensor (3, H, W) [if image available]
            - 'label': Target label (scalar) [if labels available]
            - 'athlete_id': Athlete identifier
        """
        sample = {}

        # Tabular features
        sample['tabular'] = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        # Get athlete ID
        athlete_id = self.tabular_df.iloc[idx][self.id_column]
        sample['athlete_id'] = athlete_id

        # Text features
        if self.text_df is not None:
            text_row = self.text_df[self.text_df[self.id_column] == athlete_id]
            if not text_row.empty:
                text = text_row.iloc[0]['text']

                if self.text_model == 'clip':
                    text_inputs = self.text_tokenizer(
                        text=text,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_text_length
                    )
                else:
                    text_inputs = self.text_tokenizer(
                        text,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_text_length
                    )

                sample['text_input_ids'] = text_inputs['input_ids'].squeeze(0)
                sample['text_attention_mask'] = text_inputs['attention_mask'].squeeze(0)
            else:
                # Return zero tensors if text not found
                sample['text_input_ids'] = torch.zeros(self.max_text_length, dtype=torch.long)
                sample['text_attention_mask'] = torch.zeros(self.max_text_length, dtype=torch.long)

        # Image features
        if self.image_dir is not None:
            # Try multiple image extensions
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = self.image_dir / f"{athlete_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path and image_path.exists():
                try:
                    image = Image.open(image_path).convert('RGB')

                    if self.image_processor is not None:
                        # Use HuggingFace processor
                        if self.vision_model == 'clip':
                            image_inputs = self.image_processor(images=image, return_tensors='pt')
                            sample['image'] = image_inputs['pixel_values'].squeeze(0)
                        else:
                            image_inputs = self.image_processor(images=image, return_tensors='pt')
                            sample['image'] = image_inputs['pixel_values'].squeeze(0)
                    else:
                        # Use torchvision transforms
                        sample['image'] = self.image_transform(image)

                except Exception as e:
                    logger.warning(f"Error loading image {image_path}: {e}")
                    # Return zero tensor as fallback
                    sample['image'] = torch.zeros(3, *self.image_size)
            else:
                # Image not found, return zero tensor
                sample['image'] = torch.zeros(3, *self.image_size)

        # Label
        if self.labels is not None:
            sample['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample

    def get_feature_names(self) -> List[str]:
        """Get list of tabular feature names"""
        return self.feature_columns

    def get_num_features(self) -> int:
        """Get number of tabular features"""
        return len(self.feature_columns)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for multimodal data

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary with stacked tensors
        """
        collated = {}

        # Stack tabular features
        if 'tabular' in batch[0]:
            collated['tabular'] = torch.stack([sample['tabular'] for sample in batch])

        # Stack text features
        if 'text_input_ids' in batch[0]:
            collated['text_input_ids'] = torch.stack([sample['text_input_ids'] for sample in batch])
            collated['text_attention_mask'] = torch.stack([sample['text_attention_mask'] for sample in batch])

        # Stack images
        if 'image' in batch[0]:
            collated['image'] = torch.stack([sample['image'] for sample in batch])

        # Stack labels
        if 'label' in batch[0]:
            collated['label'] = torch.stack([sample['label'] for sample in batch])

        # Collect athlete IDs
        if 'athlete_id' in batch[0]:
            collated['athlete_id'] = [sample['athlete_id'] for sample in batch]

        return collated


def create_multimodal_loaders(
    train_tabular: str,
    val_tabular: str,
    test_tabular: Optional[str] = None,
    train_text: Optional[str] = None,
    val_text: Optional[str] = None,
    test_text: Optional[str] = None,
    train_image_dir: Optional[str] = None,
    val_image_dir: Optional[str] = None,
    test_image_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    vision_model: str = 'clip',
    text_model: str = 'bert',
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Factory function to create train/val/test dataloaders

    Args:
        train_tabular: Path to training tabular CSV
        val_tabular: Path to validation tabular CSV
        test_tabular: Optional path to test tabular CSV
        train_text: Optional path to training text CSV
        val_text: Optional path to validation text CSV
        test_text: Optional path to test text CSV
        train_image_dir: Optional training image directory
        val_image_dir: Optional validation image directory
        test_image_dir: Optional test image directory
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        vision_model: Vision encoder type
        text_model: Text encoder type
        **kwargs: Additional arguments for MultiModalSportsDataset

    Returns:
        train_loader, val_loader, test_loader (optional)
    """

    # Create datasets
    train_dataset = MultiModalSportsDataset(
        tabular_path=train_tabular,
        text_path=train_text,
        image_dir=train_image_dir,
        vision_model=vision_model,
        text_model=text_model,
        **kwargs
    )

    val_dataset = MultiModalSportsDataset(
        tabular_path=val_tabular,
        text_path=val_text,
        image_dir=val_image_dir,
        vision_model=vision_model,
        text_model=text_model,
        **kwargs
    )

    test_dataset = None
    if test_tabular is not None:
        test_dataset = MultiModalSportsDataset(
            tabular_path=test_tabular,
            text_path=test_text,
            image_dir=test_image_dir,
            vision_model=vision_model,
            text_model=text_model,
            **kwargs
        )

    # Create dataloaders
    # CRITICAL: Force num_workers=0 to prevent multiprocessing deadlocks with HuggingFace models
    # The mutex.cc blocking issue occurs when multiple processes try to initialize transformers
    # For production: pre-download all models first, then can safely use num_workers > 0
    safe_num_workers = 0  # ALWAYS use 0 for multimodal models with transformers

    logger.warning(f"Setting num_workers=0 to avoid multiprocessing deadlocks with transformers models")
    logger.info("To enable multiprocessing: pre-download all models, then restart training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=safe_num_workers,
        collate_fn=MultiModalSportsDataset.collate_fn,
        pin_memory=True,  # Safe with num_workers=0
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_num_workers,
        collate_fn=MultiModalSportsDataset.collate_fn,
        pin_memory=True,  # Safe with num_workers=0
        persistent_workers=False
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=safe_num_workers,
            collate_fn=MultiModalSportsDataset.collate_fn,
            pin_memory=True,  # Safe with num_workers=0
            persistent_workers=False
        )

    logger.info(f"Created dataloaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader) if test_loader else None}")

    return train_loader, val_loader, test_loader


class SyntheticDataAugmenter:
    """
    Synthetic data generation and augmentation for multimodal sports data

    Uses:
    - Torchvision transforms for image augmentation
    - Text paraphrasing for text augmentation
    - SMOTE/ADASYN for tabular feature augmentation
    """

    def __init__(
        self,
        image_aug_prob: float = 0.5,
        text_aug_prob: float = 0.3,
        tabular_noise_std: float = 0.05
    ):
        self.image_aug_prob = image_aug_prob
        self.text_aug_prob = text_aug_prob
        self.tabular_noise_std = tabular_noise_std

        # Image augmentation pipeline
        self.image_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomErasing(p=0.2)
        ])

    def augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random image augmentation"""
        if np.random.rand() < self.image_aug_prob:
            return self.image_augment(image)
        return image

    def augment_tabular(self, features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tabular features"""
        noise = torch.randn_like(features) * self.tabular_noise_std
        return features + noise

    def augment_text(self, text: str) -> str:
        """
        Simple text augmentation (placeholder)
        For production, use back-translation or paraphrasing models
        """
        if np.random.rand() < self.text_aug_prob:
            # Simple synonym replacement (placeholder)
            # In production, use nlpaug or similar
            return text
        return text


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data for testing
    sample_df = pd.DataFrame({
        'athlete_id': range(100),
        'workload': np.random.rand(100) * 1000,
        'acute_chronic_ratio': np.random.rand(100) * 2,
        'previous_injuries': np.random.randint(0, 5, 100),
        'injury': np.random.randint(0, 2, 100)
    })
    sample_df.to_csv('sample_train.csv', index=False)

    # Test dataset
    dataset = MultiModalSportsDataset(
        tabular_path='sample_train.csv',
        vision_model='clip',
        text_model='bert'
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Feature names: {dataset.get_feature_names()}")
    print(f"Number of features: {dataset.get_num_features()}")

    # Test sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Tabular shape: {sample['tabular'].shape}")
    if 'label' in sample:
        print(f"Label: {sample['label']}")
