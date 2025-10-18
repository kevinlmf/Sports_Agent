"""
Vision-Language Multimodal Model for Sports Injury Risk Prediction

Architecture:
    Vision Encoder (CLIP/ViT) → Vision Embeddings
    Text Encoder (BERT/BioBERT) → Text Embeddings
    Tabular Features → MLP Embeddings
    ↓
    Cross-Modal Attention Fusion
    ↓
    Risk Prediction Head (Binary/Survival)

Features:
- Pre-trained vision and language encoders
- LoRA adapters for efficient fine-tuning
- Cross-attention fusion for multimodal integration
- Interpretability via attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPModel, CLIPProcessor,
    ViTModel, ViTImageProcessor,
    BertModel, AutoTokenizer,
    AutoModel
)
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class TabularEncoder(nn.Module):
    """
    MLP encoder for tabular features
    Projects numerical features to embedding space
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 512, 768],
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tabular features (batch_size, input_dim)

        Returns:
            embeddings: (batch_size, output_dim)
        """
        return self.encoder(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing vision, language, and tabular modalities

    Uses multi-head attention to compute cross-modal interactions:
    - Vision ↔ Text
    - Vision ↔ Tabular
    - Text ↔ Tabular
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Cross-attention layers
        self.vision_text_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.vision_tabular_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.text_tabular_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Fusion projection
        self.fusion_proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(
        self,
        vision_emb: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        tabular_emb: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            vision_emb: (batch, embed_dim)
            text_emb: (batch, embed_dim)
            tabular_emb: (batch, embed_dim)
            return_attention: Whether to return attention weights

        Returns:
            fused_embedding: (batch, embed_dim)
            attention_weights: Optional dict of attention weights
        """
        # Ensure all embeddings are 3D for attention
        if vision_emb is not None and vision_emb.dim() == 2:
            vision_emb = vision_emb.unsqueeze(1)
        if text_emb is not None and text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)
        if tabular_emb is not None and tabular_emb.dim() == 2:
            tabular_emb = tabular_emb.unsqueeze(1)

        attention_weights = {} if return_attention else None

        # Collect available modalities
        modalities = []

        # Vision-Text cross-attention
        if vision_emb is not None and text_emb is not None:
            vt_attn, vt_weights = self.vision_text_attn(vision_emb, text_emb, text_emb)
            vt_attn = self.norm1(vt_attn + vision_emb)
            modalities.append(vt_attn.squeeze(1))
            if return_attention:
                attention_weights['vision_text'] = vt_weights

        # Vision-Tabular cross-attention
        if vision_emb is not None and tabular_emb is not None:
            vn_attn, vn_weights = self.vision_tabular_attn(vision_emb, tabular_emb, tabular_emb)
            vn_attn = self.norm2(vn_attn + vision_emb)
            modalities.append(vn_attn.squeeze(1))
            if return_attention:
                attention_weights['vision_tabular'] = vn_weights

        # Text-Tabular cross-attention
        if text_emb is not None and tabular_emb is not None:
            tn_attn, tn_weights = self.text_tabular_attn(text_emb, tabular_emb, tabular_emb)
            tn_attn = self.norm3(tn_attn + text_emb)
            modalities.append(tn_attn.squeeze(1))
            if return_attention:
                attention_weights['text_tabular'] = tn_weights

        # Handle case where only one modality is available
        if len(modalities) == 0:
            # Fallback to tabular only
            if tabular_emb is not None:
                return tabular_emb.squeeze(1), attention_weights
            else:
                raise ValueError("At least one modality must be provided")

        # Concatenate and fuse
        if len(modalities) == 1:
            fused = modalities[0]
        else:
            fused = torch.cat(modalities, dim=-1)
            fused = self.fusion_proj(fused)

        return fused, attention_weights


class VisionLanguageRiskModel(nn.Module):
    """
    Complete Vision-Language Multimodal Model for Injury Risk Prediction

    Components:
    1. Vision Encoder: CLIP or ViT
    2. Text Encoder: BERT or BioBERT
    3. Tabular Encoder: MLP
    4. Cross-Modal Fusion: Multi-head attention
    5. Risk Prediction Head: Binary classification or survival analysis

    Args:
        tabular_input_dim: Number of tabular features
        vision_encoder: Type of vision encoder ('clip', 'vit', 'resnet')
        text_encoder: Type of text encoder ('bert', 'biobert', 'clip')
        embed_dim: Embedding dimension for all modalities
        n_heads: Number of attention heads
        use_lora: Whether to use LoRA adapters
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        dropout: Dropout rate
        freeze_vision: Freeze vision encoder weights
        freeze_text: Freeze text encoder weights
        task: Prediction task ('classification', 'survival')
    """

    def __init__(
        self,
        tabular_input_dim: int,
        vision_encoder: str = 'clip',
        text_encoder: str = 'bert',
        embed_dim: int = 768,
        n_heads: int = 8,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        dropout: float = 0.1,
        freeze_vision: bool = True,
        freeze_text: bool = True,
        task: str = 'classification',
        num_classes: int = 2
    ):
        super().__init__()

        self.tabular_input_dim = tabular_input_dim
        self.vision_encoder_type = vision_encoder
        self.text_encoder_type = text_encoder
        self.embed_dim = embed_dim
        self.use_lora = use_lora
        self.task = task

        # Initialize vision encoder
        self._init_vision_encoder(vision_encoder, freeze_vision)

        # Initialize text encoder
        self._init_text_encoder(text_encoder, freeze_text)

        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dims=[256, 512, embed_dim],
            dropout=dropout
        )

        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        # Prediction head
        if task == 'classification':
            self.prediction_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes)
            )
        elif task == 'survival':
            # Cox proportional hazards style output
            self.prediction_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1)  # Log hazard ratio
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_r, lora_alpha)

        logger.info(f"Initialized VisionLanguageRiskModel with:")
        logger.info(f"  Vision: {vision_encoder}, Text: {text_encoder}, Tabular: {tabular_input_dim} features")
        logger.info(f"  Embed dim: {embed_dim}, LoRA: {use_lora}, Task: {task}")

    def _init_vision_encoder(self, vision_encoder: str, freeze: bool):
        """Initialize vision encoder"""
        import os
        # Prevent multiprocessing issues during model loading
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        try:
            if vision_encoder == 'clip':
                logger.info("Loading CLIP vision encoder (this may take a while on first run)...")
                self.vision_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
                self.vision_proj = nn.Linear(512, self.embed_dim)  # CLIP output is 512
            elif vision_encoder == 'vit':
                logger.info("Loading ViT vision encoder (this may take a while on first run)...")
                self.vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
                self.vision_proj = nn.Linear(768, self.embed_dim)  # ViT output is 768
            else:
                raise ValueError(f"Unknown vision encoder: {vision_encoder}")

            logger.info(f"Successfully loaded {vision_encoder} encoder")
        except Exception as e:
            logger.error(f"Failed to load vision encoder {vision_encoder}: {e}")
            raise

        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            logger.info(f"Froze {vision_encoder} encoder weights")

    def _init_text_encoder(self, text_encoder: str, freeze: bool):
        """Initialize text encoder"""
        import os
        # Prevent multiprocessing issues during model loading
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        try:
            if text_encoder == 'bert':
                logger.info("Loading BERT text encoder (this may take a while on first run)...")
                self.text_model = BertModel.from_pretrained('bert-base-uncased')
                self.text_proj = nn.Linear(768, self.embed_dim)
            elif text_encoder == 'biobert':
                logger.info("Loading BioBERT text encoder (this may take a while on first run)...")
                self.text_model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
                self.text_proj = nn.Linear(768, self.embed_dim)
            elif text_encoder == 'clip':
                logger.info("Loading CLIP text encoder (this may take a while on first run)...")
                # Use CLIP text encoder
                self.text_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
                self.text_proj = nn.Linear(512, self.embed_dim)
            else:
                raise ValueError(f"Unknown text encoder: {text_encoder}")

            logger.info(f"Successfully loaded {text_encoder} encoder")
        except Exception as e:
            logger.error(f"Failed to load text encoder {text_encoder}: {e}")
            raise

        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
            logger.info(f"Froze {text_encoder} encoder weights")

    def _apply_lora(self, r: int, alpha: int):
        """Apply LoRA adapters to vision and text encoders"""
        try:
            from peft import LoraConfig, get_peft_model

            # LoRA config
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["query", "value"],  # Apply to Q,V in attention
                lora_dropout=0.1,
                bias="none"
            )

            # Apply to vision model
            self.vision_model = get_peft_model(self.vision_model, lora_config)
            logger.info(f"Applied LoRA to vision encoder (r={r}, alpha={alpha})")

            # Apply to text model
            self.text_model = get_peft_model(self.text_model, lora_config)
            logger.info(f"Applied LoRA to text encoder (r={r}, alpha={alpha})")

        except ImportError:
            logger.warning("peft library not installed. Install with: pip install peft")

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings

        Args:
            images: (batch, 3, H, W)

        Returns:
            embeddings: (batch, embed_dim)
        """
        if self.vision_encoder_type == 'clip':
            outputs = self.vision_model.get_image_features(pixel_values=images)
        elif self.vision_encoder_type == 'vit':
            outputs = self.vision_model(pixel_values=images).last_hidden_state[:, 0]  # CLS token
        else:
            outputs = self.vision_model(images)

        return self.vision_proj(outputs)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode text to embeddings

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            embeddings: (batch, embed_dim)
        """
        if self.text_encoder_type == 'clip':
            outputs = self.text_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0]  # CLS token

        return self.text_proj(outputs)

    def forward(
        self,
        tabular: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass

        Args:
            tabular: Numerical features (batch, n_features)
            images: Images (batch, 3, H, W)
            text_input_ids: Text token IDs (batch, seq_len)
            text_attention_mask: Text attention mask (batch, seq_len)
            return_attention: Return attention weights

        Returns:
            logits: Classification logits (batch, num_classes) or hazard (batch, 1)
            attention_weights: Optional attention weights dictionary
        """
        # Encode modalities
        tabular_emb = self.tabular_encoder(tabular)

        vision_emb = None
        if images is not None:
            vision_emb = self.encode_vision(images)

        text_emb = None
        if text_input_ids is not None and text_attention_mask is not None:
            text_emb = self.encode_text(text_input_ids, text_attention_mask)

        # Cross-modal fusion
        fused_emb, attention_weights = self.cross_modal_fusion(
            vision_emb=vision_emb,
            text_emb=text_emb,
            tabular_emb=tabular_emb,
            return_attention=return_attention
        )

        # Prediction
        logits = self.prediction_head(fused_emb)

        if return_attention:
            return logits, attention_weights
        return logits

    def get_embedding(
        self,
        tabular: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get fused multimodal embedding without prediction head

        Useful for:
        - Feature extraction
        - Transfer learning
        - Visualization
        """
        # Encode modalities
        tabular_emb = self.tabular_encoder(tabular)

        vision_emb = None
        if images is not None:
            vision_emb = self.encode_vision(images)

        text_emb = None
        if text_input_ids is not None and text_attention_mask is not None:
            text_emb = self.encode_text(text_input_ids, text_attention_mask)

        # Cross-modal fusion
        fused_emb, _ = self.cross_modal_fusion(
            vision_emb=vision_emb,
            text_emb=text_emb,
            tabular_emb=tabular_emb,
            return_attention=False
        )

        return fused_emb


def create_vision_language_model(
    tabular_input_dim: int,
    config: Optional[Dict] = None
) -> VisionLanguageRiskModel:
    """
    Factory function to create VisionLanguageRiskModel

    Args:
        tabular_input_dim: Number of tabular features
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    if config is None:
        config = {
            'vision_encoder': 'clip',
            'text_encoder': 'bert',
            'embed_dim': 768,
            'n_heads': 8,
            'use_lora': False,
            'lora_r': 16,
            'lora_alpha': 32,
            'dropout': 0.1,
            'freeze_vision': True,
            'freeze_text': True,
            'task': 'classification',
            'num_classes': 2
        }

    model = VisionLanguageRiskModel(
        tabular_input_dim=tabular_input_dim,
        **config
    )

    return model


if __name__ == '__main__':
    # Test model
    logging.basicConfig(level=logging.INFO)

    model = create_vision_language_model(
        tabular_input_dim=50,
        config={
            'vision_encoder': 'clip',
            'text_encoder': 'bert',
            'embed_dim': 768,
            'use_lora': False
        }
    )

    # Test forward pass
    batch_size = 4
    tabular = torch.randn(batch_size, 50)
    images = torch.randn(batch_size, 3, 224, 224)
    text_input_ids = torch.randint(0, 1000, (batch_size, 128))
    text_attention_mask = torch.ones(batch_size, 128)

    logits = model(
        tabular=tabular,
        images=images,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask
    )

    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
