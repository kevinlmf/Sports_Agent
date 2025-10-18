"""
Interpretability Module for Sports Injury Risk Models

Provides multiple explanation methods:
1. SHAP (SHapley Additive exPlanations) - Feature importance
2. Grad-CAM (Gradient-weighted Class Activation Mapping) - Visual explanations
3. Attention Visualization - Attention weight heatmaps
4. Feature Attribution - Input sensitivity analysis

Use cases:
- Model debugging
- Clinical decision support
- Trust and transparency
- Regulatory compliance (e.g., GDPR right to explanation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for tabular features

    Computes Shapley values for each feature:
        φ_i = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)! / |F|!] · [f(S∪{i}) - f(S)]

    Where:
    - φ_i: SHAP value for feature i
    - F: Set of all features
    - S: Subset of features
    - f: Model prediction function

    References:
        Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Trained model to explain
            background_data: Background dataset for SHAP (e.g., training mean)
            device: Device to run on
        """
        self.model = model.to(device).eval()
        self.background_data = background_data.to(device)
        self.device = device

        logger.info(f"SHAP Explainer initialized with background shape: {background_data.shape}")

    def explain_instance(
        self,
        instance: torch.Tensor,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Compute SHAP values for a single instance using sampling

        Args:
            instance: Input instance (n_features,)
            n_samples: Number of samples for approximation

        Returns:
            shap_values: SHAP value for each feature (n_features,)
        """
        try:
            import shap
        except ImportError:
            logger.error("shap library not installed. Install with: pip install shap")
            return self._naive_feature_importance(instance)

        # Create explainer
        def predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(x_tensor)
                if out.dim() > 1 and out.size(1) > 1:
                    # Classification: return probability of positive class
                    return F.softmax(out, dim=1)[:, 1].cpu().numpy()
                else:
                    # Regression/Survival: return raw output
                    return out.squeeze().cpu().numpy()

        explainer = shap.KernelExplainer(
            predict_fn,
            self.background_data.cpu().numpy()
        )

        # Compute SHAP values
        shap_values = explainer.shap_values(
            instance.cpu().numpy(),
            nsamples=n_samples
        )

        return shap_values

    def _naive_feature_importance(
        self,
        instance: torch.Tensor,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Naive feature importance via permutation (fallback if SHAP not available)

        Args:
            instance: Input instance
            n_samples: Number of permutation samples

        Returns:
            importance: Feature importance scores
        """
        instance = instance.to(self.device)

        # Baseline prediction
        with torch.no_grad():
            baseline_pred = self.model(instance.unsqueeze(0))

        n_features = instance.size(0)
        importance = np.zeros(n_features)

        # Permute each feature
        for i in range(n_features):
            perturbed_instance = instance.clone()

            # Sample from background distribution
            background_vals = self.background_data[:, i]
            sampled_vals = background_vals[torch.randint(0, len(background_vals), (n_samples,))]

            delta = 0
            for val in sampled_vals:
                perturbed_instance[i] = val
                with torch.no_grad():
                    perturbed_pred = self.model(perturbed_instance.unsqueeze(0))
                delta += torch.abs(perturbed_pred - baseline_pred).item()

            importance[i] = delta / n_samples

        return importance

    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance bar chart

        Args:
            shap_values: SHAP values (n_features,)
            feature_names: Feature names
            save_path: Optional path to save figure
        """
        # Sort by absolute SHAP value
        abs_shap = np.abs(shap_values)
        sorted_idx = np.argsort(abs_shap)[::-1][:20]  # Top 20

        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(sorted_idx)),
            abs_shap[sorted_idx],
            color=['red' if v < 0 else 'green' for v in shap_values[sorted_idx]]
        )
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('|SHAP Value|')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        plt.show()


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations

    Computes class-discriminative saliency maps:
        L^c_GradCAM = ReLU(Σ_k α_k^c · A^k)

    Where:
    - α_k^c = (1/Z) Σ_i Σ_j (∂y^c/∂A^k_ij): Importance weight for feature map k
    - A^k: Activations of feature map k
    - y^c: Score for class c

    References:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Vision model to explain
            target_layer: Layer to compute CAM on (usually last conv layer)
            device: Device to run on
        """
        self.model = model.to(device).eval()
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

        logger.info(f"Grad-CAM initialized on layer: {target_layer}")

    def _register_hooks(self):
        """Register forward and backward hooks"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap

        Args:
            image: Input image (1, 3, H, W)
            target_class: Target class index (if None, use predicted class)

        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        image = image.to(self.device)
        image.requires_grad = True

        # Forward pass
        output = self.model(image)

        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute weights
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Generate CAM
        cam = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / (cam.max() + 1e-8)  # Normalize

        return cam

    def visualize_cam(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        save_path: Optional[str] = None,
        alpha: float = 0.5
    ):
        """
        Overlay CAM on original image

        Args:
            image: Original image (H, W, 3)
            cam: CAM heatmap (H', W')
            save_path: Optional save path
            alpha: Overlay transparency
        """
        # Resize CAM to image size
        from scipy.ndimage import zoom
        cam_resized = zoom(cam, (image.shape[0] / cam.shape[0], image.shape[1] / cam.shape[1]))

        # Create heatmap
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]

        # Overlay
        overlayed = alpha * heatmap + (1 - alpha) * image / 255.0

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(overlayed)
        plt.title('Overlay')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Grad-CAM visualization to {save_path}")

        plt.tight_layout()
        plt.show()


class AttentionVisualizer:
    """
    Attention Weight Visualization for Transformer models

    Visualizes:
    1. Self-attention heatmaps
    2. Cross-modal attention patterns
    3. Layer-wise attention evolution
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Transformer model with attention weights
            device: Device to run on
        """
        self.model = model.to(device).eval()
        self.device = device

    def extract_attention_weights(
        self,
        inputs: Dict[str, torch.Tensor],
        return_all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from model

        Args:
            inputs: Model inputs
            return_all_layers: Return weights from all layers

        Returns:
            attention_weights: Dictionary of attention tensors
        """
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass with attention return
        with torch.no_grad():
            output, attention_dict = self.model(**inputs, return_attention=True)

        return attention_dict

    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: str = 'Attention Heatmap',
        save_path: Optional[str] = None
    ):
        """
        Plot attention weight heatmap

        Args:
            attention_weights: Attention tensor (n_heads, seq_len, seq_len) or (seq_len, seq_len)
            row_labels: Labels for rows (queries)
            col_labels: Labels for columns (keys)
            title: Plot title
            save_path: Optional save path
        """
        # Handle multi-head attention
        if attention_weights.dim() == 4:
            # (batch, n_heads, seq, seq) -> average over batch and heads
            attention_weights = attention_weights.mean(dim=[0, 1])
        elif attention_weights.dim() == 3:
            # (batch or n_heads, seq, seq) -> average
            attention_weights = attention_weights.mean(dim=0)

        attention_weights = attention_weights.cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights,
            xticklabels=col_labels if col_labels else False,
            yticklabels=row_labels if row_labels else False,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {save_path}")

        plt.tight_layout()
        plt.show()

    def plot_cross_modal_attention(
        self,
        attention_dict: Dict[str, torch.Tensor],
        modality_names: List[str] = ['Vision', 'Text', 'Tabular'],
        save_dir: Optional[str] = None
    ):
        """
        Plot cross-modal attention patterns

        Args:
            attention_dict: Dictionary with keys like 'vision_text', 'vision_tabular', etc.
            modality_names: Names of modalities
            save_dir: Directory to save plots
        """
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for key, attn_weights in attention_dict.items():
            title = f"Cross-Modal Attention: {key.replace('_', ' → ')}"
            save_path = Path(save_dir) / f"{key}_attention.png" if save_dir else None

            self.plot_attention_heatmap(
                attention_weights,
                title=title,
                save_path=str(save_path) if save_path else None
            )


class IntegratedGradients:
    """
    Integrated Gradients for input attribution

    Computes attribution by integrating gradients along path from baseline to input:
        IG_i(x) = (x_i - x'_i) · ∫_{α=0}^1 (∂F(x' + α(x-x'))/∂x_i) dα

    Where:
    - x: Input
    - x': Baseline (e.g., zeros, mean)
    - F: Model output

    References:
        Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Model to explain
            device: Device to run on
        """
        self.model = model.to(device).eval()
        self.device = device

    def compute_attributions(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        n_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute integrated gradients

        Args:
            inputs: Input tensor (batch, ...)
            baseline: Baseline tensor (same shape as inputs)
            target_class: Target class for classification
            n_steps: Number of integration steps

        Returns:
            attributions: Attribution scores (same shape as inputs)
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        baseline = baseline.to(self.device)
        inputs = inputs.to(self.device)

        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, n_steps).to(self.device)

        # Accumulate gradients
        gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad = True

            # Forward pass
            output = self.model(interpolated)

            # Get target score
            if target_class is not None:
                score = output[:, target_class].sum()
            else:
                score = output.sum()

            # Backward pass
            score.backward()

            gradients.append(interpolated.grad.detach())

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Compute attributions
        attributions = (inputs - baseline) * avg_gradients

        return attributions


def create_shap_explainer(
    model: nn.Module,
    background_data: torch.Tensor
) -> SHAPExplainer:
    """Factory function to create SHAP explainer"""
    return SHAPExplainer(model, background_data)


def create_gradcam_explainer(
    model: nn.Module,
    target_layer: nn.Module
) -> GradCAM:
    """Factory function to create Grad-CAM explainer"""
    return GradCAM(model, target_layer)


def create_attention_visualizer(
    model: nn.Module
) -> AttentionVisualizer:
    """Factory function to create attention visualizer"""
    return AttentionVisualizer(model)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(50, 2)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    background = torch.randn(100, 50)

    # Test SHAP
    print("\n=== Testing SHAP Explainer ===")
    shap_explainer = create_shap_explainer(model, background)

    instance = torch.randn(50)
    shap_values = shap_explainer.explain_instance(instance, n_samples=100)
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Top 5 important features: {np.argsort(np.abs(shap_values))[-5:]}")
