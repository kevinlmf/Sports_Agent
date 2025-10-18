"""
LoRA (Low-Rank Adaptation) Fine-tuning for Sports Injury Risk Models

LoRA enables efficient fine-tuning of large pre-trained models by:
1. Freezing the original model weights
2. Injecting trainable low-rank matrices into transformer layers
3. Reducing trainable parameters by 10-100x while maintaining performance

Mathematical Formulation:
    W' = W + ΔW
    ΔW = BA  where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)

References:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

logger = logging.getLogger(__name__)


class LoRAManager:
    """
    Manages LoRA adapter configuration and application

    Features:
    - Automatic target module detection
    - Selective layer freezing
    - Parameter-efficient training
    - Adapter merging and saving
    """

    def __init__(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: TaskType = TaskType.SEQ_CLS
    ):
        """
        Args:
            r: LoRA rank (smaller = fewer parameters)
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout for LoRA layers
            target_modules: Which modules to apply LoRA to
            bias: Bias handling ("none", "all", "lora_only")
            task_type: Task type for PEFT
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["query", "value"]
        self.bias = bias
        self.task_type = task_type

        logger.info(f"LoRA Config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Target modules: {self.target_modules}")

    def get_config(self) -> LoraConfig:
        """Get LoRA configuration object"""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type
        )

    def apply_lora(
        self,
        model: nn.Module,
        print_trainable_params: bool = True
    ) -> nn.Module:
        """
        Apply LoRA adapters to model

        Args:
            model: Base model to adapt
            print_trainable_params: Print parameter statistics

        Returns:
            Model with LoRA adapters
        """
        lora_config = self.get_config()

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        if print_trainable_params:
            self.print_trainable_parameters(model)

        return model

    @staticmethod
    def print_trainable_parameters(model: nn.Module):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_params = 0

        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_params

        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {percentage:.2f}%"
        )

    @staticmethod
    def merge_and_save(
        model: nn.Module,
        save_path: str,
        merge_weights: bool = True
    ):
        """
        Save LoRA adapter weights

        Args:
            model: Model with LoRA adapters
            save_path: Path to save adapter weights
            merge_weights: Whether to merge LoRA weights into base model
        """
        if merge_weights:
            model = model.merge_and_unload()

        model.save_pretrained(save_path)
        logger.info(f"Saved LoRA model to {save_path}")


class KnowledgeDistillation:
    """
    Knowledge Distillation for model compression

    Distills knowledge from a large teacher model to a smaller student model:
        L = α·L_CE(y_student, y_true) + (1-α)·L_KD(y_student, y_teacher)

    Where L_KD uses soft targets with temperature scaling:
        L_KD = KL(softmax(z_teacher/T), softmax(z_student/T))

    References:
        Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            teacher_model: Large pre-trained teacher
            student_model: Smaller student to train
            temperature: Softmax temperature for soft targets
            alpha: Weight for hard target loss (1-alpha for soft)
            device: Device to run on
        """
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        logger.info(f"Knowledge Distillation: T={temperature}, α={alpha}")
        logger.info(f"Teacher params: {sum(p.numel() for p in teacher_model.parameters()):,}")
        logger.info(f"Student params: {sum(p.numel() for p in student_model.parameters()):,}")

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        task: str = 'classification'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss

        Args:
            student_logits: Student predictions (batch, num_classes)
            teacher_logits: Teacher predictions (batch, num_classes)
            labels: True labels (batch,)
            task: 'classification' or 'survival'

        Returns:
            total_loss: Combined distillation loss
            loss_dict: Dictionary of loss components
        """
        # Hard target loss (with true labels)
        if task == 'classification':
            hard_loss = nn.CrossEntropyLoss()(student_logits, labels)

            # Soft target loss (with teacher predictions)
            soft_student = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
            soft_teacher = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
            soft_loss = nn.functional.kl_div(
                soft_student,
                soft_teacher,
                reduction='batchmean'
            ) * (self.temperature ** 2)

        elif task == 'survival':
            # MSE loss for survival analysis
            hard_loss = nn.MSELoss()(student_logits, labels.unsqueeze(1).float())
            soft_loss = nn.MSELoss()(student_logits, teacher_logits)

        else:
            raise ValueError(f"Unknown task: {task}")

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        loss_dict = {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        task: str = 'classification'
    ) -> Dict[str, float]:
        """
        Single training step with distillation

        Args:
            batch: Batch dictionary with inputs and labels
            optimizer: Optimizer
            task: Task type

        Returns:
            Loss dictionary
        """
        self.student.train()

        # Move batch to device
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(self.device)

        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(**inputs)

        # Student predictions
        student_logits = self.student(**inputs)

        # Compute distillation loss
        loss, loss_dict = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            task=task
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss_dict


class ProgressiveDistillation:
    """
    Progressive (layer-wise) knowledge distillation

    Aligns intermediate layers between teacher and student:
        L_total = L_output + Σ_i λ_i·L_layer(h_teacher^i, h_student^i)

    Useful for:
    - Better feature alignment
    - Improved convergence
    - Higher student performance
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        layer_weights: Optional[List[float]] = None,
        temperature: float = 3.0,
        alpha: float = 0.5
    ):
        """
        Args:
            teacher_model: Teacher model
            student_model: Student model
            layer_weights: Weights for each intermediate layer loss
            temperature: Distillation temperature
            alpha: Balance between hard and soft losses
        """
        self.teacher = teacher_model.eval()
        self.student = student_model
        self.layer_weights = layer_weights or [0.1, 0.2, 0.3, 0.4]
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_layer_loss(
        self,
        teacher_hidden: torch.Tensor,
        student_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between intermediate layer outputs

        Args:
            teacher_hidden: Teacher layer output (batch, seq, dim)
            student_hidden: Student layer output (batch, seq, dim)

        Returns:
            layer_loss: MSE between hidden states
        """
        # If dimensions don't match, project student to teacher dimension
        if teacher_hidden.shape != student_hidden.shape:
            student_hidden = nn.functional.adaptive_avg_pool1d(
                student_hidden.transpose(1, 2),
                teacher_hidden.size(1)
            ).transpose(1, 2)

        return nn.functional.mse_loss(student_hidden, teacher_hidden)


def create_lora_model(
    base_model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Factory function to create LoRA-adapted model

    Args:
        base_model: Base model to adapt
        r: LoRA rank
        lora_alpha: LoRA alpha
        target_modules: Modules to apply LoRA to

    Returns:
        Model with LoRA adapters
    """
    manager = LoRAManager(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules
    )

    return manager.apply_lora(base_model)


def create_distillation_trainer(
    teacher_model: nn.Module,
    student_model: nn.Module,
    temperature: float = 3.0,
    alpha: float = 0.5
) -> KnowledgeDistillation:
    """
    Factory function to create distillation trainer

    Args:
        teacher_model: Teacher model
        student_model: Student model
        temperature: Distillation temperature
        alpha: Hard/soft loss balance

    Returns:
        KnowledgeDistillation trainer
    """
    return KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=temperature,
        alpha=alpha
    )


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.fc = nn.Linear(100, hidden_size)
            self.classifier = nn.Linear(hidden_size, 2)

        def forward(self, x):
            return self.classifier(self.fc(x))

    # Test LoRA
    print("\n=== Testing LoRA ===")
    base_model = DummyModel()
    lora_model = create_lora_model(base_model, r=8, lora_alpha=16)

    # Test Distillation
    print("\n=== Testing Knowledge Distillation ===")
    teacher = DummyModel(hidden_size=1024)
    student = DummyModel(hidden_size=256)

    distiller = create_distillation_trainer(
        teacher_model=teacher,
        student_model=student,
        temperature=3.0,
        alpha=0.7
    )

    # Dummy batch
    batch = {
        'x': torch.randn(4, 100),
        'label': torch.randint(0, 2, (4,))
    }

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    loss_dict = distiller.train_step(batch, optimizer, task='classification')

    print(f"Distillation losses: {loss_dict}")
