#!/usr/bin/env python
"""
MLOps Demo for Sports Injury Risk Prediction
Demonstrates experiment tracking, profiling, and monitoring
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

def demo_mlflow_tracking():
    """Demonstrate MLflow experiment tracking"""
    print("\n" + "="*70)
    print("MLflow Experiment Tracking Demo")
    print("="*70 + "\n")

    try:
        import mlflow

        # Set experiment
        mlflow.set_experiment("sports_injury_risk")

        with mlflow.start_run(run_name=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            params = {
                "model_type": "TransCHIME",
                "learning_rate": 0.001,
                "batch_size": 32,
                "num_epochs": 10,
                "optimizer": "AdamW"
            }
            mlflow.log_params(params)

            # Simulate training metrics
            for epoch in range(5):
                metrics = {
                    "train_loss": 0.5 * np.exp(-0.2 * epoch) + np.random.uniform(0, 0.1),
                    "train_auc": 0.7 + 0.05 * epoch + np.random.uniform(0, 0.02),
                    "val_loss": 0.6 * np.exp(-0.15 * epoch) + np.random.uniform(0, 0.1),
                    "val_auc": 0.65 + 0.055 * epoch + np.random.uniform(0, 0.02),
                }
                mlflow.log_metrics(metrics, step=epoch)

            # Log final results
            mlflow.log_metrics({
                "final_auc": 0.89,
                "final_f1": 0.86,
                "final_precision": 0.87,
                "final_recall": 0.85
            })

            print("✓ MLflow tracking completed")
            print(f"  • Parameters logged: {len(params)}")
            print(f"  • Training epochs: 5")
            print(f"  • Final AUC: 0.89")
            print(f"  • Run ID: {mlflow.active_run().info.run_id}")

    except ImportError:
        print("⚙️  MLflow not installed")
        print("   Install with: pip install mlflow")
        print("   Start UI with: mlflow ui")


def demo_wandb_tracking():
    """Demonstrate W&B experiment tracking"""
    print("\n" + "="*70)
    print("Weights & Biases Tracking Demo")
    print("="*70 + "\n")

    try:
        import wandb

        # Initialize (in offline mode for demo)
        wandb.init(
            project="sports-injury-risk",
            name=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode="offline",
            config={
                "model": "TransCHIME",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10
            }
        )

        # Log metrics
        for step in range(5):
            wandb.log({
                "train/loss": 0.5 * np.exp(-0.2 * step) + np.random.uniform(0, 0.1),
                "train/auc": 0.7 + 0.05 * step,
                "val/loss": 0.6 * np.exp(-0.15 * step) + np.random.uniform(0, 0.1),
                "val/auc": 0.65 + 0.055 * step,
            })

        wandb.finish()

        print("✓ W&B tracking completed (offline mode)")
        print("  • Metrics logged: train/val loss, auc")
        print("  • Log location: wandb/offline-run-*")
        print("  • To sync: wandb sync wandb/offline-run-*")

    except ImportError:
        print("⚙️  Weights & Biases not installed")
        print("   Install with: pip install wandb")
        print("   Login with: wandb login")


def demo_pytorch_profiler():
    """Demonstrate PyTorch profiler"""
    print("\n" + "="*70)
    print("PyTorch Profiler Demo")
    print("="*70 + "\n")

    # Simple model for profiling
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    model = SimpleModel()
    inputs = torch.randn(32, 100)

    # Profile the model
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            outputs = model(inputs)
            loss = outputs.sum()
            loss.backward()

    # Print statistics
    print("Top 5 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))

    print("\n✓ Profiling completed")
    print("  • Forward passes: 10")
    print("  • Memory profiling: Available")
    print("  • Export format: Chrome trace (JSON)")


def demo_model_versioning():
    """Demonstrate model versioning and checkpointing"""
    print("\n" + "="*70)
    print("Model Versioning Demo")
    print("="*70 + "\n")

    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Simulate model checkpoint
    checkpoint = {
        "epoch": 10,
        "model_state_dict": {},  # Placeholder
        "optimizer_state_dict": {},  # Placeholder
        "metrics": {
            "train_loss": 0.234,
            "val_loss": 0.289,
            "auc_roc": 0.89,
            "f1_score": 0.86
        },
        "config": {
            "model_type": "TransCHIME",
            "num_layers": 6,
            "hidden_dim": 512,
            "num_heads": 8
        },
        "timestamp": datetime.now().isoformat()
    }

    # Version naming convention
    version = "v1.0.0"
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{version}.pt")

    print(f"Checkpoint structure:")
    print(f"  • Model state: model_state_dict")
    print(f"  • Optimizer state: optimizer_state_dict")
    print(f"  • Training metrics: {checkpoint['metrics']}")
    print(f"  • Configuration: {checkpoint['config']}")
    print(f"  • Timestamp: {checkpoint['timestamp']}")
    print(f"\n✓ Model versioning strategy defined")
    print(f"  • Location: {checkpoint_dir}")
    print(f"  • Naming: model_v{version}.pt")
    print(f"  • Format: PyTorch .pt format")


def demo_monitoring():
    """Demonstrate model monitoring setup"""
    print("\n" + "="*70)
    print("Model Monitoring Demo")
    print("="*70 + "\n")

    print("Production Monitoring Stack:")
    print()
    print("1. Metrics Collection:")
    print("   • Request latency (p50, p95, p99)")
    print("   • Throughput (requests/second)")
    print("   • Error rate")
    print("   • Model prediction distribution")
    print()
    print("2. Data Quality Monitoring:")
    print("   • Input drift detection")
    print("   • Missing value rates")
    print("   • Feature range validation")
    print("   • Outlier detection")
    print()
    print("3. Model Performance:")
    print("   • Real-time accuracy tracking")
    print("   • Prediction confidence distribution")
    print("   • Class imbalance monitoring")
    print()
    print("4. Infrastructure:")
    print("   • CPU/GPU utilization")
    print("   • Memory usage")
    print("   • Disk I/O")
    print("   • Network latency")
    print()
    print("5. Alerting Rules:")
    print("   • Latency > 100ms")
    print("   • Error rate > 1%")
    print("   • Accuracy drop > 5%")
    print("   • Input drift detected")

    print("\n✓ Monitoring infrastructure documented")
    print("  • Tools: Prometheus + Grafana")
    print("  • Dashboards: Performance, Data Quality, Infrastructure")
    print("  • Alerts: Configured thresholds")


def main():
    """Run all MLOps demos"""
    print("\n" + "="*70)
    print("Sports Injury Risk Prediction - MLOps Demo")
    print("="*70)

    demo_mlflow_tracking()
    demo_wandb_tracking()
    demo_pytorch_profiler()
    demo_model_versioning()
    demo_monitoring()

    print("\n" + "="*70)
    print("MLOps Demo Complete")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Experiment tracking with MLflow and W&B")
    print("  ✓ Performance profiling with PyTorch Profiler")
    print("  ✓ Model versioning and checkpointing strategy")
    print("  ✓ Production monitoring setup")
    print("\nNext Steps:")
    print("  • Set up MLflow server: mlflow ui")
    print("  • Configure W&B: wandb login")
    print("  • Deploy monitoring: docker-compose up prometheus grafana")
    print()


if __name__ == "__main__":
    main()
