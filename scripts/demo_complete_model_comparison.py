#!/usr/bin/env python3
"""
Complete Model Comparison Benchmark
Compares ALL model types: Traditional, Deep Learning (LSTM/GRU/Transformer), and Multimodal (VLM)

This script provides a comprehensive comparison with actual model training and evaluation,
not just static benchmarks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import sys
import warnings
import logging
from typing import Dict, Any, List, Tuple

# Prevent multiprocessing deadlocks with transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data Generation Functions
# ============================================================================

def generate_tabular_data(n_samples=5000, n_features=20, random_state=42):
    """Generate synthetic tabular training data"""
    np.random.seed(random_state)

    # Features
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)

    # Complex target with non-linear relationships
    y = pd.Series(
        (X.iloc[:, 0] * X.iloc[:, 1] + X.iloc[:, 2]**2 +
         np.sin(X.iloc[:, 3]) + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    )

    return X, y


def generate_sequence_data(X_tabular, y, seq_len=10, random_state=42):
    """
    Convert tabular data to sequence data for LSTM/GRU/Transformer

    Args:
        X_tabular: Tabular features (n_samples, n_features)
        y: Labels (n_samples,)
        seq_len: Sequence length

    Returns:
        X_seq: (n_samples, seq_len, n_features)
        y: (n_samples,)
    """
    np.random.seed(random_state)

    n_samples, n_features = X_tabular.shape
    X_seq = np.zeros((n_samples, seq_len, n_features))

    # Convert to numpy if it's a DataFrame
    if isinstance(X_tabular, pd.DataFrame):
        X_tabular = X_tabular.values
    if isinstance(y, pd.Series):
        y = y.values

    # Create sequences by adding temporal variation
    for i in range(n_samples):
        base_features = X_tabular[i]
        # Add temporal evolution with small noise
        for t in range(seq_len):
            noise = np.random.randn(n_features) * 0.1
            trend = base_features * (1 + t * 0.05)  # Small trend over time
            X_seq[i, t, :] = trend + noise

    return X_seq, y


def generate_multimodal_data(X_tabular, y, output_dir='data/synthetic_multimodal', random_state=42):
    """
    Generate synthetic multimodal data (tabular + images + text)

    Args:
        X_tabular: Tabular features
        y: Labels
        output_dir: Directory to save synthetic data

    Returns:
        Paths to generated data files
    """
    np.random.seed(random_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(X_tabular)

    # 1. Save tabular data with athlete IDs
    tabular_df = X_tabular.copy()
    tabular_df['athlete_id'] = [f'athlete_{i:04d}' for i in range(n_samples)]
    tabular_df['injury'] = y.values

    # 2. Generate synthetic text data (medical notes)
    text_templates = [
        "Athlete shows good recovery progress. No pain reported during training.",
        "Minor discomfort in lower back. Recommended rest and physiotherapy.",
        "High workload this week. Monitoring for signs of fatigue.",
        "Previous injury fully healed. Cleared for full training.",
        "Reported muscle tightness. Preventive treatment initiated."
    ]

    text_data = []
    for i in range(n_samples):
        athlete_id = f'athlete_{i:04d}'
        # Select text based on injury status
        if y.iloc[i] == 1:
            text = np.random.choice(text_templates[1:3])  # More concerning notes
        else:
            text = np.random.choice(text_templates)
        text_data.append({'athlete_id': athlete_id, 'text': text})

    text_df = pd.DataFrame(text_data)

    # 3. Generate synthetic images (random noise images as placeholders)
    # In real scenarios, these would be biomechanical images, posture analysis, etc.
    image_dir = output_dir / 'images'
    image_dir.mkdir(exist_ok=True)

    # We'll just save image paths, not generate actual images to save space
    # The VLM loader will handle missing images gracefully

    # Save data files
    train_tabular = output_dir / 'train_tabular.csv'
    train_text = output_dir / 'train_text.csv'

    # Split data
    train_idx = list(range(int(0.8 * n_samples)))
    val_idx = list(range(int(0.8 * n_samples), n_samples))

    tabular_df.iloc[train_idx].to_csv(train_tabular, index=False)
    tabular_df.iloc[val_idx].to_csv(output_dir / 'val_tabular.csv', index=False)

    text_df.iloc[train_idx].to_csv(train_text, index=False)
    text_df.iloc[val_idx].to_csv(output_dir / 'val_text.csv', index=False)

    logger.info(f"Generated multimodal data in {output_dir}")

    return {
        'train_tabular': str(train_tabular),
        'val_tabular': str(output_dir / 'val_tabular.csv'),
        'train_text': str(train_text),
        'val_text': str(output_dir / 'val_text.csv'),
        'image_dir': None  # No actual images for simplicity
    }


# ============================================================================
# Benchmark Functions for Each Model Type
# ============================================================================

def benchmark_traditional_model(model_class, model_name, X_train, y_train, X_test, y_test, **kwargs):
    """Benchmark traditional ML models (LR, RF, XGBoost)"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'='*60}")

    results = {
        'model_name': model_name,
        'model_type': 'traditional',
        'status': 'pending'
    }

    try:
        # Initialize model
        logger.info(f"  🔧 Initializing {model_name}...")
        model = model_class(**kwargs)

        # Training
        logger.info(f"  🏋️  Training...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Inference
        logger.info(f"  🔮 Running inference...")
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred

        # Calculate metrics
        logger.info(f"  📊 Computing metrics...")
        results.update({
            'status': 'success',
            'training_time': round(training_time, 2),
            'inference_time_ms': round(inference_time, 3),
            'auc_roc': round(roc_auc_score(y_test, y_pred_proba), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'accuracy': round(np.mean(y_pred == y_test), 4)
        })

        # Model size estimation
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            n_features = X_train.shape[1]

        if hasattr(model, 'coef_'):
            n_params = model.coef_.size
        elif hasattr(model, 'n_estimators'):
            n_params = model.n_estimators * n_features * 100
        else:
            n_params = n_features * 10

        results['n_parameters'] = n_params
        results['memory_mb'] = round(n_params * 4 / (1024 * 1024), 2)

        logger.info(f"\n  ✅ {model_name} completed!")
        logger.info(f"     AUC-ROC: {results['auc_roc']:.4f}")
        logger.info(f"     Training Time: {results['training_time']:.2f}s")

    except Exception as e:
        logger.error(f"  ❌ {model_name} failed: {str(e)}")
        results['status'] = 'failed'
        results['error'] = str(e)

    return results


def benchmark_sequence_model(model_class, model_name, X_train, y_train, X_test, y_test, **kwargs):
    """Benchmark sequence models (LSTM, GRU, Transformer)"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'='*60}")

    results = {
        'model_name': model_name,
        'model_type': 'deep_learning',
        'status': 'pending'
    }

    try:
        # Initialize model
        logger.info(f"  🔧 Initializing {model_name}...")
        model = model_class(**kwargs)

        # Training
        logger.info(f"  🏋️  Training...")
        start_time = time.time()

        # Train with validation data
        val_split = int(0.8 * len(X_train))
        X_train_sub = X_train[:val_split]
        y_train_sub = y_train[:val_split]
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]

        train_results = model.fit(
            X_train_sub, y_train_sub,
            validation_data=(X_val, y_val),
            epochs=20,  # Reduced for demo
            batch_size=32,
            verbose=False
        )

        training_time = time.time() - start_time

        # Inference
        logger.info(f"  🔮 Running inference...")
        start_time = time.time()
        y_pred_proba = model.predict_proba(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000

        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        logger.info(f"  📊 Computing metrics...")
        results.update({
            'status': 'success',
            'training_time': round(training_time, 2),
            'inference_time_ms': round(inference_time, 3),
            'auc_roc': round(roc_auc_score(y_test, y_pred_proba), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'accuracy': round(np.mean(y_pred == y_test), 4)
        })

        # Estimate model parameters
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            n_params = sum(p.numel() for p in model.model.parameters())
        else:
            # Rough estimation for LSTM
            n_params = kwargs.get('hidden_size', 128) ** 2 * kwargs.get('num_layers', 2) * 4

        results['n_parameters'] = n_params
        results['memory_mb'] = round(n_params * 4 / (1024 * 1024), 2)

        logger.info(f"\n  ✅ {model_name} completed!")
        logger.info(f"     AUC-ROC: {results['auc_roc']:.4f}")
        logger.info(f"     Parameters: {n_params:,}")
        logger.info(f"     Training Time: {results['training_time']:.2f}s")

    except Exception as e:
        logger.error(f"  ❌ {model_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        results['status'] = 'failed'
        results['error'] = str(e)

    return results


def benchmark_vlm_model(model_name, data_paths, n_features, epochs=5, use_lora=False):
    """Benchmark Vision-Language Multimodal models"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'='*60}")

    results = {
        'model_name': model_name,
        'model_type': 'multimodal',
        'status': 'pending'
    }

    try:
        from src.models.multimodal.vision_language_risk_model import VisionLanguageRiskModel
        from src.data_pipeline.multimodal_loader import create_multimodal_loaders

        # Check if transformers library is available
        try:
            import transformers
        except ImportError:
            logger.warning("  ⚠️  Transformers library not installed. Skipping VLM models.")
            results['status'] = 'skipped'
            results['error'] = 'transformers library not installed'
            return results

        logger.info(f"  🔧 Initializing {model_name}...")
        logger.info(f"     Creating multimodal data loaders...")

        # Create data loaders
        train_loader, val_loader, _ = create_multimodal_loaders(
            train_tabular=data_paths['train_tabular'],
            val_tabular=data_paths['val_tabular'],
            train_text=data_paths['train_text'],
            val_text=data_paths['val_text'],
            train_image_dir=data_paths['image_dir'],
            val_image_dir=data_paths['image_dir'],
            batch_size=16,  # Smaller batch for VLM
            vision_model='clip',
            text_model='bert'
        )

        # Initialize model
        logger.info(f"     Initializing VLM model (use_lora={use_lora})...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = VisionLanguageRiskModel(
            tabular_input_dim=n_features,
            vision_encoder='clip',
            text_encoder='bert',
            embed_dim=768,
            use_lora=use_lora,
            lora_r=16 if use_lora else None,
            freeze_vision=True,
            freeze_text=True,
            task='classification',
            num_classes=2
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"     Total parameters: {total_params:,}")
        logger.info(f"     Trainable parameters: {trainable_params:,}")

        # Training
        logger.info(f"  🏋️  Training for {epochs} epochs...")
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.001
        )
        criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                tabular = batch['tabular'].to(device)
                labels = batch['label'].to(device)

                # Optional: use text if available
                text_input_ids = batch.get('text_input_ids')
                text_attention_mask = batch.get('text_attention_mask')

                if text_input_ids is not None:
                    text_input_ids = text_input_ids.to(device)
                    text_attention_mask = text_attention_mask.to(device)

                optimizer.zero_grad()
                outputs = model(
                    tabular=tabular,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask
                )

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            if epoch % 2 == 0:
                logger.info(f"     Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")

        training_time = time.time() - start_time

        # Evaluation
        logger.info(f"  🔮 Running evaluation...")
        model.eval()
        all_preds = []
        all_labels = []

        inference_start = time.time()

        with torch.no_grad():
            for batch in val_loader:
                tabular = batch['tabular'].to(device)
                labels = batch['label'].to(device)

                text_input_ids = batch.get('text_input_ids')
                text_attention_mask = batch.get('text_attention_mask')

                if text_input_ids is not None:
                    text_input_ids = text_input_ids.to(device)
                    text_attention_mask = text_attention_mask.to(device)

                outputs = model(
                    tabular=tabular,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask
                )

                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        total_inference_time = time.time() - inference_start
        inference_time_ms = (total_inference_time / len(all_labels)) * 1000

        # Calculate metrics
        logger.info(f"  📊 Computing metrics...")
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        binary_preds = (all_preds > 0.5).astype(int)

        results.update({
            'status': 'success',
            'training_time': round(training_time, 2),
            'inference_time_ms': round(inference_time_ms, 3),
            'auc_roc': round(roc_auc_score(all_labels, all_preds), 4),
            'f1_score': round(f1_score(all_labels, binary_preds), 4),
            'precision': round(precision_score(all_labels, binary_preds), 4),
            'recall': round(recall_score(all_labels, binary_preds), 4),
            'accuracy': round(np.mean(binary_preds == all_labels), 4),
            'n_parameters': trainable_params if use_lora else total_params,
            'memory_mb': round((total_params * 4) / (1024 * 1024), 2)
        })

        logger.info(f"\n  ✅ {model_name} completed!")
        logger.info(f"     AUC-ROC: {results['auc_roc']:.4f}")
        logger.info(f"     Parameters: {trainable_params:,} trainable / {total_params:,} total")

    except Exception as e:
        logger.error(f"  ❌ {model_name} failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        results['status'] = 'failed'
        results['error'] = str(e)

    return results


# ============================================================================
# Main Benchmark Orchestration
# ============================================================================

def run_complete_benchmark(output_dir='results/complete_model_comparison'):
    """Run comprehensive benchmark of all model types"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("COMPLETE MODEL COMPARISON BENCHMARK")
    print("Traditional + Deep Learning + Multimodal VLM")
    print("="*80)

    # ========================================================================
    # 1. Generate Data
    # ========================================================================
    print("\n📦 Generating training data...")
    X_tab, y = generate_tabular_data(n_samples=5000, n_features=20)
    X_train_tab, X_test_tab, y_train, y_test = train_test_split(
        X_tab, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_tab_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_tab),
        columns=X_train_tab.columns
    )
    X_test_tab_scaled = pd.DataFrame(
        scaler.transform(X_test_tab),
        columns=X_test_tab.columns
    )

    print(f"   ✓ Tabular data: {len(X_train_tab)} train, {len(X_test_tab)} test")
    print(f"   ✓ Features: {X_train_tab.shape[1]}")
    print(f"   ✓ Class distribution: {dict(y_train.value_counts())}")

    # Generate sequence data for DL models
    print(f"\n📦 Generating sequence data for DL models...")
    X_train_seq, y_train_seq = generate_sequence_data(X_train_tab_scaled, y_train, seq_len=10)
    X_test_seq, y_test_seq = generate_sequence_data(X_test_tab_scaled, y_test, seq_len=10)
    print(f"   ✓ Sequence data shape: {X_train_seq.shape}")

    # Generate multimodal data for VLM
    print(f"\n📦 Generating multimodal data for VLM...")
    multimodal_paths = generate_multimodal_data(X_tab, y)
    print(f"   ✓ Multimodal data generated")

    # ========================================================================
    # 2. Benchmark Traditional Models
    # ========================================================================
    print("\n" + "="*80)
    print("PART 1: TRADITIONAL MODELS")
    print("="*80)

    all_results = []

    try:
        from src.methods.traditional.logistic_regression import LogisticInjuryPredictor
        result = benchmark_traditional_model(
            LogisticInjuryPredictor, 'Logistic Regression',
            X_train_tab_scaled, y_train, X_test_tab_scaled, y_test
        )
        all_results.append(result)
    except Exception as e:
        logger.warning(f"Logistic Regression unavailable: {e}")

    try:
        from src.methods.traditional.random_forest import RandomForestInjuryPredictor
        result = benchmark_traditional_model(
            RandomForestInjuryPredictor, 'Random Forest',
            X_train_tab_scaled, y_train, X_test_tab_scaled, y_test,
            n_estimators=100
        )
        all_results.append(result)
    except Exception as e:
        logger.warning(f"Random Forest unavailable: {e}")

    try:
        from src.methods.traditional.xgboost_model import XGBoostInjuryPredictor
        result = benchmark_traditional_model(
            XGBoostInjuryPredictor, 'XGBoost',
            X_train_tab_scaled, y_train, X_test_tab_scaled, y_test,
            n_estimators=100
        )
        all_results.append(result)
    except Exception as e:
        logger.warning(f"XGBoost unavailable: {e}")

    # ========================================================================
    # 3. Benchmark Deep Learning Sequence Models
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: DEEP LEARNING SEQUENCE MODELS")
    print("="*80)

    try:
        from src.methods.dl_seq.lstm_model import LSTMInjuryPredictor
        result = benchmark_sequence_model(
            LSTMInjuryPredictor, 'LSTM',
            X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            backend='pytorch'
        )
        all_results.append(result)
    except Exception as e:
        logger.warning(f"LSTM unavailable: {e}")

    try:
        from src.methods.dl_seq.gru_model import GRUInjuryPredictor
        result = benchmark_sequence_model(
            GRUInjuryPredictor, 'GRU',
            X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            hidden_size=128,
            num_layers=2,
            bidirectional=True
        )
        all_results.append(result)
    except Exception as e:
        logger.warning(f"GRU unavailable: {e}")

    try:
        from src.methods.dl_seq.transformer_model import TransformerInjuryPredictor
        result = benchmark_sequence_model(
            TransformerInjuryPredictor, 'Transformer',
            X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            d_model=128,
            nhead=4,
            num_encoder_layers=2
        )
        all_results.append(result)
    except Exception as e:
        logger.warning(f"Transformer unavailable: {e}")

    # ========================================================================
    # 4. Benchmark Vision-Language Multimodal Models
    # ========================================================================
    print("\n" + "="*80)
    print("PART 3: VISION-LANGUAGE MULTIMODAL MODELS")
    print("="*80)

    # VLM Full (all parameters)
    result = benchmark_vlm_model(
        'VLM (Full)',
        multimodal_paths,
        n_features=X_train_tab.shape[1],
        epochs=5,
        use_lora=False
    )
    all_results.append(result)

    # VLM + LoRA (parameter-efficient)
    result = benchmark_vlm_model(
        'VLM + LoRA',
        multimodal_paths,
        n_features=X_train_tab.shape[1],
        epochs=5,
        use_lora=True
    )
    all_results.append(result)

    # ========================================================================
    # 5. Save Results and Generate Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING RESULTS")
    print("="*80)

    # Filter successful results
    successful_results = [r for r in all_results if r['status'] == 'success']

    if not successful_results:
        logger.error("❌ No models completed successfully!")
        return

    # Save results
    with open(output_dir / 'complete_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to: {output_dir / 'complete_benchmark_results.json'}")

    # Generate visualizations
    print("\n📊 Generating comparison visualizations...")
    create_complete_comparison_plots(successful_results, output_dir)

    # Print summary table
    print_complete_summary_table(successful_results)

    print("\n" + "="*80)
    print("✅ COMPLETE BENCHMARK FINISHED")
    print("="*80)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   • complete_benchmark_results.json")
    print(f"   • performance_by_type.png")
    print(f"   • efficiency_comparison.png")
    print(f"   • comprehensive_comparison.png")


# ============================================================================
# Visualization Functions
# ============================================================================

def create_complete_comparison_plots(results: List[Dict], output_dir: Path):
    """Create comprehensive comparison visualizations"""
    sns.set_style("whitegrid")

    # 1. Performance by Model Type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison (All Types)', fontsize=16, fontweight='bold')

    model_names = [r['model_name'] for r in results]
    model_types = [r['model_type'] for r in results]

    # Color by model type
    type_colors = {'traditional': '#1f77b4', 'deep_learning': '#ff7f0e', 'multimodal': '#2ca02c'}
    colors = [type_colors[t] for t in model_types]

    metrics = ['auc_roc', 'f1_score', 'precision', 'recall']
    metric_labels = ['AUC-ROC', 'F1 Score', 'Precision', 'Recall']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        values = [r[metric] for r in results]

        bars = ax.barh(model_names, values, color=colors)
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Target: 0.80')

        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', fontsize=9)

        if idx == 0:
            ax.legend(fontsize=9)

    # Add legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=type_colors['traditional'], label='Traditional'),
        Patch(facecolor=type_colors['deep_learning'], label='Deep Learning'),
        Patch(facecolor=type_colors['multimodal'], label='Multimodal')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_type.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: performance_by_type.png")
    plt.close()

    # 2. Efficiency Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Efficiency Comparison', fontsize=16, fontweight='bold')

    # Training time
    ax = axes[0]
    training_times = [r['training_time'] for r in results]
    bars = ax.bar(range(len(model_names)), training_times, color=colors)
    ax.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Training Speed', fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    for bar, value in zip(bars, training_times):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(training_times)*0.02,
               f'{value:.1f}s', ha='center', fontsize=9)

    # Inference time
    ax = axes[1]
    inference_times = [r['inference_time_ms'] for r in results]
    bars = ax.bar(range(len(model_names)), inference_times, color=colors)
    ax.set_ylabel('Inference Time (ms/sample)', fontsize=11, fontweight='bold')
    ax.set_title('Inference Speed', fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    for bar, value in zip(bars, inference_times):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(inference_times)*0.02,
               f'{value:.2f}ms', ha='center', fontsize=9)

    # Parameters
    ax = axes[2]
    params = [r['n_parameters'] / 1e6 for r in results]  # in millions
    bars = ax.bar(range(len(model_names)), params, color=colors)
    ax.set_ylabel('Parameters (Millions)', fontsize=11, fontweight='bold')
    ax.set_title('Model Size', fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    for bar, value in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(params)*0.02,
               f'{value:.1f}M', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: efficiency_comparison.png")
    plt.close()

    # 3. Comprehensive scatter plot: Performance vs Efficiency
    fig, ax = plt.subplots(figsize=(12, 8))

    auc_scores = [r['auc_roc'] for r in results]
    sizes = [r['n_parameters'] / 1e3 for r in results]  # Size of bubble

    scatter = ax.scatter(inference_times, auc_scores, s=sizes, c=colors, alpha=0.6, edgecolors='black')

    for i, name in enumerate(model_names):
        ax.annotate(name, (inference_times[i], auc_scores[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel('Inference Time (ms/sample)', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance vs Inference Speed\n(Bubble size = parameter count)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend for model types
    legend_elements = [
        Patch(facecolor=type_colors['traditional'], label='Traditional'),
        Patch(facecolor=type_colors['deep_learning'], label='Deep Learning'),
        Patch(facecolor=type_colors['multimodal'], label='Multimodal')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: comprehensive_comparison.png")
    plt.close()


def print_complete_summary_table(results: List[Dict]):
    """Print formatted summary table with all models"""
    print("\n" + "="*120)
    print("COMPLETE MODEL COMPARISON SUMMARY")
    print("="*120)

    # Header
    print(f"{'Model':<25} {'Type':<15} {'AUC-ROC':<10} {'F1':<8} {'Params':<12} "
          f"{'Train(s)':<10} {'Infer(ms)':<11} {'Memory(MB)':<12}")
    print("-" * 120)

    # Sort by model type then AUC
    type_order = {'traditional': 0, 'deep_learning': 1, 'multimodal': 2}
    sorted_results = sorted(results, key=lambda x: (type_order[x['model_type']], -x['auc_roc']))

    for result in sorted_results:
        params_str = f"{result['n_parameters']/1e6:.1f}M" if result['n_parameters'] > 1e6 else f"{result['n_parameters']/1e3:.1f}K"
        print(f"{result['model_name']:<25} "
              f"{result['model_type']:<15} "
              f"{result['auc_roc']:<10.4f} "
              f"{result['f1_score']:<8.4f} "
              f"{params_str:<12} "
              f"{result['training_time']:<10.2f} "
              f"{result['inference_time_ms']:<11.3f} "
              f"{result['memory_mb']:<12.2f}")

    print("="*120)

    # Best performers by category
    traditional = [r for r in results if r['model_type'] == 'traditional']
    dl = [r for r in results if r['model_type'] == 'deep_learning']
    multimodal = [r for r in results if r['model_type'] == 'multimodal']

    print(f"\n🏆 Best Performers by Category:")

    if traditional:
        best_trad = max(traditional, key=lambda x: x['auc_roc'])
        print(f"   Traditional: {best_trad['model_name']} (AUC: {best_trad['auc_roc']:.4f})")

    if dl:
        best_dl = max(dl, key=lambda x: x['auc_roc'])
        print(f"   Deep Learning: {best_dl['model_name']} (AUC: {best_dl['auc_roc']:.4f})")

    if multimodal:
        best_mm = max(multimodal, key=lambda x: x['auc_roc'])
        print(f"   Multimodal: {best_mm['model_name']} (AUC: {best_mm['auc_roc']:.4f})")

    # Overall best
    best_overall = max(results, key=lambda x: x['auc_roc'])
    print(f"\n🥇 Overall Best: {best_overall['model_name']} (AUC: {best_overall['auc_roc']:.4f})")

    # Fastest
    fastest = min(results, key=lambda x: x['inference_time_ms'])
    print(f"⚡ Fastest Inference: {fastest['model_name']} ({fastest['inference_time_ms']:.3f}ms/sample)")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("SPORTS INJURY RISK PREDICTION")
    print("Complete Model Comparison: Traditional + DL + Multimodal")
    print("="*80)

    run_complete_benchmark()

    print("\n✨ Benchmark complete! Check results/complete_model_comparison/ for details")
    print("\n📊 Generated files:")
    print("   • complete_benchmark_results.json - Raw benchmark data")
    print("   • performance_by_type.png - Performance metrics by model type")
    print("   • efficiency_comparison.png - Training/inference time and model size")
    print("   • comprehensive_comparison.png - Performance vs efficiency scatter plot")


if __name__ == '__main__':
    main()
