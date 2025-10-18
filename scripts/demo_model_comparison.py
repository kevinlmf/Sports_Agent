#!/usr/bin/env python3
"""
Model Comparison and Benchmarking
Compares performance of Traditional, Deep Learning, and Multimodal models
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import sys
sys.path.insert(0, '.')


def generate_training_data(n_samples=5000):
    """Generate synthetic training data"""
    np.random.seed(42)
    import pandas as pd

    # Features
    feature_names = [f'feature_{i}' for i in range(20)]
    X = pd.DataFrame(np.random.randn(n_samples, 20), columns=feature_names)

    # Complex target with non-linear relationships
    y = pd.Series(
        (X.iloc[:, 0] * X.iloc[:, 1] + X.iloc[:, 2]**2 +
         np.sin(X.iloc[:, 3]) + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_dl_data(X, y, seq_length=10):
    """Prepare sequential data for deep learning models"""
    import torch

    # Convert to numpy if DataFrame
    if hasattr(X, 'values'):
        X_np = X.values
    else:
        X_np = X

    if hasattr(y, 'values'):
        y_np = y.values
    else:
        y_np = y

    # Create sequences
    X_seq = []
    y_seq = []

    for i in range(len(X_np) - seq_length + 1):
        X_seq.append(X_np[i:i+seq_length])
        y_seq.append(y_np[i+seq_length-1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Convert to torch tensors
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.LongTensor(y_seq)

    return X_tensor, y_tensor


def benchmark_model(model_class, model_name, X_train, y_train, X_test, y_test, **kwargs):
    """Benchmark a single model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    results = {
        'model_name': model_name,
        'status': 'pending'
    }

    try:
        # Check if this is a DL model (has epochs parameter)
        is_dl_model = 'epochs' in kwargs

        # Initialize model
        print(f"  🔧 Initializing {model_name}...")

        # Extract training parameters for DL models
        if is_dl_model:
            epochs = kwargs.pop('epochs', 20)
            batch_size = kwargs.pop('batch_size', 32)
            learning_rate = kwargs.pop('learning_rate', 0.001)
            model = model_class(**kwargs)
        else:
            model = model_class(**kwargs)

        # Training
        print(f"  🏋️  Training...")
        start_time = time.time()

        if is_dl_model:
            # DL models need additional parameters
            import torch
            # Convert to numpy if needed
            if isinstance(X_train, torch.Tensor):
                X_train_np = X_train.numpy()
                y_train_np = y_train.numpy()
            else:
                X_train_np = X_train
                y_train_np = y_train

            model.fit(X_train_np, y_train_np,
                     epochs=epochs,
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     verbose=False)
        else:
            model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Inference
        print(f"  🔮 Running inference...")
        start_time = time.time()

        if is_dl_model:
            import torch
            # Convert to numpy if needed
            if isinstance(X_test, torch.Tensor):
                X_test_np = X_test.numpy()
                y_test_np = y_test.numpy()
            else:
                X_test_np = X_test
                y_test_np = y_test

            y_pred = model.predict(X_test_np)
        else:
            y_pred = model.predict(X_test)
            y_test_np = y_test

        inference_time = (time.time() - start_time) / len(y_test_np) * 1000  # ms per sample

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            if is_dl_model:
                y_pred_proba = model.predict_proba(X_test_np)[:, 1]
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred

        # Calculate metrics
        print(f"  📊 Computing metrics...")
        results.update({
            'status': 'success',
            'training_time': round(training_time, 2),
            'inference_time_ms': round(inference_time, 3),
            'auc_roc': round(roc_auc_score(y_test_np, y_pred_proba), 4),
            'f1_score': round(f1_score(y_test_np, y_pred), 4),
            'precision': round(precision_score(y_test_np, y_pred), 4),
            'recall': round(recall_score(y_test_np, y_pred), 4),
            'accuracy': round(np.mean(y_pred == y_test_np), 4)
        })

        # Model size estimation
        if is_dl_model:
            # For DL models, count actual parameters
            import torch
            if hasattr(model, 'model') and model.model is not None:
                n_params = sum(p.numel() for p in model.model.parameters())
            else:
                # Rough estimate based on hidden size
                hidden_size = kwargs.get('hidden_size', 64)
                n_features = X_train_np.shape[-1] if is_dl_model else X_train.shape[1]
                n_params = hidden_size * n_features * 10  # rough estimate
        else:
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            else:
                n_features = X_train.shape[1]

            # Estimate parameters
            if hasattr(model, 'coef_'):
                n_params = model.coef_.size
            elif hasattr(model, 'get_params'):
                try:
                    n_estimators = model.n_estimators if hasattr(model, 'n_estimators') else 1
                    n_params = n_estimators * n_features * 100  # rough estimate
                except:
                    n_params = n_features * 10
            else:
                n_params = n_features * 10

        results['n_parameters'] = n_params
        results['memory_mb'] = round(n_params * 4 / (1024 * 1024), 2)  # Assuming float32

        print(f"\n  ✅ {model_name} completed successfully!")
        print(f"     AUC-ROC: {results['auc_roc']:.4f}")
        print(f"     F1 Score: {results['f1_score']:.4f}")
        print(f"     Training Time: {results['training_time']:.2f}s")
        print(f"     Inference: {results['inference_time_ms']:.3f}ms/sample")

    except Exception as e:
        print(f"  ❌ {model_name} failed: {str(e)}")
        results['status'] = 'failed'
        results['error'] = str(e)

    return results


def run_full_benchmark(output_dir='results/model_comparison'):
    """Run comprehensive model benchmarking"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MODEL COMPARISON BENCHMARK")
    print("="*70)

    # Generate data
    print("\n📦 Generating training data...")
    X_train, X_test, y_train, y_test = generate_training_data(n_samples=5000)
    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Test samples: {len(X_test)}")
    print(f"   ✓ Features: {X_train.shape[1]}")

    # Model configurations
    models_to_test = []

    # Traditional Models
    print("\n🔍 Loading traditional models...")
    try:
        from src.methods.traditional.logistic_regression import LogisticInjuryPredictor
        models_to_test.append(('Logistic Regression', LogisticInjuryPredictor, {}))
        print("   ✓ Logistic Regression")
    except Exception as e:
        print(f"   ⚠️  Logistic Regression unavailable: {e}")

    try:
        from src.methods.traditional.random_forest import RandomForestInjuryPredictor
        models_to_test.append(('Random Forest', RandomForestInjuryPredictor, {'n_estimators': 100}))
        print("   ✓ Random Forest")
    except Exception as e:
        print(f"   ⚠️  Random Forest unavailable: {e}")

    try:
        from src.methods.traditional.xgboost_model import XGBoostInjuryPredictor
        models_to_test.append(('XGBoost', XGBoostInjuryPredictor, {'n_estimators': 100}))
        print("   ✓ XGBoost")
    except Exception as e:
        print(f"   ⚠️  XGBoost unavailable: {e}")

    # Deep Learning Models
    print("\n🧠 Loading deep learning models...")
    print("   ⚙️  DL models require special data preparation (sequences/tensors)")

    dl_models_available = []

    try:
        from src.methods.dl_seq.lstm_model import LSTMInjuryPredictor
        dl_models_available.append(('LSTM', LSTMInjuryPredictor, {
            'input_size': 20,
            'hidden_size': 64,
            'num_layers': 2,
            'num_classes': 2,
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001
        }))
        print("   ✓ LSTM")
    except Exception as e:
        print(f"   ⚠️  LSTM unavailable: {e}")

    try:
        from src.methods.dl_seq.gru_model import GRUInjuryPredictor
        dl_models_available.append(('GRU', GRUInjuryPredictor, {
            'input_size': 20,
            'hidden_size': 64,
            'num_layers': 2,
            'num_classes': 2,
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001
        }))
        print("   ✓ GRU")
    except Exception as e:
        print(f"   ⚠️  GRU unavailable: {e}")

    try:
        from src.methods.dl_seq.transformer_model import TransformerInjuryPredictor
        dl_models_available.append(('Transformer', TransformerInjuryPredictor, {
            'input_size': 20,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'num_classes': 2,
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001
        }))
        print("   ✓ Transformer")
    except Exception as e:
        print(f"   ⚠️  Transformer unavailable: {e}")

    # Benchmark all models
    print("\n" + "="*70)
    print("RUNNING BENCHMARKS")
    print("="*70)

    all_results = []

    # Benchmark traditional models
    print("\n--- Traditional Models ---")
    for model_name, model_class, kwargs in models_to_test:
        result = benchmark_model(model_class, model_name, X_train, y_train, X_test, y_test, **kwargs)
        all_results.append(result)

    # Benchmark deep learning models with sequence data
    if dl_models_available:
        print("\n--- Deep Learning Models ---")
        print("📦 Preparing sequential data for DL models...")

        # Prepare sequence data
        X_train_seq, y_train_seq = prepare_dl_data(X_train, y_train, seq_length=10)
        X_test_seq, y_test_seq = prepare_dl_data(X_test, y_test, seq_length=10)

        print(f"   ✓ Train sequences: {len(X_train_seq)}")
        print(f"   ✓ Test sequences: {len(X_test_seq)}")
        print(f"   ✓ Sequence shape: {X_train_seq.shape}")

        for model_name, model_class, kwargs in dl_models_available:
            result = benchmark_model(model_class, model_name, X_train_seq, y_train_seq,
                                   X_test_seq, y_test_seq, **kwargs)
            all_results.append(result)

    # Filter successful results
    successful_results = [r for r in all_results if r['status'] == 'success']

    if not successful_results:
        print("\n❌ No models completed successfully. Cannot generate comparison.")
        return

    # Save results
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to: {output_dir / 'benchmark_results.json'}")

    # Generate visualizations
    print("\n📊 Generating comparison visualizations...")
    create_comparison_plots(successful_results, output_dir)

    # Print summary table
    print_summary_table(successful_results)

    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   • benchmark_results.json")
    print(f"   • performance_comparison.png")
    print(f"   • efficiency_comparison.png")
    print(f"   • radar_comparison.png")


def create_comparison_plots(results, output_dir):
    """Create comprehensive comparison visualizations"""
    sns.set_style("whitegrid")

    # 1. Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    model_names = [r['model_name'] for r in results]
    metrics = ['auc_roc', 'f1_score', 'precision', 'recall']
    metric_labels = ['AUC-ROC', 'F1 Score', 'Precision', 'Recall']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        values = [r[metric] for r in results]

        bars = ax.barh(model_names, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(results))))
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Target: 0.80')

        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', fontsize=9)

        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'performance_comparison.png'}")
    plt.close()

    # 2. Efficiency Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Model Efficiency Comparison', fontsize=16, fontweight='bold')

    # Training time
    ax = axes[0]
    training_times = [r['training_time'] for r in results]
    bars = ax.bar(model_names, training_times, color=plt.cm.plasma(np.linspace(0.3, 0.9, len(results))))
    ax.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Training Speed', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, training_times):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(training_times)*0.02,
               f'{value:.2f}s', ha='center', fontsize=9)

    # Inference time
    ax = axes[1]
    inference_times = [r['inference_time_ms'] for r in results]
    bars = ax.bar(model_names, inference_times, color=plt.cm.plasma(np.linspace(0.3, 0.9, len(results))))
    ax.set_ylabel('Inference Time (ms/sample)', fontsize=11, fontweight='bold')
    ax.set_title('Inference Speed', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, inference_times):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(inference_times)*0.02,
               f'{value:.3f}ms', ha='center', fontsize=9)

    # Memory usage
    ax = axes[2]
    memory = [r['memory_mb'] for r in results]
    bars = ax.bar(model_names, memory, color=plt.cm.plasma(np.linspace(0.3, 0.9, len(results))))
    ax.set_ylabel('Memory (MB)', fontsize=11, fontweight='bold')
    ax.set_title('Memory Footprint', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, memory):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(memory)*0.02,
               f'{value:.2f}MB', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'efficiency_comparison.png'}")
    plt.close()

    # 3. Radar Chart (Multi-dimensional comparison)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    categories = ['AUC-ROC', 'F1 Score', 'Precision', 'Recall', 'Speed\n(1/training_time)']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results)))

    for idx, result in enumerate(results):
        # Normalize training speed (lower is better, so invert)
        max_train_time = max([r['training_time'] for r in results])
        speed_score = 1 - (result['training_time'] / max_train_time)

        values = [
            result['auc_roc'],
            result['f1_score'],
            result['precision'],
            result['recall'],
            speed_score
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Multi-Dimensional Model Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'radar_comparison.png'}")
    plt.close()


def print_summary_table(results):
    """Print formatted summary table"""
    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)

    # Header
    print(f"{'Model':<20} {'AUC-ROC':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} "
          f"{'Train(s)':<10} {'Infer(ms)':<11} {'Memory(MB)':<12}")
    print("-" * 100)

    # Sort by AUC-ROC
    sorted_results = sorted(results, key=lambda x: x['auc_roc'], reverse=True)

    for result in sorted_results:
        print(f"{result['model_name']:<20} "
              f"{result['auc_roc']:<10.4f} "
              f"{result['f1_score']:<8.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<8.4f} "
              f"{result['training_time']:<10.2f} "
              f"{result['inference_time_ms']:<11.3f} "
              f"{result['memory_mb']:<12.2f}")

    print("="*100)

    # Best performers
    best_auc = max(results, key=lambda x: x['auc_roc'])
    fastest_train = min(results, key=lambda x: x['training_time'])
    fastest_infer = min(results, key=lambda x: x['inference_time_ms'])

    print(f"\n🏆 Best Performance: {best_auc['model_name']} (AUC-ROC: {best_auc['auc_roc']:.4f})")
    print(f"⚡ Fastest Training: {fastest_train['model_name']} ({fastest_train['training_time']:.2f}s)")
    print(f"🚀 Fastest Inference: {fastest_infer['model_name']} ({fastest_infer['inference_time_ms']:.3f}ms/sample)")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("SPORTS INJURY RISK PREDICTION - MODEL COMPARISON")
    print("="*70)

    run_full_benchmark()

    print("\n✨ Comparison complete! Check results/model_comparison/ for details")


if __name__ == '__main__':
    main()
