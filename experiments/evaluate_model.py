"""
Model Evaluation Experiment Script
Comprehensive evaluation and analysis of trained models
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import joblib

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.trainer import BaseTrainer
from src.core.metrics import MetricsCalculator
from src.core.calibration import CalibrationManager
from src.core.interpret import InterpretabilityManager
from src.eval.drift import DriftDetector
from src.data.loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_data(model_path: str, data_config: Dict[str, Any]) -> tuple:
    """Load model and test data"""
    logger.info(f"Loading model from {model_path}")

    # Load model
    trainer = BaseTrainer().load(model_path)

    # Load test data
    if data_config['type'] == 'csv':
        from src.data.loader import CSVLoader
        data_loader = CSVLoader(data_config['path'])
        df = data_loader.load()
    else:
        raise ValueError(f"Unsupported data type: {data_config['type']}")

    # Separate features and labels
    target_column = data_config['target_column']
    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info(f"Loaded test data with shape: {X.shape}")

    return trainer, X, y


def comprehensive_evaluation(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path
) -> Dict[str, Any]:
    """Comprehensive model evaluation"""
    logger.info("Starting comprehensive model evaluation...")

    results = {}

    # Basic prediction
    y_pred = trainer.predict(X)
    y_pred_proba = trainer.predict_proba(X)[:, 1]

    # Calculate all metrics
    metrics_calc = MetricsCalculator(task_type="binary")
    metrics = metrics_calc.calculate_all_metrics(y, y_pred, y_pred_proba)
    results['metrics'] = metrics

    # Metrics under different thresholds
    threshold_metrics = metrics_calc.calculate_threshold_metrics(y, y_pred_proba)
    results['threshold_metrics'] = threshold_metrics

    # Save threshold analysis results
    threshold_path = output_dir / 'threshold_analysis.csv'
    threshold_metrics.to_csv(threshold_path, index=False)

    # Confusion matrix analysis
    cm_results = metrics_calc.get_confusion_matrix_metrics(y, y_pred)
    results['confusion_matrix'] = cm_results

    # Classification report
    classification_report = metrics_calc.get_classification_report(y, y_pred)
    results['classification_report'] = classification_report

    logger.info("Basic evaluation completed")
    logger.info(f"Test AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
    logger.info(f"Test AUC-PR: {metrics.get('auc_pr', 'N/A'):.4f}")
    logger.info(f"Test F1: {metrics.get('f1', 'N/A'):.4f}")

    return results


def calibration_analysis(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path
) -> Dict[str, Any]:
    """Probability calibration analysis"""
    logger.info("Analyzing model calibration...")

    calibration_mgr = CalibrationManager()
    y_pred_proba = trainer.predict_proba(X)[:, 1]

    # Evaluate current calibration performance
    calibration_eval = calibration_mgr.evaluate_calibration(y, y_pred_proba)

    # Compare different calibration methods
    calibration_comparison = calibration_mgr.compare_calibration_methods(
        trainer.model, X, y, methods=['platt', 'isotonic', 'beta']
    )

    results = {
        'original_calibration': calibration_eval,
        'calibration_comparison': calibration_comparison
    }

    # Save calibration comparison results
    calibration_path = output_dir / 'calibration_comparison.csv'
    calibration_comparison.to_csv(calibration_path, index=False)

    # Plot calibration curve
    plot_calibration_curve(y, y_pred_proba, output_dir)

    logger.info("Calibration analysis completed")
    logger.info(f"ECE: {calibration_eval['ece']:.4f}")
    logger.info(f"Brier Score: {calibration_eval['brier_score']:.4f}")

    return results


def interpretability_analysis(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """Interpretability analysis"""
    logger.info("Performing interpretability analysis...")

    interpreter = InterpretabilityManager(trainer.model, model_type='sklearn')
    interpreter.set_feature_names(feature_names)

    # Global interpretation
    global_explanation = interpreter.explain_global(X, method='shap', max_display=20)

    # Local interpretation (select a few samples)
    sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
    local_explanation = interpreter.explain_local(X, sample_indices, method='shap')

    # Generate complete report
    report = interpreter.generate_interpretation_report(
        X, y, output_dir=str(output_dir / 'interpretability')
    )

    results = {
        'global_explanation': global_explanation,
        'local_explanation': local_explanation,
        'report': report
    }

    logger.info("Interpretability analysis completed")

    return results


def fairness_analysis(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive_features: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """Fairness analysis"""
    logger.info("Performing fairness analysis...")

    results = {}
    y_pred = trainer.predict(X)
    y_pred_proba = trainer.predict_proba(X)[:, 1]

    for feature in sensitive_features:
        if feature not in X.columns:
            logger.warning(f"Sensitive feature {feature} not found in data")
            continue

        feature_results = {}
        unique_values = X[feature].unique()

        # Calculate metrics for each group
        group_metrics = []
        for value in unique_values:
            mask = X[feature] == value
            if mask.sum() == 0:
                continue

            group_y = y[mask]
            group_pred = y_pred[mask]
            group_pred_proba = y_pred_proba[mask]

            metrics_calc = MetricsCalculator(task_type="binary")
            group_metric = metrics_calc.calculate_all_metrics(
                group_y, group_pred, group_pred_proba
            )

            group_metric['group'] = value
            group_metric['size'] = mask.sum()
            group_metrics.append(group_metric)

        group_df = pd.DataFrame(group_metrics)
        feature_results['group_metrics'] = group_df

        # Calculate fairness metrics
        if len(group_df) >= 2:
            # Statistical Parity
            positive_rates = group_df['true_positives'] / group_df['size']
            feature_results['statistical_parity_diff'] = positive_rates.max() - positive_rates.min()

            # Equalized Odds
            tpr_diff = group_df['recall'].max() - group_df['recall'].min()
            feature_results['equalized_odds_diff'] = tpr_diff

        results[feature] = feature_results

        # Save results
        group_path = output_dir / f'fairness_{feature}.csv'
        group_df.to_csv(group_path, index=False)

    logger.info("Fairness analysis completed")

    return results


def robustness_analysis(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
) -> Dict[str, Any]:
    """Robustness analysis"""
    logger.info("Performing robustness analysis...")

    results = {}
    original_metrics = MetricsCalculator(task_type="binary").calculate_all_metrics(
        y, trainer.predict(X), trainer.predict_proba(X)[:, 1]
    )

    robustness_results = []

    for noise_level in noise_levels:
        # Add Gaussian noise
        X_noisy = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        noise = np.random.normal(0, noise_level, X_noisy[numerical_cols].shape)
        X_noisy[numerical_cols] += noise

        # Prediction
        try:
            y_pred_noisy = trainer.predict(X_noisy)
            y_pred_proba_noisy = trainer.predict_proba(X_noisy)[:, 1]

            # Calculate metrics
            noisy_metrics = MetricsCalculator(task_type="binary").calculate_all_metrics(
                y, y_pred_noisy, y_pred_proba_noisy
            )

            # Calculate performance degradation
            auc_drop = original_metrics.get('auc_roc', 0) - noisy_metrics.get('auc_roc', 0)
            f1_drop = original_metrics.get('f1', 0) - noisy_metrics.get('f1', 0)

            robustness_results.append({
                'noise_level': noise_level,
                'auc_roc': noisy_metrics.get('auc_roc', 0),
                'f1': noisy_metrics.get('f1', 0),
                'auc_drop': auc_drop,
                'f1_drop': f1_drop
            })

        except Exception as e:
            logger.warning(f"Error at noise level {noise_level}: {e}")

    results['noise_analysis'] = pd.DataFrame(robustness_results)

    # Save results
    robustness_path = output_dir / 'robustness_analysis.csv'
    results['noise_analysis'].to_csv(robustness_path, index=False)

    logger.info("Robustness analysis completed")

    return results


def generate_plots(
    trainer: BaseTrainer,
    X: pd.DataFrame,
    y: pd.Series,
    evaluation_results: Dict[str, Any],
    output_dir: Path
):
    """Generate evaluation plots"""
    logger.info("Generating evaluation plots...")

    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    y_pred_proba = trainer.predict_proba(X)[:, 1]
    y_pred = trainer.predict(X)

    # 1. ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {evaluation_results["metrics"]["auc_roc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. PR curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {evaluation_results["metrics"]["auc_pr"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Injury', 'Injury'],
                yticklabels=['No Injury', 'Injury'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Predicted probability distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(y_pred_proba[y == 0], bins=50, alpha=0.7, label='No Injury', density=True)
    plt.hist(y_pred_proba[y == 1], bins=50, alpha=0.7, label='Injury', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Predicted Probability Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([y_pred_proba[y == 0], y_pred_proba[y == 1]],
                labels=['No Injury', 'Injury'])
    plt.ylabel('Predicted Probability')
    plt.title('Predicted Probability by Class')

    plt.tight_layout()
    plt.savefig(plots_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Threshold analysis
    threshold_metrics = evaluation_results.get('threshold_metrics')
    if threshold_metrics is not None and not threshold_metrics.empty:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(threshold_metrics['threshold'], threshold_metrics['precision'], label='Precision')
        plt.plot(threshold_metrics['threshold'], threshold_metrics['recall'], label='Recall')
        plt.plot(threshold_metrics['threshold'], threshold_metrics['f1'], label='F1')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(threshold_metrics['threshold'], threshold_metrics['accuracy'])
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(threshold_metrics['threshold'], threshold_metrics.get('sensitivity', []),
                 label='Sensitivity (TPR)')
        plt.plot(threshold_metrics['threshold'], threshold_metrics.get('specificity', []),
                 label='Specificity (TNR)')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Sensitivity & Specificity vs Threshold')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(threshold_metrics['threshold'], threshold_metrics.get('ppv', []),
                 label='PPV')
        plt.plot(threshold_metrics['threshold'], threshold_metrics.get('npv', []),
                 label='NPV')
        plt.xlabel('Threshold')
        plt.ylabel('Predictive Value')
        plt.title('Predictive Values vs Threshold')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"Plots saved to {plots_dir}")


def plot_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_dir: Path):
    """Plot calibration curve"""
    from sklearn.calibration import calibration_curve

    plt.figure(figsize=(10, 8))

    # Main calibration curve
    plt.subplot(2, 2, 1)
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)

    # Histogram
    plt.subplot(2, 2, 2)
    plt.hist(y_pred_proba, bins=50, density=True, alpha=0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Predicted Probability Distribution')

    # Reliability diagram
    plt.subplot(2, 2, 3)
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(y_true[in_bin].mean())
            bin_confidences.append(y_pred_proba[in_bin].mean())

    plt.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, label='Accuracy')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title('Reliability Diagram')
    plt.legend()

    # ECE visualization
    plt.subplot(2, 2, 4)
    if bin_centers and bin_accuracies and bin_confidences:
        calibration_errors = [abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)]
        plt.bar(bin_centers, calibration_errors, width=0.08, alpha=0.7)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Calibration Error')
        plt.title('Calibration Error by Bin')

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_evaluation_report(
    results: Dict[str, Any],
    output_dir: Path
):
    """Save evaluation report"""
    logger.info("Saving evaluation report...")

    # Create summary report
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'metrics_summary': results.get('evaluation', {}).get('metrics', {}),
        'calibration_summary': results.get('calibration', {}).get('original_calibration', {}),
    }

    # Save JSON format report
    import json
    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Save text format report
    text_report_path = output_dir / 'evaluation_report.txt'
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("Sports Injury Risk Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Basic metrics
        metrics = report.get('metrics_summary', {})
        f.write("Model Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}\n")
        f.write(f"AUC-PR: {metrics.get('auc_pr', 'N/A'):.4f}\n")
        f.write(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"Precision: {metrics.get('precision', 'N/A'):.4f}\n")
        f.write(f"Recall: {metrics.get('recall', 'N/A'):.4f}\n")
        f.write(f"F1-Score: {metrics.get('f1', 'N/A'):.4f}\n")
        f.write(f"Brier Score: {metrics.get('brier_score', 'N/A'):.4f}\n\n")

        # Calibration metrics
        calibration = report.get('calibration_summary', {})
        f.write("Calibration Metrics:\n")
        f.write("-" * 30 + "\n")
        ece = calibration.get('ece', 'N/A')
        mce = calibration.get('mce', 'N/A')
        reliability = calibration.get('reliability', 'N/A')
        ece_str = f"{ece:.4f}" if isinstance(ece, (int, float)) else str(ece)
        mce_str = f"{mce:.4f}" if isinstance(mce, (int, float)) else str(mce)
        reliability_str = f"{reliability:.4f}" if isinstance(reliability, (int, float)) else str(reliability)
        f.write(f"ECE: {ece_str}\n")
        f.write(f"MCE: {mce_str}\n")
        f.write(f"Reliability: {reliability_str}\n\n")

    logger.info(f"Evaluation report saved to {output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate trained sports injury risk models')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    parser.add_argument('--output', type=str, help='Output directory (optional)')
    parser.add_argument('--analysis', type=str, nargs='+',
                       choices=['basic', 'calibration', 'interpretability', 'fairness', 'robustness'],
                       default=['basic', 'calibration'],
                       help='Types of analysis to perform')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'evaluation_results_{timestamp}')

    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir}")

    try:
        # Load model and data
        trainer, X, y = load_model_and_data(args.model, config['evaluation']['data'])
        feature_names = X.columns.tolist()

        results = {}

        # Basic evaluation
        if 'basic' in args.analysis:
            evaluation_results = comprehensive_evaluation(trainer, X, y, output_dir)
            results['evaluation'] = evaluation_results

            # Generate basic plots
            generate_plots(trainer, X, y, evaluation_results, output_dir)

        # Calibration analysis
        if 'calibration' in args.analysis:
            calibration_results = calibration_analysis(trainer, X, y, output_dir)
            results['calibration'] = calibration_results

        # Interpretability analysis
        if 'interpretability' in args.analysis:
            interpretability_results = interpretability_analysis(
                trainer, X, y, feature_names, output_dir
            )
            results['interpretability'] = interpretability_results

        # Fairness analysis
        if 'fairness' in args.analysis:
            sensitive_features = config.get('evaluation', {}).get('sensitive_features', [])
            if sensitive_features:
                fairness_results = fairness_analysis(trainer, X, y, sensitive_features, output_dir)
                results['fairness'] = fairness_results
            else:
                logger.warning("No sensitive features specified for fairness analysis")

        # Robustness analysis
        if 'robustness' in args.analysis:
            robustness_results = robustness_analysis(trainer, X, y, output_dir)
            results['robustness'] = robustness_results

        # Save evaluation report
        save_evaluation_report(results, output_dir)

        logger.info("Model evaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()