"""
模型比较实验脚本
比较多个训练好的模型的性能
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
import joblib

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.trainer import BaseTrainer
from src.core.metrics import MetricsCalculator
from src.data.loader import DataLoader
from evaluate_model import load_model_and_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_multiple_models(model_paths: List[str], model_names: List[str] = None) -> Dict[str, BaseTrainer]:
    """加载多个模型"""
    models = {}

    if model_names is None:
        model_names = [Path(path).stem for path in model_paths]

    for path, name in zip(model_paths, model_names):
        try:
            logger.info(f"Loading model: {name} from {path}")
            trainer = BaseTrainer().load(path)
            models[name] = trainer
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")

    return models


def compare_basic_metrics(
    models: Dict[str, BaseTrainer],
    X: pd.DataFrame,
    y: pd.Series
) -> pd.DataFrame:
    """比较基础指标"""
    logger.info("Comparing basic metrics...")

    results = []
    metrics_calc = MetricsCalculator(task_type="binary")

    for name, trainer in models.items():
        try:
            # 预测
            y_pred = trainer.predict(X)
            y_pred_proba = trainer.predict_proba(X)[:, 1]

            # 计算指标
            metrics = metrics_calc.calculate_all_metrics(y, y_pred, y_pred_proba)
            metrics['model'] = name
            results.append(metrics)

        except Exception as e:
            logger.error(f"Error evaluating model {name}: {e}")

    return pd.DataFrame(results)


def statistical_significance_test(
    models: Dict[str, BaseTrainer],
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """统计显著性检验"""
    logger.info("Performing statistical significance tests...")

    from scipy import stats

    # 获取所有模型的AUC分数
    model_scores = {}
    for name, trainer in models.items():
        y_pred_proba = trainer.predict_proba(X)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y, y_pred_proba)
        model_scores[name] = auc_score

    # Bootstrap重采样进行显著性检验
    bootstrap_results = {}

    for name1, trainer1 in models.items():
        bootstrap_results[name1] = {}

        for name2, trainer2 in models.items():
            if name1 >= name2:  # 避免重复比较
                continue

            # Bootstrap重采样
            bootstrap_diffs = []

            for _ in range(n_bootstrap):
                # 有放回采样
                indices = np.random.choice(len(X), len(X), replace=True)
                X_boot = X.iloc[indices]
                y_boot = y.iloc[indices]

                # 计算两个模型的AUC差异
                try:
                    y_pred1 = trainer1.predict_proba(X_boot)[:, 1]
                    y_pred2 = trainer2.predict_proba(X_boot)[:, 1]

                    auc1 = roc_auc_score(y_boot, y_pred1)
                    auc2 = roc_auc_score(y_boot, y_pred2)

                    bootstrap_diffs.append(auc1 - auc2)
                except:
                    continue

            if bootstrap_diffs:
                bootstrap_diffs = np.array(bootstrap_diffs)

                # 计算置信区间
                ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
                ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

                # 显著性检验
                is_significant = not (ci_lower <= 0 <= ci_upper)

                bootstrap_results[name1][name2] = {
                    'mean_diff': np.mean(bootstrap_diffs),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'is_significant': is_significant,
                    'p_value_approx': np.mean(np.abs(bootstrap_diffs) <= np.abs(np.mean(bootstrap_diffs)))
                }

    return {
        'model_scores': model_scores,
        'bootstrap_results': bootstrap_results
    }


def plot_model_comparison(
    models: Dict[str, BaseTrainer],
    X: pd.DataFrame,
    y: pd.Series,
    comparison_df: pd.DataFrame,
    output_dir: Path
):
    """绘制模型比较图表"""
    logger.info("Generating model comparison plots...")

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # 设置颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    color_map = dict(zip(models.keys(), colors))

    # 1. ROC曲线比较
    plt.figure(figsize=(10, 8))

    for name, trainer in models.items():
        y_pred_proba = trainer.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc_score = comparison_df[comparison_df['model'] == name]['auc_roc'].iloc[0]

        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})',
                color=color_map[name], linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. PR曲线比较
    plt.figure(figsize=(10, 8))

    for name, trainer in models.items():
        y_pred_proba = trainer.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        auc_pr = comparison_df[comparison_df['model'] == name]['auc_pr'].iloc[0]

        plt.plot(recall, precision, label=f'{name} (AUC-PR = {auc_pr:.3f})',
                color=color_map[name], linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'pr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 指标雷达图
    metrics_to_plot = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 角度
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # 闭合圆形

    for name in models.keys():
        model_data = comparison_df[comparison_df['model'] == name].iloc[0]
        values = [model_data[metric] for metric in metrics_to_plot]
        values += values[:1]  # 闭合圆形

        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color_map[name])
        ax.fill(angles, values, alpha=0.25, color=color_map[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_to_plot])
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison (Radar Chart)', size=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(plots_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 指标条形图比较
    metrics_subset = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']

    fig, axes = plt.subplots(1, len(metrics_subset), figsize=(20, 6))

    for i, metric in enumerate(metrics_subset):
        ax = axes[i]

        values = []
        names = []
        colors_list = []

        for name in models.keys():
            value = comparison_df[comparison_df['model'] == name][metric].iloc[0]
            values.append(value)
            names.append(name)
            colors_list.append(color_map[name])

        bars = ax.bar(names, values, color=colors_list, alpha=0.7)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(plots_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 预测概率分布比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (name, trainer) in enumerate(models.items()):
        if idx >= 4:  # 最多显示4个模型
            break

        ax = axes[idx]
        y_pred_proba = trainer.predict_proba(X)[:, 1]

        ax.hist(y_pred_proba[y == 0], bins=30, alpha=0.7,
               label='No Injury', density=True, color='skyblue')
        ax.hist(y_pred_proba[y == 1], bins=30, alpha=0.7,
               label='Injury', density=True, color='salmon')

        ax.set_title(f'{name} - Predicted Probability Distribution')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_dir / 'probability_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_prediction_agreement(
    models: Dict[str, BaseTrainer],
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path
) -> Dict[str, Any]:
    """分析模型间预测一致性"""
    logger.info("Analyzing prediction agreement between models...")

    # 获取所有模型的预测
    predictions = {}
    probabilities = {}

    for name, trainer in models.items():
        predictions[name] = trainer.predict(X)
        probabilities[name] = trainer.predict_proba(X)[:, 1]

    # 创建预测矩阵
    pred_df = pd.DataFrame(predictions)
    prob_df = pd.DataFrame(probabilities)

    # 计算一致性
    results = {}

    # 预测一致性（二元预测）
    agreement_matrix = np.zeros((len(models), len(models)))
    model_names = list(models.keys())

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i != j:
                agreement = (pred_df[name1] == pred_df[name2]).mean()
                agreement_matrix[i, j] = agreement
            else:
                agreement_matrix[i, j] = 1.0

    results['prediction_agreement_matrix'] = agreement_matrix
    results['model_names'] = model_names

    # 概率相关性
    prob_corr = prob_df.corr()
    results['probability_correlation'] = prob_corr

    # 集成预测（简单平均）
    ensemble_prob = prob_df.mean(axis=1)
    ensemble_pred = (ensemble_prob > 0.5).astype(int)

    # 评估集成预测
    metrics_calc = MetricsCalculator(task_type="binary")
    ensemble_metrics = metrics_calc.calculate_all_metrics(y, ensemble_pred, ensemble_prob)
    results['ensemble_metrics'] = ensemble_metrics

    # 可视化
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # 一致性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix,
                xticklabels=model_names,
                yticklabels=model_names,
                annot=True, fmt='.3f', cmap='Blues',
                square=True)
    plt.title('Model Prediction Agreement Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / 'prediction_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 概率相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_corr,
                annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True)
    plt.title('Model Probability Correlation Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / 'probability_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results


def generate_comparison_report(
    comparison_df: pd.DataFrame,
    significance_results: Dict[str, Any],
    agreement_results: Dict[str, Any],
    output_dir: Path
):
    """生成比较报告"""
    logger.info("Generating comparison report...")

    # HTML报告
    html_path = output_dir / 'model_comparison_report.html'

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            .best {{ background-color: #d4edda; }}
            .section {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>Sports Injury Risk Model Comparison Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="section">
            <h2>Model Performance Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>AUC-ROC</th>
                    <th>AUC-PR</th>
                    <th>F1-Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Accuracy</th>
                </tr>
    """

    # 找出每个指标的最佳值
    best_metrics = {}
    for col in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']:
        if col in comparison_df.columns:
            best_metrics[col] = comparison_df[col].max()

    # 添加模型行
    for _, row in comparison_df.iterrows():
        html_content += "<tr>"
        html_content += f"<td>{row['model']}</td>"

        for col in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'accuracy']:
            if col in row:
                value = row[col]
                is_best = abs(value - best_metrics.get(col, 0)) < 1e-6
                class_attr = 'class="best"' if is_best else ''
                html_content += f"<td {class_attr}>{value:.4f}</td>"
            else:
                html_content += "<td>N/A</td>"

        html_content += "</tr>"

    html_content += """
            </table>
        </div>

        <div class="section">
            <h2>Statistical Significance</h2>
    """

    # 添加显著性检验结果
    model_scores = significance_results.get('model_scores', {})
    bootstrap_results = significance_results.get('bootstrap_results', {})

    html_content += "<h3>Model AUC Scores</h3><ul>"
    for model, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
        html_content += f"<li>{model}: {score:.4f}</li>"
    html_content += "</ul>"

    if bootstrap_results:
        html_content += "<h3>Pairwise Comparisons</h3><ul>"
        for model1, comparisons in bootstrap_results.items():
            for model2, result in comparisons.items():
                significance = "significant" if result['is_significant'] else "not significant"
                html_content += f"""
                <li>{model1} vs {model2}:
                    Mean difference = {result['mean_diff']:.4f},
                    CI = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}],
                    {significance}</li>
                """
        html_content += "</ul>"

    html_content += """
        </div>

        <div class="section">
            <h2>Model Agreement Analysis</h2>
    """

    # 添加集成结果
    ensemble_metrics = agreement_results.get('ensemble_metrics', {})
    html_content += f"""
            <h3>Ensemble Performance</h3>
            <p>Simple averaging of model probabilities:</p>
            <ul>
                <li>AUC-ROC: {ensemble_metrics.get('auc_roc', 'N/A'):.4f}</li>
                <li>AUC-PR: {ensemble_metrics.get('auc_pr', 'N/A'):.4f}</li>
                <li>F1-Score: {ensemble_metrics.get('f1', 'N/A'):.4f}</li>
                <li>Precision: {ensemble_metrics.get('precision', 'N/A'):.4f}</li>
                <li>Recall: {ensemble_metrics.get('recall', 'N/A'):.4f}</li>
            </ul>
    """

    html_content += """
        </div>

        <div class="section">
            <h2>Recommendations</h2>
    """

    # 添加推荐
    best_model = comparison_df.loc[comparison_df['auc_roc'].idxmax()]
    html_content += f"""
            <p><strong>Best Individual Model:</strong> {best_model['model']}
               (AUC-ROC: {best_model['auc_roc']:.4f})</p>
    """

    if ensemble_metrics.get('auc_roc', 0) > best_model['auc_roc']:
        html_content += "<p><strong>Recommendation:</strong> Consider using ensemble prediction as it outperforms individual models.</p>"
    else:
        html_content += f"<p><strong>Recommendation:</strong> Use {best_model['model']} for deployment.</p>"

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # CSV报告
    csv_path = output_dir / 'model_comparison.csv'
    comparison_df.to_csv(csv_path, index=False)

    logger.info(f"Comparison report saved to {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='Model file paths')
    parser.add_argument('--names', type=str, nargs='+', help='Model names (optional)')
    parser.add_argument('--output', type=str, help='Output directory (optional)')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'model_comparison_{timestamp}')

    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir}")

    try:
        # 加载模型
        models = load_multiple_models(args.models, args.names)

        if len(models) < 2:
            raise ValueError("At least 2 models are required for comparison")

        # 加载测试数据（使用第一个模型的配置）
        _, X, y = load_model_and_data(args.models[0], config['evaluation']['data'])

        logger.info(f"Comparing {len(models)} models on {len(X)} test samples")

        # 基础指标比较
        comparison_df = compare_basic_metrics(models, X, y)

        # 统计显著性检验
        significance_results = statistical_significance_test(models, X, y)

        # 预测一致性分析
        agreement_results = analyze_prediction_agreement(models, X, y, output_dir)

        # 生成比较图表
        plot_model_comparison(models, X, y, comparison_df, output_dir)

        # 生成比较报告
        generate_comparison_report(
            comparison_df,
            significance_results,
            agreement_results,
            output_dir
        )

        # 打印简要结果
        logger.info("Model comparison completed!")
        logger.info("Performance Summary:")
        for _, row in comparison_df.iterrows():
            logger.info(f"  {row['model']}: AUC-ROC={row['auc_roc']:.4f}, F1={row['f1']:.4f}")

        best_model = comparison_df.loc[comparison_df['auc_roc'].idxmax()]
        logger.info(f"Best model: {best_model['model']} (AUC-ROC: {best_model['auc_roc']:.4f})")

        ensemble_auc = agreement_results['ensemble_metrics'].get('auc_roc', 0)
        logger.info(f"Ensemble AUC-ROC: {ensemble_auc:.4f}")

        logger.info(f"Detailed results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()