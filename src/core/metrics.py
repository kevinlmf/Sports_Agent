"""
评估指标计算模块
支持二分类、生存分析等不同任务的评估指标
"""

from typing import Dict, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import warnings

try:
    from sksurv.metrics import concordance_index_censored, brier_score_survival
    SURVIVAL_AVAILABLE = True
except ImportError:
    SURVIVAL_AVAILABLE = False
    warnings.warn("scikit-survival not available. Survival metrics will not work.")


class MetricsCalculator:
    """指标计算器"""

    def __init__(self, task_type: str = "binary"):
        self.task_type = task_type

    def calculate_all_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_pred_proba: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
        duration: Optional[Union[np.ndarray, pd.Series]] = None,
        event: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, float]:
        """计算所有相关指标"""

        if self.task_type == "binary":
            return self._calculate_binary_metrics(
                y_true, y_pred, y_pred_proba, sample_weight
            )
        elif self.task_type == "survival":
            return self._calculate_survival_metrics(
                y_true, y_pred, duration, event, sample_weight
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _calculate_binary_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_pred_proba: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """计算二分类指标"""

        metrics = {}

        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_pred_proba is not None:
            y_pred_proba = np.array(y_pred_proba)

        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        metrics['precision'] = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)

        # 混淆矩阵相关指标
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            metrics['true_positives'] = tp
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn

            # 特异性（真阴性率）
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

            # 敏感性（真阳性率，等同于recall）
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

            # 阳性预测值和阴性预测值
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # 等同于precision
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0

            # 阳性似然比和阴性似然比
            if metrics['specificity'] > 0:
                metrics['positive_lr'] = metrics['sensitivity'] / (1 - metrics['specificity'])
            else:
                metrics['positive_lr'] = float('inf')

            if metrics['sensitivity'] > 0:
                metrics['negative_lr'] = (1 - metrics['sensitivity']) / metrics['specificity']
            else:
                metrics['negative_lr'] = 0

        # 概率相关指标
        if y_pred_proba is not None:
            try:
                # ROC-AUC
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weight)

                # PR-AUC
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba, sample_weight=sample_weight)
                metrics['auc_pr'] = auc(recall_vals, precision_vals)

                # Brier Score (越小越好)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba, sample_weight=sample_weight)

                # Log Loss (越小越好)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba, sample_weight=sample_weight)

                # 校准相关指标
                calibration_metrics = self._calculate_calibration_metrics(y_true, y_pred_proba)
                metrics.update(calibration_metrics)

            except Exception as e:
                warnings.warn(f"Error calculating probability-based metrics: {e}")

        return metrics

    def _calculate_survival_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        duration: Optional[Union[np.ndarray, pd.Series]] = None,
        event: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """计算生存分析指标"""

        if not SURVIVAL_AVAILABLE:
            raise ImportError("scikit-survival is required for survival metrics")

        metrics = {}

        # 转换为numpy数组
        if duration is not None and event is not None:
            duration = np.array(duration)
            event = np.array(event).astype(bool)

            # C-index (Concordance Index)
            try:
                c_index = concordance_index_censored(event, duration, y_pred)[0]
                metrics['c_index'] = c_index
            except Exception as e:
                warnings.warn(f"Error calculating C-index: {e}")
                metrics['c_index'] = np.nan

        # 如果有时间点预测，计算时间依赖的Brier Score
        if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
            try:
                # 假设y_pred是生存函数值在不同时间点的预测
                time_points = np.linspace(duration.min(), duration.max(), y_pred.shape[1])
                brier_scores = []

                for i, t in enumerate(time_points):
                    # 计算在时间点t的Brier Score
                    survival_probs = y_pred[:, i]
                    actual_survival = (duration > t) | ((duration <= t) & ~event)
                    brier_score = np.mean((survival_probs - actual_survival.astype(float)) ** 2)
                    brier_scores.append(brier_score)

                metrics['mean_brier_score'] = np.mean(brier_scores)
                metrics['integrated_brier_score'] = np.trapz(brier_scores, time_points) / (time_points[-1] - time_points[0])

            except Exception as e:
                warnings.warn(f"Error calculating time-dependent Brier scores: {e}")

        return metrics

    def _calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """计算校准相关指标"""

        metrics = {}

        try:
            # 校准曲线
            prob_true, prob_pred = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
            )

            # ECE (Expected Calibration Error)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # 找到落在当前bin中的预测
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    # 计算bin内的平均置信度和准确率
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()

                    # ECE累加
                    ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

            metrics['ece'] = ece

            # MCE (Maximum Calibration Error)
            mce = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)

                if in_bin.sum() > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    mce = max(mce, abs(avg_confidence_in_bin - accuracy_in_bin))

            metrics['mce'] = mce

            # Reliability (平均校准误差)
            reliability = np.mean(np.abs(prob_true - prob_pred))
            metrics['reliability'] = reliability

        except Exception as e:
            warnings.warn(f"Error calculating calibration metrics: {e}")
            metrics['ece'] = np.nan
            metrics['mce'] = np.nan
            metrics['reliability'] = np.nan

        return metrics

    def calculate_threshold_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred_proba: Union[np.ndarray, pd.Series],
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """计算不同阈值下的指标"""

        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            metrics = self._calculate_binary_metrics(y_true, y_pred, y_pred_proba)
            metrics['threshold'] = threshold

            results.append(metrics)

        return pd.DataFrame(results)

    def get_classification_report(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        target_names: Optional[List[str]] = None
    ) -> str:
        """生成详细的分类报告"""

        return classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )

    def get_confusion_matrix_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        normalize: Optional[str] = None
    ) -> Dict[str, Union[np.ndarray, str]]:
        """获取混淆矩阵及相关指标"""

        cm = confusion_matrix(y_true, y_pred, normalize=normalize)

        result = {
            'confusion_matrix': cm,
            'classification_report': self.get_classification_report(y_true, y_pred)
        }

        return result


class MetricsTracker:
    """指标跟踪器 - 用于训练过程中的指标监控"""

    def __init__(self):
        self.metrics_history = []
        self.best_metrics = {}
        self.current_epoch = 0

    def update(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """更新指标"""
        if epoch is None:
            epoch = self.current_epoch
            self.current_epoch += 1

        metrics_with_epoch = {**metrics, 'epoch': epoch}
        self.metrics_history.append(metrics_with_epoch)

        # 更新最佳指标
        for key, value in metrics.items():
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value

    def get_best_metric(self, metric_name: str) -> float:
        """获取最佳指标值"""
        return self.best_metrics.get(metric_name, 0.0)

    def get_history_df(self) -> pd.DataFrame:
        """获取历史指标DataFrame"""
        return pd.DataFrame(self.metrics_history)

    def plot_metrics(self, metrics_to_plot: Optional[List[str]] = None):
        """绘制指标曲线"""
        try:
            import matplotlib.pyplot as plt

            df = self.get_history_df()
            if df.empty:
                print("No metrics to plot")
                return

            if metrics_to_plot is None:
                metrics_to_plot = [col for col in df.columns if col != 'epoch']

            fig, axes = plt.subplots(
                nrows=(len(metrics_to_plot) + 1) // 2,
                ncols=2,
                figsize=(15, 5 * ((len(metrics_to_plot) + 1) // 2))
            )
            axes = axes.flatten() if len(metrics_to_plot) > 2 else [axes]

            for i, metric in enumerate(metrics_to_plot):
                if metric in df.columns:
                    axes[i].plot(df['epoch'], df[metric], marker='o')
                    axes[i].set_title(f'{metric.title()} over Epochs')
                    axes[i].set_xlabel('Epoch')
                    axes[i].set_ylabel(metric.title())
                    axes[i].grid(True)

            # 隐藏多余的子图
            for i in range(len(metrics_to_plot), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error plotting metrics: {e}")


# Standalone convenience functions for backward compatibility
def compute_auc_roc(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_proba: Union[np.ndarray, pd.Series],
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """
    Compute AUC-ROC score.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for the positive class
        sample_weight: Optional sample weights

    Returns:
        AUC-ROC score
    """
    return roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weight)