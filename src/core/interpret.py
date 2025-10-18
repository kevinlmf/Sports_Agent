"""
模型可解释性模块
提供SHAP、特征重要性等可解释性方法
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# SHAP相关导入
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. SHAP-based interpretations will not work.")

# Matplotlib相关导入
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting will not work.")

# 深度学习可解释性
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class InterpretabilityManager:
    """可解释性管理器"""

    def __init__(self, model, model_type: str = "sklearn"):
        """
        Args:
            model: 训练好的模型
            model_type: 模型类型 ('sklearn', 'xgboost', 'lightgbm', 'pytorch', 'tensorflow')
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = None
        self.explainer = None

    def set_feature_names(self, feature_names: List[str]):
        """设置特征名称"""
        self.feature_names = feature_names

    def explain_global(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        method: str = "shap",
        max_display: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """全局可解释性分析"""

        if method == "shap":
            return self._explain_global_shap(X, max_display=max_display, **kwargs)
        elif method == "permutation":
            return self._explain_global_permutation(X, **kwargs)
        elif method == "feature_importance":
            return self._explain_global_feature_importance(max_display=max_display)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def explain_local(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_idx: Union[int, List[int]],
        method: str = "shap",
        **kwargs
    ) -> Dict[str, Any]:
        """局部可解释性分析"""

        if method == "shap":
            return self._explain_local_shap(X, instance_idx, **kwargs)
        elif method == "lime":
            return self._explain_local_lime(X, instance_idx, **kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def _explain_global_shap(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_display: int = 20,
        sample_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """使用SHAP进行全局解释"""

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP-based explanations")

        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # 采样以提高计算效率
        if sample_size is not None and len(X_array) > sample_size:
            indices = np.random.choice(len(X_array), sample_size, replace=False)
            X_sample = X_array[indices]
        else:
            X_sample = X_array

        # 根据模型类型选择合适的解释器
        try:
            if self.model_type in ["xgboost", "lightgbm"]:
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "sklearn":
                if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
                    # 树模型
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # 线性模型或其他
                    self.explainer = shap.Explainer(self.model, X_sample)
            else:
                self.explainer = shap.Explainer(self.model, X_sample)

            # 计算SHAP值
            shap_values = self.explainer(X_sample)

            # 处理多输出情况
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 取正类的SHAP值

            # 计算特征重要性
            if hasattr(shap_values, 'values'):
                importance_scores = np.abs(shap_values.values).mean(axis=0)
            else:
                importance_scores = np.abs(shap_values).mean(axis=0)

            # 排序特征重要性
            sorted_indices = np.argsort(importance_scores)[::-1][:max_display]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_scores = importance_scores[sorted_indices]

            results = {
                'method': 'shap_global',
                'feature_importance': dict(zip(sorted_features, sorted_scores)),
                'shap_values': shap_values,
                'explainer': self.explainer,
                'feature_names': feature_names
            }

            return results

        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}")
            return {'error': str(e)}

    def _explain_global_feature_importance(self, max_display: int = 20) -> Dict[str, Any]:
        """使用模型内置特征重要性"""

        importance_scores = None
        feature_names = self.feature_names

        if hasattr(self.model, 'feature_importances_'):
            # 树模型的特征重要性
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 线性模型的系数
            importance_scores = np.abs(self.model.coef_).flatten()
        else:
            return {'error': 'Model does not have built-in feature importance'}

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]

        # 排序
        sorted_indices = np.argsort(importance_scores)[::-1][:max_display]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]

        return {
            'method': 'feature_importance',
            'feature_importance': dict(zip(sorted_features, sorted_scores)),
            'all_scores': importance_scores,
            'feature_names': feature_names
        }

    def _explain_global_permutation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        scoring: str = 'accuracy',
        n_repeats: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """使用排列重要性进行全局解释"""

        from sklearn.inspection import permutation_importance

        if y is None:
            raise ValueError("y is required for permutation importance")

        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # 计算排列重要性
        perm_importance = permutation_importance(
            self.model, X_array, y,
            scoring=scoring,
            n_repeats=n_repeats,
            **kwargs
        )

        # 排序
        sorted_indices = np.argsort(perm_importance.importances_mean)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = perm_importance.importances_mean[sorted_indices]

        return {
            'method': 'permutation_importance',
            'feature_importance': dict(zip(sorted_features, sorted_scores)),
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'feature_names': feature_names
        }

    def _explain_local_shap(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_idx: Union[int, List[int]],
        **kwargs
    ) -> Dict[str, Any]:
        """使用SHAP进行局部解释"""

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP-based explanations")

        if self.explainer is None:
            # 先进行全局分析以初始化解释器
            self.explain_global(X, method="shap", **kwargs)

        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # 选择实例
        if isinstance(instance_idx, int):
            instance_idx = [instance_idx]

        X_instances = X_array[instance_idx]

        # 计算SHAP值
        shap_values = self.explainer(X_instances)

        # 处理多输出情况
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return {
            'method': 'shap_local',
            'shap_values': shap_values,
            'instances': X_instances,
            'instance_idx': instance_idx,
            'feature_names': feature_names,
            'explainer': self.explainer
        }

    def plot_global_importance(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[str] = None,
        **kwargs
    ):
        """绘制全局特征重要性图"""

        if not PLOTTING_AVAILABLE:
            warnings.warn("Matplotlib not available for plotting")
            return

        feature_importance = explanation.get('feature_importance', {})
        if not feature_importance:
            print("No feature importance to plot")
            return

        # 准备数据
        features = list(feature_importance.keys())
        scores = list(feature_importance.values())

        # 创建水平条形图
        plt.figure(figsize=(10, max(6, len(features) * 0.4)))
        bars = plt.barh(range(len(features)), scores)

        # 设置标签
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Feature Importance ({explanation.get("method", "Unknown")})')

        # 颜色渐变
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # 添加数值标签
        for i, score in enumerate(scores):
            plt.text(score + max(scores) * 0.01, i, f'{score:.4f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_shap_summary(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[str] = None,
        **kwargs
    ):
        """绘制SHAP汇总图"""

        if not SHAP_AVAILABLE or not PLOTTING_AVAILABLE:
            warnings.warn("SHAP and Matplotlib are required for SHAP plots")
            return

        shap_values = explanation.get('shap_values')
        feature_names = explanation.get('feature_names')

        if shap_values is None:
            print("No SHAP values to plot")
            return

        # 绘制汇总图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False, **kwargs)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_shap_waterfall(
        self,
        explanation: Dict[str, Any],
        instance_idx: int = 0,
        save_path: Optional[str] = None,
        **kwargs
    ):
        """绘制SHAP瀑布图（单个实例）"""

        if not SHAP_AVAILABLE or not PLOTTING_AVAILABLE:
            warnings.warn("SHAP and Matplotlib are required for SHAP plots")
            return

        shap_values = explanation.get('shap_values')
        if shap_values is None:
            print("No SHAP values to plot")
            return

        # 绘制瀑布图
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_values[instance_idx], show=False, **kwargs)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_interpretation_report(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        output_dir: str = "interpretation_results",
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """生成完整的可解释性报告"""

        if methods is None:
            methods = ["shap", "feature_importance"]

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        report = {
            'model_type': self.model_type,
            'dataset_shape': X.shape if hasattr(X, 'shape') else (len(X), len(X[0])),
            'methods': {},
            'plots': {}
        }

        # 全局解释
        for method in methods:
            try:
                if method == "permutation" and y is None:
                    warnings.warn("Skipping permutation importance: y is required")
                    continue

                explanation = self.explain_global(X, method=method, y=y)
                report['methods'][method] = explanation

                # 生成图表
                if method in ["shap", "feature_importance", "permutation"]:
                    plot_path = output_dir / f"{method}_importance.png"
                    self.plot_global_importance(explanation, save_path=str(plot_path))
                    report['plots'][f'{method}_importance'] = str(plot_path)

                if method == "shap":
                    shap_summary_path = output_dir / "shap_summary.png"
                    self.plot_shap_summary(explanation, save_path=str(shap_summary_path))
                    report['plots']['shap_summary'] = str(shap_summary_path)

            except Exception as e:
                warnings.warn(f"Failed to generate {method} explanation: {e}")
                report['methods'][method] = {'error': str(e)}

        # 保存报告
        import json
        report_path = output_dir / "interpretation_report.json"
        with open(report_path, 'w') as f:
            # 处理无法JSON序列化的对象
            json_report = {}
            for key, value in report.items():
                if key == 'methods':
                    json_methods = {}
                    for method_name, method_result in value.items():
                        if isinstance(method_result, dict) and 'feature_importance' in method_result:
                            json_methods[method_name] = {
                                'method': method_result.get('method'),
                                'feature_importance': method_result['feature_importance']
                            }
                        else:
                            json_methods[method_name] = str(method_result)
                    json_report[key] = json_methods
                else:
                    json_report[key] = value

            json.dump(json_report, f, indent=2)

        print(f"Interpretation report saved to {output_dir}")
        return report


class AdvancedInterpreter:
    """高级可解释性分析器（包含深度学习模型的特殊方法）"""

    def __init__(self, model, model_type: str = "pytorch"):
        self.model = model
        self.model_type = model_type

    def integrated_gradients(
        self,
        X: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        steps: int = 50
    ) -> np.ndarray:
        """Integrated Gradients方法（适用于深度学习模型）"""

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Integrated Gradients")

        if baseline is None:
            baseline = np.zeros_like(X)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).requires_grad_(True)
        baseline_tensor = torch.FloatTensor(baseline)

        # 计算积分路径
        alphas = torch.linspace(0, 1, steps)
        gradients = []

        for alpha in alphas:
            # 插值输入
            interpolated_input = baseline_tensor + alpha * (X_tensor - baseline_tensor)
            interpolated_input.requires_grad_(True)

            # 前向传播
            output = self.model(interpolated_input)
            if len(output.shape) > 1:
                output = output[:, 1]  # 取正类概率

            # 计算梯度
            gradients_batch = torch.autograd.grad(
                outputs=output.sum(),
                inputs=interpolated_input,
                create_graph=False,
                retain_graph=False
            )[0]

            gradients.append(gradients_batch)

        # 计算平均梯度
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # 计算积分梯度
        integrated_grads = (X_tensor - baseline_tensor) * avg_gradients

        return integrated_grads.detach().numpy()