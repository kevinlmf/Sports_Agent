"""
概率校准模块
提供多种概率校准方法，提高模型预测概率的可靠性
"""

from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Temperature scaling will not work.")


class PlattScaling(BaseEstimator, ClassifierMixin):
    """Platt Scaling校准方法"""

    def __init__(self):
        self.calibrator = LogisticRegression()

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray):
        """拟合校准器"""
        # 转换概率为logits（避免数值问题）
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)

        self.calibrator.fit(logits, y_true)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """校准概率预测"""
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)

        calibrated_proba = self.calibrator.predict_proba(logits)
        return calibrated_proba

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """转换概率"""
        return self.predict_proba(probabilities)[:, 1]


class IsotonicCalibration(BaseEstimator, ClassifierMixin):
    """Isotonic Regression校准方法"""

    def __init__(self, out_of_bounds='clip'):
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray):
        """拟合校准器"""
        self.calibrator.fit(probabilities, y_true)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """校准概率预测"""
        calibrated_proba_pos = self.calibrator.predict(probabilities)
        calibrated_proba_neg = 1 - calibrated_proba_pos

        return np.column_stack([calibrated_proba_neg, calibrated_proba_pos])

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """转换概率"""
        return self.calibrator.predict(probabilities)


class TemperatureScaling:
    """Temperature Scaling校准方法（适用于深度学习模型）"""

    def __init__(self, max_iter: int = 50, lr: float = 0.01):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Temperature Scaling")

        self.max_iter = max_iter
        self.lr = lr
        self.temperature = None

    def fit(self, logits: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor]):
        """拟合温度参数"""
        # 转换为PyTorch张量
        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        if isinstance(y_true, np.ndarray):
            y_true = torch.LongTensor(y_true)

        # 初始化温度参数
        self.temperature = nn.Parameter(torch.ones(1))

        # 优化器
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        def eval():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, y_true)
            loss.backward()
            return loss

        # 优化温度参数
        optimizer.step(eval)

        return self

    def transform(self, logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """应用温度缩放"""
        if self.temperature is None:
            raise ValueError("Temperature scaling not fitted yet!")

        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)

        with torch.no_grad():
            calibrated_logits = logits / self.temperature
            probabilities = torch.softmax(calibrated_logits, dim=1)

        return probabilities.numpy()


class BetaCalibration(BaseEstimator, ClassifierMixin):
    """Beta Calibration校准方法"""

    def __init__(self, parameters="abm"):
        """
        parameters: str
            - "abm": 拟合a, b, m参数
            - "ab": 拟合a, b参数，m=1
        """
        self.parameters = parameters
        self.a_ = None
        self.b_ = None
        self.m_ = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray):
        """拟合Beta校准参数"""
        from scipy.optimize import minimize

        # 转换概率以避免数值问题
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

        def beta_calibration_loss(params):
            if self.parameters == "abm":
                a, b, m = params
            else:  # "ab"
                a, b = params
                m = 1.0

            # Beta校准公式
            calibrated_proba = 1 / (1 + np.exp(a * np.log(probabilities / (1 - probabilities)) + b))
            if m != 1:
                calibrated_proba = calibrated_proba ** m

            # 交叉熵损失
            calibrated_proba = np.clip(calibrated_proba, 1e-7, 1 - 1e-7)
            loss = -np.mean(y_true * np.log(calibrated_proba) + (1 - y_true) * np.log(1 - calibrated_proba))

            return loss

        # 优化参数
        if self.parameters == "abm":
            initial_params = [1.0, 0.0, 1.0]
            bounds = [(0.1, 10), (-5, 5), (0.1, 5)]
        else:
            initial_params = [1.0, 0.0]
            bounds = [(0.1, 10), (-5, 5)]

        result = minimize(
            beta_calibration_loss,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )

        if self.parameters == "abm":
            self.a_, self.b_, self.m_ = result.x
        else:
            self.a_, self.b_ = result.x
            self.m_ = 1.0

        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """校准概率预测"""
        if self.a_ is None:
            raise ValueError("Beta calibration not fitted yet!")

        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

        # Beta校准公式
        calibrated_proba_pos = 1 / (1 + np.exp(self.a_ * np.log(probabilities / (1 - probabilities)) + self.b_))
        if self.m_ != 1:
            calibrated_proba_pos = calibrated_proba_pos ** self.m_

        calibrated_proba_neg = 1 - calibrated_proba_pos

        return np.column_stack([calibrated_proba_neg, calibrated_proba_pos])

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """转换概率"""
        return self.predict_proba(probabilities)[:, 1]


class CalibrationManager:
    """校准管理器 - 统一管理各种校准方法"""

    def __init__(self):
        self.available_methods = {
            'platt': PlattScaling,
            'isotonic': IsotonicCalibration,
            'beta': BetaCalibration
        }

        if TORCH_AVAILABLE:
            self.available_methods['temperature'] = TemperatureScaling

    def calibrate_model(
        self,
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        method: str = 'isotonic',
        cv: int = 5,
        **kwargs
    ) -> BaseEstimator:
        """校准模型"""

        if method == 'sklearn_builtin':
            # 使用sklearn内置的校准方法
            calibrated_model = CalibratedClassifierCV(
                base_estimator=model,
                method='isotonic' if method == 'sklearn_builtin' else method,
                cv=cv
            )
            calibrated_model.fit(X, y)
            return calibrated_model

        elif method in self.available_methods:
            # 使用自定义校准方法
            calibrator_class = self.available_methods[method]
            calibrator = calibrator_class(**kwargs)

            # 获取未校准的概率
            if hasattr(model, 'predict_proba'):
                # 使用交叉验证获得未偏倚的概率预测
                uncalibrated_proba = cross_val_predict(
                    model, X, y, cv=cv, method='predict_proba'
                )[:, 1]
            else:
                raise ValueError("Model must have predict_proba method for calibration")

            # 拟合校准器
            calibrator.fit(uncalibrated_proba, y)

            # 创建校准后的模型包装器
            calibrated_model = CalibratedModel(model, calibrator)
            return calibrated_model

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = 'uniform'
    ) -> Dict[str, Any]:
        """评估校准效果"""

        # 计算校准曲线
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy=strategy
        )

        # 计算ECE（Expected Calibration Error）
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

        # 计算Brier Score
        brier_score = np.mean((y_pred_proba - y_true) ** 2)

        return {
            'ece': ece,
            'brier_score': brier_score,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            }
        }

    def compare_calibration_methods(
        self,
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        methods: Optional[list] = None,
        cv: int = 5
    ) -> pd.DataFrame:
        """比较不同校准方法的效果"""

        if methods is None:
            methods = ['platt', 'isotonic', 'beta']

        results = []

        # 原始模型（未校准）
        uncalibrated_proba = cross_val_predict(
            model, X, y, cv=cv, method='predict_proba'
        )[:, 1]

        uncalibrated_eval = self.evaluate_calibration(y, uncalibrated_proba)
        results.append({
            'method': 'uncalibrated',
            'ece': uncalibrated_eval['ece'],
            'brier_score': uncalibrated_eval['brier_score']
        })

        # 各种校准方法
        for method in methods:
            try:
                calibrated_model = self.calibrate_model(model, X, y, method=method, cv=cv)
                calibrated_proba = cross_val_predict(
                    calibrated_model, X, y, cv=cv, method='predict_proba'
                )[:, 1]

                calibrated_eval = self.evaluate_calibration(y, calibrated_proba)
                results.append({
                    'method': method,
                    'ece': calibrated_eval['ece'],
                    'brier_score': calibrated_eval['brier_score']
                })
            except Exception as e:
                warnings.warn(f"Error with {method} calibration: {e}")

        return pd.DataFrame(results)


class CalibratedModel:
    """校准模型包装器"""

    def __init__(self, base_model: BaseEstimator, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def fit(self, X, y, **kwargs):
        """拟合模型"""
        return self.base_model.fit(X, y, **kwargs)

    def predict(self, X):
        """预测"""
        return self.base_model.predict(X)

    def predict_proba(self, X):
        """预测概率（校准后）"""
        # 获取原始概率
        uncalibrated_proba = self.base_model.predict_proba(X)

        # 校准正类概率
        if hasattr(self.calibrator, 'predict_proba'):
            calibrated_proba = self.calibrator.predict_proba(uncalibrated_proba[:, 1])
        else:
            calibrated_pos = self.calibrator.transform(uncalibrated_proba[:, 1])
            calibrated_proba = np.column_stack([1 - calibrated_pos, calibrated_pos])

        return calibrated_proba

    def __getattr__(self, name):
        """代理其他属性到基础模型"""
        return getattr(self.base_model, name)


# Standalone convenience functions for backward compatibility
def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform',
    ax=None,
    **kwargs
):
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for the positive class
        n_bins: Number of bins for calibration curve
        strategy: Strategy for binning ('uniform' or 'quantile')
        ax: Matplotlib axis to plot on (creates new if None)
        **kwargs: Additional arguments passed to matplotlib plot

    Returns:
        Matplotlib figure and axis
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for plotting calibration curves")

    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy=strategy
    )

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', **kwargs)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    # Labels and formatting
    ax.set_xlabel('Mean predicted probability', fontsize=12)
    ax.set_ylabel('Fraction of positives', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return fig, ax