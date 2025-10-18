"""
Logistic Regression for injury risk prediction
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import logging

logger = logging.getLogger(__name__)


class LogisticInjuryPredictor:
    """逻辑回归伤病风险预测器"""

    def __init__(self,
                 random_state: int = 42,
                 class_weight: str = 'balanced',
                 max_iter: int = 1000):
        """
        初始化逻辑回归预测器

        Args:
            random_state: 随机种子
            class_weight: 类别权重策略
            max_iter: 最大迭代次数
        """
        self.random_state = random_state
        self.class_weight = class_weight
        self.max_iter = max_iter

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            validation_split: float = 0.2,
            tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        训练逻辑回归模型

        Args:
            X: 特征数据
            y: 目标变量
            validation_split: 验证集比例
            tune_hyperparameters: 是否进行超参数调优

        Returns:
            训练结果字典
        """
        logger.info("Starting logistic regression training")

        # 保存特征名称
        self.feature_names = list(X.columns)

        # 处理缺失值
        X_clean = X.fillna(0)

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X_clean)

        if tune_hyperparameters:
            self.model = self._tune_hyperparameters(X_scaled, y)
        else:
            self.model = LogisticRegression(
                random_state=self.random_state,
                class_weight=self.class_weight,
                max_iter=self.max_iter
            )
            self.model.fit(X_scaled, y)

        self.is_fitted = True

        # 计算训练指标
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]

        results = {
            'model_type': 'logistic_regression',
            'train_auc': roc_auc_score(y, y_prob),
            'train_accuracy': self.model.score(X_scaled, y),
            'feature_importance': self._get_feature_importance(),
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'n_features': len(self.feature_names),
            'n_samples': len(X)
        }

        logger.info(f"Training completed. AUC: {results['train_auc']:.3f}")

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X.fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X.fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        return self.model.predict_proba(X_scaled)

    def get_risk_scores(self, X: pd.DataFrame) -> pd.Series:
        """获取伤病风险分数（0-1）"""
        probabilities = self.predict_proba(X)
        return pd.Series(probabilities[:, 1], index=X.index)

    def _tune_hyperparameters(self, X: np.ndarray, y: pd.Series) -> LogisticRegression:
        """超参数调优"""
        logger.info("Tuning hyperparameters")

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]  # only for elasticnet
        }

        # 处理不兼容的参数组合
        base_model = LogisticRegression(
            random_state=self.random_state,
            class_weight=self.class_weight,
            max_iter=self.max_iter
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV AUC: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_

    def _get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（系数绝对值）"""
        if not self.is_fitted:
            return {}

        coefficients = np.abs(self.model.coef_[0])
        importance_dict = dict(zip(self.feature_names, coefficients))

        # 按重要性排序
        importance_dict = dict(sorted(importance_dict.items(),
                                    key=lambda x: x[1], reverse=True))

        return importance_dict

    def get_top_risk_factors(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """获取顶部风险因子"""
        importance = self._get_feature_importance()
        return list(importance.items())[:top_k]

    def explain_prediction(self, X: pd.DataFrame, index: int) -> Dict[str, Any]:
        """解释单个预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")

        sample = X.iloc[index:index+1]
        risk_score = self.get_risk_scores(sample)[0]

        # 特征贡献分析
        X_clean = sample.fillna(0)
        X_scaled = self.scaler.transform(X_clean)

        coefficients = self.model.coef_[0]
        feature_contributions = X_scaled[0] * coefficients

        # 排序特征贡献
        contrib_dict = dict(zip(self.feature_names, feature_contributions))
        sorted_contributions = sorted(contrib_dict.items(),
                                    key=lambda x: abs(x[1]), reverse=True)

        return {
            'risk_score': risk_score,
            'prediction': 'High Risk' if risk_score > 0.5 else 'Low Risk',
            'top_risk_factors': sorted_contributions[:10],
            'top_protective_factors': [item for item in sorted_contributions if item[1] < 0][:5]
        }

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'max_iter': self.max_iter
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data.get('random_state', 42)
        self.class_weight = model_data.get('class_weight', 'balanced')
        self.max_iter = model_data.get('max_iter', 1000)
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

    def cross_validate(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: int = 5) -> Dict[str, Any]:
        """交叉验证评估"""
        from sklearn.model_selection import cross_val_score, cross_validate

        X_clean = X.fillna(0)
        X_scaled = self.scaler.fit_transform(X_clean)

        model = LogisticRegression(
            random_state=self.random_state,
            class_weight=self.class_weight,
            max_iter=self.max_iter
        )

        scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']

        cv_results = cross_validate(
            model, X_scaled, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            return_train_score=True
        )

        results = {}
        for metric in scoring:
            results[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
            results[f'{metric}_std'] = cv_results[f'test_{metric}'].std()

        logger.info(f"Cross-validation AUC: {results['roc_auc_mean']:.3f} ± {results['roc_auc_std']:.3f}")

        return results

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        if not self.is_fitted:
            return {"status": "Model not fitted"}

        return {
            "model_type": "Logistic Regression",
            "n_features": len(self.feature_names),
            "class_weight": self.class_weight,
            "regularization": getattr(self.model, 'penalty', 'l2'),
            "C": getattr(self.model, 'C', 1.0),
            "solver": getattr(self.model, 'solver', 'lbfgs'),
            "top_features": list(self._get_feature_importance().keys())[:10]
        }