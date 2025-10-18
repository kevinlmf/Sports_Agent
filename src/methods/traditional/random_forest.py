"""
Random Forest for injury risk prediction
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import logging

logger = logging.getLogger(__name__)


class RandomForestInjuryPredictor(BaseEstimator, ClassifierMixin):
    """随机森林伤病风险预测器"""

    def __init__(self,
                 n_estimators: int = 100,
                 random_state: int = 42,
                 class_weight: str = 'balanced',
                 n_jobs: int = -1):
        """
        初始化随机森林预测器

        Args:
            n_estimators: 树的数量
            random_state: 随机种子
            class_weight: 类别权重策略
            n_jobs: 并行作业数
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs

        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'n_jobs': self.n_jobs
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _more_tags(self):
        """Set sklearn tags to identify this as a classifier."""
        return {'requires_fit': True,
                'requires_y': True,
                'requires_positive_X': False,
                'binary_only': True,
                '_xfail_checks': {'check_parameters_default_constructible'}}

    @property
    def classes_(self):
        """Classes labels."""
        return np.array([0, 1])

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            validation_split: float = 0.2,
            tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        训练随机森林模型

        Args:
            X: 特征数据
            y: 目标变量
            validation_split: 验证集比例
            tune_hyperparameters: 是否进行超参数调优

        Returns:
            训练结果字典
        """
        logger.info("Starting Random Forest training")

        # 保存特征名称
        self.feature_names = list(X.columns)

        # 处理缺失值
        X_clean = X.fillna(X.median())

        if tune_hyperparameters:
            self.model = self._tune_hyperparameters(X_clean, y)
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs
            )
            self.model.fit(X_clean, y)

        self.is_fitted = True

        # 计算训练指标
        y_pred = self.model.predict(X_clean)
        y_prob = self.model.predict_proba(X_clean)[:, 1]

        results = {
            'model_type': 'random_forest',
            'train_auc': roc_auc_score(y, y_prob),
            'train_accuracy': self.model.score(X_clean, y),
            'feature_importance': self._get_feature_importance(),
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'oob_score': getattr(self.model, 'oob_score_', None)
        }

        logger.info(f"Training completed. AUC: {results['train_auc']:.3f}")

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X.fillna(X.median())
        return self.model.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X.fillna(X.median())
        return self.model.predict_proba(X_clean)

    def get_risk_scores(self, X: pd.DataFrame) -> pd.Series:
        """获取伤病风险分数（0-1）"""
        probabilities = self.predict_proba(X)
        return pd.Series(probabilities[:, 1], index=X.index)

    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """超参数调优"""
        logger.info("Tuning hyperparameters")

        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        base_model = RandomForestClassifier(
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            oob_score=True
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        # 减少搜索空间以加快速度
        reduced_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 4],
            'max_features': ['sqrt', None]
        }

        grid_search = GridSearchCV(
            base_model,
            reduced_param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1,  # RF已经并行了
            verbose=1
        )

        grid_search.fit(X, y)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV AUC: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_

    def _get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_fitted:
            return {}

        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances))

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
        risk_score = self.get_risk_scores(sample).iloc[0]

        # 使用SHAP或简单的特征重要性解释
        feature_values = sample.iloc[0]
        feature_importance = self._get_feature_importance()

        # 计算每个特征对预测的贡献（简化版）
        contributions = []
        for feature, value in feature_values.items():
            if pd.notna(value) and feature in feature_importance:
                # 简化的贡献计算：特征值 * 重要性
                contrib = value * feature_importance[feature]
                contributions.append((feature, contrib, value))

        # 按贡献绝对值排序
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            'risk_score': risk_score,
            'prediction': 'High Risk' if risk_score > 0.5 else 'Low Risk',
            'top_contributors': [(f, c, v) for f, c, v in contributions[:10]],
            'feature_importance_rank': list(feature_importance.keys())[:10]
        }

    def get_tree_paths(self, X: pd.DataFrame, tree_idx: int = 0) -> List[str]:
        """获取决策路径（仅用于理解模型）"""
        if not self.is_fitted or tree_idx >= len(self.model.estimators_):
            return []

        tree = self.model.estimators_[tree_idx]
        X_clean = X.fillna(X.median())

        # 获取叶子节点路径
        leaf_indices = tree.apply(X_clean)
        paths = []

        for i, leaf_idx in enumerate(leaf_indices):
            path = f"Sample {i} -> Leaf {leaf_idx}"
            paths.append(path)

        return paths

    def analyze_feature_interactions(self) -> Dict[str, Any]:
        """分析特征交互（基于树结构）"""
        if not self.is_fitted:
            return {}

        # 收集所有树的分割特征信息
        feature_splits = {}
        for tree in self.model.estimators_:
            for node_idx in range(tree.tree_.node_count):
                if tree.tree_.feature[node_idx] >= 0:  # 非叶子节点
                    feature_idx = tree.tree_.feature[node_idx]
                    feature_name = self.feature_names[feature_idx]

                    if feature_name not in feature_splits:
                        feature_splits[feature_name] = {
                            'split_count': 0,
                            'avg_threshold': 0,
                            'thresholds': []
                        }

                    feature_splits[feature_name]['split_count'] += 1
                    threshold = tree.tree_.threshold[node_idx]
                    feature_splits[feature_name]['thresholds'].append(threshold)

        # 计算平均分割阈值
        for feature, info in feature_splits.items():
            if info['thresholds']:
                info['avg_threshold'] = np.mean(info['thresholds'])

        return feature_splits

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'n_jobs': self.n_jobs
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.n_estimators = model_data.get('n_estimators', 100)
        self.random_state = model_data.get('random_state', 42)
        self.class_weight = model_data.get('class_weight', 'balanced')
        self.n_jobs = model_data.get('n_jobs', -1)
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

    def cross_validate(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: int = 5) -> Dict[str, Any]:
        """交叉验证评估"""
        from sklearn.model_selection import cross_val_score, cross_validate

        X_clean = X.fillna(X.median())

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs
        )

        scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']

        cv_results = cross_validate(
            model, X_clean, y,
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
            "model_type": "Random Forest",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "max_features": self.model.max_features,
            "n_features": len(self.feature_names),
            "oob_score": getattr(self.model, 'oob_score_', None),
            "top_features": list(self._get_feature_importance().keys())[:10]
        }