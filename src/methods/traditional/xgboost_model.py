"""
XGBoost for injury risk prediction
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import logging

logger = logging.getLogger(__name__)


class XGBoostInjuryPredictor:
    """XGBoost伤病风险预测器"""

    def __init__(self,
                 n_estimators: int = 100,
                 random_state: int = 42,
                 use_gpu: bool = False,
                 n_jobs: int = -1):
        """
        初始化XGBoost预测器

        Args:
            n_estimators: 树的数量
            random_state: 随机种子
            use_gpu: 是否使用GPU加速
            n_jobs: 并行作业数
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs

        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            validation_split: float = 0.2,
            tune_hyperparameters: bool = True,
            early_stopping_rounds: int = 10) -> Dict[str, Any]:
        """
        训练XGBoost模型

        Args:
            X: 特征数据
            y: 目标变量
            validation_split: 验证集比例
            tune_hyperparameters: 是否进行超参数调优
            early_stopping_rounds: 早停轮次

        Returns:
            训练结果字典
        """
        logger.info("Starting XGBoost training")

        # 保存特征名称
        self.feature_names = list(X.columns)

        # 处理缺失值
        X_clean = X.fillna(-999)  # XGBoost可以处理缺失值，用特殊值标记

        # 计算类别权重
        pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0

        if tune_hyperparameters:
            self.model = self._tune_hyperparameters(X_clean, y, validation_split, early_stopping_rounds)
        else:
            # 划分训练和验证集用于早停
            X_train, X_val, y_train, y_val = train_test_split(
                X_clean, y, test_size=validation_split,
                stratify=y, random_state=self.random_state
            )

            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                scale_pos_weight=pos_weight,
                tree_method='gpu_hist' if self.use_gpu else 'auto',
                n_jobs=self.n_jobs,
                eval_metric='auc'
            )

            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )

        self.is_fitted = True

        # 计算训练指标
        y_pred = self.model.predict(X_clean)
        y_prob = self.model.predict_proba(X_clean)[:, 1]

        results = {
            'model_type': 'xgboost',
            'train_auc': roc_auc_score(y, y_prob),
            'train_accuracy': self.model.score(X_clean, y),
            'feature_importance': self._get_feature_importance(),
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'best_score': getattr(self.model, 'best_score', None)
        }

        logger.info(f"Training completed. AUC: {results['train_auc']:.3f}")

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X.fillna(-999)
        return self.model.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X.fillna(-999)
        return self.model.predict_proba(X_clean)

    def get_risk_scores(self, X: pd.DataFrame) -> pd.Series:
        """获取伤病风险分数（0-1）"""
        probabilities = self.predict_proba(X)
        return pd.Series(probabilities[:, 1], index=X.index)

    def _tune_hyperparameters(self,
                            X: pd.DataFrame,
                            y: pd.Series,
                            validation_split: float,
                            early_stopping_rounds: int) -> xgb.XGBClassifier:
        """超参数调优"""
        logger.info("Tuning hyperparameters")

        # 计算类别权重
        pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }

        # 减少搜索空间
        reduced_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }

        base_model = xgb.XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=pos_weight,
            tree_method='gpu_hist' if self.use_gpu else 'auto',
            n_jobs=self.n_jobs,
            eval_metric='auc'
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model,
            reduced_param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1,  # XGBoost已经并行了
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

        # XGBoost支持多种重要性类型
        importance_types = ['weight', 'gain', 'cover']
        importances = {}

        for imp_type in importance_types:
            try:
                imp_scores = self.model.get_booster().get_score(importance_type=imp_type)
                if imp_scores:
                    importances[imp_type] = imp_scores
            except:
                pass

        # 默认使用gain重要性，如果没有则使用sklearn的feature_importances_
        if 'gain' in importances:
            importance_dict = importances['gain']
        elif hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            importance_dict = {}

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

        # 获取SHAP值（如果可用）
        explanation = self._get_shap_explanation(sample)
        if explanation is None:
            # 降级到简单的特征重要性解释
            explanation = self._get_simple_explanation(sample)

        explanation.update({
            'risk_score': risk_score,
            'prediction': 'High Risk' if risk_score > 0.5 else 'Low Risk'
        })

        return explanation

    def _get_shap_explanation(self, sample: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """使用SHAP解释预测"""
        try:
            import shap

            explainer = shap.TreeExplainer(self.model)
            X_clean = sample.fillna(-999)
            shap_values = explainer.shap_values(X_clean)

            # 获取特征贡献
            contributions = list(zip(self.feature_names, shap_values[0]))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            return {
                'shap_values': contributions[:10],
                'base_value': explainer.expected_value,
                'explanation_type': 'shap'
            }
        except ImportError:
            logger.warning("SHAP not available, using simple explanation")
            return None
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return None

    def _get_simple_explanation(self, sample: pd.DataFrame) -> Dict[str, Any]:
        """简单的特征重要性解释"""
        feature_values = sample.iloc[0]
        feature_importance = self._get_feature_importance()

        contributions = []
        for feature, value in feature_values.items():
            if pd.notna(value) and feature in feature_importance:
                contrib = value * feature_importance[feature]
                contributions.append((feature, contrib, value))

        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            'top_contributors': contributions[:10],
            'feature_importance_rank': list(feature_importance.keys())[:10],
            'explanation_type': 'simple'
        }

    def get_learning_curve(self) -> Dict[str, List]:
        """获取学习曲线数据"""
        if not self.is_fitted or not hasattr(self.model, 'evals_result_'):
            return {}

        evals_result = self.model.evals_result_
        return evals_result

    def plot_feature_importance(self, top_k: int = 20) -> None:
        """绘制特征重要性图"""
        try:
            import matplotlib.pyplot as plt

            importance = self._get_feature_importance()
            if not importance:
                logger.warning("No feature importance available")
                return

            # 取前k个特征
            top_features = list(importance.items())[:top_k]
            features, scores = zip(*top_features)

            plt.figure(figsize=(10, max(6, len(features) * 0.3)))
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('XGBoost Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'use_gpu': self.use_gpu,
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
        self.use_gpu = model_data.get('use_gpu', False)
        self.n_jobs = model_data.get('n_jobs', -1)
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

    def cross_validate(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: int = 5) -> Dict[str, Any]:
        """交叉验证评估"""
        from sklearn.model_selection import cross_val_score, cross_validate

        X_clean = X.fillna(-999)
        pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            scale_pos_weight=pos_weight,
            tree_method='gpu_hist' if self.use_gpu else 'auto',
            n_jobs=self.n_jobs,
            eval_metric='auc'
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
            "model_type": "XGBoost",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "learning_rate": self.model.learning_rate,
            "subsample": self.model.subsample,
            "colsample_bytree": self.model.colsample_bytree,
            "n_features": len(self.feature_names),
            "best_iteration": getattr(self.model, 'best_iteration', None),
            "use_gpu": self.use_gpu,
            "top_features": list(self._get_feature_importance().keys())[:10]
        }