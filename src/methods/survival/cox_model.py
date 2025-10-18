"""
Cox Proportional Hazards model for injury risk prediction
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.preprocessing import OneHotEncoder
    from sksurv.metrics import concordance_index_censored
    SURVIVAL_AVAILABLE = True
except ImportError:
    SURVIVAL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Survival analysis packages not available. Install lifelines and scikit-survival.")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import joblib

logger = logging.getLogger(__name__)


class CoxInjuryPredictor:
    """Cox比例风险模型伤病风险预测器"""

    def __init__(self,
                 penalizer: float = 0.1,
                 l1_ratio: float = 0.0,
                 backend: str = 'lifelines',  # 'lifelines' or 'sksurv'
                 random_state: int = 42):
        """
        初始化Cox模型预测器

        Args:
            penalizer: 正则化参数
            l1_ratio: L1正则化比例 (0=Ridge, 1=Lasso)
            backend: 使用的生存分析库
            random_state: 随机种子
        """
        if not SURVIVAL_AVAILABLE:
            raise ImportError("Survival analysis packages not available")

        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.backend = backend.lower()
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def prepare_survival_data(self,
                            loads_df: pd.DataFrame,
                            injuries_df: pd.DataFrame,
                            follow_up_days: int = 365) -> pd.DataFrame:
        """
        准备生存分析数据

        Args:
            loads_df: 负荷数据
            injuries_df: 伤病数据
            follow_up_days: 随访天数

        Returns:
            生存分析数据集
        """
        survival_data = []

        for player_id in loads_df['player_id'].unique():
            player_loads = loads_df[loads_df['player_id'] == player_id].sort_values('date')
            player_injuries = injuries_df[injuries_df['player_id'] == player_id].sort_values('onset_date')

            # 为每个负荷记录创建生存数据
            for idx, load_row in player_loads.iterrows():
                observation_date = load_row['date']
                end_date = observation_date + timedelta(days=follow_up_days)

                # 查找观察期内的第一次伤病
                future_injuries = player_injuries[
                    (player_injuries['onset_date'] > observation_date) &
                    (player_injuries['onset_date'] <= end_date)
                ]

                if not future_injuries.empty:
                    # 发生伤病
                    first_injury = future_injuries.iloc[0]
                    time_to_event = (first_injury['onset_date'] - observation_date).days
                    event_observed = 1
                    severity = first_injury.get('severity', 'UNKNOWN')
                else:
                    # 审查（未观察到伤病）
                    time_to_event = follow_up_days
                    event_observed = 0
                    severity = 'NONE'

                # 收集特征
                survival_record = {
                    'player_id': player_id,
                    'observation_date': observation_date,
                    'time_to_event': time_to_event,
                    'event_observed': event_observed,
                    'severity': severity
                }

                # 添加负荷特征
                for col in load_row.index:
                    if col not in ['player_id', 'date']:
                        survival_record[col] = load_row[col]

                survival_data.append(survival_record)

        survival_df = pd.DataFrame(survival_data)

        # 过滤掉时间为0的记录
        survival_df = survival_df[survival_df['time_to_event'] > 0]

        logger.info(f"Created survival dataset with {len(survival_df)} observations")
        logger.info(f"Event rate: {survival_df['event_observed'].mean():.3f}")

        return survival_df

    def fit(self,
            survival_df: pd.DataFrame,
            duration_col: str = 'time_to_event',
            event_col: str = 'event_observed',
            feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        训练Cox模型

        Args:
            survival_df: 生存分析数据
            duration_col: 持续时间列名
            event_col: 事件观察列名
            feature_columns: 特征列名列表

        Returns:
            训练结果字典
        """
        logger.info(f"Starting Cox model training with {self.backend} backend")

        if feature_columns is None:
            # 自动选择数值特征
            exclude_cols = [duration_col, event_col, 'player_id', 'observation_date', 'severity']
            feature_columns = [col for col in survival_df.columns
                             if col not in exclude_cols and survival_df[col].dtype in ['int64', 'float64']]

        self.feature_names = feature_columns
        logger.info(f"Using {len(feature_columns)} features for Cox model")

        # 准备数据
        X = survival_df[feature_columns].fillna(0)
        durations = survival_df[duration_col]
        events = survival_df[event_col].astype(bool)

        # 特征标准化
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=feature_columns,
            index=X.index
        )

        if self.backend == 'lifelines':
            return self._fit_lifelines(X_scaled, durations, events)
        elif self.backend == 'sksurv':
            return self._fit_sksurv(X_scaled, durations, events)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _fit_lifelines(self, X: pd.DataFrame, durations: pd.Series, events: pd.Series) -> Dict[str, Any]:
        """使用lifelines训练Cox模型"""
        # 准备lifelines格式数据
        data = X.copy()
        data['duration'] = durations
        data['event'] = events

        # 创建模型
        self.model = CoxPHFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio
        )

        # 训练模型
        self.model.fit(data, duration_col='duration', event_col='event')

        self.is_fitted = True

        # 计算训练指标
        concordance = self.model.concordance_index_

        # 获取系数
        hazard_ratios = self.model.hazard_ratios_
        coefficients = self.model.params_

        results = {
            'model_type': 'cox_lifelines',
            'concordance_index': concordance,
            'hazard_ratios': hazard_ratios.to_dict(),
            'coefficients': coefficients.to_dict(),
            'n_features': len(self.feature_names),
            'n_samples': len(data),
            'n_events': events.sum(),
            'event_rate': events.mean(),
            'log_likelihood': self.model.log_likelihood_,
            'AIC': self.model.AIC_,
            'partial_AIC': self.model.AIC_partial_
        }

        logger.info(f"Cox model training completed. C-index: {concordance:.3f}")
        return results

    def _fit_sksurv(self, X: pd.DataFrame, durations: pd.Series, events: pd.Series) -> Dict[str, Any]:
        """使用scikit-survival训练Cox模型"""
        # 准备sksurv格式数据
        y = np.array([(event, duration) for event, duration in zip(events, durations)],
                    dtype=[('event', bool), ('duration', float)])

        # 创建模型
        if self.l1_ratio == 0:
            # Pure Ridge
            from sksurv.linear_model import CoxPHSurvivalAnalysis
            self.model = CoxPHSurvivalAnalysis(alpha=self.penalizer)
        else:
            # ElasticNet Cox
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            self.model = CoxnetSurvivalAnalysis(
                l1_ratio=self.l1_ratio,
                alpha_min_ratio=0.01,
                max_iter=1000
            )

        # 训练模型
        self.model.fit(X.values, y)

        self.is_fitted = True

        # 计算训练指标
        risk_scores = self.model.predict(X.values)
        concordance = concordance_index_censored(events, durations, risk_scores)[0]

        # 获取系数
        if hasattr(self.model, 'coef_'):
            coefficients = dict(zip(self.feature_names, self.model.coef_))
            hazard_ratios = dict(zip(self.feature_names, np.exp(self.model.coef_)))
        else:
            # For CoxnetSurvivalAnalysis
            coefficients = dict(zip(self.feature_names, self.model.coef_[0]))
            hazard_ratios = dict(zip(self.feature_names, np.exp(self.model.coef_[0])))

        results = {
            'model_type': 'cox_sksurv',
            'concordance_index': concordance,
            'hazard_ratios': hazard_ratios,
            'coefficients': coefficients,
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'n_events': events.sum(),
            'event_rate': events.mean()
        }

        logger.info(f"Cox model training completed. C-index: {concordance:.3f}")
        return results

    def predict_survival_function(self, X: pd.DataFrame, times: np.ndarray = None) -> Dict[str, np.ndarray]:
        """预测生存函数"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X[self.feature_names].fillna(0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_clean),
            columns=self.feature_names,
            index=X_clean.index
        )

        if self.backend == 'lifelines':
            if times is None:
                times = np.linspace(1, 365, 100)

            survival_functions = self.model.predict_survival_function(X_scaled, times=times)
            return {
                'times': times,
                'survival_probs': survival_functions.values.T  # Shape: (n_samples, n_times)
            }
        else:
            # sksurv backend
            if hasattr(self.model, 'predict_survival_function'):
                survival_functions = self.model.predict_survival_function(X_scaled.values)
                if times is None:
                    times = survival_functions[0].x

                survival_probs = np.array([sf(times) for sf in survival_functions])
                return {
                    'times': times,
                    'survival_probs': survival_probs
                }
            else:
                logger.warning("Survival function prediction not available for this model")
                return {'times': np.array([]), 'survival_probs': np.array([])}

    def predict_risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        """预测风险分数（越高越危险）"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X[self.feature_names].fillna(0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_clean),
            columns=self.feature_names,
            index=X_clean.index
        )

        if self.backend == 'lifelines':
            # lifelines中风险分数为线性预测值
            risk_scores = self.model.predict_partial_hazard(X_scaled).values
        else:
            # sksurv中直接预测风险分数
            risk_scores = self.model.predict(X_scaled.values)

        return risk_scores

    def predict_injury_probability(self, X: pd.DataFrame, time_horizon: int = 30) -> np.ndarray:
        """预测指定时间内的伤病概率"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        survival_data = self.predict_survival_function(X, times=np.array([time_horizon]))

        if len(survival_data['survival_probs']) > 0:
            # 伤病概率 = 1 - 生存概率
            injury_probs = 1 - survival_data['survival_probs'][:, 0]
        else:
            # 降级到风险分数
            risk_scores = self.predict_risk_scores(X)
            # 将风险分数转换为概率（sigmoid变换）
            injury_probs = 1 / (1 + np.exp(-risk_scores))

        return injury_probs

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（基于系数绝对值）"""
        if not self.is_fitted:
            return {}

        if self.backend == 'lifelines':
            coefficients = self.model.params_
        else:
            if hasattr(self.model, 'coef_'):
                coefficients = pd.Series(self.model.coef_, index=self.feature_names)
            else:
                coefficients = pd.Series(self.model.coef_[0], index=self.feature_names)

        # 按系数绝对值排序
        importance = coefficients.abs().sort_values(ascending=False)
        return importance.to_dict()

    def get_hazard_ratios(self) -> Dict[str, float]:
        """获取风险比"""
        if not self.is_fitted:
            return {}

        if self.backend == 'lifelines':
            return self.model.hazard_ratios_.to_dict()
        else:
            if hasattr(self.model, 'coef_'):
                hazard_ratios = np.exp(self.model.coef_)
            else:
                hazard_ratios = np.exp(self.model.coef_[0])

            return dict(zip(self.feature_names, hazard_ratios))

    def explain_prediction(self, X: pd.DataFrame, index: int) -> Dict[str, Any]:
        """解释单个预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")

        sample = X.iloc[index:index+1]
        risk_score = self.predict_risk_scores(sample)[0]
        injury_prob_30d = self.predict_injury_probability(sample, time_horizon=30)[0]

        # 特征贡献分析
        feature_values = sample[self.feature_names].iloc[0]
        hazard_ratios = self.get_hazard_ratios()

        contributions = []
        for feature, value in feature_values.items():
            if pd.notna(value) and feature in hazard_ratios:
                hr = hazard_ratios[feature]
                # 风险贡献 = (HR - 1) * 标准化特征值
                scaled_value = self.scaler.transform([[value]])[0][0] if len(self.feature_names) == 1 else \
                             self.scaler.transform(sample[self.feature_names].fillna(0))[0][list(self.feature_names).index(feature)]
                contribution = (hr - 1) * scaled_value
                contributions.append((feature, contribution, hr, value))

        # 按贡献绝对值排序
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            'risk_score': risk_score,
            'injury_probability_30d': injury_prob_30d,
            'risk_level': 'High' if injury_prob_30d > 0.2 else 'Medium' if injury_prob_30d > 0.1 else 'Low',
            'top_risk_factors': [(f, c, hr, v) for f, c, hr, v in contributions[:10] if c > 0],
            'top_protective_factors': [(f, abs(c), hr, v) for f, c, hr, v in contributions[:10] if c < 0]
        }

    def cross_validate(self,
                      survival_df: pd.DataFrame,
                      duration_col: str = 'time_to_event',
                      event_col: str = 'event_observed',
                      feature_columns: List[str] = None,
                      cv: int = 5) -> Dict[str, Any]:
        """交叉验证评估"""
        if feature_columns is None:
            exclude_cols = [duration_col, event_col, 'player_id', 'observation_date', 'severity']
            feature_columns = [col for col in survival_df.columns
                             if col not in exclude_cols and survival_df[col].dtype in ['int64', 'float64']]

        X = survival_df[feature_columns].fillna(0)
        durations = survival_df[duration_col]
        events = survival_df[event_col].astype(bool)

        # 使用player-based splitting避免数据泄漏
        if 'player_id' in survival_df.columns:
            unique_players = survival_df['player_id'].unique()
            np.random.seed(self.random_state)
            np.random.shuffle(unique_players)

            fold_size = len(unique_players) // cv
            cv_scores = []

            for i in range(cv):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < cv - 1 else len(unique_players)

                test_players = unique_players[start_idx:end_idx]
                train_players = np.setdiff1d(unique_players, test_players)

                train_mask = survival_df['player_id'].isin(train_players)
                test_mask = survival_df['player_id'].isin(test_players)

                X_train, X_test = X[train_mask], X[test_mask]
                durations_train, durations_test = durations[train_mask], durations[test_mask]
                events_train, events_test = events[train_mask], events[test_mask]

                # 训练模型
                temp_model = CoxInjuryPredictor(
                    penalizer=self.penalizer,
                    l1_ratio=self.l1_ratio,
                    backend=self.backend,
                    random_state=self.random_state
                )

                temp_survival_df = pd.DataFrame(X_train)
                temp_survival_df['duration'] = durations_train
                temp_survival_df['event'] = events_train

                temp_model.fit(temp_survival_df, feature_columns=feature_columns)

                # 评估
                risk_scores = temp_model.predict_risk_scores(pd.DataFrame(X_test, columns=feature_columns))
                c_index = concordance_index_censored(events_test, durations_test, risk_scores)[0]
                cv_scores.append(c_index)

        else:
            # 简单随机分割
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            cv_scores = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                durations_train, durations_test = durations.iloc[train_idx], durations.iloc[test_idx]
                events_train, events_test = events.iloc[train_idx], events.iloc[test_idx]

                # 类似的训练和评估过程...
                pass

        results = {
            'cv_concordance_mean': np.mean(cv_scores),
            'cv_concordance_std': np.std(cv_scores),
            'cv_scores': cv_scores
        }

        logger.info(f"Cross-validation C-index: {results['cv_concordance_mean']:.3f} ± {results['cv_concordance_std']:.3f}")
        return results

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'penalizer': self.penalizer,
            'l1_ratio': self.l1_ratio,
            'backend': self.backend,
            'random_state': self.random_state
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Cox model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.penalizer = model_data.get('penalizer', 0.1)
        self.l1_ratio = model_data.get('l1_ratio', 0.0)
        self.backend = model_data.get('backend', 'lifelines')
        self.random_state = model_data.get('random_state', 42)
        self.is_fitted = True

        logger.info(f"Cox model loaded from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        if not self.is_fitted:
            return {"status": "Model not fitted"}

        summary = {
            "model_type": f"Cox Proportional Hazards ({self.backend})",
            "backend": self.backend,
            "penalizer": self.penalizer,
            "l1_ratio": self.l1_ratio,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "is_fitted": self.is_fitted
        }

        if self.backend == 'lifelines' and hasattr(self.model, 'concordance_index_'):
            summary.update({
                "concordance_index": self.model.concordance_index_,
                "log_likelihood": self.model.log_likelihood_,
                "AIC": self.model.AIC_
            })

        # 添加顶部特征
        importance = self.get_feature_importance()
        if importance:
            summary["top_features"] = list(importance.keys())[:10]

        return summary