"""
特征工程模块
实现rolling窗口、EMA、back-to-back、伤病史编码等特征工程
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """特征工程配置"""
    # Rolling窗口配置
    rolling_windows: List[int] = None  # [7, 14, 28]

    # EMA配置
    ema_spans: List[int] = None  # [7, 14, 28]

    # 伤病史特征配置
    injury_history_days: int = 365  # 伤病史回溯天数

    # Back-to-back比赛配置
    back_to_back_threshold: int = 3  # 连续比赛天数阈值

    # 负荷累积配置
    acute_chronic_ratios: List[Tuple[int, int]] = None  # [(7, 28), (14, 42)]

    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [7, 14, 28]
        if self.ema_spans is None:
            self.ema_spans = [7, 14, 28]
        if self.acute_chronic_ratios is None:
            self.acute_chronic_ratios = [(7, 28), (14, 42)]


class BaseFeatureEngineer:
    """基础特征工程器"""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.scalers = {}
        self.encoders = {}

    def fit_transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """拟合并转换特征"""
        return self.transform(data, fit=True)

    def transform(self, data: Dict[str, pd.DataFrame], fit: bool = False) -> Dict[str, pd.DataFrame]:
        """转换特征"""
        transformed_data = {}

        for name, df in data.items():
            if not df.empty:
                transformed_data[name] = df.copy()

        return transformed_data


class PlayerFeatureEngineer(BaseFeatureEngineer):
    """球员特征工程器"""

    def transform(self, players_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """转换球员特征"""
        df = players_df.copy()

        # BMI计算
        if 'height' in df.columns and 'weight' in df.columns:
            df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

        # 位置编码
        if 'position' in df.columns:
            if fit or 'position' not in self.encoders:
                self.encoders['position'] = LabelEncoder()
                df['position_encoded'] = self.encoders['position'].fit_transform(df['position'].fillna('UNKNOWN'))
            else:
                df['position_encoded'] = self.encoders['position'].transform(df['position'].fillna('UNKNOWN'))

        # 主脚编码
        if 'dominant_foot' in df.columns:
            df['dominant_foot_R'] = (df['dominant_foot'] == 'R').astype(int)

        # 年龄分组
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                   bins=[0, 23, 28, 32, 100],
                                   labels=['young', 'prime', 'veteran', 'senior'])
            df['age_group_encoded'] = df['age_group'].cat.codes

        # 身体指标标准化
        numeric_features = ['age', 'height', 'weight', 'bmi']
        existing_features = [f for f in numeric_features if f in df.columns]

        if existing_features:
            if fit or 'player_numeric' not in self.scalers:
                self.scalers['player_numeric'] = StandardScaler()
                df[existing_features] = self.scalers['player_numeric'].fit_transform(df[existing_features].fillna(df[existing_features].mean()))
            else:
                df[existing_features] = self.scalers['player_numeric'].transform(df[existing_features].fillna(df[existing_features].mean()))

        return df


class LoadFeatureEngineer(BaseFeatureEngineer):
    """负荷特征工程器"""

    def transform(self, loads_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """转换负荷特征"""
        if loads_df.empty:
            return loads_df

        df = loads_df.copy()

        # 确保按球员和日期排序
        df = df.sort_values(['player_id', 'date'])

        # 基础衍生特征
        df = self._create_basic_features(df)

        # Rolling窗口特征
        df = self._create_rolling_features(df)

        # EMA特征
        df = self._create_ema_features(df)

        # Acute-Chronic比率
        df = self._create_acute_chronic_ratios(df)

        # Back-to-back比赛特征
        df = self._create_back_to_back_features(df)

        # 负荷变化特征
        df = self._create_load_change_features(df)

        return df

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基础衍生特征"""
        # 强度指标
        if all(col in df.columns for col in ['distance_high_intensity', 'distance_total']):
            df['intensity_ratio'] = df['distance_high_intensity'] / (df['distance_total'] + 1e-6)

        if all(col in df.columns for col in ['sprints_count', 'minutes_played']):
            df['sprint_rate'] = df['sprints_count'] / (df['minutes_played'] + 1e-6)

        if all(col in df.columns for col in ['distance_total', 'minutes_played']):
            df['avg_speed'] = (df['distance_total'] * 1000 / 60) / (df['minutes_played'] + 1e-6)

        # 综合负荷指标
        load_features = ['minutes_played', 'distance_total', 'distance_high_intensity', 'sprints_count']
        existing_load_features = [f for f in load_features if f in df.columns]

        if len(existing_load_features) >= 2:
            # 使用主成分分析的思路，简化为加权平均
            weights = [0.3, 0.3, 0.25, 0.15]  # 对应上述特征的权重
            df['composite_load'] = sum(df[f] * w for f, w in
                                     zip(existing_load_features[:len(weights)],
                                         weights[:len(existing_load_features)]))

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建Rolling窗口特征"""
        for window in self.config.rolling_windows:
            # 基础负荷指标的rolling统计
            load_cols = ['minutes_played', 'distance_total', 'distance_high_intensity', 'composite_load']

            for col in load_cols:
                if col in df.columns:
                    # 按球员分组计算rolling特征
                    df[f'{col}_rolling_{window}d_mean'] = (
                        df.groupby('player_id')[col]
                        .rolling(window=window, min_periods=1)
                        .mean().reset_index(0, drop=True)
                    )

                    df[f'{col}_rolling_{window}d_std'] = (
                        df.groupby('player_id')[col]
                        .rolling(window=window, min_periods=1)
                        .std().reset_index(0, drop=True)
                    )

                    df[f'{col}_rolling_{window}d_max'] = (
                        df.groupby('player_id')[col]
                        .rolling(window=window, min_periods=1)
                        .max().reset_index(0, drop=True)
                    )

        return df

    def _create_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建指数移动平均特征"""
        for span in self.config.ema_spans:
            load_cols = ['minutes_played', 'distance_total', 'composite_load']

            for col in load_cols:
                if col in df.columns:
                    df[f'{col}_ema_{span}d'] = (
                        df.groupby('player_id')[col]
                        .ewm(span=span, adjust=False)
                        .mean().reset_index(0, drop=True)
                    )

        return df

    def _create_acute_chronic_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建急性-慢性负荷比率"""
        for acute_days, chronic_days in self.config.acute_chronic_ratios:
            load_cols = ['composite_load', 'distance_total']

            for col in load_cols:
                if col in df.columns:
                    acute = (df.groupby('player_id')[col]
                           .rolling(window=acute_days, min_periods=1)
                           .mean().reset_index(0, drop=True))

                    chronic = (df.groupby('player_id')[col]
                             .rolling(window=chronic_days, min_periods=1)
                             .mean().reset_index(0, drop=True))

                    df[f'{col}_ac_ratio_{acute_days}_{chronic_days}'] = acute / (chronic + 1e-6)

        return df

    def _create_back_to_back_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建Back-to-back比赛特征"""
        # 计算距离上次比赛的天数
        df['days_since_last_game'] = (
            df.groupby('player_id')['date']
            .diff().dt.days.fillna(999)
        )

        # Back-to-back标记
        df['is_back_to_back'] = (df['days_since_last_game'] <= self.config.back_to_back_threshold).astype(int)

        # 连续比赛计数
        df['consecutive_games'] = 0
        for player_id in df['player_id'].unique():
            mask = df['player_id'] == player_id
            player_data = df[mask].copy()

            consecutive = 0
            consecutive_counts = []

            for _, row in player_data.iterrows():
                if row['is_back_to_back']:
                    consecutive += 1
                else:
                    consecutive = 0
                consecutive_counts.append(consecutive)

            df.loc[mask, 'consecutive_games'] = consecutive_counts

        return df

    def _create_load_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建负荷变化特征"""
        load_cols = ['minutes_played', 'distance_total', 'composite_load']

        for col in load_cols:
            if col in df.columns:
                # 相对于前一场比赛的变化
                df[f'{col}_change'] = (
                    df.groupby('player_id')[col]
                    .pct_change().fillna(0)
                )

                # 相对于7天平均的偏差
                if f'{col}_rolling_7d_mean' in df.columns:
                    df[f'{col}_deviation_from_7d'] = (
                        (df[col] - df[f'{col}_rolling_7d_mean']) /
                        (df[f'{col}_rolling_7d_mean'] + 1e-6)
                    )

        return df


class InjuryFeatureEngineer(BaseFeatureEngineer):
    """伤病特征工程器"""

    def transform(self, injuries_df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
        """转换伤病特征"""
        if injuries_df.empty:
            return injuries_df

        df = injuries_df.copy()
        reference_date = reference_date or datetime.now()

        # 伤病严重程度编码
        severity_mapping = {'MINOR': 1, 'MODERATE': 2, 'SEVERE': 3}
        if 'severity' in df.columns:
            df['severity_score'] = df['severity'].map(severity_mapping).fillna(0)

        # 伤病类型编码
        if 'injury_type' in df.columns:
            if 'injury_type' not in self.encoders:
                self.encoders['injury_type'] = LabelEncoder()
                df['injury_type_encoded'] = self.encoders['injury_type'].fit_transform(df['injury_type'].fillna('UNKNOWN'))
            else:
                df['injury_type_encoded'] = self.encoders['injury_type'].transform(df['injury_type'].fillna('UNKNOWN'))

        # 身体部位编码
        if 'body_part' in df.columns:
            if 'body_part' not in self.encoders:
                self.encoders['body_part'] = LabelEncoder()
                df['body_part_encoded'] = self.encoders['body_part'].fit_transform(df['body_part'].fillna('UNKNOWN'))

        # 伤病机制编码
        if 'mechanism' in df.columns:
            mechanism_mapping = {'contact': 1, 'non-contact': 2, 'overuse': 3}
            df['mechanism_score'] = df['mechanism'].map(mechanism_mapping).fillna(0)

        # 距离参考日期的天数
        if 'onset_date' in df.columns:
            df['days_since_injury'] = (reference_date - df['onset_date']).dt.days

        return df


class InjuryHistoryFeatureEngineer(BaseFeatureEngineer):
    """伤病历史特征工程器"""

    def create_injury_history_features(self,
                                     loads_df: pd.DataFrame,
                                     injuries_df: pd.DataFrame) -> pd.DataFrame:
        """为每个负荷记录创建伤病历史特征"""
        if injuries_df.empty:
            return self._add_empty_injury_features(loads_df)

        df = loads_df.copy()

        # 初始化伤病历史特征
        injury_features = [
            'total_injuries', 'recent_injuries_30d', 'recent_injuries_90d', 'recent_injuries_365d',
            'days_since_last_injury', 'avg_injury_severity', 'max_injury_severity',
            'muscle_injuries', 'joint_injuries', 'ligament_injuries',
            'contact_injuries', 'noncontact_injuries', 'overuse_injuries',
            'total_days_injured', 'avg_days_per_injury', 'injury_rate_per_year'
        ]

        for feature in injury_features:
            df[feature] = 0.0

        # 为每个球员和日期计算伤病历史特征
        for player_id in df['player_id'].unique():
            player_injuries = injuries_df[injuries_df['player_id'] == player_id].copy()
            if player_injuries.empty:
                continue

            player_loads = df[df['player_id'] == player_id].copy()

            for idx, load_row in player_loads.iterrows():
                current_date = load_row['date']

                # 获取当前日期之前的所有伤病
                past_injuries = player_injuries[
                    player_injuries['onset_date'] < current_date
                ].copy()

                if past_injuries.empty:
                    continue

                # 计算各种伤病历史特征
                features = self._calculate_injury_features(past_injuries, current_date)

                # 更新DataFrame
                for feature_name, value in features.items():
                    df.loc[idx, feature_name] = value

        return df

    def _calculate_injury_features(self,
                                 past_injuries: pd.DataFrame,
                                 current_date: datetime) -> Dict[str, float]:
        """计算单个时间点的伤病历史特征"""
        features = {}

        # 基础统计
        features['total_injuries'] = len(past_injuries)

        if features['total_injuries'] == 0:
            return {f: 0.0 for f in ['total_injuries', 'recent_injuries_30d', 'recent_injuries_90d',
                                   'recent_injuries_365d', 'days_since_last_injury', 'avg_injury_severity']}

        # 时间窗口内的伤病数
        for days in [30, 90, 365]:
            cutoff_date = current_date - timedelta(days=days)
            recent_injuries = past_injuries[past_injuries['onset_date'] >= cutoff_date]
            features[f'recent_injuries_{days}d'] = len(recent_injuries)

        # 距离最后一次伤病的天数
        last_injury_date = past_injuries['onset_date'].max()
        features['days_since_last_injury'] = (current_date - last_injury_date).days

        # 伤病严重程度
        if 'severity_score' in past_injuries.columns:
            features['avg_injury_severity'] = past_injuries['severity_score'].mean()
            features['max_injury_severity'] = past_injuries['severity_score'].max()
        else:
            features['avg_injury_severity'] = 0.0
            features['max_injury_severity'] = 0.0

        # 按伤病类型分类
        if 'injury_type' in past_injuries.columns:
            type_counts = past_injuries['injury_type'].value_counts()
            features['muscle_injuries'] = type_counts.get('MUSCLE', 0)
            features['joint_injuries'] = type_counts.get('JOINT', 0)
            features['ligament_injuries'] = type_counts.get('LIGAMENT', 0)

        # 按伤病机制分类
        if 'mechanism' in past_injuries.columns:
            mechanism_counts = past_injuries['mechanism'].value_counts()
            features['contact_injuries'] = mechanism_counts.get('contact', 0)
            features['noncontact_injuries'] = mechanism_counts.get('non-contact', 0)
            features['overuse_injuries'] = mechanism_counts.get('overuse', 0)

        # 总受伤天数和平均天数
        if 'days_out' in past_injuries.columns:
            total_days = past_injuries['days_out'].sum()
            features['total_days_injured'] = total_days
            features['avg_days_per_injury'] = total_days / len(past_injuries) if len(past_injuries) > 0 else 0

            # 伤病率（每年）
            injury_span_days = (current_date - past_injuries['onset_date'].min()).days
            if injury_span_days > 0:
                features['injury_rate_per_year'] = (len(past_injuries) * 365) / injury_span_days
            else:
                features['injury_rate_per_year'] = 0

        return features

    def _add_empty_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为没有伤病历史的数据添加空特征"""
        injury_features = [
            'total_injuries', 'recent_injuries_30d', 'recent_injuries_90d', 'recent_injuries_365d',
            'days_since_last_injury', 'avg_injury_severity', 'max_injury_severity',
            'muscle_injuries', 'joint_injuries', 'ligament_injuries',
            'contact_injuries', 'noncontact_injuries', 'overuse_injuries',
            'total_days_injured', 'avg_days_per_injury', 'injury_rate_per_year'
        ]

        for feature in injury_features:
            df[feature] = 0.0

        return df


class ComprehensiveFeatureEngineer:
    """综合特征工程器"""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.player_engineer = PlayerFeatureEngineer(config)
        self.load_engineer = LoadFeatureEngineer(config)
        self.injury_engineer = InjuryFeatureEngineer(config)
        self.injury_history_engineer = InjuryHistoryFeatureEngineer(config)

    def fit_transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """拟合并转换所有特征"""
        return self.transform(data, fit=True)

    def transform(self, data: Dict[str, pd.DataFrame], fit: bool = False) -> Dict[str, Any]:
        """转换所有特征"""
        result = {}

        # 转换球员特征
        if 'players' in data and not data['players'].empty:
            result['players'] = self.player_engineer.transform(data['players'], fit=fit)

        # 转换负荷特征
        if 'game_loads' in data and not data['game_loads'].empty:
            loads_with_features = self.load_engineer.transform(data['game_loads'], fit=fit)

            # 添加伤病历史特征
            if 'injuries' in data:
                loads_with_features = self.injury_history_engineer.create_injury_history_features(
                    loads_with_features, data['injuries']
                )

            result['game_loads'] = loads_with_features

        # 转换训练负荷特征（如果有）
        if 'training_loads' in data and not data['training_loads'].empty:
            result['training_loads'] = self.load_engineer.transform(data['training_loads'], fit=fit)

        # 转换伤病特征
        if 'injuries' in data and not data['injuries'].empty:
            result['injuries'] = self.injury_engineer.transform(data['injuries'], fit=fit)

        # 创建建模数据集
        if 'game_loads' in result:
            result['modeling_dataset'] = self._create_modeling_dataset(result)

        return result

    def _create_modeling_dataset(self, transformed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """创建用于建模的数据集"""
        loads_df = transformed_data['game_loads'].copy()

        # 合并球员信息
        if 'players' in transformed_data:
            players_df = transformed_data['players'].copy()
            # 选择关键的球员特征
            player_features = ['player_id', 'age', 'bmi', 'position_encoded', 'dominant_foot_R']
            existing_features = [f for f in player_features if f in players_df.columns]

            if existing_features:
                loads_df = loads_df.merge(players_df[existing_features], on='player_id', how='left')

        # 选择建模特征
        feature_columns = self._select_modeling_features(loads_df)

        # 处理缺失值
        loads_df[feature_columns] = loads_df[feature_columns].fillna(0)

        return loads_df[['player_id', 'date'] + feature_columns]

    def _select_modeling_features(self, df: pd.DataFrame) -> List[str]:
        """选择用于建模的特征"""
        # 基础负荷特征
        base_features = [
            'minutes_played', 'distance_total', 'distance_high_intensity', 'composite_load',
            'intensity_ratio', 'sprint_rate', 'avg_speed'
        ]

        # Rolling窗口特征
        rolling_features = []
        for window in self.config.rolling_windows:
            for metric in ['composite_load', 'distance_total']:
                rolling_features.extend([
                    f'{metric}_rolling_{window}d_mean',
                    f'{metric}_rolling_{window}d_std'
                ])

        # EMA特征
        ema_features = []
        for span in self.config.ema_spans:
            for metric in ['composite_load', 'distance_total']:
                ema_features.append(f'{metric}_ema_{span}d')

        # Acute-Chronic比率
        ac_features = []
        for acute_days, chronic_days in self.config.acute_chronic_ratios:
            for metric in ['composite_load', 'distance_total']:
                ac_features.append(f'{metric}_ac_ratio_{acute_days}_{chronic_days}')

        # Back-to-back特征
        btb_features = ['days_since_last_game', 'is_back_to_back', 'consecutive_games']

        # 负荷变化特征
        change_features = []
        for metric in ['composite_load', 'distance_total']:
            change_features.extend([
                f'{metric}_change',
                f'{metric}_deviation_from_7d'
            ])

        # 伤病历史特征
        injury_history_features = [
            'total_injuries', 'recent_injuries_30d', 'recent_injuries_90d',
            'days_since_last_injury', 'avg_injury_severity',
            'muscle_injuries', 'noncontact_injuries', 'injury_rate_per_year'
        ]

        # 球员特征
        player_features = ['age', 'bmi', 'position_encoded', 'dominant_foot_R']

        # 组合所有特征
        all_features = (base_features + rolling_features + ema_features +
                       ac_features + btb_features + change_features +
                       injury_history_features + player_features)

        # 只保留存在的特征
        existing_features = [f for f in all_features if f in df.columns]

        logger.info(f"Selected {len(existing_features)} features for modeling")

        return existing_features


# 为了向后兼容，提供一个简化的FeatureEngineer类
class FeatureEngineer:
    """简化的特征工程器，用于快速开始示例"""

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """简单的特征转换，用于演示"""
        result_df = df.copy()

        # 位置编码（简单数值编码）
        if 'position' in result_df.columns:
            position_map = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
            result_df['position_encoded'] = result_df['position'].map(position_map).fillna(-1)

        # 基本特征工程
        if 'age' in result_df.columns and 'height' in result_df.columns and 'weight' in result_df.columns:
            # BMI计算
            result_df['bmi'] = result_df['weight'] / ((result_df['height'] / 100) ** 2)

            # 创建一些复合特征
            if 'games_played' in result_df.columns and 'minutes_played' in result_df.columns:
                result_df['minutes_per_game'] = result_df['minutes_played'] / (result_df['games_played'] + 1)

            if 'training_load' in result_df.columns and 'match_intensity' in result_df.columns:
                result_df['total_load'] = result_df['training_load'] + result_df['match_intensity']

        # 删除不需要的列（包括原始的字符串列）
        cols_to_drop = []
        if 'player_id' in result_df.columns:
            cols_to_drop.append('player_id')
        if 'position' in result_df.columns:
            cols_to_drop.append('position')

        if cols_to_drop:
            result_df = result_df.drop(columns=cols_to_drop)

        return result_df