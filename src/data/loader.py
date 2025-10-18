"""
数据加载器模块
支持从CSV/Parquet/SQL等多种数据源加载数据到DataFrame
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
from abc import ABC, abstractmethod

from .contracts import (
    PlayerProfile, GameLoad, TrainingLoad, InjuryRecord,
    DataSchema, SchemaValidator, GAME_LOAD_SCHEMA, INJURY_SCHEMA
)

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """数据加载器基类"""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """加载数据"""
        pass


class CSVLoader(DataLoader):
    """CSV文件加载器"""

    def __init__(self, file_path: Union[str, Path], **kwargs):
        self.file_path = Path(file_path)
        self.kwargs = kwargs

    def load(self) -> pd.DataFrame:
        """从CSV文件加载数据"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path, **self.kwargs)
            logger.info(f"Loaded {len(df)} rows from {self.file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV {self.file_path}: {str(e)}")
            raise


class ParquetLoader(DataLoader):
    """Parquet文件加载器"""

    def __init__(self, file_path: Union[str, Path], **kwargs):
        self.file_path = Path(file_path)
        self.kwargs = kwargs

    def load(self) -> pd.DataFrame:
        """从Parquet文件加载数据"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.file_path}")

        try:
            df = pd.read_parquet(self.file_path, **self.kwargs)
            logger.info(f"Loaded {len(df)} rows from {self.file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load Parquet {self.file_path}: {str(e)}")
            raise


class SQLLoader(DataLoader):
    """SQL数据库加载器"""

    def __init__(self, connection_string: str, query: str):
        self.connection_string = connection_string
        self.query = query

    def load(self) -> pd.DataFrame:
        """从SQL数据库加载数据"""
        try:
            # 支持SQLite和其他数据库
            if 'sqlite' in self.connection_string:
                conn = sqlite3.connect(self.connection_string.replace('sqlite:///', ''))
            else:
                # 这里可以扩展支持其他数据库
                raise NotImplementedError("Only SQLite is currently supported")

            df = pd.read_sql_query(self.query, conn)
            conn.close()

            logger.info(f"Loaded {len(df)} rows from database")
            return df
        except Exception as e:
            logger.error(f"Failed to load from database: {str(e)}")
            raise


class SportsDataLoader:
    """体育伤病风险数据加载器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器

        Args:
            config: 配置字典，包含数据源路径和参数
        """
        self.config = config
        self.validator = SchemaValidator()

    def load_players(self) -> pd.DataFrame:
        """加载球员基本信息"""
        loader_config = self.config.get('players', {})
        loader = self._create_loader(loader_config)

        df = loader.load()

        # 数据类型转换
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if 'height' in df.columns:
            df['height'] = pd.to_numeric(df['height'], errors='coerce')
        if 'weight' in df.columns:
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

        return df

    def load_game_loads(self) -> pd.DataFrame:
        """加载比赛负荷数据"""
        loader_config = self.config.get('game_loads', {})
        loader = self._create_loader(loader_config)

        df = loader.load()

        # 验证schema
        required_cols = GAME_LOAD_SCHEMA['required_columns']
        self.validator.validate_dataframe_schema(df, required_cols)

        # 数据类型转换
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        numeric_cols = ['minutes_played', 'distance_total', 'distance_high_intensity',
                       'sprints_count', 'max_speed']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def load_training_loads(self) -> pd.DataFrame:
        """加载训练负荷数据"""
        loader_config = self.config.get('training_loads', {})
        if not loader_config:
            logger.warning("No training loads configuration found")
            return pd.DataFrame()

        loader = self._create_loader(loader_config)
        df = loader.load()

        # 数据类型转换
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'duration' in df.columns:
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        if 'rpe' in df.columns:
            df['rpe'] = pd.to_numeric(df['rpe'], errors='coerce')

        return df

    def load_injuries(self) -> pd.DataFrame:
        """加载伤病记录数据"""
        loader_config = self.config.get('injuries', {})
        loader = self._create_loader(loader_config)

        df = loader.load()

        # 验证schema
        required_cols = INJURY_SCHEMA['required_columns']
        self.validator.validate_dataframe_schema(df, required_cols)

        # 数据类型转换
        date_cols = ['onset_date', 'return_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'days_out' in df.columns:
            df['days_out'] = pd.to_numeric(df['days_out'], errors='coerce')

        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据"""
        data = {}

        try:
            data['players'] = self.load_players()
            data['game_loads'] = self.load_game_loads()
            data['training_loads'] = self.load_training_loads()
            data['injuries'] = self.load_injuries()

            logger.info("Successfully loaded all data sources")

            # 数据概览
            for name, df in data.items():
                logger.info(f"{name}: {len(df)} rows, {len(df.columns)} columns")

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

        return data

    def _create_loader(self, config: Dict[str, Any]) -> DataLoader:
        """根据配置创建相应的数据加载器"""
        source_type = config.get('type', 'csv')

        if source_type == 'csv':
            return CSVLoader(config['path'], **config.get('params', {}))
        elif source_type == 'parquet':
            return ParquetLoader(config['path'], **config.get('params', {}))
        elif source_type == 'sql':
            return SQLLoader(config['connection'], config['query'])
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")


class DataPreprocessor:
    """数据预处理器"""

    @staticmethod
    def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
        """清理球员数据"""
        df = df.copy()

        # 移除重复记录
        df = df.drop_duplicates(subset=['player_id'])

        # 处理缺失值
        df['position'] = df['position'].fillna('UNKNOWN')
        df['dominant_foot'] = df['dominant_foot'].fillna('R')

        # 数据合理性检查
        df = df[(df['age'] >= 16) & (df['age'] <= 45)]
        df = df[(df['height'] >= 150) & (df['height'] <= 220)]
        df = df[(df['weight'] >= 50) & (df['weight'] <= 120)]

        return df

    @staticmethod
    def clean_load_data(df: pd.DataFrame) -> pd.DataFrame:
        """清理负荷数据"""
        df = df.copy()

        # 排序
        df = df.sort_values(['player_id', 'date'])

        # 移除异常值
        df = df[df['minutes_played'] >= 0]
        df = df[df['distance_total'] >= 0]

        # 计算衍生指标
        df['load_intensity'] = df['distance_high_intensity'] / (df['distance_total'] + 1e-6)
        df['sprint_rate'] = df['sprints_count'] / (df['minutes_played'] + 1e-6)

        return df

    @staticmethod
    def clean_injury_data(df: pd.DataFrame) -> pd.DataFrame:
        """清理伤病数据"""
        df = df.copy()

        # 计算伤病天数（如果缺失）
        if 'days_out' not in df.columns or df['days_out'].isna().any():
            mask = df['return_date'].notna() & df['onset_date'].notna()
            df.loc[mask, 'days_out'] = (df.loc[mask, 'return_date'] -
                                       df.loc[mask, 'onset_date']).dt.days

        # 根据天数推断严重程度（如果缺失）
        if 'severity' not in df.columns:
            df['severity'] = 'UNKNOWN'

        mask_minor = df['days_out'] < 7
        mask_moderate = (df['days_out'] >= 7) & (df['days_out'] <= 21)
        mask_severe = df['days_out'] > 21

        df.loc[mask_minor, 'severity'] = 'MINOR'
        df.loc[mask_moderate, 'severity'] = 'MODERATE'
        df.loc[mask_severe, 'severity'] = 'SEVERE'

        return df


def create_sample_data() -> Dict[str, pd.DataFrame]:
    """创建示例数据（用于测试）"""
    np.random.seed(42)

    # 创建示例球员数据
    players_data = []
    positions = ['GK', 'DEF', 'MID', 'FWD']

    for i in range(50):
        players_data.append({
            'player_id': f'P{i:03d}',
            'name': f'Player {i}',
            'age': np.random.randint(18, 35),
            'height': np.random.normal(180, 10),
            'weight': np.random.normal(75, 10),
            'position': np.random.choice(positions),
            'dominant_foot': np.random.choice(['L', 'R'], p=[0.1, 0.9])
        })

    players_df = pd.DataFrame(players_data)

    # 创建示例负荷数据
    game_loads_data = []
    start_date = datetime.now() - timedelta(days=365)

    for player_id in players_df['player_id']:
        for day in range(0, 365, 7):  # 每周一场比赛
            if np.random.random() > 0.1:  # 90%参赛率
                date = start_date + timedelta(days=day)
                game_loads_data.append({
                    'player_id': player_id,
                    'date': date,
                    'minutes_played': max(0, np.random.normal(75, 20)),
                    'distance_total': max(0, np.random.normal(10, 2)),
                    'distance_high_intensity': max(0, np.random.normal(1.5, 0.5)),
                    'sprints_count': max(0, int(np.random.normal(15, 5))),
                    'max_speed': max(0, np.random.normal(28, 3)),
                    'rpe': np.random.randint(1, 11)
                })

    game_loads_df = pd.DataFrame(game_loads_data)

    # 创建示例伤病数据
    injuries_data = []
    injury_types = ['MUSCLE', 'JOINT', 'LIGAMENT', 'BONE']

    for i, player_id in enumerate(players_df['player_id'].sample(20)):  # 40%球员有伤病记录
        injury_date = start_date + timedelta(days=np.random.randint(0, 300))
        days_out = max(1, int(np.random.exponential(14)))

        injuries_data.append({
            'player_id': player_id,
            'injury_id': f'INJ{i:03d}',
            'onset_date': injury_date,
            'return_date': injury_date + timedelta(days=days_out),
            'injury_type': np.random.choice(injury_types),
            'severity': 'MINOR' if days_out < 7 else 'MODERATE' if days_out <= 21 else 'SEVERE',
            'body_part': np.random.choice(['hamstring', 'ankle', 'knee', 'shoulder']),
            'mechanism': np.random.choice(['contact', 'non-contact', 'overuse']),
            'occurred_during': np.random.choice(['match', 'training']),
            'days_out': days_out
        })

    injuries_df = pd.DataFrame(injuries_data)

    return {
        'players': players_df,
        'game_loads': game_loads_df,
        'training_loads': pd.DataFrame(),  # 空的训练数据
        'injuries': injuries_df
    }