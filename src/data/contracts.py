"""
数据契约定义模块
定义输入数据的schema和验证规则，防止脏数据进入系统
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np


class PlayerPosition(Enum):
    """球员位置枚举"""
    GOALKEEPER = "GK"
    DEFENDER = "DEF"
    MIDFIELDER = "MID"
    FORWARD = "FWD"
    UNKNOWN = "UNK"


class InjuryType(Enum):
    """伤病类型枚举"""
    MUSCLE = "MUSCLE"
    JOINT = "JOINT"
    BONE = "BONE"
    LIGAMENT = "LIGAMENT"
    CONCUSSION = "CONCUSSION"
    OTHER = "OTHER"


class InjurySeverity(Enum):
    """伤病严重程度"""
    MINOR = "MINOR"  # < 7 days
    MODERATE = "MODERATE"  # 7-21 days
    SEVERE = "SEVERE"  # > 21 days


@dataclass
class PlayerProfile:
    """球员基本信息契约"""
    player_id: str
    name: str
    age: int
    height: float  # cm
    weight: float  # kg
    position: PlayerPosition
    dominant_foot: str  # L/R

    def __post_init__(self):
        if not 16 <= self.age <= 45:
            raise ValueError(f"Age {self.age} out of valid range [16, 45]")
        if not 150 <= self.height <= 220:
            raise ValueError(f"Height {self.height} out of valid range [150, 220]")
        if not 50 <= self.weight <= 120:
            raise ValueError(f"Weight {self.weight} out of valid range [50, 120]")


@dataclass
class GameLoad:
    """比赛负荷数据契约"""
    player_id: str
    date: datetime
    match_id: Optional[str]

    # 基础负荷指标
    minutes_played: float
    distance_total: float  # km
    distance_high_intensity: float  # km, >19.8 km/h
    sprints_count: int
    max_speed: float  # km/h

    # 心率相关
    avg_heart_rate: Optional[int] = None
    max_heart_rate: Optional[int] = None
    heart_rate_zones: Optional[Dict[str, float]] = None  # Zone1-5 percentages

    # 生理指标
    rpe: Optional[int] = None  # Rate of Perceived Exertion (1-10)
    wellness_score: Optional[int] = None  # (1-10)
    sleep_quality: Optional[int] = None  # (1-10)

    def __post_init__(self):
        if not 0 <= self.minutes_played <= 120:
            raise ValueError(f"Minutes played {self.minutes_played} invalid")
        if self.distance_total < 0 or self.distance_total > 20:
            raise ValueError(f"Total distance {self.distance_total} km invalid")


@dataclass
class TrainingLoad:
    """训练负荷数据契约"""
    player_id: str
    date: datetime
    session_type: str  # "strength", "endurance", "technical", "tactical"

    duration: int  # minutes
    rpe: Optional[int] = None
    load_score: Optional[float] = None  # RPE * duration

    # 专项训练指标
    strength_volume: Optional[float] = None  # total kg lifted
    running_distance: Optional[float] = None  # km
    ball_touches: Optional[int] = None


@dataclass
class InjuryRecord:
    """伤病记录契约"""
    player_id: str
    injury_id: str
    onset_date: datetime
    return_date: Optional[datetime]

    injury_type: InjuryType
    severity: InjurySeverity
    body_part: str
    mechanism: str  # "contact", "non-contact", "overuse"

    # 上下文信息
    occurred_during: str  # "match", "training", "other"
    surface_type: Optional[str] = None  # "grass", "artificial", "indoor"
    weather_condition: Optional[str] = None

    # 恢复相关
    days_out: Optional[int] = None
    treatment_type: Optional[str] = None

    def __post_init__(self):
        if self.return_date and self.return_date < self.onset_date:
            raise ValueError("Return date cannot be before onset date")

        if self.days_out is not None and self.days_out < 0:
            raise ValueError("Days out cannot be negative")


@dataclass
class DataSchema:
    """完整数据集契约"""
    players: List[PlayerProfile]
    game_loads: List[GameLoad]
    training_loads: List[TrainingLoad]
    injuries: List[InjuryRecord]

    # 数据质量要求
    min_history_days: int = 90  # 最少历史数据天数
    max_missing_rate: float = 0.3  # 最大缺失率

    def validate_completeness(self) -> Dict[str, Any]:
        """验证数据完整性"""
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # 检查基础数据
        if not self.players:
            validation_report["errors"].append("No player profiles found")
            validation_report["valid"] = False

        if not self.game_loads and not self.training_loads:
            validation_report["errors"].append("No load data found")
            validation_report["valid"] = False

        # 检查数据一致性
        player_ids = {p.player_id for p in self.players}
        load_player_ids = {gl.player_id for gl in self.game_loads} | {tl.player_id for tl in self.training_loads}

        missing_players = load_player_ids - player_ids
        if missing_players:
            validation_report["errors"].append(f"Missing player profiles for IDs: {missing_players}")
            validation_report["valid"] = False

        return validation_report


class SchemaValidator:
    """Schema验证器"""

    @staticmethod
    def validate_dataframe_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """验证DataFrame列结构"""
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True

    @staticmethod
    def validate_data_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> List[str]:
        """验证数据类型"""
        errors = []
        for col, expected_type in type_mapping.items():
            if col in df.columns:
                try:
                    if expected_type == 'datetime':
                        pd.to_datetime(df[col])
                    elif expected_type == 'numeric':
                        pd.to_numeric(df[col])
                    elif expected_type == 'string':
                        df[col].astype(str)
                except Exception as e:
                    errors.append(f"Column {col} type validation failed: {str(e)}")
        return errors

    @staticmethod
    def validate_value_ranges(df: pd.DataFrame, range_mapping: Dict[str, tuple]) -> List[str]:
        """验证数值范围"""
        errors = []
        for col, (min_val, max_val) in range_mapping.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if not out_of_range.empty:
                    errors.append(f"Column {col} has {len(out_of_range)} values out of range [{min_val}, {max_val}]")
        return errors


# 预定义的验证配置
GAME_LOAD_SCHEMA = {
    'required_columns': ['player_id', 'date', 'minutes_played', 'distance_total'],
    'type_mapping': {
        'player_id': 'string',
        'date': 'datetime',
        'minutes_played': 'numeric',
        'distance_total': 'numeric',
        'distance_high_intensity': 'numeric',
        'sprints_count': 'numeric'
    },
    'range_mapping': {
        'minutes_played': (0, 120),
        'distance_total': (0, 20),
        'distance_high_intensity': (0, 10),
        'max_speed': (0, 40)
    }
}

INJURY_SCHEMA = {
    'required_columns': ['player_id', 'onset_date', 'injury_type', 'severity'],
    'type_mapping': {
        'player_id': 'string',
        'onset_date': 'datetime',
        'return_date': 'datetime',
        'days_out': 'numeric'
    },
    'range_mapping': {
        'days_out': (0, 365)
    }
}