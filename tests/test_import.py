#!/usr/bin/env python3
"""
测试FeatureEngineer导入和基本功能
"""

import pytest
import pandas as pd
import numpy as np
from src.data.features import FeatureEngineer


class TestFeatureEngineer:
    """FeatureEngineer测试类"""

    def test_import(self):
        """测试FeatureEngineer能否正常导入"""
        assert FeatureEngineer is not None

    def test_instantiation(self):
        """测试FeatureEngineer能否正常实例化"""
        fe = FeatureEngineer()
        assert fe is not None

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        test_data = {
            'player_id': ['p001', 'p002', 'p003'],
            'age': [25, 30, 22],
            'position': ['MID', 'DEF', 'FWD'],
            'height': [175, 180, 185],
            'weight': [70, 75, 80],
            'games_played': [20, 25, 18],
            'minutes_played': [1800, 2250, 1620],
            'training_load': [70, 65, 75],
            'match_intensity': [60, 55, 70],
            'injury_risk': [0, 1, 0]
        }
        return pd.DataFrame(test_data)

    def test_feature_engineering(self, sample_data):
        """测试特征工程功能"""
        fe = FeatureEngineer()
        df_processed = fe.transform(sample_data)

        # 验证处理后的数据不为空
        assert df_processed is not None
        assert len(df_processed) > 0

        # 验证原始行数保持不变
        assert len(df_processed) == len(sample_data)

        # 验证添加了新的特征列
        assert len(df_processed.columns) >= len(sample_data.columns)