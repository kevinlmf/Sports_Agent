"""
Data drift detection module
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """数据漂移检测器"""

    def __init__(self):
        self.reference_stats = None

    def fit_reference(self, X: pd.DataFrame) -> None:
        """设置参考数据集的统计信息"""
        self.reference_stats = {
            'mean': X.mean(),
            'std': X.std(),
            'min': X.min(),
            'max': X.max()
        }
        logger.info("Reference statistics computed")

    def detect_drift(self, X: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """检测数据漂移"""
        if self.reference_stats is None:
            raise ValueError("Must fit reference data first")

        current_stats = {
            'mean': X.mean(),
            'std': X.std(),
            'min': X.min(),
            'max': X.max()
        }

        # 简单的统计差异检测
        drift_scores = {}
        for feature in X.columns:
            if feature in self.reference_stats['mean']:
                ref_mean = self.reference_stats['mean'][feature]
                curr_mean = current_stats['mean'][feature]
                ref_std = self.reference_stats['std'][feature]

                if ref_std > 0:
                    drift_scores[feature] = abs(curr_mean - ref_mean) / ref_std
                else:
                    drift_scores[feature] = 0

        # 判断是否发生漂移
        drifted_features = [f for f, score in drift_scores.items() if score > threshold]

        return {
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'drift_scores': drift_scores,
            'threshold': threshold
        }