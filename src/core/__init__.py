"""
Core training and evaluation components for sports injury risk prediction.
"""

from .trainer import BaseTrainer, SklearnTrainer, DeepLearningTrainer, HyperparameterTuner
from .metrics import MetricsCalculator, MetricsTracker
from .calibration import CalibrationManager, PlattScaling, IsotonicCalibration, BetaCalibration
from .interpret import InterpretabilityManager, AdvancedInterpreter

__all__ = [
    # Trainers
    'BaseTrainer',
    'SklearnTrainer',
    'DeepLearningTrainer',
    'HyperparameterTuner',

    # Metrics
    'MetricsCalculator',
    'MetricsTracker',

    # Calibration
    'CalibrationManager',
    'PlattScaling',
    'IsotonicCalibration',
    'BetaCalibration',

    # Interpretability
    'InterpretabilityManager',
    'AdvancedInterpreter'
]