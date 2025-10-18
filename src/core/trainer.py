"""
General Trainer Module
Unified training interface supporting binary classification and survival analysis
"""

import os
import json
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib

from .metrics import MetricsCalculator
from .calibration import CalibrationManager


class BaseTrainer:
    """Base Trainer Class"""

    def __init__(
        self,
        task_type: str = "binary",  # binary, survival
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.model = None
        self.training_history = []
        self.best_params = None
        self.metrics_calc = MetricsCalculator(task_type=task_type)
        self.calibration_mgr = CalibrationManager()

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseTrainer":
        """Train model"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return binary predictions
            predictions = self.model.predict(X)
            return np.column_stack([1 - predictions, predictions])

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model"""
        if self.task_type == "binary":
            y_pred_proba = self.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.predict(X)
            y_pred_proba = None

        return self.metrics_calc.calculate_all_metrics(
            y_true=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            sample_weight=sample_weight
        )

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: str = "roc_auc",
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Cross validation"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        if self.task_type == "binary":
            cv_splitter = StratifiedKFold(
                n_splits=cv,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold
            cv_splitter = KFold(
                n_splits=cv,
                shuffle=True,
                random_state=self.random_state
            )

        scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=self.n_jobs,
            **kwargs
        )

        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'cv_type': type(cv_splitter).__name__
        }

    def calibrate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: str = "isotonic",
        cv: int = 5
    ) -> "BaseTrainer":
        """Probability calibration"""
        if self.task_type != "binary":
            self.logger.warning("Calibration only supports binary classification")
            return self

        if self.model is None:
            raise ValueError("Model not fitted yet!")

        self.model = self.calibration_mgr.calibrate_model(
            model=self.model,
            X=X,
            y=y,
            method=method,
            cv=cv
        )

        self.logger.info(f"Model calibrated using {method} method")
        return self

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'task_type': self.task_type,
            'random_state': self.random_state,
            'training_history': self.training_history,
            'best_params': self.best_params,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> "BaseTrainer":
        """Load model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.task_type = model_data['task_type']
        self.random_state = model_data.get('random_state', 42)
        self.training_history = model_data.get('training_history', [])
        self.best_params = model_data.get('best_params')

        self.logger.info(f"Model loaded from {filepath}")
        return self


class SklearnTrainer(BaseTrainer):
    """Scikit-learn Model Trainer"""

    def __init__(
        self,
        model: BaseEstimator,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = model
        self.hyperparams = hyperparams or {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "SklearnTrainer":
        """Train sklearn model"""
        # Set hyperparameters
        if self.hyperparams:
            self.model = self.base_model.set_params(**self.hyperparams)
        else:
            self.model = self.base_model

        # Train model
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight

        self.model.fit(X, y, **fit_params)

        # Record training history
        train_metrics = self.evaluate(X, y, sample_weight)
        history_entry = {
            'epoch': 0,
            'train_metrics': train_metrics,
            'timestamp': datetime.now().isoformat()
        }

        if validation_data is not None:
            X_val, y_val = validation_data[:2]
            val_sample_weight = validation_data[2] if len(validation_data) > 2 else None
            val_metrics = self.evaluate(X_val, y_val, val_sample_weight)
            history_entry['val_metrics'] = val_metrics

        self.training_history.append(history_entry)

        if self.verbose:
            self.logger.info("Training completed")
            self.logger.info(f"Train metrics: {train_metrics}")
            if validation_data is not None:
                self.logger.info(f"Validation metrics: {val_metrics}")

        return self


class DeepLearningTrainer(BaseTrainer):
    """Deep Learning Model Trainer"""

    def __init__(
        self,
        model_class,
        model_params: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_params = model_params or {}
        self.training_params = training_params or {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "DeepLearningTrainer":
        """Train deep learning model"""
        # Initialize model
        self.model = self.model_class(**self.model_params)

        # Set training parameters
        epochs = self.training_params.get('epochs', 100)
        batch_size = self.training_params.get('batch_size', 32)
        early_stopping = self.training_params.get('early_stopping', True)
        patience = self.training_params.get('patience', 10)

        # Training process
        best_val_metric = float('-inf') if self.task_type == "binary" else float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train one epoch
            train_loss = self.model.train_epoch(
                X, y, batch_size=batch_size, sample_weight=sample_weight
            )

            # Calculate metrics
            train_metrics = self.evaluate(X, y, sample_weight)
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'timestamp': datetime.now().isoformat()
            }

            if validation_data is not None:
                X_val, y_val = validation_data[:2]
                val_sample_weight = validation_data[2] if len(validation_data) > 2 else None
                val_metrics = self.evaluate(X_val, y_val, val_sample_weight)
                history_entry['val_metrics'] = val_metrics

                # Early stopping check
                if early_stopping:
                    current_val_metric = val_metrics.get('auc_roc', val_metrics.get('c_index', 0))
                    if current_val_metric > best_val_metric:
                        best_val_metric = current_val_metric
                        patience_counter = 0
                        # Save best model state
                        self.model.save_checkpoint()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        # Restore best model state
                        self.model.load_checkpoint()
                        break

            self.training_history.append(history_entry)

            if self.verbose and epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: {train_metrics}")

        return self


class HyperparameterTuner:
    """Hyperparameter Tuner"""

    def __init__(
        self,
        trainer_class,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.trainer_class = trainer_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_trainer = None
        self.tuning_results = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> "HyperparameterTuner":
        """Perform hyperparameter tuning"""
        from sklearn.model_selection import ParameterGrid

        param_combinations = list(ParameterGrid(self.param_grid))
        best_score = float('-inf')

        for i, params in enumerate(param_combinations):
            # Create trainer
            trainer = self.trainer_class(
                hyperparams=params,
                random_state=self.random_state,
                verbose=False,
                **kwargs
            )

            # Cross validation
            trainer.fit(X, y)
            cv_results = trainer.cross_validate(
                X, y, cv=self.cv, scoring=self.scoring
            )

            # Record results
            result = {
                'params': params,
                'mean_score': cv_results['mean'],
                'std_score': cv_results['std'],
                'scores': cv_results['scores']
            }
            self.tuning_results.append(result)

            # Update best model
            if cv_results['mean'] > best_score:
                best_score = cv_results['mean']
                self.best_trainer = trainer
                self.best_trainer.best_params = params

            print(f"Combination {i+1}/{len(param_combinations)}: "
                  f"Score = {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")

        return self

    def get_best_trainer(self) -> BaseTrainer:
        """Get best trainer"""
        if self.best_trainer is None:
            raise ValueError("Tuner not fitted yet!")
        return self.best_trainer

    def get_tuning_results(self) -> pd.DataFrame:
        """Get tuning results"""
        return pd.DataFrame(self.tuning_results)


# Alias for backward compatibility
InjuryModelTrainer = SklearnTrainer