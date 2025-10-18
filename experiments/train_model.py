"""
Model Training Experiment Script
Supports multi-model training and hyperparameter optimization
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import DataLoader, SportsDataLoader
from src.data.features import FeatureEngineer
from src.data.validate import DataQualityValidator as DataValidator, ValidationLevel
from src.methods.traditional.random_forest import RandomForestInjuryPredictor as RandomForestModel
from src.methods.traditional.logistic_regression import LogisticInjuryPredictor as LogisticRegressionModel
from src.methods.traditional.xgboost_model import XGBoostInjuryPredictor as XGBoostModel
from src.core.trainer import SklearnTrainer, HyperparameterTuner
from src.core.metrics import MetricsCalculator, MetricsTracker
from src.core.interpret import InterpretabilityManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Setup directory structure"""
    base_dir = Path(config.get('output_dir', 'results'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    directories = {
        'base': base_dir,
        'experiment': base_dir / f"experiment_{timestamp}",
        'models': base_dir / f"experiment_{timestamp}" / 'models',
        'metrics': base_dir / f"experiment_{timestamp}" / 'metrics',
        'plots': base_dir / f"experiment_{timestamp}" / 'plots',
        'interpretability': base_dir / f"experiment_{timestamp}" / 'interpretability'
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def load_and_prepare_data(config: Dict[str, Any]) -> tuple:
    """Load and prepare data"""
    logger.info("Loading and preparing data...")

    # Initialize components
    feature_engineer = FeatureEngineer()
    data_validator = DataValidator()

    # Load data
    data_config = config['data']
    if data_config['type'] == 'csv':
        # Use CSVLoader to load data
        from src.data.loader import CSVLoader
        csv_loader = CSVLoader(data_config['path'])
        df = csv_loader.load()
    elif data_config['type'] == 'synthetic':
        # Generate synthetic data for demonstration
        df = generate_synthetic_data(
            n_samples=data_config.get('n_samples', 1000),
            random_state=config.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unsupported data type: {data_config['type']}")

    logger.info(f"Loaded data with shape: {df.shape}")

    # Simple data validation (for synthetic data)
    if data_config['type'] == 'synthetic':
        logger.info("Skipping detailed validation for synthetic data")
        # Basic checks
        if df.empty:
            raise ValueError("Generated synthetic data is empty")
        if df.isnull().all().any():
            logger.warning("Some columns are entirely null in synthetic data")
    else:
        # Perform complete validation on real data
        validation_results = data_validator.validate_all({'data': df})
        if data_validator.has_errors():
            logger.error("Data validation failed")
            for result in validation_results:
                if result.level == ValidationLevel.ERROR:
                    logger.error(f"{result.check_name}: {result.message}")
            raise ValueError("Data validation failed with errors")

        # Log warning information
        for result in validation_results:
            if result.level == ValidationLevel.WARNING:
                logger.warning(f"{result.check_name}: {result.message}")
            elif result.level == ValidationLevel.INFO:
                logger.info(f"{result.check_name}: {result.message}")

    # Feature engineering
    feature_config = config.get('feature_engineering', {})
    if feature_config.get('enabled', True):
        df = feature_engineer.transform(df)
        logger.info(f"Feature engineering completed. New shape: {df.shape}")

    # Separate features and labels
    target_column = data_config['target_column']
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Data split
    split_config = config.get('data_split', {})
    test_size = split_config.get('test_size', 0.2)
    val_size = split_config.get('val_size', 0.1)
    random_state = config.get('random_state', 42)

    # Train/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train/validation split
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )
    else:
        X_train, X_val, y_train, y_val = X_temp, None, y_temp, None

    logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape if X_val is not None else 'None'}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns.tolist()


def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic sports injury data for demonstration"""
    np.random.seed(random_state)

    data = {
        'player_id': [f'player_{i:04d}' for i in range(n_samples)],
        'age': np.random.randint(18, 35, n_samples),
        'position': np.random.choice(['GK', 'DEF', 'MID', 'FWD'], n_samples),
        'height': np.random.normal(180, 10, n_samples),
        'weight': np.random.normal(75, 8, n_samples),
        'games_played': np.random.randint(0, 50, n_samples),
        'minutes_played': np.random.randint(0, 4000, n_samples),
        'recent_injury': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'training_load': np.random.normal(70, 15, n_samples),
        'match_intensity': np.random.normal(60, 20, n_samples),
        'injury_history_count': np.random.poisson(1.5, n_samples)
    }

    # Generate target variable (injury risk)
    # Create reasonable injury risk based on combination of some features
    risk_factors = (
        (data['age'] - 18) / 17 * 0.3 +  # Age factor
        data['recent_injury'] * 0.4 +     # Recent injury
        (data['injury_history_count'] / 5) * 0.2 +  # Injury history
        (data['training_load'] - 50) / 50 * 0.1     # Training load
    )

    # Add noise and convert to probability
    risk_probabilities = 1 / (1 + np.exp(-(risk_factors + np.random.normal(0, 0.5, n_samples))))
    data['injury_risk'] = np.random.binomial(1, risk_probabilities, n_samples)

    return pd.DataFrame(data)


def get_model(model_name: str, config: Dict[str, Any]):
    """Get model instance"""
    model_config = config['models'][model_name]
    model_type = model_config['type']
    params = model_config.get('params', {})

    if model_type == 'random_forest':
        # RandomForestInjuryPredictor only supports specific parameters
        supported_params = ['n_estimators', 'random_state', 'class_weight', 'n_jobs']
        filtered_params = {k: v for k, v in params.items() if k in supported_params}
        return RandomForestModel(**filtered_params)
    elif model_type == 'logistic_regression':
        # LogisticInjuryPredictor only supports specific parameters
        supported_params = ['random_state', 'class_weight', 'max_iter']
        filtered_params = {k: v for k, v in params.items() if k in supported_params}
        return LogisticRegressionModel(**filtered_params)
    elif model_type == 'xgboost':
        # XGBoostInjuryPredictor only supports specific parameters
        supported_params = ['n_estimators', 'random_state', 'use_gpu', 'n_jobs']
        filtered_params = {k: v for k, v in params.items() if k in supported_params}
        return XGBoostModel(**filtered_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_single_model(
    model_name: str,
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    feature_names: Optional[list] = None
) -> Dict[str, Any]:
    """Train single model"""
    logger.info(f"Training model: {model_name}")

    # Get model
    model = get_model(model_name, config)

    # Create trainer
    trainer = SklearnTrainer(
        model=model,
        random_state=config.get('random_state', 42),
        verbose=True
    )

    # Prepare validation data
    validation_data = (X_val, y_val) if X_val is not None else None

    # Train model
    trainer.fit(X_train, y_train, validation_data=validation_data)

    # Evaluate model
    train_metrics = trainer.evaluate(X_train, y_train)
    results = {
        'model_name': model_name,
        'trainer': trainer,
        'train_metrics': train_metrics,
        'training_history': trainer.training_history
    }

    if validation_data:
        val_metrics = trainer.evaluate(X_val, y_val)
        results['val_metrics'] = val_metrics

    # Cross-validation (handle models incompatible with sklearn)
    try:
        cv_results = trainer.cross_validate(X_train, y_train, cv=5)
        results['cv_results'] = cv_results
    except Exception as e:
        logger.warning(f"Cross-validation failed for {model_name}: {str(e)}")
        results['cv_results'] = {'test_score': [np.nan], 'train_score': [np.nan]}

    logger.info(f"Model {model_name} training completed")
    logger.info(f"Train AUC: {train_metrics.get('auc_roc', 'N/A'):.4f}")
    if 'val_metrics' in results:
        logger.info(f"Val AUC: {results['val_metrics'].get('auc_roc', 'N/A'):.4f}")
    cv_results_dict = results.get('cv_results', {})
    if 'mean' in cv_results_dict and 'std' in cv_results_dict:
        logger.info(f"CV AUC: {cv_results_dict['mean']:.4f} ± {cv_results_dict['std']:.4f}")
    else:
        logger.info("CV AUC: Not available")

    return results


def hyperparameter_tuning(
    model_name: str,
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """Hyperparameter tuning"""
    logger.info(f"Starting hyperparameter tuning for {model_name}")

    tuning_config = config['hyperparameter_tuning']
    if not tuning_config.get('enabled', False):
        logger.info("Hyperparameter tuning disabled")
        return {}

    model_config = config['models'][model_name]
    param_grid = model_config.get('param_grid', {})

    if not param_grid:
        logger.warning(f"No parameter grid defined for {model_name}")
        return {}

    # Get base model class
    model = get_model(model_name, config)

    # Create hyperparameter tuner
    tuner = HyperparameterTuner(
        trainer_class=SklearnTrainer,
        param_grid=param_grid,
        cv=tuning_config.get('cv', 5),
        scoring=tuning_config.get('scoring', 'roc_auc'),
        n_jobs=tuning_config.get('n_jobs', -1),
        random_state=config.get('random_state', 42)
    )

    # Execute tuning
    tuner.fit(X_train, y_train, model=model)

    # Get best trainer
    best_trainer = tuner.get_best_trainer()

    logger.info(f"Hyperparameter tuning completed for {model_name}")
    logger.info(f"Best parameters: {best_trainer.best_params}")

    return {
        'best_trainer': best_trainer,
        'tuning_results': tuner.get_tuning_results(),
        'best_params': best_trainer.best_params
    }


def evaluate_models(
    results: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    directories: Dict[str, Path]
) -> pd.DataFrame:
    """Evaluate all models and generate comparison report"""
    logger.info("Evaluating models on test set...")

    evaluation_results = []

    for model_name, model_results in results.items():
        if 'trainer' not in model_results:
            continue

        trainer = model_results['trainer']

        # Test set evaluation
        test_metrics = trainer.evaluate(X_test, y_test)

        # Organize results
        eval_result = {
            'model': model_name,
            'train_auc': model_results['train_metrics'].get('auc_roc', np.nan),
            'val_auc': model_results.get('val_metrics', {}).get('auc_roc', np.nan),
            'test_auc': test_metrics.get('auc_roc', np.nan),
            'cv_auc_mean': model_results['cv_results'].get('mean', np.nan),
            'cv_auc_std': model_results['cv_results'].get('std', np.nan),
            'test_precision': test_metrics.get('precision', np.nan),
            'test_recall': test_metrics.get('recall', np.nan),
            'test_f1': test_metrics.get('f1', np.nan),
            'test_brier_score': test_metrics.get('brier_score', np.nan)
        }

        evaluation_results.append(eval_result)

        # Save individual model results
        model_save_path = directories['models'] / f"{model_name}_model.joblib"
        trainer.save(model_save_path)

        logger.info(f"Model {model_name} - Test AUC: {test_metrics.get('auc_roc', 'N/A'):.4f}")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(evaluation_results)

    # Save comparison results
    comparison_path = directories['metrics'] / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)

    # Print best model
    if not comparison_df.empty:
        best_model = comparison_df.loc[comparison_df['test_auc'].idxmax()]
        logger.info(f"Best model: {best_model['model']} with test AUC: {best_model['test_auc']:.4f}")

    return comparison_df


def generate_interpretability_report(
    best_model_name: str,
    results: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
    directories: Dict[str, Path]
):
    """Generate interpretability report"""
    logger.info("Generating interpretability report...")

    if best_model_name not in results or 'trainer' not in results[best_model_name]:
        logger.warning("No best model found for interpretability analysis")
        return

    trainer = results[best_model_name]['trainer']
    model = trainer.model

    # Create interpretability manager
    interpreter = InterpretabilityManager(model, model_type='sklearn')
    interpreter.set_feature_names(feature_names)

    # Generate complete interpretability report
    report = interpreter.generate_interpretation_report(
        X=X_test,
        y=y_test,
        output_dir=str(directories['interpretability']),
        methods=['shap', 'feature_importance']
    )

    logger.info(f"Interpretability report saved to {directories['interpretability']}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train sports injury risk prediction models')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--models', type=str, nargs='+', help='Models to train (if not specified, train all models in config)')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup directories
    directories = setup_directories(config)

    logger.info(f"Experiment directory: {directories['experiment']}")

    # Save configuration file copy
    config_copy_path = directories['experiment'] / 'config.yaml'
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    try:
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_prepare_data(config)

        # Determine models to train
        models_to_train = args.models if args.models else list(config['models'].keys())

        # Train model
        results = {}
        for model_name in models_to_train:
            if model_name not in config['models']:
                logger.warning(f"Model {model_name} not found in config, skipping...")
                continue

            # Hyperparameter tuning
            if args.tune:
                tuning_results = hyperparameter_tuning(model_name, config, X_train, y_train)
                if tuning_results and 'best_trainer' in tuning_results:
                    # Use best trainer after tuning
                    best_trainer = tuning_results['best_trainer']
                    results[model_name] = {
                        'model_name': model_name,
                        'trainer': best_trainer,
                        'train_metrics': best_trainer.evaluate(X_train, y_train),
                        'tuning_results': tuning_results['tuning_results']
                    }

                    if X_val is not None:
                        results[model_name]['val_metrics'] = best_trainer.evaluate(X_val, y_val)

                    # Cross-validation
                    cv_results = best_trainer.cross_validate(X_train, y_train, cv=5)
                    results[model_name]['cv_results'] = cv_results

                    continue

            # Standard training
            model_results = train_single_model(
                model_name, config, X_train, y_train, X_val, y_val, feature_names
            )
            results[model_name] = model_results

        # Evaluate model
        comparison_df = evaluate_models(results, X_test, y_test, directories)

        # Generate interpretability report (for best model)
        if not comparison_df.empty:
            best_model_name = comparison_df.loc[comparison_df['test_auc'].idxmax()]['model']
            generate_interpretability_report(
                best_model_name, results, X_test, y_test, feature_names, directories
            )

        logger.info("Training experiment completed successfully!")
        logger.info(f"Results saved to: {directories['experiment']}")

    except Exception as e:
        logger.error(f"Training experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()