"""
Sports Injury Risk Prediction - Quick Start Example
Demonstrates how to use this system for injury risk prediction
"""

import sys
from pathlib import Path

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.methods.traditional.random_forest import RandomForestInjuryPredictor as RandomForestModel
from src.core.trainer import SklearnTrainer
from src.core.metrics import MetricsCalculator
from src.core.interpret import InterpretabilityManager

def generate_sample_data(n_samples=100):
    """Generate sample data"""
    np.random.seed(42)

    data = {
        'player_id': [f'player_{i:03d}' for i in range(n_samples)],
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

    # Generate target variable
    risk_factors = (
        (data['age'] - 18) / 17 * 0.3 +
        np.array(data['recent_injury']) * 0.4 +
        (np.array(data['injury_history_count']) / 5) * 0.2 +
        (np.array(data['training_load']) - 50) / 50 * 0.1
    )

    risk_probabilities = 1 / (1 + np.exp(-(risk_factors + np.random.normal(0, 0.5, n_samples))))
    data['injury_risk'] = np.random.binomial(1, risk_probabilities, n_samples)

    return pd.DataFrame(data)

def main():
    """Main demonstration function"""
    print("🏃‍♂️ Sports Injury Risk Prediction - Quick Start")
    print("=" * 50)

    # 1. Generate sample data
    print("\n1️⃣ Generating sample data...")
    df = generate_sample_data(500)
    print(f"   Data shape: {df.shape}")
    print(f"   Injury rate: {df['injury_risk'].mean():.2%}")

    # 2. Feature engineering
    print("\n2️⃣ Performing feature engineering...")
    feature_engineer = FeatureEngineer()
    df_processed = feature_engineer.transform(df)
    print(f"   Processed shape: {df_processed.shape}")

    # 3. Data splitting
    print("\n3️⃣ Splitting train and test data...")
    target_col = 'injury_risk'
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 4. Model training
    print("\n4️⃣ Training Random Forest model...")
    model = RandomForestModel(n_estimators=100, random_state=42)
    trainer = SklearnTrainer(model=model, random_state=42)
    trainer.fit(X_train, y_train, validation_data=(X_test, y_test))

    # 5. Model evaluation
    print("\n5️⃣ Evaluating model performance...")
    metrics_calc = MetricsCalculator(task_type="binary")

    # Training set performance
    train_metrics = trainer.evaluate(X_train, y_train)
    print(f"   Training AUC-ROC: {train_metrics['auc_roc']:.4f}")

    # Test set performance
    test_metrics = trainer.evaluate(X_test, y_test)
    print(f"   Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"   Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"   Test Precision: {test_metrics['precision']:.4f}")
    print(f"   Test Recall: {test_metrics['recall']:.4f}")

    # 6. Cross validation (skip due to compatibility issues)
    print("\n6️⃣ Performing cross validation...")
    try:
        cv_results = trainer.cross_validate(X_train, y_train, cv=5)
        print(f"   CV AUC-ROC: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    except Exception as e:
        print(f"   Skipping cross validation (compatibility issue): {str(e)[:100]}...")

    # 7. Interpretability analysis
    print("\n7️⃣ Analyzing model interpretability...")
    try:
        # 直接从RandomForest模型获取特征重要性
        if hasattr(trainer.model, 'model') and hasattr(trainer.model.model, 'feature_importances_'):
            feature_importances = trainer.model.model.feature_importances_
            feature_names = X.columns.tolist()
            importance_pairs = list(zip(feature_names, feature_importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)

            print("   Important feature ranking:")
            for i, (feature, importance) in enumerate(importance_pairs[:5]):
                print(f"     {i+1}. {feature}: {importance:.4f}")
        else:
            print("   Unable to get feature importance")
    except Exception as e:
        print(f"   Skipping interpretability analysis (error): {str(e)[:100]}...")

    # 8. Single prediction example
    print("\n8️⃣ Single player prediction example...")
    sample_player = X_test.iloc[0:1]
    prediction = trainer.predict(sample_player)[0]
    probability = trainer.predict_proba(sample_player)[0, 1]

    print(f"   Player feature sample:")
    for feature, value in sample_player.iloc[0].head(5).items():
        print(f"     {feature}: {value}")

    print(f"   Prediction result: {'High risk' if prediction == 1 else 'Low risk'}")
    print(f"   Injury probability: {probability:.2%}")

    # 9. Risk level classification
    print("\n9️⃣ Test set risk level distribution...")
    all_probabilities = trainer.predict_proba(X_test)[:, 1]

    low_risk = np.sum(all_probabilities < 0.3)
    medium_risk = np.sum((all_probabilities >= 0.3) & (all_probabilities < 0.7))
    high_risk = np.sum(all_probabilities >= 0.7)

    print(f"   Low risk (<30%): {low_risk} players ({low_risk/len(X_test):.1%})")
    print(f"   Medium risk (30-70%): {medium_risk} players ({medium_risk/len(X_test):.1%})")
    print(f"   High risk (>70%): {high_risk} players ({high_risk/len(X_test):.1%})")

    # 10. Save model
    print("\n🔟 Saving trained model...")
    model_path = project_root / "models" / "demo_model.joblib"
    model_path.parent.mkdir(exist_ok=True)
    trainer.save(model_path)
    print(f"   Model saved to: {model_path}")

    print("\n✅ Quick start demo completed!")
    print("\n📋 Next steps you can try:")
    print("   - Use experiments/train_model.py for complete model training")
    print("   - Use experiments/evaluate_model.py for detailed model evaluation")
    print("   - Start API service for online prediction")

if __name__ == "__main__":
    main()