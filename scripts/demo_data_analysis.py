#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Sports Injury Risk Prediction
Generates comprehensive visualizations and statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def generate_sample_data(n_samples=1000):
    """Generate sample sports injury data for demonstration"""
    np.random.seed(42)

    # Feature generation
    data = {
        'age': np.random.randint(18, 40, n_samples),
        'training_hours_per_week': np.random.exponential(8, n_samples).clip(2, 30),
        'previous_injuries': np.random.poisson(1.5, n_samples),
        'bmi': np.random.normal(23, 3, n_samples).clip(17, 35),
        'recovery_days': np.random.gamma(2, 5, n_samples),
        'injury_severity': np.random.choice(['None', 'Minor', 'Moderate', 'Severe'],
                                           n_samples,
                                           p=[0.6, 0.25, 0.1, 0.05])
    }

    # Target: Injury risk (binary)
    risk_score = (
        0.3 * (data['age'] > 30) +
        0.2 * (data['training_hours_per_week'] > 15) +
        0.4 * (data['previous_injuries'] > 2) +
        0.1 * (data['bmi'] > 27) +
        np.random.normal(0, 0.15, n_samples)
    )
    data['injury_occurred'] = (risk_score > 0.5).astype(int)

    return data


def create_eda_visualizations(data, output_dir='results/eda'):
    """Create comprehensive EDA visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # 1. Distribution Analysis
    print("\n📊 1. Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold')

    numeric_features = ['age', 'training_hours_per_week', 'previous_injuries',
                       'bmi', 'recovery_days']

    for idx, feature in enumerate(numeric_features):
        ax = axes[idx // 3, idx % 3]

        # Histogram with KDE
        ax.hist(data[feature], bins=30, alpha=0.6, color='steelblue', edgecolor='black')
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')

        # Add statistics
        mean_val = np.mean(data[feature])
        median_val = np.median(data[feature])
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=9)

    # Injury occurrence pie chart
    ax = axes[1, 2]
    injury_counts = [np.sum(data['injury_occurred'] == 0), np.sum(data['injury_occurred'] == 1)]
    ax.pie(injury_counts, labels=['No Injury', 'Injury'], autopct='%1.1f%%',
           colors=['#90EE90', '#FF6B6B'], startangle=90)
    ax.set_title('Injury Occurrence Rate', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'feature_distributions.png'}")
    plt.close()

    # 2. Correlation Analysis
    print("\n📈 2. Correlation Analysis")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create correlation matrix
    corr_data = {k: v for k, v in data.items() if k in numeric_features + ['injury_occurred']}
    corr_matrix = np.corrcoef([corr_data[k] for k in corr_data.keys()])

    # Heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=[k.replace('_', '\n') for k in corr_data.keys()],
                yticklabels=[k.replace('_', '\n') for k in corr_data.keys()],
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'correlation_matrix.png'}")
    plt.close()

    # 3. Risk Factor Analysis
    print("\n⚠️  3. Risk Factor Analysis")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Injury Risk Factor Analysis', fontsize=16, fontweight='bold')

    # Age vs Injury
    ax = axes[0, 0]
    injury_yes = [data['age'][i] for i in range(len(data['age'])) if data['injury_occurred'][i] == 1]
    injury_no = [data['age'][i] for i in range(len(data['age'])) if data['injury_occurred'][i] == 0]
    ax.hist([injury_no, injury_yes], bins=20, label=['No Injury', 'Injury'],
            color=['green', 'red'], alpha=0.6)
    ax.set_xlabel('Age', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Age Distribution by Injury Status', fontweight='bold')
    ax.legend()

    # Training Hours vs Injury
    ax = axes[0, 1]
    injury_yes = [data['training_hours_per_week'][i] for i in range(len(data['training_hours_per_week']))
                  if data['injury_occurred'][i] == 1]
    injury_no = [data['training_hours_per_week'][i] for i in range(len(data['training_hours_per_week']))
                 if data['injury_occurred'][i] == 0]
    ax.boxplot([injury_no, injury_yes], labels=['No Injury', 'Injury'],
                patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
    ax.set_ylabel('Training Hours/Week', fontsize=11)
    ax.set_title('Training Load by Injury Status', fontweight='bold')

    # Previous Injuries vs New Injury
    ax = axes[1, 0]
    injury_yes = [data['previous_injuries'][i] for i in range(len(data['previous_injuries']))
                  if data['injury_occurred'][i] == 1]
    injury_no = [data['previous_injuries'][i] for i in range(len(data['previous_injuries']))
                 if data['injury_occurred'][i] == 0]
    ax.hist([injury_no, injury_yes], bins=range(0, 8), label=['No Injury', 'Injury'],
            color=['green', 'red'], alpha=0.6)
    ax.set_xlabel('Previous Injuries', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Previous Injury History Impact', fontweight='bold')
    ax.legend()

    # BMI vs Injury
    ax = axes[1, 1]
    injury_yes = [data['bmi'][i] for i in range(len(data['bmi']))
                  if data['injury_occurred'][i] == 1]
    injury_no = [data['bmi'][i] for i in range(len(data['bmi']))
                 if data['injury_occurred'][i] == 0]
    ax.scatter([0]*len(injury_no), injury_no, alpha=0.3, color='green', label='No Injury')
    ax.scatter([1]*len(injury_yes), injury_yes, alpha=0.3, color='red', label='Injury')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Injury', 'Injury'])
    ax.set_ylabel('BMI', fontsize=11)
    ax.set_title('BMI Distribution by Injury Status', fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'risk_factor_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'risk_factor_analysis.png'}")
    plt.close()

    # 4. Statistical Summary
    print("\n📋 4. Statistical Summary")
    stats = {
        'dataset_size': len(data['age']),
        'injury_rate': f"{np.mean(data['injury_occurred']) * 100:.1f}%",
        'feature_stats': {}
    }

    for feature in numeric_features:
        stats['feature_stats'][feature] = {
            'mean': float(np.mean(data[feature])),
            'std': float(np.std(data[feature])),
            'min': float(np.min(data[feature])),
            'max': float(np.max(data[feature])),
            'median': float(np.median(data[feature]))
        }

    # Save statistics
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ Saved: {output_dir / 'statistics.json'}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Dataset Size: {stats['dataset_size']}")
    print(f"Injury Rate: {stats['injury_rate']}")
    print("\nFeature Statistics:")
    for feature, values in stats['feature_stats'].items():
        print(f"\n{feature.replace('_', ' ').title()}:")
        print(f"  Mean: {values['mean']:.2f} | Std: {values['std']:.2f}")
        print(f"  Range: [{values['min']:.2f}, {values['max']:.2f}]")

    print("\n" + "="*70)
    print("✅ EDA Complete! Check results/eda/ for visualizations")
    print("="*70)


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("SPORTS INJURY RISK PREDICTION - DATA ANALYSIS")
    print("="*70)

    # Generate sample data
    print("\n📦 Generating sample data...")
    data = generate_sample_data(n_samples=1000)
    print(f"   ✓ Generated {len(data['age'])} samples")

    # Create visualizations
    create_eda_visualizations(data)

    print("\n✨ Analysis complete! Visualizations saved to results/eda/")


if __name__ == '__main__':
    main()
