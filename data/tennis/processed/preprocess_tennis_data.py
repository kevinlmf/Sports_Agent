#!/usr/bin/env python3
"""
网球运动员数据预处理脚本
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

def load_data(data_dir):
    """加载原始数据"""
    print("加载数据...")

    main_df = pd.read_csv(f"{data_dir}/tennis_players_main.csv")
    ts_df = pd.read_csv(f"{data_dir}/tennis_players_timeseries.csv")

    print(f"✓ 主数据: {main_df.shape}")
    print(f"✓ 时间序列: {ts_df.shape}")

    return main_df, ts_df

def engineer_features(df):
    """特征工程"""
    print("\n特征工程...")

    # BMI
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

    # 胜率
    df['win_rate'] = df['matches_won_12m'] / df['matches_played_12m']

    # 比赛密度
    df['match_density'] = df['matches_played_12m'] / 52  # 每周平均比赛数

    # 训练比赛比
    df['training_match_ratio'] = df['training_hours_per_week'] / (df['matches_played_12m'] / 52)

    # 恢复充足度
    df['recovery_adequacy'] = df['recovery_time_hours_per_week'] / df['training_hours_per_week']

    # 疲劳指数
    df['fatigue_index'] = (df['fatigue_score'] + df['stress_level']) / 2

    # 睡眠充足度
    df['sleep_adequacy'] = df['avg_sleep_hours'] * df['sleep_quality_score'] / 10

    # 伤病倾向
    df['injury_proneness'] = (
        df['previous_injuries_count'] * 0.3 +
        df['days_injured_12m'] / 365 * 0.5 +
        (1 if df['injury_severity_last'] == 'Severe' else 0.5 if df['injury_severity_last'] == 'Moderate' else 0) * 0.2
    )

    # 年龄组
    df['age_group'] = pd.cut(df['age'], bins=[0, 22, 28, 35, 50], labels=['Young', 'Prime', 'Veteran', 'Senior'])

    # 工作负荷风险区间
    df['ac_ratio_risk'] = pd.cut(df['ac_ratio'],
                                   bins=[0, 0.8, 1.0, 1.3, 1.5, 10],
                                   labels=['Very Low', 'Low', 'Optimal', 'Elevated', 'Very High'])

    print(f"✓ 添加了 {len(df.columns) - len(df.columns)} 个新特征")

    return df

def prepare_datasets(df):
    """准备训练、验证、测试集"""
    print("\n准备数据集...")

    # 选择特征
    feature_columns = [
        # 基本信息
        'age', 'height_cm', 'weight_kg', 'bmi', 'years_pro',

        # 比赛数据
        'matches_played_12m', 'win_rate', 'tournaments_played_12m',
        'grand_slams_played_12m', 'match_density',
        'avg_match_duration_min', 'avg_games_per_match',
        'service_games_won_pct', 'break_points_saved_pct',

        # 训练数据
        'training_hours_per_week', 'court_time_hours_per_week',
        'gym_time_hours_per_week', 'training_match_ratio',
        'avg_training_intensity', 'training_strain',

        # 工作负荷
        'acute_workload', 'chronic_workload', 'ac_ratio',

        # 生理指标
        'resting_heart_rate', 'vo2_max', 'body_fat_percentage',

        # 伤病历史
        'previous_injuries_count', 'days_injured_12m',
        'time_since_last_injury_days', 'injury_proneness',

        # 恢复
        'avg_sleep_hours', 'sleep_quality_score', 'sleep_adequacy',
        'stress_level', 'fatigue_score', 'fatigue_index',
        'recovery_time_hours_per_week', 'recovery_adequacy',

        # 旅行
        'travel_days_12m', 'countries_visited_12m', 'time_zones_crossed_12m',
    ]

    X = df[feature_columns].fillna(0)
    y = df['injury_risk_label']

    # 分割数据
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    print(f"✓ 训练集: {X_train.shape}")
    print(f"✓ 验证集: {X_val.shape}")
    print(f"✓ 测试集: {X_test.shape}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 转换回DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, feature_columns

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                       scaler, feature_columns, output_dir):
    """保存处理后的数据"""
    print(f"\n保存处理后的数据到 {output_dir}...")

    # 合并X和y
    train_df = X_train.copy()
    train_df['injury_risk_label'] = y_train
    train_df.to_csv(f"{output_dir}/train.csv", index=False)

    val_df = X_val.copy()
    val_df['injury_risk_label'] = y_val
    val_df.to_csv(f"{output_dir}/val.csv", index=False)

    test_df = X_test.copy()
    test_df['injury_risk_label'] = y_test
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    # 保存标准化器参数
    import joblib
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    # 保存特征列表
    with open(f"{output_dir}/feature_columns.json", 'w') as f:
        json.dump(feature_columns, f, indent=2)

    print(f"✓ 训练集: {output_dir}/train.csv")
    print(f"✓ 验证集: {output_dir}/val.csv")
    print(f"✓ 测试集: {output_dir}/test.csv")
    print(f"✓ 标准化器: {output_dir}/scaler.pkl")

if __name__ == '__main__':
    import sys

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else './data/raw'
    processed_dir = sys.argv[2] if len(sys.argv) > 2 else './data/processed'

    print("=" * 70)
    print("网球运动员数据预处理")
    print("=" * 70)

    # 加载数据
    main_df, ts_df = load_data(raw_dir)

    # 特征工程
    main_df = engineer_features(main_df)

    # 准备数据集
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_columns = prepare_datasets(main_df)

    # 保存
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                       scaler, feature_columns, processed_dir)

    print("\n" + "=" * 70)
    print("数据预处理完成！")
    print("=" * 70)
