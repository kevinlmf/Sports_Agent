#!/usr/bin/env python3
"""
网球运动员数据收集脚本
从多个数据源收集ATP/WTA运动员的数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
import random

def generate_tennis_player_data(n_players=200):
    """
    生成模拟的网球运动员数据

    包含：
    - 基本信息：姓名、年龄、性别、身高、体重
    - 比赛数据：比赛场次、胜率、排名
    - 训练数据：训练时长、训练强度
    - 伤病历史：过往伤病记录
    - 生理指标：心率、体能指数
    """

    np.random.seed(42)
    random.seed(42)

    print(f"生成 {n_players} 名网球运动员的数据...")

    # 姓名列表
    first_names = ['Rafael', 'Roger', 'Novak', 'Andy', 'Dominic', 'Stefanos',
                   'Daniil', 'Alexander', 'Carlos', 'Casper', 'Serena', 'Venus',
                   'Naomi', 'Ashleigh', 'Simona', 'Iga', 'Garbine', 'Karolina']
    last_names = ['Nadal', 'Federer', 'Djokovic', 'Murray', 'Thiem', 'Tsitsipas',
                  'Medvedev', 'Zverev', 'Alcaraz', 'Ruud', 'Williams', 'Osaka',
                  'Barty', 'Halep', 'Swiatek', 'Muguruza', 'Pliskova', 'Kvitova']

    data = []

    for i in range(n_players):
        player = {
            # 基本信息
            'player_id': f'P{i+1:04d}',
            'name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'age': np.random.randint(18, 38),
            'gender': np.random.choice(['M', 'F'], p=[0.5, 0.5]),
            'height_cm': np.random.randint(160, 200),
            'weight_kg': np.random.randint(55, 95),
            'country': np.random.choice(['USA', 'Spain', 'Serbia', 'UK', 'France',
                                        'Germany', 'Australia', 'Russia', 'Japan']),

            # 职业信息
            'years_pro': np.random.randint(1, 20),
            'current_ranking': i + 1 + np.random.randint(-10, 10),
            'highest_ranking': max(1, i + 1 - np.random.randint(0, 20)),
            'career_prize_money': np.random.randint(100000, 100000000),

            # 比赛数据（过去12个月）
            'matches_played_12m': np.random.randint(30, 80),
            'matches_won_12m': np.random.randint(15, 60),
            'tournaments_played_12m': np.random.randint(10, 25),
            'grand_slams_played_12m': np.random.randint(0, 4),

            # 比赛表现
            'avg_match_duration_min': np.random.randint(60, 180),
            'avg_games_per_match': np.random.randint(15, 35),
            'avg_sets_per_match': round(np.random.uniform(2.0, 3.5), 2),
            'service_games_won_pct': round(np.random.uniform(60, 90), 1),
            'break_points_saved_pct': round(np.random.uniform(40, 80), 1),

            # 训练数据（每周平均）
            'training_hours_per_week': round(np.random.uniform(15, 35), 1),
            'court_time_hours_per_week': round(np.random.uniform(10, 25), 1),
            'gym_time_hours_per_week': round(np.random.uniform(5, 15), 1),
            'recovery_time_hours_per_week': round(np.random.uniform(3, 10), 1),

            # 训练强度
            'avg_training_intensity': round(np.random.uniform(6, 10), 1),  # 1-10 scale
            'peak_training_load': round(np.random.uniform(500, 2000), 0),  # arbitrary units
            'training_monotony': round(np.random.uniform(1.0, 2.5), 2),
            'training_strain': round(np.random.uniform(1000, 8000), 0),

            # 急慢比 (Acute:Chronic Workload Ratio)
            'acute_workload': round(np.random.uniform(500, 1500), 0),
            'chronic_workload': round(np.random.uniform(800, 1200), 0),
        }

        # 计算急慢比
        player['ac_ratio'] = round(player['acute_workload'] / player['chronic_workload'], 2)

        # 生理指标
        player['resting_heart_rate'] = np.random.randint(45, 70)
        player['max_heart_rate'] = np.random.randint(170, 200)
        player['vo2_max'] = round(np.random.uniform(50, 75), 1)  # ml/kg/min
        player['body_fat_percentage'] = round(np.random.uniform(8, 20), 1)
        player['muscle_mass_kg'] = round(np.random.uniform(25, 45), 1)

        # 伤病历史
        player['previous_injuries_count'] = np.random.randint(0, 10)
        player['days_injured_12m'] = np.random.randint(0, 150)
        player['injury_types'] = random.choice([
            'None', 'Knee', 'Ankle', 'Shoulder', 'Elbow', 'Wrist', 'Back',
            'Hamstring', 'Achilles', 'Abdominal', 'Multiple'
        ])
        player['time_since_last_injury_days'] = np.random.randint(0, 365)
        player['injury_severity_last'] = random.choice(['None', 'Minor', 'Moderate', 'Severe'])

        # 睡眠和恢复
        player['avg_sleep_hours'] = round(np.random.uniform(6, 10), 1)
        player['sleep_quality_score'] = round(np.random.uniform(5, 10), 1)  # 1-10
        player['stress_level'] = round(np.random.uniform(3, 9), 1)  # 1-10
        player['fatigue_score'] = round(np.random.uniform(2, 8), 1)  # 1-10

        # 旅行负荷
        player['travel_days_12m'] = np.random.randint(50, 200)
        player['countries_visited_12m'] = np.random.randint(5, 30)
        player['time_zones_crossed_12m'] = np.random.randint(10, 50)

        # 场地类型（比赛分布）
        player['matches_hard_court_pct'] = round(np.random.uniform(30, 60), 1)
        player['matches_clay_court_pct'] = round(np.random.uniform(15, 40), 1)
        player['matches_grass_court_pct'] = round(np.random.uniform(5, 20), 1)

        # 计算伤病风险（目标变量）
        # 基于多个因素的综合评分
        risk_score = 0

        # 年龄因素
        risk_score += (player['age'] - 25) * 0.5 if player['age'] > 25 else 0

        # 伤病历史
        risk_score += player['previous_injuries_count'] * 2
        risk_score += player['days_injured_12m'] * 0.05

        # 训练负荷
        if player['ac_ratio'] > 1.5 or player['ac_ratio'] < 0.8:
            risk_score += 10
        risk_score += max(0, player['training_strain'] - 5000) * 0.002

        # 比赛负荷
        risk_score += player['matches_played_12m'] * 0.1
        risk_score += player['travel_days_12m'] * 0.02

        # 恢复因素
        risk_score -= player['avg_sleep_hours'] * 2
        risk_score += player['fatigue_score'] * 1.5
        risk_score += player['stress_level'] * 1.2

        # 添加随机噪声
        risk_score += np.random.normal(0, 5)

        # 归一化到0-1
        player['injury_risk_score'] = max(0, min(1, risk_score / 100))

        # 二分类标签（高风险 vs 低风险）
        player['injury_risk_label'] = 1 if player['injury_risk_score'] > 0.5 else 0

        # 生成时间戳
        player['data_collection_date'] = (datetime.now() - timedelta(days=np.random.randint(0, 7))).strftime('%Y-%m-%d')

        data.append(player)

    df = pd.DataFrame(data)

    print(f"✓ 成功生成 {len(df)} 名运动员的数据")
    print(f"  - 特征数量: {len(df.columns)}")
    print(f"  - 高风险运动员: {df['injury_risk_label'].sum()}")
    print(f"  - 低风险运动员: {len(df) - df['injury_risk_label'].sum()}")

    return df

def add_time_series_data(df, n_weeks=12):
    """
    为每个运动员添加时间序列数据（过去12周的训练负荷）
    """
    print(f"\n生成过去 {n_weeks} 周的时间序列数据...")

    time_series_data = []

    for idx, player in df.iterrows():
        player_id = player['player_id']
        base_workload = player['chronic_workload']

        for week in range(n_weeks, 0, -1):
            week_date = datetime.now() - timedelta(weeks=week)

            # 模拟工作负荷的波动
            variation = np.random.uniform(0.7, 1.3)
            weekly_load = base_workload * variation

            time_series_data.append({
                'player_id': player_id,
                'week_date': week_date.strftime('%Y-%m-%d'),
                'week_number': n_weeks - week + 1,
                'weekly_training_load': round(weekly_load, 0),
                'matches_this_week': np.random.randint(0, 3),
                'training_days': np.random.randint(4, 7),
                'rest_days': 7 - np.random.randint(4, 7),
                'wellness_score': round(np.random.uniform(5, 10), 1),
                'soreness_score': round(np.random.uniform(1, 8), 1),
            })

    ts_df = pd.DataFrame(time_series_data)

    print(f"✓ 成功生成 {len(ts_df)} 条时间序列记录")

    return ts_df

def save_data(df, ts_df, output_dir):
    """保存数据到CSV文件"""
    print(f"\n保存数据到 {output_dir}...")

    # 保存主数据
    main_file = f"{output_dir}/tennis_players_main.csv"
    df.to_csv(main_file, index=False)
    print(f"✓ 主数据已保存: {main_file}")

    # 保存时间序列数据
    ts_file = f"{output_dir}/tennis_players_timeseries.csv"
    ts_df.to_csv(ts_file, index=False)
    print(f"✓ 时间序列数据已保存: {ts_file}")

    # 保存数据字典
    data_dict = {
        'dataset': 'Tennis Player Injury Risk Data',
        'description': 'Comprehensive tennis player data including match statistics, training load, physiological metrics, and injury history',
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_players': len(df),
        'total_features': len(df.columns),
        'target_variable': 'injury_risk_label',
        'high_risk_count': int(df['injury_risk_label'].sum()),
        'low_risk_count': int(len(df) - df['injury_risk_label'].sum()),
        'features': {
            'basic_info': ['player_id', 'name', 'age', 'gender', 'height_cm', 'weight_kg', 'country'],
            'career_stats': ['years_pro', 'current_ranking', 'highest_ranking', 'career_prize_money'],
            'match_data': ['matches_played_12m', 'matches_won_12m', 'tournaments_played_12m', 'grand_slams_played_12m'],
            'training_data': ['training_hours_per_week', 'court_time_hours_per_week', 'gym_time_hours_per_week'],
            'workload': ['acute_workload', 'chronic_workload', 'ac_ratio', 'training_strain'],
            'physiology': ['resting_heart_rate', 'max_heart_rate', 'vo2_max', 'body_fat_percentage'],
            'injury_history': ['previous_injuries_count', 'days_injured_12m', 'injury_types', 'time_since_last_injury_days'],
            'recovery': ['avg_sleep_hours', 'sleep_quality_score', 'stress_level', 'fatigue_score'],
            'target': ['injury_risk_score', 'injury_risk_label']
        }
    }

    dict_file = f"{output_dir}/data_dictionary.json"
    with open(dict_file, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"✓ 数据字典已保存: {dict_file}")

    return main_file, ts_file

if __name__ == '__main__':
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else './data/raw'
    n_players = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    print("=" * 70)
    print("网球运动员数据生成器")
    print("=" * 70)

    # 生成主数据
    df = generate_tennis_player_data(n_players)

    # 生成时间序列数据
    ts_df = add_time_series_data(df, n_weeks=12)

    # 保存数据
    main_file, ts_file = save_data(df, ts_df, output_dir)

    print("\n" + "=" * 70)
    print("数据生成完成！")
    print("=" * 70)
    print(f"\n文件列表:")
    print(f"  1. {main_file}")
    print(f"  2. {ts_file}")
    print(f"  3. {output_dir}/data_dictionary.json")
