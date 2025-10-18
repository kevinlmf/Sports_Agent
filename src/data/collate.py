"""
序列打包模块
为LSTM/Transformer等序列模型准备数据，处理padding和masking
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SequenceCollator:
    """序列数据整理器"""

    def __init__(self,
                 sequence_length: int = 30,
                 prediction_horizon: int = 7,
                 min_sequence_length: int = 10,
                 padding_value: float = 0.0):
        """
        初始化序列整理器

        Args:
            sequence_length: 输入序列长度（天）
            prediction_horizon: 预测窗口长度（天）
            min_sequence_length: 最小有效序列长度
            padding_value: 填充值
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.min_sequence_length = min_sequence_length
        self.padding_value = padding_value

    def create_sequences(self,
                        data: pd.DataFrame,
                        target_column: str = None,
                        feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        创建序列数据

        Args:
            data: 输入数据，包含player_id, date和特征列
            target_column: 目标列名（用于监督学习）
            feature_columns: 特征列名列表

        Returns:
            包含序列数据的字典
        """
        if feature_columns is None:
            # 自动检测数值特征列
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns
                             if col not in ['player_id'] and col != target_column]

        sequences = []
        targets = []
        metadata = []

        # 按球员分组处理
        for player_id in data['player_id'].unique():
            player_data = data[data['player_id'] == player_id].sort_values('date')

            if len(player_data) < self.min_sequence_length:
                logger.warning(f"Player {player_id} has insufficient data: {len(player_data)} records")
                continue

            # 创建该球员的序列
            player_sequences = self._create_player_sequences(
                player_data, feature_columns, target_column
            )

            sequences.extend(player_sequences['sequences'])
            targets.extend(player_sequences['targets'])
            metadata.extend(player_sequences['metadata'])

        logger.info(f"Created {len(sequences)} sequences from {data['player_id'].nunique()} players")

        result = {
            'sequences': sequences,
            'metadata': metadata,
            'feature_names': feature_columns,
            'n_features': len(feature_columns)
        }

        if targets:
            result['targets'] = targets

        return result

    def _create_player_sequences(self,
                               player_data: pd.DataFrame,
                               feature_columns: List[str],
                               target_column: str = None) -> Dict[str, List]:
        """为单个球员创建序列"""
        sequences = []
        targets = []
        metadata = []

        player_id = player_data['player_id'].iloc[0]

        # 滑动窗口创建序列
        for i in range(len(player_data) - self.sequence_length + 1):
            # 输入序列
            sequence_data = player_data.iloc[i:i+self.sequence_length]
            sequence_features = sequence_data[feature_columns].values.astype(np.float32)

            # 检查序列有效性
            if self._is_valid_sequence(sequence_data):
                sequences.append(sequence_features)

                # 元数据
                metadata.append({
                    'player_id': player_id,
                    'sequence_start': sequence_data['date'].iloc[0],
                    'sequence_end': sequence_data['date'].iloc[-1],
                    'sequence_index': len(sequences) - 1
                })

                # 目标值（如果提供）
                if target_column and target_column in player_data.columns:
                    # 预测未来一段时间的目标
                    target_start_idx = i + self.sequence_length
                    target_end_idx = min(target_start_idx + self.prediction_horizon,
                                       len(player_data))

                    if target_start_idx < len(player_data):
                        target_data = player_data.iloc[target_start_idx:target_end_idx]
                        # 可以是二分类（是否受伤）或回归（伤病风险分数）
                        target_value = target_data[target_column].max()  # 取窗口内最大值
                        targets.append(target_value)
                    else:
                        # 如果没有未来数据，跳过这个序列
                        sequences.pop()
                        metadata.pop()

        return {
            'sequences': sequences,
            'targets': targets,
            'metadata': metadata
        }

    def _is_valid_sequence(self, sequence_data: pd.DataFrame) -> bool:
        """检查序列是否有效"""
        # 检查时间连续性（允许一定的间隔）
        dates = pd.to_datetime(sequence_data['date'])
        max_gap = (dates.max() - dates.min()).days

        # 如果时间跨度过大（超过序列长度的2倍），认为不连续
        if max_gap > self.sequence_length * 2:
            return False

        # 检查数据质量（非空值比例）
        non_null_ratio = sequence_data.notna().sum().sum() / (len(sequence_data) * len(sequence_data.columns))
        if non_null_ratio < 0.7:  # 至少70%的数据非空
            return False

        return True

    def pad_sequences(self, sequences: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        对序列进行padding

        Args:
            sequences: 序列列表

        Returns:
            (padded_sequences, mask): padding后的序列和mask
        """
        if not sequences:
            return np.array([]), np.array([])

        # 转换为tensor进行padding
        tensor_sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]

        # Padding
        padded = pad_sequence(tensor_sequences, batch_first=True,
                            padding_value=self.padding_value)

        # 创建mask（True表示真实数据，False表示padding）
        max_len = padded.shape[1]
        mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            mask[i, :len(seq)] = True

        return padded.numpy(), mask.numpy()

    def create_injury_prediction_dataset(self,
                                       loads_data: pd.DataFrame,
                                       injuries_data: pd.DataFrame,
                                       feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        创建伤病预测数据集

        Args:
            loads_data: 负荷数据
            injuries_data: 伤病数据
            feature_columns: 特征列

        Returns:
            伤病预测数据集
        """
        # 为每个负荷记录创建伤病标签
        labeled_data = self._create_injury_labels(loads_data, injuries_data)

        # 创建序列
        sequences_data = self.create_sequences(
            labeled_data,
            target_column='injury_target',
            feature_columns=feature_columns
        )

        return sequences_data

    def _create_injury_labels(self,
                            loads_data: pd.DataFrame,
                            injuries_data: pd.DataFrame) -> pd.DataFrame:
        """为负荷数据创建伤病标签"""
        data = loads_data.copy()
        data['injury_target'] = 0

        if injuries_data.empty:
            logger.warning("No injury data available for labeling")
            return data

        # 为每个球员处理
        for player_id in data['player_id'].unique():
            player_injuries = injuries_data[injuries_data['player_id'] == player_id]
            if player_injuries.empty:
                continue

            player_loads = data[data['player_id'] == player_id].copy()

            # 为每个负荷记录检查未来是否有伤病
            for idx, load_row in player_loads.iterrows():
                load_date = pd.to_datetime(load_row['date'])

                # 检查未来prediction_horizon天内是否有伤病
                future_date = load_date + timedelta(days=self.prediction_horizon)

                # 查找在此期间开始的伤病
                future_injuries = player_injuries[
                    (player_injuries['onset_date'] >= load_date) &
                    (player_injuries['onset_date'] <= future_date)
                ]

                if not future_injuries.empty:
                    data.loc[idx, 'injury_target'] = 1

        injury_rate = data['injury_target'].mean()
        logger.info(f"Created injury labels with {injury_rate:.3f} positive rate")

        return data


class TransformerCollator(SequenceCollator):
    """Transformer模型专用的数据整理器"""

    def __init__(self,
                 sequence_length: int = 30,
                 prediction_horizon: int = 7,
                 min_sequence_length: int = 10,
                 padding_value: float = 0.0,
                 add_positional_encoding: bool = True):
        super().__init__(sequence_length, prediction_horizon, min_sequence_length, padding_value)
        self.add_positional_encoding = add_positional_encoding

    def create_sequences(self,
                        data: pd.DataFrame,
                        target_column: str = None,
                        feature_columns: List[str] = None) -> Dict[str, Any]:
        """创建Transformer序列数据"""
        base_result = super().create_sequences(data, target_column, feature_columns)

        if self.add_positional_encoding and base_result['sequences']:
            # 添加位置编码
            base_result['sequences'] = self._add_positional_encoding(base_result['sequences'])

        return base_result

    def _add_positional_encoding(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """添加位置编码"""
        enhanced_sequences = []

        for seq in sequences:
            seq_len, n_features = seq.shape

            # 简单的正弦位置编码
            pos_encoding = np.zeros((seq_len, n_features))
            position = np.arange(seq_len).reshape(-1, 1)

            for i in range(n_features):
                if i % 2 == 0:
                    pos_encoding[:, i] = np.sin(position / np.power(10000, i / n_features)).flatten()
                else:
                    pos_encoding[:, i] = np.cos(position / np.power(10000, (i-1) / n_features)).flatten()

            # 将位置编码与原始特征拼接
            enhanced_seq = seq + 0.1 * pos_encoding  # 小的权重避免覆盖原始信息
            enhanced_sequences.append(enhanced_seq)

        return enhanced_sequences


class DataLoader:
    """PyTorch数据加载器适配器"""

    def __init__(self, sequences_data: Dict[str, Any], batch_size: int = 32, shuffle: bool = True):
        self.sequences_data = sequences_data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.sequences_data['sequences'])

    def __getitem__(self, idx):
        sequence = self.sequences_data['sequences'][idx]
        result = {'sequence': sequence, 'metadata': self.sequences_data['metadata'][idx]}

        if 'targets' in self.sequences_data:
            result['target'] = self.sequences_data['targets'][idx]

        return result

    def get_batches(self):
        """生成批次数据"""
        indices = list(range(len(self)))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_sequences = [self.sequences_data['sequences'][idx] for idx in batch_indices]
            batch_metadata = [self.sequences_data['metadata'][idx] for idx in batch_indices]

            # Padding批次内的序列
            collator = SequenceCollator()
            padded_sequences, mask = collator.pad_sequences(batch_sequences)

            batch_data = {
                'sequences': padded_sequences,
                'mask': mask,
                'metadata': batch_metadata
            }

            if 'targets' in self.sequences_data:
                batch_targets = [self.sequences_data['targets'][idx] for idx in batch_indices]
                batch_data['targets'] = np.array(batch_targets)

            yield batch_data


def create_cross_validation_splits(sequences_data: Dict[str, Any],
                                 n_splits: int = 5,
                                 split_by_player: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    创建交叉验证划分

    Args:
        sequences_data: 序列数据
        n_splits: 划分数量
        split_by_player: 是否按球员划分（避免数据泄漏）

    Returns:
        [(train_indices, val_indices), ...] 列表
    """
    if not split_by_player:
        # 简单随机划分
        n_samples = len(sequences_data['sequences'])
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        splits = []
        fold_size = n_samples // n_splits

        for i in range(n_splits):
            start_val = i * fold_size
            end_val = (i + 1) * fold_size if i < n_splits - 1 else n_samples

            val_indices = indices[start_val:end_val].tolist()
            train_indices = np.concatenate([indices[:start_val], indices[end_val:]]).tolist()
            splits.append((train_indices, val_indices))

        return splits

    # 按球员划分
    player_ids = [meta['player_id'] for meta in sequences_data['metadata']]
    unique_players = list(set(player_ids))
    np.random.shuffle(unique_players)

    splits = []
    players_per_fold = len(unique_players) // n_splits

    for i in range(n_splits):
        start_idx = i * players_per_fold
        end_idx = (i + 1) * players_per_fold if i < n_splits - 1 else len(unique_players)

        val_players = set(unique_players[start_idx:end_idx])
        train_players = set(unique_players) - val_players

        train_indices = [idx for idx, player_id in enumerate(player_ids) if player_id in train_players]
        val_indices = [idx for idx, player_id in enumerate(player_ids) if player_id in val_players]

        splits.append((train_indices, val_indices))

    return splits