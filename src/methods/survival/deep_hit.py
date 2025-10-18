"""
Deep Hit model for competing risks survival analysis
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DeepHitNet(nn.Module):
    """Deep Hit Network for competing risks"""

    def __init__(self,
                 input_size: int,
                 num_risks: int,
                 num_time_bins: int,
                 hidden_layers: List[int] = [64, 32],
                 dropout: float = 0.3,
                 activation: str = 'relu',
                 batch_norm: bool = True):
        super().__init__()

        self.input_size = input_size
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins

        # 共享特征提取器
        shared_layers = []
        in_features = input_size

        for i, out_features in enumerate(hidden_layers):
            shared_layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(out_features))

            if activation.lower() == 'relu':
                shared_layers.append(nn.ReLU())
            elif activation.lower() == 'elu':
                shared_layers.append(nn.ELU())
            elif activation.lower() == 'selu':
                shared_layers.append(nn.SELU())

            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))

            in_features = out_features

        self.shared_net = nn.Sequential(*shared_layers)

        # 风险特定的输出层
        self.risk_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, num_time_bins),
                nn.Softmax(dim=1)
            ) for _ in range(num_risks)
        ])

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch_size, input_size)

        Returns:
            outputs: List of (batch_size, num_time_bins) for each risk
        """
        shared_features = self.shared_net(x)

        # 每个风险的时间分布
        risk_outputs = []
        for risk_layer in self.risk_layers:
            risk_output = risk_layer(shared_features)
            risk_outputs.append(risk_output)

        return risk_outputs


class DeepHitLoss(nn.Module):
    """Deep Hit Loss function"""

    def __init__(self, alpha: float = 0.5, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha  # 平衡likelihood和ranking loss的权重
        self.sigma = sigma  # ranking loss中的温度参数

    def forward(self, risk_outputs, time_bins, risk_indicators, event_indicators):
        """
        计算Deep Hit损失

        Args:
            risk_outputs: List of (batch_size, num_time_bins) for each risk
            time_bins: (batch_size,) - 时间区间索引
            risk_indicators: (batch_size, num_risks) - 风险类型指示符
            event_indicators: (batch_size,) - 是否发生事件

        Returns:
            loss: 总损失
        """
        batch_size = time_bins.size(0)
        num_risks = len(risk_outputs)

        # Likelihood Loss
        likelihood_loss = 0.0
        for risk_idx in range(num_risks):
            risk_pred = risk_outputs[risk_idx]  # (batch_size, num_time_bins)

            # 选择对应的时间区间概率
            risk_probs = torch.gather(risk_pred, 1, time_bins.unsqueeze(1)).squeeze()

            # 只对发生该风险的样本计算损失
            risk_mask = risk_indicators[:, risk_idx] * event_indicators
            if risk_mask.sum() > 0:
                likelihood_loss += -torch.log(risk_probs[risk_mask] + 1e-8).mean()

        # Ranking Loss (for censored data)
        ranking_loss = 0.0
        censored_mask = (event_indicators == 0)

        if censored_mask.sum() > 0:
            for risk_idx in range(num_risks):
                risk_pred = risk_outputs[risk_idx]

                # 计算累积风险概率
                cumulative_risk = torch.cumsum(risk_pred, dim=1)

                for i in range(batch_size):
                    if censored_mask[i]:
                        # 对于审查样本，其累积风险应该低于相同时间点发生事件的样本
                        censored_time = time_bins[i]
                        censored_cum_risk = cumulative_risk[i, censored_time]

                        # 找到在相同或更早时间发生事件的样本
                        event_mask = (event_indicators == 1) & (time_bins <= censored_time)
                        if event_mask.sum() > 0:
                            event_cum_risks = cumulative_risk[event_mask, censored_time]
                            ranking_diff = censored_cum_risk - event_cum_risks
                            ranking_loss += torch.sum(torch.sigmoid(ranking_diff / self.sigma))

        # 总损失
        total_loss = self.alpha * likelihood_loss + (1 - self.alpha) * ranking_loss
        return total_loss


class DeepHitInjuryPredictor:
    """Deep Hit 竞争风险伤病预测器"""

    def __init__(self,
                 hidden_layers: List[int] = [64, 32],
                 dropout: float = 0.3,
                 activation: str = 'relu',
                 batch_norm: bool = True,
                 num_time_bins: int = 20,
                 alpha: float = 0.5,
                 sigma: float = 0.1,
                 device: str = 'auto',
                 random_state: int = 42):
        """
        初始化DeepHit预测器

        Args:
            hidden_layers: 隐藏层大小列表
            dropout: Dropout比率
            activation: 激活函数
            batch_norm: 是否使用批归一化
            num_time_bins: 时间区间数量
            alpha: likelihood和ranking loss平衡参数
            sigma: ranking loss温度参数
            device: 计算设备
            random_state: 随机种子
        """
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.num_time_bins = num_time_bins
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state

        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 模型相关
        self.model = None
        self.scaler = StandardScaler()
        self.risk_encoder = LabelEncoder()
        self.feature_names = None
        self.risk_types = None
        self.time_bins = None
        self.is_fitted = False
        self.training_history = {}
        self.input_size = None
        self.num_risks = None

        # 设置随机种子
        self._set_random_seed()

        logger.info(f"Initialized DeepHit predictor on {self.device}")

    def _set_random_seed(self):
        """设置随机种子"""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

    def prepare_competing_risks_data(self,
                                   loads_df: pd.DataFrame,
                                   injuries_df: pd.DataFrame,
                                   follow_up_days: int = 365,
                                   risk_column: str = 'injury_type') -> pd.DataFrame:
        """
        准备竞争风险数据

        Args:
            loads_df: 负荷数据
            injuries_df: 伤病数据
            follow_up_days: 随访天数
            risk_column: 风险类型列名

        Returns:
            竞争风险数据集
        """
        competing_risks_data = []

        for player_id in loads_df['player_id'].unique():
            player_loads = loads_df[loads_df['player_id'] == player_id].sort_values('date')
            player_injuries = injuries_df[injuries_df['player_id'] == player_id].sort_values('onset_date')

            # 为每个负荷记录创建竞争风险数据
            for idx, load_row in player_loads.iterrows():
                observation_date = load_row['date']
                end_date = observation_date + pd.Timedelta(days=follow_up_days)

                # 查找观察期内的第一次伤病
                future_injuries = player_injuries[
                    (player_injuries['onset_date'] > observation_date) &
                    (player_injuries['onset_date'] <= end_date)
                ]

                if not future_injuries.empty:
                    # 发生伤病 - 选择最早的伤病
                    first_injury = future_injuries.iloc[0]
                    time_to_event = (first_injury['onset_date'] - observation_date).days
                    event_observed = 1
                    risk_type = first_injury.get(risk_column, 'UNKNOWN')
                else:
                    # 审查（未观察到伤病）
                    time_to_event = follow_up_days
                    event_observed = 0
                    risk_type = 'CENSORED'

                # 收集特征
                risk_record = {
                    'player_id': player_id,
                    'observation_date': observation_date,
                    'time_to_event': time_to_event,
                    'event_observed': event_observed,
                    'risk_type': risk_type
                }

                # 添加负荷特征
                for col in load_row.index:
                    if col not in ['player_id', 'date']:
                        risk_record[col] = load_row[col]

                competing_risks_data.append(risk_record)

        competing_risks_df = pd.DataFrame(competing_risks_data)

        # 过滤掉时间为0的记录
        competing_risks_df = competing_risks_df[competing_risks_df['time_to_event'] > 0]

        logger.info(f"Created competing risks dataset with {len(competing_risks_df)} observations")

        # 统计各种风险类型
        risk_counts = competing_risks_df[competing_risks_df['event_observed'] == 1]['risk_type'].value_counts()
        logger.info(f"Risk type distribution: {risk_counts.to_dict()}")

        return competing_risks_df

    def _create_time_bins(self, time_to_event: np.ndarray) -> np.ndarray:
        """创建时间区间"""
        max_time = np.max(time_to_event)
        self.time_bins = np.linspace(0, max_time, self.num_time_bins + 1)
        return self.time_bins

    def _discretize_time(self, time_to_event: np.ndarray) -> np.ndarray:
        """将连续时间离散化到时间区间"""
        return np.digitize(time_to_event, self.time_bins) - 1

    def fit(self,
            competing_risks_df: pd.DataFrame,
            duration_col: str = 'time_to_event',
            event_col: str = 'event_observed',
            risk_col: str = 'risk_type',
            feature_columns: List[str] = None,
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 64,
            learning_rate: float = 0.001,
            early_stopping_patience: int = 15,
            verbose: bool = True) -> Dict[str, Any]:
        """
        训练DeepHit模型

        Args:
            competing_risks_df: 竞争风险数据
            duration_col: 持续时间列名
            event_col: 事件观察列名
            risk_col: 风险类型列名
            feature_columns: 特征列名列表
            validation_split: 验证集比例
            epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率
            early_stopping_patience: 早停耐心值
            verbose: 是否显示训练过程

        Returns:
            训练结果字典
        """
        logger.info("Starting DeepHit training")

        # 特征选择
        if feature_columns is None:
            exclude_cols = [duration_col, event_col, risk_col, 'player_id', 'observation_date']
            feature_columns = [col for col in competing_risks_df.columns
                             if col not in exclude_cols and
                             competing_risks_df[col].dtype in ['int64', 'float64']]

        self.feature_names = feature_columns
        self.input_size = len(feature_columns)
        logger.info(f"Using {len(feature_columns)} features for DeepHit model")

        # 准备数据
        X = competing_risks_df[feature_columns].fillna(0)
        durations = competing_risks_df[duration_col].values
        events = competing_risks_df[event_col].values
        risks = competing_risks_df[risk_col].values

        # 创建时间区间
        self.time_bins = self._create_time_bins(durations)
        time_bin_indices = self._discretize_time(durations)

        # 编码风险类型
        event_risks = risks[events == 1]  # 只对实际发生的事件编码
        self.risk_encoder.fit(event_risks)
        self.risk_types = list(self.risk_encoder.classes_)
        self.num_risks = len(self.risk_types)

        logger.info(f"Found {self.num_risks} risk types: {self.risk_types}")

        # 创建风险指示符矩阵
        risk_indicators = np.zeros((len(risks), self.num_risks))
        for i, risk in enumerate(risks):
            if events[i] == 1 and risk in self.risk_types:
                risk_idx = self.risk_encoder.transform([risk])[0]
                risk_indicators[i, risk_idx] = 1

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        # 数据分割
        if 'player_id' in competing_risks_df.columns:
            # 按球员分割
            unique_players = competing_risks_df['player_id'].unique()
            np.random.seed(self.random_state)
            np.random.shuffle(unique_players)

            n_val_players = int(len(unique_players) * validation_split)
            val_players = unique_players[:n_val_players]
            train_players = unique_players[n_val_players:]

            train_mask = competing_risks_df['player_id'].isin(train_players)
            val_mask = competing_risks_df['player_id'].isin(val_players)

            X_train, X_val = X_scaled[train_mask], X_scaled[val_mask]
            time_bins_train, time_bins_val = time_bin_indices[train_mask], time_bin_indices[val_mask]
            risk_indicators_train, risk_indicators_val = risk_indicators[train_mask], risk_indicators[val_mask]
            events_train, events_val = events[train_mask], events[val_mask]
        else:
            # 简单随机分割
            n_samples = len(X)
            indices = np.random.permutation(n_samples)
            split_idx = int(n_samples * (1 - validation_split))

            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            X_train, X_val = X_scaled[train_indices], X_scaled[val_indices]
            time_bins_train, time_bins_val = time_bin_indices[train_indices], time_bin_indices[val_indices]
            risk_indicators_train, risk_indicators_val = risk_indicators[train_indices], risk_indicators[val_indices]
            events_train, events_val = events[train_indices], events[val_indices]

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.LongTensor(time_bins_train).to(self.device),
            torch.FloatTensor(risk_indicators_train).to(self.device),
            torch.FloatTensor(events_train).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.LongTensor(time_bins_val).to(self.device),
            torch.FloatTensor(risk_indicators_val).to(self.device),
            torch.FloatTensor(events_val).to(self.device)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        self.model = DeepHitNet(
            input_size=self.input_size,
            num_risks=self.num_risks,
            num_time_bins=self.num_time_bins,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm
        ).to(self.device)

        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = DeepHitLoss(alpha=self.alpha, sigma=self.sigma)

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for batch_X, batch_time_bins, batch_risk_indicators, batch_events in train_loader:
                optimizer.zero_grad()

                risk_outputs = self.model(batch_X)
                loss = criterion(risk_outputs, batch_time_bins, batch_risk_indicators, batch_events)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # 验证阶段
            val_loss = self._evaluate(val_loader, criterion)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    # 恢复最佳模型
                    self.model.load_state_dict(self.best_model_state)
                    break

        self.is_fitted = True
        self.training_history = history

        # 计算最终指标
        final_results = {
            'model_type': 'deephit',
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'final_val_loss': val_loss,
            'training_history': history,
            'n_features': self.input_size,
            'n_samples': len(competing_risks_df),
            'n_events': events.sum(),
            'event_rate': events.mean(),
            'num_risks': self.num_risks,
            'risk_types': self.risk_types,
            'num_time_bins': self.num_time_bins,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

        logger.info("DeepHit training completed")
        return final_results

    def _evaluate(self, data_loader, criterion):
        """评估模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_X, batch_time_bins, batch_risk_indicators, batch_events in data_loader:
                risk_outputs = self.model(batch_X)
                loss = criterion(risk_outputs, batch_time_bins, batch_risk_indicators, batch_events)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def predict_risk_probabilities(self, X: pd.DataFrame, time_horizons: List[int] = [30, 90, 180, 365]) -> Dict[str, np.ndarray]:
        """预测各种风险在不同时间点的概率"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_clean)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            risk_outputs = self.model(X_tensor)

        # 将输出转换为numpy
        risk_probs = [output.cpu().numpy() for output in risk_outputs]

        # 计算累积概率
        results = {}
        for horizon in time_horizons:
            # 找到对应的时间区间
            time_bin = np.digitize([horizon], self.time_bins)[0] - 1
            time_bin = min(time_bin, self.num_time_bins - 1)

            horizon_probs = {}
            for risk_idx, risk_type in enumerate(self.risk_types):
                # 累积到指定时间点的概率
                cumulative_prob = np.sum(risk_probs[risk_idx][:, :time_bin+1], axis=1)
                horizon_probs[risk_type] = cumulative_prob

            results[f'{horizon}d'] = horizon_probs

        return results

    def predict_dominant_risk(self, X: pd.DataFrame, time_horizon: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """预测主导风险类型和概率"""
        risk_predictions = self.predict_risk_probabilities(X, [time_horizon])
        horizon_key = f'{time_horizon}d'

        if horizon_key not in risk_predictions:
            return np.array([]), np.array([])

        # 找到每个样本的最大概率风险
        all_probs = []
        for risk_type in self.risk_types:
            all_probs.append(risk_predictions[horizon_key][risk_type])

        all_probs = np.array(all_probs).T  # (n_samples, n_risks)

        dominant_risk_indices = np.argmax(all_probs, axis=1)
        dominant_risk_probs = np.max(all_probs, axis=1)
        dominant_risk_names = [self.risk_types[idx] for idx in dominant_risk_indices]

        return np.array(dominant_risk_names), dominant_risk_probs

    def explain_prediction(self, X: pd.DataFrame, index: int, time_horizon: int = 30) -> Dict[str, Any]:
        """解释单个预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")

        sample = X.iloc[index:index+1]

        # 预测风险概率
        risk_predictions = self.predict_risk_probabilities(sample, [time_horizon])
        horizon_key = f'{time_horizon}d'

        # 主导风险
        dominant_risk, dominant_prob = self.predict_dominant_risk(sample, time_horizon)

        return {
            'dominant_risk': dominant_risk[0] if len(dominant_risk) > 0 else 'UNKNOWN',
            'dominant_risk_probability': dominant_prob[0] if len(dominant_prob) > 0 else 0.0,
            'all_risk_probabilities': risk_predictions[horizon_key] if horizon_key in risk_predictions else {},
            'risk_level': 'High' if dominant_prob[0] > 0.2 else 'Medium' if dominant_prob[0] > 0.1 else 'Low',
            'time_horizon_days': time_horizon
        }

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'risk_encoder': self.risk_encoder,
            'feature_names': self.feature_names,
            'risk_types': self.risk_types,
            'time_bins': self.time_bins,
            'model_config': {
                'input_size': self.input_size,
                'num_risks': self.num_risks,
                'num_time_bins': self.num_time_bins,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout,
                'activation': self.activation,
                'batch_norm': self.batch_norm,
                'alpha': self.alpha,
                'sigma': self.sigma
            },
            'training_history': self.training_history,
            'random_state': self.random_state
        }

        torch.save(model_data, filepath)
        logger.info(f"DeepHit model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = torch.load(filepath, map_location=self.device)

        config = model_data['model_config']
        self.input_size = config['input_size']
        self.num_risks = config['num_risks']
        self.num_time_bins = config['num_time_bins']
        self.hidden_layers = config['hidden_layers']
        self.dropout = config['dropout']
        self.activation = config['activation']
        self.batch_norm = config['batch_norm']
        self.alpha = config['alpha']
        self.sigma = config['sigma']

        self.model = DeepHitNet(
            input_size=self.input_size,
            num_risks=self.num_risks,
            num_time_bins=self.num_time_bins,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm
        ).to(self.device)

        self.model.load_state_dict(model_data['model_state_dict'])
        self.scaler = model_data['scaler']
        self.risk_encoder = model_data['risk_encoder']
        self.feature_names = model_data['feature_names']
        self.risk_types = model_data['risk_types']
        self.time_bins = model_data['time_bins']
        self.training_history = model_data.get('training_history', {})
        self.random_state = model_data.get('random_state', 42)
        self.is_fitted = True

        logger.info(f"DeepHit model loaded from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        summary = {
            "model_type": "DeepHit (Competing Risks)",
            "device": str(self.device),
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "is_fitted": self.is_fitted
        }

        if self.input_size is not None:
            summary.update({
                "input_size": self.input_size,
                "n_features": len(self.feature_names) if self.feature_names else 0,
                "num_risks": self.num_risks,
                "num_time_bins": self.num_time_bins
            })

        if self.risk_types is not None:
            summary["risk_types"] = self.risk_types

        if self.is_fitted:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            summary.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })

        if self.training_history:
            history = self.training_history
            summary["training_epochs"] = len(history.get('train_loss', []))

        return summary