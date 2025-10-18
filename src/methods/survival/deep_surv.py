"""
Deep Survival Analysis model (DeepSurv) for injury risk prediction
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

try:
    from sksurv.metrics import concordance_index_censored
    SURVIVAL_METRICS_AVAILABLE = True
except ImportError:
    SURVIVAL_METRICS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sksurv not available for survival metrics")

logger = logging.getLogger(__name__)


class DeepSurvNet(nn.Module):
    """Deep Survival Network"""

    def __init__(self,
                 input_size: int,
                 hidden_layers: List[int] = [64, 32, 16],
                 dropout: float = 0.3,
                 activation: str = 'relu',
                 batch_norm: bool = True):
        super().__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        layers = []
        in_features = input_size

        # 构建隐藏层
        for i, out_features in enumerate(hidden_layers):
            # 线性层
            layers.append(nn.Linear(in_features, out_features))

            # 批归一化
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))

            # 激活函数
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'elu':
                layers.append(nn.ELU())
            elif activation.lower() == 'selu':
                layers.append(nn.SELU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = out_features

        # 输出层（单个神经元，无激活函数）
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

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
            risk_scores: (batch_size, 1) - 风险分数（线性输出）
        """
        return self.network(x)


class PartialLogLikelihoodLoss(nn.Module):
    """部分对数似然损失（Cox回归损失）"""

    def __init__(self, regularization: float = 0.01):
        super().__init__()
        self.regularization = regularization

    def forward(self, risk_scores, durations, events, model=None):
        """
        计算部分对数似然损失

        Args:
            risk_scores: (batch_size, 1) - 模型输出的风险分数
            durations: (batch_size,) - 生存时间
            events: (batch_size,) - 事件指示符
            model: 模型实例（用于L2正则化）

        Returns:
            loss: 标量损失值
        """
        batch_size = risk_scores.size(0)
        risk_scores = risk_scores.squeeze()

        # 排序（按生存时间降序）
        sorted_indices = torch.argsort(durations, descending=True)
        sorted_risk_scores = risk_scores[sorted_indices]
        sorted_events = events[sorted_indices]

        # 计算风险集合的累积和
        risk_exp = torch.exp(sorted_risk_scores)
        cumulative_sum = torch.cumsum(risk_exp, dim=0)

        # 部分对数似然
        log_likelihood = torch.sum(
            sorted_events * (sorted_risk_scores - torch.log(cumulative_sum))
        )

        # 平均负对数似然
        n_events = torch.sum(events)
        if n_events > 0:
            loss = -log_likelihood / n_events
        else:
            loss = torch.tensor(0.0, device=risk_scores.device)

        # L2正则化
        if model is not None and self.regularization > 0:
            l2_reg = torch.tensor(0., device=risk_scores.device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += self.regularization * l2_reg

        return loss


class DeepSurvInjuryPredictor:
    """Deep Survival Analysis 伤病风险预测器"""

    def __init__(self,
                 hidden_layers: List[int] = [64, 32, 16],
                 dropout: float = 0.3,
                 activation: str = 'relu',
                 batch_norm: bool = True,
                 regularization: float = 0.01,
                 device: str = 'auto',
                 random_state: int = 42):
        """
        初始化DeepSurv预测器

        Args:
            hidden_layers: 隐藏层大小列表
            dropout: Dropout比率
            activation: 激活函数
            batch_norm: 是否使用批归一化
            regularization: L2正则化强度
            device: 计算设备
            random_state: 随机种子
        """
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.regularization = regularization
        self.random_state = random_state

        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 模型相关
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.training_history = {}
        self.input_size = None

        # 设置随机种子
        self._set_random_seed()

        logger.info(f"Initialized DeepSurv predictor on {self.device}")

    def _set_random_seed(self):
        """设置随机种子"""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

    def prepare_survival_data(self,
                            loads_df: pd.DataFrame,
                            injuries_df: pd.DataFrame,
                            follow_up_days: int = 365) -> pd.DataFrame:
        """准备生存分析数据（与Cox模型相同的逻辑）"""
        from .cox_model import CoxInjuryPredictor
        cox_predictor = CoxInjuryPredictor()
        return cox_predictor.prepare_survival_data(loads_df, injuries_df, follow_up_days)

    def fit(self,
            survival_df: pd.DataFrame,
            duration_col: str = 'time_to_event',
            event_col: str = 'event_observed',
            feature_columns: List[str] = None,
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 64,
            learning_rate: float = 0.001,
            early_stopping_patience: int = 15,
            verbose: bool = True) -> Dict[str, Any]:
        """
        训练DeepSurv模型

        Args:
            survival_df: 生存分析数据
            duration_col: 持续时间列名
            event_col: 事件观察列名
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
        logger.info("Starting DeepSurv training")

        # 特征选择
        if feature_columns is None:
            exclude_cols = [duration_col, event_col, 'player_id', 'observation_date', 'severity']
            feature_columns = [col for col in survival_df.columns
                             if col not in exclude_cols and survival_df[col].dtype in ['int64', 'float64']]

        self.feature_names = feature_columns
        self.input_size = len(feature_columns)
        logger.info(f"Using {len(feature_columns)} features for DeepSurv model")

        # 准备数据
        X = survival_df[feature_columns].fillna(0)
        durations = survival_df[duration_col].values
        events = survival_df[event_col].astype(bool).values

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        # 数据分割
        n_samples = len(X)
        if 'player_id' in survival_df.columns:
            # 按球员分割避免数据泄漏
            unique_players = survival_df['player_id'].unique()
            np.random.seed(self.random_state)
            np.random.shuffle(unique_players)

            n_val_players = int(len(unique_players) * validation_split)
            val_players = unique_players[:n_val_players]
            train_players = unique_players[n_val_players:]

            train_mask = survival_df['player_id'].isin(train_players)
            val_mask = survival_df['player_id'].isin(val_players)

            X_train, X_val = X_scaled[train_mask], X_scaled[val_mask]
            durations_train, durations_val = durations[train_mask], durations[val_mask]
            events_train, events_val = events[train_mask], events[val_mask]
        else:
            # 简单随机分割
            indices = np.random.permutation(n_samples)
            split_idx = int(n_samples * (1 - validation_split))

            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            X_train, X_val = X_scaled[train_indices], X_scaled[val_indices]
            durations_train, durations_val = durations[train_indices], durations[val_indices]
            events_train, events_val = events[train_indices], events[val_indices]

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(durations_train).to(self.device),
            torch.FloatTensor(events_train.astype(float)).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.FloatTensor(durations_val).to(self.device),
            torch.FloatTensor(events_val.astype(float)).to(self.device)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        self.model = DeepSurvNet(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm
        ).to(self.device)

        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=verbose
        )
        criterion = PartialLogLikelihoodLoss(regularization=self.regularization)

        # 训练循环
        best_val_concordance = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_concordance': []}

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for batch_X, batch_durations, batch_events in train_loader:
                optimizer.zero_grad()

                risk_scores = self.model(batch_X)
                loss = criterion(risk_scores, batch_durations, batch_events, self.model)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # 验证阶段
            val_loss, val_concordance = self._evaluate(val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_concordance'].append(val_concordance)

            scheduler.step(val_concordance)

            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val C-index: {val_concordance:.4f}")

            # 早停检查
            if val_concordance > best_val_concordance:
                best_val_concordance = val_concordance
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
            'model_type': 'deepsurv',
            'epochs_trained': epoch + 1,
            'best_val_concordance': best_val_concordance,
            'final_val_concordance': val_concordance,
            'training_history': history,
            'n_features': self.input_size,
            'n_samples': len(survival_df),
            'n_events': events.sum(),
            'event_rate': events.mean(),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

        logger.info("DeepSurv training completed")
        return final_results

    def _evaluate(self, data_loader, criterion):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_risk_scores = []
        all_durations = []
        all_events = []

        with torch.no_grad():
            for batch_X, batch_durations, batch_events in data_loader:
                risk_scores = self.model(batch_X)
                loss = criterion(risk_scores, batch_durations, batch_events, self.model)
                total_loss += loss.item()

                all_risk_scores.extend(risk_scores.cpu().numpy().flatten())
                all_durations.extend(batch_durations.cpu().numpy())
                all_events.extend(batch_events.cpu().numpy().astype(bool))

        avg_loss = total_loss / len(data_loader)

        # 计算concordance index
        if SURVIVAL_METRICS_AVAILABLE:
            concordance = concordance_index_censored(all_events, all_durations, all_risk_scores)[0]
        else:
            # 简化的concordance计算
            concordance = self._simple_concordance(all_risk_scores, all_durations, all_events)

        return avg_loss, concordance

    def _simple_concordance(self, risk_scores, durations, events):
        """简化的concordance index计算"""
        risk_scores = np.array(risk_scores)
        durations = np.array(durations)
        events = np.array(events)

        concordant_pairs = 0
        total_pairs = 0

        n = len(risk_scores)
        for i in range(n):
            for j in range(i + 1, n):
                if events[i] and durations[i] < durations[j]:
                    total_pairs += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant_pairs += 1
                elif events[j] and durations[j] < durations[i]:
                    total_pairs += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant_pairs += 1

        return concordant_pairs / total_pairs if total_pairs > 0 else 0.5

    def predict_risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        """预测风险分数"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_clean = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_clean)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            risk_scores = self.model(X_tensor)

        return risk_scores.cpu().numpy().flatten()

    def predict_survival_function(self, X: pd.DataFrame, times: np.ndarray = None) -> Dict[str, np.ndarray]:
        """预测生存函数（基于风险分数的近似）"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        risk_scores = self.predict_risk_scores(X)

        if times is None:
            times = np.linspace(1, 365, 100)

        # 使用指数生存模型的近似
        # S(t) = exp(-lambda * t), 其中 lambda 与风险分数相关
        survival_functions = []

        for risk_score in risk_scores:
            # 将风险分数转换为风险率参数
            lambda_param = np.exp(risk_score) / 100  # 缩放因子
            survival_prob = np.exp(-lambda_param * times)
            survival_functions.append(survival_prob)

        return {
            'times': times,
            'survival_probs': np.array(survival_functions)
        }

    def predict_injury_probability(self, X: pd.DataFrame, time_horizon: int = 30) -> np.ndarray:
        """预测指定时间内的伤病概率"""
        survival_data = self.predict_survival_function(X, times=np.array([time_horizon]))
        injury_probs = 1 - survival_data['survival_probs'][:, 0]
        return injury_probs

    def explain_prediction(self, X: pd.DataFrame, index: int) -> Dict[str, Any]:
        """解释单个预测（基于梯度）"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")

        sample = X.iloc[index:index+1]
        risk_score = self.predict_risk_scores(sample)[0]
        injury_prob_30d = self.predict_injury_probability(sample, time_horizon=30)[0]

        # 计算特征梯度
        X_clean = sample[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_clean)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        X_tensor.requires_grad_(True)

        self.model.eval()
        output = self.model(X_tensor)
        output.backward()

        gradients = X_tensor.grad.cpu().numpy().flatten()

        # 特征重要性（梯度 * 输入值）
        feature_values = X_scaled.flatten()
        feature_importance = np.abs(gradients * feature_values)

        # 排序特征
        importance_pairs = list(zip(self.feature_names, feature_importance, feature_values, gradients))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)

        return {
            'risk_score': risk_score,
            'injury_probability_30d': injury_prob_30d,
            'risk_level': 'High' if injury_prob_30d > 0.2 else 'Medium' if injury_prob_30d > 0.1 else 'Low',
            'feature_importance': [(name, imp) for name, imp, _, _ in importance_pairs[:10]],
            'feature_gradients': [(name, grad) for name, _, _, grad in importance_pairs[:10]]
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
            'feature_names': self.feature_names,
            'model_config': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout,
                'activation': self.activation,
                'batch_norm': self.batch_norm,
                'regularization': self.regularization
            },
            'training_history': self.training_history,
            'random_state': self.random_state
        }

        torch.save(model_data, filepath)
        logger.info(f"DeepSurv model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = torch.load(filepath, map_location=self.device)

        config = model_data['model_config']
        self.input_size = config['input_size']
        self.hidden_layers = config['hidden_layers']
        self.dropout = config['dropout']
        self.activation = config['activation']
        self.batch_norm = config['batch_norm']
        self.regularization = config['regularization']

        self.model = DeepSurvNet(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm
        ).to(self.device)

        self.model.load_state_dict(model_data['model_state_dict'])
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data.get('training_history', {})
        self.random_state = model_data.get('random_state', 42)
        self.is_fitted = True

        logger.info(f"DeepSurv model loaded from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        summary = {
            "model_type": "DeepSurv",
            "device": str(self.device),
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "regularization": self.regularization,
            "is_fitted": self.is_fitted
        }

        if self.input_size is not None:
            summary["input_size"] = self.input_size
            summary["n_features"] = len(self.feature_names) if self.feature_names else 0

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
            if 'val_concordance' in history and history['val_concordance']:
                summary["best_val_concordance"] = max(history['val_concordance'])

        return summary