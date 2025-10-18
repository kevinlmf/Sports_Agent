"""
GRU model for injury risk prediction using both PyTorch and Flax implementations
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn_flax
    from flax.training import train_state
    import optax
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Flax/JAX not available. Using PyTorch only.")

logger = logging.getLogger(__name__)


class PyTorchGRUModel(nn.Module):
    """PyTorch GRU模型"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 计算GRU输出大小
        gru_output_size = hidden_size * (2 if bidirectional else 1)

        # 注意力机制（可选）
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None, use_attention=False):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            mask: (batch_size, seq_len) - True for valid positions
            use_attention: 是否使用注意力机制
        """
        batch_size = x.size(0)

        # GRU forward pass
        gru_out, h_n = self.gru(x)

        if use_attention:
            # 使用注意力机制
            if mask is not None:
                # 将mask转换为注意力mask (batch_size, seq_len)
                attn_mask = ~mask  # MultiheadAttention uses True for masked positions
            else:
                attn_mask = None

            attn_out, _ = self.attention(gru_out, gru_out, gru_out, key_padding_mask=attn_mask)

            # 全局平均池化或最大池化
            if mask is not None:
                # 使用mask进行池化
                mask_expanded = mask.unsqueeze(-1).expand_as(attn_out)
                attn_out = attn_out * mask_expanded
                final_hidden = attn_out.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                final_hidden = attn_out.mean(dim=1)
        else:
            # 使用最后一个时间步的隐状态
            if self.bidirectional:
                # 取前向和后向的最后隐状态
                forward_h = h_n[-2]
                backward_h = h_n[-1]
                final_hidden = torch.cat([forward_h, backward_h], dim=1)
            else:
                final_hidden = h_n[-1]

        # 分类
        output = self.classifier(final_hidden)
        return output.squeeze(-1)


if FLAX_AVAILABLE:
    class FlaxGRUModel(nn_flax.Module):
        """Flax GRU模型"""

        hidden_size: int = 128
        num_layers: int = 2
        dropout_rate: float = 0.3

        @nn_flax.compact
        def __call__(self, x, training=False):
            # Flax中的GRU实现
            batch_size, seq_len, input_size = x.shape

            # 多层GRU
            for layer_idx in range(self.num_layers):
                # GRU Cell
                carry = jnp.zeros((batch_size, self.hidden_size))
                outputs = []

                for t in range(seq_len):
                    carry, output = nn_flax.GRUCell(features=self.hidden_size)(carry, x[:, t, :])
                    outputs.append(output)

                x = jnp.stack(outputs, axis=1)

                if training and layer_idx < self.num_layers - 1:
                    x = nn_flax.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

            # 取最后时间步
            final_hidden = x[:, -1, :]

            # 分类头
            x = nn_flax.Dense(self.hidden_size // 2)(final_hidden)
            x = nn_flax.relu(x)
            if training:
                x = nn_flax.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

            x = nn_flax.Dense(self.hidden_size // 4)(x)
            x = nn_flax.relu(x)
            if training:
                x = nn_flax.Dropout(rate=self.dropout_rate // 2)(x, deterministic=not training)

            x = nn_flax.Dense(1)(x)
            x = nn_flax.sigmoid(x)

            return x.squeeze(-1)


class GRUInjuryPredictor:
    """GRU伤病风险预测器"""

    def __init__(self,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 use_attention: bool = False,
                 backend: str = 'pytorch',
                 device: str = 'auto',
                 random_state: int = 42):
        """
        初始化GRU预测器

        Args:
            hidden_size: GRU隐藏层大小
            num_layers: GRU层数
            dropout: Dropout比率
            bidirectional: 是否使用双向GRU
            use_attention: 是否使用注意力机制
            backend: 使用的深度学习框架
            device: 计算设备
            random_state: 随机种子
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.backend = backend.lower()
        self.random_state = random_state

        # 设备设置
        if device == 'auto':
            if self.backend == 'pytorch':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = 'gpu' if jax.devices('gpu') else 'cpu'
        else:
            self.device = device

        # 模型相关
        self.model = None
        self.train_state = None  # For Flax
        self.is_fitted = False
        self.input_size = None
        self.training_history = {}

        # 设置随机种子
        self._set_random_seed()

        logger.info(f"Initialized GRU predictor with {self.backend} backend on {self.device}")

    def _set_random_seed(self):
        """设置随机种子"""
        np.random.seed(self.random_state)

        if self.backend == 'pytorch':
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
        elif FLAX_AVAILABLE:
            import jax.random as random
            self.rng_key = random.PRNGKey(self.random_state)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            early_stopping_patience: int = 10,
            verbose: bool = True) -> Dict[str, Any]:
        """
        训练GRU模型

        Args:
            X: 输入序列数据 (n_samples, seq_len, n_features)
            y: 目标变量 (n_samples,)
            validation_data: 验证数据
            epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率
            early_stopping_patience: 早停耐心值
            verbose: 是否显示训练过程

        Returns:
            训练结果字典
        """
        logger.info(f"Starting GRU training with {self.backend} backend")

        self.input_size = X.shape[-1]

        if self.backend == 'pytorch':
            return self._fit_pytorch(X, y, validation_data, epochs, batch_size,
                                   learning_rate, early_stopping_patience, verbose)
        elif self.backend == 'flax' and FLAX_AVAILABLE:
            return self._fit_flax(X, y, validation_data, epochs, batch_size,
                                learning_rate, early_stopping_patience, verbose)
        else:
            raise ValueError(f"Backend {self.backend} not available or not supported")

    def _fit_pytorch(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
                    epochs: int,
                    batch_size: int,
                    learning_rate: float,
                    early_stopping_patience: int,
                    verbose: bool) -> Dict[str, Any]:
        """PyTorch训练实现"""

        # 创建模型
        self.model = PyTorchGRUModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.FloatTensor(y).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if validation_data is not None:
            val_X, val_y = validation_data
            val_dataset = TensorDataset(
                torch.FloatTensor(val_X).to(self.device),
                torch.FloatTensor(val_y).to(self.device)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.BCELoss()

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # 训练循环
        best_val_auc = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X, use_attention=self.use_attention)
                loss = criterion(outputs, batch_y)

                # L2正则化
                l2_reg = torch.tensor(0., device=self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += 1e-5 * l2_reg

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            scheduler.step()

            # 验证阶段
            if val_loader is not None:
                val_loss, val_auc = self._evaluate_pytorch(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_auc'].append(val_auc)

                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

                # 早停检查（基于AUC）
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
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
            else:
                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")

        self.is_fitted = True
        self.training_history = history

        # 计算最终指标
        final_results = {
            'model_type': 'gru_pytorch',
            'epochs_trained': epoch + 1,
            'best_val_auc': best_val_auc,
            'training_history': history,
            'use_attention': self.use_attention
        }

        if val_loader is not None:
            final_val_loss, final_val_auc = self._evaluate_pytorch(val_loader, criterion)
            final_results.update({
                'final_val_loss': final_val_loss,
                'final_val_auc': final_val_auc
            })

        logger.info("PyTorch GRU training completed")
        return final_results

    def _evaluate_pytorch(self, data_loader, criterion):
        """PyTorch模型评估"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                outputs = self.model(batch_X, use_attention=self.use_attention)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        # 计算AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_targets, all_predictions)
        except:
            auc = 0.0

        return avg_loss, auc

    def _fit_flax(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 early_stopping_patience: int,
                 verbose: bool) -> Dict[str, Any]:
        """Flax训练实现"""
        if not FLAX_AVAILABLE:
            raise ImportError("Flax/JAX not available")

        # 创建模型
        model = FlaxGRUModel(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout
        )

        # 初始化参数
        self.rng_key, init_rng = jax.random.split(self.rng_key)
        params = model.init(init_rng, jnp.ones((1, X.shape[1], X.shape[2])))['params']

        # 创建训练状态
        # 使用AdamW优化器
        tx = optax.adamw(learning_rate, weight_decay=1e-5)
        self.train_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )

        # 训练循环
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # 训练一个epoch
            epoch_loss = self._train_epoch_flax(X, y, batch_size)
            history['train_loss'].append(epoch_loss)

            # 验证
            if validation_data is not None:
                val_X, val_y = validation_data
                val_loss, val_auc = self._evaluate_flax(val_X, val_y, batch_size)
                history['val_loss'].append(val_loss)
                history['val_auc'].append(val_auc)

                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

                # 早停检查
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # 保存最佳参数
                    self.best_params = jax.tree_map(lambda x: x, self.train_state.params)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        # 恢复最佳参数
                        self.train_state = self.train_state.replace(params=self.best_params)
                        break

        self.is_fitted = True
        self.training_history = history

        final_results = {
            'model_type': 'gru_flax',
            'epochs_trained': epoch + 1,
            'best_val_auc': best_val_auc,
            'training_history': history
        }

        logger.info("Flax GRU training completed")
        return final_results

    def _train_epoch_flax(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        """Flax训练一个epoch"""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        epoch_losses = []

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = jnp.array(X[batch_indices])
            batch_y = jnp.array(y[batch_indices])

            self.rng_key, dropout_key = jax.random.split(self.rng_key)
            self.train_state, loss = self._update_flax(self.train_state, batch_X, batch_y, dropout_key)
            epoch_losses.append(loss)

        return np.mean(epoch_losses)

    def _update_flax(self, state, X, y, rng_key):
        """Flax参数更新"""
        def loss_fn(params):
            predictions = state.apply_fn({'params': params}, X, training=True, rngs={'dropout': rng_key})
            # Binary cross entropy loss
            loss = -jnp.mean(y * jnp.log(predictions + 1e-15) + (1 - y) * jnp.log(1 - predictions + 1e-15))

            # L2正则化
            l2_reg = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
            loss += 1e-5 * l2_reg

            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def _evaluate_flax(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[float, float]:
        """Flax模型评估"""
        n_samples = X.shape[0]
        all_predictions = []
        all_losses = []

        for i in range(0, n_samples, batch_size):
            batch_X = jnp.array(X[i:i + batch_size])
            batch_y = jnp.array(y[i:i + batch_size])

            predictions = self.train_state.apply_fn(
                {'params': self.train_state.params}, batch_X, training=False
            )

            loss = -jnp.mean(batch_y * jnp.log(predictions + 1e-15) +
                           (1 - batch_y) * jnp.log(1 - predictions + 1e-15))

            all_predictions.extend(predictions)
            all_losses.append(loss)

        avg_loss = np.mean(all_losses)

        # 计算AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, np.array(all_predictions))
        except:
            auc = 0.0

        return avg_loss, auc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.backend == 'pytorch':
            return self._predict_proba_pytorch(X)
        elif self.backend == 'flax':
            return self._predict_proba_flax(X)

    def _predict_proba_pytorch(self, X: np.ndarray) -> np.ndarray:
        """PyTorch预测概率"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor, use_attention=self.use_attention)

        return predictions.cpu().numpy()

    def _predict_proba_flax(self, X: np.ndarray) -> np.ndarray:
        """Flax预测概率"""
        X_jax = jnp.array(X)
        predictions = self.train_state.apply_fn(
            {'params': self.train_state.params}, X_jax, training=False
        )
        return np.array(predictions)

    def get_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """获取伤病风险分数（0-1）"""
        return self.predict_proba(X)

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == 'pytorch':
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout,
                    'bidirectional': self.bidirectional,
                    'use_attention': self.use_attention
                },
                'backend': self.backend,
                'training_history': self.training_history
            }
            torch.save(model_data, filepath)
        elif self.backend == 'flax':
            import pickle
            model_data = {
                'train_state': self.train_state,
                'model_config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout,
                    'use_attention': self.use_attention
                },
                'backend': self.backend,
                'training_history': self.training_history
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        if self.backend == 'pytorch':
            model_data = torch.load(filepath, map_location=self.device)
            config = model_data['model_config']

            self.input_size = config['input_size']
            self.hidden_size = config['hidden_size']
            self.num_layers = config['num_layers']
            self.dropout = config['dropout']
            self.bidirectional = config['bidirectional']
            self.use_attention = config.get('use_attention', False)

            self.model = PyTorchGRUModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            ).to(self.device)

            self.model.load_state_dict(model_data['model_state_dict'])

        elif self.backend == 'flax':
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.train_state = model_data['train_state']
            config = model_data['model_config']

            self.input_size = config['input_size']
            self.hidden_size = config['hidden_size']
            self.num_layers = config['num_layers']
            self.dropout = config['dropout']
            self.use_attention = config.get('use_attention', False)

        self.training_history = model_data.get('training_history', {})
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        summary = {
            "model_type": f"GRU ({self.backend})",
            "backend": self.backend,
            "device": str(self.device),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
            "is_fitted": self.is_fitted
        }

        if self.backend == 'pytorch' and self.bidirectional is not None:
            summary["bidirectional"] = self.bidirectional

        if self.input_size is not None:
            summary["input_size"] = self.input_size

        if self.training_history:
            history = self.training_history
            summary["training_epochs"] = len(history.get('train_loss', []))
            if 'val_auc' in history and history['val_auc']:
                summary["best_val_auc"] = max(history['val_auc'])

        return summary