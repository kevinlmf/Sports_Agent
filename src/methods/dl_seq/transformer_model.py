"""
Transformer model for injury risk prediction using both PyTorch and Flax implementations
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
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


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PyTorchTransformerModel(nn.Module):
    """PyTorch Transformer模型"""

    def __init__(self,
                 input_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_length: int = 500):
        super().__init__()

        self.d_model = d_model
        self.input_size = input_size

        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # 输出层
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            mask: (batch_size, seq_len) - True for valid positions
        """
        batch_size, seq_len = x.shape[:2]

        # 输入投影
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # 添加位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # 创建注意力mask (Transformer expects False for valid positions)
        if mask is not None:
            # mask: True for valid positions -> src_key_padding_mask: True for invalid positions
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        # Transformer编码
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 全局池化 - 使用注意力权重或简单平均
        if mask is not None:
            # 使用mask进行加权平均
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            encoded_masked = encoded * mask_expanded
            pooled = encoded_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            # 简单平均池化
            pooled = encoded.mean(dim=1)

        # 分类
        output = self.classifier(pooled)
        return output.squeeze(-1)


if FLAX_AVAILABLE:
    class FlaxTransformerModel(nn_flax.Module):
        """Flax Transformer模型"""

        d_model: int = 256
        nhead: int = 8
        num_encoder_layers: int = 4
        dim_feedforward: int = 512
        dropout_rate: float = 0.1

        @nn_flax.compact
        def __call__(self, x, mask=None, training=False):
            batch_size, seq_len, input_size = x.shape

            # 输入投影
            x = nn_flax.Dense(self.d_model)(x)
            x = x * jnp.sqrt(self.d_model)

            # 位置编码
            x = self._add_position_encoding(x)

            # 多层Transformer编码器
            for _ in range(self.num_encoder_layers):
                x = self._transformer_encoder_layer(x, mask, training)

            # 全局池化
            if mask is not None:
                mask_expanded = jnp.expand_dims(mask, -1)
                x_masked = x * mask_expanded
                pooled = jnp.sum(x_masked, axis=1) / jnp.sum(mask, axis=1, keepdims=True).clip(min=1)
            else:
                pooled = jnp.mean(x, axis=1)

            # 分类头
            x = nn_flax.LayerNorm()(pooled)
            if training:
                x = nn_flax.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

            x = nn_flax.Dense(self.d_model // 2)(x)
            x = nn_flax.gelu(x)
            if training:
                x = nn_flax.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

            x = nn_flax.Dense(self.d_model // 4)(x)
            x = nn_flax.gelu(x)
            if training:
                x = nn_flax.Dropout(rate=self.dropout_rate // 2)(x, deterministic=not training)

            x = nn_flax.Dense(1)(x)
            x = nn_flax.sigmoid(x)

            return x.squeeze(-1)

        def _add_position_encoding(self, x):
            """添加位置编码"""
            seq_len = x.shape[1]
            d_model = x.shape[2]

            position = jnp.arange(seq_len)[:, None]
            div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))

            pe_sin = jnp.sin(position * div_term)
            pe_cos = jnp.cos(position * div_term)

            pe = jnp.zeros((seq_len, d_model))
            pe = pe.at[:, 0::2].set(pe_sin)
            pe = pe.at[:, 1::2].set(pe_cos)

            return x + pe[None, :, :]

        def _transformer_encoder_layer(self, x, mask, training):
            """Transformer编码器层"""
            # 多头自注意力
            attn_output = nn_flax.MultiHeadDotProductAttention(
                num_heads=self.nhead,
                dropout_rate=self.dropout_rate if training else 0.0
            )(x, mask=mask)

            # 残差连接和层归一化
            x = nn_flax.LayerNorm()(x + attn_output)

            # Feed Forward Network
            ff_output = nn_flax.Dense(self.dim_feedforward)(x)
            ff_output = nn_flax.gelu(ff_output)
            if training:
                ff_output = nn_flax.Dropout(rate=self.dropout_rate)(ff_output, deterministic=not training)
            ff_output = nn_flax.Dense(self.d_model)(ff_output)

            # 残差连接和层归一化
            x = nn_flax.LayerNorm()(x + ff_output)

            return x


class TransformerInjuryPredictor:
    """Transformer伤病风险预测器"""

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_length: int = 500,
                 backend: str = 'pytorch',
                 device: str = 'auto',
                 random_state: int = 42):
        """
        初始化Transformer预测器

        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_seq_length: 最大序列长度
            backend: 使用的深度学习框架
            device: 计算设备
            random_state: 随机种子
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_length = max_seq_length
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

        logger.info(f"Initialized Transformer predictor with {self.backend} backend on {self.device}")

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
            learning_rate: float = 0.0001,
            warmup_steps: int = 1000,
            early_stopping_patience: int = 10,
            verbose: bool = True) -> Dict[str, Any]:
        """
        训练Transformer模型

        Args:
            X: 输入序列数据 (n_samples, seq_len, n_features)
            y: 目标变量 (n_samples,)
            validation_data: 验证数据
            epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率
            warmup_steps: 预热步数
            early_stopping_patience: 早停耐心值
            verbose: 是否显示训练过程

        Returns:
            训练结果字典
        """
        logger.info(f"Starting Transformer training with {self.backend} backend")

        self.input_size = X.shape[-1]

        if self.backend == 'pytorch':
            return self._fit_pytorch(X, y, validation_data, epochs, batch_size,
                                   learning_rate, warmup_steps, early_stopping_patience, verbose)
        elif self.backend == 'flax' and FLAX_AVAILABLE:
            return self._fit_flax(X, y, validation_data, epochs, batch_size,
                                learning_rate, warmup_steps, early_stopping_patience, verbose)
        else:
            raise ValueError(f"Backend {self.backend} not available or not supported")

    def _fit_pytorch(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
                    epochs: int,
                    batch_size: int,
                    learning_rate: float,
                    warmup_steps: int,
                    early_stopping_patience: int,
                    verbose: bool) -> Dict[str, Any]:
        """PyTorch训练实现"""

        # 创建模型
        self.model = PyTorchTransformerModel(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_length=self.max_seq_length
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

        # 优化器设置
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )

        # 学习率调度器（带预热）
        def get_lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.1, 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (epochs * len(train_loader) - warmup_steps))))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

        criterion = nn.BCELoss(reduction='mean')

        # 训练循环
        best_val_auc = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'learning_rate': []}
        global_step = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # 创建padding mask
                mask = (batch_X.sum(dim=-1) != 0)  # 假设全0表示padding

                outputs = self.model(batch_X, mask=mask)
                loss = criterion(outputs, batch_y)

                # Label smoothing
                loss = loss * 0.9 + 0.1 * 0.5  # 简单的label smoothing

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                global_step += 1

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 验证阶段
            if val_loader is not None:
                val_loss, val_auc = self._evaluate_pytorch(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_auc'].append(val_auc)

                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, "
                              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                # 早停检查
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
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        self.is_fitted = True
        self.training_history = history

        # 计算最终指标
        final_results = {
            'model_type': 'transformer_pytorch',
            'epochs_trained': epoch + 1,
            'best_val_auc': best_val_auc,
            'training_history': history,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

        if val_loader is not None:
            final_val_loss, final_val_auc = self._evaluate_pytorch(val_loader, criterion)
            final_results.update({
                'final_val_loss': final_val_loss,
                'final_val_auc': final_val_auc
            })

        logger.info("PyTorch Transformer training completed")
        return final_results

    def _evaluate_pytorch(self, data_loader, criterion):
        """PyTorch模型评估"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                mask = (batch_X.sum(dim=-1) != 0)
                outputs = self.model(batch_X, mask=mask)
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
                 warmup_steps: int,
                 early_stopping_patience: int,
                 verbose: bool) -> Dict[str, Any]:
        """Flax训练实现"""
        if not FLAX_AVAILABLE:
            raise ImportError("Flax/JAX not available")

        # 创建模型
        model = FlaxTransformerModel(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout_rate=self.dropout
        )

        # 初始化参数
        self.rng_key, init_rng = jax.random.split(self.rng_key)
        params = model.init(init_rng, jnp.ones((1, X.shape[1], X.shape[2])))['params']

        # 创建学习率调度
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=epochs * (len(X) // batch_size),
            end_value=learning_rate * 0.1
        )

        # 创建训练状态
        tx = optax.adamw(learning_rate=schedule, weight_decay=0.01)
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
            'model_type': 'transformer_flax',
            'epochs_trained': epoch + 1,
            'best_val_auc': best_val_auc,
            'training_history': history
        }

        logger.info("Flax Transformer training completed")
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

            # 创建mask
            mask = jnp.sum(batch_X, axis=-1) != 0

            self.rng_key, dropout_key = jax.random.split(self.rng_key)
            self.train_state, loss = self._update_flax(self.train_state, batch_X, batch_y, mask, dropout_key)
            epoch_losses.append(loss)

        return np.mean(epoch_losses)

    def _update_flax(self, state, X, y, mask, rng_key):
        """Flax参数更新"""
        def loss_fn(params):
            predictions = state.apply_fn({'params': params}, X, mask, training=True, rngs={'dropout': rng_key})
            # Binary cross entropy with label smoothing
            smoothed_y = y * 0.9 + 0.05
            loss = -jnp.mean(smoothed_y * jnp.log(predictions + 1e-15) +
                           (1 - smoothed_y) * jnp.log(1 - predictions + 1e-15))
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
            mask = jnp.sum(batch_X, axis=-1) != 0

            predictions = self.train_state.apply_fn(
                {'params': self.train_state.params}, batch_X, mask, training=False
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
            mask = (X_tensor.sum(dim=-1) != 0)
            predictions = self.model(X_tensor, mask=mask)

        return predictions.cpu().numpy()

    def _predict_proba_flax(self, X: np.ndarray) -> np.ndarray:
        """Flax预测概率"""
        X_jax = jnp.array(X)
        mask = jnp.sum(X_jax, axis=-1) != 0
        predictions = self.train_state.apply_fn(
            {'params': self.train_state.params}, X_jax, mask, training=False
        )
        return np.array(predictions)

    def get_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """获取伤病风险分数（0-1）"""
        return self.predict_proba(X)

    def get_attention_weights(self, X: np.ndarray, layer_idx: int = -1) -> np.ndarray:
        """获取注意力权重（仅PyTorch）"""
        if not self.is_fitted or self.backend != 'pytorch':
            raise ValueError("Attention weights only available for fitted PyTorch model")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        mask = (X_tensor.sum(dim=-1) != 0)

        # 需要修改模型以返回注意力权重
        # 这里提供简化实现
        with torch.no_grad():
            # 获取指定层的注意力权重
            pass  # 实际实现需要修改forward方法

        return np.array([])  # 占位返回

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
                    'd_model': self.d_model,
                    'nhead': self.nhead,
                    'num_encoder_layers': self.num_encoder_layers,
                    'dim_feedforward': self.dim_feedforward,
                    'dropout': self.dropout,
                    'max_seq_length': self.max_seq_length
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
                    'd_model': self.d_model,
                    'nhead': self.nhead,
                    'num_encoder_layers': self.num_encoder_layers,
                    'dim_feedforward': self.dim_feedforward,
                    'dropout': self.dropout
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
            self.d_model = config['d_model']
            self.nhead = config['nhead']
            self.num_encoder_layers = config['num_encoder_layers']
            self.dim_feedforward = config['dim_feedforward']
            self.dropout = config['dropout']
            self.max_seq_length = config.get('max_seq_length', 500)

            self.model = PyTorchTransformerModel(
                input_size=self.input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                max_seq_length=self.max_seq_length
            ).to(self.device)

            self.model.load_state_dict(model_data['model_state_dict'])

        elif self.backend == 'flax':
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.train_state = model_data['train_state']
            config = model_data['model_config']

            self.input_size = config['input_size']
            self.d_model = config['d_model']
            self.nhead = config['nhead']
            self.num_encoder_layers = config['num_encoder_layers']
            self.dim_feedforward = config['dim_feedforward']
            self.dropout = config['dropout']

        self.training_history = model_data.get('training_history', {})
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        summary = {
            "model_type": f"Transformer ({self.backend})",
            "backend": self.backend,
            "device": str(self.device),
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_seq_length": self.max_seq_length,
            "is_fitted": self.is_fitted
        }

        if self.input_size is not None:
            summary["input_size"] = self.input_size

        if self.backend == 'pytorch' and self.is_fitted:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            summary.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })

        if self.training_history:
            history = self.training_history
            summary["training_epochs"] = len(history.get('train_loss', []))
            if 'val_auc' in history and history['val_auc']:
                summary["best_val_auc"] = max(history['val_auc'])

        return summary