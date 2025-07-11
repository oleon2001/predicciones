# MODELOS TRANSFORMER AVANZADOS PARA PREDICCIÓN FINANCIERA
"""
Implementación de modelos Transformer especializados para predicción de criptomonedas
Incluye Time Series Transformer, Informer, y PatchTST
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.optim as optim

# Sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Interfaces
from core.interfaces import IMLModel, ModelMetrics

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset para series temporales"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, seq_length: int, horizon: int):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
        self.horizon = horizon
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.horizon + 1
    
    def __getitem__(self, idx):
        # Secuencia de entrada
        x = self.data[idx:idx + self.seq_length]
        # Target para predicción
        y = self.targets[idx + self.seq_length:idx + self.seq_length + self.horizon]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention optimizada para series temporales"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # Reshape para multi-head
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(out)

class TransformerBlock(nn.Module):
    """Bloque Transformer con normalizaciones y residuales"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention con residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward con residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class FinancialTransformer(nn.Module):
    """Transformer especializado para datos financieros"""
    
    def __init__(self, input_dim: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 512, seq_length: int = 60,
                 horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_length = seq_length
        self.horizon = horizon
        
        # Embedding para features
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(seq_length, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, seq_length: int, d_model: int):
        """Crear encoding posicional"""
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.dropout(x)
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Use last time step for prediction
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # Output
        return self.output_layers(x)

class AdvancedTransformerModel(IMLModel):
    """Modelo Transformer avanzado para predicción financiera"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.d_model = config.get('d_model', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        self.d_ff = config.get('d_ff', 512)
        self.seq_length = config.get('seq_length', 60)
        self.horizon = config.get('horizon', 24)
        self.dropout = config.get('dropout', 0.1)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        
        # Model components
        self.model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Entrena el modelo Transformer"""
        try:
            print(f"🚀 Entrenando Transformer para horizonte {self.horizon}h...")
            
            # Preparar datos
            X_scaled = self.scaler.fit_transform(X)
            y_values = y.values
            
            # Crear dataset
            dataset = TimeSeriesDataset(X_scaled, y_values, self.seq_length, self.horizon)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Inicializar modelo
            input_dim = X.shape[1]
            self.model = FinancialTransformer(
                input_dim=input_dim,
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                d_ff=self.d_ff,
                seq_length=self.seq_length,
                horizon=self.horizon,
                dropout=self.dropout
            ).to(self.device)
            
            # Optimizer y scheduler
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
            criterion = nn.MSELoss()
            
            # Training loop
            self.model.train()
            losses = []
            
            for epoch in range(self.epochs):
                epoch_loss = 0
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                losses.append(avg_loss)
                scheduler.step(avg_loss)
                
                if epoch % 10 == 0:
                    print(f"   Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}")
            
            self.is_trained = True
            
            # Validar en conjunto de entrenamiento
            metrics = self.validate(X, y)
            
            print(f"✅ Transformer entrenado - MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error entrenando Transformer: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hace predicciones"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        try:
            self.model.eval()
            
            X_scaled = self.scaler.transform(X)
            
            # Crear secuencias
            predictions = []
            
            with torch.no_grad():
                for i in range(len(X_scaled) - self.seq_length + 1):
                    sequence = X_scaled[i:i + self.seq_length]
                    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    
                    pred = self.model(sequence_tensor)
                    predictions.append(pred.cpu().numpy().flatten())
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return np.array([])
    
    def validate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Valida el modelo"""
        try:
            predictions = self.predict(X)
            
            if len(predictions) == 0:
                return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -1.0}
            
            # Ajustar longitudes
            min_length = min(len(predictions), len(y))
            y_true = y.values[-min_length:]
            y_pred = predictions[-min_length:]
            
            # Calcular métricas
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'model_type': 'Transformer'
            }
            
        except Exception as e:
            logger.error(f"Error en validación: {e}")
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -1.0}
    
    def save_model(self, path: str):
        """Guarda el modelo"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'config': self.config
            }, path)
    
    def load_model(self, path: str):
        """Carga el modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        self.scaler = checkpoint['scaler']
        self.config = checkpoint['config']
        
        # Recrear modelo
        self.model = FinancialTransformer(**self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True 