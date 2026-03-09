import torch
import torch.nn as nn
import torch.nn.functional as F
from core.attention import CustomMultiHeadAttention

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x):
        # ТВОЯ ЗАДАЧА №1: Реализовать RMSNorm
        # 1. Возведи x в квадрат
        # 2. Найди среднее значение по последней оси (dim=-1, keepdim=True)
        # 3. Прибавь self.eps (для стабильности, чтобы не делить на ноль)
        # 4. Извлеки квадратный корень (это и есть RMS - Root Mean Square)
        # 5. Подели x на RMS и умножь на self.weight
        # ...
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x/rms * self.weight
        return y


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self,x):
        # ТВОЯ ЗАДАЧА №2: Реализовать SwiGLU
        # Формула: ( SiLU(x * W1) * (x * W2) ) * W3
        # Подсказка: функция SiLU в PyTorch это F.silu()
        # Оператор * здесь - это поэлементное умножение (Hadamard product)
        # ...
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class LlamaTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.attention = CustomMultiHeadAttention(embed_dim=dim, num_heads=num_heads)
        self.feed_forward = SwiGLU(dim=dim, hidden_dim=hidden_dim)

        self.attention_norm = RMSNorm(dim=dim)
        self.ffn_norm = RMSNorm(dim=dim)

    def forward(self, x):
        # ТВОЯ ЗАДАЧА №3: Собрать Pre-Norm архитектуру с Residual связями
        # 1. Сохрани x (residual)
        # 2. Пропусти x через attention_norm
        # 3. Пропусти результат через attention
        # 4. Прибавь residual к результату (это новый x)

        # 5. Сохрани новый x (новый residual)
        # 6. Пропусти x через ffn_norm
        # 7. Пропусти результат через feed_forward
        # 8. Прибавь residual к результату

        # Верни итоговый x
        # ...
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
