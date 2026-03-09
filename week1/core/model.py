import torch
import torch.nn as nn
from core.blocks import LlamaTransformerBlock, RMSNorm
def precompute_rope_angles(dim, seq_len, theta=10000.0):
    """
    Магия RoPE (Rotary Positional Embedding).
    Мы заранее вычисляем углы поворота для каждой позиции токена.
    """
    # 1. Вычисляем частоты для каждой пары измерений
    # Используем torch.arange для создания последовательности [0, 2, 4... dim]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 2. Создаем позиции токенов [0, 1, 2... seq_len-1]
    t = torch.arange(seq_len, dtype=torch.float32)

    # 3. Внешнее произведение (каждая позиция умножается на каждую частоту)
    # Используем torch.outer!
    freqs_outer = torch.outer(t, freqs)

    # 4. Превращаем углы в комплексные числа (cos + i*sin) для быстрого вращения
    # torch.polar(абсолютное значение=1.0, угол=freqs_outer)
    complex_angles = torch.polar(torch.ones_like(freqs_outer), freqs_outer)
    return complex_angles

class MiniLlama(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        # ТВОЯ ЗАДАЧА №1: Создать Embedding слой (используй nn.Embedding)
        self.tok_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim) # Напиши код

        # ТВОЯ ЗАДАЧА №2: Создать список блоков Трансформера (используй nn.ModuleList)
        # Нам нужно создать num_layers штук LlamaTransformerBlock
        self.layers = nn.ModuleList([
            LlamaTransformerBlock(dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim) for _ in range(num_layers)
            # Напиши код генерации блоков
        ])

        # ТВОЯ ЗАДАЧА №3: Финальный RMSNorm и Линейный слой (lm_head)
        # Линейный слой должен перевести из embed_dim обратно в vocab_size
        self.norm = RMSNorm(embed_dim) # Напиши код
        self.lm_head = nn.Linear(embed_dim, vocab_size) # Напиши код

    def forward(self, tokens):

        # 1. Получаем эмбеддинги токенов
        x = self.tok_embeddings(tokens) # [B, T, C]

        # (В реальной Llama здесь применяется RoPE к Query и Key внутри Attention,
        # но для простоты архитектуры сегодня мы пропустим интеграцию RoPE внутрь твоего
        # CustomMultiHeadAttention, чтобы не сломать вчерашний код. Оставим RoPE на десерт).

        # 2. Пропускаем через все слои Трансформера
        for layer in self.layers:
            x = layer(x)

        # 3. Финальная нормализация
        x = self.norm(x)

        # 4. Получаем логиты (вероятности) для словаря
        logits = self.lm_head(x) # [B, T, Vocab_Size]

        return logits
