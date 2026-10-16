"""장소 생성 모델: 소형 디코더 트랜스포머 (M3 T3-2, ADR-0003).

사전학습 백본 없이 처음부터 학습한다. 입력은 tokens.py의 토큰열이고 다음 토큰의 분포를 낸다.
"""

import torch
from torch import nn

from scripts.semantic_id.tokens import VOCAB_SIZE

MAX_LEN = 32  # 헤더 6 + 방문 최대 6곳 × 4 = 30


class PlaceLM(nn.Module):
    def __init__(self, d_model: int = 256, layers: int = 4, heads: int = 4, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.config = {"d_model": d_model, "layers": layers, "heads": heads, "dropout": dropout, "ffn_mult": ffn_mult}
        self.tok = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * ffn_mult, dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight  # 입력·출력 임베딩 공유
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[1]
        x = self.drop(self.tok(tokens) + self.pos(torch.arange(n, device=tokens.device)))
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), 1)
        x = self.blocks(x, mask=causal)
        return self.head(self.norm(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
