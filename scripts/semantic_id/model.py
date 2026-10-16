"""장소 생성 모델: 소형 디코더 트랜스포머 (M3 T3-2, ADR-0003).

사전학습 백본 없이 처음부터 학습한다. 입력은 tokens.py의 토큰열이고 다음 토큰의 분포를 낸다.
어텐션을 직접 구현해 ONNX로 내보낼 때 길이·배치가 고정되지 않게 했다. 가중치 이름은
`nn.TransformerEncoder`(norm_first, gelu)와 같아서 그 구현으로 학습한 체크포인트도 그대로 읽힌다.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from scripts.semantic_id.tokens import VOCAB_SIZE

MAX_LEN = 32  # 헤더 6 + 방문 최대 6곳 × 4 = 30


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)
        pos = torch.arange(t, device=x.device)
        scores = scores.masked_fill(pos[None, :] > pos[:, None], float("-inf"))  # 뒤 토큰은 보지 않는다
        out = (self.drop(torch.softmax(scores, dim=-1)) @ v).transpose(1, 2).reshape(b, t, d)
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.self_attn = SelfAttention(d_model, heads, dropout)
        self.linear1 = nn.Linear(d_model, ffn)
        self.linear2 = nn.Linear(ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.norm1(x)))
        return x + self.drop(self.linear2(self.drop(F.gelu(self.linear1(self.norm2(x))))))


class Stack(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PlaceLM(nn.Module):
    def __init__(self, d_model: int = 256, layers: int = 4, heads: int = 4, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.config = {"d_model": d_model, "layers": layers, "heads": heads, "dropout": dropout, "ffn_mult": ffn_mult}
        self.tok = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.blocks = Stack([Block(d_model, heads, d_model * ffn_mult, dropout) for _ in range(layers)])
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
        if isinstance(m, SelfAttention):
            nn.init.normal_(m.in_proj_weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.tok(tokens) + self.pos(torch.arange(tokens.shape[1], device=tokens.device)))
        return self.head(self.norm(self.blocks(x)))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
