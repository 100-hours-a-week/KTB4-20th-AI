"""학습한 모델을 ONNX로 내보내고 PyTorch 출력과 같은지 확인한다 (M5 T5-1, ADR-0004).

내보내는 그래프는 토큰열(배치, 길이)을 받아 마지막 위치의 다음 토큰 로그확률(배치, 어휘)을 낸다.
빔 서치와 후보 제한은 모델 밖(decode.py)에서 한다.

  uv run --with torch --with onnx --with onnxruntime --with numpy python -m scripts.semantic_id.export_onnx \\
      --model-dir "$V/m3/model_a" --out "$V/m3/model_a/model.onnx"
"""

import argparse
import pathlib
import random
import sys

from scripts.semantic_id.tokens import VOCAB_SIZE

TOLERANCE = 1e-3  # 로그확률 최대 절대 오차


def load_model(model_dir: pathlib.Path):
    import torch

    from scripts.semantic_id.model import PlaceLM

    ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
    model = PlaceLM(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model.eval()


def wrap_last_logprobs(model):
    import torch
    from torch import nn

    class LastLogProbs(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, tokens):
            return torch.log_softmax(self.inner(tokens)[:, -1, :], dim=-1)

    return LastLogProbs(model).eval()


def export(model, out: pathlib.Path) -> None:
    import torch

    wrapped = wrap_last_logprobs(model)
    sample = torch.randint(0, VOCAB_SIZE, (2, 10), dtype=torch.long)
    torch.onnx.export(
        wrapped, (sample,), str(out), input_names=["tokens"], output_names=["logprobs"],
        dynamic_axes={"tokens": {0: "batch", 1: "length"}, "logprobs": {0: "batch"}}, opset_version=17, dynamo=False,
    )


def parity(model, onnx_path: pathlib.Path, cases: int = 20, seed: int = 20261016) -> float:
    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    wrapped = wrap_last_logprobs(model)
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(cases):
        batch, length = rng.randint(1, 12), rng.randint(6, 30)
        tokens = torch.tensor([[rng.randrange(VOCAB_SIZE) for _ in range(length)] for _ in range(batch)], dtype=torch.long)
        with torch.no_grad():
            expected = wrapped(tokens).numpy()
        got = session.run(None, {"tokens": tokens.numpy()})[0]
        worst = max(worst, float(np.abs(expected - got).max()))
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"이미 있어서 덮어쓰지 않음: {args.out}", file=sys.stderr)
        return 1
    model = load_model(args.model_dir)
    export(model, args.out)
    worst = parity(model, args.out)
    print(f"ONNX 저장: {args.out} ({args.out.stat().st_size / 1e6:.1f}MB) · PyTorch와 최대 절대 오차 {worst:.2e} (허용 {TOLERANCE})")
    return 0 if worst <= TOLERANCE else 1


if __name__ == "__main__":
    raise SystemExit(main())
