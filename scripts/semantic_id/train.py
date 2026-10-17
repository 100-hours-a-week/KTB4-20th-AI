"""생성 모델을 학습한다 (M3 T3-2, ADR-0003).

학습용 하루 시퀀스로 학습하고 개발용 손실이 가장 낮은 시점의 가중치를 저장한다(조기 종료).
최종용은 열지 않는다. 손실은 DB 범위 안으로 판정된 방문의 4개 토큰에만 건다.

  uv run --with torch --with numpy python -m scripts.semantic_id.train --v1 "$V" --out-dir "$V/m3/model_a" --d-model 256 --layers 4
"""

import argparse
import json
import pathlib
import random
import sys
import time

from scripts.semantic_id.split import iter_days
from scripts.semantic_id.tokens import PAD
from scripts.semantic_id.train_data import build_examples

SEED = 20261016


def load_examples(v1: pathlib.Path, part: str) -> list[dict]:
    if part == "final":
        raise ValueError("학습 코드는 최종용을 열지 않는다")
    visit_map = json.loads((v1 / "m3/visit_map.json").read_text(encoding="utf-8"))
    places = json.loads((v1 / "m3/places.json").read_text(encoding="utf-8"))
    codes = {p["google_place_id"]: p["code"] for p in places if p["code"] is not None}
    days = iter_days(v1 / "m1/days.jsonl", v1 / "m1/split.json", part)
    return build_examples(days, visit_map, codes)


def batches(examples: list[dict], size: int, shuffle: bool, rng: random.Random):
    import torch

    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), size):
        chunk = [examples[i] for i in order[start:start + size]]
        width = max(len(e["tokens"]) for e in chunk)
        toks = torch.full((len(chunk), width), PAD, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.float)
        for r, e in enumerate(chunk):
            toks[r, :len(e["tokens"])] = torch.tensor(e["tokens"])
            mask[r, :len(e["mask"])] = torch.tensor(e["mask"], dtype=torch.float)
        yield toks, mask


def masked_loss(model, toks, mask):
    import torch.nn.functional as F

    logits = model(toks[:, :-1])
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), toks[:, 1:].reshape(-1), reduction="none")
    m = mask[:, 1:].reshape(-1)
    return (ce * m).sum() / m.sum().clamp(min=1), m.sum()


def evaluate(model, examples, device) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for toks, mask in batches(examples, 256, False, random.Random(0)):
            loss, n = masked_loss(model, toks.to(device), mask.to(device))
            total += float(loss) * float(n)
            count += float(n)
    return total / max(count, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    if (args.out_dir / "model.pt").exists():
        print(f"이미 있어서 덮어쓰지 않음: {args.out_dir}", file=sys.stderr)
        return 1
    import numpy as np
    import torch

    from scripts.semantic_id.model import PlaceLM

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train, dev = load_examples(args.v1, "train"), load_examples(args.v1, "dev")
    model = PlaceLM(args.d_model, args.layers, args.heads, args.dropout).to(device)
    print(f"장치 {device} · 학습 {len(train)}일, 개발 {len(dev)}일 · 파라미터 {model.num_parameters() / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    steps_total = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_total, pct_start=0.1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve, best, bad, started = [], float("inf"), 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0.0
        for toks, mask in batches(train, args.batch, True, rng):
            loss, n = masked_loss(model, toks.to(device), mask.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * float(n)
            count += float(n)
        dev_loss = evaluate(model, dev, device)
        curve.append({"epoch": epoch, "train": round(total / count, 4), "dev": round(dev_loss, 4)})
        print(f"epoch {epoch:3d} train {total / count:.4f} dev {dev_loss:.4f} ({time.time() - started:.0f}s)")
        if dev_loss < best - 1e-4:
            best, bad = dev_loss, 0
            torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "config": model.config}, args.out_dir / "model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    meta = {"best_dev_loss": round(best, 4), "epochs_run": len(curve), "seed": args.seed, "params": model.num_parameters(),
            "train_days": len(train), "dev_days": len(dev), "args": {k: str(v) for k, v in vars(args).items()}, "curve": curve}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"최저 개발 손실 {best:.4f} · 저장 {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
