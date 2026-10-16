"""RQ-VAE로 DB 장소 임베딩을 코드 묶음(Semantic ID)으로 바꾼다 (M2 T2-4, ADR-0003).

구조(TIGER 설정): 인코더 입력 → 512 → 256 → 128 → 잠재 32, 3단계 잔차 양자화(단계마다 코드북 256개), 디코더는 대칭.
손실 = 복원 + 코드북(β 없음) + β × 커밋(β = 0.25). 코드북은 학습 전에 인코더 출력의 k-means로 초기화하고,
학습 중 쓰이지 않는 코드(죽은 코드)는 주기적으로 다시 뽑는다.

학습은 DB 장소만으로 하고, 로그 방문지는 학습된 인코더로 코드만 얻는다(ADR-0003). torch는 프로젝트 의존성이 아니라
`uv run --with torch --with numpy`로 임시 환경에서 쓴다.

  uv run --with torch --with numpy python -m scripts.semantic_id.rqvae --embed-dir "$OUT/embed_qwen3" --out-dir "$OUT/rqvae_qwen3"
"""

import argparse
import itertools
import json
import pathlib
import sys

LEVELS = 3
CODEBOOK = 256
LATENT = 32
HIDDEN = (512, 256, 128)
BETA = 0.25
SEED = 20261016


def build_model(in_dim: int):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class RQVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = (in_dim, *HIDDEN, LATENT)
            enc = []
            for a, b in itertools.pairwise(dims):
                enc += [nn.Linear(a, b), nn.ReLU()]
            self.encoder = nn.Sequential(*enc[:-1])  # 마지막 활성화는 뺀다
            rdims = dims[::-1]
            dec = []
            for a, b in itertools.pairwise(rdims):
                dec += [nn.Linear(a, b), nn.ReLU()]
            self.decoder = nn.Sequential(*dec[:-1])
            self.codebooks = nn.Parameter(torch.randn(LEVELS, CODEBOOK, LATENT) * 0.1)

        def quantize(self, z):
            residual = z
            quantized = torch.zeros_like(z)
            codes = []
            codebook_loss = z.new_zeros(())
            commit_loss = z.new_zeros(())
            for lvl in range(LEVELS):
                cb = self.codebooks[lvl]
                dist = (residual**2).sum(1, keepdim=True) - 2 * residual @ cb.T + (cb**2).sum(1)[None]
                idx = dist.argmin(1)
                q = cb[idx]
                codebook_loss = codebook_loss + F.mse_loss(q, residual.detach())
                commit_loss = commit_loss + F.mse_loss(residual, q.detach())
                quantized = quantized + q
                residual = residual - q.detach()
                codes.append(idx)
            z_q = z + (quantized - z).detach()  # 직선 통과(straight-through)
            return z_q, torch.stack(codes, 1), codebook_loss, commit_loss

        def forward(self, x):
            z = self.encoder(x)
            z_q, codes, codebook_loss, commit_loss = self.quantize(z)
            recon = self.decoder(z_q)
            recon_loss = ((recon - x) ** 2).sum(1).mean()
            return recon_loss + codebook_loss + BETA * commit_loss, recon_loss, codes

        @torch.no_grad()
        def encode_codes(self, x):
            return self.quantize(self.encoder(x))[1]

    return RQVAE()


def _kmeans(points, k: int, iters: int = 25, seed: int = SEED):
    import torch

    g = torch.Generator().manual_seed(seed)
    centers = points[torch.randperm(points.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(points, centers).argmin(1)
        for j in range(k):
            members = points[assign == j]
            if len(members):
                centers[j] = members.mean(0)
    return centers


def init_codebooks(model, x) -> None:
    # 단계마다 남은 잔차를 k-means로 나눠 코드북을 초기화한다(코드북 붕괴를 막음)
    import torch

    with torch.no_grad():
        residual = model.encoder(x)
        for lvl in range(LEVELS):
            centers = _kmeans(residual, CODEBOOK, seed=SEED + lvl)
            model.codebooks.data[lvl] = centers
            assign = torch.cdist(residual, centers).argmin(1)
            residual = residual - centers[assign]


def revive_dead_codes(model, x, generator) -> int:
    # 한 번도 쓰이지 않은 코드를 잔차가 큰 샘플 근처로 다시 뽑는다
    import torch

    revived = 0
    with torch.no_grad():
        residual = model.encoder(x)
        for lvl in range(LEVELS):
            cb = model.codebooks.data[lvl]
            assign = torch.cdist(residual, cb).argmin(1)
            counts = torch.bincount(assign, minlength=CODEBOOK)
            dead = (counts == 0).nonzero().flatten()
            if len(dead):
                pick = torch.randperm(residual.shape[0], generator=generator)[: len(dead)]
                cb[dead] = residual[pick] + 0.01 * torch.randn(len(dead), LATENT, generator=generator)
                revived += len(dead)
            residual = residual - model.codebooks.data[lvl][torch.cdist(residual, model.codebooks.data[lvl]).argmin(1)]
    return revived


def train(x, epochs: int = 300, batch_size: int = 1024, lr: float = 1e-3, seed: int = SEED, log=print):
    import torch

    torch.manual_seed(seed)
    model = build_model(x.shape[1])
    init_codebooks(model, x)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch_size):
            batch = x[perm[i:i + batch_size]]
            loss, recon, _ = model(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(recon) * len(batch)
        if epoch % 50 == 0 or epoch == 1:
            revived = revive_dead_codes(model, x, gen) if epoch % 50 == 0 else 0
            log(f"  epoch {epoch}: 복원 손실 {total / n:.4f}, 다시 뽑은 죽은 코드 {revived}")
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-dir", type=pathlib.Path, required=True, help="embed.py의 결과 폴더")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    if (args.out_dir / "meta.json").exists():
        print(f"이미 있어서 다시 학습하지 않음: {args.out_dir}", file=sys.stderr)
        return 1
    import numpy as np
    import torch

    from scripts.semantic_id import codes as codes_mod

    vectors = np.load(args.embed_dir / "embeddings.npy").astype("float32")
    items = [json.loads(line) for line in (args.embed_dir / "texts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    db_rows = [n for n, i in enumerate(items) if i["source"] == "db"]
    x_db = torch.from_numpy(vectors[db_rows])
    print(f"DB 장소 {len(db_rows)}개로 RQ-VAE 학습 (입력 {x_db.shape[1]}차원, {args.epochs} epoch)")
    model = train(x_db, epochs=args.epochs, seed=args.seed)

    model.eval()
    with torch.no_grad():
        all_codes = model.encode_codes(torch.from_numpy(vectors)).numpy().astype("int16")
        recon = float(((model.decoder(model.quantize(model.encoder(x_db))[0]) - x_db) ** 2).sum(1).mean())
    db_keys = [items[n]["key"] for n in db_rows]
    db_codes = {k: tuple(int(c) for c in all_codes[n]) for k, n in zip(db_keys, db_rows)}
    with_collision = codes_mod.assign_collision_ids(db_codes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "codes.npy", all_codes)  # embed-dir의 texts.jsonl과 같은 행 순서
    (args.out_dir / "db_semantic_ids.json").write_text(
        json.dumps({k: list(v) for k, v in sorted(with_collision.items())}, ensure_ascii=False) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), args.out_dir / "rqvae.pt")
    meta = {
        "embed_dir": args.embed_dir.name, "levels": LEVELS, "codebook": CODEBOOK, "latent": LATENT, "hidden": list(HIDDEN),
        "beta": BETA, "epochs": args.epochs, "seed": args.seed, "recon_loss_db": round(recon, 4),
        "usage_db": codes_mod.usage_stats(list(db_codes.values()), LEVELS, CODEBOOK),
        "collisions_db": codes_mod.collision_stats(db_codes),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("단계별 사용 코드:", meta["usage_db"])
    print("충돌:", meta["collisions_db"])
    print(f"저장: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
