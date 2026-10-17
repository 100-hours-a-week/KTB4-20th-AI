"""후보 제한 결정론 빔 서치 (M3 T3-3, ADR-0003).

후보 장소(지역, 품질 필터, 회피조건, 이미 쓴 장소 등을 적용한 집합)의 코드로 트라이를 만들고,
모델이 낸 확률 중 트라이에 있는 다음 토큰만 고른다. 그래서 출력은 항상 후보 집합 안의 실제 장소다.
같은 입력과 같은 가중치에서는 같은 결과가 나온다(점수가 같으면 토큰 번호가 작은 쪽).

next_logprobs(prefixes) → 각 접두사 다음 토큰의 로그확률(어휘 전체)을 돌려주는 함수를 받는다.
모델과 분리해 두었으므로 PyTorch 모델, ONNX, 시험용 가짜 모델에 모두 쓸 수 있다.
"""

from collections.abc import Callable, Iterable, Sequence

from scripts.semantic_id.tokens import place_tokens

Trie = dict  # 토큰 → 하위 Trie. 잎(깊이 4)은 {"gid": google_place_id}


def build_trie(codes_by_gid: dict[str, Sequence[int]]) -> Trie:
    root: Trie = {}
    for gid in sorted(codes_by_gid):
        node = root
        for token in place_tokens(codes_by_gid[gid]):
            node = node.setdefault(token, {})
        node["gid"] = gid  # 정수 키(토큰)와 겹치지 않는 문자열 키
    return root


def beam_search(
    context: Sequence[int], trie: Trie, next_logprobs: Callable[[list[list[int]]], list[Sequence[float]]], width: int,
) -> list[tuple[str, float]]:
    """context 다음에 올 장소를 trie 안에서 width개까지 찾는다. 반환: [(google_place_id, 로그확률)] 점수 내림차순."""
    beams: list[tuple[float, list[int], Trie]] = [(0.0, [], trie)]
    for _ in range(4):
        logprobs = next_logprobs([list(context) + prefix for _, prefix, _ in beams])
        expanded: list[tuple[float, list[int], Trie]] = []
        for (score, prefix, node), lp in zip(beams, logprobs, strict=True):
            for token, child in node.items():
                if token == "gid":
                    continue
                expanded.append((score + float(lp[token]), [*prefix, token], child))
        expanded.sort(key=lambda b: (-b[0], b[1]))
        beams = expanded[:width]
        if not beams:
            return []
    return [(node["gid"], score) for score, _, node in beams]


def rank_candidates(
    context: Sequence[int], candidates: dict[str, Sequence[int]], next_logprobs, width: int,
) -> list[str]:
    trie = build_trie(candidates)
    return [gid for gid, _ in beam_search(context, trie, next_logprobs, width)]


def torch_next_logprobs(model, device) -> Callable[[list[list[int]]], list[Sequence[float]]]:
    import torch

    model.eval()

    @torch.no_grad()
    def fn(prefixes: list[list[int]]):
        toks = torch.tensor(prefixes, dtype=torch.long, device=device)
        return torch.log_softmax(model(toks)[:, -1, :].float(), dim=-1).cpu().tolist()

    return fn


def exclude(candidates: dict[str, Sequence[int]], used: Iterable[str]) -> dict[str, Sequence[int]]:
    used = set(used)
    return {g: c for g, c in candidates.items() if g not in used}
