"""생성 모델의 입력 토큰 설계와 하루 시퀀스 → 토큰열 변환 (M3 T3-1, ADR-0003).

토큰열: [BOS, 취향 4개, 지역 1개, (c1, c2, c3, n) × 방문 수]
  - 취향: 카테고리마다 0/1 토큰 하나(HISTORY_CULTURE, NATURE_HEALING, ACTIVITY, CONVENIENCE_SHOPPING 순서)
  - 지역: 서울, 부산, 제주, 경주, 전주
  - 장소: Semantic ID (c1, c2, c3, n). 단계마다 따로 번호 구간을 써서 같은 숫자라도 단계가 다르면 다른 토큰이다
  - 회피조건과 슬롯 제약은 입력에 넣지 않고 디코딩 후보 제한으로 처리한다(ADR-0003)

학습 손실은 정답으로 쓸 수 있는 방문(DB 장소에 대응된 방문)의 4개 토큰에만 건다.
대응되지 않은 방문은 가장 가까운 DB 장소의 코드를 앞 문맥으로만 쓴다(T2-6).
"""

from collections.abc import Sequence

from scripts.semantic_id.preference import MISSION_CATEGORIES

CATEGORIES = tuple(MISSION_CATEGORIES)  # 입력 순서. 딕셔너리 정의 순서를 그대로 쓴다
REGIONS = ("서울", "부산", "제주", "경주", "전주")
LEVELS = 3
CODEBOOK = 256
MAX_COLLISION = 8  # 충돌 구분 번호 상한(실측 최대 묶음 5)
TOKENS_PER_PLACE = 4

PAD, BOS = 0, 1
PREF_BASE = 2
REGION_BASE = PREF_BASE + 2 * len(CATEGORIES)
CODE_BASE = REGION_BASE + len(REGIONS)
COLLISION_BASE = CODE_BASE + LEVELS * CODEBOOK
VOCAB_SIZE = COLLISION_BASE + MAX_COLLISION
HEADER_LEN = 1 + len(CATEGORIES) + 1  # BOS + 취향 + 지역

Code = tuple[int, int, int, int]


def place_tokens(code: Sequence[int]) -> list[int]:
    c1, c2, c3, n = code
    if not (0 <= c1 < CODEBOOK and 0 <= c2 < CODEBOOK and 0 <= c3 < CODEBOOK and 0 <= n < MAX_COLLISION):
        raise ValueError(f"범위 밖 코드: {tuple(code)}")
    return [CODE_BASE + c1, CODE_BASE + CODEBOOK + c2, CODE_BASE + 2 * CODEBOOK + c3, COLLISION_BASE + n]


def token_to_code_part(token: int) -> tuple[int, int]:
    """장소 토큰 → (자리 0~3, 값). 장소 토큰이 아니면 오류."""
    if CODE_BASE <= token < COLLISION_BASE:
        return (token - CODE_BASE) // CODEBOOK, (token - CODE_BASE) % CODEBOOK
    if COLLISION_BASE <= token < VOCAB_SIZE:
        return 3, token - COLLISION_BASE
    raise ValueError(f"장소 토큰이 아님: {token}")


def header_tokens(preference: dict[str, int], region: str) -> list[int]:
    out = [BOS]
    for k, category in enumerate(CATEGORIES):
        out.append(PREF_BASE + 2 * k + int(bool(preference[category])))
    out.append(REGION_BASE + REGIONS.index(region))
    return out


def encode_day(
    preference: dict[str, int], region: str, codes: Sequence[Sequence[int]], targets: Sequence[bool]
) -> tuple[list[int], list[int]]:
    """하루를 (토큰열, 손실 마스크)로 바꾼다. targets[j]가 참인 방문만 손실에 쓴다."""
    if len(codes) != len(targets):
        raise ValueError("codes와 targets의 길이가 다름")
    tokens = header_tokens(preference, region)
    mask = [0] * len(tokens)
    for code, is_target in zip(codes, targets, strict=True):
        tokens += place_tokens(code)
        mask += [int(is_target)] * TOKENS_PER_PLACE
    return tokens, mask
