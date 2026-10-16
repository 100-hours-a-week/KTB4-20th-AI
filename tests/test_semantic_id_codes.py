import pytest

from scripts.semantic_id import codes


def test_collision_ids_are_assigned_in_key_order():
    got = codes.assign_collision_ids({"b": (1, 2, 3), "a": (1, 2, 3), "c": (9, 9, 9)})
    assert got == {"a": (1, 2, 3, 0), "b": (1, 2, 3, 1), "c": (9, 9, 9, 0)}


def test_collision_ids_do_not_depend_on_input_order():
    one = codes.assign_collision_ids({"x": (0, 0, 0), "y": (0, 0, 0)})
    two = codes.assign_collision_ids({"y": (0, 0, 0), "x": (0, 0, 0)})
    assert one == two


def test_collision_stats():
    stats = codes.collision_stats({"a": (1, 1, 1), "b": (1, 1, 1), "c": (2, 2, 2), "d": (3, 3, 3)})
    assert stats["places"] == 4 and stats["distinct_codes"] == 3
    assert stats["colliding_places"] == 2 and stats["max_group"] == 2


def test_usage_stats_counts_used_codes_and_perplexity():
    out = codes.usage_stats([(0, 0, 0), (0, 1, 0), (1, 2, 0), (1, 3, 0)], levels=3, codebook_size=256)
    assert out[0]["used"] == 2 and out[0]["perplexity"] == 2.0
    assert out[2]["used"] == 1 and out[2]["perplexity"] == 1.0


def test_prefix_agreement():
    pairs = [((1, 2, 3), (1, 2, 3)), ((1, 2, 3), (1, 2, 9)), ((1, 2, 3), (1, 9, 9)), ((1, 2, 3), (9, 9, 9))]
    assert codes.prefix_agreement(pairs) == {1: 0.75, 2: 0.5, 3: 0.25}


def test_random_pair_agreement_is_reproducible_and_low_for_diverse_codes():
    by_key = {f"k{i}": (i % 256, (i * 7) % 256, (i * 13) % 256) for i in range(2000)}
    a = codes.random_pair_agreement(by_key, samples=5000, seed=1)
    assert a == codes.random_pair_agreement(by_key, samples=5000, seed=1)
    assert a[3] <= a[2] <= a[1] < 0.05


def test_rqvae_trains_and_returns_valid_codes():
    torch = pytest.importorskip("torch")
    from scripts.semantic_id import rqvae

    g = torch.Generator().manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(600, 24, generator=g), dim=1)
    model = rqvae.train(x, epochs=2, batch_size=256, seed=0, log=lambda *_: None)
    model.eval()
    got = model.encode_codes(x)
    assert got.shape == (600, rqvae.LEVELS)
    assert int(got.min()) >= 0 and int(got.max()) < rqvae.CODEBOOK
    again = rqvae.train(x, epochs=2, batch_size=256, seed=0, log=lambda *_: None)
    again.eval()
    assert torch.equal(got, again.encode_codes(x))  # 같은 시드면 같은 코드
