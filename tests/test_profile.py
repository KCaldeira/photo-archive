import numpy as np

from photosel.profile import desirability, diverse_pick, monte_carlo_topk, power_mean


def test_desirability_ramps():
    d = desirability(np.array([0.0, 0.5, 1.0, 2.0]), "up", 0.5, 1.0, eps=0.1)
    assert np.allclose(d, [0.1, 0.1, 1.0, 1.0])
    d = desirability(np.array([0.0, 0.02, 0.085, 0.5]), "down", 0.02, 0.15, eps=0.05)
    assert d[0] == 1.0 and d[1] == 1.0 and 0.05 < d[2] < 1.0 and d[3] == 0.05


def test_power_mean_limits():
    D = np.array([[0.9, 0.1], [0.5, 0.5]])
    w = np.array([1.0, 1.0])
    assert np.allclose(power_mean(D, w, 1.0), [0.5, 0.5])
    assert np.allclose(power_mean(D, w, 0.0), [0.3, 0.5])            # geometric
    assert np.allclose(power_mean(D, w, 1e-7), power_mean(D, w, 0.0), atol=1e-5)
    assert np.all(np.abs(power_mean(D, w, -60.0) - D.min(axis=1)) < 0.02)   # -> minimum


def test_one_deficient_metric_loses_to_uniformly_good():
    # Photo 0: excellent on 9 metrics, at the floor on one. Photo 1: uniformly good.
    D = np.array([[1.0] * 9 + [0.05], [0.75] * 10])
    w = np.ones(10)
    assert power_mean(D, w, 0.0)[1] > power_mean(D, w, 0.0)[0]
    p = monte_carlo_topk(D, w, k=1, draws=200, p_range=(-2.0, 0.0), seed=3)
    assert p[1] > p[0]


def test_monte_carlo_reproducible():
    rng = np.random.default_rng(0)
    D = rng.uniform(0.05, 1, size=(50, 4))
    w = np.array([1, 2, 1, 0.5])
    a = monte_carlo_topk(D, w, k=10, draws=100, seed=7)
    b = monte_carlo_topk(D, w, k=10, draws=100, seed=7)
    assert np.array_equal(a, b) and abs(a.sum() - 10) < 1e-9


def test_diverse_pick_skips_near_duplicates():
    base = np.eye(8)[0]
    E = np.stack([base, base * 0.999 + 0.04 * np.eye(8)[1], np.eye(8)[2], np.eye(8)[3]])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    score = np.array([1.0, 0.99, 0.5, 0.4])
    picks = diverse_pick(score, E, 3, lam=1.0, sim_floor=0.75)
    assert picks[0] == 0 and 1 not in picks[:2]
