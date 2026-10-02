"""Sliding-window rate limiter, driven by a fake clock."""

from app import ratelimit
from app.ratelimit import RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_blocks_after_the_limit_and_reports_the_wait():
    clock = Clock()
    limiter = RateLimiter(limit=2, window_s=60, clock=clock)
    assert limiter.retry_after("a") == 0
    limiter.hit("a")
    clock.now += 10
    limiter.hit("a")
    assert limiter.retry_after("a") == 50  # until the first hit leaves the window
    assert limiter.retry_after("b") == 0  # keys are independent


def test_window_slides():
    clock = Clock()
    limiter = RateLimiter(limit=2, window_s=60, clock=clock)
    limiter.hit("a")
    clock.now += 30
    limiter.hit("a")
    clock.now += 30  # the first hit is now exactly one window old
    assert limiter.retry_after("a") == 0
    limiter.hit("a")
    assert limiter.retry_after("a") == 30


def test_checking_does_not_consume_quota():
    limiter = RateLimiter(limit=1, window_s=60, clock=Clock())
    for _ in range(5):
        assert limiter.retry_after("a") == 0


def test_expired_keys_are_swept(monkeypatch):
    monkeypatch.setattr(ratelimit, "MAX_TRACKED_KEYS", 3)
    clock = Clock()
    limiter = RateLimiter(limit=1, window_s=60, clock=clock)
    for key in ("a", "b", "c"):
        limiter.hit(key)
    clock.now += 61
    limiter.hit("d")
    assert set(limiter._hits) == {"d"}
