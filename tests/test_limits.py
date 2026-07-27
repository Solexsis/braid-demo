from __future__ import annotations

import asyncio

import pytest

from app.limits import ConcurrencyGate, Overloaded, RateLimiter


def test_rate_limiter_allows_then_blocks():
    limiter = RateLimiter(requests=3, window_s=10.0)
    for i in range(3):
        limiter.check("1.2.3.4", now=i * 0.1)
    with pytest.raises(Overloaded) as exc:
        limiter.check("1.2.3.4", now=0.4)
    assert exc.value.retry_after >= 1


def test_rate_limiter_window_rolls_off():
    limiter = RateLimiter(requests=1, window_s=5.0)
    limiter.check("a", now=0.0)
    with pytest.raises(Overloaded):
        limiter.check("a", now=1.0)
    limiter.check("a", now=6.0)


def test_rate_limiter_is_per_key():
    limiter = RateLimiter(requests=1, window_s=60.0)
    limiter.check("a", now=0.0)
    limiter.check("b", now=0.0)


def test_rate_limiter_disabled():
    limiter = RateLimiter(requests=0)
    assert not limiter.enabled
    for i in range(100):
        limiter.check("a", now=float(i))


def test_gate_serializes_and_tracks_in_flight():
    async def scenario():
        gate = ConcurrencyGate(slots=1, queue_limit=4)
        order = []

        async def work(name, delay):
            async with gate:
                order.append(f"{name}:start")
                assert gate.in_flight == 1
                await asyncio.sleep(delay)
                order.append(f"{name}:end")

        await asyncio.gather(work("a", 0.02), work("b", 0.01))
        assert gate.in_flight == 0
        # Never interleaved.
        assert order in (
            ["a:start", "a:end", "b:start", "b:end"],
            ["b:start", "b:end", "a:start", "a:end"],
        )

    asyncio.run(scenario())


def test_gate_rejects_when_the_queue_is_full():
    async def scenario():
        gate = ConcurrencyGate(slots=1, queue_limit=1)
        release = asyncio.Event()

        async def hold():
            async with gate:
                await release.wait()

        async def wait():
            async with gate:
                pass

        held = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        queued = asyncio.create_task(wait())
        await asyncio.sleep(0.01)
        with pytest.raises(Overloaded, match="queue is full"):
            async with gate:
                pass
        release.set()
        await asyncio.gather(held, queued)

    asyncio.run(scenario())
