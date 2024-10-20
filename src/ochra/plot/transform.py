import math
from collections.abc import Collection, Sequence
from bisect import bisect_left
from typing import Optional



def histogram(data: Collection[float], ticks: Sequence[float]) -> Sequence[tuple[float, float, int]]:
    counts = [0] * (len(ticks) - 1)
    for x in data:
        if ticks[0] <= x <= ticks[-1]:
            i = bisect_left(ticks, x)
            counts[i - 1] += 1
    return [(ticks[i], ticks[i + 1], count) for i, count in enumerate(counts)]


def gaussian_kde(data: Collection[float], ticks: Sequence[float], bandwidth: Optional[float] = None) -> Sequence[tuple[float, float]]:
    from scipy.stats import gaussian_kde

    kde = gaussian_kde([x for x in data if not math.isnan(x)], bw_method=bandwidth)
    y = kde(ticks)
    return list(zip(ticks, y))
