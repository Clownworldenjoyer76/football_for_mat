
from typing import Tuple

def implied_prob(american: int) -> float:
    a = int(american)
    return (100/(-a)) if a < 0 else (100/(a+100))

def remove_vig_2way(p_over_raw: float, p_under_raw: float) -> Tuple[float, float]:
    k = p_over_raw + p_under_raw
    if k <= 0:
        return 0.5, 0.5
    return p_over_raw / k, p_under_raw / k
