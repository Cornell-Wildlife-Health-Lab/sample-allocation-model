import logging
from typing import List
import numpy as np
import math
logger = logging.getLogger(__name__)

def posterior_calculate(
    theta0: float,
    U_history: List[float],
    Max_prevalence: float = 0.01,
    num_states: int = 3,
    Speed_spread: float = 0.2
):
    """
    Calculate the final posterior belief distribution over prevalence states.

    Args:
        theta0 (float): Introduction rate (0 <= theta0 <= 1).
        U_history (List[float]): Observed sampling counts (each must be >= 0).
        Max_prevalence (float): Maximum prevalence (ignored here).
        num_states (int): Number of discretized states (ignored here).
        Speed_spread (float): Spread rate (ignored here).

    Returns:
        b (np.ndarray): Final posterior belief over states.

    Raises:
        ValueError: On invalid introduction rate or sampling history
        Exception: Logs the full traceback to log.txt and re-raises.
    """
    # Validate inputs
    if not isinstance(theta0, (int, float)) or math.isnan(theta0) or not (0.0 < theta0 < 1.0):
        raise ValueError(f"Introduction rate must be a number in (0,1), got {theta0!r}")
    if not isinstance(U_history, (list, np.ndarray)):
        raise ValueError("The sampling history must be a list or numpy array of non-negative integers.")
    for idx, u in enumerate(U_history):
        if (not isinstance(u, (int,float)) or u < 0 or math.isnan(u)):
            raise ValueError(f"Historical sampling size (for the past [{idx}]th year) must be integer >= 0, got {u}")

    # Special case: no historical data
    if len(U_history) == 0:
        b = np.zeros(num_states, dtype=float)
        b[0] = 1.0
        return b

    try:
        Prevalence = np.linspace(0, Max_prevalence, num_states)
        P = np.zeros((num_states, num_states))
        P[0, 0], P[0, 1] = 1-theta0, theta0
        for i in range(1, num_states-1):
            P[i, i], P[i, i+1] = 1-Speed_spread, Speed_spread
        P[-1, -1] = 1.0

        # Initialize prior belief at disease‑free state
        b = np.zeros(num_states, dtype=float)
        b[0] = 1.0

        # Update
        for t, u in enumerate(U_history):
            F = (1 - Prevalence) ** u
            b_pred = b @ P
            numer = b_pred * F  # element‑wise
            denom = numer.sum()
            if denom <= 0:
                denom = np.finfo(float).eps
            b = numer / denom
        return b

    except Exception:
        logger.exception("Unhandled error in posterior_calculate")
        raise
