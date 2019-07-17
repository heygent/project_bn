from typing import NamedTuple
import numpy as np


class KalmanResult(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    gain: np.ndarray


def get_estimate(
    estimate: np.ndarray,
    timedelta: float,
    control: float,
    process_noise: np.ndarray = np.zeros((2, 1)),
):
    A = np.array([1, timedelta], [0, 1])
    B = np.array([[0.5 * timedelta ** 2], [timedelta]])
    return A @ estimate + B @ [control] + process_noise


def get_gain(state_cov, mea_cov):
    """
    Calcola il Kalman Gain. La formula completa è P_kp H.T / (H P_kp H.T + R).
    H serve a convertire la forma della matrice in maniera compatibile a
    quella del Kalman Gain. In questo caso non c'è necessità di usarla
    perché sarebbe uguale all'identità.

    Bisogna fare al quadrato mat. covarianza?
    """

    return state_cov / (state_cov + mea_cov)


def get_state_cov(state_cov, timedelta, process_noise=np.zeros((2, 2))):
    A = np.array([1, timedelta], [0, 1])
    return A @ state_cov @ A.T + process_noise


def kalman_filter(
    state, state_cov, measurements, mea_cov, control, process_noise_cov
):
    pass
