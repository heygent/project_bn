from typing import NamedTuple
import numpy as np
from numpy.linalg import inv


class KalmanResult(NamedTuple):
    state: np.ndarray
    state_cov: np.ndarray
    gain: np.ndarray


def get_a_priori_estimate(
    old_prediction: np.ndarray, deltat: float, control: float
):
    A = np.array([1, deltat], [0, 1])
    B = np.array([[0.5 * deltat ** 2], [deltat]])
    return A @ old_prediction + B @ [control]


def get_a_priori_error_cov(state_cov, timedelta, Q=np.zeros((2, 2))):
    """
    Q: covarianza dovuta al rumore del processo
    """
    A = np.array([[1, timedelta], [0, 1]])
    return A @ state_cov @ A.T + Q


def get_a_posteriori_estimate(a_priori_estimate, measurement, gain):
    return a_priori_estimate + gain @ measurement


def get_a_posteriori_error_cov(previous_cov, gain):
    return (np.identity(2) - gain) @ previous_cov


def get_gain(state_cov, mea_cov):
    """
    Calcola il Kalman Gain. La formula completa è P_kp H.T / (H P_kp H.T + R).
    H serve a convertire la forma della matrice in maniera compatibile a
    quella del Kalman Gain. In questo caso non c'è necessità di usarla
    perché sarebbe uguale all'identità.

    Il denominatore è uguale a S_k su wikipedia
    """
    return state_cov @ inv(state_cov + mea_cov)


def kalman_filter(
    state, state_cov, control, measurements, mea_cov, timedelta, pnoise_cov
):
    for measurement in measurements:
        state = get_a_priori_estimate(state, timedelta, control)
        state_cov = get_a_priori_error_cov(state_cov, timedelta, pnoise_cov)
        gain = get_gain(state_cov, mea_cov)
        state = get_a_posteriori_estimate(state, measurement, gain)
        state_cov = get_a_posteriori_error_cov(state_cov, gain)
        yield KalmanResult(state, state_cov, gain)
