from typing import NamedTuple
import numpy as np
from numpy.linalg import inv


class KalmanResult(NamedTuple):
    state: np.ndarray
    state_cov: np.ndarray
    gain: np.ndarray


def get_a_priori_estimate(state: np.ndarray, timedelta: float, control: float):
    A = np.array([1, timedelta], [0, 1])
    B = np.array([[0.5 * timedelta ** 2], [timedelta]])
    return A @ state + B @ [control]


def get_a_priori_error_cov(state_cov, timedelta, Q=np.zeros((2, 2))):
    """
    Q: covarianza dovuta al rumore del processo
    """
    A = np.array([[1, timedelta], [0, 1]])
    return A @ state_cov @ A.T + Q


def get_a_posteriori_estimate(a_priori_estimate, measurement, gain):
    return a_priori_estimate + gain @ (measurement - a_priori_estimate)


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
    state,
    state_cov,
    mea_cov,
    control=0,
    timedelta=1,
    pnoise_cov=np.zeros((2, 2)),
):
    measurement = yield
    while True:
        state = get_a_priori_estimate(state, timedelta, control)
        state_cov = get_a_priori_error_cov(state_cov, timedelta, pnoise_cov)
        gain = get_gain(state_cov, mea_cov)
        state = get_a_posteriori_estimate(state, measurement, gain)
        state_cov = get_a_posteriori_error_cov(state_cov, gain)
        measurement = yield KalmanResult(state, state_cov, gain)


def simulation():

    # stato iniziale reale
    real_state = np.array([[0, 0]])

    # rumore del processo
    Q = np.diag([1, 2])

    # rumore dei sensori
    R = np.diag([2, 1])

    kalman_filter(
        state=np.array([1, 1]),
        state_cov=np.diag([2, 2]),
        mea_cov=np.diag([1, 2]),
        control=1,
        timedelta=1,
        pnoise_cov=np.diag([2, 2]),
    )

    for i in range(1, 60):

        w = np.array([1, 2])

        nuovo_stato_pallina = get_a_priori_estimate(real_state, 1, 2) + w

        misurazioni_sensori = np.array([1, 2], [3, 4])

        
