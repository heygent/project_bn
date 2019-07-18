from itertools import islice
import numpy as np
from kalman.kalman import simulate_moving_object


def test_kalman_simulation():
    initial_state = np.array([10, 3]).reshape(2, 1)
    state_cov = np.diag((2, 2))
    kalman_initial_state = np.array([12, 2.5]).reshape(2, 1)
    mea_cov = np.diag((4, 5))
    simulation = simulate_moving_object(
        initial_state,
        state_cov,
        mea_cov,
        kalman_state=kalman_initial_state,
        acceleration=2,
    )

    for real_state, kalman_res in islice(simulation, 50):
        print(real_state)
        print(kalman_res)
