from itertools import islice

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from kalman_filter.kalman import simulate_moving_object


def kalman_simulation():
    initial_state = np.array([10, 0.5]).reshape(2, 1)
    state_cov = np.diag((100, 4))
    kalman_initial_state = np.array([12, 2.5]).reshape(2, 1)
    mea_cov = np.diag((4, 0.1))
    simulation = simulate_moving_object(
        initial_state,
        state_cov,
        mea_cov,
        kalman_state=kalman_initial_state,
        acceleration=1,
    )
    real_states = []
    kalman_results = []

    for real_state, measured_state, kalman_res in islice(simulation, 30):
        real_states.append(real_state)
        kalman_results.append(kalman_res)

    sim_data = dict(
        state=real_states,
        kalman_results=kalman_results,
        state_x=[r[0][0] for r in real_states],
        predicted_x=[s.state[0][0] for s in kalman_results],
        state_v=[r[1][0] for r in real_states],
        predicted_v=[s.state[1][0] for s in kalman_results],
        time=list(range(30)),
    )

    return pd.DataFrame(sim_data)


if __name__ == "__main__":
    df = kalman_simulation()
    x_predictions = np.array([k.state[0][0] for k in df["kalman_results"]])
    x_cov = np.array([k.state_cov[0][0] for k in df["kalman_results"]])
    pl.plot(df["time"], df["state_x"])
    pl.plot(df["time"], x_predictions)
    pl.fill_between(df["time"], x_predictions - x_cov, x_predictions +
                    x_cov, color='green')
    pl.show()
