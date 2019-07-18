from itertools import islice
from collections import namedtuple
import numpy as np
from kalman_filter.kalman import simulate_moving_object
import plotnine as p9
import altair as alt
import pandas as pd


def kalman_simulation():
    initial_state = np.array([10, 0.5]).reshape(2, 1)
    state_cov = np.diag((4, 0.01))
    kalman_initial_state = np.array([12, 2.5]).reshape(2, 1)
    mea_cov = np.diag((4, 0.1))
    simulation = simulate_moving_object(
        initial_state,
        state_cov,
        mea_cov,
        kalman_state=kalman_initial_state,
        acceleration=10,
    )
    real_states = []
    kalman_results = []

    for real_state, measured_state, kalman_res in islice(simulation, 50):
        real_states.append(real_state)
        kalman_results.append(kalman_res)

    sim_data = dict(
        # state=real_states,
        # filter_results=kalman_results,
        state_x=[r[0][0] for r in real_states],
        predicted_x=[s.state[0][0] for s in kalman_results],
        state_v=[r[1][0] for r in real_states],
        predicted_v=[s.state[1][0] for s in kalman_results],
        time=list(range(50)),
    )

    return pd.DataFrame(sim_data)


if __name__ == "__main__":
    df = kalman_simulation()
    p = (
            p9.ggplot(data=df, mapping=p9.aes("time", "state_x"))
            + p9.geom_point()
    )
