from math import isclose

import pytest

from project_bn.map_ask import map_ask
from project_bn.nets import make_sprinkler_plus


@pytest.mark.parametrize(
    "Ms,e,bn,expected_map_value,expected_assignments",
    [
        (
            ["Season", "Sprinkler"],
            dict(GrassWet=True, RoadWet=False),
            make_sprinkler_plus(),
            0.7238186557,
            dict(Season="summer", Sprinkler="true"),
        )
    ],
)
def test_map(Ms, e, bn, expected_map_value, expected_assignments):
    assignments, map_value = map_ask(Ms, e, bn)
    assert isclose(map_value, expected_map_value)
    assert assignments == expected_assignments


@pytest.mark.parametrize(
    "e,bn,expected_mpe_value,expected_assignments",
    [
        (
            dict(GrassWet=True, RoadWet=False),
            make_sprinkler_plus(),
            0.6843176151770903,
            dict(Season="summer", Sprinkler=True, Cloudy=False, Rain="no"),
        )
    ],
)
def test_mpe(e, bn, expected_mpe_value, expected_assignments):
    pytest.fail("Not implemented")
