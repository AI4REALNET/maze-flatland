"""File to holds the tests for the tree observation builder."""

from __future__ import annotations

import random

from flatland.core.grid.grid_utils import distance_on_rail
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.prediction_builders.malf_shortest_path_predictor import MalfShortestPathPredictorForRailEnv
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.graph_based_directional import (
    GraphDirectionalObservationConversion,
)
from maze_flatland.test.env_instantation import create_core_env, create_dummy_env_dead_end


def test_equivalence_in_obs_depth_2_and_3():
    """Checks that graph depth in obs builder can be lowered from 3 to 2 with no drawbacks."""
    seed = 1234
    random.seed(seed)
    env_depth_2 = FlatlandEnvironment(
        core_env=create_core_env(5, 30, 30, 2, 0, {1: 1}),
        action_conversion={'train_move': DirectionalAC()},
        observation_conversion={'train_move': GraphDirectionalObservationConversion(True, graph_depth=2)},
    )
    env_depth_3 = FlatlandEnvironment(
        core_env=create_core_env(5, 30, 30, 2, 0, {1: 1}),
        action_conversion={'train_move': DirectionalAC()},
        observation_conversion={'train_move': GraphDirectionalObservationConversion(True, graph_depth=3)},
    )

    for e in (env_depth_2, env_depth_3):
        e.seed(seed)

    obs_depth_2 = env_depth_2.reset()
    obs_depth_3 = env_depth_3.reset()

    def check_obs(obs1, obs2):
        """Asserts that two observations are identical."""
        assert obs1.keys() == obs2.keys()
        for k in obs1:
            assert (obs1[k] == obs2[k]).all(), f'mismatch in the observation with key {k}'

    check_obs(obs_depth_2, obs_depth_3)
    done = False
    while not done:
        a = {'train_move': random.randint(1, 3)}
        obs_depth_2, rew_depth_2, done, _ = env_depth_2.step(a)
        obs_depth_3, rew_depth_3, _, _ = env_depth_3.step(a)
        assert rew_depth_2 == rew_depth_3
        check_obs(obs_depth_2, obs_depth_3)


def test_conflict():
    for seed in [694, 546789, 987]:
        print(f'Seed: {seed}')
        random.seed(seed)
        env = FlatlandEnvironment(
            core_env=create_core_env(10, 30, 30, 2, 0, {1: 1}),
            action_conversion={'train_move': DirectionalAC()},
            observation_conversion={'train_move': GraphDirectionalObservationConversion(True, graph_depth=2)},
        )
        env.seed(seed)
        _ = env.reset()
        done = False
        # init an instance of shortest path instance for the tests.
        short_path_builder = MalfShortestPathPredictorForRailEnv(
            max_depth=20, exclude_off_map_trains=False, consider_departure_delay=True
        )
        short_path_builder.set_env(env.rail_env)
        short_path_builder.reset()
        while not done:
            action = {'train_move': 2}
            _, r, done, _ = env.step(action)
            ms = env.get_maze_state()
            tree_switch_obs = env.observation_conversion.current_obs
            for branch in tree_switch_obs.edges:
                if isinstance(branch, float):
                    continue
                node = branch.node
                idx_conflicting_agent = node.idx_conflicting_agent
                if idx_conflicting_agent == -1:
                    continue
                # make sure cell of conflict is on the other agent the shortest path at the given distance.
                short_path_ca = short_path_builder.get(idx_conflicting_agent)
                cell_from_short_path = tuple(
                    short_path_ca[idx_conflicting_agent][node.dist_other_agent_to_conflict][1:3]
                )
                assert distance_on_rail(node.cell_of_conflict, cell_from_short_path, 'Manhattan') == 0
                # check that if conflicting agent is out of map then it always has an alternative path.
                if ms.trains[idx_conflicting_agent].has_not_yet_departed():
                    assert node.clashing_agent_has_alternative
                    delayed_time_to_target = 1 + (
                        ms.trains[idx_conflicting_agent].target_distance / ms.trains[idx_conflicting_agent].speed
                    )
                    assert (
                        max(
                            delayed_time_to_target - ms.trains[idx_conflicting_agent].time_left_to_scheduled_arrival,
                            0,
                        )
                        == node.ca_expected_delay_for_alternative_path
                    )


def test_reversible_dead_end_success_to_target():
    """Test that the tree observation can correctly
    see the connection to the target through the dead-ends."""
    env_double_dead_ends = create_dummy_env_dead_end(2)
    # replace obs conv
    env_double_dead_ends.observation_conversion_dict['train_move'] = GraphDirectionalObservationConversion(True)
    env_double_dead_ends.seed(1234)
    dict_obs_double_dead_ends = env_double_dead_ends.observation_conversion.convert_to_dict(
        env_double_dead_ends.reset()
    )
    # ensure that double dead ends can successfully lead the agent to the target.
    assert (
        dict_obs_double_dead_ends['F-L--distance'] == 2.0
    ), f'Unexpected distance: {dict_obs_double_dead_ends["F-L--distance"]}'
    assert (
        dict_obs_double_dead_ends['F-L--target_dist'] == 6
    ), f'Unexpected target distance: {dict_obs_double_dead_ends["F-L--target_dist"]}'
    assert dict_obs_double_dead_ends['F-L--deadlock'] == 0


def test_dead_end_not_connected_to_target():
    """Test that the tree observation
    correctly handles the non-reversible dead-end.
    """
    env_single_dead_end = create_dummy_env_dead_end(1)
    # replace obs conv
    env_single_dead_end.observation_conversion_dict['train_move'] = GraphDirectionalObservationConversion(True)
    env_single_dead_end.seed(1234)
    dict_obs_single_dead_end = env_single_dead_end.observation_conversion.convert_to_dict(env_single_dead_end.reset())
    # check that single dead end cannot reach the target when going to dead-end
    assert (
        dict_obs_single_dead_end['F-L--distance'] == 2.0
    ), f'Unexpected distance: {dict_obs_single_dead_end["F-L--distance"]}'
    assert (
        dict_obs_single_dead_end['F-L--target_dist'] == -1
    ), f'Unexpected target distance: {dict_obs_single_dead_end["F-L--target_dist"]}'
    assert dict_obs_single_dead_end['F-L--deadlock'] == 0


def test_no_reversible_dead_end():
    """Test that the tree observation correctly handles the straight cell going outside the boundaries.
    Note that these are not reversible dead-ends but straight rails going out of map.
    """
    env_no_dead_end = create_dummy_env_dead_end(0)
    # replace obs conv
    env_no_dead_end.observation_conversion_dict['train_move'] = GraphDirectionalObservationConversion(True)
    env_no_dead_end.seed(1234)
    dict_obs_no_dead_end = env_no_dead_end.observation_conversion.convert_to_dict(env_no_dead_end.reset())
    # check that single dead end cannot reach the target when going to dead-end
    assert dict_obs_no_dead_end['F-L--distance'] == -1, f'Unexpected distance: {dict_obs_no_dead_end["F-L--distance"]}'
    assert (
        dict_obs_no_dead_end['F-L--target_dist'] == -1
    ), f'Unexpected target distance: {dict_obs_no_dead_end["F-L--target_dist"]}'
    assert dict_obs_no_dead_end['F-L--deadlock'] == 0
