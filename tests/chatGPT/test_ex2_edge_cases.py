import os
from importlib import util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = ROOT / "original"
STUDENT_PATH = Path(
    os.environ.get("EX2_STUDENT_PATH", ROOT / "my" / "versions" / "ex2-v1.py")
)


if str(ORIGINAL_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_DIR))

import ext_elev


def load_controller():
    spec = util.spec_from_file_location("student_ex2_v1", STUDENT_PATH)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_controller(problem):
    student = load_controller()
    api = ext_elev.create_elevators_game(problem)
    return student.Controller(api)


def test_reset_is_preferred_in_a_dead_end_state():
    problem = {
        "seed": 0,
        "height": 1,
        "Elevators": {
            0: (0, (0, 1), 10),
            1: (1, (1,), 10),
        },
        "Persons": {
            10: (1, 1, 0),
        },
        "elevator_chosen_action_prob": {0: 0.0, 1: 1.0},
        "person_chosen_action_prob": {10: 1.0},
        "persons_reward": {10: [100]},
        "goal_reward": 50,
        "horizon": 10,
    }
    controller = make_controller(problem)

    state = (
        ((0, 0, 1), (1, 1, 0)),
        ((10, ("in", 0)),),
        1,
    )

    action = controller.choose_next_action(state)
    assert action == "RESET"


def test_boarding_a_working_elevator_beats_moving_a_broken_one():
    problem = {
        "seed": 0,
        "height": 2,
        "Elevators": {
            0: (0, (0, 1, 2), 10),
            1: (2, (2,), 10),
        },
        "Persons": {
            11: (2, 1, 0),
        },
        "elevator_chosen_action_prob": {0: 0.0, 1: 1.0},
        "person_chosen_action_prob": {11: 1.0},
        "persons_reward": {11: [1]},
        "goal_reward": 50,
        "horizon": 10,
    }
    controller = make_controller(problem)

    state = (
        ((0, 0, 0), (1, 2, 0)),
        ((11, ("floor", 2)),),
        1,
    )

    action = controller.choose_next_action(state)
    assert action == "ENTER{11,1}"


def test_immediate_delivery_should_be_taken_over_broken_elevator_motion():
    problem = {
        "seed": 0,
        "height": 2,
        "Elevators": {
            0: (0, (0, 1, 2), 10),
            1: (2, (2,), 10),
        },
        "Persons": {
            10: (0, 1, 1),
            11: (2, 1, 2),
        },
        "elevator_chosen_action_prob": {0: 0.0, 1: 1.0},
        "person_chosen_action_prob": {10: 1.0, 11: 1.0},
        "persons_reward": {10: [100], 11: [1]},
        "goal_reward": 50,
        "horizon": 10,
    }
    controller = make_controller(problem)

    state = (
        ((0, 0, 1), (1, 2, 0)),
        ((10, ("in", 0)), (11, ("in", 1))),
        2,
    )

    action = controller.choose_next_action(state)
    assert action == "EXIT{11,1}"