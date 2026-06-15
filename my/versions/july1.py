"""AI assistance disclosure: drafted with GitHub Copilot (GPT-5.4 mini).

Hybrid controller for the stochastic multi-elevator MDP. It combines a
depth-limited expectimax search over legal actions with a domain-specific
heuristic that values high-reward people, transfer floors, and goal reward.
"""

import math

import ext_elev

id = ["000000000"]


class Controller:
    """Stochastic multi-elevator controller.

    The policy searches a shallow but exact stochastic tree and falls back to a
    heuristic value estimate when the search horizon is exhausted.
    """

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self._max_steps = int(game.get_max_steps())
        self._initial_state = game.get_initial_state()
        self._goal_reward = float(game.get_goal_reward())

        self._capacities = game.get_capacities()
        self._reachable = {
            eid: tuple(sorted(floors))
            for eid, floors in game.get_reachable().items()
        }
        self._reachable_sets = {
            eid: frozenset(floors)
            for eid, floors in self._reachable.items()
        }

        _, persons_t, _ = self._initial_state
        self._person_ids = tuple(sorted(pid for pid, _ in persons_t))
        self._elevator_ids = tuple(sorted(self._capacities))

        self._person_weight = {
            pid: game.get_person_weight(pid) for pid in self._person_ids
        }
        self._person_goal = {
            pid: game.get_person_goal(pid) for pid in self._person_ids
        }
        self._person_prob = {
            pid: float(game.get_person_action_prob(pid)) for pid in self._person_ids
        }
        self._elevator_prob = {
            eid: float(game.get_elevator_action_prob(eid)) for eid in self._elevator_ids
        }
        self._person_mean_reward = {
            pid: (
                sum(game.get_person_reward(pid)) / len(game.get_person_reward(pid))
                if len(game.get_person_reward(pid)) > 0
                else 0.0
            )
            for pid in self._person_ids
        }

        self._state_info_cache = {}
        self._legal_action_cache = {}
        self._transition_cache = {}
        self._value_cache = {}
        self._policy_cache = {}
        self._heuristic_cache = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def choose_next_action(self, state):
        steps_left = self._max_steps - self.game.get_current_steps()
        if steps_left <= 0:
            return "RESET"

        depth = self._choose_depth(state, steps_left)
        self._best_value(state, steps_left, depth)
        return self._policy_cache.get((state, steps_left, depth), "RESET")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _choose_depth(self, state, steps_left):
        _, persons_t, _ = state
        remaining = len(persons_t)

        if steps_left <= 2:
            return steps_left
        if remaining <= 1:
            return min(steps_left, 8)
        if remaining <= 2:
            return min(steps_left, 7)
        if remaining <= 3:
            return min(steps_left, 7)   # was 6
        return min(steps_left, 5)

    def _best_value(self, state, steps_left, depth_left):
        key = (state, steps_left, depth_left)
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached

        if steps_left <= 0:
            value = 0.0
            self._value_cache[key] = value
            self._policy_cache[key] = "RESET"
            return value

        if depth_left <= 0:
            value = self._heuristic(state, steps_left)
            self._value_cache[key] = value
            self._policy_cache[key] = "RESET"
            return value

        best_value = -math.inf
        best_action = "RESET"

        for action in self._legal_actions(state):
            q_value = self._action_value(state, action, steps_left, depth_left)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        self._value_cache[key] = best_value
        self._policy_cache[key] = best_action
        return best_value

    def _action_value(self, state, action, steps_left, depth_left):
        outcomes = self._transition_outcomes(state, action)
        next_steps = steps_left - 1
        next_depth = depth_left - 1
        total = 0.0

        for prob, next_state, immediate_reward in outcomes:
            if prob <= 0.0:
                continue
            continuation = (
                self._best_value(next_state, next_steps, next_depth)
                if next_steps > 0
                else 0.0
            )
            total += prob * (immediate_reward + continuation)

        if action == "RESET":
            total -= self._reset_penalty(state)

        return total

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def _decode_state(self, state):
        cached = self._state_info_cache.get(state)
        if cached is not None:
            return cached

        elevators_t, persons_t, remaining = state
        elevator_floor = {}
        elevator_load = {}
        for eid, floor, load in elevators_t:
            elevator_floor[eid] = floor
            elevator_load[eid] = load

        person_loc = {}
        people_on_floor = {}
        people_in_elevator = {}
        for pid, loc in persons_t:
            person_loc[pid] = loc
            if loc[0] == "floor":
                people_on_floor.setdefault(loc[1], []).append(pid)
            else:
                people_in_elevator.setdefault(loc[1], []).append(pid)

        info = (elevator_floor, elevator_load, person_loc, people_on_floor, people_in_elevator, remaining)
        self._state_info_cache[state] = info
        return info

    def _legal_actions(self, state):
        cached = self._legal_action_cache.get(state)
        if cached is not None:
            return cached

        elevator_floor, elevator_load, _, people_on_floor, people_in_elevator, _ = self._decode_state(state)

        actions = []

        # Exit actions first: if someone can be delivered, that is the most
        # urgent kind of move.
        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            inside = people_in_elevator.get(eid, [])
            goal_first = []
            other = []
            for pid in inside:
                if floor == self._person_goal[pid]:
                    goal_first.append(pid)
                else:
                    other.append(pid)
            goal_first.sort(key=lambda pid: (-self._person_mean_reward[pid], pid))
            other.sort()
            for pid in goal_first + other:
                actions.append(f"EXIT{{{pid},{eid}}}")

        # Enter actions next.
        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            load = elevator_load[eid]
            candidates = []
            for pid in people_on_floor.get(floor, []):
                if load + self._person_weight[pid] <= self._capacities[eid]:
                    candidates.append(pid)
            candidates.sort(key=lambda pid: (-self._person_mean_reward[pid], pid))
            for pid in candidates:
                actions.append(f"ENTER{{{pid},{eid}}}")

        # Move actions are pruned to the most relevant floors per elevator.
        floor_scores = self._floor_scores(state)
        for eid in self._elevator_ids:
            floor = elevator_floor[eid]
            targets = []
            for target in self._reachable[eid]:
                score = floor_scores.get(target, 0.0)
                if target == floor:
                    score += 0.1  # keep self-loops available but low priority
                targets.append((score, target))
            targets.sort(key=lambda item: (-item[0], item[1]))
            keep_count = len(targets) if len(targets) <= 4 else 3
            for _, target in targets[:keep_count]:
                actions.append(f"MOVE{{{eid},{target}}}")

        actions.append("RESET")
        actions = tuple(actions)
        self._legal_action_cache[state] = actions
        return actions

    def _floor_scores(self, state):
        elevators_t, persons_t, _ = state
        score = {}

        total_reward = sum(self._person_mean_reward[pid] for pid, _ in persons_t)

        for eid, _, _ in elevators_t:
            for floor in self._reachable[eid]:
                score[floor] = score.get(floor, 0.0) + 0.15 * total_reward

        for pid, loc in persons_t:
            reward = self._person_mean_reward[pid]
            goal = self._person_goal[pid]
            score[goal] = score.get(goal, 0.0) + 1.50 * reward
            if loc[0] == "floor":
                score[loc[1]] = score.get(loc[1], 0.0) + 2.50 * reward
            else:
                floor = self._decode_state(state)[0][loc[1]]
                score[floor] = score.get(floor, 0.0) + 1.20 * reward

        # Floors shared by multiple elevators are good transfer points.
        for floor in score:
            reach_count = sum(1 for eid in self._elevator_ids if floor in self._reachable_sets[eid])
            score[floor] += 0.40 * reach_count

        return score

    # ------------------------------------------------------------------
    # Transition model
    # ------------------------------------------------------------------
    def _transition_outcomes(self, state, action):
        key = (state, action)
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached

        if action == "RESET":
            outcomes = ((1.0, self._initial_state, 0.0),)
            self._transition_cache[key] = outcomes
            return outcomes

        kind, first, second = self._parse_action(action)
        if kind == "MOVE":
            outcomes = self._move_outcomes(state, first, second)
        elif kind == "ENTER":
            outcomes = self._enter_outcomes(state, first, second)
        elif kind == "EXIT":
            outcomes = self._exit_outcomes(state, first, second)
        else:
            outcomes = ((1.0, state, 0.0),)

        self._transition_cache[key] = outcomes
        return outcomes

    def _move_outcomes(self, state, eid, target_floor):
        elevators_t, persons_t, remaining = state
        success_prob = self._elevator_prob[eid]
        elevator_floor, _, _, _, _, _ = self._decode_state(state)
        current_floor = elevator_floor[eid]

        success_state = self._replace_elevator(state, eid, target_floor)
        if success_prob >= 1.0:
            return ((1.0, success_state, 0.0),)

        failure_floors = tuple(sorted((self._reachable_sets[eid] - {target_floor}) | {current_floor}))
        if not failure_floors:
            return ((1.0, success_state, 0.0),)

        fail_prob = (1.0 - success_prob) / len(failure_floors)
        outcomes = [(success_prob, success_state, 0.0)]
        for floor in failure_floors:
            fail_state = self._replace_elevator(state, eid, floor)
            outcomes.append((fail_prob, fail_state, 0.0))
        return tuple(outcomes)

    def _enter_outcomes(self, state, pid, eid):
        success_prob = self._person_prob[pid]
        success_state = self._enter_success_state(state, pid, eid)
        if success_prob >= 1.0:
            return ((1.0, success_state, 0.0),)
        return ((success_prob, success_state, 0.0), (1.0 - success_prob, state, 0.0))

    def _exit_outcomes(self, state, pid, eid):
        success_prob = self._person_prob[pid]
        success_state, reward = self._exit_success_state(state, pid, eid)
        if success_prob >= 1.0:
            return ((1.0, success_state, reward),)
        return ((success_prob, success_state, reward), (1.0 - success_prob, state, 0.0))

    def _replace_elevator(self, state, eid, new_floor):
        elevators_t, persons_t, remaining = state
        elevators = list(elevators_t)
        for idx, (cur_eid, floor, load) in enumerate(elevators):
            if cur_eid == eid:
                elevators[idx] = (cur_eid, new_floor, load)
                break
        return (tuple(sorted(elevators)), persons_t, remaining)

    def _enter_success_state(self, state, pid, eid):
        elevators_t, persons_t, remaining = state
        elevators = list(elevators_t)
        persons = list(persons_t)

        weight = self._person_weight[pid]
        for idx, (cur_eid, floor, load) in enumerate(elevators):
            if cur_eid == eid:
                elevators[idx] = (cur_eid, floor, load + weight)
                break

        for idx, (cur_pid, loc) in enumerate(persons):
            if cur_pid == pid:
                persons[idx] = (cur_pid, ("in", eid))
                break

        return (tuple(sorted(elevators)), tuple(sorted(persons)), remaining)

    def _exit_success_state(self, state, pid, eid):
        elevators_t, persons_t, remaining = state
        elevators = list(elevators_t)
        persons = list(persons_t)

        current_floor = None
        for idx, (cur_eid, floor, load) in enumerate(elevators):
            if cur_eid == eid:
                current_floor = floor
                elevators[idx] = (cur_eid, floor, load - self._person_weight[pid])
                break

        if current_floor is None:
            raise ValueError(f"Unknown elevator {eid}")

        if current_floor == self._person_goal[pid]:
            next_remaining = remaining - 1
            if next_remaining == 0:
                return self._initial_state, self._goal_reward + self._person_mean_reward[pid]

            next_persons = tuple(sorted((cur_pid, loc) for cur_pid, loc in persons if cur_pid != pid))
            return (tuple(sorted(elevators)), next_persons, next_remaining), self._person_mean_reward[pid]

        for idx, (cur_pid, loc) in enumerate(persons):
            if cur_pid == pid:
                persons[idx] = (cur_pid, ("floor", current_floor))
                break

        return (tuple(sorted(elevators)), tuple(sorted(persons)), remaining), 0.0

    # ------------------------------------------------------------------
    # Heuristic
    # ------------------------------------------------------------------
    def _heuristic(self, state, steps_left):
        key = (state, steps_left)
        cached = self._heuristic_cache.get(key)
        if cached is not None:
            return cached

        _, persons_t, _ = state
        if not persons_t:
            self._heuristic_cache[key] = self._goal_reward
            return self._goal_reward

        elevator_floor, elevator_load, _, _, _, _ = self._decode_state(state)
        total = 0.0
        total_cost = 0.0

        for pid, loc in persons_t:
            cost = self._estimate_person_cost(state, pid, loc, elevator_floor, elevator_load)
            if math.isinf(cost):
                self._heuristic_cache[key] = -1e9
                return -1e9

            reward = self._person_mean_reward[pid]
            total += reward * (0.95 ** min(cost, steps_left))
            total_cost += cost

        total += self._goal_reward * (0.95 ** min(total_cost, steps_left))
        self._heuristic_cache[key] = total
        return total

    def _reset_penalty(self, state):
        _, persons_t, _ = state
        remaining = len(persons_t)
        return 3.0 + 1.5 * remaining

    def _estimate_person_cost(self, state, pid, loc, elevator_floor, elevator_load):
        goal = self._person_goal[pid]
        weight = self._person_weight[pid]
        p_prob = self._person_prob[pid]

        if loc[0] == "floor":
            start_floor = loc[1]
            best = math.inf

            for eid in self._elevator_ids:
                if elevator_load[eid] + weight > self._capacities[eid]:
                    continue
                if start_floor not in self._reachable_sets[eid]:
                    continue

                move_to_pickup = 0.0 if elevator_floor[eid] == start_floor else (1.0 / self._elevator_prob[eid])
                enter_cost = 1.0 / p_prob

                if goal in self._reachable_sets[eid]:
                    move_to_goal = 0.0 if start_floor == goal else (1.0 / self._elevator_prob[eid])
                    cost = move_to_pickup + enter_cost + move_to_goal + enter_cost
                    best = min(best, cost)

                # Transfer via another elevator on a shared floor.
                for eid2 in self._elevator_ids:
                    if eid2 == eid:
                        continue
                    shared = self._reachable_sets[eid] & self._reachable_sets[eid2]
                    if goal not in self._reachable_sets[eid2]:
                        continue
                    if elevator_load[eid2] + weight > self._capacities[eid2]:
                        continue
                    for meet in shared:
                        move_to_meet = 0.0 if meet == start_floor else (1.0 / self._elevator_prob[eid])
                        move_e2_to_meet = 0.0 if elevator_floor[eid2] == meet else (1.0 / self._elevator_prob[eid2])
                        move_e2_to_goal = 0.0 if meet == goal else (1.0 / self._elevator_prob[eid2])
                        cost = (
                            move_to_pickup + enter_cost + move_to_meet + enter_cost
                            + move_e2_to_meet + enter_cost + move_e2_to_goal + enter_cost
                        )
                        best = min(best, cost)

            return best

        # Person is already inside an elevator.
        eid = loc[1]
        current_floor = elevator_floor[eid]
        best = math.inf

        if goal in self._reachable_sets[eid]:
            move_to_goal = 0.0 if current_floor == goal else (1.0 / self._elevator_prob[eid])
            best = min(best, move_to_goal + (1.0 / p_prob))

        # Transfer to another elevator if needed.
        for eid2 in self._elevator_ids:
            if eid2 == eid:
                continue
            if goal not in self._reachable_sets[eid2]:
                continue
            shared = self._reachable_sets[eid] & self._reachable_sets[eid2]
            if elevator_load[eid2] + weight > self._capacities[eid2]:
                continue
            for meet in shared:
                move_to_meet = 0.0 if current_floor == meet else (1.0 / self._elevator_prob[eid])
                move_e2_to_meet = 0.0 if elevator_floor[eid2] == meet else (1.0 / self._elevator_prob[eid2])
                move_e2_to_goal = 0.0 if meet == goal else (1.0 / self._elevator_prob[eid2])
                cost = move_to_meet + (1.0 / p_prob) + move_e2_to_meet + (1.0 / p_prob) + move_e2_to_goal + (1.0 / p_prob)
                best = min(best, cost)

        return best

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_action(action):
        if action == "RESET":
            return ("RESET", None, None)

        kind, rest = action.split("{", 1)
        rest = rest[:-1]
        left, right = rest.split(",")
        return kind, int(left), int(right)