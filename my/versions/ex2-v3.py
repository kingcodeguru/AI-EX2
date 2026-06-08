import ext_elev
import numpy as np
import re
import sys

# Increase recursion depth for deep VI trees
sys.setrecursionlimit(10000)

id = ["000000000"]

# Global caches to share computation across runs of the same problem
_GLOBAL_MEMO_V = {}
_GLOBAL_MEMO_PI = {}
_GLOBAL_MEMO_TRANS = {}
_GLOBAL_MEMO_LEGAL = {}


class Controller:
    """Stochastic multi-elevator controller using Value Iteration."""

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.horizon = game.get_max_steps()
        self.initial_state = game.get_initial_state()
        self.goal_reward = game.get_goal_reward()
        self.reachable = game.get_reachable()
        self.capacities = game.get_capacities()

        # Pre-process elevator info
        self.elev_ids = sorted(self.reachable.keys())
        all_floors = set()
        for floors in self.reachable.values():
            all_floors.update(floors)
        self.max_floor = max(all_floors) if all_floors else 0
        self.height = self.max_floor + 1

        # Pre-process person info
        _, init_persons, _ = self.initial_state
        self.pers_ids = sorted([p[0] for p in init_persons])

        self.eid_to_idx = {eid: i for i, eid in enumerate(self.elev_ids)}
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.pers_ids)}

        self.elev_probs = {eid: game.get_elevator_action_prob(eid) for eid in self.elev_ids}
        self.pers_probs = {pid: game.get_person_action_prob(pid) for pid in self.pers_ids}
        self.pers_goals = {pid: game.get_person_goal(pid) for pid in self.pers_ids}
        self.pers_weights = {pid: game.get_person_weight(pid) for pid in self.pers_ids}

        self.pers_exp_rewards = {}
        for pid in self.pers_ids:
            rewards = game.get_person_reward(pid)
            self.pers_exp_rewards[pid] = sum(rewards) / len(rewards) if rewards else 0.0

        # Create a unique key for the problem to use global caches
        self.problem_key = (
            tuple(sorted((eid, tuple(sorted(fs))) for eid, fs in self.reachable.items())),
            tuple(sorted(self.capacities.items())),
            tuple(sorted(self.elev_probs.items())),
            tuple(sorted(self.pers_probs.items())),
            tuple(sorted(self.pers_goals.items())),
            tuple(sorted(self.pers_weights.items())),
            tuple(sorted(self.pers_exp_rewards.items())),
            self.goal_reward,
            self.horizon
        )

        if self.problem_key not in _GLOBAL_MEMO_V:
            _GLOBAL_MEMO_V[self.problem_key] = {}
            _GLOBAL_MEMO_PI[self.problem_key] = {}
            _GLOBAL_MEMO_TRANS[self.problem_key] = {}
            _GLOBAL_MEMO_LEGAL[self.problem_key] = {}

        self.memo_v = _GLOBAL_MEMO_V[self.problem_key]
        self.memo_pi = _GLOBAL_MEMO_PI[self.problem_key]
        self.memo_trans = _GLOBAL_MEMO_TRANS[self.problem_key]
        self.memo_legal = _GLOBAL_MEMO_LEGAL[self.problem_key]

        self.init_int_state = self.external_to_internal(self.initial_state)

    def external_to_internal(self, state):
        """Convert engine state to a compact internal representation.
        Internal state: (tuple_of_elevator_floors, tuple_of_person_locations)
        Person location: floor_idx (0..height-1), or (height + elevator_idx), or -1 (delivered)
        """
        elevators_t, persons_t, _ = state
        
        # Elevator floors
        e_floors = [0] * len(self.elev_ids)
        for eid, f, w in elevators_t:
            e_floors[self.eid_to_idx[eid]] = f
            
        # Person locations
        p_locs = [-1] * len(self.pers_ids)
        for pid, loc in persons_t:
            idx = self.pid_to_idx[pid]
            if loc[0] == 'floor':
                p_locs[idx] = loc[1]
            elif loc[0] == 'in':
                p_locs[idx] = self.height + self.eid_to_idx[loc[1]]
                
        return (tuple(e_floors), tuple(p_locs))

    def get_legal_actions(self, int_state):
        if int_state in self.memo_legal:
            return self.memo_legal[int_state]
            
        e_floors, p_locs = int_state
        actions = ["RESET"]
        
        # MOVE actions
        for i, eid in enumerate(self.elev_ids):
            for f in self.reachable[eid]:
                actions.append(f"MOVE{{{eid},{f}}}")
        
        # Current weights in elevators
        e_weights = [0] * len(self.elev_ids)
        for j, p_loc in enumerate(p_locs):
            if p_loc >= self.height:
                e_weights[p_loc - self.height] += self.pers_weights[self.pers_ids[j]]
                
        # ENTER actions
        for j, pid in enumerate(self.pers_ids):
            p_loc = p_locs[j]
            if 0 <= p_loc < self.height: # person on floor
                w_p = self.pers_weights[pid]
                for i, eid in enumerate(self.elev_ids):
                    if e_floors[i] == p_loc and e_weights[i] + w_p <= self.capacities[eid]:
                        actions.append(f"ENTER{{{pid},{eid}}}")
                        
        # EXIT actions
        for j, pid in enumerate(self.pers_ids):
            p_loc = p_locs[j]
            if p_loc >= self.height: # person in elevator
                e_idx = p_loc - self.height
                actions.append(f"EXIT{{{pid},{self.elev_ids[e_idx]}}}")
                
        self.memo_legal[int_state] = actions
        return actions

    def get_transitions(self, int_state, action):
        key = (int_state, action)
        if key in self.memo_trans:
            return self.memo_trans[key]
        
        e_floors, p_locs = int_state
        
        if action == "RESET":
            res = [(self.init_int_state, 1.0, 0.0)]
            self.memo_trans[key] = res
            return res
            
        # Parse action - use simple string slicing/splitting for speed
        # format is NAME{arg1,arg2}
        idx1 = action.find('{')
        name = action[:idx1]
        args = action[idx1+1:-1].split(',')
        arg1, arg2 = int(args[0]), int(args[1])
        
        transitions = []
        if name == "MOVE":
            eid, target_f = arg1, arg2
            e_idx = self.eid_to_idx[eid]
            cur_f = e_floors[e_idx]
            p_succ = self.elev_probs[eid]
            
            # Success
            new_e = list(e_floors)
            new_e[e_idx] = target_f
            transitions.append(((tuple(new_e), p_locs), p_succ, 0.0))
            
            # Failure
            reachable_e = self.reachable[eid]
            options = sorted({cur_f} | (set(reachable_e) - {target_f}))
            p_fail_each = (1.0 - p_succ) / len(options)
            for opt_f in options:
                new_e_f = list(e_floors)
                new_e_f[e_idx] = opt_f
                transitions.append(((tuple(new_e_f), p_locs), p_fail_each, 0.0))
                
        elif name == "ENTER":
            pid, eid = arg1, arg2
            p_idx = self.pid_to_idx[pid]
            e_idx = self.eid_to_idx[eid]
            p_succ = self.pers_probs[pid]
            
            # Success
            new_p = list(p_locs)
            new_p[p_idx] = self.height + e_idx
            transitions.append(((e_floors, tuple(new_p)), p_succ, 0.0))
            
            # Failure
            transitions.append((int_state, 1.0 - p_succ, 0.0))
            
        elif name == "EXIT":
            pid, eid = arg1, arg2
            p_idx = self.pid_to_idx[pid]
            e_idx = self.eid_to_idx[eid]
            cur_f = e_floors[e_idx]
            p_succ = self.pers_probs[pid]
            
            # Failure
            transitions.append((int_state, 1.0 - p_succ, 0.0))
            
            # Success
            if cur_f == self.pers_goals[pid]:
                # Delivered
                exp_r = self.pers_exp_rewards[pid]
                new_p = list(p_locs)
                new_p[p_idx] = -1
                
                num_rem = sum(1 for loc in new_p if loc != -1)
                if num_rem == 0:
                    # Global goal reached
                    transitions.append((self.init_int_state, p_succ, exp_r + self.goal_reward))
                else:
                    transitions.append(((e_floors, tuple(new_p)), p_succ, exp_r))
            else:
                # Regular exit (to floor)
                new_p = list(p_locs)
                new_p[p_idx] = cur_f
                transitions.append(((e_floors, tuple(new_p)), p_succ, 0.0))
        
        # Merge identical next states to reduce branching
        merged = {}
        for ns, p, r in transitions:
            if ns not in merged:
                merged[ns] = [0.0, 0.0]
            merged[ns][0] += p
            merged[ns][1] += p * r
            
        res = [(ns, p, r_sum / p if p > 0 else 0.0) for ns, (p, r_sum) in merged.items() if p > 0]
        self.memo_trans[key] = res
        return res

    def get_v(self, int_state, t):
        if t == 0:
            return 0.0
        state_t = (int_state, t)
        if state_t in self.memo_v:
            return self.memo_v[state_t]
        
        best_v = -1e12
        best_a = "RESET"
        
        for a in self.get_legal_actions(int_state):
            v = 0.0
            for ns, p, r in self.get_transitions(int_state, a):
                v += p * (r + self.get_v(ns, t - 1))
            
            if v > best_v:
                best_v = v
                best_a = a
                
        self.memo_v[state_t] = best_v
        self.memo_pi[state_t] = best_a
        return best_v

    def choose_next_action(self, state):
        int_state = self.external_to_internal(state)
        curr_steps = self.game.get_current_steps()
        t = self.horizon - curr_steps
        if t <= 0:
            return "RESET"
        
        # Run VI starting from current state
        self.get_v(int_state, t)
        
        return self.memo_pi.get((int_state, t), "RESET")
