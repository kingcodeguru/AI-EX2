"""AI assistance disclosure: drafted with advanced AI.

Version 5: Aggressive Iterative Deepening Expectimax with Exception-Based 
Time-Out Guardrails and Root-Level Action Prioritization.
"""

import math
import time
import ext_elev

id = ["000000000"]


class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.horizon = game.get_max_steps()
        self.initial_state = game.get_initial_state()
        self.goal_reward = game.get_goal_reward()
        self.reachable = game.get_reachable()
        self.capacities = game.get_capacities()

        # Pre-process elevator and person configurations
        self.elev_ids = sorted(self.reachable.keys())
        self.pers_ids = sorted([p[0] for p in self.initial_state[1]])
        
        self.eid_to_idx = {eid: i for i, eid in enumerate(self.elev_ids)}
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.pers_ids)}

        all_floors = set()
        for floors in self.reachable.values():
            all_floors.update(floors)
        self.height = max(all_floors) + 1 if all_floors else 0

        self.elev_probs = {eid: game.get_elevator_action_prob(eid) for eid in self.elev_ids}
        self.pers_probs = {pid: game.get_person_action_prob(pid) for pid in self.pers_ids}
        self.pers_goals = {pid: game.get_person_goal(pid) for pid in self.pers_ids}
        self.pers_weights = {pid: game.get_person_weight(pid) for pid in self.pers_ids}
        
        self.pers_exp_rewards = {}
        for pid in self.pers_ids:
            rewards = game.get_person_reward(pid)
            self.pers_exp_rewards[pid] = sum(rewards) / len(rewards) if rewards else 0.0

        # Pre-compute exact isolated step costs for the heuristic fallback
        self._single_agent_costs = self._precompute_exact_single_agent_costs()
        
        self.memo_trans = {}
        self.memo_legal = {}
        self.memo_expectimax = {}  # Persistent cache across steps of the same instance
        self.init_int_state = self.external_to_internal(self.initial_state)

        # Dynamic Time Budgeting Safety Thresholds
        self.total_time_limit = 20.0 + 0.5 * self.horizon
        self.time_spent = 0.0
        self.step_start_time = 0.0
        self.time_budget = 0.0
        self.nodes_expanded = 0

    def external_to_internal(self, state):
        elevators_t, persons_t, _ = state
        e_floors = [0] * len(self.elev_ids)
        e_loads = [0] * len(self.elev_ids)
        for eid, f, w in elevators_t:
            idx = self.eid_to_idx[eid]
            e_floors[idx] = f
            e_loads[idx] = w
            
        p_locs = [-1] * len(self.pers_ids)
        for pid, loc in persons_t:
            idx = self.pid_to_idx[pid]
            if loc[0] == 'floor':
                p_locs[idx] = loc[1]
            elif loc[0] == 'in':
                p_locs[idx] = self.height + self.eid_to_idx[loc[1]]
                
        return (tuple(e_floors), tuple(e_loads), tuple(p_locs))

    def _precompute_exact_single_agent_costs(self):
        costs = {}
        for pid in self.pers_ids:
            goal = self.pers_goals[pid]
            p_prob = self.pers_probs[pid]
            for f in range(self.height):
                best_cost = math.inf
                for eid in self.elev_ids:
                    if f in self.reachable[eid] and goal in self.reachable[eid]:
                        e_prob = self.elev_probs[eid]
                        cost = (1.0 / p_prob) + (1.0 / e_prob) + (1.0 / p_prob)
                        if cost < best_cost:
                            best_cost = cost
                costs[(pid, f)] = best_cost
        return costs

    def get_legal_actions(self, int_state):
        if int_state in self.memo_legal:
            return self.memo_legal[int_state]
            
        e_floors, e_loads, p_locs = int_state
        actions = ["RESET"]
        
        for i, eid in enumerate(self.elev_ids):
            for f in self.reachable[eid]:
                actions.append(f"MOVE{{{eid},{f}}}")
        
        for j, pid in enumerate(self.pers_ids):
            p_loc = p_locs[j]
            if 0 <= p_loc < self.height:
                w_p = self.pers_weights[pid]
                for i, eid in enumerate(self.elev_ids):
                    if e_floors[i] == p_loc and e_loads[i] + w_p <= self.capacities[eid]:
                        actions.append(f"ENTER{{{pid},{eid}}}")
                        
        for j, pid in enumerate(self.pers_ids):
            p_loc = p_locs[j]
            if p_loc >= self.height:
                e_idx = p_loc - self.height
                actions.append(f"EXIT{{{pid},{self.elev_ids[e_idx]}}}")
                
        self.memo_legal[int_state] = actions
        return actions

    def get_transitions(self, int_state, action):
        key = (int_state, action)
        if key in self.memo_trans:
            return self.memo_trans[key]
        
        e_floors, e_loads, p_locs = int_state
        if action == "RESET":
            res = [(self.init_int_state, 1.0, 0.0)]
            self.memo_trans[key] = res
            return res
            
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
            
            new_e = list(e_floors)
            new_e[e_idx] = target_f
            transitions.append(((tuple(new_e), e_loads, p_locs), p_succ, 0.0))
            
            reachable_e = self.reachable[eid]
            options = sorted({cur_f} | (set(reachable_e) - {target_f}))
            p_fail_each = (1.0 - p_succ) / len(options)
            for opt_f in options:
                new_e_f = list(e_floors)
                new_e_f[e_idx] = opt_f
                transitions.append(((tuple(new_e_f), e_loads, p_locs), p_fail_each, 0.0))
                
        elif name == "ENTER":
            pid, eid = arg1, arg2
            p_idx = self.pid_to_idx[pid]
            e_idx = self.eid_to_idx[eid]
            p_succ = self.pers_probs[pid]
            
            new_p = list(p_locs)
            new_p[p_idx] = self.height + e_idx
            new_l = list(e_loads)
            new_l[e_idx] += self.pers_weights[pid]
            transitions.append(((e_floors, tuple(new_l), tuple(new_p)), p_succ, 0.0))
            transitions.append((int_state, 1.0 - p_succ, 0.0))
            
        elif name == "EXIT":
            pid, eid = arg1, arg2
            p_idx = self.pid_to_idx[pid]
            e_idx = self.eid_to_idx[eid]
            cur_f = e_floors[e_idx]
            p_succ = self.pers_probs[pid]
            
            transitions.append((int_state, 1.0 - p_succ, 0.0))
            new_l = list(e_loads)
            new_l[e_idx] -= self.pers_weights[pid]
            
            if cur_f == self.pers_goals[pid]:
                exp_r = self.pers_exp_rewards[pid]
                new_p = list(p_locs)
                new_p[p_idx] = -1
                
                if all(loc == -1 for loc in new_p):
                    transitions.append((self.init_int_state, p_succ, exp_r + self.goal_reward))
                else:
                    transitions.append(((e_floors, tuple(new_l), tuple(new_p)), p_succ, exp_r))
            else:
                new_p = list(p_locs)
                new_p[p_idx] = cur_f
                transitions.append(((e_floors, tuple(new_l), tuple(new_p)), p_succ, 0.0))
        
        merged = {}
        for ns, p, r in transitions:
            if ns not in merged:
                merged[ns] = [0.0, 0.0]
            merged[ns][0] += p
            merged[ns][1] += p * r
            
        res = [(ns, p, r_sum / p if p > 0 else 0.0) for ns, (p, r_sum) in merged.items() if p > 0]
        self.memo_trans[key] = res
        return res

    def _heuristic(self, int_state, t_left):
        e_floors, _, p_locs = int_state
        total_expected = 0.0
        max_cost = 0.0
        
        for j, p_loc in enumerate(p_locs):
            if p_loc == -1:
                continue
            pid = self.pers_ids[j]
            cost = self._single_agent_costs.get((pid, p_loc if p_loc < self.height else e_floors[p_loc - self.height]), 10.0)
            max_cost = max(max_cost, cost)
            total_expected += self.pers_exp_rewards[pid] * (0.95 ** min(cost, t_left))
            
        if any(loc != -1 for loc in p_locs):
            total_expected += self.goal_reward * (0.95 ** min(max_cost, t_left))
        else:
            total_expected = self.goal_reward
            
        return total_expected

    def _expectimax_search(self, int_state, depth, t_left):
        if depth == 0 or t_left == 0:
            return self._heuristic(int_state, t_left)
            
        self.nodes_expanded += 1
        if self.nodes_expanded % 128 == 0:
            if time.process_time() - self.step_start_time > self.time_budget:
                raise TimeoutError()
                
        state_key = (int_state, depth, t_left)
        if state_key in self.memo_expectimax:
            return self.memo_expectimax[state_key]
            
        best_v = -math.inf
        for a in self.get_legal_actions(int_state):
            v = 0.0
            for ns, p, r in self.get_transitions(int_state, a):
                if p > 0:
                    v += p * (r + self._expectimax_search(ns, depth - 1, t_left - 1))
            if v > best_v:
                best_v = v
                
        self.memo_expectimax[state_key] = best_v
        return best_v

    def choose_next_action(self, state):
        self.step_start_time = time.process_time()
        curr_steps = self.game.get_current_steps()
        t_left = self.horizon - curr_steps
        
        if t_left <= 0:
            return "RESET"
            
        int_state = self.external_to_internal(state)
        
        # Aggressive time distribution target: Use up to 93% of overall engine allocation safely
        time_remaining = (self.total_time_limit * 0.93) - self.time_spent
        self.time_budget = max(0.1, (time_remaining / max(1, t_left)) * 4.0)
        
        self.nodes_expanded = 0
        best_action = "RESET"
        
        legal_actions = list(self.get_legal_actions(int_state))
        action_scores = {a: -math.inf for a in legal_actions}
        
        try:
            # Iterative Deepening out to the structural limits of the remaining step horizon
            for depth in range(1, t_left + 1):
                new_action_scores = {}
                for a in legal_actions:
                    v = 0.0
                    for ns, p, r in self.get_transitions(int_state, a):
                        if p > 0:
                            v += p * (r + self._expectimax_search(ns, depth - 1, t_left - 1))
                    new_action_scores[a] = v
                
                # Commit the values of the fully completed depth evaluation layer
                action_scores = new_action_scores
                best_action = max(action_scores, key=action_scores.get)
                
                # Sort root legal choices so the highest yield branches are traversed first in depth+1
                legal_actions.sort(key=lambda a: -action_scores.get(a, -math.inf))
                
        except TimeoutError:
            # Clean exit handling: Fall back immediately onto the best choice from the deepest completed layer
            if action_scores and max(action_scores.values()) > -math.inf:
                best_action = max(action_scores, key=action_scores.get)
                
        self.time_spent += (time.process_time() - self.step_start_time)
        return best_action