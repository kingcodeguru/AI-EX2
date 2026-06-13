# ---------------------------------------------------------------------------
# AI / external-source disclosure (required by the assignment):
#   An LLM (Anthropic Claude) was used to DIAGNOSE under-performance and to
#   suggest two improvements, which I reviewed and validated across all
#   provided instances before submitting:
#     1) End-game handling in choose_next_action (marked "ENDGAME"): when the
#        remaining steps cannot complete the full multi-person episode, deliver
#        the best reward-per-step person (never dropping a boarded passenger).
#     2) A realistic-cost copy of the delivery costs (self.p_costs_real, using
#        expected attempts 1/p instead of 1/p^2) used ONLY for the farm-vs-full
#        strategy decision, so a broken elevator is not over-penalised when
#        deciding whether full delivery is worthwhile. Planning/heuristics are
#        unchanged.
#   All other logic is my own original implementation.
# ---------------------------------------------------------------------------
import time
import heapq
import itertools
import re
import ext_elev

id = ["216700930"]

class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.horizon = game.get_max_steps()
        self.goal_reward = game.get_goal_reward()
        self.initial_state = game.get_initial_state()
        self.e_cap = game.get_capacities()
        self.e_reach = game.get_reachable()
        self.e_probs = {eid: game.get_elevator_action_prob(eid) for eid in self.e_cap}
        self.p_weights = {}; self.p_goals = {}; self.p_probs = {}; self.p_rewards_avg = {}
        _, persons_init_t, _ = self.initial_state
        for (pid, _) in persons_init_t:
            self.p_weights[pid] = self.game.get_person_weight(pid)
            self.p_goals[pid] = self.game.get_person_goal(pid)
            self.p_probs[pid] = self.game.get_person_action_prob(pid)
            rewards = self.game.get_person_reward(pid)
            self.p_rewards_avg[pid] = sum(rewards) / len(rewards)
        self.p_costs = {}
        for (pid, _) in persons_init_t:
            self.p_costs[pid] = self._precompute_costs(pid)
        # Realistic-cost copy (expected attempts = 1/p, NOT 1/p^2) used ONLY for the
        # farm-vs-full strategy decision, so a broken elevator is not over-penalised
        # there. Planning/heuristic keep the 1/p^2 avoidance behaviour untouched.
        self.p_costs_real = {}
        for (pid, _) in persons_init_t:
            self.p_costs_real[pid] = self._precompute_costs(pid, realistic=True)
        self.shared_floors = set()
        all_reaches = list(self.e_reach.values())
        for i in range(len(all_reaches)):
            for j in range(i + 1, len(all_reaches)):
                self.shared_floors.update(set(all_reaches[i]).intersection(all_reaches[j]))
        self.person_cost_factor = 0.845
        self.heuristic_scale = 0.80
        self.carpool_discount = 0.80
        self.total_allowed_time = (20 + 0.5 * self.horizon) * 0.90
        self.start_time = time.time()
        self.farm_threshold = 0.70
        self.rho = 0.1
        self._decide_strategy()
        self.plan = []; self.expected_state = None; self.last_action = None; self.last_state = None

    def _precompute_costs(self, pid, realistic=False):
        goal_f = self.p_goals[pid]; p_prob = self.p_probs[pid]; p_weight = self.p_weights[pid]
        p_act_cost = 1.0 / p_prob if p_prob > 0 else float('inf')
        dist = {}; pq = []; goal_node = ('floor', goal_f); dist[goal_node] = 0.0
        heapq.heappush(pq, (0.0, goal_node))
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            if u[0] == 'floor':
                f = u[1]
                for eid, reach in self.e_reach.items():
                    if f in reach and p_weight <= self.e_cap[eid]:
                        v = ('in', eid, f); nc = d + p_act_cost
                        if nc < dist.get(v, float('inf')): dist[v] = nc; heapq.heappush(pq, (nc, v))
            elif u[0] == 'in':
                eid, f = u[1], u[2]; e_prob = self.e_probs[eid]
                if e_prob > 0:
                    e_act_cost = (1.0 / e_prob) if realistic else (1.0 / (e_prob ** 2))
                else:
                    e_act_cost = float('inf')
                for prev_f in self.e_reach[eid]:
                    if prev_f != f:
                        v = ('in', eid, prev_f); nc = d + e_act_cost
                        if nc < dist.get(v, float('inf')): dist[v] = nc; heapq.heappush(pq, (nc, v))
                v = ('floor', f); nc = d + p_act_cost
                if nc < dist.get(v, float('inf')): dist[v] = nc; heapq.heappush(pq, (nc, v))
        return dist

    def _decide_strategy(self):
        _, persons_t, _ = self.initial_state
        all_pids = [p[0] for p in persons_t]
        total_cost_all = 0
        for pid in all_pids:
            start_f = next(p[1][1] for p in persons_t if p[0] == pid)
            total_cost_all += self.p_costs_real[pid].get(('floor', start_f), float('inf'))
        profit_all = (sum(self.p_rewards_avg.values()) + self.goal_reward) / (total_cost_all + 0.1)
        best_rate = -1; best_subset = set()
        max_subset_size = min(4, len(all_pids))
        for size in range(1, max_subset_size + 1):
            for subset in itertools.combinations(all_pids, size):
                expected_reward = sum(self.p_rewards_avg[pid] for pid in subset)
                max_cost = 0
                for pid in subset:
                    start_f = next(p[1][1] for p in persons_t if p[0] == pid)
                    c = self.p_costs_real[pid].get(('floor', start_f), float('inf'))
                    max_cost = max(max_cost, c)
                if max_cost == float('inf'): continue
                expected_steps = max_cost + (len(subset) - 1) * 2.0
                rate = expected_reward / (expected_steps + 1.0)
                if rate > best_rate: best_rate = rate; best_subset = set(subset); self.rho = rate
        rl_like = self._is_rl_like(); rl_factor = 0.8 if rl_like else 1.0
        if best_rate > profit_all * self.farm_threshold * rl_factor:
            self.target_pids = best_subset; self.farming = True; self.rho = best_rate
        else:
            self.target_pids = set(all_pids); self.farming = False; self.rho = max(best_rate, profit_all)
        self.allowed_elevators = set(self.e_cap.keys())
        needed_floors = set(); total_target_weight = 0
        if self.target_pids:
            for pid in self.target_pids:
                start_f = next(p[1][1] for p in persons_t if p[0] == pid)
                needed_floors.add(start_f); needed_floors.add(self.p_goals[pid]); total_target_weight += self.p_weights[pid]
            best_anchor = None; best_anchor_prob = -1
            for eid, reach in self.e_reach.items():
                if needed_floors.issubset(reach) and self.e_cap[eid] >= total_target_weight:
                    if self.e_probs[eid] > best_anchor_prob: best_anchor_prob = self.e_probs[eid]; best_anchor = eid
            if best_anchor is not None and best_anchor_prob >= 0.90:
                self.allowed_elevators = {best_anchor}

    def _is_rl_like(self):
        vals = list(self.p_rewards_avg.values())
        if not vals: return False
        mean = sum(vals) / len(vals); mx = max(vals)
        return mx > 1.5 * mean

    def _compute_reward_left(self, state):
        _, persons = state; reward = 0.0
        if self.farming:
            all_delivered = all(loc[0] == 'floor' and loc[1] == self.p_goals[pid] for pid, loc in persons if pid in self.target_pids)
            if all_delivered: reward = self.goal_reward
            else:
                for pid, loc in persons:
                    if pid in self.target_pids: reward += self.p_rewards_avg[pid]
                reward += self.goal_reward
        else:
            for pid, loc in persons:
                if pid in self.target_pids: reward += self.p_rewards_avg[pid]
            reward += self.goal_reward
        return reward

    def _get_det_successors(self, state):
        elevs, persons = state; succs = []
        e_dict = {e[0]: (e[1], e[2]) for e in elevs}
        mandatory_exits = []
        for i, (pid, loc) in enumerate(persons):
            if pid not in self.target_pids: continue
            if loc[0] == 'in':
                eid = loc[1]
                if eid not in self.allowed_elevators: continue
                e_f, _ = e_dict[eid]
                if e_f == self.p_goals[pid]:
                    cost = (1.0 / self.p_probs[pid]) * self.person_cost_factor
                    new_elevs = tuple((e[0], e[1], e[2] - self.p_weights[pid]) if e[0] == eid else e for e in elevs)
                    new_persons = tuple((p[0], ('floor', e_f)) if p[0] == pid else p for p in persons)
                    mandatory_exits.append((f"EXIT{{{pid},{eid}}}", (new_elevs, new_persons), cost))
        if mandatory_exits: return mandatory_exits
        for i, (pid, loc) in enumerate(persons):
            if pid not in self.target_pids: continue
            if loc[0] == 'in':
                eid = loc[1]
                if eid not in self.allowed_elevators: continue
                e_f, _ = e_dict[eid]
                cost = (1.0 / self.p_probs[pid]) * self.person_cost_factor
                new_elevs = tuple((e[0], e[1], e[2] - self.p_weights[pid]) if e[0] == eid else e for e in elevs)
                new_persons = tuple((p[0], ('floor', e_f)) if p[0] == pid else p for p in persons)
                succs.append((f"EXIT{{{pid},{eid}}}", (new_elevs, new_persons), cost))
        for i, (pid, loc) in enumerate(persons):
            if pid not in self.target_pids: continue
            if loc[0] == 'floor':
                f = loc[1]
                for eid, (e_f, e_w) in e_dict.items():
                    if eid not in self.allowed_elevators: continue
                    if e_f == f and e_w + self.p_weights[pid] <= self.e_cap[eid]:
                        current_discount = self.carpool_discount if e_w > 0 else 1.0
                        cost = ((1.0 / self.p_probs[pid]) * self.person_cost_factor) * current_discount
                        new_elevs = tuple((e[0], e[1], e[2] + self.p_weights[pid]) if e[0] == eid else e for e in elevs)
                        new_persons = tuple((p[0], ('in', eid)) if p[0] == pid else p for p in persons)
                        succs.append((f"ENTER{{{pid},{eid}}}", (new_elevs, new_persons), cost))
        interesting_floors = set(self.shared_floors)
        for pid, loc in persons:
            if pid not in self.target_pids: continue
            if loc[0] == 'floor': interesting_floors.add(loc[1])
            elif loc[0] == 'in': interesting_floors.add(self.p_goals[pid])
        for eid, (e_f, e_w) in e_dict.items():
            if eid not in self.allowed_elevators: continue
            cost = 1.0 / (self.e_probs[eid] ** 2)
            for target_f in self.e_reach[eid]:
                if target_f != e_f:
                    if target_f not in interesting_floors: continue
                    new_elevs = tuple((e[0], target_f, e[2]) if e[0] == eid else e for e in elevs)
                    succs.append((f"MOVE{{{eid},{target_f}}}", (new_elevs, persons), cost))
        return succs

    def _is_goal(self, state):
        _, persons = state
        for pid, loc in persons:
            if pid in self.target_pids:
                if loc[0] != 'floor' or loc[1] != self.p_goals[pid]: return False
        return True

    def _h(self, state):
        _, persons = state; total = 0.0
        for pid, loc in persons:
            if pid in self.target_pids:
                if loc[0] == 'floor' and loc[1] == self.p_goals[pid]: continue
                if loc[0] == 'in':
                    eid = loc[1]; e_f = next(e[1] for e in state[0] if e[0] == eid)
                    c = self.p_costs[pid].get(('in', eid, e_f), float('inf')); c *= self.carpool_discount
                else:
                    c = self.p_costs[pid].get(('floor', loc[1]), float('inf'))
                total += c
        return total * self.heuristic_scale

    def _min_steps_to_deliver_pid(self, state, pid):
        elevs, persons = state; goal = self.p_goals[pid]
        loc = next((l for p, l in persons if p == pid), None)
        if loc is None: return 0
        e_dict = {eid: (floor, load) for eid, floor, load in elevs}
        if loc[0] == 'in':
            eid = loc[1]; e_f = e_dict[eid][0]
            return 1 if e_f == goal else 2
        floor = loc[1]; best = float('inf')
        for eid, (e_f, _) in e_dict.items():
            if eid not in self.allowed_elevators: continue
            if e_f == floor: best = min(best, 2 if floor == goal else 3)
            elif floor in self.e_reach[eid]: best = min(best, 3 if floor == goal else 4)
        return best

    def _run_a_star(self, start_state, time_limit):
        start_time = time.time(); counter = itertools.count()
        queue = [(self._h(start_state), next(counter), 0, start_state, [])]; visited = set()
        while queue:
            if time.time() - start_time > time_limit: return []
            f, _, g, state, path = heapq.heappop(queue)
            if self._is_goal(state): return path
            if state in visited: continue
            visited.add(state)
            for act, nxt, cost in self._get_det_successors(state):
                if nxt not in visited:
                    new_g = g + cost; new_f = new_g + self._h(nxt)
                    heapq.heappush(queue, (new_f, next(counter), new_g, nxt, path + [(act, nxt)]))
        return []

    def _expectimax_lookahead(self, state, depth, time_limit_remaining):
        if depth <= 0 or time_limit_remaining < 0.001: return None
        _, persons = state; best_act = None; best_val = float('-inf')
        for act, succ_state, cost in self._get_det_successors(state):
            nums = [int(s) for s in re.findall(r'\d+', act)]
            if act.startswith("MOVE"): p_success = self.e_probs.get(nums[0], 0.95) if nums else 0.95
            elif act.startswith("ENTER") or act.startswith("EXIT"): p_success = self.p_probs.get(nums[0], 0.95) if nums else 0.95
            else: p_success = 1.0
            reward_if_success = self._compute_reward_left(succ_state); h_cost = self._h(succ_state)
            val_success = reward_if_success - self.rho * (cost + h_cost)
            val_fail = self._compute_reward_left(state) - self.rho * cost
            if act.startswith("MOVE"): val_fail -= self.rho * 4.0
            exp_value = p_success * val_success + (1.0 - p_success) * val_fail
            if act.startswith("MOVE"):
                eid_str = act[5:].split(',')[0]
                if eid_str.isdigit(): exp_value *= self.e_probs.get(int(eid_str), 1.0)
            if exp_value > best_val: best_val = exp_value; best_act = act
        return best_act

    def choose_next_action(self, full_state):
        elevs, persons, _ = full_state; a_star_state = (elevs, persons)
        elapsed_total = time.time() - self.start_time
        time_left = max(0.1, self.total_allowed_time - elapsed_total)
        steps_left = max(1, self.horizon - self.game.get_current_steps())
        step_timeout = min(6.0, (time_left / steps_left) * 8)
        viable_pids = set()
        for pid in self.target_pids:
            if self._min_steps_to_deliver_pid(a_star_state, pid) <= steps_left: viable_pids.add(pid)
        if not viable_pids: return "RESET"
        original_targets = self.target_pids; self.target_pids = viable_pids
        if self.farming:
            target_delivered = True
            for pid, loc in persons:
                if pid in self.target_pids:
                    if loc[0] != 'floor' or loc[1] != self.p_goals[pid]: target_delivered = False; break
            if target_delivered:
                self.plan = []; self.last_action = "RESET"; self.last_state = a_star_state
                self.target_pids = original_targets; return "RESET"
        if self.last_action is not None:
            if self.expected_state == a_star_state: pass
            elif self.last_state == a_star_state and any(self.last_action.startswith(prefix) for prefix in ["ENTER", "EXIT"]):
                self.target_pids = original_targets; return self.last_action
            else: self.plan = []
        if not self.plan: self.plan = self._run_a_star(a_star_state, step_timeout)
        if (not self.farming) and len(viable_pids) > 1 and self.plan and len(self.plan) > steps_left:
            boarded_pids = {pid for pid, loc in persons if loc[0] == 'in' and pid in viable_pids}
            if boarded_pids: self.target_pids = boarded_pids
            else:
                best_pid, best_rate = None, -1.0
                for pid in viable_pids:
                    st = self._min_steps_to_deliver_pid(a_star_state, pid)
                    if 0 < st <= steps_left:
                        rate = self.p_rewards_avg[pid] / st
                        if rate > best_rate: best_rate, best_pid = rate, pid
                if best_pid is not None: self.target_pids = {best_pid}
            single_plan = self._run_a_star(a_star_state, step_timeout)
            if single_plan: self.plan = single_plan
        if self.plan:
            act, next_state = self.plan.pop(0)
            self.last_action = act; self.last_state = a_star_state; self.expected_state = next_state
            self.target_pids = original_targets; return act
        fallback_act = self._expectimax_lookahead(a_star_state, depth=1, time_limit_remaining=step_timeout)
        self.target_pids = original_targets
        if fallback_act:
            self.last_action = fallback_act; self.last_state = a_star_state; return fallback_act
        return "RESET"