"""
=============================================================================
Stochastic Multi-Elevator Controller: Maximum Reward Deep Planner
=============================================================================

ALGORITHMIC ARCHITECTURE:
-----------------------------------------------------------------------------
1. STOCHASTIC SHORTEST PATHS (Dijkstra)
   During initialization, we calculate the exact minimum expected steps to 
   deliver each person. Edge weights are treated as the expected number of 
   attempts (1 / p_success). This perfectly maps multi-elevator relay routes.

2. DYNAMIC DISCOUNTED-REWARD HEURISTIC
   The heuristic dynamically calculates the cost to deliver all remaining 
   passengers from the *current* elevator positions. It scores the state based 
   on: Reward * (0.98 ^ Expected_Cost). If all passengers can be delivered, 
   it includes the massive global goal_reward. This organically balances 
   greediness with the remaining time budget.

3. ITERATIVE DEEPENING EXPECTIMAX
   The agent dynamically tracks wall-clock time. It searches depth 1, then 2, 
   then 3, calculating exact transition probabilities and expected values, 
   until its time budget for the current turn runs low.
=============================================================================
"""

import heapq
import time
import ext_elev

id = ["000000000"]  # IMPORTANT: Update with your actual Bar-Ilan ID

INF = float('inf')

class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self._t0 = time.process_time()
        
        self.horizon = game.get_max_steps()
        self.initial_state = game.get_initial_state()
        self.goal_reward = game.get_goal_reward()
        
        # Determine total allowed time (95% of theoretical max to be safe)
        self.total_time_limit = (20.0 + 0.5 * self.horizon) * 0.95
        
        self.reachable = game.get_reachable()
        self.capacities = game.get_capacities()
        
        _, persons_t, _ = self.initial_state
        self.elev_ids = sorted(self.reachable.keys())
        self.pers_ids = sorted([p[0] for p in persons_t])
        
        self.eid_to_idx = {eid: i for i, eid in enumerate(self.elev_ids)}
        self.idx_to_eid = {i: eid for i, eid in enumerate(self.elev_ids)}
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.pers_ids)}
        
        all_floors = set()
        for floors in self.reachable.values():
            all_floors.update(floors)
        self.height = max(all_floors) + 1 if all_floors else 0
        
        # Shared floors mapping to prune useless moves
        floor_counts = {}
        for e in self.elev_ids:
            for f in self.reachable[e]:
                floor_counts[f] = floor_counts.get(f, 0) + 1
        self.shared_floors = {f for f, c in floor_counts.items() if c > 1}
        
        self.p_goals = {p: game.get_person_goal(p) for p in self.pers_ids}
        self.p_weights = {p: game.get_person_weight(p) for p in self.pers_ids}
        self.p_probs = {p: max(1e-9, game.get_person_action_prob(p)) for p in self.pers_ids}
        self.e_probs = {e: max(1e-9, game.get_elevator_action_prob(e)) for e in self.elev_ids}
        
        self.p_rewards = {}
        for p in self.pers_ids:
            rews = game.get_person_reward(p)
            self.p_rewards[p] = sum(rews) / len(rews) if rews else 0.0

        self.init_int_state = self._internalize(self.initial_state)

        # Precompute universal stochastic shortest paths
        self._precompute_dijkstra()

    def _internalize(self, state):
        """Converts slow string/tuple states into fast flat integer tuples."""
        elevs, persons, rem = state
        e_floors = [0] * len(self.elev_ids)
        e_loads = [0] * len(self.elev_ids)
        
        for e in elevs:
            idx = self.eid_to_idx[e[0]]
            e_floors[idx] = e[1]
            e_loads[idx] = e[2]
            
        p_locs = [-1] * len(self.pers_ids)
        for p in persons:
            idx = self.pid_to_idx[p[0]]
            loc = p[1]
            if loc[0] == 'floor':
                p_locs[idx] = loc[1]
            else:
                p_locs[idx] = self.height + self.eid_to_idx[loc[1]]
                
        return (tuple(e_floors), tuple(e_loads), tuple(p_locs), rem)

    def _precompute_dijkstra(self):
        """Backwards Dijkstra treating probabilities as Expected Step Costs (1/p)."""
        self.dist = {}
        for pid in self.pers_ids:
            g = self.p_goals[pid]
            w = self.p_weights[pid]
            pp = self.p_probs[pid]
            
            pq = [(0.0, ('floor', g))]
            dist_map = { ('floor', g): 0.0 }
            
            while pq:
                d, node = heapq.heappop(pq)
                if d > dist_map.get(node, INF): continue
                
                if node[0] == 'floor':
                    f = node[1]
                    for eid in self.elev_ids:
                        if f in self.reachable[eid] and w <= self.capacities[eid]:
                            v = ('in', eid, f)
                            c = d + 1.0 / pp
                            if c < dist_map.get(v, INF):
                                dist_map[v] = c
                                heapq.heappush(pq, (c, v))
                else:
                    _, eid, f = node
                    pe = self.e_probs[eid]
                    
                    # Expected cost to back-propagate a MOVE
                    for prev_f in self.reachable[eid]:
                        if prev_f != f:
                            v = ('in', eid, prev_f)
                            c = d + 1.0 / pe
                            if c < dist_map.get(v, INF):
                                dist_map[v] = c
                                heapq.heappush(pq, (c, v))
                                
                    # Expected cost to back-propagate an ENTER
                    v = ('floor', f)
                    c = d + 1.0 / pp
                    if c < dist_map.get(v, INF):
                        dist_map[v] = c
                        heapq.heappush(pq, (c, v))
                        
            self.dist[pid] = dist_map

    def _get_dynamic_cost(self, p_idx, e_floors, p_locs):
        """Exact expected steps to deliver a person from current active positions."""
        loc = p_locs[p_idx]
        if loc == -1: return 0.0  # Already delivered
        
        pid = self.pers_ids[p_idx]
        
        if loc >= self.height:  # Inside an elevator
            e_idx = loc - self.height
            e_id = self.idx_to_eid[e_idx]
            f = e_floors[e_idx]
            return self.dist[pid].get(('in', e_id, f), INF)
            
        else:  # Waiting on a floor
            f = loc
            best_cost = INF
            for e_idx, e_id in enumerate(self.elev_ids):
                if f in self.reachable[e_id] and self.p_weights[pid] <= self.capacities[e_id]:
                    e_f = e_floors[e_idx]
                    # Cost to move empty elevator to user (if not already there)
                    move_cost = 0.0 if e_f == f else (1.0 / self.e_probs[e_id])
                    enter_cost = 1.0 / self.p_probs[pid]
                    deliv_cost = self.dist[pid].get(('in', e_id, f), INF)
                    
                    total = move_cost + enter_cost + deliv_cost
                    if total < best_cost:
                        best_cost = total
            return best_cost

    def _heuristic(self, int_state, t_left):
        """Value function evaluates exponential time-decay reward."""
        e_floors, _, p_locs, rem = int_state
        if t_left <= 0: return 0.0
        
        h_val = 0.0
        max_cost = 0.0
        all_deliverable = True
        active_count = 0
        
        # Decay factor: 0.98 ensures faster deliveries are strictly preferred
        gamma = 0.98 
        
        for p_idx, loc in enumerate(p_locs):
            if loc != -1:
                active_count += 1
                cost = self._get_dynamic_cost(p_idx, e_floors, p_locs)
                
                if cost > t_left or cost == INF:
                    all_deliverable = False
                else:
                    pid = self.pers_ids[p_idx]
                    h_val += self.p_rewards[pid] * (gamma ** cost)
                    if cost > max_cost:
                        max_cost = cost
                        
        # If we can clear the rest of the board within the horizon, claim the goal reward!
        if all_deliverable and rem > 0:
            h_val += self.goal_reward * (gamma ** max_cost)
            
        return h_val

    def _get_legal_actions(self, int_state):
        e_floors, e_loads, p_locs, _ = int_state
        
        # Use integer codes for blazing fast lookahead: 0=RESET, 1=MOVE, 2=ENTER, 3=EXIT
        actions = [(0, 0, 0)] 
        
        # Action Pruning: Only MOVE to floors that contain active targets, goals, or shared transfers
        useful_floors = set(self.shared_floors)
        for p_idx, loc in enumerate(p_locs):
            if loc != -1:
                useful_floors.add(self.p_goals[self.pers_ids[p_idx]])
                if loc < self.height:
                    useful_floors.add(loc)
                    
        for e_idx, eid in enumerate(self.elev_ids):
            cur_f = e_floors[e_idx]
            for f in self.reachable[eid]:
                if f != cur_f and f in useful_floors:
                    actions.append((1, eid, f))
                    
        for p_idx, pid in enumerate(self.pers_ids):
            loc = p_locs[p_idx]
            if loc != -1:
                w = self.p_weights[pid]
                if loc < self.height:  # On floor
                    f = loc
                    for e_idx, eid in enumerate(self.elev_ids):
                        if e_floors[e_idx] == f and e_loads[e_idx] + w <= self.capacities[eid]:
                            actions.append((2, pid, eid))
                else:  # In elevator
                    eid = self.idx_to_eid[loc - self.height]
                    actions.append((3, pid, eid))
                    
        return actions

    def _get_transitions(self, int_state, action):
        e_floors, e_loads, p_locs, rem = int_state
        act_type = action[0]
        
        if act_type == 0:  # RESET
            return [(1.0, self.init_int_state, 0.0)]
            
        elif act_type == 1:  # MOVE
            eid, target_f = action[1], action[2]
            e_idx = self.eid_to_idx[eid]
            p_succ = self.e_probs[eid]
            
            new_e = list(e_floors)
            new_e[e_idx] = target_f
            outcomes = [(p_succ, (tuple(new_e), e_loads, p_locs, rem), 0.0)]
            
            fail_floors = [f for f in self.reachable[eid] if f != target_f]
            if fail_floors:
                p_fail = (1.0 - p_succ) / len(fail_floors)
                for f in fail_floors:
                    new_e_f = list(e_floors)
                    new_e_f[e_idx] = f
                    outcomes.append((p_fail, (tuple(new_e_f), e_loads, p_locs, rem), 0.0))
            return outcomes
            
        elif act_type == 2:  # ENTER
            pid, eid = action[1], action[2]
            p_idx = self.pid_to_idx[pid]
            e_idx = self.eid_to_idx[eid]
            p_succ = self.p_probs[pid]
            
            new_loads = list(e_loads)
            new_loads[e_idx] += self.p_weights[pid]
            new_locs = list(p_locs)
            new_locs[p_idx] = self.height + e_idx
            
            succ_state = (e_floors, tuple(new_loads), tuple(new_locs), rem)
            return [(p_succ, succ_state, 0.0), (1.0 - p_succ, int_state, 0.0)]
            
        elif act_type == 3:  # EXIT
            pid, eid = action[1], action[2]
            p_idx = self.pid_to_idx[pid]
            e_idx = self.eid_to_idx[eid]
            p_succ = self.p_probs[pid]
            cur_f = e_floors[e_idx]
            
            new_loads = list(e_loads)
            new_loads[e_idx] -= self.p_weights[pid]
            new_locs = list(p_locs)
            
            if cur_f == self.p_goals[pid]:
                new_locs[p_idx] = -1
                new_rem = rem - 1
                reward = self.p_rewards[pid]
                
                if new_rem == 0:
                    succ_state = self.init_int_state
                    reward += self.goal_reward
                else:
                    succ_state = (e_floors, tuple(new_loads), tuple(new_locs), new_rem)
                return [(p_succ, succ_state, reward), (1.0 - p_succ, int_state, 0.0)]
            else:
                new_locs[p_idx] = cur_f
                succ_state = (e_floors, tuple(new_loads), tuple(new_locs), rem)
                return [(p_succ, succ_state, 0.0), (1.0 - p_succ, int_state, 0.0)]

    def _expectimax(self, int_state, depth, t_left, memo):
        if depth == 0 or t_left == 0:
            return self._heuristic(int_state, t_left)
            
        key = (int_state, depth)
        if key in memo: return memo[key]
        
        best_val = -float('inf')
        for act in self._get_legal_actions(int_state):
            val = 0.0
            for prob, next_s, r in self._get_transitions(int_state, act):
                if prob > 0:
                    val += prob * (r + self._expectimax(next_s, depth - 1, t_left - 1, memo))
            
            # Tiny penalty prevents arbitrary looping when things are equidistant
            if act[0] == 0: val -= 1e-4 
            if val > best_val: best_val = val
            
        memo[key] = best_val
        return best_val

    def _format_action(self, action):
        if action[0] == 0: return "RESET"
        if action[0] == 1: return f"MOVE{{{action[1]},{action[2]}}}"
        if action[0] == 2: return f"ENTER{{{action[1]},{action[2]}}}"
        if action[0] == 3: return f"EXIT{{{action[1]},{action[2]}}}"

    def choose_next_action(self, state):
        int_state = self._internalize(state)
        curr_steps = self.game.get_current_steps()
        t_left = self.horizon - curr_steps
        if t_left <= 0: return "RESET"
        
        # --- DYNAMIC TIME MANAGEMENT ---
        elapsed = time.process_time() - self._t0
        time_rem = self.total_time_limit - elapsed
        budget = (time_rem / max(1, t_left)) * 1.5  # Allowed process time for this specific turn
        
        best_act = (0, 0, 0)
        
        # Iterative Deepening Expectimax
        for depth in range(1, 6):  # Go deeper until budget runs out (Max depth 5)
            memo = {}
            current_best_val = -float('inf')
            current_best_act = (0, 0, 0)
            
            for act in self._get_legal_actions(int_state):
                val = 0.0
                for prob, next_s, r in self._get_transitions(int_state, act):
                    if prob > 0:
                        val += prob * (r + self._expectimax(next_s, depth - 1, t_left - 1, memo))
                
                # Tie breakers: Prefer progress over resetting, prefer interaction over moving
                if act[0] == 0: val -= 1e-4
                if act[0] in [2, 3]: val += 1e-6
                
                if val > current_best_val:
                    current_best_val = val
                    current_best_act = act
                    
            best_act = current_best_act
            
            # Check clock: If we spent more than 40% of this turn's budget, do not risk diving another layer deeper
            step_elapsed = time.process_time() - self._t0 - elapsed
            if step_elapsed > budget * 0.4:
                break
                
        return self._format_action(best_act)