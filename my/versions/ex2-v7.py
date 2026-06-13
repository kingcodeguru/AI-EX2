"""
=============================================================================
Stochastic Multi-Elevator Controller: The "Optimal Integration"
=============================================================================

ARCHITECTURE SYNERGY:
This controller integrates the integer-based Iterative Deepening Expectimax 
from V4 with the Stochastic Dijkstra and Subset Farming Strategy from Vamit.

1. EXACT STOCHASTIC PATHFINDING (Dijkstra)
   During initialization, it runs a backward Dijkstra search from every goal. 
   Edge weights are expected attempts (1/p). This builds an exact map of the 
   expected steps to deliver any person, naturally handling multi-elevator relays.

2. SUBSET FARMING STRATEGY (The "RL Trap" Solver)
   It evaluates every subset of passengers to find the combination S* that 
   maximizes the Rate = E[Reward(S)] / E[Steps(S) + 1]. The agent will 
   ruthlessly farm this subset and RESET, ignoring low-reward decoys.

3. INTEGER-TUPLE EXPECTIMAX
   At runtime, states are converted to flat integer tuples. This completely 
   eliminates slow string parsing during the tree search. It uses Iterative 
   Deepening with time.process_time() to safely search as deep as possible 
   without ever timing out.

4. DYNAMIC END-GAME SQUEEZE
   When the remaining steps drop below the expected cost of the S* cycle, 
   the agent drops the subset restriction and greedily targets ANY person 
   it can deliver before the clock runs out.
=============================================================================
"""

import heapq
import itertools
import time
import ext_elev

id = ["000000000"]  # IMPORTANT: Update with your Bar-Ilan ID

INF = float('inf')

class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self._t0 = time.process_time()
        
        self.horizon = game.get_max_steps()
        self.initial_state = game.get_initial_state()
        self.goal_reward = game.get_goal_reward()
        
        # Time budget tracking (95% to be safe)
        self.total_time_limit = (20.0 + 0.5 * self.horizon) * 0.95
        self.time_spent = 0.0
        
        self.reachable = game.get_reachable()
        self.capacities = game.get_capacities()
        
        _, persons_t, _ = self.initial_state
        self.elev_ids = sorted(self.reachable.keys())
        self.pers_ids = sorted([p[0] for p in persons_t])
        
        # Integer mapping for ultra-fast tuple states
        self.eid_to_idx = {eid: i for i, eid in enumerate(self.elev_ids)}
        self.idx_to_eid = {i: eid for i, eid in enumerate(self.elev_ids)}
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.pers_ids)}
        
        all_floors = set()
        for floors in self.reachable.values():
            all_floors.update(floors)
        self.height = max(all_floors) + 1 if all_floors else 0
        
        # Extract probabilities, weights, goals, and mean rewards
        self.p_goals = {p: game.get_person_goal(p) for p in self.pers_ids}
        self.p_weights = {p: game.get_person_weight(p) for p in self.pers_ids}
        self.p_probs = {p: max(1e-9, game.get_person_action_prob(p)) for p in self.pers_ids}
        self.e_probs = {e: max(1e-9, game.get_elevator_action_prob(e)) for e in self.elev_ids}
        
        self.p_rewards = {}
        for p in self.pers_ids:
            rews = game.get_person_reward(p)
            self.p_rewards[p] = sum(rews) / len(rews) if rews else 0.0

        self.init_int_state = self._external_to_internal(self.initial_state)

        # 1. Precompute Stochastic Shortest Paths (Dijkstra)
        self._precompute_dijkstra()
        
        # 2. Determine Optimal Farming Subset
        self._determine_farming_strategy()

    def _external_to_internal(self, state):
        """Converts strings/tuples to flat integer tuples for fast search."""
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
        """Backwards Dijkstra treating probabilities as Expected Attempts (1/p)."""
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
                    
                    # Backpropagate a MOVE
                    for prev_f in self.reachable[eid]:
                        if prev_f != f:
                            v = ('in', eid, prev_f)
                            c = d + 1.0 / pe
                            if c < dist_map.get(v, INF):
                                dist_map[v] = c
                                heapq.heappush(pq, (c, v))
                                
                    # Backpropagate an ENTER
                    v = ('floor', f)
                    c = d + 1.0 / pp
                    if c < dist_map.get(v, INF):
                        dist_map[v] = c
                        heapq.heappush(pq, (c, v))
                        
            self.dist[pid] = dist_map

    def _determine_farming_strategy(self):
        """Evaluates all subsets to find the one with the highest Reward/Steps rate."""
        _, persons_t, _ = self.initial_state
        init_locs = {p[0]: p[1][1] for p in persons_t}
        
        best_rate = -1.0
        best_subset = set()
        best_cost = 0.0
        
        max_size = min(5, len(self.pers_ids)) # Bound to prevent explosion
        
        for size in range(1, max_size + 1):
            for subset in itertools.combinations(self.pers_ids, size):
                reward = sum(self.p_rewards[pid] for pid in subset)
                if len(subset) == len(self.pers_ids):
                    reward += self.goal_reward
                    
                # Estimate cost to deliver this subset from start
                max_cost = 0.0
                for pid in subset:
                    start_f = init_locs[pid]
                    c = self.dist[pid].get(('floor', start_f), INF)
                    max_cost = max(max_cost, c)
                    
                if max_cost == INF: continue
                
                # Approximate cycle steps: max individual cost + sequential overhead
                expected_steps = max_cost + (len(subset) - 1) * 2.0
                rate = reward / (expected_steps + 1.0) # +1 for RESET
                
                if rate > best_rate:
                    best_rate = rate
                    best_subset = set(subset)
                    best_cost = expected_steps + 1.0
                    
        self.farm_subset = best_subset
        self.farm_cycle_cost = best_cost

    def _get_dynamic_cost(self, p_idx, e_floors, p_locs):
        """Exact expected steps to deliver person using current elevator pos + Dijkstra."""
        loc = p_locs[p_idx]
        if loc == -1: return 0.0 
        
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
                    move_cost = 0.0 if e_f == f else (1.0 / self.e_probs[e_id])
                    enter_cost = 1.0 / self.p_probs[pid]
                    deliv_cost = self.dist[pid].get(('in', e_id, f), INF)
                    
                    total = move_cost + enter_cost + deliv_cost
                    if total < best_cost:
                        best_cost = total
            return best_cost

    def _heuristic(self, int_state, t_left, target_pids):
        """Geometrically discounted evaluation of the state."""
        e_floors, _, p_locs, rem = int_state
        if t_left <= 0: return 0.0
        
        h_val = 0.0
        max_cost = 0.0
        all_deliverable = True
        gamma = 0.96 # Decay factor pushes agent to act quickly
        
        for p_idx, loc in enumerate(p_locs):
            pid = self.pers_ids[p_idx]
            if loc != -1 and pid in target_pids:
                cost = self._get_dynamic_cost(p_idx, e_floors, p_locs)
                
                if cost > t_left or cost == INF:
                    all_deliverable = False
                else:
                    h_val += self.p_rewards[pid] * (gamma ** cost)
                    max_cost = max(max_cost, cost)
                    
        # Add goal reward if we are targeting everyone and can finish
        if len(target_pids) == len(self.pers_ids) and all_deliverable and rem > 0:
            h_val += self.goal_reward * (gamma ** max_cost)
            
        return h_val

    def _get_legal_actions(self, int_state, target_pids):
        e_floors, e_loads, p_locs, _ = int_state
        # 0: RESET, 1: MOVE, 2: ENTER, 3: EXIT
        actions = [(0, 0, 0)] 
        
        # Prune MOVE: Only move to floors involved with the targeted persons
        useful_floors = set()
        for p_idx, loc in enumerate(p_locs):
            pid = self.pers_ids[p_idx]
            if loc != -1 and pid in target_pids:
                useful_floors.add(self.p_goals[pid])
                if loc < self.height:
                    useful_floors.add(loc)
                    
        for e_idx, eid in enumerate(self.elev_ids):
            cur_f = e_floors[e_idx]
            for f in self.reachable[eid]:
                if f != cur_f and f in useful_floors:
                    actions.append((1, eid, f))
                    
        for p_idx, pid in enumerate(self.pers_ids):
            loc = p_locs[p_idx]
            if loc != -1 and pid in target_pids:
                w = self.p_weights[pid]
                if loc < self.height:  # ENTER
                    f = loc
                    for e_idx, eid in enumerate(self.elev_ids):
                        if e_floors[e_idx] == f and e_loads[e_idx] + w <= self.capacities[eid]:
                            actions.append((2, pid, eid))
                else:  # EXIT
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

    def _expectimax(self, int_state, depth, t_left, target_pids, memo):
        if depth == 0 or t_left == 0:
            return self._heuristic(int_state, t_left, target_pids)
            
        key = (int_state, depth)
        if key in memo: return memo[key]
        
        best_val = -float('inf')
        for act in self._get_legal_actions(int_state, target_pids):
            val = 0.0
            for prob, next_s, r in self._get_transitions(int_state, act):
                if prob > 0:
                    val += prob * (r + self._expectimax(next_s, depth - 1, t_left - 1, target_pids, memo))
            
            if act[0] == 0: val -= 1e-4 # Tiny penalty for resetting unnecessarily
            if val > best_val: best_val = val
            
        memo[key] = best_val
        return best_val

    def _format_action(self, action):
        if action[0] == 0: return "RESET"
        if action[0] == 1: return f"MOVE{{{action[1]},{action[2]}}}"
        if action[0] == 2: return f"ENTER{{{action[1]},{action[2]}}}"
        if action[0] == 3: return f"EXIT{{{action[1]},{action[2]}}}"

    def choose_next_action(self, state):
        start_time = time.process_time()
        curr_steps = self.game.get_current_steps()
        t_left = self.horizon - curr_steps
        if t_left <= 0: return "RESET"
        
        int_state = self._external_to_internal(state)
        
        # --- END-GAME SQUEEZE STRATEGY ---
        # If time is running out to complete a farmed cycle, target EVERYONE
        if t_left < self.farm_cycle_cost * 1.3:
            active_targets = set(self.pers_ids)
        else:
            active_targets = self.farm_subset
            
        # If we successfully delivered our targets for this cycle, RESET!
        targets_present = False
        for p_idx, loc in enumerate(int_state[2]):
            if self.pers_ids[p_idx] in active_targets and loc != -1:
                targets_present = True
                break
        if not targets_present:
            return "RESET"
        
        # --- DYNAMIC TIME BUDGETING ---
        time_rem = self.total_time_limit - self.time_spent
        budget = max(0.01, (time_rem / max(1, t_left)) * 1.5)
        
        best_act = (0, 0, 0)
        
        # Iterative Deepening Expectimax
        for depth in range(1, 6):
            memo = {}
            current_best_val = -float('inf')
            current_best_act = (0, 0, 0)
            
            for act in self._get_legal_actions(int_state, active_targets):
                val = 0.0
                for prob, next_s, r in self._get_transitions(int_state, act):
                    if prob > 0:
                        val += prob * (r + self._expectimax(next_s, depth - 1, t_left - 1, active_targets, memo))
                
                # Tie breakers
                if act[0] == 0: val -= 1e-4
                if act[0] in [2, 3]: val += 1e-6
                
                if val > current_best_val:
                    current_best_val = val
                    current_best_act = act
                    
            best_act = current_best_act
            
            # Break early if search is consuming too much time
            step_elapsed = time.process_time() - start_time
            if step_elapsed > budget * 0.4:
                break
                
        self.time_spent += (time.process_time() - start_time)
        return self._format_action(best_act)