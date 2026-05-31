"""
=============================================================================
Stochastic Multi-Elevator Controller: Method & Implementation Details
=============================================================================

METHODOLOGY: 1-Step Lookahead Expectimax
-----------------------------------------------------------------------------
This controller solves the stochastic Multi-Elevator Markov Decision Process (MDP) 
by calculating the exact expected Q-value of every legal action from the current 
state. Since elevators and people can fail their actions, every action leads to 
multiple possible next states. 

For a given state `s` and action `a`, the Q-value is computed as:
    Q(s, a) = Sum_{s'} P(s' | s, a) * [ R(s, a, s') + V(s') ]
Where:
    - P(s' | s, a) is the transition probability (success or failure).
    - R(s, a, s') is the immediate reward (if a person is delivered).
    - V(s') is the heuristic value of the next state.

The action with the highest Q(s, a) is chosen.

HEURISTIC DESIGN: Deterministic Relaxation & Reward Decay
-----------------------------------------------------------------------------
To estimate V(s') rapidly without exhausting the step limit, the controller 
uses a pre-calculated distance heuristic based on a deterministic relaxation 
of the problem.

1. Precomputation (Dijkstra's Algorithm): 
   During initialization, the controller calculates the exact minimum expected 
   actions for a person to reach their goal from any floor or any elevator. 
   Instead of raw steps, it uses the expected cost of an action: 
   Cost = 1 / Probability of Success.

2. State Evaluation V(s'):
   For any given state, the heuristic estimates the expected reward of the 
   remaining people. To enforce speed and implicitly penalize wasted steps, 
   the heuristic applies an exponential decay (gamma = 0.95) to the reward 
   based on the expected distance:
       V(s') = Sum_{p in remaining} Expected_Reward(p) * (0.95 ^ expected_steps)

   This ensures the agent highly values states where people are closer to 
   their destinations, naturally pruning paths that wander or waste time.

PRUNING & EDGE CASES:
-----------------------------------------------------------------------------
- Infinite distances (unreachable states) are heavily penalized (-999999).
- RESET is naturally de-prioritized to prevent the agent from giving up 
  arbitrarily early, but remains an escape hatch if the state is permanently 
  stuck.
- When the goal is reached, the heuristic accounts for the global goal reward 
  plus the recursive value of the freshly reset initial state.
=============================================================================
"""
import ext_elev
import heapq

id = ["000000000"]

class Controller:
    """Stochastic multi-elevator controller.
    
    Uses 1-Step Lookahead Expectimax with a pre-computed distance heuristic 
    to rapidly evaluate Q-values for all legal actions.
    """

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.initial_state = game.get_initial_state()
        self.capacities = game.get_capacities()
        self.reachable = game.get_reachable()
        self.goal_reward = game.get_goal_reward()

        _, persons_t, _ = self.initial_state
        self.all_persons = [p[0] for p in persons_t]
        self.all_elevators = list(self.capacities.keys())

        self.person_goals = {p: game.get_person_goal(p) for p in self.all_persons}
        self.person_weights = {p: game.get_person_weight(p) for p in self.all_persons}

        # Calculate Expected Rewards for delivery
        self.person_rewards = {}
        for p in self.all_persons:
            rewards = game.get_person_reward(p)
            self.person_rewards[p] = sum(rewards) / len(rewards)

        # Probabilities
        self.P_e = {e: game.get_elevator_action_prob(e) for e in self.all_elevators}
        self.P_p = {p: game.get_person_action_prob(p) for p in self.all_persons}

        # Expected Action Costs (1 / Probability of Success)
        self.MOVE_COST = {e: (1.0 / self.P_e[e] if self.P_e[e] > 1e-5 else 1e5) for e in self.all_elevators}
        self.ENTER_COST = {p: (1.0 / self.P_p[p] if self.P_p[p] > 1e-5 else 1e5) for p in self.all_persons}
        self.EXIT_COST = {p: (1.0 / self.P_p[p] if self.P_p[p] > 1e-5 else 1e5) for p in self.all_persons}

        # Precompute exact deterministic shortest paths (in expected actions) to goals
        self.base_dist = {}
        for p in self.all_persons:
            goal = self.person_goals[p]
            ENTER_C = self.ENTER_COST[p]
            EXIT_C = self.EXIT_COST[p]

            dist = {}
            pq = [(0, f'f_{goal}')]
            while pq:
                d, u = heapq.heappop(pq)
                if u in dist: continue
                dist[u] = d

                if u.startswith('f_'):
                    f = int(u[2:])
                    # If person exits e at f, elevator e must move there and person exits
                    for e, r_floors in self.reachable.items():
                        if f in r_floors:
                            v = f'e_{e}'
                            if v not in dist:
                                heapq.heappush(pq, (d + self.MOVE_COST[e] + EXIT_C, v))
                else:
                    e = int(u[2:])
                    # If person enters e at f, cost is simply ENTER
                    for f in self.reachable[e]:
                        v = f'f_{f}'
                        if v not in dist:
                            heapq.heappush(pq, (d + ENTER_C, v))
            self.base_dist[p] = dist

        self.h_cache = {}
        # Pre-cache the heuristic for initial state to avoid recursion on goal reset
        self.initial_h = 0
        self.initial_h = self._heuristic(self.initial_state)

    def _heuristic(self, state):
        """Evaluates state based on remaining expected actions to deliver all people."""
        if state in self.h_cache:
            return self.h_cache[state]

        elevators_t, persons_t, rem = state
        
        # When goal is reached, state resets allowing for further rewards
        if rem == 0:
            return self.goal_reward + self.initial_h

        total_h = 0
        elev_floors = {e[0]: e[1] for e in elevators_t}

        for pid, loc in persons_t:
            R = self.person_rewards[pid]
            ENTER_C = self.ENTER_COST[pid]

            if loc[0] == 'floor':
                f = loc[1]
                best_cost = float('inf')
                for e, r_floors in self.reachable.items():
                    if f in r_floors:
                        cost = ENTER_C + self.base_dist[pid].get(f'e_{e}', float('inf'))
                        if elev_floors[e] != f:
                            cost += self.MOVE_COST[e]
                        if cost < best_cost:
                            best_cost = cost
                expected_actions = best_cost
            else:
                e = loc[1]
                expected_actions = self.base_dist[pid].get(f'e_{e}', float('inf'))
                goal = self.person_goals[pid]
                if elev_floors[e] == goal:
                    expected_actions -= self.MOVE_COST[e]

            # Severely penalize states where a person is trapped
            if expected_actions == float('inf'):
                self.h_cache[state] = -999999
                return -999999

            # Using 0.95 acts as an intrinsic step penalty, driving agent to be fast
            total_h += R * (0.95 ** expected_actions)

        self.h_cache[state] = total_h
        return total_h

    def _get_legal_actions(self, state):
        """Yields all legal actions in the current state."""
        actions = []
        elevators_t, persons_t, _ = state
        
        elev_floors = {}
        elev_loads = {}
        for eid, f, w in elevators_t:
            elev_floors[eid] = f
            elev_loads[eid] = w

        persons_on_floor = {}
        persons_in_elev = {}

        for pid, loc in persons_t:
            if loc[0] == 'floor':
                f = loc[1]
                persons_on_floor.setdefault(f, []).append((pid, self.person_weights[pid]))
            else:
                e = loc[1]
                persons_in_elev.setdefault(e, []).append(pid)

        # MOVE
        for eid, f, _ in elevators_t:
            for target in self.reachable[eid]:
                if target != f:
                    actions.append(f"MOVE{{{eid},{target}}}")

        # ENTER
        for eid, f, w in elevators_t:
            for pid, p_w in persons_on_floor.get(f, []):
                if w + p_w <= self.capacities[eid]:
                    actions.append(f"ENTER{{{pid},{eid}}}")

        # EXIT
        for eid, pids in persons_in_elev.items():
            for pid in pids:
                actions.append(f"EXIT{{{pid},{eid}}}")

        actions.append("RESET")
        return actions

    def _get_outcomes(self, state, action):
        """Simulates action outcomes and returns [(prob, next_state, immediate_reward)]."""
        elevators_t, persons_t, rem = state
        if action == "RESET":
            return [(1.0, self.initial_state, 0.0)]

        op = action[:4]
        
        if op == "MOVE":
            inner = action[5:-1].split(',')
            e, target = int(inner[0]), int(inner[1])
            P_success = self.P_e[e]

            new_elevs_succ = []
            f_old = -1
            for elev in elevators_t:
                if elev[0] == e:
                    new_elevs_succ.append((e, target, elev[2]))
                    f_old = elev[1]
                else:
                    new_elevs_succ.append(elev)
                    
            succ_state = (tuple(new_elevs_succ), persons_t, rem)
            outcomes = [(P_success, succ_state, 0.0)]

            # Failure bounds
            fail_floors = set([f_old])
            for f in self.reachable[e]:
                if f != target:
                    fail_floors.add(f)
            fail_floors = list(fail_floors)
            
            if fail_floors:
                P_fail_each = (1.0 - P_success) / len(fail_floors)
                if P_fail_each > 0:
                    for ff in fail_floors:
                        new_elevs_fail = []
                        for elev in elevators_t:
                            if elev[0] == e:
                                new_elevs_fail.append((e, ff, elev[2]))
                            else:
                                new_elevs_fail.append(elev)
                        fail_state = (tuple(new_elevs_fail), persons_t, rem)
                        outcomes.append((P_fail_each, fail_state, 0.0))

            return outcomes

        elif op == "ENTE":
            inner = action[6:-1].split(',')
            p, e = int(inner[0]), int(inner[1])
            P_success = self.P_p[p]

            new_elevs = []
            for elev in elevators_t:
                if elev[0] == e:
                    new_elevs.append((e, elev[1], elev[2] + self.person_weights[p]))
                else:
                    new_elevs.append(elev)

            new_persons = []
            for per in persons_t:
                if per[0] == p:
                    new_persons.append((p, ('in', e)))
                else:
                    new_persons.append(per)

            succ_state = (tuple(new_elevs), tuple(new_persons), rem)
            outcomes = [(P_success, succ_state, 0.0)]
            if P_success < 1.0:
                outcomes.append((1.0 - P_success, state, 0.0))
            return outcomes

        elif op == "EXIT":
            inner = action[5:-1].split(',')
            p, e = int(inner[0]), int(inner[1])
            P_success = self.P_p[p]

            e_floor = -1
            for elev in elevators_t:
                if elev[0] == e:
                    e_floor = elev[1]
                    break

            is_goal = (e_floor == self.person_goals[p])

            new_elevs = []
            for elev in elevators_t:
                if elev[0] == e:
                    new_elevs.append((e, elev[1], elev[2] - self.person_weights[p]))
                else:
                    new_elevs.append(elev)

            new_persons = []
            for per in persons_t:
                if per[0] == p:
                    if not is_goal:
                        new_persons.append((p, ('floor', e_floor)))
                else:
                    new_persons.append(per)

            reward = self.person_rewards[p] if is_goal else 0.0
            new_rem = rem - 1 if is_goal else rem
            succ_state = (tuple(new_elevs), tuple(new_persons), new_rem)

            if new_rem == 0:
                reward += self.goal_reward
                succ_state = self.initial_state

            outcomes = [(P_success, succ_state, reward)]
            if P_success < 1.0:
                outcomes.append((1.0 - P_success, state, 0.0))
            return outcomes

    def _get_q_value(self, state, action):
        """Calculates expected Q-Value for a given action."""
        outcomes = self._get_outcomes(state, action)
        q = 0
        for prob, next_s, imm_rew in outcomes:
            q += prob * (imm_rew + self._heuristic(next_s))
        return q

    def choose_next_action(self, state):
        """Return one of: 'MOVE{e,f}', 'ENTER{p,e}', 'EXIT{p,e}', 'RESET'"""
        legal_actions = self._get_legal_actions(state)
        best_a = None
        best_q = -float('inf')

        for a in legal_actions:
            q = self._get_q_value(state, a)
            
            # De-prioritize resets to prevent giving up arbitrarily early
            if a == "RESET":
                q -= 1e6

            if q > best_q:
                best_q = q
                best_a = a

        if best_a is None:
            return "RESET"
            
        return best_a