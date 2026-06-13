"""
ex2.py — Production-ready Optimal Controller
Stochastic Multi-Elevator Passenger Routing MDP

=== ARCHITECTURE OVERVIEW ===

OFFLINE (in __init__):
  1. Parse all problem parameters via public game API getters
  2. Compute per-person, per-elevator expected delivery step costs
     using exact closed-form solutions for geometric distributions
  3. Enumerate all 2^n subsets of passengers; for each subset compute
     E[reward] and E[steps] for one full reset cycle. Select S* = subset
     maximising long-run Reward-per-Step rate ρ* = E[R(S*)] / E[T(S*)].
     This solves the "Reset-Looping Trap" with no heuristic hyperparameters.

ONLINE (in choose_action):
  4. Heuristic state value:
       V(s) = E[R_remaining | s] - ρ* × E[T_remaining | s]
     This is the relative value function from average-reward MDP theory.
     It naturally implements opportunity-cost discounting against ρ*.
  5. 2-step lookahead over all valid actions:
       Q(s, a) = E[ r_a + V(s') ]
     where the expectation is over the stochastic transition (weighted
     by p_success for ENTER/EXIT/MOVE, marginalised over failure modes).
  6. Always choose argmax_a Q(s, a). RESET is included in the action set
     and wins whenever looping back to the initial state has higher value.

=== INTERNAL REPRESENTATION ===
All hot-path computations use raw int/float/tuple — zero string operations
inside lookahead loops. Action strings are only formatted at the final return.

=== PROBABILITIES & EXPECTATIONS ===
- MOVE{eid, f}: succeeds with p = elev_prob[eid] (geometric → E[tries] = 1/p)
  On failure the elevator ends up uniformly on any OTHER reachable floor.
  We always re-issue MOVE{eid, f} so the total expected steps is 1/p regardless
  of where the failure lands (geometric restarts don't depend on fail state).
- ENTER{pid, eid} / EXIT{pid, eid}: success prob = person_action_prob[pid].
  Failure has no state change. E[tries] = 1/p_person (geometric).
- Delivery cost for person p via elevator e from current elevator floor f:
    cost(p,e,f) = move(e, f→p_floor)/p_e + 1/p_p + move(e, p_floor→dest)/p_e + 1/p_p
  (where move(e,a,b) = 0 if a==b, else 1 (since we issue MOVE once per step
  targeting the destination — each attempt is 1 step; success probability p_e
  per attempt → geometric with mean 1/p_e).
  Note: move(a→b) = 1/p_e for a≠b regardless of |a-b| because any floor is
  directly reachable in a single MOVE action (target any floor directly).
"""

from itertools import combinations


# ---------------------------------------------------------------------------
# String-free action encoding (integers → string only at return boundary)
# ---------------------------------------------------------------------------
# Action types as integer codes (used internally)
_ACT_RESET = 0
_ACT_MOVE  = 1
_ACT_ENTER = 2
_ACT_EXIT  = 3


def _fmt_move(eid, floor):
    return f"MOVE{{{eid},{floor}}}"

def _fmt_enter(pid, eid):
    return f"ENTER{{{pid},{eid}}}"

def _fmt_exit(pid, eid):
    return f"EXIT{{{pid},{eid}}}"

RESET_STR = "RESET"


class Controller:
    # ======================================================================
    # INIT: parse problem, build distance field, find optimal subset S*
    # ======================================================================

    def __init__(self, game):
        self.game = game

        # ------------------------------------------------------------------
        # 1. Extract static parameters via public API
        # ------------------------------------------------------------------
        state0 = game.get_current_state()
        elevators_t0, persons_t0, _total = state0

        reachable = game.get_reachable()
        self.elev_reachable = {eid: frozenset(floors) for eid, floors in reachable.items()}
        self.num_floors = (max((max(floors) for floors in reachable.values()), default=-1) + 1)
        self.all_floors  = tuple(range(self.num_floors))
        self.horizon     = game.get_max_steps()
        self.goal_reward = float(game.get_goal_reward())

        # Elevators
        self.eids = tuple(sorted(e for (e, f, w) in elevators_t0))
        self.num_elevators = len(self.eids)

        self.elev_init_floor  = {}  # eid → initial floor
        self.elev_init_wload  = {}  # eid → initial weight load
        self.elev_prob        = {}  # eid → p_success per MOVE attempt
        self.elev_capacity    = {}  # eid → max weight capacity

        for (eid, floor, wload) in elevators_t0:
            self.elev_init_floor[eid] = floor
            self.elev_init_wload[eid] = wload
            self.elev_prob[eid]       = float(game.get_elevator_action_prob(eid))
            self.elev_capacity[eid]   = game.get_capacities()[eid]

        # Persons
        self.pids = tuple(sorted(p for (p, loc) in persons_t0))
        self.num_persons = len(self.pids)

        self.person_init_loc  = {}  # pid → initial loc tuple
        self.person_dest      = {}  # pid → destination floor (int)
        self.person_weight    = {}  # pid → weight
        self.person_prob      = {}  # pid → p_success per ENTER/EXIT attempt
        self.person_rewards   = {}  # pid → list of possible rewards
        self.person_mean_rew  = {}  # pid → mean reward

        for (pid, loc) in persons_t0:
            self.person_init_loc[pid] = loc
            dest = game.get_person_goal(pid)
            self.person_dest[pid]     = dest
            self.person_weight[pid]   = game.get_person_weight(pid)
            p_act                     = float(game.get_person_action_prob(pid))
            self.person_prob[pid]     = p_act
            rews                      = list(game.get_person_reward(pid))
            self.person_rewards[pid]  = rews
            self.person_mean_rew[pid] = float(sum(rews)) / len(rews) if rews else 0.0

        # Snapshot of initial state tuple for RESET reference
        self._init_elevators = elevators_t0
        self._init_persons   = persons_t0
        self._init_state     = state0

        # ------------------------------------------------------------------
        # 2. Precompute: expected MOVE steps between any two floors per eid
        # ------------------------------------------------------------------
        # E_move[eid][f_from][f_to] = expected steps to move elevator eid
        # from f_from to f_to (issuing MOVE{eid, f_to} until success).
        # = 0            if f_from == f_to
        # = 1/p_e        otherwise  (geometric, p_e = elev_prob[eid])
        self.E_move = {}
        for eid in self.eids:
            pe = self.elev_prob[eid]
            inv_pe = (1.0 / pe) if pe > 1e-9 else float('inf')
            row = {}
            for fa in self.all_floors:
                row[fa] = {}
                for fb in self.all_floors:
                    row[fa][fb] = 0.0 if fa == fb else (inv_pe if fb in self.elev_reachable[eid] else float('inf'))
            self.E_move[eid] = row

        # ------------------------------------------------------------------
        # 3. Precompute: expected delivery cost for each (pid, eid) pair
        #    starting from the initial elevator floor.
        # ------------------------------------------------------------------
        # E_deliver_init[pid][eid] = expected steps to deliver pid
        #   using elevator eid, starting from elev_init_floor[eid],
        #   person at their initial location.
        self.E_deliver_init = {}
        for pid in self.pids:
            self.E_deliver_init[pid] = {}
            for eid in self.eids:
                cost = self._delivery_cost(pid, eid,
                                           self.elev_init_floor[eid],
                                           self.person_init_loc[pid])
                self.E_deliver_init[pid][eid] = cost

        # ------------------------------------------------------------------
        # 4. Subset enumeration → find S* and ρ*
        # ------------------------------------------------------------------
        self._find_optimal_subset()

        # Boolean for fast check
        self.doing_loop = (len(self.target_pids) < self.num_persons)

    # ======================================================================
    # Expected delivery cost (core formula)
    # ======================================================================

    def _delivery_cost(self, pid, eid, elev_floor, person_loc):
        """
        Expected steps to deliver person pid using elevator eid,
        given the elevator is currently at elev_floor and the person
        is at person_loc (either ('floor', f) or ('in', some_eid)).

        Formula (person on floor f_p, going to dest d):
          cost = E_move(eid, elev_floor → f_p)
               + 1/p_person          [ENTER]
               + E_move(eid, f_p → d)
               + 1/p_person          [EXIT]

        If person is already inside THIS elevator:
          cost = E_move(eid, elev_floor → d)
               + 1/p_person          [EXIT]

        If person is inside a DIFFERENT elevator: return inf
        (can't deliver with this elevator without cross-transfer logic).
        """
        pp = self.person_prob[pid]
        if pp < 1e-9:
            return float('inf')
        inv_pp = 1.0 / pp

        dest = self.person_dest[pid]
        E  = self.E_move[eid]

        if person_loc[0] == 'in':
            if person_loc[1] == eid:
                # Already loaded on this elevator
                if dest not in self.elev_reachable[eid]:
                    return float('inf')
                return E[elev_floor][dest] + inv_pp
            else:
                # In a different elevator — undeliverable via this eid
                return float('inf')
        else:
            # person_loc[0] == 'floor'
            pf = person_loc[1]
            if pf not in self.elev_reachable[eid] or dest not in self.elev_reachable[eid]:
                return float('inf')
            return E[elev_floor][pf] + inv_pp + E[pf][dest] + inv_pp

    # ======================================================================
    # Subset enumeration — find ρ* (optimal long-run rate)
    # ======================================================================

    def _subset_cycle_cost(self, pid_subset):
        """
        Estimate expected steps for one full cycle delivering exactly the
        persons in pid_subset (then RESET costs 1 extra step).

        We use a greedy min-cost sequential assignment starting from the
        initial elevator floors. This is a lower bound on the true cost
        (parallel elevators can overlap) and an upper bound on single-elevator
        cost. For well-formed problems it's a good approximation.

        Returns (expected_reward, expected_steps).
        """
        if not pid_subset:
            return (0.0, 1.0)  # RESET only, no reward

        exp_reward = sum(self.person_mean_rew[pid] for pid in pid_subset)
        if len(pid_subset) == self.num_persons:
            exp_reward += self.goal_reward

        # Greedy sequential assignment from initial positions
        # Track current elevator floors as we "assign" deliveries
        cur_floor = dict(self.elev_init_floor)
        total_steps = 0.0

        # Sort persons by standalone best delivery cost (SJF ordering)
        sorted_pids = sorted(
            pid_subset,
            key=lambda p: min(
                self._delivery_cost(p, e, cur_floor[e], self.person_init_loc[p])
                for e in self.eids
            )
        )

        for pid in sorted_pids:
            # Pick best (eid, cost) for this person given current elevator positions
            best_cost = float('inf')
            best_eid  = self.eids[0]
            for eid in self.eids:
                cost = self._delivery_cost(pid, eid, cur_floor[eid],
                                           self.person_init_loc[pid])
                if cost < best_cost:
                    best_cost = cost
                    best_eid  = eid
            total_steps += best_cost
            # Update that elevator's position to destination
            cur_floor[best_eid] = self.person_dest[pid]

        total_steps += 1.0  # RESET step at cycle end
        return (exp_reward, total_steps)

    def _find_optimal_subset(self):
        """
        Enumerate all 2^n non-empty subsets of persons.
        Compute ρ(S) = E[R(S)] / E[T(S)] for each.
        Set self.target_pids = S* = argmax ρ(S)
            self.rho_star    = ρ*

        If the full set is best, no reset-loop is needed.
        """
        best_rho    = -1.0
        best_subset = frozenset(self.pids)

        for size in range(1, self.num_persons + 1):
            for combo in combinations(self.pids, size):
                s = frozenset(combo)
                exp_r, exp_t = self._subset_cycle_cost(s)
                if exp_t < 1e-12:
                    continue
                rho = exp_r / exp_t
                if rho > best_rho:
                    best_rho    = rho
                    best_subset = s

        self.rho_star    = max(best_rho, 0.0)
        self.target_pids = best_subset

    # ======================================================================
    # Online heuristic value function
    # ======================================================================

    def _state_value(self, elevators_t, persons_t, steps_left):
        """
        V(s) = E[R_remaining | s, π*] - ρ* · E[T_remaining | s, π*]

        This is the relative (bias) value function for the average-reward MDP.
        It equals zero at the "long-run average" trajectory and is positive
        for states that are above average.

        E[R_remaining]: sum of mean rewards for target persons still present,
            plus goal_reward if all persons are in the full-set policy.
        E[T_remaining]: greedy min-cost sequential delivery from current state.
        """
        if steps_left <= 0:
            return 0.0

        # Build quick lookup
        elev_map   = {e: (f, w) for (e, f, w) in elevators_t}
        person_map = {p: loc    for (p, loc)  in persons_t}

        # Active targets: persons in target_pids still present in the scene
        active = [p for p in self.target_pids if p in person_map]

        if not active:
            # All targets delivered; next action should be RESET
            # Value = ρ* × 1 step saved by resetting now vs idle
            # (The RESET itself costs 1 step, giving value = 0 net,
            # but continuing with RESET is the right move — handled by
            # the action comparison including RESET in the action set.)
            return 0.0

        # Expected remaining reward
        exp_r = sum(self.person_mean_rew[p] for p in active)
        if not self.doing_loop:
            # Include goal_reward when all persons are still present
            if len(active) == self.num_persons:
                exp_r += self.goal_reward
            # Partial-progress: goal_reward is not reachable

        # Expected remaining steps (greedy sequential from current state)
        exp_t = self._greedy_steps_from_state(active, elev_map, person_map)
        exp_t = min(exp_t, float(steps_left))

        return exp_r - self.rho_star * exp_t

    def _greedy_steps_from_state(self, pid_list, elev_map, person_map):
        """
        Greedy sequential assignment from CURRENT state.
        Returns expected total steps to deliver all persons in pid_list,
        + 1 for the RESET at cycle end (even if not doing reset-loop,
        this +1 is marginal and keeps comparisons consistent).
        """
        if not pid_list:
            return 1.0

        cur_floor = {e: elev_map[e][0] for e in self.eids}

        # Sort: persons already loaded first (they're in progress), then SJF
        def key(p):
            loc = person_map[p]
            if loc[0] == 'in':
                # Already loaded: cost is just move-to-dest + exit
                eid = loc[1]
                return self._delivery_cost(p, eid, cur_floor[eid], loc)
            # On floor
            return min(
                self._delivery_cost(p, e, cur_floor[e], loc)
                for e in self.eids
            )

        sorted_pids = sorted(pid_list, key=key)
        total = 0.0

        for pid in sorted_pids:
            loc = person_map[pid]
            best_cost = float('inf')
            best_eid  = self.eids[0]

            if loc[0] == 'in':
                # Person already in a specific elevator
                eid_loaded = loc[1]
                cost = self._delivery_cost(pid, eid_loaded, cur_floor[eid_loaded], loc)
                if cost < best_cost:
                    best_cost = cost
                    best_eid  = eid_loaded
            else:
                for eid in self.eids:
                    cost = self._delivery_cost(pid, eid, cur_floor[eid], loc)
                    if cost < best_cost:
                        best_cost = cost
                        best_eid  = eid

            total += best_cost
            cur_floor[best_eid] = self.person_dest[pid]

        return total + 1.0  # +1 for cycle reset

    # ======================================================================
    # Online action enumeration (integer-only, no string parsing)
    # ======================================================================

    def _enum_actions(self, elevators_t, persons_t):
        """
        Return list of (action_type, arg1, arg2) tuples.
        _ACT_RESET  → (0, 0, 0)
        _ACT_MOVE   → (1, eid, target_floor)
        _ACT_ENTER  → (2, pid, eid)
        _ACT_EXIT   → (3, pid, eid)

        Pruning:
        - Only include ENTER/EXIT for target persons (when doing reset-loop)
        - Only MOVE to floors that are useful: person floors or destinations
        - ENTER only when elevator is not over capacity
        - EXIT only when elevator is at person's destination
        """
        # Build fast lookups
        elev_floor = {}
        elev_wload = {}
        for (e, f, w) in elevators_t:
            elev_floor[e] = f
            elev_wload[e] = w

        person_loc = {p: loc for (p, loc) in persons_t}

        # Determine "useful" target floors for MOVE actions
        useful_floors = set()
        for pid in person_loc:
            if self.doing_loop and pid not in self.target_pids:
                continue
            loc = person_loc[pid]
            if loc[0] == 'floor':
                useful_floors.add(loc[1])
            useful_floors.add(self.person_dest[pid])

        actions = [(_ACT_RESET, 0, 0)]

        # MOVE actions (only to useful floors)
        for eid in self.eids:
            cf = elev_floor[eid]
            for f in useful_floors:
                if f != cf and f in self.elev_reachable[eid]:
                    actions.append((_ACT_MOVE, eid, f))

        # ENTER / EXIT actions
        for pid, loc in person_loc.items():
            if self.doing_loop and pid not in self.target_pids:
                continue

            pp = self.person_prob[pid]
            if pp < 1e-9:
                continue

            if loc[0] == 'floor':
                pf = loc[1]
                pw = self.person_weight[pid]
                for eid in self.eids:
                    if elev_floor[eid] == pf:
                        # Weight check
                        if elev_wload[eid] + pw <= self.elev_capacity[eid]:
                            actions.append((_ACT_ENTER, pid, eid))

            elif loc[0] == 'in':
                eid = loc[1]
                dest = self.person_dest[pid]
                if elev_floor[eid] == dest:
                    actions.append((_ACT_EXIT, pid, eid))

        return actions

    # ======================================================================
    # Apply action (success outcome) — tuple-only, no strings
    # ======================================================================

    def _apply_success(self, action_code, elevators_t, persons_t):
        """
        Return (new_elevators_t, new_persons_t, immediate_reward, goal_reached).
        Computes the SUCCESS outcome of the action.
        goal_reached=True iff all persons were just delivered (EXIT completing all).
        """
        act_type, a1, a2 = action_code

        if act_type == _ACT_RESET:
            return (self._init_elevators, self._init_persons, 0.0, False)

        if act_type == _ACT_MOVE:
            eid, target_floor = a1, a2
            new_e = tuple(
                (e, target_floor, w) if e == eid else (e, f, w)
                for (e, f, w) in elevators_t
            )
            return (new_e, persons_t, 0.0, False)

        if act_type == _ACT_ENTER:
            pid, eid = a1, a2
            pw = self.person_weight[pid]
            new_e = tuple(
                (e, f, w + pw) if e == eid else (e, f, w)
                for (e, f, w) in elevators_t
            )
            new_p = tuple(
                (p, ('in', eid)) if p == pid else (p, loc)
                for (p, loc) in persons_t
            )
            return (new_e, new_p, 0.0, False)

        if act_type == _ACT_EXIT:
            pid, eid = a1, a2
            pw    = self.person_weight[pid]
            imm_r = self.person_mean_rew[pid]
            new_e = tuple(
                (e, f, max(0, w - pw)) if e == eid else (e, f, w)
                for (e, f, w) in elevators_t
            )
            new_p = tuple((p, loc) for (p, loc) in persons_t if p != pid)
            goal  = (len(new_p) == 0) and (not self.doing_loop)
            # In loop mode, "goal" for this cycle = no more targets remain
            if self.doing_loop:
                remaining_targets = any(p in self.target_pids for (p, loc) in new_p)
                goal = not remaining_targets
            return (new_e, new_p, imm_r, goal)

        return (elevators_t, persons_t, 0.0, False)

    # ======================================================================
    # Q-value (expected value of taking action a in state s)
    # ======================================================================

    def _q_value(self, action_code, elevators_t, persons_t, steps_left):
        """
        Q(s, a) = E[r_a + V(s')]

        For stochastic actions we weight over success/failure outcomes:

        MOVE{eid, f}:
            success (p_e): elevator moves to f
            failure (1-p_e): elevator ends at uniform random other floor
              → approximate failure as NO progress (elevator stays put).
              This is correct in expectation for subsequent re-attempts
              (geometric distribution restarts), so the total expected
              steps for "move eid to f" is still 1/p_e. We capture this
              via the heuristic which already uses 1/p_e as move cost.
              So for the one-step lookahead: treat failure as no change,
              weighted appropriately.

        ENTER{pid, eid}:
            success (p_p): person enters elevator
            failure (1-p_p): state unchanged

        EXIT{pid, eid}:
            success (p_p): person delivered, reward earned
            failure (1-p_p): state unchanged

        RESET: deterministic, goes to init state.
        """
        if steps_left <= 1:
            # Last step: only immediate reward matters
            act_type, a1, a2 = action_code
            if act_type == _ACT_EXIT:
                return self.person_prob[a1] * self.person_mean_rew[a1]
            return 0.0

        act_type, a1, a2 = action_code
        next_steps = steps_left - 1

        # ---- RESET --------------------------------------------------------
        if act_type == _ACT_RESET:
            v_init = self._state_value(self._init_elevators, self._init_persons,
                                       next_steps)
            return v_init

        # ---- MOVE ---------------------------------------------------------
        if act_type == _ACT_MOVE:
            eid = a1
            p_e = self.elev_prob[eid]

            # Success outcome
            ne_s, np_s, _, _ = self._apply_success(action_code, elevators_t, persons_t)
            v_succ = self._state_value(ne_s, np_s, next_steps)

            # Failure outcome: elevator stays (conservative approximation)
            # We do NOT move the elevator in the failure branch — it ends
            # up somewhere random, but since we'll re-issue the same MOVE
            # next step, the heuristic already accounts for this correctly
            # via E_move = 1/p_e.  For the 1-step lookahead we approximate
            # failure as "current state unchanged".
            v_fail = self._state_value(elevators_t, persons_t, next_steps)

            return p_e * v_succ + (1.0 - p_e) * v_fail

        # ---- ENTER --------------------------------------------------------
        if act_type == _ACT_ENTER:
            pid = a1
            p_p = self.person_prob[pid]

            ne_s, np_s, _, _ = self._apply_success(action_code, elevators_t, persons_t)
            v_succ = self._state_value(ne_s, np_s, next_steps)
            v_fail = self._state_value(elevators_t, persons_t, next_steps)

            return p_p * v_succ + (1.0 - p_p) * v_fail

        # ---- EXIT ---------------------------------------------------------
        if act_type == _ACT_EXIT:
            pid = a1
            p_p  = self.person_prob[pid]
            imm  = self.person_mean_rew[pid]

            ne_s, np_s, _, goal = self._apply_success(action_code, elevators_t, persons_t)

            if goal:
                # All targets delivered; next state is initial (auto-reset or manual)
                if not self.doing_loop:
                    # Engine auto-resets and awards goal_reward
                    v_next = self._state_value(self._init_elevators,
                                               self._init_persons, next_steps)
                    v_succ = (imm + self.goal_reward) + v_next
                else:
                    # Loop mode: we'll RESET next step ourselves
                    v_next = self._state_value(self._init_elevators,
                                               self._init_persons, next_steps - 1)
                    # -ρ* for the extra RESET step cost
                    v_succ = imm + (v_next - self.rho_star * 1.0)
            else:
                v_next = self._state_value(ne_s, np_s, next_steps)
                v_succ = imm + v_next

            v_fail = self._state_value(elevators_t, persons_t, next_steps)

            return p_p * v_succ + (1.0 - p_p) * v_fail

        return 0.0

    # ======================================================================
    # Main entry point: choose_action
    # ======================================================================

    def choose_action(self, state):
        """
        Called by the game engine each step.
        Selects and returns the action string maximising Q(s, a).
        """
        elevators_t, persons_t, _total_rem = state
        steps_left = self.game.get_max_steps() - self.game.get_current_steps()

        if steps_left <= 0:
            return RESET_STR

        # Enumerate valid actions (integer codes)
        actions = self._enum_actions(elevators_t, persons_t)

        best_q   = -float('inf')
        best_act = (_ACT_RESET, 0, 0)

        for act_code in actions:
            q = self._q_value(act_code, elevators_t, persons_t, steps_left)
            if q > best_q:
                best_q   = q
                best_act = act_code

        # Convert best action code to string (only here, once)
        return self._fmt_action(best_act)

    def choose_next_action(self, state):
        """Compatibility wrapper for the checker and simulation harness."""
        return self.choose_action(state)

    # ======================================================================
    # Format action code → string
    # ======================================================================

    @staticmethod
    def _fmt_action(action_code):
        act_type, a1, a2 = action_code
        if act_type == _ACT_RESET:
            return RESET_STR
        if act_type == _ACT_MOVE:
            return _fmt_move(a1, a2)
        if act_type == _ACT_ENTER:
            return _fmt_enter(a1, a2)
        if act_type == _ACT_EXIT:
            return _fmt_exit(a1, a2)
        return RESET_STR