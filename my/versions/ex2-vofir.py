import heapq
import time
import ext_elev

id = ["000000000"]


# ====================================================================== #
#  Stochastic Multi-Elevator Routing controller (Finite Horizon MDP)     #
#                                                                        #
#  Strategy                                                              #
#  --------                                                              #
#  The episode is a renewal process: we repeatedly run a "cycle" that     #
#  collects reward and returns the board to its initial layout -- either  #
#  via the engine's automatic reset after a full clear (which also pays   #
#  goal_reward), or via an explicit RESET after delivering a profitable   #
#  subset of persons. The best long-run policy farms whichever cycle has  #
#  the highest reward-per-step rate, so __init__ measures the farming     #
#  efficiency of *every* subset of persons and picks the winner. That is  #
#  the "RESET farming" decision: farm one lucrative person on repeat, or  #
#  clear the whole board for the goal bonus.                              #
#                                                                        #
#  Routing is solved with a stochastic shortest-path search (Dijkstra,    #
#  edge weight 1/p for any action with success probability p), which      #
#  also discovers multi-elevator relays through shared floors. Each       #
#  person keeps *several* candidate delivery plans (one per feasible      #
#  elevator plus the optimal relay) so the controller can use whichever   #
#  elevator is best positioned -- this load-balances work across cars.    #
#  Broken / low-probability elevators are only ever used when strictly    #
#  necessary, because their heavy edge weights make the search avoid them.#
#                                                                        #
#  choose_next_action runs a depth-limited Expectimax (lookahead value    #
#  iteration) whose leaf evaluation is the renewal-reward potential       #
#      F(s) = R_remaining(s) - rho * g(s)                                 #
#  where g(s) is the expected number of steps to finish the current       #
#  cycle and rho is the farmed cycle's reward rate. Acting greedily on    #
#  E[ r + F(s') ] is the optimal one-step rule for the renewal value      #
#  function; the extra plies refine tie-breaks and resolve the immediate  #
#  stochastic branching exactly. A symmetry-aware cache keyed on          #
#  (canonical_state, depth) prunes redundant subtrees, and only "useful"  #
#  actions (those advancing a target person, plus RESET) are expanded.    #
# ====================================================================== #

INF = float("inf")


class Plan:
    """One concrete delivery scheme for a person: an ordered list of legs
    (elevator, pickup_floor, dropoff_floor) plus pre-computed cost arrays."""
    __slots__ = ("legs", "suffix", "intr", "suffix_min", "intr_min")

    def __init__(self, legs, pe, q):
        self.legs = legs
        n = len(legs)
        # expected-cost (1/p) arrays
        self.intr = []
        for (e, a, b) in legs:
            c = 2.0 / q                       # enter + exit
            if a != b:
                c += 1.0 / pe[e]              # carry move
            self.intr.append(c)
        self.suffix = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            self.suffix[i] = self.suffix[i + 1] + (1.0 / pe[legs[i][0]]) + self.intr[i]
        # optimistic unit-cost arrays (every action succeeds) for end-game gating
        self.intr_min = [2 + (1 if a != b else 0) for (e, a, b) in legs]
        self.suffix_min = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            self.suffix_min[i] = self.suffix_min[i + 1] + 1 + self.intr_min[i]


class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game

        # ---- static configuration (read once, public API only) ---- #
        init_state = game.get_initial_state()
        elevators_t, persons_t, _ = init_state
        self.initial_state = init_state
        self.max_steps = game.get_max_steps()
        self.goal_reward = float(game.get_goal_reward())

        reachable = game.get_reachable()
        capacities = game.get_capacities()

        self.elev_ids = [eid for (eid, _, _) in elevators_t]
        self.reachable = {e: frozenset(reachable[e]) for e in self.elev_ids}
        self.cap = {e: capacities[e] for e in self.elev_ids}
        self.pe = {e: float(game.get_elevator_action_prob(e)) for e in self.elev_ids}
        self.init_efloor = {eid: fl for (eid, fl, _) in elevators_t}

        self.person_ids = [pid for (pid, _) in persons_t]
        self.start = {pid: loc[1] for (pid, loc) in persons_t}
        self.goal = {p: game.get_person_goal(p) for p in self.person_ids}
        self.weight = {p: game.get_person_weight(p) for p in self.person_ids}
        self.qp = {p: float(game.get_person_action_prob(p)) for p in self.person_ids}
        self.rewards = {p: tuple(game.get_person_reward(p)) for p in self.person_ids}
        self.Erew = {p: sum(self.rewards[p]) / len(self.rewards[p])
                     for p in self.person_ids}

        self.all_persons = frozenset(self.person_ids)

        # ---- per-person candidate delivery plans ---- #
        self.plans = {}            # plans[p] = list[Plan]
        self.deliverable = set()
        for p in self.person_ids:
            plist = self._build_plans(p)
            self.plans[p] = plist
            if plist:
                self.deliverable.add(p)

        self.min_from_init = {
            p: (self._person_minsteps(p, ('floor', self.start[p]), self.init_efloor)
                if self.plans[p] else INF)
            for p in self.person_ids
        }

        # ---- choose which cycle to farm (the core strategic decision) ---- #
        self.S_star, self.allpersons_flag, self.rho, self.cycle_cost = \
            self._choose_cycle()
        self.target = frozenset(self.S_star)

        # ---- lookahead config ---- #
        # Branching is tiny (only useful actions are expanded) and the symmetry
        # cache collapses subtrees, so a few plies are cheap. Multi-person relay
        # problems are prone to a shuttle livelock that *deeper* search amplifies,
        # so depth is kept modest there; single/two-person farming benefits from
        # an extra ply for the end-game squeeze.
        live = max(1, len(self.target))
        self.depth = 5 if live <= 1 else (4 if live <= 2 else 3)
        self.tol = 0.0
        self._cache = {}

        # Wall-clock guard: stay well under the grader's per-seed time budget by
        # shrinking the lookahead depth if an episode ever runs long.
        self._t0 = time.perf_counter()
        self._soft = 0.7

    # ------------------------------------------------------------------ #
    #  Plan construction (stochastic shortest paths)                     #
    # ------------------------------------------------------------------ #
    def _build_plans(self, p):
        """Return candidate delivery plans for p: a single-leg plan for every
        elevator able to carry p start->goal directly, plus the globally
        optimal (possibly relay) plan from Dijkstra. Empty => undeliverable."""
        s, g, w = self.start[p], self.goal[p], self.weight[p]
        seen = set()
        plans = []

        for e in self.elev_ids:
            if s in self.reachable[e] and g in self.reachable[e] and w <= self.cap[e]:
                legs = ((e, s, g),)
                if legs not in seen:
                    seen.add(legs)
                    plans.append(Plan(list(legs), self.pe, self.qp[p]))

        relay = self._dijkstra_plan(p)
        if relay is not None:
            legs = tuple(relay)
            if legs not in seen:
                seen.add(legs)
                plans.append(Plan(relay, self.pe, self.qp[p]))
        return plans

    def _dijkstra_plan(self, p):
        """Min expected-cost delivery path (handles multi-elevator relays)."""
        s, g, w, q = self.start[p], self.goal[p], self.weight[p], self.qp[p]
        start = ('F', s)
        dist = {start: 0.0}
        prev = {start: None}
        cnt = 0
        pq = [(0.0, 0, start)]

        while pq:
            d, _, node = heapq.heappop(pq)
            if d > dist.get(node, INF):
                continue
            if node == 'DONE':
                break
            for (nbr, cost) in self._neighbors(node, g, w, q):
                nd = d + cost
                if nd < dist.get(nbr, INF):
                    dist[nbr] = nd
                    prev[nbr] = node
                    cnt += 1
                    heapq.heappush(pq, (nd, cnt, nbr))

        if 'DONE' not in prev:
            return None
        path = []
        cur = 'DONE'
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        legs = []
        cur_e = cur_a = cur_f = None
        for node in path:
            if node == 'DONE':
                if cur_e is not None:
                    legs.append((cur_e, cur_a, cur_f))
                break
            if node[0] == 'F':
                if cur_e is not None:
                    legs.append((cur_e, cur_a, cur_f))
                    cur_e = None
                cur_f = node[1]
            else:
                _, e, f = node
                if cur_e is None:
                    cur_e, cur_a = e, f
                cur_f = f
        return legs

    def _neighbors(self, node, g, w, q):
        out = []
        if node[0] == 'F':
            f = node[1]
            for e in self.elev_ids:
                if f in self.reachable[e] and w <= self.cap[e]:
                    out.append((('I', e, f), 1.0 / self.pe[e] + 1.0 / q))
        else:
            _, e, f = node
            for f2 in self.reachable[e]:
                if f2 != f:
                    out.append((('I', e, f2), 1.0 / self.pe[e]))
            if f == g:
                out.append(('DONE', 1.0 / q))
            else:
                out.append((('F', f), 1.0 / q))
        return out

    # ------------------------------------------------------------------ #
    #  Cycle selection (RESET farming)                                   #
    # ------------------------------------------------------------------ #
    def _choose_cycle(self):
        deliverable = sorted(self.deliverable)
        n = len(deliverable)
        if n == 0:
            return frozenset(), False, 0.0, 1.0

        best = None
        for mask in range(1, 1 << n):
            S = [deliverable[i] for i in range(n) if (mask >> i) & 1]
            S_set = frozenset(S)
            allflag = (S_set == self.all_persons)
            reward = sum(self.Erew[p] for p in S)
            if allflag:
                reward += self.goal_reward
            cost = self._cycle_cost(S_set, allflag)
            if cost == INF or cost <= 0:
                continue
            rate = reward / cost
            if best is None or rate > best[0]:
                best = (rate, S_set, allflag, cost)

        if best is None:
            return frozenset(), False, 0.0, 1.0
        return best[1], best[2], best[0], best[3]

    def _cycle_cost(self, S_set, allflag):
        """Deterministic expected-cost rollout of the greedy delivery policy
        for S_set from the initial board (charges 1/p per primitive action,
        assumes success). +1 for the explicit RESET on a strict subset."""
        efl = dict(self.init_efloor)
        ew = {e: 0 for e in self.elev_ids}
        ploc = {p: ('floor', self.start[p]) for p in self.person_ids}

        cost = 0.0
        for _ in range(400):
            present = [p for p in S_set if p in ploc]
            if not present:
                if not allflag:
                    cost += 1.0
                return cost
            best_act, best_after, best_prob = None, INF, 1.0
            for (act, prob) in self._candidate_actions(efl, ew, ploc, present):
                if act[0] == 'RESET':
                    continue
                nefl, nw, nploc = self._apply_success(act, efl, ew, ploc)
                npres = [p for p in S_set if p in nploc]
                ng = self._g_raw(npres, nploc, nefl, INF)
                if ng < best_after - 1e-9:
                    best_after, best_act, best_prob = ng, act, prob
            if best_act is None:
                return INF
            cost += 1.0 / best_prob
            efl, ew, ploc = self._apply_success(best_act, efl, ew, ploc)
        return INF

    # ------------------------------------------------------------------ #
    #  Cost-to-go (g) and renewal potential (F)                          #
    # ------------------------------------------------------------------ #
    def _person_ctg(self, p, loc, efl):
        """Expected steps to finish delivering p, minimised over its candidate
        plans, given actual elevator floors for the immediate leg."""
        best = INF
        if loc[0] == 'in':
            e = loc[1]
            f = efl[e]
            for plan in self.plans[p]:
                legs = plan.legs
                for i in range(len(legs)):
                    if legs[i][0] != e:
                        continue
                    b = legs[i][2]
                    c = (0.0 if f == b else 1.0 / self.pe[e]) + (1.0 / self.qp[p])
                    c += plan.suffix[i + 1]
                    if c < best:
                        best = c
                    break
        else:
            f = loc[1]
            for plan in self.plans[p]:
                legs = plan.legs
                for i in range(len(legs)):
                    if legs[i][1] != f:
                        continue
                    e = legs[i][0]
                    rep = 0.0 if efl[e] == f else (1.0 / self.pe[e])
                    c = rep + plan.intr[i] + plan.suffix[i + 1]
                    if c < best:
                        best = c
                    break
        return best

    def _person_minsteps(self, p, loc, efl):
        """Optimistic remaining action count (assume every action succeeds)."""
        best = INF
        if loc[0] == 'in':
            e = loc[1]
            f = efl[e]
            for plan in self.plans[p]:
                legs = plan.legs
                for i in range(len(legs)):
                    if legs[i][0] != e:
                        continue
                    b = legs[i][2]
                    c = (0 if f == b else 1) + 1 + plan.suffix_min[i + 1]
                    if c < best:
                        best = c
                    break
        else:
            f = loc[1]
            for plan in self.plans[p]:
                legs = plan.legs
                for i in range(len(legs)):
                    if legs[i][1] != f:
                        continue
                    e = legs[i][0]
                    rep = 0 if efl[e] == f else 1
                    c = rep + plan.intr_min[i] + plan.suffix_min[i + 1]
                    if c < best:
                        best = c
                    break
        return best

    def _g_raw(self, present, ploc, efl, steps_left):
        g = 0.0
        for p in present:
            c = self._person_ctg(p, ploc[p], efl)
            if c <= steps_left:
                g += c
        return g

    def _potential(self, efl, ploc, steps_left):
        """Renewal-reward leaf value  F(s) = R_remaining(s) - rho * g(s).

        g(s) is the expected steps to finish the current cycle (sum of
        per-person cost-to-go); R_remaining the reward still collectable this
        cycle. Acting greedily on E[r + F(s')] is the optimal one-step rule
        for the renewal value function. Persons that can no longer finish
        within the horizon are pruned from both terms (end-game awareness)."""
        present = [p for p in self.target if p in ploc]
        g = R = 0.0
        n_pursuable = 0
        for p in present:
            if self._person_minsteps(p, ploc[p], efl) <= steps_left:
                g += self._person_ctg(p, ploc[p], efl)
                R += self.Erew[p]
                n_pursuable += 1
        if self.allpersons_flag and present and n_pursuable == len(present):
            R += self.goal_reward
        return R - self.rho * g

    # ------------------------------------------------------------------ #
    #  Action generation + transition model                              #
    # ------------------------------------------------------------------ #
    def _candidate_actions(self, efl, ew, ploc, present):
        """Useful legal actions only: those advancing a present target person
        along one of its plans, plus RESET. Returns (action, success_prob)."""
        acts = []
        seen = set()

        def add(a, prob):
            if a not in seen:
                seen.add(a)
                acts.append((a, prob))

        for p in present:
            loc = ploc[p]
            if loc[0] == 'in':
                e = loc[1]
                f = efl[e]
                b = None
                for plan in self.plans[p]:
                    for (le, a, bb) in plan.legs:
                        if le == e:
                            b = bb
                            break
                    if b is not None:
                        break
                if b is None:
                    continue
                if f == b:
                    add(('EXIT', p, e), self.qp[p])
                else:
                    add(('MOVE', e, b), self.pe[e])
            else:
                f = loc[1]
                for plan in self.plans[p]:
                    for (le, a, bb) in plan.legs:
                        if a != f:
                            continue
                        if efl[le] == f:
                            if ew[le] + self.weight[p] <= self.cap[le]:
                                add(('ENTER', p, le), self.qp[p])
                        else:
                            add(('MOVE', le, f), self.pe[le])
                        break
        acts.append((('RESET',), 1.0))
        return acts

    def _apply_success(self, act, efl, ew, ploc):
        kind = act[0]
        if kind == 'RESET':
            return (dict(self.init_efloor),
                    {e: 0 for e in self.elev_ids},
                    {p: ('floor', self.start[p]) for p in self.person_ids})
        nefl, nw, nploc = dict(efl), dict(ew), dict(ploc)
        if kind == 'MOVE':
            _, e, f = act
            nefl[e] = f
        elif kind == 'ENTER':
            _, p, e = act
            nploc[p] = ('in', e)
            nw[e] += self.weight[p]
        elif kind == 'EXIT':
            _, p, e = act
            f = nefl[e]
            nw[e] -= self.weight[p]
            if f == self.goal[p]:
                del nploc[p]
            else:
                nploc[p] = ('floor', f)
        return nefl, nw, nploc

    def _outcomes(self, act, efl, ew, ploc):
        """Stochastic outcome distribution: (prob, reward, nefl, nw, nploc)."""
        kind = act[0]
        if kind == 'RESET':
            yield (1.0, 0.0,
                   dict(self.init_efloor),
                   {e: 0 for e in self.elev_ids},
                   {p: ('floor', self.start[p]) for p in self.person_ids})
            return
        if kind == 'MOVE':
            _, e, target = act
            pe = self.pe[e]
            nefl = dict(efl); nefl[e] = target
            yield (pe, 0.0, nefl, ew, ploc)
            cur = efl[e]
            options = sorted(set([cur]) | (set(self.reachable[e]) - {target}))
            pf = (1.0 - pe) / len(options)
            for fo in options:
                nf = dict(efl); nf[e] = fo
                yield (pf, 0.0, nf, ew, ploc)
            return
        if kind == 'ENTER':
            _, p, e = act
            q = self.qp[p]
            nploc = dict(ploc); nploc[p] = ('in', e)
            nw = dict(ew); nw[e] += self.weight[p]
            yield (q, 0.0, efl, nw, nploc)
            yield (1.0 - q, 0.0, efl, ew, ploc)
            return
        # EXIT
        _, p, e = act
        q = self.qp[p]
        f = efl[e]
        nw = dict(ew); nw[e] -= self.weight[p]
        if f == self.goal[p]:
            nploc = dict(ploc); del nploc[p]
            r = self.Erew[p]
            if len(nploc) == 0:                                   # full clear
                yield (q, r + self.goal_reward,
                       dict(self.init_efloor),
                       {ee: 0 for ee in self.elev_ids},
                       {pp: ('floor', self.start[pp]) for pp in self.person_ids})
            else:
                yield (q, r, efl, nw, nploc)
        else:
            nploc = dict(ploc); nploc[p] = ('floor', f)
            yield (q, 0.0, efl, nw, nploc)
        yield (1.0 - q, 0.0, efl, ew, ploc)

    # ------------------------------------------------------------------ #
    #  Expectimax lookahead                                              #
    # ------------------------------------------------------------------ #
    def _canon(self, efl, ploc):
        ev = tuple(sorted((e, efl[e]) for e in self.elev_ids))
        pv = tuple(sorted(ploc.items()))
        return (ev, pv)

    def _g_only(self, efl, ploc, steps_left):
        """The g(s) term alone: expected steps to finish the current cycle."""
        g = 0.0
        for p in self.target:
            loc = ploc.get(p)
            if loc is not None and self._person_minsteps(p, loc, efl) <= steps_left:
                g += self._person_ctg(p, loc, efl)
        return g

    def _value(self, efl, ew, ploc, depth, steps_left):
        """Expectimax value: max_a E[ r + value(s') ], leaf = F potential."""
        if depth <= 0 or steps_left <= 0:
            return self._potential(efl, ploc, steps_left)
        key = (self._canon(efl, ploc), depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        present = [p for p in self.target if p in ploc]
        best = -INF
        for (act, _prob) in self._candidate_actions(efl, ew, ploc, present):
            ev = 0.0
            for (prob, r, nefl, nw, nploc) in self._outcomes(act, efl, ew, ploc):
                ev += prob * (r + self._value(nefl, nw, nploc,
                                              depth - 1, steps_left - 1))
            if ev > best:
                best = ev
        if best == -INF:
            best = self._potential(efl, ploc, steps_left)
        self._cache[key] = best
        return best

    # ------------------------------------------------------------------ #
    #  Public entry point                                                #
    # ------------------------------------------------------------------ #
    def choose_next_action(self, state):
        elevators_t, persons_t, _total = state
        efl = {eid: fl for (eid, fl, _) in elevators_t}
        ew = {eid: w for (eid, _, w) in elevators_t}
        ploc = {pid: loc for (pid, loc) in persons_t}

        steps_left = self.max_steps - self.game.get_current_steps()
        present = [p for p in self.target if p in ploc]

        # Strict-subset cycle finished -> restart with RESET while there is
        # (optimistically) time to reset and finish one more delivery.
        if not present:
            if self.S_star:
                cheapest = min((self.min_from_init[p] for p in self.target),
                               default=INF)
                if steps_left >= 1 + cheapest:
                    return "RESET"
            return self._fallback_action(elevators_t)

        self._cache = {}
        depth = self.depth if steps_left >= self.depth else max(1, steps_left)
        # Throttle depth if we are spending too long this episode.
        elapsed = time.perf_counter() - self._t0
        if elapsed > self._soft:
            depth = min(depth, 2)
        if elapsed > self._soft * 1.3:
            depth = 1
        acts = self._candidate_actions(efl, ew, ploc, present)

        # Maximise the renewal value E[ r + F(s') ]. Break ties (within TOL)
        # by making the most *progress* -- smallest expected remaining cycle
        # cost E[g(s')] -- then prefer committal / reliable actions. Pure
        # progress descent provably avoids the shuttle livelock the renewal
        # potential is otherwise prone to, while the value term drives all the
        # farming / reset decisions.
        PRIORITY = {'EXIT': 3, 'ENTER': 2, 'MOVE': 1, 'RESET': 0}
        TOL = self.tol
        best_act = None
        best_val = -INF
        best_key = None
        for (act, prob) in acts:
            ev = 0.0
            g_ev = 0.0
            for (pr, r, nefl, nw, nploc) in self._outcomes(act, efl, ew, ploc):
                ev += pr * (r + self._value(nefl, nw, nploc,
                                            depth - 1, steps_left - 1))
                g_ev += pr * self._g_only(nefl, nploc, steps_left - 1)
            key = (-g_ev, PRIORITY[act[0]], prob)
            if best_act is None or ev > best_val + TOL:
                best_act, best_val, best_key = act, ev, key
            elif ev >= best_val - TOL and key > best_key:
                best_act, best_key = act, key
                if ev > best_val:
                    best_val = ev

        if best_act is None:
            return self._fallback_action(elevators_t)
        return self._format(best_act)

    # ------------------------------------------------------------------ #
    #  Utilities                                                         #
    # ------------------------------------------------------------------ #
    def _format(self, act):
        kind = act[0]
        if kind == 'RESET':
            return "RESET"
        if kind == 'MOVE':
            return "MOVE{%d,%d}" % (act[1], act[2])
        if kind == 'ENTER':
            return "ENTER{%d,%d}" % (act[1], act[2])
        return "EXIT{%d,%d}" % (act[1], act[2])

    def _fallback_action(self, elevators_t):
        for (eid, cur_f, _) in elevators_t:
            for f in self.reachable[eid]:
                if f != cur_f:
                    return "MOVE{%d,%d}" % (eid, f)
        return "RESET"
