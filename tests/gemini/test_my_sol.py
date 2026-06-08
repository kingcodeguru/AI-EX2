import sys
import os
import time
import importlib.util

# Setup paths
sys.path.insert(0, os.path.abspath('original'))
import ext_elev
import ex2_check

# Import my controller
spec = importlib.util.spec_from_file_location("ex2", "my/versions/ex2-v3.py")
ex2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex2)

def solve(api):
    controller = ex2.Controller(api)
    for _ in range(api.get_max_steps()):
        action = controller.choose_next_action(api.get_current_state())
        api.submit_next_action(action)
        if api.get_done():
            break
    return api.get_current_reward()

# Run m5_easy
name, problem = [p for p in ex2_check.PROBLEMS if p[0] == 'm5_easy'][0]
print(f"Testing {name}...")
problem["seed"] = 42
api = ext_elev.create_elevators_game(problem, debug=False)
start = time.perf_counter()
reward = solve(api)
print(f"Result: reward={reward}, time={time.perf_counter()-start:.4f}s")
