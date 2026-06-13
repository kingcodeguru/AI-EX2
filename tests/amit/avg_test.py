import os
import re

BASELINE_DIR = "baseline"
SOLUTION_FILE = "Solution.txt"

GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
RESET  = '\033[0m'
BOLD   = '\033[1m'


def parse_solution(path):
    """Return {problem: avg_reward} using the last occurrence of each problem."""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "reward_average=" not in line:
                continue
            try:
                name = line.split(":")[0].strip()
                avg  = float(line.split("reward_average=")[1].split("|")[0].strip())
                results[name] = avg
            except Exception:
                pass
    return results


def parse_baselines(base_dir):
    """Return {config: {problem: avg_score}} from each subfolder's .log files."""
    baselines = {}
    for config in sorted(os.listdir(base_dir)):
        config_path = os.path.join(base_dir, config)
        if not os.path.isdir(config_path):
            continue
        scores = {}
        for fname in os.listdir(config_path):
            if not fname.endswith(".log"):
                continue
            problem = fname[:-4]
            fpath = os.path.join(config_path, fname)
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    m = re.search(r'#\s*avg=([0-9.]+)', line)
                    if m:
                        scores[problem] = float(m.group(1))
        if scores:
            baselines[config] = scores
    return baselines


def run_test():
    my_scores = parse_solution(SOLUTION_FILE)
    if not my_scores:
        print(f"{RED}No results found in {SOLUTION_FILE}. Run ex2_check.py first.{RESET}")
        return

    baselines = parse_baselines(BASELINE_DIR)
    if not baselines:
        print(f"{RED}No baseline data found in '{BASELINE_DIR}/'.{RESET}")
        return

    configs   = sorted(baselines.keys())
    problems  = sorted(my_scores.keys())

    col_w  = 10
    name_w = 12

    header = f"{'Problem':<{name_w}} | {'My Avg':>{col_w}}"
    for c in configs:
        header += f" | {c:>{col_w}}"
    header += f" | {'Beats':>6}"
    print(f"\n{BOLD}=== AVERAGE BASELINE COMPARISON ==={RESET}\n")
    print(header)
    print("-" * len(header))

    total_beats = 0
    total_comps = 0

    for prob in problems:
        my = my_scores[prob]
        row = f"{prob:<{name_w}} | {my:>{col_w}.2f}"
        beats = 0
        n = 0
        for c in configs:
            bl = baselines[c].get(prob)
            if bl is None:
                row += f" | {'N/A':>{col_w}}"
                continue
            n += 1
            if my >= bl:
                beats += 1
                row += f" | {GREEN}{bl:>{col_w}.2f}{RESET}"
            else:
                row += f" | {RED}{bl:>{col_w}.2f}{RESET}"
        total_beats += beats
        total_comps += n
        beat_str = f"{beats}/{n}"
        color = GREEN if beats == n else (YELLOW if beats > 0 else RED)
        row += f" | {color}{beat_str:>6}{RESET}"
        print(row)

    print("-" * len(header))
    pct = 100.0 * total_beats / total_comps if total_comps else 0
    print(f"\n{BOLD}Summary:{RESET} Beat {total_beats}/{total_comps} "
          f"(problem × baseline) pairs  ({pct:.1f}%)\n")

    if total_beats == total_comps:
        print(f"{GREEN}{BOLD}Perfect — you beat every baseline on every problem!{RESET}\n")
    elif pct >= 80:
        print(f"{GREEN}Strong performance — beating most baselines.{RESET}\n")
    elif pct >= 50:
        print(f"{YELLOW}Decent — room to improve on weaker baselines.{RESET}\n")
    else:
        print(f"{RED}Needs work — failing more than half the comparisons.{RESET}\n")


if __name__ == "__main__":
    run_test()
