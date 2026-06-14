import os
import sys
import readchar

def input(prompt):
    print(prompt, end='', flush=True)
    return readchar.readchar()

def run(version, scripts):
    print("What test do you want to run?")
    for i, (name, _) in enumerate(scripts, start=1):
        print(f"{i}. {name}'s tests")
    choice = input("Enter your choice: ")
    if choice in [str(i) for i, _ in enumerate(scripts, start=1)]:
        _, command = scripts[int(choice) - 1]
        os.system(f"{command} {version}")

def run_simulation(version):
    print("What simulation do you want to run?")
    simulations = [('David', './simulations/run.sh')]
    run(version, simulations)

def run_test(version):
    print("What test do you want to run?")
    tests = [('David', './tests/david/run.sh'),
             ('Amit', './tests/amit/run.sh')]
    run(version, tests)


def resolve_version_file(argv):
    if len(argv) > 1 and argv[1] == "-f":
        if len(argv) < 3:
            raise SystemExit("usage: my/main.py [-f <version-file>] [version]")
        return argv[2]

    version = argv[1] if len(argv) > 1 else "1"
    return f"ex2-v{version}.py"

def main(argv):
    version_file = resolve_version_file(argv)
    while True:
        print("What do you want to do?")
        print("1. run a test")
        print("2. run simulation")
        print("3. exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            run_test(version_file)
        elif choice == "2":
            run_simulation(version_file)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")



if __name__ == "__main__":
    main(sys.argv)