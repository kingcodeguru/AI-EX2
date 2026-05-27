import os
import sys
import readchar

def input(prompt):
    print(prompt, end='', flush=True)
    return readchar.readchar()

def run_test(version):
    print("What test do you want to run?")
    tests = [('David', './tests/david/run.sh')]
    for i, (name, _) in enumerate(tests, start=1):
        print(f"{i}. {name}'s tests")
    choice = input("Enter your choice: ")
    if choice in [str(i) for i, _ in enumerate(tests, start=1)]:
        _, command = tests[int(choice) - 1]
        os.system(f"{command} {version}")
    else:
        print("Invalid choice. Please try again.")

def main(argv):
    version = argv[1] if len(argv) > 1 else "1"
    while True:
        print("What do you want to do?")
        print("1. run a test")
        print("2. exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            run_test(version)
        elif choice == "2":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")



if __name__ == "__main__":
    main(sys.argv)