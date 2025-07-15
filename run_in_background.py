import subprocess
import sys
import os
import time

def launch_process(command_list, log_path):
    """
    Launch a background process with output redirected to log_path.
    """
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        command_list,
        stdout=log_file,
        stderr=log_file
    )
    return process.pid

def run_all(script_path, python_executable=sys.executable):
    """
    Launch:
    1. visdom -port 8997
    2. visdom -port 8998
    3. A target Python script

    All in background, each logging to its own file.
    """

    if not os.path.isfile(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return

    print("[INFO] Launching background services...")

    visdom_pid_1 = launch_process(["visdom", "-port", "8997"], "visdom_8997.log")
    print(f"[OK] Visdom launched on port 8997 (PID {visdom_pid_1})")

    visdom_pid_2 = launch_process(["visdom", "-port", "8998"], "visdom_8998.log")
    print(f"[OK] Visdom launched on port 8998 (PID {visdom_pid_2})")

    # Optional: wait briefly to ensure visdom servers start before continuing
    time.sleep(2)

    script_log_path = "background_script.log"
    script_pid = launch_process([python_executable, script_path], script_log_path)
    print(f"[OK] Script launched: {script_path} (PID {script_pid})")

    print("\n[SUMMARY]")
    print(f"Visdom 8997 PID: {visdom_pid_1} (log: visdom_8997.log)")
    print(f"Visdom 8998 PID: {visdom_pid_2} (log: visdom_8998.log)")
    print(f"Script PID:      {script_pid} (log: {script_log_path})")
    print("\nTo stop: kill <PID>")

# Example usage
if __name__ == "__main__":
    target_script = "train_MBD_Parse2022.py"  # <-- Replace this
    run_all(target_script)
