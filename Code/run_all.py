import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "outputs"
LOG_FILE = LOG_DIR / "run_all_log.txt"

TRAIN_COMMANDS = [
    "python3 src/train.py",
    "python3 src/train_dialted.py",
    "python3 src/train_vgg16.py",
    "python3 src/train_vgg16_dialated.py",
    "python3 src/train_segnet.py",
    "python3 src/train_segnet_dialated.py",
    "python3 src/train_lraspp.py",
    "python3 src/train_lraspp_dialated.py",
]


def run_one_command(command, log_file_handle):
    separator = "=" * 80
    header = f"\n{separator}\nRUNNING: {command}\n{separator}\n"
    print(header, end="")
    log_file_handle.write(header)
    log_file_handle.flush()

    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    for line in process.stdout:
        print(line, end="")
        log_file_handle.write(line)

    process.wait()
    log_file_handle.flush()
    return process.returncode


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()

    with open(LOG_FILE, "w", encoding="utf-8") as log_file:
        log_file.write(f"Run started: {start_time.isoformat()}\n")
        log_file.write(f"Working directory: {ROOT_DIR}\n")
        log_file.flush()

        failed = []
        for command in TRAIN_COMMANDS:
            code = run_one_command(command, log_file)
            if code != 0:
                failed.append((command, code))

        end_time = datetime.now()
        summary = [
            "\n" + "=" * 80,
            "RUN SUMMARY",
            "=" * 80,
            f"Started: {start_time.isoformat()}",
            f"Ended:   {end_time.isoformat()}",
            f"Log file: {LOG_FILE}",
        ]

        if failed:
            summary.append("Status: FAILED")
            for command, code in failed:
                summary.append(f"- {command} (exit code {code})")
        else:
            summary.append("Status: SUCCESS")

        summary_text = "\n".join(summary) + "\n"
        print(summary_text, end="")
        log_file.write(summary_text)

    if failed:
        raise SystemExit(1)


main()
