import argparse
import glob
import logging
import os
import subprocess
import time

from tqdm import tqdm


def run_one(filename, timeout):
    if timeout is None:
        timeout = 40
    passing = True
    logging.info(f"Running {filename}")
    start = time.time()
    env = dict(os.environ, WANDB_DISABLED="true", IS_CI="1", MPLBACKEND="Agg")
    process = subprocess.Popen(["python", filename], env=env, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.poll() != 0:
            passing = False
            logging.error(f"{filename} FAILED")
            logging.error(stdout.decode("UTF-8"))
            logging.error(stderr.decode("UTF-8"))
        else:
            logging.info(f"Completed in {time.time() - start:.2f} seconds")
    except subprocess.TimeoutExpired:
        process.terminate()
        passing = False
        logging.error(f"{filename} FAILED, TOOK LONGER THAN {timeout} SECONDS")
    return passing


def run_many(fnames, timeout=None):
    failed = []
    sols = sorted(fnames)

    for sol in tqdm(sols, desc="Running solution files"):
        passing = run_one(sol, timeout=timeout)
        if not passing:
            failed.append(sol)
    if failed:
        logging.warning(f"Failed Tests: {', '.join(failed)}")
    else:
        logging.info(f"All tests passed!")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="*", default=[])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--soft", "-s", action="store_true")
    parser.add_argument("--timeout", "-t", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARNING))

    if not args.filename:
        fnames = glob.glob("*_solution.py")
    else:
        fnames = args.filename

    failed = run_many(fnames, timeout=args.timeout)

    if not args.soft:
        assert not failed
    else:
        logging.info(f"Tests completed, {len(failed)} failed")
