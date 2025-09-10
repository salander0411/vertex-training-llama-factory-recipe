"""Entrypoint for Vertex Distributed Training container."""

import argparse
import os
import sys
from collections.abc import Sequence
from subprocess import STDOUT, check_output, run

import subprocess as sp

from absl import app, flags, logging
from util.cluster_spec import get_cluster_spec, ClusterInfo
 
from retrying import retry

import shutil

NCCL_PLUGIN_PATH = os.getenv("NCCL_PLUGIN_PATH")
NCCL_INIT_SCRIPT = os.getenv("NCCL_INIT_SCRIPT")


# PyTorch barrier call which synchronizes all of the nodes before launching the training process.
# This makes sure that processes will block until all processes are ready.
# Improves the reliability of spot VM usage for multi-node training jobs
@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)
def barrier_with_retry() -> None:
    import torch
    logging.info("Starting barrier on RANK {}".format(os.environ["RANK"]))
    torch.distributed.init_process_group()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    logging.info("Finished barrier on RANK {}".format(os.environ["RANK"]))


def main(unused_argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser()
        
    parser.add_argument(
        "--nproc-per",
        type=int,
        help="Number of processes per node",
        default=8,
    )

    args, unknown = parser.parse_known_args()
    '''    
    for key, val in os.environ.items():
        logging.info("ENV %s=%s", key, val)
    '''

     # Set NCCL Environment
    if NCCL_INIT_SCRIPT and Path(NCCL_INIT_SCRIPT).exists():
        logging.info(f"Sourcing NCCL init script from {NCCL_INIT_SCRIPT} and applying workarounds...")
        command = (
            f'source "{NCCL_INIT_SCRIPT}" && '
            'unset NCCL_NET && '
            'export NCCL_NET_PLUGIN=/usr/local/gib/lib64/libnccl-net_internal.so && '
            'env | grep "^NCCL_"'
        )

        # Execute the command in a bash shell and capture its standard output.
        proc = sp.run(["bash", "-c", command], capture_output=True, text=True, check=True,)

        # Unset NCCL_NET explicitly in the current Python process's environment,
        if 'NCCL_NET' in os.environ:
            logging.info("Unsetting NCCL_NET in the current environment.")
            del os.environ['NCCL_NET']

        # Parse the KEY=VALUE output from the subshell and apply it to the parent environment.
        logging.info("Applying new NCCL environment variables to the current process:")
        for line in proc.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                logging.info(f"  Setting: {key}={value}")
                os.environ[key] = value
    else:
        logging.warning(
        f"NCCL_INIT_SCRIPT is not set or not found at '{NCCL_INIT_SCRIPT}'. "
        "Skipping NCCL environment setup."
        )

    try:
        logging.info(f"Running ldconfig for {NCCL_PLUGIN_PATH}...")
        sp.run(["ldconfig", str(NCCL_PLUGIN_PATH)], check=True, capture_output=True)
    except (sp.CalledProcessError, FileNotFoundError) as e:
        logging.warning(f"ldconfig command failed, which might be okay: {e}")

    logging.info("Final environment variables for torchrun:")
    for key, val in sorted(os.environ.items()):
        logging.info("ENV %s=%s", key, val)

    cluster_spec = get_cluster_spec()
    primary_node_addr = cluster_spec.primary_node_addr
    primary_node_port = cluster_spec.primary_node_port
    node_rank = cluster_spec.node_rank
    num_nodes = cluster_spec.num_nodes

    cmd = [
        "torchrun",
        f"--nproc-per-node={os.getenv('GPUS_PER_NODE', '8')}",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
    ]
    if num_nodes > 1:
        cmd += [
            "--max-restarts=3",
            "--rdzv-backend=static",
            f'--rdzv_id={os.getenv("CLOUD_ML_JOB_ID", primary_node_port)}',
            f"--rdzv-endpoint={primary_node_addr}:{primary_node_port}"
        ]
    cmd += [
        "/LLaMA-Factory/src/llamafactory/launcher.py"
    ]
    cmd += unknown

    logging.info("launching with cmd: \n%s", " \\\n".join(cmd))
    #barrier_with_retry()
    run(cmd, stdout=sys.stdout, stderr=sys.stdout, check=True)

    
if __name__ == "__main__":
    logging.get_absl_handler().python_handler.stream = sys.stdout
    app.run(main, flags_parser=lambda _args: flags.FLAGS(_args, known_only=True))