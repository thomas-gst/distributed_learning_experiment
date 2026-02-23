import subprocess
import sys
import time

# -------------------------
# CONFIGURATION
# -------------------------

USER = "thomas.gastellu"
SCRIPT = "train_ddp_mnist.py"
PROJECT_DIR = "~/venvs/csc_multi_modal/distributed_learning_experiment"

MASTER_ADDR = "porsche.polytechnique.fr"
MASTER_PORT = "12355"

NODES = [
    "porsche.polytechnique.fr",  # master (rank 0)
    "maserati.polytechnique.fr",  # worker1 (rank 1)
    "ferrari.polytechnique.fr",  # worker2 (rank 2)
]

# -------------------------

NNODES = len(NODES)


def launch_node(ip, rank):
    cmd = f"""
    cd {PROJECT_DIR} && \
    torchrun \
        --nnodes={NNODES} \
        --nproc_per_node=1 \
        --node_rank={rank} \
        --master_addr={MASTER_ADDR} \
        --master_port={MASTER_PORT} \
        {SCRIPT}
    """
   
    # SSH into worker
    ssh_cmd = f'ssh {USER}@{ip} "{cmd}"'
    print(f"Launching worker {ip} (rank {rank})")
    return subprocess.Popen(ssh_cmd, shell=True)


def main():
    processes = []

    for rank, ip in enumerate(NODES):
        p = launch_node(ip, rank)
        processes.append(p)
        time.sleep(1)  # small delay helps avoid race conditions

    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()