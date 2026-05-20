This repository contains the configuration and code for setting up a **Distributed GPU Training MLOps pipeline** on Nebius Cloud. It serves as an educational assignment to learn how to provision cloud infrastructure, containerize ML workloads, and orchestrate distributed training using PyTorch DDP and SkyPilot.

### Overall Purpose
The goal is to fine-tune a Causal Language Model (like `facebook/opt-1.3b` or `opt-2.7b`) on the `wikitext` dataset using Distributed Data Parallel (DDP) across multiple GPU nodes. The pipeline automates the deployment and execution of this distributed workload.

### Architecture
The architecture is composed of four main components interacting together:

1. **Infrastructure (Nebius Managed Kubernetes - mk8s)**
   - The foundation is a managed Kubernetes cluster configured with a node group of GPU instances (e.g., H100 or L40S presets). This provides the raw compute power needed for the training job.
   - A Nebius Container Registry is used to host the Docker image so the Kubernetes cluster can pull it during job execution.

2. **Orchestration (SkyPilot)**
   - **SkyPilot** is used as the orchestration layer to abstract away the complexity of Kubernetes job management.
   - It runs an API server (either managed or self-deployed) and accepts job submissions via the `train_job.yaml` configuration.
   - The `train_job.yaml` defines the cluster requirements (`H100:1`, `num_nodes: 2`), the Docker image to use, environment variables, and the execution commands.
   - SkyPilot automatically maps its internal environment variables (like `$SKYPILOT_NODE_IPS`, `$SKYPILOT_NODE_RANK`, `$SKYPILOT_NUM_GPUS_PER_NODE`) to set up the master node address and coordinate the distributed setup.

3. **Containerization (`Dockerfile`)**
   - The environment is standardized using a Docker image based on NVIDIA's optimized PyTorch container (`nvcr.io/nvidia/pytorch:25.12-py3`).
   - It installs all required ML dependencies such as `transformers`, `datasets`, `accelerate`, `peft`, `trl`, and `wandb`, ensuring reproducibility across all nodes in the cluster.

4. **Training Script (`train.py`)**
   - This script executes the actual machine learning workload.
   - It initializes the distributed process group using the `nccl` backend.
   - It loads a causal language model and the `wikitext` dataset from Hugging Face, handles the tokenization and block packing.
   - Finally, it uses the Hugging Face `Trainer` to perform the distributed training for 500 steps.
   - The script is launched by `torchrun` (configured in the `train_job.yaml`), which spawns the necessary processes per node and connects them to the master address for gradient synchronization.