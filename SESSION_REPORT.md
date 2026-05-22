# Nebius Academy — Distributed GPU Training: Full Session Report

## What We Were Building

This homework was about setting up a complete **distributed ML training pipeline on cloud infrastructure**. The goal: take a pre-written training script, containerize it, provision a GPU Kubernetes cluster on Nebius Cloud, and launch a real multi-node distributed training job using PyTorch DDP orchestrated by SkyPilot. It's a representative slice of what MLOps looks like in production.

---

## Core Concepts

### Managed Kubernetes (MK8S)
Kubernetes is a container orchestration platform — it schedules and manages containers (pods) across a cluster of machines. **Nebius MK8S** is their managed version: you provision it via the console, and Nebius handles the control plane. You just get nodes. In this homework the cluster had **2 nodes, each with 1 NVIDIA L40S GPU (48GB VRAM)**, connected over a high-speed internal network.

### Docker Containers
Training jobs run inside Docker containers to ensure reproducibility — every dependency (PyTorch, HuggingFace Transformers, NCCL) is baked into the image at build time. Your image was hosted on **Nebius Container Registry** and referenced in the job YAML. The base image was `nvcr.io/nvidia/pytorch:25.12-py3` — NVIDIA's official PyTorch image with CUDA drivers pre-installed.

### SkyPilot
SkyPilot is an open-source **cloud-agnostic ML job orchestrator**. Instead of writing raw Kubernetes manifests, you write a simple YAML describing what you need (GPUs, memory, Docker image, run command) and SkyPilot translates that into K8s pods, manages SSH access, streams logs, and handles teardown. It supports AWS, GCP, Azure, and Kubernetes clusters including Nebius MK8S.

Key SkyPilot concepts:
- **`sky launch`** — provisions cluster + submits job
- **`sky exec`** — submits job to existing cluster
- **`sky logs`** — streams job output
- **`sky queue`** — shows job status
- **`sky down`** — tears down cluster and stops billing
- **`sky api`** — SkyPilot runs a local API server that all CLI commands talk to; restarting it reloads credentials

### PyTorch DDP (Distributed Data Parallel)
DDP is PyTorch's standard strategy for training on multiple GPUs across multiple machines. The key idea:

- **Every GPU gets a full copy of the model** (unlike model parallelism where the model is split)
- The dataset is **sharded** — each GPU processes a different batch simultaneously
- After each backward pass, **gradients are synchronized** across all GPUs using an `AllReduce` operation — every GPU ends up with the same averaged gradient
- All GPUs update their weights identically, so all model copies stay in sync

This means you get **near-linear scaling**: 2 GPUs ≈ 2× the throughput of 1 GPU, with minimal overhead.

### NCCL (NVIDIA Collective Communications Library)
NCCL is the low-level library that implements the `AllReduce` gradient sync. It automatically uses the fastest available transport — NVLink (within a node), InfiniBand, or Ethernet (across nodes). In your setup it used the pod network between the two K8s nodes. Setting `NCCL_DEBUG=INFO` makes it print initialization details — which network interface it chose, bandwidth, topology — and that output was a required deliverable.

### torchrun
`torchrun` is PyTorch's launcher for DDP jobs. It:
- Spawns one process per GPU on each node
- Sets environment variables (`LOCAL_RANK`, `RANK`, `WORLD_SIZE`) that DDP uses internally
- Handles the rendezvous (all processes finding each other) via `MASTER_ADDR` + `MASTER_PORT`

In your `train_job.yaml`, SkyPilot injected `SKYPILOT_NODE_RANK`, `SKYPILOT_NUM_NODES`, and `SKYPILOT_NODE_IPS` which were used to configure `torchrun` correctly on each node.

### The Training Job
- **Model:** `facebook/opt-1.3b` — a 1.3 billion parameter causal language model by Meta
- **Dataset:** `Salesforce/wikitext` (wikitext-2-v1) — standard LM benchmark dataset
- **Task:** Causal language modeling (next token prediction)
- **Steps:** 500
- **Batch size:** 4 per GPU × 2 GPUs = effective batch size 8
- **Precision:** bf16 (bfloat16) — halves memory vs fp32
- **Duration:** ~49 minutes on 2× L40S

---

## What Went Wrong and Why

### 1. Token Authentication Loop
**Problem:** `nebius mk8s v1 cluster get-token` rejected the cluster ID as a positional argument.
**Root cause:** The CLI expected the ID via `-f stdin` as JSON, not as a positional arg.
**Fix:**
```bash
nebius mk8s v1 cluster get-token --format json -f /dev/stdin \
  <<< '{"metadata":{"id":"mk8scluster-e00tkktzk7hv4ecx1c"}}'
```
**Learning:** Always check `--help` carefully; Nebius CLI conventions differ from tools like `gcloud` or `aws`.

### 2. SkyPilot Timezone Crash
**Problem:** `sky status`, `sky queue` crashed with `ValueError: ZoneInfo keys may not be absolute paths, got: /UTC`.
**Root cause:** The Codespace had a misconfigured system timezone (`/UTC` instead of `UTC`), and the `pendulum` library used by SkyPilot couldn't handle the leading slash.
**Fix:** `export TZ=UTC` before any `sky` command. Made permanent with `echo 'export TZ=UTC' >> ~/.bashrc`.

### 3. Head Pod Crash + Lost Weights (First Run)
**Problem:** Woke up to find the head pod in `ContainerStatusUnknown`, SkyPilot showing cluster as `INIT`, job history gone.
**Root cause:** The head pod was evicted/OOM-killed overnight. Kubernetes doesn't automatically restart pods in `ContainerStatusUnknown`.
**Recovery:** Deleted the dead pod, ran `sky start ddp-cluster` to recreate it.
**Bigger problem:** `output_dir="/tmp/output"` — `/tmp` is ephemeral container storage. When the head pod died, all saved checkpoints were wiped. The worker only had `rng_state_1.pth` (rank 1's RNG state), not the actual model weights (which only rank 0 saves).
**Fix:** Changed `output_dir` to `/root/sky_workdir/output` — this path is on the persistent pod volume and survives pod restarts.

### 4. OOM Kill on opt-2.7b
**Problem:** Second run with `facebook/opt-2.7b` ended with exit code 137 (kernel OOM kill) during final model save.
**Root cause:** opt-2.7b in bf16 = ~5.4GB just for weights. Add optimizer states (Adam keeps 2 copies of all parameters = ~10.8GB), activations, and gradients — the L40S's 48GB was exhausted during the final `save_pretrained` which temporarily holds multiple copies of the model in memory.
**Fix:** Reverted to `opt-1.3b` which fits comfortably. To run 2.7b properly you'd need FSDP (Fully Sharded Data Parallel) or gradient checkpointing.

### 5. Missing Training Log
**Problem:** Tore down the cluster with `sky down` before saving logs. `sky logs` requires a live cluster.
**Root cause:** Rushed teardown without saving the NCCL log — a required assignment deliverable worth 20 points.
**Fix:** Need to rerun the job. The correct workflow is always:
```bash
sky logs ddp-cluster 1 > training_log.txt  # BEFORE sky down
sky down ddp-cluster
```

---

## Key Learnings

- **Save logs before teardown** — `sky logs` needs a live cluster; once `sky down` runs, logs are gone
- **Never checkpoint to `/tmp`** — use `/root/sky_workdir/` or object storage (S3/GCS)
- **Rank 0 owns checkpoints** — in standard DDP, only rank 0 (head pod) saves model weights; the head pod is a single point of failure for your checkpoints
- **For true resilience, use object storage** — set `output_dir="s3://your-bucket/checkpoints"` so checkpoints survive any pod failure
- **Token expiry is frequent** — Nebius IAM tokens are short-lived; every new shell session and every SkyPilot API server restart needs a fresh token
- **`export TZ=UTC` is mandatory** in this Codespace — add it to `~/.bashrc` and always set it before `sky api start`
- **opt-2.7b needs memory optimization** — FSDP, gradient checkpointing (`gradient_checkpointing=True` in `TrainingArguments`), or reduced batch size to run safely on L40S
- **`save_steps=50, save_total_limit=3`** is a safe cadence — you lose at most 50 steps of work on any crash, and only keep 3 checkpoints to save disk

---

## Cost Breakdown

From the Nebius billing screenshot, the full session (April 22 – May 22) cost:

| Resource | Usage | Cost |
|---|---|---|
| L40S GPU hours | 40.92 GPU·h | $26.60 |
| L40S vCPU hours | 327.35 vCPU·h | $1.96 |
| L40S RAM | 1309.40 GiB·h | $2.10 |
| Network SSD | 4138.97 GiB·h | $0.40 |
| **Total + VAT** | | **$37.06** |

Effective GPU rate: **~$0.65/GPU·hour** for preemptible L40S — very competitive. The final successful run cost roughly **$2.80** all-in.
