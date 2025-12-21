# LLAMA3.1-405B Disaggregated Multi-Node SLURM Script

This SLURM script launches a disaggregated Dynamo + TensorRT-LLM inference setup for Llama3.1-405B across multiple nodes, separating context processing and generation workloads for optimal performance and resource utilization.

## Overview

The script implements a **disaggregated serving benchmarking** that contains the following component:

- **Context Servers**: Handle prompt processing and KV-cache generation
- **Generation Servers**: Handle token generation and sampling
- **Orchestrator**: Coordinates between context and generation workers
- **MLPerf Loadgen Measuring Client**: Using MLPerf loadgen to measure the backend throughput at latency

## Architecture

18 Nodes

- Dynamo OpenAI Frontend (port 8000), ETCD (port 2379/2380(), NATS (port 4222) on head node
- 14 Prefill Workers, each TP4, one on each node
- 4 Decode Workers, each TP4, one on each node
- MLPerf Benchmark Client (connects to Dynamo OpenAI Frontend on head node)

## Prerequisites

### 1. Hardware Requirements

- **18 nodes** with **4 GB200 GPUs each** (72 total GPUs)
- **SLURM cluster** with Enroot support
- **High-speed interconnect** for multi-node communication
- **Large shared storage** for model weights (~230GB for FP4 quantized model)

### 2. Software Requirements

1. [TRTLLM](https://github.com/NVIDIA/TensorRT-LLM): cd2f9fa62f1c7d9fa35c23518092d45d022f226c
2. [DYNAMO](https://github.com/ai-dynamo/dynamo.git): f9b1757f32d0793abbb625ed4467eab133924671

#### Build Enroot Image

First, import the required enroot image:

```bash
# linux/arm64 for GB200
PLATFORM="arm64"
DYNAMO_COMMIT="f9b1757f32d0793abbb625ed4467eab133924671"
TRTLLM_COMMIT="cd2f9fa62f1c7d9fa35c23518092d45d022f226c"
IMAGE="mlpinf_dynamo_trtllm:${DYNAMO_COMMIT}-${TRTLLM_COMMIT}-${PLATFORM}"

# Clone and build iamge
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
git checkout "${DYNAMO_COMMIT}"
./container/build.sh --framework tensorrtllm --tensorrtllm-commit "${TRTLLM_COMMIT}" --platform "linux/$PLATFORM"
# This produces an image named "dynamo:latest-tensorrtllm", retag to match $IMAGE
docker tag dynamo:latest-tensorrtllm $IMAGE

# TODO(submitter): Update sbatch script to use this IMAGE for deploying Dynamo
# Export docker image to squash file
LOCAL_IMAGE="${IMAGE}.sqsh"
ENROOT_TRANSFER_TIMEOUT=3600 enroot import -o "${LOCAL_IMAGE}" "docker://${IMAGE}"
```

NOTE: `LOCAL_IMAGE` here should replace the one in `disagg_18node.sbatch` for launching Dynamo TRTLLM workers

#### Download Model Checkpoint

Download the FP4 quantized model from HuggingFace:

1. **Install required tools**:

```bash
pip install huggingface_hub
```

2. **Download the model**:

```bash
huggingface-cli download nvidia/Llama-3.1-405B-Instruct-FP4 \
    --local-dir <MLPERF_SCRATCH_PATH>/models/Llama3.1-405B/fp4-quantized-modelopt/llama3.1-405b-instruct-hf-torch-fp4
```

```

**Required directory structure**:

```

<MLPERF_SCRATCH_PATH>/
└── models/
└── Llama3.1-405B/
└── fp4-quantized-modelopt/
└── llama3.1-405b-instruct-hf-torch-fp4/
├── config.json
├── model-\*.safetensors
├── tokenizer.json
└── ...

````

### 3. Model Information

Based on the [NVIDIA Llama-3.1-405B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP4) model:

- **Parameters**: 405 billion parameters
- **Quantization**: FP4

## Configuration

### Step 1:

The script includes several environment variables that are set automatically, typically you don't need to change.

```bash
# Build environment configuration
export ENV="dev"
export BUILD_CONTEXT="aarch64-Grace"
export CUDA_VER="12.9"
export TRT_VER="10.11.0.33"
export MITTEN_HASH="82a930d962ce6bd8aed38cf185a2acfbbfd6b84b"
export INSTALL_RGAT_DEPS="0"
export INSTALL_TRTLLM=""
export INSTALL_LOADGEN=""

# Python path configuration
export PYTHONPATH=/work/.llm_aarch64/lib/python3.12/site-packages/:${PYTHONPATH}
````

### Step 2: Update MLPerf Script Paths

Edit the script variables to match your environment:

```bash
# Step 1: Base image (automatically set)
export BASE_IMAGE="nvcr.io/nvidia/mlperf/mlperf-inference:mlpinf-v5.1-cuda12.9-pytorch25.05-ubuntu24.04-aarch64-Grace"

# Step 2: Scratch path for models and datasets
export MLPERF_SCRATCH_PATH="/your/path/to/mlperf_inference_storage"

# Step3: image is built via instructions in closed/NVIDIA/pyxis/README.md
CONTAINER_IMAGE=/path/to/your/mlperf-inference.sqsh

# Step 4: Container mounts (modify paths as needed)
# TODO(submitter): must mount mlperf-inference repo-root to /mlperf-inference/
REPO_ROOT="/your/path/to/mlperf-inference"
CONTAINER_MOUNTS="${REPO_ROOT}:/mlperf-inference/,${REPO_ROOT}/closed/NVIDIA/:/work,${MLPERF_SCRATCH_PATH}:/home/mlperf_inference_storage"

# Step 5: Working directory
WORK_DIR="$REPO_ROOT/closed/NVIDIA/code/llama3_1-405b/tensorrt/run_disagg_405B"

# Step 6: Model directory (HF quantized FP4 model)
MODEL_DIR=/your/path/to/llama3.1-405b-instruct-hf-torch-fp4
```

### Step 3: Configure Dynamo Disaggregated Setup

The dynamo deployment currently uses a helper script from the Dynamo repo
for multinode slurm environments:
https://github.com/ai-dynamo/dynamo/blob/45934797234430f6c96a439c4be58da501a1391e/components/backends/trtllm/multinode/srun_disaggregated.sh#L1

There are some details on the script and its related environment variables here:
https://github.com/ai-dynamo/dynamo/blob/45934797234430f6c96a439c4be58da501a1391e/components/backends/trtllm/multinode/multinode-examples.md?plain=1#L1

The steps below expose the relevant variables to configure and the `.sbatch` script
calls the helper `srun_disaggregated.sh` script under the hood.

In the future, these steps could probably be simplified by moving the Dynamo helper scripts
into this repository, but for now this approach keeps a single source of truth in the Dynamo
repository.

Current **optimal configuration** (best tokens/s/gpu):

```bash
cat > "${FULL_PREFILL_ENGINE_CONFIG_FILE}" <<EOF
backend: pytorch
tensor_parallel_size: 4
pipeline_parallel_size: 1

max_batch_size: 128
max_num_tokens: 4096
# Dataset peaks at 20010 tokens, and need a multiple of block size (32) here
max_seq_len: 20192

kv_cache_dtype: fp8
disable_overlap_scheduler: true
enable_chunked_prefill: true

kv_cache_config:
  free_gpu_memory_fraction: 0.95
  # Disable kv cache block reuse for consistent benchmarking purposes
  # but in a real service you may want this enabled.
  enable_block_reuse: false

cache_transceiver_config:
  # Dataset peaks at 20010 tokens, and need a multiple of block size (32) here
  max_num_tokens: 20192

scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  context_chunking_policy: FIRST_COME_FIRST_SERVED
EOF

cat > "${FULL_DECODE_ENGINE_CONFIG_FILE}" <<EOF
backend: pytorch
tensor_parallel_size: 4
pipeline_parallel_size: 1

max_batch_size: 512
max_num_tokens: 512
# Dataset peaks at 20010 tokens, and need a multiple of block size (32) here
max_seq_len: 20192

kv_cache_dtype: fp8
disable_overlap_scheduler: false

use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 96
  - 128
  - 150
  - 180
  - 192
  - 224
  - 256
  - 288
  - 352
  - 416
  - 480
  - 512

kv_cache_config:
  free_gpu_memory_fraction: 0.95
  # Disable kv cache block reuse for consistent benchmarking purposes
  # but in a real service you may want this enabled.
  enable_block_reuse: false

cache_transceiver_config:
  # Dataset peaks at 20010 tokens, and need a multiple of block size (32) here
  max_num_tokens: 20192
EOF

## --- Dynamo Deployment Env Vars for dynamo's srun_disaggregated.sh helper script --- ##

# TODO(submitter): update these in the sbatch script
export IMAGE=$LOCAL_IMAGE # image we built in prerequisites section above
export MODEL_PATH="/your/path/to/model/llama3.1-405b-instruct-hf-torch-fp4"
export SERVED_MODEL_NAME="meta-llama/Llama-3.1-405B-Instruct"

# 14P / 4D for NVL72 (double 7P2D config for NVL36)
export NUM_GPUS_PER_NODE=4
# Number of nodes needed per decode worker (ex: 1 for TP4, 2 for TP8, etc.)
export NUM_DECODE_NODES=1
export NUM_DECODE_WORKERS=4
# Number of nodes needed per prefill worker
export NUM_PREFILL_NODES=1
export NUM_PREFILL_WORKERS=14

# Set prefill_first to improve TTFT for 4.5s TTFT latency budget
export DISAGGREGATION_STRATEGY="prefill_first"
```

**Total GPU Usage**: 72 GPUs (14 context servers × 4 GPUs + 2 generation servers × 8 GPUs)

### Step 4: SLURM Configuration

```bash
#SBATCH --job-name=mlperf_405B_test
#SBATCH --nodes=18              # 18 nodes
#SBATCH --partition=36x2-a01r   # Update to your partition
#SBATCH --account=gtc_inference # Update to your account
#SBATCH --time=4:00:00          # 4 hour time limit
#SBATCH --output=./slurm_llama405b_dynamo_disagg_18node_a01r_%j.txt
#SBATCH --container-remap-root
#SBATCH --container-mount-home
```

### Step 5: Server Configuration

The script includes specific server timeout settings:

- **Request timeout**: 7200 seconds
- **Response timeout**: 7200 seconds
- **Warm-up time**: 120 seconds

## How to Run

### 1. Submit the Job

```bash
sbatch disagg_18node.sbatch
```

### 2. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Monitor main job log
tail -f slurm_llama405b_dynamo_disagg_18node_a01r_<jobid>.txt
```

### 3. Check Benchmark Results

```bash
# Performance results
cat ${LOG_DIR}/mlperf_benchmark_performance_<timestamp>.log

# Accuracy results
cat ${LOG_DIR}/mlperf_benchmark_accuracy_<timestamp>.log
```

## Performance Tuning

### Server Target QPS Configuration

For optimal performance, tune the server target QPS in:

```
open/NVIDIA/configs/GB200-NVL72_GB200-NVL-186GB_aarch64x72/Interactive/llama3_1-405b.py
```

**Recommended scaling**:

- **72 GPUs**: `server_target_qps = 15.75`

## Script Execution Flow

1. **Container Initialization**: Start the enroot container across all nodes
2. **Worker Startup**: Launch server, context, and generation workers using `srun_disaggregated.sh`
3. **Warm-up Period**: Wait 120 seconds for workers and server to initialize
4. **Benchmark Execution**: Run both performance and accuracy benchmarks
5. **Result Collection**: Save results to log files

## Output Files

### Logs

- `slurm_llama405b_dynamo_disagg_18node_a01r_<jobid>.txt` - Main SLURM job output
- `${LOG_DIR}/mlperf_benchmark_performance_<timestamp>.log` - Performance benchmark results
- `${LOG_DIR}/mlperf_benchmark_accuracy_<timestamp>.log` - Accuracy benchmark results
