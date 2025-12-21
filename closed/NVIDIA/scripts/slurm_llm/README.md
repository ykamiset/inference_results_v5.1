# Multi-node LLM benchmarks using slurm

Meant for launching trtllm server endpoints which spawn LLM inference engines across nodes in the same NVL domain.  
**For GB300, kindly use [scripts/slurm_llm/run-trtllm-serve.sh](./run-trtllm-serve.sh)**

## Terminology

There are two categories of instances:
- `local_node_instance`
  - A single model instance resides on a single node, using number of GPUs <= num GPUS on a node.
  - On GB200 NVL, `llama2-70b`, `llama3_1-405b` and `mixtral-8x7b`
- `cross_node_instance`
  - A single model instance resides on multiple nodes, using number of GPUs > num GPUs on a node.
  - On GB200 NVL, `deepseek-r1` with `EP`, `TP or PP` and `ADP` > 4

Batching modes:
- `IFB`: In-flight batching. May be local- or cross- node
- `Disaggregated`: __Not supported as of now.__

## Prerequisites:
1. You need a `sqsh` image. See [closed/NVIDIA/pyxis/README.md](../../../pyxis/README.md)
```bash
$ make -f Makefile.pyxis build_base_sqsh \
    ARCH=x86_64|aarch64 \ # must match the compute node arch
    INSTALL_TRTLLM=1 INSTALL_LOADGEN=1 \
    EXTRA_SRUN_FLAGS="--container-remap-root"
```

2. Please generate a huggingface access token from your account and substitute it in the prefix scripts [here](./cross_node_instances/prefix.sh) and [here](./local_node_instance/prefix.sh)

## TLDR Repro
Repro scripts may be found at [closed/NVIDIA/scripts/slurm_llm/repros](./repros)

### Local-node instance repro (llama, mixtral)
The target scripts are:
- [`local_node_instance/run_servers_and_harness.sh`](./local_node_instance/run_servers_and_harness.sh):
  - runs a single slurm job with multiple nodes.
  - calls `make run_llm_server` on each node to spawn server(s) per node.
  - calls `make run_harness` on one node only with parsed endpoint urls.
  - Usage can be seen with `local_node_instance/run_servers_and_harness.sh -h`
  - Example usage:
```bash
sbatch \
    <slurm_flags>
    local_node_instance/run_servers_and_harness.sh \
    --mlperf_container_image=/path/to/sqsh \
    --mlperf_scratch_path=/path/to/scratch \
    --trt_engine_artefacts=/path/to/high/vol/storage \ # This is location of engine build only if --trtllm_backend=trt
    --scenario=Offline|Server \
    --benchmark_name=llama2-70b|mixtral-8x7b|llama3_1-405b \
    --core_type=trtllm_endpoint \
    --num_instances_per_node=4 \ # num GPUs a single instance occupies, <= 4
    --system_name=SYSTEM_ID \
    --trtllm_backend=trt|torch
```

### Cross-node instance repro (DS-R1)
The target scripts are:
- [`cross_node_instance/run_server.sh`](./cross_node_instances/run_server.sh):
  - runs a single slurm job with multiple nodes.
  - calls `make run_llm_server` once as a cross node MPI command (with `--nodes=$nodes_per_instance`, `--ntasks-per-node=$gpus_per_node`)
  - On successful server launch, this is exposed as an endpoint at `localhost:30000` of the master node.
  - Once the server is launched, the job perennially `wait`s for servers to be killed.
  - Usage can be seen with `cross_node_instance/run_server.sh -h`
  - Example usage:
```bash
sbatch \
    <slurm_flags>
    cross_node_instances/run_server.sh \
    --mlperf_container_image=/path/to/sqsh \
    --mlperf_scratch_path=/path/to/scratch \
    --trt_engine_artefacts=/path/to/high/vol/storage \ # This is ignored for now
    --scenario=Offline|Server \
    --benchmark_name=deepseek-r1 \
    --core_type=trtllm_endpoint \
    --gpus_per_instance=8  # must be > 4
```
For multiple DP ranks, set the number of nodes accordingly. For example, for GB200 NVL, to run 2 ranks of DS-R1 on 8 GPUs each, use `--nodes=4` (16 GPUs) and `--gpus_per_instance=8`.

Once the server is launched, you must run the benchmark client manually:

- [`cross_node_instance/run_client.sh`](./cross_node_instances/run_client.sh)
  - Runs the `make run_harness` client
  - Please add the `trtllm_server_urls` flag manually to provide the endpoint(s) that the trtllm-serve instances are running on. Usually the enpoints are on every alternate node in slurm nodelist.
  - Usage:
```bash
srun --overlap \
    --jobid=<server_jobid> \
    -w <node_name> \
    --container-name=mlperf_inference-run_llm_server \
    --container-workdir=/work \
    --output=out.txt \
    cross_node_instances/run_client.sh \
    --num_slurm_tasks=$gpus_per_instance \
    --model_path=value \
    --scenario=Offline|Server|Interactive \
    --system_name=<SYSTEM_ID> 
```

Auto-detected system will likely be incorrect in multi-node experiments. System name can be overriden using `SYSTEM_NAME` env variable in both server and client scripts.

If you encounter errors related to huggingface-cli will launching servers, try to add `sleep 15` in `prefix.sh`.
