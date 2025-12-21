# Running GB300 v5.1
Only `deepseek-r1` and `llama3_1-405b` functional. 

## Container for CTK 13.0
GB300 uses CUDA13.0. Please use the container `nvcr.io/nvidia/mlperf/mlperf-inference:mlpinf-v5.1-cuda13.0-pytorch25.08-ubuntu24.04-aarch64-Grace-release` for GB300 mlperf runs.
We recommend you first import this into a sqsh image, and then build incremental images from this. 

```bash
enroot import -o /path/to/base_image.sqsh docker://nvcr.io/nvidia/mlperf/mlperf-inference:mlpinf-v5.1-cuda13.0-pytorch25.08-ubuntu24.04-aarch64-Grace-release
```


## Installing TRTLLM for GB300
GB300 uses sm103 compute capability. We have packaged the trtllm build folders under `/opt` in the GB300 cuda13.0 container. 

To install, start a srun sandbox with:
```bash
$ srun --container-image=/path/to/base_image.sqsh (other_srun_flags) --pty bash
$ cp /opt/gb300-trtllm-builds.tar.gz /your/path
```

untar the tarball. There will be 2 trtllm folders:
- `gb110_bringup_add_mlpinf` - use this for `deepseek-r1`
- `gb110_bringup_enable_sm103` - use this for `llama3_1-405B0`


To install either, use:
```bash
$ srun --container-image=/path/to/base_image.sqsh --container-save=/path/to/image_with_trtllm.sqsh --container-mounts=... --pty bash
$ pip install /path/to/trtllm/folder
```

For `llama3_1-405B` - you may use `scripts/slurm_llm/local_node_instances` artifacts. 
For `deepseek-r1`, please see below section.

## `deepseek-r1`

See `./run-trtllm-serve --help`:
```bash
$ ./run-trtllm-serve.sh --help
Usage: ./run-trtllm-serve.sh [OPTIONS]
Options:
  --mode MODE               Set mode trtllm-(bench/serve). Use serve for mlperf benchmarking
  --trtllm_container_image IMAGE  Set trtllm container image
  --mlperf_container_image IMAGE  Set mlperf container image
  --mlperf_scratch_space space  Set mlperf scratch space
  --scenario Offline|Server
  --num_requests NUM_REQUESTS  Set number of requests (trtllm-bench only)
  --dummy_weights           Use dummy weights (trtllm-bench only)
  --concurrency NUM         Set concurrency level (trtllm-bench only)
  --run_client         To run mlperf harness (trtllm-serve only)
  --nsys PATH            Enable nsys profiling with path, profiling skipped if not set
  --nsys_extra_flags FLAGS  Set extra nsys flags
  --profile_iter_range RANGE              Set iteration range (default: 3000-3200)
  --nsys_name NAME          Set nsys output name
  --nsys_prefix PREFIX      Set nsys prefix command
  --help, -h                Show this help message
```
- This script launches one/multiple `trtllm-serve` instances via `trtllm-llmapi-launch` across nodes in a slurm job allocation. 
- If `--run_client` is specified, then it will launch a single task with `make run_harness`:
    - the trtllm-serve endpoints are launched in order of `scontrol show hostnames $SLURM_NODELIST`
    - warmup is disabled - please make sure to hard code the time sufficient to load engine + start server before the harness job step is launched
- else, waits

### yml config
* since this ignores `make run_llm_server` the config/ dir of the system will be ignored. please set the config accordingly. 
