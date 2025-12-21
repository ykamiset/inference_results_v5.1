#!/bin/bash


## This script will launch servers on each node (by calling make run_llm_server)
## It will run perennially until each server is manually killed via SIGINT

repo_root=$(git rev-parse --show-toplevel)
username=$(whoami)
output_dir=$repo_root/closed/NVIDIA/build/slurm_logs
trtllm_backend="torch"

usage="sbatch \\
    run_servers.sh \\
    --mlperf_container_image=value \\
    --mlperf_scratch_path=value \\
    --trt_engine_artefacts=value \\
    --scenario=value \\
    --benchmark_name=value \\
    --core_type=value \\
    --trtllm_backend=torch|trt"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mlperf_container_image=*)
            mlperf_container_image="${1#*=}"
            shift
            ;;
        --mlperf_scratch_path=*)
            mlperf_scratch_path="${1#*=}"
            shift
            ;;
        --trt_engine_artefacts=*)
            trt_engine_artefacts="${1#*=}"
            shift
            ;;
        --scenario=*)
            scenario="${1#*=}"
            shift
            ;;
        --benchmark_name=*)
            benchmark_name="${1#*=}"
            shift
            ;;
        --core_type=*)
            core_type="${1#*=}"
            shift
            ;;
        --trtllm_backend=*)
            trtllm_backend="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $usage"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $usage"
            exit 1
            ;;
    esac
done

if [ -z "$mlperf_container_image" ]; then
    echo "Error: mlperf_container_image is not provided"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$scenario" ]; then
    echo "Error: scenario not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$benchmark_name" ]; then
    echo "Error: benchmark_name not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$core_type" ]; then
    echo "Error: core_type not specified"
    echo "Usage: $usage"
    exit 1
fi

set -x

export scenario="${scenario,,}"

export engine_dir="/home/artefacts/engines/mlperf-engine_$benchmark_name.$scenario.$core_type"

export container_workdir="/work"
export actual_workdir="${repo_root}/closed/NVIDIA"
export script_dir=$container_workdir/scripts/slurm_llm

export server_container_name="mlperf_inference"
export container_mount="$actual_workdir:$container_workdir,$mlperf_scratch_path:/home/mlperf_inference_storage,$trt_engine_artefacts:/home/artefacts"

export RUN_ARGS="--benchmarks=$benchmark_name \\
 --scenarios=$scenario \\
 --core_type=$core_type \\
 --engine_dir=$engine_dir \\
 --server_in_foreground"

if [ "$trtllm_backend" == "trt" ]; then
    export RUN_ARGS="$RUN_ARGS --trtllm_runtime_flags=trtllm_backend:cpp"
fi

export server_srun_header="srun --container-image=$mlperf_container_image --container-mounts=$container_mount --container-workdir=$container_workdir"

### If there is a unified file system, build the engine first which all trtllm-serve instances can use
$server_srun_header \
 --container-name=$server_container_name-generate_engines \
 --nodes=1 \
 --output=$output_dir/slurm-$SLURM_JOB_ID-generate_engines.txt \
 --export=script_dir,RUN_ARGS \
 --mpi=pmix \
 /bin/bash -c 'source $script_dir/local_node_instance/prefix.sh && make generate_engines'

### Get node names
node_list=$(scontrol show hostnames $SLURM_NODELIST)

### Launch server on each node
for node in $node_list; do
    $server_srun_header \
        --container-name=$server_container_name-run_llm_server \
        --nodes=1 \
        --export=script_dir,RUN_ARGS \
        -w $node \
        --output=$output_dir/slurm-$SLURM_JOB_ID-$node-server-launch-log.txt \
        /bin/bash -c 'source $script_dir/local_node_instance/prefix.sh && make run_llm_server' &
done

### Launch an overlapping and interactive srun job to run `make run_harness RUN_ARGS="..."`
### srun --jobid=<JOB_ID> --overlap --container-name=mlperf_inference_server --pty bash

### SIGINT the servers manually to exit the job:
### srun --jobid=<JOB_ID> --overlap --container-name=mlperf_inference_server pkill -2 trtllm-serve

wait
