#!/bin/bash

## This script will launch servers on each node (by calling make run_llm_server)
## It will then call make run_harness to run the accuracy and performance tests on a single node

repo_root=$(git rev-parse --show-toplevel)
username=$(whoami)

output_dir=$repo_root/closed/NVIDIA/build/slurm_logs
trtllm_backend="torch"

usage="sbatch \\
    run_servers_and_harness.sh \\
    --mlperf_container_image=value \\
    --mlperf_scratch_path=value \\
    --trt_engine_artefacts=value \\
    --scenario=value \\
    --benchmark_name=value \\
    --core_type=value \\
    --num_instances_per_node=value \\
    --system_name=value \\
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
        --num_instances_per_node=*)
            num_instances_per_node="${1#*=}"
            shift
            ;;
        --system_name=*)
            system_name="${1#*=}"
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

if [ -z "$num_instances_per_node" ]; then
    echo "Error: num_instances_per_node not specified"
    echo "Usage: $usage"
    exit 1
fi

if [ -z "$system_name" ]; then
    echo "Warning: system_name not specified - will default to single node system detection"
else
    export SYSTEM_NAME=$system_name
fi

format_hostnames() {
    local num_servers=$1
    shift
    local hosts=("$@")
    local result=""

    for host in "${hosts[@]}"; do
        for ((i=0; i<num_servers; i++)); do
            if [[ -n "$result" ]]; then
                result+=","
            fi
            result+="${host}:$((30000 + i))"
        done
    done

    echo "$result"
}

set -x

export scenario="${scenario,,}"
node_list=$(scontrol show hostnames $SLURM_NODELIST)
export endpoints=$(format_hostnames $num_instances_per_node $node_list)

export engine_dir="/home/artefacts/engines/mlperf-engine_$benchmark_name.$scenario.$core_type"

export container_workdir="/work"
export script_dir="$container_workdir/scripts/slurm_llm"
export actual_workdir="${repo_root}/closed/NVIDIA"

export container_name_suffix="mlperf_inference"
export container_mount="$actual_workdir:$container_workdir,$mlperf_scratch_path:/home/mlperf_inference_storage,$trt_engine_artefacts:/home/artefacts"

export base_run_args="--benchmarks=$benchmark_name \
 --scenarios=$scenario \
 --core_type=$core_type \
 --engine_dir=$engine_dir"

if [ "$trtllm_backend" == "trt" ]; then
    export base_run_args="$base_run_args --trtllm_runtime_flags=trtllm_backend:cpp"
fi

export srun_header="srun --container-image=$mlperf_container_image --container-mounts=$container_mount --container-workdir=$container_workdir"


### If there is a unified file system, build the engine first which all trtllm-serve instances can use
$srun_header \
    --container-name=$container_name_suffix-generate_engines \
    --nodes=1 \
    --export=RUN_ARGS="$base_run_args",script_dir,SYSTEM_NAME \
    --output=$output_dir/slurm-$SLURM_JOB_ID-generate_engines.txt \
    --mpi=pmix \
    /bin/bash -c 'source $script_dir/local_node_instance/prefix.sh && make generate_engines'

### Get node names
node_list=$(scontrol show hostnames $SLURM_NODELIST)

### Launch server on each node
export RUN_ARGS="$base_run_args \
 --server_in_foreground"

for node in $node_list; do
    $srun_header \
        --export=RUN_ARGS,script_dir,SYSTEM_NAME \
        --container-name=$container_name_suffix-run_llm_server \
        --nodes=1 \
        -w $node \
        --output=$output_dir/slurm-$SLURM_JOB_ID-$node-server-launch-log.txt \
        /bin/bash -c 'source $script_dir/local_node_instance/prefix.sh && make run_llm_server' &
done

unset RUN_ARGS
export RUN_ARGS="$base_run_args \
 --trtllm_server_urls=$endpoints \
 --test_mode=AccuracyOnly"

## Accuracy run
$srun_header --overlap \
    --container-name=$container_name_suffix-run_harness_accuracy \
    --nodes=1 \
    --export=RUN_ARGS,script_dir,SYSTEM_NAME \
    --output=$output_dir/slurm-$SLURM_JOB_ID-run_harness_accuracy.txt \
    /bin/bash -c 'source $script_dir/local_node_instance/prefix.sh && make run_harness'

unset RUN_ARGS
export RUN_ARGS="$base_run_args \
 --trtllm_server_urls=$endpoints \
 --test_mode=PerformanceOnly"

## Performance run
$srun_header --overlap \
    --container-name=$container_name_suffix-run_harness_performance \
    --nodes=1 \
    --export=RUN_ARGS,script_dir,SYSTEM_NAME \
    --output=$output_dir/slurm-$SLURM_JOB_ID-run_harness_performance.txt \
    /bin/bash -c 'source $script_dir/local_node_instance/prefix.sh && make run_harness'

srun --overlap \
    --container-name=$container_name_suffix-run_llm_server \
    --ntasks-per-node=1 \
    /bin/bash -c 'pkill -9 make'

wait
