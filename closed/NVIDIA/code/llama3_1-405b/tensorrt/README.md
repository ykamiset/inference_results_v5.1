# Llama3.1

## Getting started

### Download Model

Please download model files by following the mlcommons README.md with instructions:

```bash
# following steps: https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b#get-model
# If your company has license concern, please download the model from the following link: https://llama3-1.mlcommons.org/
export CHECKPOINT_PATH=build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout be673f326cab4cd22ccfef76109faf68e41aa5f1
```

### Download and Prepare Data

Please download data files by following the mlcommons README.md with instructions.
Please move the downloaded pickle into expected path and follow steps to run the required data pre-processing:

```bash
# follow: https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b#get-dataset
# to download file: mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl, mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl

# make sure you are in mlperf's container
make prebuild

# move into right directory
mv mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl build/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl
mv mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl build/data/llama3.1-405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl

# run pre-process step for llama3
python3 code/llama3_1-405b/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed-data
```

Make sure after the 2 steps above, you have:

1. model downloaded at: `build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct/`
2. preprocessed data at `build/preprocessed_data/llama3.1-405b/`:

- `build/preprocessed_data/llama3.1-405b/input_lens.npy`
- `build/preprocessed_data/llama3.1-405b/input_ids_padded.npy`
- `build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_512_processed_fp16_calibration/data.parquet`

## Build and run the benchmarks

Please follow the steps below in MLPerf container. Note that the quantization is done in the generate_engines step, so you don't need to do it separately.

```bash
# make sure you are in mlperf's container
make prebuild
# if not, make sure you already built TRTLLM as well as mlperf harnesses needed for GPTJ run.
make build
# If the latest TRTLLM is already built in the steps above, you can expedite the build. You don't need to run make build if loadgen, TRTLLM, and harnesses are already built on the latest commit.
SKIP_TRTLLM_BUILD=1 make build

# Please update configs/llama3_1-405b to include your custom machine config before building the engine
make generate_engines RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=Offline"
make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=Offline"
```

## Interactive Scenario with disagg serving using NVL 72
please refer to the documentation on how to run interactive server scenario using disagg serving on GB200 NVL72 in run_disagg_405B/README.md

