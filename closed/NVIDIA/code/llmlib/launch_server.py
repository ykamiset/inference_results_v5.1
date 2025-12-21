# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import shutil
import subprocess
import sys

from code import G_BENCHMARK_MODULES
from code.common import logging
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.triton.base_config import G_TRITON_BASE_CONFIG
from code.common.workload import EngineIndex, Workload
import code.fields.general as general_fields
from code.fields import harness as harness_fields
from code.llmlib.builder import TRTLLMBuilderOp, HFQuantizerOp
import code.llmlib.fields as llm_fields
from nvmitten.configurator import autoconfigure, bind
from nvmitten.nvidia.accelerator import GPU
from nvmitten.pipeline import Operation

from .config import TritonHarnessConfig, TrtllmEndpointConfig, TrtllmDisaggEndpointConfig


@autoconfigure
@bind(llm_fields.triton_num_models_per_server, "num_models_per_server")
class GenerateTritonConfigOp(Operation):
    def __init__(self, overwrite=True, num_models_per_server=1):
        """
        Args:
            overwrite (bool): Skip generation if repo exists, else overwrite
            num_models_per_server (int): Number of models to load on each Triton server
        """
        self.system_id = DETECTED_SYSTEM.extras['id']
        self.overwrite = overwrite
        self.workload_name = EngineIndex().wl.benchmark.valstr.lower()
        self.model_version = '1'
        self.num_models_per_server = num_models_per_server
        self.scenario = EngineIndex().wl.scenario.valstr.lower()

        harness_config = TritonHarnessConfig()
        self.trtllm_runtime_flags = harness_config.runtime_flags
        self.tp_size = self.trtllm_runtime_flags['tensor_parallelism']
        self.pp_size = self.trtllm_runtime_flags['pipeline_parallelism']

        self.generation_config = harness_config.gen_config
        self.decoupled = self.generation_config.streaming and self.scenario != 'offline'

        gpus = EngineIndex().wl.system.accelerators[GPU]
        logging.info("Accelerators: ")
        for gpu in gpus:
            logging.info(gpu.pretty_string())
        self.num_gpus = len(gpus)

        self.num_gpus_per_model = self.tp_size * self.pp_size
        num_models = self.num_gpus // self.num_gpus_per_model
        self.num_servers = num_models // self.num_models_per_server

        self.model_store_path_prefix = f"/work/build/triton_model_repos/{self.system_id}/{self.workload_name}/{self.scenario}/repo"

    def run(self, scratch_space, dependency_outputs):
        engine_dir = dependency_outputs[TRTLLMBuilderOp]["engine_dir"]
        assert engine_dir.exists(), f"Engine directory not found at: {engine_dir}"
        assert (engine_dir / "rank0.engine").exists(), "Please specify valid --engine_dir in RUN_ARGS, no engine found at {engine_dir}"

        for repo_idx in range(self.num_servers):
            model_path_str = f"{self.model_store_path_prefix}_{repo_idx}"
            model_repo_path = Path(model_path_str)
            if model_repo_path.exists():
                if not self.overwrite:
                    logging.info(f"Directory {model_path_str} exists, skipping regeneration")
                    continue
                logging.info(f"Directory {model_path_str} already exists, this will be overwritten")
                shutil.rmtree(model_path_str)
            else:
                logging.info(f"Creating {model_path_str}")

            triton_model_name_prefix = "model"
            for m_idx in range(self.num_models_per_server):
                model_idx = m_idx + (repo_idx * self.num_models_per_server)
                triton_model_name = f"{triton_model_name_prefix}-{str(model_idx)}"

                gpu_start_idx = model_idx * self.num_gpus_per_model
                gpu_start_idx %= self.num_gpus // self.num_servers
                gpu_idcs = list(range(gpu_start_idx, gpu_start_idx + self.num_gpus_per_model))
                gpu_idcs = list(map(str, gpu_idcs))
                gpu_idcs = ','.join(gpu_idcs)
                model_dir = model_repo_path.joinpath(triton_model_name, self.model_version)
                model_dir.mkdir(parents=True, exist_ok=False)
                config_file_path = model_repo_path.joinpath(triton_model_name, "config.pbtxt")

                logging.info(f"\tUsing TRTLLM engine at {engine_dir}")

                engine_file_name = str(engine_dir)

                with config_file_path.open(mode='w', encoding='utf-8') as f:
                    f.write(G_TRITON_BASE_CONFIG.format(
                        model_name=triton_model_name,
                        is_decoupled=self.decoupled,
                        beam_width=self.generation_config.runtime_beam_width,
                        engine_path=engine_file_name,
                        gpu_device_idx=gpu_idcs,
                        enable_chunked_context=self.trtllm_runtime_flags['enable_chunked_context'],
                        max_num_tokens=self.trtllm_runtime_flags['max_num_tokens']))
            logging.info(f"Generated triton repository at {model_repo_path}")

        return {"triton_server_repos_path": model_repo_path.parent, "num_gpus_per_model": self.num_gpus_per_model}

    @classmethod
    def output_keys(cls):
        return ["triton_server_repos_path", "num_gpus_per_model"]

    @classmethod
    def immediate_dependencies(cls):
        return {TRTLLMBuilderOp}


@autoconfigure
@bind(general_fields.log_dir)
class RunTritonServerOp(Operation):
    """
        Operation to run tritonserver instance(s)
        - Uses triton's `launch_triton_server.py` script at tensorrtllm_backend/scripts/
        - Will depend on GenerateTritonConfigOp. Will start a separate tritonserver instance for each repo, with CUDA_VISIBLE_DEVICES set accordingly
        - Inside each repo, there may be one or more models - all these are exposed via the same tritonserver instance.
        - Returns the list of tritonserver URLs.
    """

    SCRIPT_PATH = Path("/work/build/triton-inference-server/out/tensorrtllm/scripts/launch_triton_server.py")
    TRITON_SERVER_PATH = Path("/opt/tritonserver/bin/tritonserver")

    def __init__(self, log_dir: Path):
        super().__init__()
        self.server_repos_path = None
        self.num_gpus_per_model = None
        self.log_dir = log_dir
        self.harness_config = TritonHarnessConfig()

    def run(self, scratch_space, dependency_outputs):
        self.server_repos_path = Path(dependency_outputs[GenerateTritonConfigOp]["triton_server_repos_path"])
        self.num_gpus_per_model = dependency_outputs[GenerateTritonConfigOp]["num_gpus_per_model"]

        server_repos = [path for path in self.server_repos_path.iterdir() if path.is_dir()]

        # for each path, run a tritonserver instance like:
        # CUDA_VISIBLE_DEVICES=.. python3 /work/build/triton-inference-server/out/tensorrtllm/scripts/launch_triton_server.py \
        # --tritonserver=/opt/tritonserver/bin/tritonserver \
        # --model_repo=<repo> --tensorrt_llm_model_name=<models> --world_size=<tp*pp>
        # --grpc_port=<> --http_port=<> --metrics_port=<>
        grpc_urls = []
        server_idx = 0
        for server_repo in server_repos:
            model_names = [path.name for path in server_repo.iterdir() if path.is_dir()]
            num_models = len(model_names)
            num_gpus_per_server = num_models * self.num_gpus_per_model
            model_names = ','.join(model_names)

            gpu_ids = list(range(server_idx * num_gpus_per_server, (server_idx + 1) * num_gpus_per_server))
            gpu_ids = ','.join(map(str, gpu_ids))

            # we do 1 core per triton server endpoint, and only support DP
            grpc_url = self.harness_config.get_endpoint_url_for_dp(server_idx)

            # Extract ports from the URL
            grpc_port = grpc_url.split(':')[-1]
            http_port = 8080 + server_idx
            metrics_port = 8888 + server_idx

            cmd = ['python3', str(self.SCRIPT_PATH),
                   '--tritonserver', str(self.TRITON_SERVER_PATH),
                   '--model_repo', str(server_repo),
                   '--tensorrt_llm_model_name', str(model_names),
                   '--world_size', str(self.num_gpus_per_model),
                   '--grpc_port', str(grpc_port),
                   '--http_port', str(http_port),
                   '--metrics_port', str(metrics_port),
                   '--log-file', str(self.log_dir / f'tritonserver_log_{server_idx}.log'),
                   '--log',  # weirdly the script needs this to log into file
                   '--force']
            grpc_urls.append(grpc_url)
            logging.info(f"Starting tritonserver instance {server_idx} with command:\n\tCUDA_VISIBLE_DEVICES={gpu_ids} {' '.join(cmd)}")
            subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_ids}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            server_idx += 1

        return {"triton_server_urls": grpc_urls}

    @classmethod
    def output_keys(cls):
        return ["triton_server_urls"]

    @classmethod
    def immediate_dependencies(cls):
        return {GenerateTritonConfigOp}


@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(Workload.FIELD, "workload")
class RunTrtllmServeOp(Operation):
    """ Operation to run trtllm-serve endpoint(s) using trtllm-serve cli """

    def __init__(self,
                 workload: Workload,
                 server_in_foreground: bool = False):
        super().__init__()
        self.wl = workload
        self.blocking = server_in_foreground

        # Merge user flags with defaults
        self.harness_config = TrtllmEndpointConfig()
        self.trtllm_build_flags = self.harness_config.build_flags
        self.trtllm_runtime_flags = self.harness_config.runtime_flags
        self.trtllm_checkpoint_flags = self.harness_config.checkpoint_flags
        self.extra_config_yaml_contents = self.harness_config.extra_config_yaml
        assert self.harness_config.core_type == harness_fields.CoreType.TRTLLM_ENDPOINT

        if self.harness_config.capture_server_logs_dir is not None:
            self.log_dir = self.harness_config.capture_server_logs_dir
        else:
            self.log_dir = self.harness_config.log_dir

        # Get GPU devices
        gpus = DETECTED_SYSTEM.accelerators[GPU]
        self.devices = [gpu.gpu_index for gpu in gpus]

    def run(self, scratch_space, dependency_outputs):
        # 1. Get checkpoint / engine path
        if self.harness_config.runtime_flags['trtllm_backend'] == 'pytorch':
            target_path = dependency_outputs[HFQuantizerOp]["quantized_checkpoint_path"]
            assert Path(target_path).exists(), f"Checkpoint path {target_path} does not exist."

            self.tokenizer_path = target_path
        else:
            target_path = dependency_outputs[TRTLLMBuilderOp]["engine_dir"]
            assert Path(target_path).exists(), f"Engine directory {target_path} does not exist."

            model_repo = G_BENCHMARK_MODULES[self.wl.benchmark].load(('HF_MODEL_REPO',)).HF_MODEL_REPO
            model_name, _ = list(model_repo.items())[0]

            # TODO(vir):
            # re-use local HF model path even in TRT mode
            # instead of passing HF repo
            self.tokenizer_path = model_name

        # 2. Calculate number of trtllm-serve commands to launch on this node
        is_mpi_launch = self.harness_config.is_mpi_task
        gpus_per_server = self.harness_config.get_instance_size()
        launch_endpoints = self.harness_config.trtllm_endpoint_urls

        # 3. Create extra args yaml file
        extra_config_path = Path(self.log_dir) / "trtllm_serve_extra_conf.yaml"
        with extra_config_path.open('w') as f:
            f.write(self.extra_config_yaml_contents)
        logging.info(f"Extra Config YAML Contents:\n{self.extra_config_yaml_contents}")

        # 4. Launch trtllm-serve processes
        server_processes = []
        for index in range(len(launch_endpoints)):
            endpoint_url = launch_endpoints[index]
            endpoint_port = endpoint_url.split(':')[-1]
            env = os.environ.copy()

            cmd = []
            if is_mpi_launch:
                # TODO(vir): remove hardcoded string
                cmd = ['/work/build/TRTLLM/tensorrt_llm/llmapi/trtllm-llmapi-launch', 'trtllm-serve']
                gpu_ids = None

            else:
                # Calculate GPU assignment
                cmd = ['trtllm-serve']
                start_gpu = index * gpus_per_server
                gpu_ids = list(range(start_gpu, start_gpu + gpus_per_server))
                env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

            cmd.extend([
                str(target_path),
                '--host', '0.0.0.0',
                '--port', str(endpoint_port),
                '--extra_llm_api_options', str(extra_config_path.absolute())
            ])

            # Add optional arguments only if they are not None
            optional_args = {
                '--num_postprocess_workers': self.trtllm_runtime_flags['num_postprocess_workers'],
                '--tp_size': self.harness_config.tensor_parallelism,
                '--pp_size': self.harness_config.pipeline_parallelism,
                '--ep_size': self.harness_config.moe_expert_parallelism,
                '--max_num_tokens': self.trtllm_runtime_flags['max_num_tokens'],
                '--max_batch_size': self.trtllm_runtime_flags['max_batch_size'],
                '--max_seq_len': self.trtllm_build_flags['max_seq_len'],
                '--max_beam_width': self.trtllm_build_flags['max_beam_width'],
                '--tokenizer': self.tokenizer_path,
            }

            if self.trtllm_runtime_flags['trtllm_backend'] == 'pytorch':
                optional_args |= {'--backend': self.trtllm_runtime_flags['trtllm_backend']}

            for arg_name, arg_value in optional_args.items():
                if arg_value is not None:
                    cmd.extend([arg_name, str(arg_value)])

            if 'disable_gc' in self.trtllm_runtime_flags and self.trtllm_runtime_flags['disable_gc']:
                cmd.append('--disable_gc')

            log_file = self.log_dir / f'trtllm_serve_{index}.log'
            with open(log_file, 'w') as f:
                f.write(f"Launch ENV:\n{env}\n\n")
                f.write(f"Launch CMD:\n{' '.join(cmd)}\n\n")
                f.write(f"Extra Config:\n{self.extra_config_yaml_contents}\n\n")
                server_processes.append(subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT
                ))

            logging.info(f"Launched {endpoint_url}")
            logging.info(f"  CMD: {' '.join(cmd)}")
            logging.info(f"  GPU devices: {gpu_ids}")
            logging.info(f"  Log file: {log_file}")

        if self.blocking:
            for process in server_processes:
                process.wait()

        return {"trtllm_endpoint_urls": launch_endpoints}

    @classmethod
    def output_keys(cls):
        return ["trtllm_endpoint_urls"]

    @classmethod
    def immediate_dependencies(cls):
        if TrtllmEndpointConfig().runtime_flags['trtllm_backend'] == 'pytorch':
            return {HFQuantizerOp}
        else:
            return {TRTLLMBuilderOp}


@bind(general_fields.log_dir)
@autoconfigure
class GenerateTrtllmDisaggConfigOp(Operation):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir

        self.harness_config = TrtllmDisaggEndpointConfig()
        self.trtllm_build_flags = self.harness_config.build_flags
        self.trtllm_runtime_flags = self.harness_config.runtime_flags
        self.trtllm_checkpoint_flags = self.harness_config.checkpoint_flags
        self.extra_config_yaml_contents = self.harness_config.extra_config_yaml
        assert self.harness_config.core_type == harness_fields.CoreType.TRTLLM_DISAGG

    def run(self, scratch_space, dependency_outputs):
        assert self.trtllm_runtime_flags['trtllm_backend'] == 'pytorch', "Can only be used with pytorch backend"
        orchestrator_mode = not self.harness_config.is_mpi_task

        target_path = dependency_outputs[HFQuantizerOp]["quantized_checkpoint_path"]
        assert Path(target_path).exists(), f"Checkpoint path {target_path} does not exist."

        if self.harness_config.disagg_config_path is not None:
            disagg_config_file_path = Path(self.harness_config.disagg_config_path)
        else:
            disagg_config_file_path = Path(self.log_dir) / "trtllm_serve_disagg_config.yaml"

        if disagg_config_file_path.exists():
            logging.warning(f"Disagg config file {disagg_config_file_path} already exists, overwriting it.")

        # TODO(vir): remove hardcoded path
        gen_config_script_path = Path("/work/build/TRTLLM/docs/source/scripts/disaggregated/gen_yaml.py")
        assert gen_config_script_path.exists(), f"Disagg config script {gen_config_script_path} does not exist"

        logging.info(f"Generating disagg config yaml file at {disagg_config_file_path}")
        gen_script_args = {
            "config": disagg_config_file_path,
            "model": target_path,
            "num_ctx_servers": self.trtllm_runtime_flags['num_ctx_servers'],
            "ctx_tp_size": self.trtllm_runtime_flags['ctx_tp_size'],
            "ctx_batch_size": self.trtllm_runtime_flags['ctx_batch_size'],
            "ctx_max_num_tokens": self.trtllm_runtime_flags['ctx_max_num_tokens'],
            "ctx_enable_attention_dp": self.trtllm_runtime_flags['ctx_enable_attention_dp'],
            "num_gen_servers": self.trtllm_runtime_flags['num_gen_servers'],
            "gen_tp_size": self.trtllm_runtime_flags['gen_tp_size'],
            "gen_batch_size": self.trtllm_runtime_flags['gen_batch_size'],
            "gen_max_num_tokens": self.trtllm_runtime_flags['gen_max_num_tokens'],
            "gen_enable_attention_dp": self.trtllm_runtime_flags['gen_enable_attention_dp'],
            "gen_gpu_memory_fraction": self.trtllm_runtime_flags['gen_gpu_memory_fraction'],
            "worker_start_port": self.trtllm_runtime_flags['worker_start_port'],
            "server_port": self.trtllm_runtime_flags['server_port'],
            "nsys_on": self.trtllm_runtime_flags['nsys_on'],
        }
        if orchestrator_mode:
            # Get GPU devices to determine num_local_gpus
            gpus = DETECTED_SYSTEM.accelerators[GPU]
            num_local_gpus = len(gpus)

            # Use mpirun when in orchestrator mode
            gen_config_cmd = [
                "mpirun",
                "-n", str(num_local_gpus),
                "python3",
                str(gen_config_script_path),
                *[f"--{key}={value}" for key, value in gen_script_args.items() if value],
            ]
        else:
            # Use direct python3 when in leader mode
            gen_config_cmd = [
                "python3",
                str(gen_config_script_path),
                *[f"--{key}={value}" for key, value in gen_script_args.items() if value],
            ]

        # create custom env
        custom_env = os.environ.copy()
        custom_env['TRTLLM_ENABLE_PDL'] = str(int(self.trtllm_runtime_flags['enable_pdl'] == 1))

        if os.environ.get('SLURM_JOB_NODELIST') is None:
            custom_env['SLURM_JOB_NODELIST'] = 'localhost'
            custom_env['SLURM_TASKS_PER_NODE'] = str(int(self.harness_config.global_size))

        if orchestrator_mode:
            logging.info(f"Generating disagg config yaml file with mpirun (orchestrator mode) using {num_local_gpus} GPUs:\n{' '.join(gen_config_cmd)}")
        else:
            logging.info(f"Generating disagg config yaml file with command (MPI mode):\n{' '.join(gen_config_cmd)}")
        subprocess.run(gen_config_cmd, check=True, env=custom_env)

        logging.info(f"Disagg config YAML file generated to: {disagg_config_file_path}")
        return {"disagg_config_file_path": disagg_config_file_path}

    @classmethod
    def output_keys(cls):
        return ["disagg_config_file_path"]

    @classmethod
    def immediate_dependencies(cls):
        return {HFQuantizerOp}


@autoconfigure
@bind(general_fields.log_dir)
class RunTrtllmServeDisaggOp(Operation):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir

        # Merge user flags with defaults
        self.harness_config = TrtllmDisaggEndpointConfig()
        self.trtllm_build_flags = self.harness_config.build_flags
        self.trtllm_runtime_flags = self.harness_config.runtime_flags
        self.trtllm_checkpoint_flags = self.harness_config.checkpoint_flags
        self.extra_config_yaml_contents = self.harness_config.extra_config_yaml
        assert self.harness_config.core_type == harness_fields.CoreType.TRTLLM_DISAGG

        # Get GPU devices
        gpus = DETECTED_SYSTEM.accelerators[GPU]
        self.devices = [gpu.gpu_index for gpu in gpus]

    def run(self, scratch_space, dependency_outputs):
        assert self.trtllm_runtime_flags['trtllm_backend'] == 'pytorch', "Can only be used with pytorch backend"
        orchestrator_mode = not self.harness_config.is_mpi_task

        # we expect disagg config to be specified in run_llm_server leader mode
        self.disagg_config_file_path = self.harness_config.disagg_config_path
        assert Path(self.disagg_config_file_path).exists(), f"Disagg config file {self.disagg_config_file_path} does not exist"

        if orchestrator_mode:
            self._launch_orchestrator_mode()
        else:
            self._launch_leader_mode()

    def _launch_orchestrator_mode(self):
        raise NotImplementedError("Orchestrator mode is not implemented yet. Please use leader mode for now.")

    def _launch_leader_mode(self):
        # TODO(vir): make this a field ?
        launch_type = os.getenv("MLPERF_DISAGG_LAUNCH_TYPE", "worker")
        assert launch_type in ["worker", "leader"], f"Invalid launch type: {launch_type}"

        if launch_type == "leader":
            # launch trtllm-disagg leader process
            leader_log_file_path = self.log_dir / f"leader_log__{os.getenv('HOSTNAME', 'unknown')}_{os.getpid()}.log"

            # Redirect stdout/stderr to log file before exec
            logging.info(f"Starting disagg leader process inline")
            logging.info(f"Log file: {leader_log_file_path}")

            # Open log file and write initial info
            with open(leader_log_file_path, 'a') as log_file:
                log_file.write(f"Launch CMD: trtllm-serve disaggregated -c {self.disagg_config_file_path} -t 1800 -r 1800\n\n")
                log_file.flush()

                # Redirect stdout and stderr to log file
                os.dup2(log_file.fileno(), sys.stdout.fileno())
                os.dup2(log_file.fileno(), sys.stderr.fileno())

            # Replace current process with leader command
            # This preserves the MPI environment
            # TODO(vir): fix hardcoded path
            os.execv("/work/.llm_x86_64/bin/trtllm-serve", [
                "trtllm-serve",
                "disaggregated",
                "--config", str(self.disagg_config_file_path),

                # TODO(vir): change defaults if needed
                "--server_start_timeout", "1800",
                "--request_timeout", "1800"
            ])

        else:  # worker processes
            # launch trtllm-disagg worker processes
            # TODO(vir): remove hardcoded path
            worker_script_path = Path("/work/build/TRTLLM/docs/source/scripts/disaggregated/start_worker.sh")
            worker_log_file_path = self.log_dir / f"worker_log__{os.getenv('HOSTNAME', 'unknown')}_{os.getpid()}.log"

            # Redirect stdout/stderr to log file before exec
            logging.info(f"Starting disagg worker process inline")
            logging.info(f"Log file: {worker_log_file_path}")

            # Open log file and write initial info
            with open(worker_log_file_path, 'a') as log_file:
                log_file.write(f"Launch CMD: bash {worker_script_path} {self.disagg_config_file_path} {self.trtllm_runtime_flags['enable_pdl']}\n\n")
                log_file.flush()

                # Redirect stdout and stderr to log file
                os.dup2(log_file.fileno(), sys.stdout.fileno())
                os.dup2(log_file.fileno(), sys.stderr.fileno())

            # Replace current process with the worker command
            # This preserves the MPI environment
            os.execv("/bin/bash", [
                "bash",
                str(worker_script_path),
                str(self.disagg_config_file_path),
                str(self.trtllm_runtime_flags['enable_pdl'])
            ])

    @classmethod
    def immediate_dependencies(cls):
        return {}
