# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from typing import Any, Dict, List

from nvmitten.configurator import Field

import pathlib


__doc__ = """LLM Harness Fields

Used to control components of the LLM Harness and the TRTLLM backend.
"""

capture_server_logs_dir = Field(
    "capture_server_logs_dir",
    description="Path where server-logs, and harness logs will be placed in",
    from_string=pathlib.Path)

trtllm_lib_path = Field(
    "trtllm_lib_path",
    description="Path to TensorRT-LLM repo root",
    from_string=pathlib.Path)

tensor_parallelism = Field(
    "tensor_parallelism",
    description="Tensor Parallelism",
    from_string=int)

pipeline_parallelism = Field(
    "pipeline_parallelism",
    description="Pipeline Parallelism",
    from_string=int)

moe_expert_parallelism = Field(
    "moe_expert_parallelism",
    description="Expert Parallelism (for MOE models only)",
    from_string=int)

quantizer_outdir = Field(
    "llm_quantizer_outdir",
    description="Path to store the quantized checkpoints for TRTLLM",
    from_string=pathlib.Path)

quantizer_lib_path_override = Field(
    "quantizer_lib_path_override",
    description="Path to the TRTLLM library to use for quantization",
    from_string=pathlib.Path)

llm_gen_config_path = Field(
    "llm_gen_config_path",
    description="The path to the json files storing the generation configs.",
    from_string=pathlib.Path)

use_token_latencies = Field(
    "use_token_latencies",
    description="If enabled, uses token latencies.",
    from_string=bool)

disable_progress_display = Field(
    "disable_progress_display",
    description="Disable LLMHarness progress display rendering (default=False).",
    from_string=bool)

enable_sort = Field(
    "enable_sort",
    description="(Placeholder, not functional) Enable sorting of requests by token length (default=True).",
    from_string=bool)

enable_ttft_latency_tracker = Field(
    "enable_ttft_latency_tracker",
    description="Enable latency tracker for TTFT(default=False).",
    from_string=bool)

triton_num_clients_per_server = Field(
    "triton_num_clients_per_server",
    description="Number of gRPC clients to use (each in separate process space) (default=1)",
    from_string=int)

triton_num_models_per_server = Field(
    "triton_num_models_per_server",
    description="Number of models to load on each Triton server (default=1)",
    from_string=int)


def parse_server_endpoint_list(s: str) -> List[str]:
    """Parse the server endpoints from a string

    Format:
    'host_name_0:port_0,host_name_1:port_1,...'

    Args:
        s (str): String containing server endpoints separated by commas, with each endpoint
                separated by a colon.

    Returns:
        List[str]: List of server endpoints.
    """
    return s.split(',')


triton_server_urls = Field(
    "triton_server_urls",
    description="Triton server URLs (default=0.0.0.0:8001)",
    from_string=parse_server_endpoint_list)


trtllm_server_urls = Field(
    "trtllm_server_urls",
    description="TRTLLM server URLs (default=0.0.0.0:30000)",
    from_string=parse_server_endpoint_list)

server_in_foreground = Field(
    "server_in_foreground",
    description="Run the server process in foreground instead of child zombie procs",
    from_string=bool)


def parse_trtllm_flags(s: str) -> Dict[str, Any]:
    """Parse TRTLLM flags from a string.

    Format:
    'key1:value1,key2:value2,...'

    Args:
        s (str): String containing key-value pairs separated by commas, with each pair
                separated by a colon.

    Returns:
        Dict[str, str]: Dictionary mapping keys to their corresponding values.
    """
    _d = {}
    for kv in s.split(','):
        k, v = kv.split(':', 1)
        _d[k] = v
    return _d


trtllm_build_flags = Field(
    "trtllm_build_flags",
    description="TRTLLM build flags",
    from_string=parse_trtllm_flags)

trtllm_checkpoint_flags = Field(
    "trtllm_checkpoint_flags",
    description="TRTLLM checkpoint flags",
    from_string=parse_trtllm_flags)

trtllm_runtime_flags = Field(
    "trtllm_runtime_flags",
    description="TRTLLM runtime flags",
    from_string=parse_trtllm_flags)

show_steady_state_progress = Field(
    "show_steady_state_progress",
    description="Show steady state information in progress bar",
    from_string=bool)

warmup_iterations = Field(
    "warmup_iterations",
    description="Number of warmup iterations before actual benchmark. (default: auto-determined)",
    from_string=int)

trtllm_disagg_config_path = Field(
    "trtllm_disagg_config_path",
    description="Path to the TRTLLM disaggregated config file. Required and used ONLY in --core_type=trtllm_disagg.",
    from_string=pathlib.Path)
