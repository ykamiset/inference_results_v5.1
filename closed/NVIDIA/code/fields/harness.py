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
from enum import Enum
import pathlib

from nvmitten.configurator import Field

from code.common.constants import AuditTest


__doc__ = """Harness control flags

Settings for benchmark harness runs.
"""


test_run = Field(
    "test_run",
    description="If set, will set min_duration to 1 minute (60000ms). For Offline and Server, min_query_count is set to 1.",
    from_string=bool)

glog_verbosity = Field(
    "glog_verbosity",
    description="Enable verbose output",
    from_string=int)

py_harness = Field(
    "py_harness",
    description=("Chooses to use the new Python-C++ harness. Now only supports LWIS benchmarks: "
                 "If the benchmark is not supported, the default harness will automatically be chosen instead"),
    from_string=bool)

gpu_indices = Field(
    "gpu_indices",
    description="Comma-separated list of gpu indices.",
    from_string=lambda s: [int(dev) for dev in s.split(',')])

numa_config = Field(
    "numa_config",
    description="""Manually set NUMA settings:
    GPU and CPU cores for each NUMA node are specified as a string with the following convention:

    ```
    [Node 0 config]&[Node 1 config]&[Node 2 config]&...
    ```

    Each `[Node n config]` can be configured as `[GPU ID(s)]:[CPU ID(s)]`.
    Each `ID(s)` can be single digit, comma-separated digits, or digits with dash.
    ex.
        "numa_config": "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"

    In this example, example `3:0-15,64-79` means GPU 3 and CPU 0,1,2,...15,64,65,66,...79 are in the same node,
    and since this key is the very first of elements connected with &, they are in node 0.""")

profiler = Field(
    "profiler",
    description="[INTERNAL ONLY] Select profiler to use.",
    argparse_opts={
        "choices": ["nsys", "nvprof", "ncu", "pic-c"]
    })

use_jemalloc = Field(
    "use_jemalloc",
    description="Use libjemalloc.so.2 as the malloc(3) implementation",
    from_string=bool)

audit_test = Field(
    "audit_test",
    description="The audit test to run, if an audit-related action is chosed.",
    from_string=AuditTest.get_match)

no_audit_verify = Field(
    "no_audit_verify",
    description=("If set, skip the verification step for the audit harness. Ignored if not "
                 "running audit harness."),
    from_string=bool)

vboost_slider = Field(
    "vboost_slider",
    description=("Control clock-propagation ratios between GPC-XBAR. "
                 "Look at `nvidia-smi boost-slider --vboost`."),
    from_string=int)

tensor_path = Field(
    "tensor_path",
    description="Path to preprocessed samples in .npy format",
    from_string=str)  # TODO: tensor_path can be a comma-separated list. Handle this later.

gpu_copy_streams = Field(
    "gpu_copy_streams",
    description="Number of copy streams to use for GPU",
    from_string=int)

gpu_inference_streams = Field(
    "gpu_inference_streams",
    description="Number of inference streams to use for GPU.",
    from_string=int)

max_dlas = Field(
    "max_dlas",
    description="Max number of DLAs to use per device",
    from_string=int)

dla_copy_streams = Field(
    "dla_copy_streams",
    description="Number of copy streams to use for DLA",
    from_string=int)

dla_inference_streams = Field(
    "dla_inference_streams",
    description="Number of inference streams to use for DLA",
    from_string=int)

run_infer_on_copy_streams = Field(
    "run_infer_on_copy_streams",
    description="Run inference on copy streams",
    from_string=bool)

warmup_duration = Field(
    "warmup_duration",
    description="Minimum duration to perform warmup for (s)",
    from_string=float)

use_direct_host_access = Field(
    "use_direct_host_access",
    description="Use direct access to host memory for all devices. (SoC Unified Memory)",
    from_string=bool)

use_deque_limit = Field(
    "use_deque_limit",
    description="Use a max number of elements dequed from work queue (LWIS only)",
    from_string=bool)

deque_timeout_usec = Field(
    "deque_timeout_usec",
    description="Timeout in us for deque from work queue (LWIS only)",
    from_string=int)

use_batcher_thread_per_device = Field(
    "use_batcher_thread_per_device",
    description="Enable a separate batcher thread per device",
    from_string=bool)

use_cuda_thread_per_device = Field(
    "use_cuda_thread_per_device",
    description="Enable a separate cuda thread per device",
    from_string=bool)

start_from_device = Field(
    "start_from_device",
    description="If enabled, assumes that inputs start from device memory in QSL",
    from_string=bool)

end_on_device = Field(
    "end_on_device",
    description="Allows output to remain device memory for QuerySampleComplete.",
    from_string=bool)

coalesced_tensor = Field(
    "coalesced_tensor",
    description="Turn on if all the samples are coalesced into one single npy file (LWIS Only)",
    from_string=bool)

map_path = Field(
    "map_path",
    description="Path to map file for samples. Not used if coalesced_tensor is True (BERT, DLRMv2).",
    from_string=pathlib.Path)

assume_contiguous = Field(
    "assume_contiguous",
    description="Assume that the data in a query is already contiguous (LWIS Only)",
    from_string=bool)

complete_threads = Field(
    "complete_threads",
    description="Number of threads per device for sending responses",
    from_string=int)

use_same_context = Field(
    "use_same_context",
    description="Use the same TRT context for all copy streams (shape must be static and gpu_inference_streams must be 1).",
    from_string=bool)

use_spin_wait = Field(
    "use_spin_wait",
    description="Use spin waiting for LWIS. Recommended for single stream",
    from_string=bool)

use_graphs = Field(
    "use_graphs",
    description="Enable CUDA graphs.",
    from_string=bool)

server_num_issue_query_threads = Field(
    "server_num_issue_query_threads",
    description="Number of IssueQuery threads to use for Loadgen in Server scenario",
    from_string=int)


class CoreType(Enum):
    """Enum for supported core types"""
    TRTLLM_EXECUTOR = "trtllm_executor"
    TRTLLM_ENDPOINT = "trtllm_endpoint"
    TRTLLM_DISAGG = "trtllm_disagg"
    TRTLLM_HLAPI = "trtllm_hlapi"
    TRITON_GRPC = "triton_grpc"
    DUMMY = "dummy"

    @classmethod
    def from_string(cls, s: str) -> 'CoreType':
        """Parse core type from string to CoreType enum.

        Args:
            s (str): String representation of core type (e.g., 'trtllm_executor')

        Returns:
            CoreType: The corresponding CoreType enum value

        Raises:
            ValueError: If the string doesn't match any CoreType value
        """
        for core_type in cls:
            if core_type.value == s:
                return core_type
        raise ValueError(f"Invalid core type: {s}. Valid options are: {', '.join([ct.value for ct in cls])}")


core_type = Field(
    "core_type",
    description=f"Type of core to use (choices: {', '.join([ct.value for ct in CoreType])})",
    from_string=CoreType.from_string)
