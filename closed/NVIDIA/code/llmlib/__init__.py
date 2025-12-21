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

from code import G_BENCHMARK_MODULES
import contextlib
import gc
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Tuple

import numpy as np

from code.common.workload import Workload
from code.fields import general as general_fields
from code.fields import harness as harness_fields
from code.fields.harness import CoreType
from code.ops.harness import PyHarnessOp
from code.ops.loadgen import LoadgenConfFilesOp
import mlperf_loadgen as lg
from nvmitten.configurator import autoconfigure, bind

from . import fields as llm_fields
from .config import TrtllmDisaggEndpointConfig, TrtllmEndpointConfig, TrtllmHlApiConfig
from .cores import LLMRequest, BackendRegistry
from .factory import LLMServerFactory
from .utils import LazyImport
from .utils import prefix_logger as logging

# we use faulthandler for better stack traces
import faulthandler
faulthandler.enable()


# NOTE(vir):
# Lazy import builder operations to avoid trtllm dependency
# we can use _load() in Operation.immediate_dependencies()
TRTLLMBuilderOp = LazyImport('code.llmlib.builder', 'TRTLLMBuilderOp')
HFQuantizerOp = LazyImport('code.llmlib.builder', 'HFQuantizerOp')


@autoconfigure
@bind(Workload.FIELD)
@bind(llm_fields.disable_progress_display)
@bind(llm_fields.warmup_iterations)
@bind(general_fields.verbose)
@bind(general_fields.verbose_nvtx)
@bind(harness_fields.core_type)
class LLMHarnessOp(PyHarnessOp):
    """LLM Harness Operation"""

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            set: Set of operation classes that this operation depends on.
        """
        return {LoadgenConfFilesOp}

    def __init__(self,
                 *args,
                 workload: Optional[Workload] = None,
                 verbose: bool = False,
                 verbose_nvtx: bool = False,
                 disable_progress_display: bool = False,
                 core_type: Optional[CoreType] = None,
                 warmup_iterations: Optional[int] = None,
                 **kwargs):
        assert workload is not None, "Workload is required"
        assert workload.benchmark.is_llm, "LLMHarnessOp only supports LLM workloads"
        self.wl = workload
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.disable_progress_display = disable_progress_display
        self.warmup_iterations = warmup_iterations

        # use default core type if not provided
        if core_type is None:
            core_type = G_BENCHMARK_MODULES[self.wl.benchmark].load(('DEFAULT_CORE_TYPE',)).DEFAULT_CORE_TYPE
        self.core_type = core_type

        _qsl_t = G_BENCHMARK_MODULES[self.wl.benchmark].load().DataLoader
        super().__init__(_qsl_t, *args, **kwargs)

        self.server = None

    def issue_queries(self, query_samples: List[lg.QuerySample]):
        """Issue queries to the SUT.

        Args:
            query_samples (List[lg.QuerySample]): List of query samples to issue.
        """
        if self._qsl_inst is None:
            logging.warning("QSL instance not set. Skipping issue_queries() call.")
            return

        qsl_ids = [sample.id for sample in query_samples]
        qsl_indices = [sample.index for sample in query_samples]
        input_tokens = self._qsl_inst.get_input_tokens(qsl_indices)
        stop_tokens = self._qsl_inst.get_stop_tokens(qsl_indices)
        queries = [LLMRequest(request_id=qsl_id, input_tokens=inp_tok, stop_tokens=stop_tok)
                   for qsl_id, inp_tok, stop_tok in zip(qsl_ids, input_tokens, stop_tokens)]
        self.server.issue_queries(queries)

    def flush_queries(self):
        """Flush queries from the SUT.

        Args:
            query_samples (List[lg.QuerySample]): List of query samples to flush.
        """
        self.server.flush_queries()

    def get_backend_kwargs(self, dependency_outputs):
        """ Get backend-specific kwargs for LLMServerFactory. """
        return {}

    @contextlib.contextmanager
    def wrap_lg_test(self, scratch_space, dependency_outputs):
        with contextlib.ExitStack() as stack:
            try:
                # Use factory to create LLMServer instance
                self.server = LLMServerFactory.create_server(
                    backend_type=self.core_type,
                    workload=self.wl,
                    disable_progress_display=self.disable_progress_display,
                    verbose=self.verbose,
                    verbose_nvtx=self.verbose_nvtx,
                    **self.get_backend_kwargs(dependency_outputs),
                )

                # Ensure server readiness
                logging.info("Warming up the server...")
                self.server.warm_up(warmup_iters=self.warmup_iterations)
                logging.info("Server warmup completed.")

                # Start metric capture for endpoint-based cores
                stack.enter_context(self.start_external_metric_capture())

                # Disable automatic garbage collection for test-run
                # We run GC opportunistically in LLMServer
                logging.warning(f"Disabled automatic garbage collection for the test run.")
                gc.disable()
                gc.collect()

                yield None
            finally:
                logging.info("Test Complete. Cleaning up LLMServer...")

                # we notify LLMServer to cleanup using stop_work()
                # this is where we dump all stats and metrics to file and cleanup the server and cores
                if self.server:
                    self.server.stop_work()

                # re-enable automatic GC
                gc.enable()

    @contextlib.contextmanager
    def start_external_metric_capture(self):
        """Context manager to start and stop metric capture for TRTLLM endpoint-based cores."""
        metric_capture_process = None

        if self.core_type not in [CoreType.TRTLLM_ENDPOINT, CoreType.TRTLLM_DISAGG]:
            # Not an endpoint-based core, nothing to do
            yield
            return

        try:
            # Get endpoint URLs from the backend configuration
            backend_config = BackendRegistry.get(self.core_type).CONFIG_T()
            endpoint_urls = backend_config.trtllm_endpoint_urls
            assert endpoint_urls, "No endpoint URLs found"

            if endpoint_urls:
                metric_capture_script = Path(__file__).parent / "metric_capture.py"
                assert metric_capture_script.exists(), f"Metric capture script not found at: {metric_capture_script}"

                # Create endpoint_harness_logs subdirectory
                endpoint_logs_dir = Path(backend_config.log_dir) / "endpoint_harness_logs"
                endpoint_logs_dir.mkdir(parents=True, exist_ok=True)

                # Build command-line arguments
                cmd = [
                    sys.executable,
                    str(metric_capture_script),
                    "--endpoints", ','.join(endpoint_urls),
                    "--output-dir", str(endpoint_logs_dir),
                    "--poll-interval", "2.0"
                ]

                # Add server logs directory if capture_server_logs_dir is set
                if backend_config.capture_server_logs_dir is not None:
                    cmd.extend(["--server-logs-dir", str(backend_config.capture_server_logs_dir)])

                logging.info(f"Starting background metric capture for endpoints: {endpoint_urls} ")
                metric_capture_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True  # Run in new session group, independent of parent
                )

            yield

        finally:
            if metric_capture_process is not None:
                # NOTE(vir):
                # metrics_capture.py will capture SIGTERM, stop awaiting
                # anymore iteration stats
                metric_capture_process.terminate()
                logging.info(f"Signaled metric capture process to shutdown ; "
                             f"Files will be saved in: {endpoint_logs_dir}")


@autoconfigure
class TrtllmExecutorClientHarnessOp(LLMHarnessOp):
    """LLM Harness Operation with TRTLLM executor"""

    @classmethod
    def immediate_dependencies(cls):
        return {LoadgenConfFilesOp, TRTLLMBuilderOp._load()}

    def get_backend_kwargs(self, dependency_outputs):
        return super().get_backend_kwargs(dependency_outputs) | {
            'engine_dir': dependency_outputs[TRTLLMBuilderOp._load()]["engine_dir"],
        }


@autoconfigure
class TrtllmServeClientHarnessOp(LLMHarnessOp):
    """LLM Harness Operation with trtllm-serve endpoint based inference"""

    @classmethod
    def immediate_dependencies(cls):
        if TrtllmEndpointConfig().runtime_flags['trtllm_backend'] == 'pytorch':
            return {LoadgenConfFilesOp, HFQuantizerOp._load()}
        else:
            return {LoadgenConfFilesOp, TRTLLMBuilderOp._load()}

    def get_backend_kwargs(self, dependency_outputs):
        if TrtllmEndpointConfig().runtime_flags['trtllm_backend'] == 'pytorch':
            model_path = dependency_outputs[HFQuantizerOp._load()]["quantized_checkpoint_path"]
        else:
            model_path = dependency_outputs[TRTLLMBuilderOp._load()]["engine_dir"]

        return super().get_backend_kwargs(dependency_outputs) | {
            'model_path': model_path
        }


@autoconfigure
class TrtllmDisaggServeClientHarnessOp(TrtllmServeClientHarnessOp):
    """LLM Harness Operation with trtllm-serve-disag endpoint based inference"""

    @classmethod
    def immediate_dependencies(cls):
        assert TrtllmDisaggEndpointConfig().runtime_flags['trtllm_backend'] == 'pytorch'
        return {LoadgenConfFilesOp, HFQuantizerOp._load()}

    def get_backend_kwargs(self, dependency_outputs):
        return super().get_backend_kwargs(dependency_outputs) | {
            'model_path': dependency_outputs[HFQuantizerOp._load()]["quantized_checkpoint_path"],
        }


@autoconfigure
class TrtllmHLApiClientHarnessOp(LLMHarnessOp):
    """LLM Harness Operation with TRT-LLM high-level API"""

    @classmethod
    def immediate_dependencies(cls):
        if TrtllmHlApiConfig().runtime_flags['trtllm_backend'] == 'pytorch':
            return {LoadgenConfFilesOp, HFQuantizerOp._load()}
        else:
            return {LoadgenConfFilesOp, TRTLLMBuilderOp._load()}

    def get_backend_kwargs(self, dependency_outputs):
        return super().get_backend_kwargs(dependency_outputs) | {
            'model_path': dependency_outputs[HFQuantizerOp._load()]["quantized_checkpoint_path"],
        }


@autoconfigure
class TritonClientHarnessOp(LLMHarnessOp):
    """LLM Harness Operation with Triton executor"""
    pass
