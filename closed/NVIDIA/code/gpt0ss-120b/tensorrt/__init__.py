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

from code.llmlib import TrtllmServeClientHarnessOp, CoreType, TrtllmHLApiClientHarnessOp
from code.llmlib.launch_server import RunTrtllmServeOp  # noqa: F401
from code.llmlib.builder import LLMComponentEngine, TRTLLMBuilderOp, TRTLLMQuantizerOp, HFQuantizerOp  # noqa: F401
from .constants import GPTOSS_120BComponent as Component
from .dataset import GPTOSS_120BDataset as DataLoader  # noqa: F401

COMPONENT_MAP = {
    Component.GPTOSS_120B: None,
}
VALID_COMPONENT_SETS = {"gpu": [{Component.GPTOSS_120B}]}
DEFAULT_CORE_TYPE = CoreType.TRTLLM_ENDPOINT
HF_MODEL_REPO = {"openai/gpt-oss-120b": 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'}

ComponentEngine = LLMComponentEngine
CalibrateEngineOp = TRTLLMQuantizerOp
EngineBuilderOp = TRTLLMBuilderOp
TrtllmServeBenchmarkHarnessOp = TrtllmServeClientHarnessOp
TrtllmHLApiBenchmarkHarnessOp = TrtllmHLApiClientHarnessOp
