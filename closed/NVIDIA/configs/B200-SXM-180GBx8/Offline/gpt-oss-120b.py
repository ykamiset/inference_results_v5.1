import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'gptoss-120b': 1024,
        },
        model_fields.input_dtype: 'int32',
        llm_fields.llm_gen_config_path: 'code/gpt0ss-120b/tensorrt/generation_config.json',
        loadgen_fields.min_duration: 600000,
        loadgen_fields.offline_expected_qps: 30,
        model_fields.precision: 'fp8',
        harness_fields.tensor_path: 'build/preprocessed_data/gptoss-120b/v4/perf/',
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 1,
        llm_fields.trtllm_build_flags: {
            'max_beam_width': 1,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'multiple_profiles': 'enable',
            'use_paged_context_fmha': 'disable',
            'use_fp8_context_fmha': 'disable',
            'max_num_tokens': 20480,
            'max_input_len': 2048,
            'max_seq_len': 32000,
            'tokens_per_block': 32,
        },
        # llm_fields.trtllm_checkpoint_flags: {
        #     'kv_cache_dtype': 'fp8',
        # },
        llm_fields.trtllm_runtime_flags: {
            'exclude_input_from_output': True,
            'use_inflight_batching': True,
            'max_num_tokens': 14336,
            'batch_scheduler_policy': 'max_util',
            'context_chunking_policy': 'first_come_first_served',
            'kvcache_free_gpu_mem_frac': 0.95,
            'cuda_graph_batch_sizes': [128, 256, 384, 512, 1024],
            'cuda_graph_padding_enabled': True,
        },
        llm_fields.use_token_latencies: True,
        harness_fields.vboost_slider: 1,
        harness_fields.use_graphs: True,
    },
}

