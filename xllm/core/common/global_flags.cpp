/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "global_flags.h"

#include <limits>

#include "brpc/reloadable_flags.h"

// NOTE: related flags should be placed together.

// --- xllm service config ---

DEFINE_string(host, "", "Host name for brpc server.");

DEFINE_int32(port, 8010, "Port for brpc server.");

DEFINE_int32(idle_timeout_s,
             -1,
             "Connection will be closed if there is no read/write operations "
             "during the last `idle_timeout_s`. -1 means wait indefinitely.");

DEFINE_int32(num_threads, 32, "Number of threads to process requests.");

DEFINE_int32(max_concurrency,
             0,
             "Limit number of requests processed in parallel.");

DEFINE_int32(
    max_concurrent_requests,
    0,
    "Maximum number of concurrent requests the xllm service can handle.");

BRPC_VALIDATE_GFLAG(max_concurrent_requests, brpc::NonNegativeInteger);

// --- model config ---

DEFINE_string(model_id, "", "hf model name.");

DEFINE_string(model, "", "Name or path of the huggingface model to use.");

DEFINE_string(backend,
              "llm",
              "Choose the backend model type. 'llm' for text-only, "
              "'vlm' for multimodal (text and images).");

DEFINE_string(task,
              "generate",
              "The task to use the model for(e.g. generate, embed).");

DEFINE_string(devices,
              "npu:0",
              "Devices to run the model on, e.g. npu:0, npu:0,npu:1.");

DEFINE_string(draft_model, "", "draft hf model path to the model file.");

DEFINE_string(draft_devices,
              "npu:0",
              "Devices to run the draft model on, e.g. npu:0, npu:0,npu:1.");

DEFINE_bool(enable_mla,
            false,
            "Whether to enable multi-head latent attention.");

// --- vlm config ---

DEFINE_int32(limit_image_per_prompt,
             4,
             "Maximum number of image per prompt. Only applicable for "
             "multimodal models.");

// --- threading config ---

DEFINE_int32(num_handling_threads, 4, "Number of handling threads.");

DEFINE_int32(num_response_handling_threads,
             4,
             "Number of response handling threads.");

// --- kvcache config ---

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_int64(max_cache_size,
             0,
             "Max gpu memory size for kv cache. Default is 0, which means "
             "cache size is caculated by available memory.");

DEFINE_double(max_memory_utilization,
              0.9,
              "The fraction of GPU memory to be used for model inference, "
              "including model weights and kv cache.");

// --- scheduler config ---

DEFINE_int32(max_tokens_per_batch, 20480, "Max number of tokens per batch.");

DEFINE_int32(max_seqs_per_batch, 256, "Max number of sequences per batch.");

DEFINE_bool(enable_schedule_overlap,
            true,
            "Whether to enable schedule overlap.");

DEFINE_double(prefill_scheduling_memory_usage_threshold,
              0.95,
              "The memory usage threshold during prefill scheduling.");

DEFINE_bool(enable_chunked_prefill, true, "Whether to enable chunked prefill.");

DEFINE_int32(max_tokens_per_chunk_for_prefill,
             512,
             "Max number of token per chunk in prefill stage.");

DEFINE_int32(chunked_match_frequency,
             2,
             "Number of sequence prefix cache match frequency.");

DEFINE_bool(use_zero_evict,
            false,
            "Use ZeroEvictionScheduler but ContinuousScheduler.");

DEFINE_int32(max_decode_token_per_sequence,
             256,
             "Max decode token per sequence.");

// --- parallel config ---

DEFINE_int32(dp_size, 1, "Data parallel size for MLA attention.");

DEFINE_int32(ep_size, 1, "Expert parallel size for MoE model.");

DEFINE_string(
    communication_backend,
    "lccl",
    "NPU communication backend.(e.g. lccl, hccl). When enable dp, use hccl.");

// --- ep load balance config ---

DEFINE_bool(enable_eplb, false, "Whether to use expert parallel load balance.");

DEFINE_int32(redundant_experts_num,
             1,
             "Number of redundant experts on per device.");

DEFINE_int64(eplb_update_interval, 1000, "EPLB update rate.");

DEFINE_double(eplb_update_threshold, 0.8, "EPLB update threshold.");

DEFINE_string(rank_tablefile, "", "ATB HCCL rank table file.");

DEFINE_int32(expert_parallel_degree, 0, "Expert parallel degree.");

// --- profile config ---

DEFINE_bool(enable_profile_step_time,
            false,
            "Whether to enable profile step time.");

DEFINE_bool(enable_profile_token_budget,
            false,
            "Whether to enable profile token budget.");

DEFINE_bool(enable_latency_aware_schedule,
            false,
            "use predicted latency for latency aware schedule.");

DEFINE_int32(profile_max_prompt_length,
             2048,
             "The max prompt length for profile.");

DEFINE_bool(enable_profile_kv_blocks,
            true,
            "true if generate kv cache for profile");

DEFINE_int32(max_global_ttft_ms,
             std::numeric_limits<int32_t>::max(),
             "all requests use single global ttft");

DEFINE_int32(max_global_tpot_ms,
             std::numeric_limits<int32_t>::max(),
             "all requests use single global ttft");

// --- prefix cache config ---

DEFINE_bool(enable_prefix_cache,
            true,
            "Whether to enable the prefix cache for the block manager.");

DEFINE_bool(enable_cache_upload,
            false,
            "Whether to upload cache info to service. This feature is only "
            "available when service routing is enabled.");

DEFINE_uint32(murmur_hash3_seed, 1024, "Default Murmur Hash seed.");

// --- serving on multi-nodes config ---

DEFINE_string(master_node_addr,
              "127.0.0.1:19888",
              "The master address for multi-node distributed serving(e.g. "
              "10.18.1.1:9999).");

DEFINE_int32(nnodes, 1, "The number of multi-nodes.");

DEFINE_int32(node_rank, 0, "The node rank.");

// --- disaggregated prefill and decode config ---

DEFINE_string(xservice_addr, "", "XService server address.");

DEFINE_bool(enable_disagg_pd,
            false,
            "Whether to enable disaggregated prefill and decode execution.");

DEFINE_int32(disagg_pd_port, 7777, "Port for brpc disagg pd server.");

DEFINE_string(instance_role,
              "DEFAULT",
              "The role of instance(e.g. DEFAULT, PREFILL, DECODE).");

DEFINE_string(kv_cache_transfer_type,
              "LlmDataDist",
              "The type of kv cache transfer(e.g. LlmDataDist, HCCL).");

DEFINE_string(kv_cache_transfer_mode,
              "PUSH",
              "The mode of kv cache transfer(e.g. PUSH, PULL).");

DEFINE_string(device_ip, "", "The device ip.");

DEFINE_int32(transfer_listen_port, 26000, "The KVCacheTranfer listen port.");

// --- worker server config ---

DEFINE_int32(max_connect_count,
             40,
             "The max count for worker try to connect to server.");

DEFINE_int32(sleep_time_second,
             3,
             "The sleep time for worker try to connect to server next time.");

// --- function call config ---

DEFINE_string(tool_call_parser,
              "",
              "Specify the parser for handling tool-call interactions(e.g. "
              "qwen25, qwen3, kimi_k2, deepseekv3).");

// --- speculative config ---

DEFINE_int32(num_speculative_tokens, 0, "Number of speculative tokens.");

DEFINE_bool(enable_atb_spec_kernel,
            false,
            "Whether to use ATB speculative kernel.");

// --- block copy config ---

DEFINE_bool(enable_block_copy_kernel,
            true,
            "Whether to use ATB block copy kernel.");

// --- service routing config ---

DEFINE_string(etcd_addr, "", "Etcd adderss for save instance meta info.");

DEFINE_bool(enable_service_routing,
            false,
            "Whether to use xllm service routing.");

DEFINE_double(heart_beat_interval, 0.5, "Heart beat interval.");

DEFINE_int32(etcd_ttl, 3, "Time to live for etcd.");

DEFINE_int32(timeout_ms,
             -1,
             "Max duration of bRPC Channel. -1 means wait indefinitely.");

// --- priority strategy config ---

DEFINE_string(priority_strategy,
              "FCFS",
              "Priority strategy for requests(e.g. FCFS, priority, deadline).");

DEFINE_bool(enable_online_preempt_offline,
            true,
            "Whether to enable online preempt offline.");

// --- kvcache store config ---

DEFINE_double(host_blocks_factor,
              0.0,
              "Host block factor, e.g. host block num = host_blocks_factor * "
              "hbm block num.");

DEFINE_bool(enable_kvcache_store, false, "Whether to use kvcache store.");

DEFINE_string(store_protocol,
              "tcp",
              "KV cache store protocol(e.g. tcp, rdma).");

DEFINE_string(store_master_server_entry,
              "",
              "The address information of the store master service.");

DEFINE_string(store_metadata_connstring,
              "",
              "The address of the kv cache store metadata service.");

// --- for computation communication parallel ---

DEFINE_bool(
    enable_multi_stream_parallel,
    false,
    "Whether to enable computation communication parallel by two streams "
    "and two micro batches in prefill stage.");

// --- for dit ---
DEFINE_int32(max_requests_per_batch, 1, "Max number of request per batch.");

// --- for beam search ---
DEFINE_bool(enable_beam_search_npu,
            false,
            "Whether to enable beam search on npu.");