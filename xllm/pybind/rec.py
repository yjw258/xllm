# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import signal
import sys
import threading
from typing import Any, Callable, List, Optional, Sequence, Union

from xllm_export import (Options, RecMaster, RequestOutput, RequestParams,
                         configure_rec_runtime)

from . import utils
from .errors import ValidationError
from .params import (BeamSearchParams, SamplingParams, _RequestParamsProxy,
                     to_request_params)

_RequestParamsLike = Union[RequestParams, _RequestParamsProxy]
_RequestParamsListLike = Optional[
    Union[_RequestParamsLike, Sequence[_RequestParamsLike]]
]
_OutputCallback = Callable[[RequestOutput], bool]

# model_types REC supports, mirroring core/util/rec_model_utils.h:
#   OneRec -> "onerec"; LlmRec (generative recommendation) -> qwen2/qwen3/qwen3_moe.
_REC_MODEL_TYPES = frozenset({"onerec", "qwen2", "qwen3", "qwen3_moe"})

# Default per-request timeout (seconds) to avoid blocking forever when a request
# never produces a terminal status.
_DEFAULT_TIMEOUT = 300.0


def _get_tqdm(
    use_tqdm: Union[bool, Callable[..., Any]],
) -> Optional[Callable[..., Any]]:
    if not use_tqdm:
        return None
    if callable(use_tqdm):
        return use_tqdm
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "tqdm is required when use_tqdm=True. "
            "Set use_tqdm=False to disable the progress bar."
        ) from exc
    return tqdm


class _OutputState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.output: Optional[RequestOutput] = None
        self.completed = False


def _make_callback(
    state: _OutputState,
    on_complete: Optional[Callable[[], None]] = None,
) -> _OutputCallback:
    def callback(output: RequestOutput) -> bool:
        should_complete = False
        with state.lock:
            state.output = output
            # Align with the c_api rec path (helper.cpp handle_inference_request):
            # a request is complete once its RequestOutput carries a status.
            if not state.completed and output.status is not None:
                state.completed = True
                should_complete = True
        if should_complete:
            if on_complete is not None:
                on_complete()
            state.event.set()
        return True

    return callback


def _wait_for_output(
    state: _OutputState,
    timeout: Optional[float],
) -> RequestOutput:
    if not state.event.wait(timeout):
        raise TimeoutError("REC request timed out")
    output = state.output
    if output is None:
        raise RuntimeError("REC request finished without output")
    if output.status is not None and not output.status.ok:
        raise ValidationError(output.status.code, output.status.message)
    return output


def _to_rec_request_params_list(
    params: _RequestParamsListLike,
    count: int,
) -> List[RequestParams]:
    if params is None:
        return [RequestParams() for _ in range(count)]
    if isinstance(params, (RequestParams, _RequestParamsProxy)):
        return [to_request_params(params, default_cls=SamplingParams)]
    params_list = list(params)
    if len(params_list) == 0:
        return [RequestParams() for _ in range(count)]
    return [
        to_request_params(item, default_cls=SamplingParams)
        for item in params_list
    ]


class REC:
    """Offline generative-recommendation engine (backend == "rec").

    Mirrors the LLM offline interface: construct with a model path, call
    ``generate`` with text prompts, then ``finish``. Supports LlmRec models
    (qwen2/qwen3/qwen3_moe) and OneRec.

    NOTE: the default startup config (mirroring the c_api rec preset) enables
    beam multi-round mode (``beam_width=128``, ``max_decode_rounds=3``), so
    ``generate`` runs through the beam pipeline by default. For plain
    (non-beam) generation, construct with ``beam_width=1, max_decode_rounds=0``.
    """

    def __init__(
        self,
        model: str,
        task: str = "generate",
        devices: str = "npu:0",
        draft_model: str = "",
        draft_devices: str = "",
        block_size: int = 1,
        max_cache_size: int = 1000000,
        max_memory_utilization: float = 0.55,
        enable_prefix_cache: bool = False,  # not supported yet
        max_tokens_per_batch: int = 4096,
        max_seqs_per_batch: int = 2,
        max_tokens_per_chunk_for_prefill: int = 0,
        num_speculative_tokens: int = 0,
        num_request_handling_threads: int = 4,
        communication_backend: str = "lccl",
        rank_tablefile: str = "",
        expert_parallel_degree: int = 0,
        enable_chunked_prefill: bool = False,
        enable_prefill_sp: bool = False,
        master_node_addr: str = "",
        transfer_listen_port: int = 26000,
        nnodes: int = 1,
        node_rank: int = 0,
        dp_size: int = 1,
        ep_size: int = 1,
        instance_name: str = "",
        enable_disagg_pd: bool = False,
        enable_pd_ooc: bool = False,
        enable_schedule_overlap: bool = False,
        kv_cache_transfer_mode: str = "PUSH",
        enable_graph: bool = True,
        enable_graph_mode_decode_no_padding: bool = True,
        enable_prefill_piecewise_graph: bool = True,
        enable_shm: bool = False,
        is_local: bool = True,
        input_shm_size: int = 1024,
        output_shm_size: int = 128,
        disable_log_stats: bool = True,
        enable_sleep_mode: bool = False,
        # Rec-specific knobs (no LLM equivalent). Defaults mirror the c_api rec
        # preset XLLM_INIT_REC_OPTIONS_DEFAULT (xllm/c_api/default.h).
        beam_width: int = 128,
        max_decode_rounds: int = 3,
        rec_worker_max_concurrency: int = 2,
        request_queue_size: int = 16,
        enable_rec_fast_sampler: bool = True,
        enable_rec_prefill_only: bool = False,
        enable_xattention_one_stage: bool = False,
        enable_block_copy_kernel: bool = False,
        enable_topk_sorted: bool = False,
        flashinfer_workspace_buffer_size: int = 128 * 1024 * 1024,
        server_idx: int = 0,
        kv_cache_dtype: str = "auto",
        use_cpp_chat_template: bool = True,
        **kwargs: Any,
    ) -> None:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if not os.path.exists(model):
            raise ValueError(f"model {model} not exists")

        # backend for a rec model_type is "llm" in the registry (only "onerec"
        # registers as "rec"), so validate against the supported rec model_types
        # rather than the registry backend, then force backend="rec" below.
        model_type, _ = utils._infer_model_type_and_backend(model)
        if model_type is None:
            raise ValueError("model_type is required for REC inference")
        if model_type not in _REC_MODEL_TYPES:
            raise ValueError(
                f"REC does not support model_type {model_type!r}; "
                f"supported: {sorted(_REC_MODEL_TYPES)}"
            )
        utils._configure_cpp_chat_template(use_cpp_chat_template, model_type)

        options = Options()
        options.model_path = model
        options.task_type = task
        options.devices = devices
        options.draft_model_path = draft_model
        options.draft_devices = draft_devices
        options.backend = "rec"
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = enable_prefix_cache
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.max_tokens_per_chunk_for_prefill = max_tokens_per_chunk_for_prefill
        options.num_speculative_tokens = num_speculative_tokens
        options.num_request_handling_threads = num_request_handling_threads
        options.communication_backend = communication_backend
        options.rank_tablefile = rank_tablefile
        options.expert_parallel_degree = expert_parallel_degree
        options.enable_chunked_prefill = enable_chunked_prefill
        options.enable_prefill_sp = enable_prefill_sp
        if master_node_addr:
            options.master_node_addr = master_node_addr
        else:
            free_port = utils.get_free_port()
            options.master_node_addr = "127.0.0.1:" + str(free_port)
        options.transfer_listen_port = transfer_listen_port
        options.nnodes = nnodes
        options.node_rank = node_rank
        options.dp_size = dp_size
        options.ep_size = ep_size
        options.instance_name = instance_name
        options.enable_disagg_pd = enable_disagg_pd
        options.enable_pd_ooc = enable_pd_ooc
        options.enable_schedule_overlap = enable_schedule_overlap
        options.kv_cache_transfer_mode = kv_cache_transfer_mode
        options.enable_graph = enable_graph
        options.enable_graph_mode_decode_no_padding = enable_graph_mode_decode_no_padding
        options.enable_prefill_piecewise_graph = enable_prefill_piecewise_graph
        options.enable_offline_inference = True
        options.spawn_worker_path = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )
        options.enable_shm = enable_shm
        options.is_local = is_local
        options.input_shm_size = input_shm_size
        options.output_shm_size = output_shm_size
        options.disable_log_stats = disable_log_stats
        options.enable_sleep_mode = enable_sleep_mode
        options.beam_width = beam_width
        options.rec_worker_max_concurrency = rec_worker_max_concurrency
        options.server_idx = server_idx
        options.kv_cache_dtype = kv_cache_dtype

        # Rec runtime knobs are read from a mix of FLAGS_ and *Config singletons;
        # configure_rec_runtime sets both consistently (mirrors c_api rec init).
        configure_rec_runtime(
            beam_width,
            max_decode_rounds,
            max_seqs_per_batch,
            max_tokens_per_batch,
            max_tokens_per_chunk_for_prefill,
            block_size,
            enable_prefix_cache,
            enable_schedule_overlap,
            enable_chunked_prefill,
            enable_graph,
            enable_prefill_piecewise_graph,
            enable_graph_mode_decode_no_padding,
            enable_rec_fast_sampler,
            enable_rec_prefill_only,
            enable_xattention_one_stage,
            enable_block_copy_kernel,
            enable_topk_sorted,
            rec_worker_max_concurrency,
            request_queue_size,
            flashinfer_workspace_buffer_size,
        )

        self.master = RecMaster(options)
        self.master.run()

        # Startup-level beam config. LlmRec beam search reads beam_width from
        # BeamSearchConfig (startup flag), not from per-request params, and only
        # runs beam search when max_decode_rounds > 0. beam_search() validates
        # against these to avoid silently degrading to single-sequence output.
        self._startup_beam_width = beam_width
        self._max_decode_rounds = max_decode_rounds

    def finish(self) -> None:
        # Offline shutdown. The graceful C++ teardown can hang (rec worker
        # thread pools), so hard-exit. Ascend TBE subprocesses may print
        # "main process disappeared" on exit; that is harmless shutdown noise.
        os._exit(0)

    def sleep(self) -> None:
        """Release device HBM in place (SleepableAllocator) without destroying
        the engine. Requires the engine to be created with
        ``enable_sleep_mode=True``. Call ``wake_up`` to re-acquire the memory.
        """
        self.master.sleep()

    def wake_up(self) -> None:
        """Re-acquire device HBM previously released by ``sleep``."""
        self.master.wake_up()

    def is_sleeping(self) -> bool:
        return self.master.is_sleeping()

    def update_weights(self, weights: Any) -> None:
        """RL weight hot-update from an iterator of (hf_name, torch.Tensor).

        ``weights`` is an iterable/generator yielding ``(name, tensor)`` with
        HuggingFace names (e.g. ``model.layers.0.self_attn.q_proj.weight``) and
        FULL tensors already on this process's NPU device. Tensors are streamed
        in per-layer batches: each batch is staged to host, and on the final
        batch all pipeline models merge in place into the wake_up'd weight
        buffers (fused qkv/gate_up split + NZ conversion done internally).
        Blocking: returns once all weights are written. Requires the engine to
        have been created with ``enable_sleep_mode=True`` (kManual loader).

        Grouping by layer only bounds per-call size; the C++ side accumulates
        across batches in the host staging buffers and merges once at the end.
        Streamed: each layer is dispatched as soon as it is complete and its
        Python tensor refs are then dropped, so the trainer can free that
        layer's device memory before the next layer is pulled (peak holds at
        most one layer, not the whole model).
        """
        import re

        layer_re = re.compile(r"(.*layers\.\d+)\.")

        pending = None  # the just-completed batch, held back so the final
        # batch can be flagged is_last (we only know a batch was the last one
        # after the iterator is exhausted).

        def _flush(batch, is_last):
            self.master.update_weights_from_tensor(batch, is_last)

        cur_key = None
        cur = []
        for name, tensor in weights:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            m = layer_re.match(name)
            key = m.group(1) if m else name.split(".")[0]
            if cur_key is not None and key != cur_key and cur:
                # A layer just completed. Flush the previously-held batch (not
                # last, since we have more), then hold the current one.
                if pending is not None:
                    _flush(pending, is_last=False)
                pending = cur
                cur = []
            cur_key = key
            cur.append((name, tensor))

        # Drain: flush the held batch, then the final one with is_last=True.
        if cur:
            if pending is not None:
                _flush(pending, is_last=False)
            _flush(cur, is_last=True)
        elif pending is not None:
            _flush(pending, is_last=True)

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        sampling_params: _RequestParamsListLike = None,
        request_params: _RequestParamsListLike = None,
        timeout: Optional[float] = _DEFAULT_TIMEOUT,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        **kwargs: Any,
    ) -> List[RequestOutput]:
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if request_params is None:
            request_params = sampling_params
        elif sampling_params is not None:
            raise ValueError(
                "sampling_params and request_params cannot both be set"
            )

        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)
        if not all(isinstance(prompt, str) for prompt in prompt_list):
            raise TypeError("prompts must be str or sequence[str]")
        if len(prompt_list) == 0:
            return []

        params_list = _to_rec_request_params_list(request_params, len(prompt_list))
        if len(params_list) not in (1, len(prompt_list)):
            raise ValueError(
                "The number of request_params must be 1 or equal to the "
                "number of prompts."
            )

        outputs: List[Optional[RequestOutput]] = [None] * len(prompt_list)
        states = [_OutputState() for _ in prompt_list]
        progress_bar = None
        progress_bar_lock = threading.Lock()
        tqdm_cls = _get_tqdm(use_tqdm)
        if tqdm_cls is not None:
            progress_bar = tqdm_cls(total=len(prompt_list), desc="Processed prompts")

        def mark_progress() -> None:
            if progress_bar is not None:
                with progress_bar_lock:
                    progress_bar.update(1)

        try:
            for index, prompt in enumerate(prompt_list):
                params = params_list[0 if len(params_list) == 1 else index]
                callback = _make_callback(states[index], mark_progress)
                self.master.handle_text_request(prompt, params, callback)

            for index, state in enumerate(states):
                output = _wait_for_output(state, timeout)
                output.prompt = prompt_list[index]
                outputs[index] = output
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return [output for output in outputs if output is not None]

    def generate_tokens(
        self,
        prompt_tokens: Sequence[int],
        request_params: Optional[_RequestParamsLike] = None,
        timeout: Optional[float] = _DEFAULT_TIMEOUT,
    ) -> RequestOutput:
        token_ids = list(prompt_tokens)
        if len(token_ids) == 0:
            raise ValueError("prompt_tokens cannot be empty")
        params = _to_rec_request_params_list(request_params, 1)[0]
        state = _OutputState()
        self.master.handle_token_request(token_ids, params, _make_callback(state))
        return _wait_for_output(state, timeout)

    def beam_search(
        self,
        prompts: Sequence[Sequence[int]],
        beam_width: int = 4,
        max_tokens: int = 512,
        request_params: _RequestParamsListLike = None,
        timeout: Optional[float] = _DEFAULT_TIMEOUT,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
    ) -> List[RequestOutput]:
        # LlmRec beam search only runs when the engine was started with
        # max_decode_rounds > 0, and the effective beam width comes from the
        # startup BeamSearchConfig, not per-request params. Reject configs that
        # would silently produce single-sequence output instead of beams.
        if self._max_decode_rounds < 1:
            raise ValueError(
                "beam_search requires the engine to be started with "
                "max_decode_rounds > 0; construct REC(..., "
                f"max_decode_rounds=M>0) (got {self._max_decode_rounds})."
            )
        if self._startup_beam_width < 2:
            raise ValueError(
                "beam_search requires the engine to be started with "
                "beam_width > 1; construct REC(..., beam_width=N>1) "
                f"(got {self._startup_beam_width})."
            )
        if beam_width != self._startup_beam_width:
            raise ValueError(
                f"beam_width={beam_width} does not match the startup "
                f"beam_width={self._startup_beam_width}; LlmRec beam search "
                "uses the startup beam_width. Restart REC with the desired "
                "beam_width or pass the matching value."
            )

        prompt_list = [list(prompt) for prompt in prompts]
        if len(prompt_list) == 0:
            return []
        if not all(len(tokens) > 0 for tokens in prompt_list):
            raise ValueError("each prompt token list must be non-empty")

        if request_params is None:
            request_params = BeamSearchParams(
                beam_width=beam_width, max_tokens=max_tokens
            )
        params_list = _to_rec_request_params_list(request_params, len(prompt_list))
        if len(params_list) not in (1, len(prompt_list)):
            raise ValueError(
                "The number of request_params must be 1 or equal to the "
                "number of prompts."
            )

        outputs: List[Optional[RequestOutput]] = [None] * len(prompt_list)
        states = [_OutputState() for _ in prompt_list]
        progress_bar = None
        progress_bar_lock = threading.Lock()
        tqdm_cls = _get_tqdm(use_tqdm)
        if tqdm_cls is not None:
            progress_bar = tqdm_cls(total=len(prompt_list), desc="Processed prompts")

        def mark_progress() -> None:
            if progress_bar is not None:
                with progress_bar_lock:
                    progress_bar.update(1)

        try:
            for index, tokens in enumerate(prompt_list):
                params = params_list[0 if len(params_list) == 1 else index]
                callback = _make_callback(states[index], mark_progress)
                self.master.handle_token_request(tokens, params, callback)

            for index, state in enumerate(states):
                output = _wait_for_output(state, timeout)
                output.prompt = prompt_list[index]
                outputs[index] = output
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return [output for output in outputs if output is not None]
