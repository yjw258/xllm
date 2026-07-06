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

# model_types REC supports: OneRec -> "onerec"; LlmRec -> qwen2/qwen3/qwen3_moe.
_REC_MODEL_TYPES = frozenset({"onerec", "qwen2", "qwen3", "qwen3_moe"})

# Default per-request timeout in milliseconds. 0 = no timeout (wait forever).
_DEFAULT_TIMEOUT_MS = 0


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
            # A request is complete once its RequestOutput carries a status.
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
    timeout_ms: int,
) -> RequestOutput:
    # timeout_ms == 0 waits indefinitely; otherwise convert ms to seconds.
    wait_seconds = None if timeout_ms <= 0 else timeout_ms / 1000.0
    if not state.event.wait(wait_seconds):
        raise TimeoutError(f"REC request timed out after {timeout_ms} ms")
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


def _normalize_prompts(prompts):
    """Normalize accepted prompt shapes into (prompt_list, is_token).

      - str / list[str]              -> text prompts,     is_token=False
      - list[int] / list[list[int]]  -> token-id prompts, is_token=True

    Raises TypeError on empty/mixed/unknown shapes.
    """
    if isinstance(prompts, str):
        return [prompts], False

    if not isinstance(prompts, (list, tuple)):
        raise TypeError(
            "prompts must be str, list[str], list[int], or list[list[int]]")

    items = list(prompts)
    if len(items) == 0:
        return [], False

    # A bare list[int] is a single token-id prompt.
    if all(isinstance(x, int) and not isinstance(x, bool) for x in items):
        return [list(items)], True

    is_text = [isinstance(x, str) for x in items]
    is_token = [
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(isinstance(t, int) and not isinstance(t, bool) for t in x)
        for x in items
    ]
    if all(is_text):
        return items, False
    if all(is_token):
        return [list(x) for x in items], True
    raise TypeError(
        "prompts must be all text (str) or all token-id (list[int]); "
        "mixed or malformed prompts are not supported")


class REC:
    """Offline generative-recommendation engine (backend == "rec").

    Mirrors the LLM offline interface: construct with a model path, call
    ``generate`` with text prompts, then ``finish``. Supports LlmRec models
    (qwen2/qwen3/qwen3_moe) and OneRec.

    NOTE: the default startup config enables beam multi-round mode
    (``beam_width=128``, ``max_decode_rounds=3``), so ``generate`` runs through
    the beam pipeline by default. For plain (non-beam) generation, construct
    with ``beam_width=1, max_decode_rounds=0``.
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
        enable_graph: bool = False,
        enable_graph_mode_decode_no_padding: bool = True,
        enable_prefill_piecewise_graph: bool = True,
        enable_shm: bool = False,
        is_local: bool = True,
        input_shm_size: int = 1024,
        output_shm_size: int = 128,
        disable_log_stats: bool = True,
        enable_sleep_mode: bool = False,
        # Rec-specific knobs (no LLM equivalent).
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
        options.server_idx = server_idx
        options.kv_cache_dtype = kv_cache_dtype

        # Set rec runtime knobs (FLAGS_/*Config singletons) and derive the
        # dual-source Options fields (beam_width / enable_graph /
        # rec_worker_max_concurrency).
        configure_rec_runtime(
            options,
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

        # Startup beam config; beam_search() validates requests against these.
        self._startup_beam_width = beam_width
        self._max_decode_rounds = max_decode_rounds

    def finish(self) -> None:
        # Hard-exit the process.
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

        ``weights`` yields ``(name, tensor)`` with HuggingFace names and FULL
        tensors already on this process's NPU device. Tensors are streamed in
        per-layer batches (peak holds ~one layer); on the final batch all
        pipeline models merge in place into the wake_up'd weight buffers (fused
        qkv/gate_up split + NZ conversion done internally). Blocking. Requires
        the engine created with ``enable_sleep_mode=True`` (kManual loader).
        """
        import re

        layer_re = re.compile(r"(.*layers\.\d+)\.")

        # Hold the just-completed batch so the last one can be flagged is_last.
        pending = None

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
                # Layer complete: flush the held batch, then hold the current one.
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

    def _submit_and_collect(self, prompt_list, is_token, params_list, timeout_ms,
                            use_tqdm):
        """Submit each prompt (text or token-id) to the right master entry and
        collect the outputs. Shared by generate() and beam_search()."""
        outputs: List[Optional[RequestOutput]] = [None] * len(prompt_list)
        states = [_OutputState() for _ in prompt_list]
        progress_bar = None
        progress_bar_lock = threading.Lock()
        tqdm_cls = _get_tqdm(use_tqdm)
        if tqdm_cls is not None:
            progress_bar = tqdm_cls(total=len(prompt_list),
                                    desc="Processed prompts")

        def mark_progress() -> None:
            if progress_bar is not None:
                with progress_bar_lock:
                    progress_bar.update(1)

        try:
            for index, prompt in enumerate(prompt_list):
                params = params_list[0 if len(params_list) == 1 else index]
                callback = _make_callback(states[index], mark_progress)
                if is_token:
                    self.master.handle_token_request(prompt, params, callback)
                else:
                    self.master.handle_text_request(prompt, params, callback)

            for index, state in enumerate(states):
                output = _wait_for_output(state, timeout_ms)
                # RequestOutput.prompt is a str field; stringify token-id prompts.
                output.prompt = (str(prompt_list[index]) if is_token
                                 else prompt_list[index])
                outputs[index] = output
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return [output for output in outputs if output is not None]

    def generate(
        self,
        prompts: Union[str, Sequence[str], Sequence[int], Sequence[Sequence[int]]],
        sampling_params: Optional[Union[
            SamplingParams,
            List[SamplingParams],
        ]] = None,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        **kwargs: Any,
    ) -> List[RequestOutput]:
        """Run generation on text or pre-tokenized prompts.

        prompts: str / list[str] (text) or list[int] / list[list[int]]
        (token-id); text and token-id cannot be mixed in one call.

        Accepts either ``sampling_params`` or ``request_params`` (via kwargs) --
        aliases for the same params, cannot both be set. Mirrors LLM.generate.
        """
        request_params = kwargs.pop("request_params", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if request_params is None:
            request_params = sampling_params
        elif sampling_params is not None:
            raise ValueError(
                "sampling_params and request_params cannot both be set"
            )

        prompt_list, is_token = _normalize_prompts(prompts)
        if len(prompt_list) == 0:
            return []

        params_list = _to_rec_request_params_list(request_params, len(prompt_list))
        if len(params_list) not in (1, len(prompt_list)):
            raise ValueError(
                "The number of request_params must be 1 or equal to the "
                "number of prompts."
            )

        return self._submit_and_collect(
            prompt_list, is_token, params_list, timeout_ms, use_tqdm)

    def beam_search(
        self,
        prompts: Union[str, Sequence[str], Sequence[int], Sequence[Sequence[int]]],
        params: Optional[Union[RequestParams, BeamSearchParams]] = None,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        **kwargs: Any,
    ) -> List[RequestOutput]:
        """Beam search on text or pre-tokenized prompts.

        prompts: str / list[str] (text) or list[int] / list[list[int]]
        (token-id); text and token-id cannot be mixed in one call. ``params`` is
        a single RequestParams / BeamSearchParams (mirrors LLM.beam_search).
        Pass ``per_token_logprobs=True`` (kwarg) for one logprob per token.
        """
        per_token_logprobs = kwargs.pop("per_token_logprobs", False)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        # Require max_decode_rounds > 0 and beam_width > 1 at startup.
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

        # Set the fast-path defaults (beam_width, logprobs, top_logprobs, top_k)
        # from the startup beam width, keeping fields the caller set explicitly.
        explicit_fields = (
            params.explicit_fields()
            if isinstance(params, _RequestParamsProxy)
            else set()
        )
        beam_params = to_request_params(params, default_cls=BeamSearchParams)
        beam_params.beam_width = self._startup_beam_width
        if "logprobs" not in explicit_fields:
            beam_params.logprobs = True
        if "top_logprobs" not in explicit_fields and beam_params.top_logprobs == 0:
            beam_params.top_logprobs = self._startup_beam_width
        if "top_k" not in explicit_fields and beam_params.top_k <= 0:
            beam_params.top_k = self._startup_beam_width
        if "per_token_logprobs" not in explicit_fields:
            beam_params.per_token_logprobs = per_token_logprobs

        return self.generate(
            prompts,
            request_params=beam_params,
            timeout_ms=timeout_ms,
            use_tqdm=use_tqdm,
        )
