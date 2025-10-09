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

#include "beam_searcher.h"

namespace xllm {
BeamSearchOutput BeamSearcher::forward(
    const torch::Tensor& logprobs,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs) const {
#if defined(USE_NPU)
  BeamSearchOutput output;

  output.out_tokens =
      torch::empty_like(logprobs, torch::kInt32).to("npu").reshape({-1, 1});
  output.out_logprobs =
      torch::empty_like(logprobs, torch::kFloat32).to("npu").reshape({-1, 1});
  output.src_seq_idxes =
      torch::empty_like(logprobs, torch::kInt32).to("npu").reshape({-1, 1});
  xllm_ops::beam_search(logprobs.reshape({-1, 1}),
                        top_tokens,
                        top_logprobs,
                        output.src_seq_idxes,
                        output.out_logprobs,
                        output.out_tokens);
  output.src_seq_idxes = output.src_seq_idxes.reshape({-1});
  output.out_logprobs = output.out_logprobs.reshape({-1});
  output.out_tokens = output.out_tokens.reshape({-1});
  return output;
#else
  LOG(FATAL) << "BeamSearcher is only implemented for NPU backend.";
#endif
}
}  // namespace xllm