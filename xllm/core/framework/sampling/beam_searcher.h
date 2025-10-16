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

#pragma once
#include <torch/torch.h>
#include <torch/types.h>

#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/beam_search.h"
#endif

namespace xllm {

struct BeamSearchOutput {
  torch::Tensor out_tokens;     // [num_seq]
  torch::Tensor out_logprobs;   // [num_seq]
  torch::Tensor src_seq_idxes;  // [num_seq]
};

class BeamSearcher {
 public:
  BeamSearcher() = default;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // logprobs: [num_seq]
  // top_tokens: [num_seq, top_k]
  // top_logprobs: [num_seq, top_k]
  BeamSearchOutput forward(const torch::Tensor& logprobs,
                           const torch::Tensor& top_tokens,
                           const torch::Tensor& top_logprobs) const;
};

}  // namespace xllm