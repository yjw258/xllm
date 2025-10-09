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
  // out_tokens是top_tokens每行的第一个元素
  output.out_tokens = top_tokens.select(/*dim=*/1, /*index=*/0);
  // out_logprobs是top_logprobs每行的第一个元素
  output.out_logprobs = top_logprobs.select(/*dim=*/1, /*index=*/0);
  output.out_logprobs = output.out_logprobs.add_(logprobs);
  // src_seq_idxes是一个递增的序列，表示每个输出token对应的输入序列索引
  output.src_seq_idxes =
      torch::arange(0,
                    logprobs.size(0),
                    /*step=*/1,
                    torch::dtype(torch::kInt32).device(logprobs.device()));
  // output.out_tokens = torch::empty_like(logprobs, torch::kInt32);
  // output.out_logprobs = torch::empty_like(logprobs, torch::kFloat32);
  // output.src_seq_idxes = torch::empty_like(logprobs, torch::kInt32);
  // xllm_ops::beam_search(logprobs,
  //                       top_tokens,
  //                       top_logprobs,
  //                       output.src_seq_idxes,
  //                       output.out_logprobs,
  //                       output.out_tokens);
  return output;
#else
  LOG(FATAL) << "BeamSearcher is only implemented for NPU backend.";
#endif
}
}  // namespace xllm