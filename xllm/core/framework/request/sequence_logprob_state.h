/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <optional>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "request_output.h"

namespace xllm {

class LogprobState {
 public:
  LogprobState(int64_t num_prompt_tokens, size_t capacity);
  ~LogprobState() = default;

  // for generated tokens
  float get_average_logprob(int64_t num_tokens);

  void generate_output_tokens_logprobs(
      size_t start_idx,
      size_t end_idx,
      const Tokenizer& tokenizer,
      std::optional<std::vector<LogProb>>& out_logprobs,
      bool skip_special_tokens,
      const std::vector<int32_t>& tokens);

  void update_logprob(size_t index, const Token& token, int64_t num_top_tokens);

  const std::vector<std::vector<float>>& get_top_logprobs() const {
    return top_logprobs_;
  }

  const std::vector<std::vector<int64_t>>& get_top_tokens() const {
    return top_tokens_;
  }

  const std::vector<std::optional<float>>& get_logprobs() const {
    return logprobs_;
  }

  float get_logprob_sum() const { return logprob_sum_; }

  void set_logprob_sum(float logprob_sum) { logprob_sum_ = logprob_sum; }

 private:
  int64_t num_prompt_tokens_;
  std::vector<std::optional<float>> logprobs_;
  // accumulated log probability of the sequence
  float acc_logprob_ = 0.0;
  int64_t last_acc_token_idx_ = -1;
  // top k log probs
  std::vector<std::vector<int64_t>> top_tokens_;
  std::vector<std::vector<float>> top_logprobs_;
  float logprob_sum_ = 0.0;
};

}  // namespace xllm