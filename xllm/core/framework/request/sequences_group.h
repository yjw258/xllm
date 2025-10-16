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

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "core/framework/sampling/sampling_params.h"
#include "mm_data.h"
#include "sequence.h"
#include "stopping_checker.h"

namespace xllm {

class SequencesGroup {
 public:
  SequencesGroup(const std::string& prompt,
                 const std::vector<int32_t>& prompt_tokens,
                 const torch::Tensor& input_embedding,
                 const MMData& mm_data,
                 const SequenceParams& sequence_params);

  bool finished() const;

  bool expand_sequences(bool share_prefix);

  void generate_outputs(std::vector<SequenceOutput>& outputs,
                        const Tokenizer& tokenizer);

  void process_beam_search();

  std::vector<std::unique_ptr<Sequence>>& sequences() { return sequences_; }

  // get sequence params
  const SequenceParams& get_sequence_params() const { return sequence_params_; }

  int32_t dp_rank() { return sequences_[0]->dp_rank(); }

 private:
  void add();

 private:
  const std::string& prompt_;                  // ref from request
  const std::vector<int32_t>& prompt_tokens_;  // ref from request
  const torch::Tensor& input_embedding_;       // ref from request
  const MMData& mm_data_;                      // ref from request
  SequenceParams sequence_params_;

 private:
  std::vector<std::unique_ptr<Sequence>> sequences_;
};

}  // namespace xllm
