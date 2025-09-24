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

#include "batch.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <vector>

#include "batch_input_builder.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/batch/mposition.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/params_utils.h"
#include "util/blocking_counter.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

Batch::Batch(Sequence* sequence) { add(sequence); }
Batch::Batch(const std::vector<Sequence*>& sequences) { add(sequences); }

void Batch::add(Sequence* sequence, uint32_t allowed_max_token) {
  CHECK(sequence != nullptr);
  CHECK(!sequence->finished());
  CHECK_GT(allowed_max_token, 0);

  sequences_.push_back(sequence);
  allowed_max_tokens_.push_back(allowed_max_token);

  const auto& input_embedding = sequence->get_input_embedding();
  if (input_embedding.defined())
    input_embeddings_vec_.emplace_back(input_embedding);

  const auto& mm_data = sequence->get_mm_data();
  // if (sequence->is_prefill_stage() &&  mm_data.valid()) // TODO:Compatible
  // With Chunked Prefill
  if ((sequence->kv_state().kv_cache_tokens_num() <
       sequence->num_prompt_tokens()) &&
      mm_data.valid())
    mm_data_vec_.emplace_back(mm_data);
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

ForwardInput Batch::prepare_forward_input(uint32_t num_decoding_tokens,
                                          uint32_t min_decoding_batch_size,
                                          const ModelArgs& args) {
  BatchInputBuilder builder(sequences_,
                            allowed_max_tokens_,
                            input_embeddings_vec_,
                            mm_data_vec_,
                            copy_in_cache_block_infos_,
                            copy_out_cache_block_infos_,
                            swap_cache_block_infos_,
                            &args);
  return builder.build_forward_input(num_decoding_tokens,
                                     min_decoding_batch_size);
}

RawForwardInput Batch::prepare_forward_input(uint32_t start_idx,
                                             uint32_t end_idx) {
  BatchInputBuilder builder(sequences_,
                            allowed_max_tokens_,
                            input_embeddings_vec_,
                            mm_data_vec_,
                            copy_in_cache_block_infos_,
                            copy_out_cache_block_infos_,
                            swap_cache_block_infos_,
                            nullptr);
  return builder.build_raw_forward_input(start_idx, end_idx);
}

void Batch::process_sample_output(const RawForwardOutput& raw_output,
                                  bool enable_schedule_overlap,
                                  ThreadPool* thread_pool) {
  // if raw_output.outputs.size() value is 0,
  // this means all sequences are in prefill stage status.
  auto start_time = std::chrono::high_resolution_clock::now();
  const int64_t num_seqs = raw_output.outputs.size();
  std::vector<int64_t> output_indices;
  output_indices.reserve(sequences_.size());
  int64_t output_idx = 0;
  for (auto* seq : sequences_) {
    if (seq->finished()) {
      output_idx++;
      output_indices.push_back(-1);
      continue;
    }
    if (update_sequence_state(seq, enable_schedule_overlap)) {
      output_indices.push_back(-1);
      continue;
    }
    CHECK_LT(output_idx, num_seqs);
    output_indices.push_back(output_idx);
    output_idx++;
  }

  CHECK_EQ(output_idx, num_seqs);

  // parallel processing function
  auto process_sequences_range = [&](size_t start_seq_idx, size_t end_seq_idx) {
    for (size_t seq_idx = start_seq_idx; seq_idx < end_seq_idx; ++seq_idx) {
      const int64_t output_idx = output_indices[seq_idx];
      if (output_idx == -1) {
        continue;
      }
      const RawSampleOutput raw_sam_output = raw_output.outputs[output_idx];
      const size_t token_size = raw_sam_output.tokens.size();
      for (size_t t_idx = 0; t_idx < token_size; t_idx++) {
        Token t(raw_sam_output.tokens[t_idx].id);
        if (raw_sam_output.tokens[t_idx].logprob.has_value()) {
          t.logprob = raw_sam_output.tokens[t_idx].logprob.value();
        }
        t.top_tokens = raw_sam_output.tokens[t_idx].top_tokens;
        t.top_logprobs = raw_sam_output.tokens[t_idx].top_logprobs;
        // always append a token, maybe true or fake token
        append_token_for_sequence(
            sequences_[seq_idx], t, t_idx, enable_schedule_overlap);

        if (raw_sam_output.tokens[t_idx].embeddings.size() > 0) {
          torch::Tensor embeddings =
              torch::tensor(raw_sam_output.tokens[t_idx].embeddings);
          sequences_[seq_idx]->update_embeddings(embeddings);
        }
        // Speculative decoding may append an EOS token at the beginning,
        // followed by bonus tokens, causing the sequence stopping check to
        // fail.
        if (sequences_[seq_idx]->finished()) {
          break;
        }
      }
    }
  };

  if (thread_pool && sequences_.size() >= thread_pool->size()) {
    // multi-thread processing
    const size_t num_threads = thread_pool->size();
    const size_t sequences_per_thread =
        (sequences_.size() + num_threads - 1) / num_threads;
    BlockingCounter counter(num_threads);

    for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
      const size_t start_seq_idx = thread_idx * sequences_per_thread;
      const size_t end_seq_idx =
          std::min(start_seq_idx + sequences_per_thread, sequences_.size());

      thread_pool->schedule([&, start_seq_idx, end_seq_idx]() {
        process_sequences_range(start_seq_idx, end_seq_idx);
        counter.decrement_count();
      });
    }
    // wait for all threads to complete
    counter.wait();
  } else {
    // single thread processing
    process_sequences_range(0, sequences_.size());
  }

  process_beam_search();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  LOG(INFO) << "Using " << (thread_pool ? thread_pool->size() : 1)
            << " threads to process sample output for " << sequences_.size()
            << " sequences, time taken: " << duration.count() << " us.";
}

void Batch::process_sample_output(const SampleOutput& sample_output,
                                  bool enable_schedule_overlap) {
  if (sample_output.embeddings.defined()) {
    const int64_t num_seqs = sample_output.embeddings.size(0);
    int64_t output_idx = 0;
    for (auto* seq : sequences_) {
      CHECK_LT(output_idx, num_seqs);
      auto cur_seq_embed =
          safe_to(sample_output.embeddings[output_idx++], torch::kFloat32);
      seq->update_embeddings(cur_seq_embed);
    }
  }

  // if sample_output.next_tokens not defined,
  // sample_output.next_tokens.size(0) value is 0,
  // this means all sequences are in prefill stage status.
  const int64_t num_seqs = sample_output.next_tokens.size(0);
  int64_t output_idx = 0;
  for (auto* seq : sequences_) {
    if (seq->finished()) {
      output_idx++;
      continue;
    }
    if (update_sequence_state(seq, enable_schedule_overlap)) {
      continue;
    }
    CHECK_LT(output_idx, num_seqs);

    const auto curr_idx = output_idx++;
    const auto token = build_token(curr_idx,
                                   sample_output.next_tokens,
                                   sample_output.logprobs,
                                   sample_output.top_tokens,
                                   sample_output.top_logprobs);

    // always append a token, maybe true or fake token
    append_token_for_sequence(seq, token, 0, enable_schedule_overlap);
  }
  CHECK_EQ(output_idx, num_seqs);
  process_beam_search();
}

bool Batch::update_sequence_state(Sequence* seq, bool enable_schedule_overlap) {
  // In chunked prefill case, if enable_schedule_overlap, we need the
  // prefill-or-not state of last stage, otherwise, we need the state
  // of current stage.
  if (FLAGS_enable_chunked_prefill) {
    if (!enable_schedule_overlap && seq->is_prefill_stage()) {
      seq->pre_scheduled_step_prefill_queue().push(true);
      // if not enable_schedule_overlap, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
      return true;
    } else if (enable_schedule_overlap &&
               seq->pre_scheduled_step_prefill_queue().front()) {
      seq->pre_scheduled_step_prefill_queue().pop();
      return true;
    }
  }
  return false;
}

void Batch::append_token_for_sequence(Sequence* seq,
                                      const Token& token,
                                      int token_idx,
                                      bool enable_schedule_overlap) {
  // always append a token, maybe true or fake token
  if (!enable_schedule_overlap) {
    seq->append_token(token);
    if (FLAGS_enable_chunked_prefill) {
      seq->pre_scheduled_step_prefill_queue().push(false);
      // if not enable_schedule_overlap, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
    }
  } else {
    // truely update the real token if enable_schedule_overlap
    seq->update_last_step_token(token, token_idx);
    if (FLAGS_enable_chunked_prefill && token_idx == 0) {
      seq->pre_scheduled_step_prefill_queue().pop();
    }
  }
}

void Batch::process_embedding_output(const torch::Tensor& output_embedding) {
  Token token(0);
  if (output_embedding.defined()) {
    int32_t slice_img_index = 0;
    for (auto* seq : sequences_) {  // TODO
      const auto& mm_data = seq->get_mm_data();

      auto pixel_values = mm_data.get_tensor_vec("pixel_values");
      constexpr const int channel = 3;
      int32_t slice_num = 0;
      for (const auto& item : pixel_values) {
        slice_num += item.size(0) / channel;
      }

      auto seq_img_embedding =
          output_embedding
              .slice(0, slice_img_index, slice_img_index + slice_num)
              .clone();
      ;
      slice_img_index += slice_num;
      seq->update_embeddings(seq_img_embedding);
      seq->append_token(token);
    }
  }
}

void Batch::process_beam_search() {
  for (auto* sequence_group : sequence_groups_) {
    sequence_group->process_beam_search();
  }
}
}  // namespace xllm
