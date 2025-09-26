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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_beam_search.h"
#include "beam_search.h"

#define CHECK_ACL_SUCCESS(expr, msg) \
  do {                               \
    auto _ret = (expr);              \
    if (_ret != ACL_SUCCESS) {       \
      LOG(ERROR) << msg;             \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)
namespace xllm_ops {

void beam_search(torch::Tensor& token_ids,
                 const torch::Tensor& log_probs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_probs,
                 torch::Tensor& src_idx) {
  xllm_ops_utils::check_tensor(token_ids, "token_ids", "beam_search");
  xllm_ops_utils::check_tensor(log_probs, "log_probs", "beam_search");
  xllm_ops_utils::check_tensor(top_tokens, "top_tokens", "beam_search");
  xllm_ops_utils::check_tensor(top_probs, "top_probs", "beam_search");
  aclTensor* token_ids_ids = nullptr;
  aclTensor* log_probs_ids = nullptr;
  aclTensor* top_tokens_ids = nullptr;
  aclTensor* top_probs_ids = nullptr;
  aclTensor* src_idx_ids = nullptr;
  int32_t device_id = token_ids.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  xllm_ops_utils::create_acltensor(&token_ids_ids, token_ids);
  xllm_ops_utils::create_acltensor(&log_probs_ids, log_probs);
  xllm_ops_utils::create_acltensor(&top_tokens_ids, top_tokens);
  xllm_ops_utils::create_acltensor(&top_probs_ids, top_probs);
  xllm_ops_utils::create_acltensor(&src_idx_ids, src_idx);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(aclnnBeamSearchGetWorkspaceSize(token_ids_ids,
                                                    log_probs_ids,
                                                    top_tokens_ids,
                                                    top_probs_ids,
                                                    token_ids_ids,
                                                    src_idx_ids,
                                                    &workspace_size,
                                                    &executor),
                    "beam_search: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "beam_search: failed to allocate workspace");
  }
  CHECK_ACL_SUCCESS(
      aclnnBeamSearch(workspace_addr, workspace_size, executor, stream),
      "beam_search: failed to perform beam search");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "beam_search: failed to synchronize stream");
  aclDestroyTensor(token_ids_ids);
  aclDestroyTensor(log_probs_ids);
  aclDestroyTensor(top_tokens_ids);
  aclDestroyTensor(top_probs_ids);
  aclDestroyTensor(src_idx_ids);
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "beam_search: failed to free workspace");
  }
}
}  // namespace xllm_ops