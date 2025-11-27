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
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "smem.h"
#include "smem_shm.h"
#include "smem_trans.h"

namespace xllm {

typedef enum {
  MF_PULL_WEIGHT = 1U << 0,
  MF_PUSH_WEIGHT = 1U << 1
} mf_weight_transfer_direction_t;

class MfWeightTransfer {
 public:
  MfWeightTransfer(int device_id,
                   int rank_id,
                   int rank_size,
                   const std::string& ip_port)
      : device_id_(device_id),
        rank_id_(rank_id),
        rank_size_(rank_size),
        ip_port_(ip_port) {
    global_session_id_.resize(rank_size);
    for (int i = 0; i < rank_size_; ++i) {
      global_session_id_[i] = "127.0.0.1:" + std::to_string(10000 + i);
    }

    const uint32_t LOG_LEVEL_WARNING = 2;
    smem_set_log_level(LOG_LEVEL_WARNING);
    auto ret = smem_init(0);
    CHECK_EQ(ret, 0) << "smem init failed, ret:" << ret << ", rank:" << rank_id;
    if (rank_id == 0) {
      ret = smem_create_config_store(ip_port.c_str());
      CHECK_EQ(ret, 0) << "smem create config store failed, ret:" << ret
                       << ", rank:" << rank_id;
    }

    smem_trans_config_init(&trans_config_);
    if (rank_id == 0) {
      trans_config_.role = SMEM_TRANS_SENDER;
    } else {
      trans_config_.role = SMEM_TRANS_RECEIVER;
    }

    session_id_ = global_session_id_[rank_id];
    trans_config_.deviceId = device_id;
    trans_config_.dataOpType = SMEMB_DATA_OP_SDMA;

    ret = smem_trans_init(&trans_config_);
    CHECK_EQ(ret, 0) << "smem trans init failed, ret:" << ret
                     << ", rank:" << rank_id;

    trans_handle_ =
        smem_trans_create(ip_port.c_str(), session_id_.c_str(), &trans_config_);
    if (trans_handle_ == nullptr) {
      LOG(ERROR) << "smem trans create failed, ret:" << ret
                 << ", rank:" << rank_id;
    }

    LOG(INFO) << "rank id: " << rank_id << ", session id: " << session_id_;
    LOG(INFO) << "rank id: " << rank_id << ", smem trans create done.";

    (void)smem_shm_config_init(&shm_config_);
    shm_config_.startConfigStoreServer = false;

    ret = smem_shm_init(
        ip_port.c_str(), rank_size, rank_id, device_id, &shm_config_);
    CHECK_EQ(ret, 0) << "smem shm init failed, ret:" << ret
                     << ", rank:" << rank_id;

    shm_handle_ = smem_shm_create(0,
                                  rank_size,
                                  rank_id,
                                  1024ULL * 1024 * 2,  // 2MB
                                  SMEMS_DATA_OP_MTE,
                                  0,
                                  &shm_gva_);
    if (shm_handle_ == nullptr) {
      LOG(ERROR) << "smem shm create failed, ret:" << ret
                 << ", rank:" << rank_id;
    }

    barrier();
  }

  ~MfWeightTransfer() {
    smem_shm_destroy(shm_handle_, 0);
    smem_shm_uninit(0);
    smem_trans_destroy(trans_handle_, 0);
    smem_trans_uninit(0);
    smem_uninit();
  }

  void init(std::vector<void*>& weight_addrs,
            std::vector<void*>& weight_buf_addrs,
            std::vector<size_t>& weight_sizes) {
    weight_addrs_ = std::move(weight_addrs);
    weight_buf_addrs_ = std::move(weight_buf_addrs);
    weight_sizes_ = std::move(weight_sizes);

    gather_peer_addrs();
    register_weight();
    barrier();
  }

  void set_weight_addrs(std::vector<void*>& weight_addrs) {
    weight_addrs_ = std::move(weight_addrs);
  }

  void set_weight_sizes(std::vector<size_t>& weight_sizes) {
    weight_sizes_ = std::move(weight_sizes);
  }

  void set_weight_buf_addrs(std::vector<void*>& weight_buf_addrs) {
    weight_buf_addrs_ = std::move(weight_buf_addrs);
  }

  void gather_peer_addrs() {
    global_weight_addrs_.resize(rank_size_);
    global_weight_buf_addrs_.resize(rank_size_);
    int ret = 0;
    for (int i = 0; i < rank_size_; ++i) {
      global_weight_addrs_[i].resize(weight_sizes_.size());
      global_weight_buf_addrs_[i].resize(weight_sizes_.size());
    }
    std::vector<void*> weight_addrs_gather(rank_size_);
    // gather weight_addrs
    for (size_t i = 0; i < weight_addrs_.size(); i++) {
      ret = smem_shm_control_allgather(shm_handle_,
                                       (char*)&weight_addrs_[i],
                                       sizeof(void*),
                                       (char*)weight_addrs_gather.data(),
                                       sizeof(void*) * rank_size_);
      CHECK_EQ(ret, 0) << "smem shm control allgather failed, ret:" << ret
                       << ", rank:" << rank_id_;
      barrier();
      for (int j = 0; j < rank_size_; ++j) {
        global_weight_addrs_[j][i] = weight_addrs_gather[j];
      }
    }

    // gather weight_buf_addrs
    std::vector<void*> weight_buf_addrs_gather(rank_size_);
    for (size_t i = 0; i < weight_buf_addrs_.size(); i++) {
      ret = smem_shm_control_allgather(shm_handle_,
                                       (char*)&weight_buf_addrs_[i],
                                       sizeof(void*),
                                       (char*)weight_buf_addrs_gather.data(),
                                       sizeof(void*) * rank_size_);
      CHECK_EQ(ret, 0) << "smem shm control allgather failed, ret:" << ret
                       << ", rank:" << rank_id_;
      barrier();
      for (int j = 0; j < rank_size_; ++j) {
        global_weight_buf_addrs_[j][i] = weight_buf_addrs_gather[j];
      }
    }
  }

  void register_weight() {
    int ret = smem_trans_batch_register_mem(
        trans_handle_,
        weight_addrs_.data(),
        weight_sizes_.data(),
        static_cast<uint32_t>(weight_sizes_.size()),
        0);
    CHECK_EQ(ret, 0) << "smem trans batch register weight mem failed, ret:"
                     << ret << ", rank:" << rank_id_;
    ret = smem_trans_batch_register_mem(
        trans_handle_,
        weight_buf_addrs_.data(),
        weight_sizes_.data(),
        static_cast<uint32_t>(weight_sizes_.size()),
        0);
    CHECK_EQ(ret, 0)
        << "smem trans batch register weight buffer mem failed, ret:" << ret
        << ", rank:" << rank_id_;
    std::this_thread::sleep_for(
        std::chrono::seconds(10UL));  // wait for register
  }

  void transfer_weight(int32_t global_rank_id,
                       mf_weight_transfer_direction_t direction,
                       bool to_buffer = false,
                       bool batched = false) {
    int ret = 0;
    auto& remote_weight_addrs = to_buffer
                                    ? global_weight_buf_addrs_[global_rank_id]
                                    : global_weight_addrs_[global_rank_id];
    auto start = std::chrono::high_resolution_clock::now();
    if (direction == mf_weight_transfer_direction_t::MF_PULL_WEIGHT) {
      if (batched) {
        ret = smem_trans_batch_read(
            trans_handle_,
            weight_addrs_.data(),
            global_session_id_[global_rank_id].c_str(),
            const_cast<const void**>(remote_weight_addrs.data()),
            weight_sizes_.data(),
            static_cast<uint32_t>(weight_sizes_.size()));
        CHECK_EQ(ret, 0) << "smem trans batch read failed, ret:" << ret
                         << ", rank:" << rank_id_;
      } else {
        for (size_t i = 0; i < weight_sizes_.size(); ++i) {
          ret = smem_trans_read(trans_handle_,
                                weight_addrs_[i],
                                global_session_id_[global_rank_id].c_str(),
                                remote_weight_addrs[i],
                                weight_sizes_[i]);
          CHECK_EQ(ret, 0) << "smem trans read failed, ret:" << ret
                           << ", rank:" << rank_id_;
        }
      }
    } else if (direction == mf_weight_transfer_direction_t::MF_PUSH_WEIGHT) {
      if (batched) {
        ret = smem_trans_batch_write(
            trans_handle_,
            const_cast<const void**>(weight_addrs_.data()),
            global_session_id_[global_rank_id].c_str(),
            remote_weight_addrs.data(),
            weight_sizes_.data(),
            static_cast<uint32_t>(weight_sizes_.size()));
        CHECK_EQ(ret, 0) << "smem trans batch write failed, ret:" << ret
                         << ", rank :" << rank_id_;
      } else {
        for (size_t i = 0; i < weight_sizes_.size(); ++i) {
          ret = smem_trans_write(trans_handle_,
                                 weight_addrs_[i],
                                 global_session_id_[global_rank_id].c_str(),
                                 remote_weight_addrs[i],
                                 weight_sizes_[i]);
          CHECK_EQ(ret, 0) << "smem trans write failed, ret:" << ret
                           << ", rank:" << rank_id_;
        }
      }
    } else {
      LOG(ERROR) << "invalid direction:" << static_cast<int>(direction);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "transfer weight cost time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
  }

  void barrier() {
    auto ret = smem_shm_control_barrier(shm_handle_);
    CHECK_EQ(ret, 0) << "smem shm control barrier failed, ret:" << ret
                     << ", rank:" << rank_id_;
  }

 private:
  int device_id_;
  int rank_id_;
  int rank_size_;
  std::string ip_port_;
  smem_trans_config_t trans_config_;
  smem_trans_t trans_handle_;
  std::string session_id_;

  smem_shm_config_t shm_config_;
  smem_shm_t shm_handle_;
  void* shm_gva_ = nullptr;

  std::vector<std::string> global_session_id_;

  std::vector<void*> weight_addrs_;
  std::vector<std::vector<void*>> global_weight_addrs_;
  std::vector<size_t> weight_sizes_;

  std::vector<void*> weight_buf_addrs_;
  std::vector<std::vector<void*>> global_weight_buf_addrs_;
};
}  // namespace xllm