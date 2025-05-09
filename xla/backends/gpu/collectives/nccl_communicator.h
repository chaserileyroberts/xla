/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COMMUNICATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {

class NcclCollectives;

// XLA collectives communicator wrapping an NCCL communicator.
class NcclCommunicator : public Communicator {
 public:
  explicit NcclCommunicator(NcclCollectives* collectives, ncclComm_t comm);
  ~NcclCommunicator() override;

  // NcclCommunicator is not copyable or movable.
  NcclCommunicator(const NcclCommunicator&) = delete;
  NcclCommunicator(NcclCommunicator&&) = delete;
  NcclCommunicator& operator=(const NcclCommunicator&) = delete;
  NcclCommunicator& operator=(NcclCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::Status HealthCheck() const final;
  absl::StatusOr<size_t> NumRanks() const final;

  absl::StatusOr<std::unique_ptr<RegisteredBufferHandle>> RegisterBuffer(
      se::DeviceMemoryBase buffer) final;

  tsl::AsyncValueRef<Event> AllReduce(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      ReductionKind reduction_kind,
                                      const Executor& executor) final;

  tsl::AsyncValueRef<Event> Broadcast(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      RankId root,
                                      const Executor& executor) final;

  tsl::AsyncValueRef<Event> ReduceScatter(se::DeviceMemoryBase send_buffer,
                                          se::DeviceMemoryBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          ReductionKind reduction_kind,
                                          const Executor& executor) final;

  tsl::AsyncValueRef<Event> AllGather(se::DeviceMemoryBase send_buffer,
                                      se::DeviceMemoryBase recv_buffer,
                                      PrimitiveType dtype, size_t count,
                                      const Executor& executor) final;

  tsl::AsyncValueRef<Event> AllToAll(
      absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final;

  tsl::AsyncValueRef<Event> CollectivePermute(
      se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) final;

  tsl::AsyncValueRef<Event> Send(se::DeviceMemoryBase send_buffer,
                                 PrimitiveType dtype, size_t count, RankId peer,
                                 const Executor& executor) final;

  tsl::AsyncValueRef<Event> Recv(se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count, RankId peer,
                                 const Executor& executor) final;

  std::string ToString() const final;

  ncclComm_t comm() const { return comm_; }

 private:
  static absl::StatusOr<se::Stream*> ToStream(const Executor& executor);

  NcclCollectives* collectives_;  // Parent NcclCollectives instance
  ncclComm_t comm_;               // Underlying NCCL communicator
  bool aborted_ = false;          // Has Abort() been called?
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_NCCL_COMMUNICATOR_H_
