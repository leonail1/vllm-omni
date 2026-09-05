// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#include <ATen/ATen.h>
#include <torch/library.h>
#include <vector>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {
void attention(const at::Tensor& source, at::Tensor& output, int64_t model,
               int64_t input_pool, int64_t output_pool,
               const c10::List<int64_t>& pause_gates) {
  auto npu_stream = c10_npu::getCurrentNPUStream(source.get_device());
  auto stream = npu_stream.stream(false);
  auto input_bytes = source.nbytes();
  auto output_bytes = output.nbytes();
  auto source_ptr = source.data_ptr();
  auto output_ptr = output.data_ptr();
  std::vector<int64_t> gates(pause_gates.begin(), pause_gates.end());
  if (input_bytes) c10_npu::NPUCachingAllocator::recordStream(source.storage().data_ptr(), npu_stream);
  if (output_bytes) c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), npu_stream);
  // Retain tensors until the host queue has submitted their native work.
  at_npu::native::OpCommand::RunOpApi("DloQueuedAttention", [=, source_hold = source, output_hold = output]() -> int {
    auto checked = [](aclError code) { TORCH_CHECK(code == ACL_SUCCESS, "DLO runtime error ", code); };
    try {
      for (auto gate : gates) checked(aclrtValueWrite(reinterpret_cast<void*>(gate), 1, 0, stream));
      if (input_bytes) checked(aclrtMemcpyAsync(reinterpret_cast<void*>(input_pool), input_bytes,
                                              source_ptr, input_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
      checked(aclmdlRIExecuteAsync(reinterpret_cast<void*>(model), stream));
      if (output_bytes) checked(aclrtMemcpyAsync(output_ptr, output_bytes,
                                               reinterpret_cast<void*>(output_pool), output_bytes,
                                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
      for (auto gate : gates) checked(aclrtValueWrite(reinterpret_cast<void*>(gate), 0, 0, stream));
    } catch (...) {
      // Resume weight admission while preserving the original submission error.
      for (auto gate : gates) aclrtValueWrite(reinterpret_cast<void*>(gate), 0, 0, stream);
      throw;
    }
    return 0;
  });
}
}  // namespace

TORCH_LIBRARY(vllm_omni_sdma, m) {
  m.def("attention(Tensor source, Tensor(a!) output, int model, int input_pool, int output_pool, int[] gates) -> ()");
  m.impl("attention", c10::DispatchKey::PrivateUse1, TORCH_FN(attention));
}
