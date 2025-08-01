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

#include <string>

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace mt = ::mlir::triton;
namespace mt_xla = ::mlir::triton::xla;
namespace ttng = mlir::triton::nvidia_gpu;

absl::Status CreateTritonPipeline(
    mlir::OpPassManager* pm, std::string arch_name, int num_warps, int num_ctas,
    int num_stages, mt::nvidia_gpu::ClusterInfo& out_cluster_info) {
  TF_ASSIGN_OR_RETURN(
      const stream_executor::CudaComputeCapability cc,
      stream_executor::CudaComputeCapability::FromString(arch_name));
  const int ccAsInt = cc.major * 10 + cc.minor;
  const int threadsPerWarp = 32;

  pm->addPass(mt_xla::CreateRoundF32ToTF32ForTf32DotRewritePass());

  // Based on make_ttir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mt::createTritonRewriteTensorPointer());
  if (!cc.IsAtLeastHopper()) {
    pm->addPass(mt::createTritonRewriteTensorDescriptorToPointer());
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonCombineOps());
  pm->addPass(mt::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createTritonLoopUnroll());

  // Based on make_ttgir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm->addPass(mt::createConvertTritonToTritonGPU(
      {absl::StrFormat("cuda:%u", ccAsInt), num_warps, threadsPerWarp,
       num_ctas}));
  pm->addPass(mt::gpu::createTritonGPUCoalesce());
  if (cc.IsAtLeastAmpere()) {
    pm->addPass(mt::gpu::createTritonGPUF32DotTC());
  }
  pm->addPass(ttng::createTritonNvidiaGPUPlanCTAPass(&out_cluster_info));
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  pm->addPass(mt::gpu::createTritonGPUAccelerateMatmul());
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands({cc.IsAtLeastAmpere()}));
  pm->addPass(ttng::createTritonNvidiaGPUOptimizeDescriptorEncodingPass());
  pm->addPass(mt::createTritonLoopAwareCSE());
  if (cc.IsAmpere() || cc.IsHopper()) {
    pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mlir::createLoopInvariantCodeMotionPass());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm->addPass(mlir::createNVGPUWarpSpecialization({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUAssignLatencies({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUScheduleLoops());
    pm->addPass(mt::gpu::createTritonGPUPipeline({num_stages}));
  } else if (cc.IsAtLeastBlackwell()) {
    pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mlir::createLoopInvariantCodeMotionPass());
    pm->addPass(mt::gpu::createTritonGPUOptimizeAccumulatorInit());
    pm->addPass(mt::gpu::createTritonGPUHoistTMEMAlloc());
    pm->addPass(ttng::createTritonNvidiaGPUPromoteLHSToTMemPass());
    pm->addPass(mt::gpu::createTritonGPUAssignLatencies({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUScheduleLoops());
    pm->addPass(
        mt::gpu::createTritonGPUAutomaticWarpSpecialization({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUPipeline({num_stages}));
    pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm->addPass(ttng::createTritonNvidiaGPURemoveTMEMTokensPass());
  } else {
    pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mt::gpu::createTritonGPUPrefetch());
  pm->addPass(
      mt::gpu::createTritonGPUOptimizeDotOperands({cc.IsAtLeastAmpere()}));
  pm->addPass(mt::gpu::createTritonGPUCoalesceAsyncCopy());
  pm->addPass(ttng::createTritonNvidiaGPUOptimizeTMemLayoutsPass());
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(ttng::createTritonNvidiaGPUInterleaveTMemPass());
  pm->addPass(mt::gpu::createTritonGPUReduceDataDuplication());
  pm->addPass(mt::gpu::createTritonGPUReorderInstructions());
  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mlir::createSymbolDCEPass());
  if (cc.IsAtLeastHopper()) {
    pm->addPass(ttng::createTritonNvidiaGPUTMALoweringPass());
  }
  pm->addPass(ttng::createTritonGPUFenceInsertion({ccAsInt}));
  pm->addPass(ttng::createTritonNvidiaGPUMMALoweringPass());
  pm->addPass(mlir::createSCCPPass());
  pm->addPass(mlir::createCanonicalizerPass());

  // Corresponds to "mod.get_tensordesc_metadata()"
  // in @triton//:third_party/nvidia/backend/compiler.py
  pm->addPass(mt_xla::CreateExtractTmaInfoPass());

  // Based on make_llir() in
  // @triton//:third_party/nvidia/backend/compiler.py
  pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm->addPass(mt::gpu::createTritonGPUAllocateWarpGroups());
  pm->addPass(mlir::createSCFToControlFlowPass());
  pm->addPass(mt::gpu::createAllocateSharedMemory());
  pm->addPass(ttng::createTritonTensorMemoryAllocationPass());
  // We could add a flag to XLA to optionally enable the following pass:
  // pm->addPass(mt::instrument::createTritonInstrumentConcurrencySanitizer());
  pm->addPass(mt::gpu::createTritonGPUGlobalScratchAllocationPass());
  pm->addPass(ttng::createTritonGPUProxyFenceInsertion({ccAsInt}));
  pm->addPass(mt::createConvertTritonGPUToLLVMPass(ccAsInt));
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mt::createConvertNVGPUToLLVM());
  pm->addPass(mt::createConvertWarpSpecializeToLLVM());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mlir::createConvertNVVMToLLVMPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.

  return absl::OkStatus();
}

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info) {
  return nvptx::LibDevicePath(
      hlo_config.debug_options().xla_gpu_cuda_data_dir());
}

}  // namespace gpu
}  // namespace xla
