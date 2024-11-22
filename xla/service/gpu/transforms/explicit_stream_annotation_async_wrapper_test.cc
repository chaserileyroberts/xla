/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/explicit_stream_annotation_async_wrapper.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/test.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ExplicitStreamAnnotationAsyncWrapperTest = HloTestBase;

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest, AnnotatedOpIsWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %sub (lhs: f32[]) -> f32[] {
    %lhs = f32[] parameter(0)
    %rhs = f32[] constant(1)
    ROOT %sub = f32[] subtract(f32[] %lhs, f32[] %rhs)
  }

  ENTRY %main () -> f32[] {
    %lhs = f32[] constant(42)
    %call1 = f32[] call(f32[] %lhs), to_apply=%sub, frontend_attributes={_xla_stream_annotation="1"}
  })";

  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %lhs.1 = f32[] constant(42)
  // CHECK: %call-start = ((f32[]), f32[]) call-start(f32[] %lhs.1), async_execution_thread="explicit", to_apply=%sub, frontend_attributes={_xla_stream_annotation="1"}
  // CHECK: ROOT %call-done = f32[] call-done(((f32[]), f32[]) %call-start), frontend_attributes={_xla_stream_annotation="1"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false}
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}
}  // namespace
}  // namespace xla::gpu
