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

#include "xla/service/gpu/transforms/explicit_nccl_group_async_wrapper.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ExplicitNcclGroupAsyncWrapperTest = HloTestBase;

TEST_F(ExplicitNcclGroupAsyncWrapperTest, AnnotatedOpIsWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %comms (a: f32[1]) -> (f32[1], f32[1]) {
    a = f32[1] parameter(0)
    x = f32[1] all-gather(f32[1] a), dimensions={0}
    y = f32[1] collective-permute(a), source_target_pairs={{0,1}}
    ROOT result = (f32[1], f32[1]) tuple(x, y)
  }

  ENTRY %main () -> (f32[1], f32[1]) {
    b = f32[1] parameter(0)
    ROOT c = (f32[1], f32[1]) call(f32[1] b), to_apply=%comms, frontend_attributes={_nccl_group=""}
  })";

  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  ExplicitNcclGroupAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %b = f32[1]{0} parameter(0)
  // CHECK: %tuple-start = ((f32[1]{0}), (f32[1]{0}, f32[1]{0})) async-start(f32[1]{0} %b), async_execution_thread="explicit", calls=%comms  
  // CHECK: ROOT %tuple-done = (f32[1]{0}, f32[1]{0}) async-done(((f32[1]{0}), (f32[1]{0}, f32[1]{0})) %tuple-start)
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}

}  // namespace
}  // namespace xla::gpu
