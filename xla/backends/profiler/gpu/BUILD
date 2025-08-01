load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "if_nvcc",
    "internal_visibility",
    "tsl_copts",
    "tsl_gpu_library",
)
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_additional_device_tracer_srcs",
)
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
    "if_cuda_newer_than",
)
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
)

tsl_gpu_library(
    name = "device_tracer",
    srcs = tf_additional_device_tracer_srcs(),
    copts = tf_profiler_copts() + tsl_copts(),
    deps = [
        ":cupti_utils",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ] + if_cuda([
        # keep sorted
        ":cupti_buffer_events",
        ":cupti_collector",
        ":cupti_tracer",
        ":cupti_wrapper",
    ]) + if_rocm([
        # keep sorted
        ":rocm_collector",
        ":rocm_tracer",
    ]),
    alwayslink = 1,
)

tsl_gpu_library(
    name = "cupti_interface",
    hdrs = if_cuda(["cupti_interface.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:types",
    ] + if_cuda(["//xla/tsl/cuda:cupti"]),
)

tsl_gpu_library(
    name = "mock_cupti",
    testonly = 1,
    hdrs = if_cuda(["mock_cupti.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_interface",
    ],
    deps = [
        "@tsl//tsl/platform:test",
    ],
)

tsl_gpu_library(
    name = "cupti_error_manager",
    srcs = if_cuda(["cupti_error_manager.cc"]),
    hdrs = if_cuda(["cupti_error_manager.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_interface",
        ":cupti_wrapper",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

xla_test(
    name = "cupti_error_manager_test",
    size = "small",
    srcs = ["cupti_error_manager_test.cc"],
    backends = ["gpu"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "config-cuda-only",
        "cuda-only",
        "no_mac",
    ],
    deps = [
        ":cupti_collector",
        ":cupti_interface",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ] + if_cuda_is_configured([
        ":cuda_test",
        ":cupti_error_manager",
        ":cupti_tracer",
        ":cupti_utils",
        ":cupti_wrapper",
        ":mock_cupti",
        "@com_google_absl//absl/memory",
        "//xla/tsl/profiler/utils:time_utils",
    ]),
)

cuda_library(
    name = "cuda_test",
    testonly = 1,
    srcs = ["cuda_test.cu.cc"],
    hdrs = ["cuda_test.h"],
    copts = if_nvcc([
        "-nvcc_options",
        "ptxas-options=-v",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
        "@tsl//tsl/platform:test",
    ],
)

# Rationale for linkstatic: The symbols in libcupti_static.a have hidden
# visibility. The wrapper will fail to find them if it's ever built as a
# shared library. This is the same issue as b/11094727. Always linking
# the wrapper statically works around the issue. An alternative would be
# to patch libcupti_static, but it's not worth the trouble considering
# that the wrapper is about the only direct user.
tsl_gpu_library(
    name = "cupti_wrapper",
    srcs = if_cuda([
        "cupti_wrapper.cc",
        "cupti_wrapper_stub.cc",
    ]),
    hdrs = if_cuda(["cupti_wrapper.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
    ] + if_cuda(["//xla/tsl/cuda:cupti"]),
)

tsl_gpu_library(
    name = "cupti_tracer",
    srcs = if_cuda(["cupti_tracer.cc"]),
    hdrs = if_cuda(["cupti_tracer.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    tags = ["config-cuda-only"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "//xla/tsl/profiler/utils:per_thread",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
    ] + if_cuda([
        # keep sorted
        ":cupti_buffer_events",
        ":cupti_collector",
        ":cupti_interface",
        ":cupti_pm_sampler_factory",
        ":cupti_utils",
        ":nvtx_utils",
    ]),
)

cc_library(
    name = "cupti_pm_sampler_factory",
    srcs = ["cupti_pm_sampler_factory.cc"],
    hdrs = [
        "cupti_pm_sampler.h",
        "cupti_pm_sampler_factory.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "config-cuda-only",
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cupti_collector",
        ":cupti_interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@local_config_cuda//cuda:cupti_headers",
    ] + if_cuda_newer_than(
        "12_6",
        [":cupti_pm_sampler_impl"],
        [":cupti_pm_sampler_stub"],
    ),
)

cc_library(
    name = "cupti_pm_sampler_stub",
    srcs = ["cupti_pm_sampler_stub.cc"],
    hdrs = [
        "cupti_pm_sampler.h",
        "cupti_pm_sampler_stub.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    copts = tf_profiler_copts() + tsl_copts(),
    deps = [
        ":cupti_collector",
        ":cupti_interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "cupti_pm_sampler_impl",
    srcs = ["cupti_pm_sampler_impl.cc"],
    hdrs = [
        "cupti_pm_sampler.h",
        "cupti_pm_sampler_impl.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "config-cuda-only",
        "cuda-only",
        "gpu",
        "manual",  # This target requires CUDA 12.6+, therefore we only built it if it was requested via a dependency.
    ],
    deps = [
        ":cupti_collector",
        ":cupti_interface",
        ":cupti_status",
        "//xla/stream_executor/cuda:cuda_status",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_config_cuda//cuda:cupti_headers",
        "@tsl//tsl/platform:errors",
    ],
)

tsl_gpu_library(
    name = "cupti_profiler",
    srcs = if_cuda(["cupti_profiler.cc"]),
    hdrs = if_cuda(["cupti_profiler.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

tsl_gpu_library(
    name = "rocm_collector",
    srcs = if_rocm(["rocm_collector.cc"]),
    hdrs = if_rocm(["rocm_collector.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/rocm:roctracer_wrapper",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:abi",
        "@tsl//tsl/platform:env_time",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
    ],
)

tsl_gpu_library(
    name = "rocm_tracer",
    srcs = if_rocm(["rocm_tracer.cc"]),
    hdrs = if_rocm(["rocm_tracer.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_collector",
        "//xla/stream_executor/rocm:roctracer_wrapper",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "//xla/tsl/profiler/utils:time_utils",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
    ],
)

tsl_gpu_library(
    name = "nvtx_utils",
    srcs = if_cuda(["nvtx_utils.cc"]),
    hdrs = if_cuda(["nvtx_utils.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    deps = [
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:macros",
    ],
)

tsl_gpu_library(
    name = "cupti_collector",
    srcs = if_cuda(["cupti_collector.cc"]),
    hdrs = ["cupti_collector.h"],
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_buffer_events",
        ":cupti_interface",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "//xla/tsl/profiler/utils:math_utils",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:timespan",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:abi",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ] + if_cuda([
        "//xla/tsl/cuda:cupti",
        "//xla/tsl/cuda",
    ]),
)

tsl_gpu_library(
    name = "cupti_buffer_events",
    srcs = if_cuda(["cupti_buffer_events.cc"]),
    hdrs = ["cupti_buffer_events.h"],
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        "//xla/tsl/profiler/utils:buffer_pool",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:thread_annotations",
    ] + if_cuda(["//xla/tsl/cuda:cupti"]),
)

tsl_gpu_library(
    name = "cupti_utils",
    srcs = if_cuda(["cupti_utils.cc"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_error_manager",
        ":cupti_interface",
        ":cupti_wrapper",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:stringpiece",
        "//xla/tsl/util:env_var",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

tsl_gpu_library(
    name = "cupti_status",
    srcs = ["cupti_status.cc"],
    hdrs = ["cupti_status.h"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "config-cuda-only",
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "cupti_buffer_events_test",
    srcs = ["cupti_buffer_events_test.cc"],
    tags = [
        "no_mac",
    ],
    deps = [
        ":cupti_buffer_events",
        ":cupti_collector",  # buildcleaner: keep
        ":cupti_utils",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "cupti_collector_test",
    size = "small",
    srcs = ["cupti_collector_test.cc"],
    backends = ["gpu"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "config-cuda-only",
        "cuda-only",
        "no_mac",
    ],
    deps = [
        ":cupti_collector",
        ":cupti_tracer",
        ":cupti_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cuda_library(
    name = "nvtx_with_cuda_kernels",
    testonly = 1,
    srcs = ["nvtx_with_cuda_kernels.cu.cc"],
    hdrs = ["nvtx_with_cuda_kernels.h"],
    copts = if_nvcc([
        "-nvcc_options",
        "ptxas-options=-v",
    ]),
    tags = ["cuda-only"],
    visibility = ["//visibility:public"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)

xla_test(
    name = "nvtx_with_cuda_kernels_test",
    size = "small",
    srcs = ["nvtx_with_cuda_kernels_test.cc"],
    backends = ["gpu"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "cuda-only",
        "no_mac",
    ],
    deps = [
        ":nvtx_with_cuda_kernels",
        "@com_google_googletest//:gtest_main",
    ],
)

cuda_library(
    name = "profile_with_cuda_kernels",
    testonly = 1,
    srcs = ["profile_with_cuda_kernels.cu.cc"],
    hdrs = ["profile_with_cuda_kernels.h"],
    copts = if_nvcc([
        "-nvcc_options",
        "ptxas-options=-v",
    ]),
    tags = [
        "config-cuda-only",
        "cuda-only",
    ],
)

xla_test(
    name = "profile_with_cuda_kernels_test",
    size = "small",
    srcs = ["profile_with_cuda_kernels_test.cc"],
    args = if_google(["--heap_check="]),  # There is a memory leak in CUPTI
    backends = ["gpu"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "config-cuda-only",
        "cuda-only",
        "no_mac",
    ],
    deps = [
        ":cupti_collector",
        ":cupti_error_manager",
        ":cupti_tracer",
        ":cupti_utils",
        ":cupti_wrapper",
        ":profile_with_cuda_kernels",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
