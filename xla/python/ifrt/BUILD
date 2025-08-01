load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/python/ifrt:ifrt.bzl", "ifrt_package_groups")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform/default:rules_cc.bzl", "cc_library")
# copybara:uncomment load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":internal"]),
)

exports_files([
    "BUILD",
])

ifrt_package_groups()

cc_library(
    name = "ifrt",
    srcs = [
        "array_spec.cc",
        "array.cc",
        "client.cc",
        "compiler.cc",
        "device.cc",
        "device_list.cc",
        "dtype.cc",
        "layout.cc",
        "remap_plan.cc",
        "executable.cc",
        "host_callback.cc",
        "index.cc",
        "index_domain.cc",
        "memory.cc",
        "program.cc",
        # TODO(hyeontaek): Move this SerDes out of `ifrt` target once `Compiler::DeserializeLoadedExecutable()` is split into executable deserialization and executable loading, and thus the method can be removed.
        "executable_serdes.cc",
        "shape.cc",
        "sharding.cc",
        "topology.cc",
        "tuple.cc",
        "value.cc",
    ],
    hdrs = [
        "array.h",
        "layout.h",
        "array_spec.h",
        "client.h",
        "compiler.h",
        "remap_plan.h",
        "device.h",
        "device_list.h",
        "dtype.h",
        "executable.h",
        "future.h",
        "host_callback.h",
        "index.h",
        "index_domain.h",
        "memory.h",
        "program.h",
        # TODO(hyeontaek): Move this SerDes out of `ifrt` target once `Compiler::DeserializeLoadedExecutable()` is split into executable deserialization and executable loading, and thus the method can be removed.
        "executable_serdes.h",
        "shape.h",
        "sharding.h",
        "topology.h",
        "tuple.h",
        "value.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":array_spec_proto_cc",
        ":attribute_map",
        ":device_proto_cc",
        ":dtype_proto_cc",
        ":execute_options_proto_cc",
        ":layout_proto_cc",
        ":ref_wrapper",
        ":remap_plan_proto_cc",
        ":serdes",
        ":serdes_default_version_accessor",
        ":serdes_version",
        ":shape_proto_cc",
        ":sharding_proto_cc",
        ":user_context",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:bitmap",
        "//xla/tsl/lib/gtl:int_type",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@highwayhash",
        "@highwayhash//:arch_specific",
        "@highwayhash//:hh_types",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "array_test",
    size = "small",
    srcs = ["array_test.cc"],
    deps = [
        ":ifrt",
        ":mock",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "attribute_map",
    srcs = ["attribute_map.cc"],
    hdrs = ["attribute_map.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":attribute_map_proto_cc",
        ":serdes_default_version_accessor",
        ":serdes_version",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "attribute_map_test",
    size = "small",
    srcs = ["attribute_map_test.cc"],
    deps = [
        ":attribute_map",
        ":serdes_test_util",
        ":serdes_version",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "client_impl_util",
    srcs = ["client_impl_util.cc"],
    hdrs = ["client_impl_util.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        ":ifrt",
        ":user_context",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_proto_library(
    name = "execute_options_proto",
    srcs = ["execute_options.proto"],
    protodeps = [":attribute_map_proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

xla_cc_test(
    name = "executable_test",
    size = "small",
    srcs = ["executable_test.cc"],
    deps = [
        ":attribute_map",
        ":execute_options_proto_cc",
        ":ifrt",
        ":serdes_test_util",
        ":serdes_version",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "future_test",
    size = "small",
    srcs = ["future_test.cc"],
    deps = [
        ":ifrt",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "index_domain_test",
    size = "small",
    srcs = ["index_domain_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "index_test",
    size = "small",
    srcs = ["index_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "layout_proto",
    srcs = ["layout.proto"],
    protodeps = [":serdes_proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

xla_cc_test(
    name = "layout_test",
    size = "small",
    srcs = ["layout_test.cc"],
    deps = [
        ":basic_device_list",
        ":ifrt",
        ":mock",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_layout",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "layout_serdes_proto",
    srcs = ["layout_serdes.proto"],
)

cc_library(
    name = "layout_serdes",
    srcs = ["layout_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":ifrt",
        ":layout_serdes_proto_cc",
        ":serdes",
        ":serdes_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "layout_serdes_test",
    size = "small",
    srcs = ["layout_serdes_test.cc"],
    deps = [
        ":ifrt",
        ":layout_serdes",
        ":serdes",
        ":serdes_test_util",
        ":serdes_version",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "memory_test",
    size = "small",
    srcs = ["memory_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ref_wrapper",
    hdrs = ["ref_wrapper.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "ref_wrapper_test",
    size = "small",
    srcs = ["ref_wrapper_test.cc"],
    deps = [
        ":ref_wrapper",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "shape_test",
    size = "small",
    srcs = ["shape_test.cc"],
    deps = [
        ":ifrt",
        ":serdes_test_util",
        ":serdes_version",
        ":shape_proto_cc",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "sharding_test",
    size = "small",
    srcs = ["sharding_test.cc"],
    deps = [
        ":device_test_util",
        ":ifrt",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":ifrt",
        ":user_context",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "device_test_util",
    testonly = True,
    srcs = ["device_test_util.cc"],
    hdrs = ["device_test_util.h"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        ":basic_device_list",
        ":ifrt",
        ":mock",
        ":test_util",
        "//xla:util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "array_impl_test_lib",
    testonly = True,
    srcs = ["array_impl_test_lib.cc"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        ":ifrt",
        ":test_util",
        ":user_context",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,
)

build_test(
    name = "array_test_no_impl",
    targets = [":array_impl_test_lib"],
)

tf_proto_library(
    name = "attribute_map_proto",
    srcs = ["attribute_map.proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

cc_library(
    name = "client_impl_test_lib",
    testonly = True,
    srcs = ["client_impl_test_lib.cc"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        ":ifrt",
        ":test_util",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
    alwayslink = True,
)

build_test(
    name = "client_test_no_impl",
    targets = [":client_impl_test_lib"],
)

cc_library(
    name = "tuple_impl_test_lib",
    testonly = True,
    srcs = ["tuple_impl_test_lib.cc"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        ":ifrt",
        ":test_util",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,
)

build_test(
    name = "tuple_test_no_impl",
    targets = [":tuple_impl_test_lib"],
)

cc_library(
    name = "mock",
    testonly = True,
    srcs = ["mock.cc"],
    hdrs = ["mock.h"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":attribute_map",
        ":basic_device_list",
        ":ifrt",
        ":user_context",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_layout",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/framework:allocator",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_for_library",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "serdes_version",
    hdrs = ["serdes_version.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        "//xla/tsl/lib/gtl:int_type",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "serdes_any_version_accessor",
    hdrs = ["serdes_any_version_accessor.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([":serdes_any_version_users"]),
    deps = [":serdes_version"],
)

cc_library(
    name = "serdes_default_version_accessor",
    hdrs = ["serdes_default_version_accessor.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        ":serdes_version",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "serdes_week_4_old_version_accessor",
    hdrs = ["serdes_week_4_old_version_accessor.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([":serdes_week_4_old_version_users"]),
    deps = [":serdes_version"],
)

cc_library(
    name = "serdes_test_util",
    testonly = True,
    srcs = ["serdes_test_util.cc"],
    hdrs = ["serdes_test_util.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":serdes_any_version_accessor",
        ":serdes_version",
        ":serdes_week_4_old_version_accessor",
    ],
)

cc_library(
    name = "serdes",
    srcs = ["serdes.cc"],
    hdrs = ["serdes.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":serdes_default_version_accessor",
        ":serdes_proto_cc",
        ":serdes_version",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "serdes_test",
    srcs = ["serdes_test.cc"],
    deps = [
        ":serdes",
        ":serdes_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

tf_proto_library(
    name = "serdes_proto",
    srcs = ["serdes.proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

cc_library(
    name = "program_serdes",
    srcs = ["program_serdes.cc"],
    hdrs = ["program_serdes.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":ifrt",
        ":serdes",
        "//xla:util",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

cc_library(
    name = "sharding_serdes",
    srcs = ["sharding_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":ifrt",
        ":serdes",
        ":serdes_version",
        ":sharding_serdes_proto_cc",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "sharding_serdes_test",
    srcs = ["sharding_serdes_test.cc"],
    deps = [
        ":device_test_util",
        ":ifrt",
        ":serdes",
        ":serdes_proto_cc",
        ":serdes_test_util",
        ":serdes_version",
        ":sharding_serdes",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "array_spec_proto",
    srcs = ["array_spec.proto"],
    protodeps = [
        ":dtype_proto",
        ":shape_proto",
        ":sharding_proto",
    ],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

xla_cc_test(
    name = "array_spec_test",
    size = "small",
    srcs = ["array_spec_test.cc"],
    deps = [
        ":array_spec_proto_cc",
        ":device_test_util",
        ":ifrt",
        ":serdes_test_util",
        ":serdes_version",
        ":sharding_serdes",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_layout",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

tf_proto_library(
    name = "device_proto",
    srcs = ["device.proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

xla_cc_test(
    name = "device_list_test",
    size = "small",
    srcs = ["device_list_test.cc"],
    deps = [
        ":device_proto_cc",
        ":device_test_util",
        ":ifrt",
        ":serdes_test_util",
        ":serdes_version",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "basic_device_list",
    srcs = ["basic_device_list.cc"],
    hdrs = ["basic_device_list.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        ":device_proto_cc",
        ":ifrt",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

tf_proto_library(
    name = "dtype_proto",
    srcs = ["dtype.proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

xla_cc_test(
    name = "dtype_test",
    size = "small",
    srcs = ["dtype_test.cc"],
    deps = [
        ":dtype_proto_cc",
        ":ifrt",
        ":serdes_test_util",
        ":serdes_version",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "remap_plan_proto",
    srcs = ["remap_plan.proto"],
    protodeps = [":array_spec_proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

xla_cc_test(
    name = "remap_plan_test",
    size = "small",
    srcs = ["remap_plan_test.cc"],
    deps = [
        ":device_test_util",
        ":ifrt",
        ":remap_plan_proto_cc",
        ":serdes_test_util",
        ":serdes_version",
        ":sharding_serdes",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_layout",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "remap_impl_test_lib",
    testonly = True,
    srcs = ["remap_impl_test_lib.cc"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
    deps = [
        ":ifrt",
        ":test_util",
        "//xla:status_macros",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

tf_proto_library(
    name = "shape_proto",
    srcs = ["shape.proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

tf_proto_library(
    name = "sharding_proto",
    srcs = ["sharding.proto"],
    protodeps = [":serdes_proto"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

tf_proto_library(
    name = "sharding_serdes_proto",
    srcs = ["sharding_serdes.proto"],
    protodeps = [
        ":device_proto",
        ":shape_proto",
        "//xla/python/ifrt/ir:sharding_param_proto",
    ],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

cc_library(
    name = "plugin_program",
    srcs = ["plugin_program.cc"],
    hdrs = ["plugin_program.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":ifrt",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "plugin_program_serdes",
    srcs = ["plugin_program_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":plugin_program",
        ":serdes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "plugin_program_serdes_test",
    srcs = ["plugin_program_serdes_test.cc"],
    deps = [
        ":plugin_program",
        ":plugin_program_serdes",
        ":serdes",
        ":serdes_proto_cc",
        ":serdes_version",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "custom_call_program",
    srcs = ["custom_call_program.cc"],
    hdrs = ["custom_call_program.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":ifrt",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/strings:cord",
        "@llvm-project//llvm:Support",
    ],
)

tf_proto_library(
    name = "custom_call_program_proto",
    srcs = ["custom_call_program.proto"],
    protodeps = [
        ":array_spec_proto",
        ":device_proto",
    ],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
)

cc_library(
    name = "custom_call_program_serdes",
    srcs = ["custom_call_program_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        ":array_spec_proto_cc",
        ":custom_call_program",
        ":custom_call_program_proto_cc",
        ":ifrt",
        ":program_serdes",
        ":serdes",
        ":serdes_version",
        ":sharding_proto_cc",
        ":sharding_serdes",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

cc_library(
    name = "user_context",
    srcs = ["user_context.cc"],
    hdrs = ["user_context.h"],
    visibility = internal_visibility([
        ":friends",
        ":internal",
        ":users",
    ]),
    deps = [
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "user_context_test",
    srcs = ["user_context_test.cc"],
    deps = [
        ":user_context",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "custom_call_program_serdes_test",
    srcs = ["custom_call_program_serdes_test.cc"],
    deps = [
        ":custom_call_program",
        ":custom_call_program_serdes",
        ":device_test_util",
        ":ifrt",
        ":program_serdes",
        ":serdes",
        ":serdes_proto_cc",
        ":serdes_test_util",
        ":serdes_version",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

# copybara:uncomment_begin
# bzl_library(
#     name = "ifrt_bzl",
#     srcs = ["ifrt.bzl"],
#     parse_tests = False,
#     visibility = ["//visibility:private"],
#     deps = ["//xla/tsl:package_groups_bzl"],
# )
#
# bzl_library(
#     name = "ifrt_google_bzl",
#     srcs = ["ifrt.google.bzl"],
#     parse_tests = False,
#     visibility = ["//visibility:private"],
# )
# copybara:uncomment_end
