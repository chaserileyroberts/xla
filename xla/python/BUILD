load("//xla:py_strict.bzl", "py_strict_test")
load("//xla:pytype.bzl", "pytype_strict_library")
load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
    "xla_py_test_deps",
)
load("//xla/python:package_groups.bzl", "XLA_PYTHON_XLA_CLIENT_USERS", "XLA_PYTHON_XLA_EXTENSION_USERS")
load("//xla/python:pywrap.bzl", "nanobind_pywrap_extension")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable", "tsl_pybind_extension")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":jax"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
        "//xla:internal",
    ],
)

package_group(
    name = "jax",
    packages = [
        "//third_party/py/jax/...",
    ],
)

exports_files(
    srcs = ["pyinit_stub.c"],
    visibility = [":friends"],
)

pytype_strict_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    visibility = internal_visibility(XLA_PYTHON_XLA_CLIENT_USERS),
    deps = if_google([
        ":_hlo_pass",
        ":_ops",
        ":_profiler",
        ":_profile_data",
        ":_xla_builder",
        "//third_party/py/jax/jaxlib:xla_client",
        "@ml_dtypes_py//ml_dtypes",
        "//third_party/py/numpy",
    ]),
)

pytype_strict_library(
    name = "xla_extension",
    srcs = ["xla_extension.py"],
    visibility = internal_visibility(XLA_PYTHON_XLA_EXTENSION_USERS),
    deps = if_google(["//third_party/py/jax/jaxlib:_jax"]),
)

tsl_pybind_extension(
    name = "status_casters_ext",
    testonly = 1,
    srcs = ["status_casters_ext.cc"],
    deps = [
        "//xla/pjrt:exceptions",
        "//xla/pjrt:status_casters",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
)

py_strict_test(
    name = "status_casters_test",
    srcs = ["status_casters_test.py"],
    main = "status_casters_test.py",
    tags = ["no_oss"],
    deps = [
        ":status_casters_ext",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = [
        "types.h",
        "version.h",
    ],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":nb_numpy",
        ":safe_static_init",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:exceptions",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt:pjrt_dtype",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@nanobind",
    ],
)

cc_library(
    name = "literal_type_casters",
    hdrs = ["literal_type_casters.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":types",
        "//xla:literal",
        "//xla:shape_util",
        "@com_google_absl//absl/container:inlined_vector",
        "@nanobind",
    ],
)

cc_library(
    name = "pprof_profile_builder",
    srcs = ["pprof_profile_builder.cc"],
    hdrs = ["pprof_profile_builder.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//xla:util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@nanobind",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:profile_proto_cc",
    ],
)

cc_library(
    name = "inspect_sharding",
    srcs = ["inspect_sharding.cc"],
    hdrs = ["inspect_sharding.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service/spmd:spmd_partitioner",
        "@com_google_absl//absl/status",
    ],
    # Always register 'InspectSharding' custom partitioning handler.
    alwayslink = 1,
)

cc_library(
    name = "custom_partition_callback",
    srcs = ["custom_partition_callback.cc"],
    hdrs = ["custom_partition_callback.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt/c:pjrt_c_api_custom_partitioner_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/service:call_inliner",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service/spmd:spmd_partitioner",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "custom_call_batch_partitioner",
    srcs = ["custom_call_batch_partitioner.cc"],
    hdrs = ["custom_call_batch_partitioner.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service/spmd:spmd_partitioner",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "debug_callback_partitioner",
    srcs = ["debug_callback_partitioner.cc"],
    hdrs = ["debug_callback_partitioner.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_sharding",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service/spmd:spmd_partitioner",
        "@com_google_absl//absl/status",
    ],
    # Always register 'DebugCallbackCustomCallPartitioner' custom partitioning handler.
    alwayslink = 1,
)

nanobind_pywrap_extension(
    name = "_hlo_pass",
    srcs = ["hlo_pass.cc"],
    pytype_srcs = ["_hlo_pass.pyi"],
    visibility = [":friends"],
    deps = [
        "//xla/hlo/ir:hlo_module_group",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/pjrt:status_casters",
        "//xla/service:call_inliner",
        "@nanobind",
    ],
)

nanobind_pywrap_extension(
    name = "_xla_builder",
    srcs = ["xla_builder.cc"],
    pytype_deps = [
        "//third_party/py/jax/jaxlib:_jax",
    ],
    pytype_srcs = ["_xla_builder.pyi"],
    deps = [
        ":nb_helpers",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/pjrt:status_casters",
        "//xla/service:name_uniquer",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
    ],
)

nanobind_pywrap_extension(
    name = "_ops",
    srcs = ["ops.cc"],
    pytype_deps = [
        ":_xla_builder",
        "//third_party/py/jax/jaxlib:_jax",
    ],
    pytype_srcs = ["_ops.pyi"],
    deps = [
        ":literal_type_casters",
        ":nb_absl_span",
        ":nb_helpers",
        ":types",
        # placeholder for index annotation deps
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:approx_topk",
        "//xla/hlo/builder/lib:approx_topk_shape",
        "//xla/hlo/builder/lib:comparators",
        "//xla/hlo/builder/lib:lu_decomposition",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:qr",
        "//xla/hlo/builder/lib:self_adjoint_eig",
        "//xla/hlo/builder/lib:sorting",
        "//xla/hlo/builder/lib:svd",
        "//xla/pjrt:status_casters",
        "//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "refine_polymorphic_shapes",
    srcs = ["refine_polymorphic_shapes.cc"],
    hdrs = ["refine_polymorphic_shapes.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/mlir_hlo:stablehlo_extension_passes",
        "//xla/service/spmd/shardy:utils",
        "//xla/service/spmd/shardy/round_trip_common:import_constants",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:base",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "safe_static_init",
    hdrs = ["safe_static_init.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@nanobind",
    ],
)

cc_library(
    name = "profiler_utils",
    srcs = ["profiler_utils.cc"],
    hdrs = ["profiler_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/plugin:plugin_tracer",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
)

nanobind_pywrap_extension(
    name = "_profiler",
    srcs = ["profiler.cc"],
    pytype_srcs = ["_profiler.pyi"],
    deps = [
        ":aggregate_profile",
        ":profiler_utils",
        ":xplane_to_profile_instructions",
        # placeholder for index annotation deps
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/cpu:python_tracer",
        "//xla/backends/profiler/plugin:plugin_tracer",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/python/profiler:profile_data_lib",
        "//xla/tsl/platform:macros",
        "//xla/tsl/profiler/rpc:profiler_server_impl",
        "//xla/tsl/profiler/rpc/client:capture_profile",
        "//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

nanobind_pywrap_extension(
    name = "_profile_data",
    srcs = ["profile_data.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    pytype_srcs = ["_profile_data.pyi"],
    deps = [
        "//xla/python/profiler:profile_data_lib",
        "@nanobind",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

py_strict_test(
    name = "profile_data_test",
    srcs = ["profile_data_test.py"],
    main = "profile_data_test.py",
    tags = [
        "no_oss",
        "optonly",
    ],
    deps =
        if_google([
            "//third_party/py/jax",
            "//pyglib:gfile",
            "@absl_py//absl/testing:absltest",
        ]),
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/log:initialize",
    ],
)

cc_library(
    name = "xplane_to_profile_instructions",
    srcs = ["xplane_to_profile_instructions.cc"],
    hdrs = ["xplane_to_profile_instructions.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:types",
        "//xla/tsl/profiler/convert:xla_op_utils",
        "//xla/tsl/profiler/utils:file_system_utils",
        "//xla/tsl/profiler/utils:tf_xplane_visitor",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "//xla/tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

xla_cc_test(
    name = "xplane_to_profile_instructions_test",
    srcs = ["xplane_to_profile_instructions_test.cc"],
    deps = [
        ":xplane_to_profile_instructions",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "//xla/tsl/profiler/convert:xla_op_utils",
        "//xla/tsl/profiler/rpc/client:save_profile",
        "//xla/tsl/profiler/utils:file_system_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc_impl",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "nb_helpers",
    hdrs = ["nb_helpers.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@local_config_python//:python_headers",
        "@nanobind",
    ],
)

cc_library(
    name = "nb_numpy",
    srcs = ["nb_numpy.cc"],
    hdrs = ["nb_numpy.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        # copybara:uncomment_begin
        # "//third_party/py/numpy:multiarray",  # build_cleaner: keep
        # copybara:uncomment_end
        "@local_config_python//:python_headers",
        "//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "nb_absl_inlined_vector",
    hdrs = ["nb_absl_inlined_vector.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@nanobind",
    ],
)

cc_library(
    name = "nb_absl_span",
    hdrs = ["nb_absl_span.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/types:span",
        "@nanobind",
    ],
)

cc_library(
    name = "nb_absl_flat_hash_map",
    hdrs = ["nb_absl_flat_hash_map.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@nanobind",
    ],
)

cc_library(
    name = "nb_absl_flat_hash_set",
    hdrs = ["nb_absl_flat_hash_set.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@nanobind",
    ],
)

cc_library(
    name = "aggregate_profile",
    srcs = ["aggregate_profile.cc"],
    hdrs = ["aggregate_profile.h"],
    deps = [
        ":xplane_to_profile_instructions",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
    ],
)

xla_cc_test(
    name = "aggregate_profile_test",
    srcs = ["aggregate_profile_test.cc"],
    deps = [
        ":aggregate_profile",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc_impl",
    ],
)

py_strict_test(
    name = "xla_compiler_test",
    srcs = ["xla_compiler_test.py"],
    main = "xla_compiler_test.py",
    tags = ["no_oss"],
    deps = [
        ":xla_extension",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

cc_library(
    name = "version",
    hdrs = ["version.h"],
    compatible_with = [],
    visibility = internal_visibility([":friends"]),
)
