cc_library(
    name = "SizedInt",
    hdrs = ["internal/SizedInt.hpp"],
)
cc_library(
    name = "BitArray",
    hdrs = ["internal/BitArray.hpp"],
    deps = [
        "@ParallelTools//ParallelTools:parallel",
    ],
)
cc_library(
    name = "helpers",
    hdrs = ["internal/helpers.hpp"],
    deps = [
        "BitArray",
        "SizedInt",
        "@ParallelTools//ParallelTools:parallel",
        "@ParallelTools//ParallelTools:reducer",
    ],
)
cc_library(
    name = "PMA",
    hdrs = ["PMA.hpp"],
    deps = [
        "helpers",
        "SizedInt",
        "@soa//StructOfArrays:soa",
    ],
)
cc_library(
    name = "TinySet_small",
    hdrs = ["TinySet_small.hpp"],
    deps = [
        "helpers",
        "BitArray",
        "SizedInt",
        "PMA",
    ],
)
cc_library(
    name = "SparseMatrix",
    hdrs = ["SparseMatrix.hpp"],
    deps = [
        "helpers",
        "BitArray",
        "TinySet_small",
        "@ParallelTools//ParallelTools:parallel",
        "@ParallelTools//ParallelTools:reducer",
        "@parlaylib//parlay:primitives",
    ],
)


package(
    default_visibility = ["//visibility:public"],
)
