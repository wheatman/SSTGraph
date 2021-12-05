#pragma once
#include <cstdint>
#include <cstdlib>
// intel cilk+
#if CILK == 1
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include <iostream>
#include <sstream>
#define parallel_for cilk_for
#define parallel_main main
#ifdef __clang__
#define parallel_for_1 _Pragma("cilk grainsize 1") cilk_for
#define parallel_for_8 _Pragma("cilk grainsize 8") cilk_for
#define parallel_for_256 _Pragma("cilk grainsize 256") cilk_for
#else
#define parallel_for_1 _Pragma("cilk grainsize = 1") cilk_for
#define parallel_for_8 _Pragma("cilk grainsize = 8") cilk_for
#define parallel_for_256 _Pragma("cilk grainsize = 256") cilk_for
#endif
[[maybe_unused]] static int getWorkers() { return __cilkrts_get_nworkers(); }

[[maybe_unused]] static int getWorkerNum() {
  return __cilkrts_get_worker_number();
}

// openmp
#elif OPENMP == 1
#include <omp.h>
#define cilk_spawn
#define cilk_sync
#define parallel_main main
#define parallel_for _Pragma("omp parallel for") for
#define parallel_for_1 _Pragma("omp parallel for schedule (static,1)") for
#define parallel_for_8 _Pragma("omp parallel for schedule (static,8)") for
#define parallel_for_256 _Pragma("omp parallel for schedule (static,256)") for

[[maybe_unused]] static int getWorkers() { return omp_get_max_threads(); }
[[maybe_unused]] static int getWorkerNum() { return omp_get_thread_num(); }

// c++
#else
#define cilk_spawn
#define cilk_sync
#define parallel_main main
#define parallel_for for
#define parallel_for_1 for
#define parallel_for_8 for
#define parallel_for_256 for
#define cilk_for for

[[maybe_unused]] static int getWorkers() { return 1; }
[[maybe_unused]] static int getWorkerNum() { return 0; }

#endif

#include <climits>

#if defined(LONG)
using intT = int32_t;
using uintT = uint32_t;
#define INT_T_MAX LONG_MAX
#define UINT_T_MAX ULONG_MAX
#else
using intT = int64_t;
using uintT = uint64_t;
#define INT_T_MAX INT_MAX
#define UINT_T_MAX UINT_MAX
#endif
