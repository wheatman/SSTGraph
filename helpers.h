#pragma once

#include "BitArray.hpp"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "SizedInt.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <sys/time.h>
#include <tuple>
#include <type_traits>
#include <vector>

template <typename T> T *newA(size_t n) { return (T *)malloc(n * sizeof(T)); }

using el_t = uint32_t;

template <typename T> struct is_tuple_impl : std::false_type {};

template <typename... Ts>
struct is_tuple_impl<std::tuple<Ts...>> : std::true_type {};

template <typename T> struct is_tuple : is_tuple_impl<std::decay_t<T>> {};

template <typename... Ts> static std::string TypeNameTuple(std::tuple<Ts...> x);

template <typename T> static std::string TypeName() {
  if constexpr (is_tuple<T>::value)
    return TypeNameTuple(T());
  if constexpr (std::is_same_v<T, bool>)
    return "bool";
  if constexpr (std::is_same_v<T, uint8_t>)
    return "uint8_t";
  if constexpr (std::is_same_v<T, uint16_t>)
    return "uint16_t";
  if constexpr (std::is_same_v<T, uint32_t>)
    return "uint32_t";
  if constexpr (std::is_same_v<T, uint64_t>)
    return "uint64_t";
  if constexpr (std::is_same_v<T, int8_t>)
    return "int8_t";
  if constexpr (std::is_same_v<T, int16_t>)
    return "int16_t";
  if constexpr (std::is_same_v<T, int32_t>)
    return "int32_t";
  if constexpr (std::is_same_v<T, int64_t>)
    return "int64_t";
  if constexpr (std::is_same_v<T, float>)
    return "float";
  if constexpr (std::is_same_v<T, double>)
    return "double";
  if constexpr (std::is_same_v<T, sized_uint<sizeof(T)>>)
    return T::name();
  return typeid(T).name();
}

template <typename... Ts>
static std::string TypeNameTuple([[maybe_unused]] std::tuple<Ts...> x) {
  std::string ret = ("" + ... + (", " + TypeName<Ts>()));
  return "{" + ret + "}";
};

// find index of first 1-bit (least significant bit)
static inline uint32_t bsf_word(uint32_t word) {
  uint32_t result;
  __asm__ volatile("bsf %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline uint32_t bsr_word(uint32_t word) {
  uint32_t result;
  __asm__ volatile("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}
#ifdef __AVX2__
uint32_t hsum_epi32_avx(__m128i x) {
  __m128i hi64 =
      _mm_unpackhi_epi64(x, x); // 3-operand non-destructive AVX lets us save a
                                // byte without needing a movdqa
  __m128i sum64 = _mm_add_epi32(hi64, x);
  __m128i hi32 = _mm_shuffle_epi32(
      sum64, _MM_SHUFFLE(2, 3, 0, 1)); // Swap the low two elements
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  return _mm_cvtsi128_si32(sum32); // movd
}
uint32_t hsum_8x32(__m256i v) {
  __m128i sum128 = _mm_add_epi32(
      _mm256_castsi256_si128(v),
      _mm256_extracti128_si256(v, 1)); // silly GCC uses a longer AXV512VL
                                       // instruction if AVX512 is enabled :/
  return hsum_epi32_avx(sum128);
}
uint32_t hsum_16x16(__m256i v) {
  __m256i sum8 = _mm256_hadd_epi16(v, v);
  __m256i sum4 = _mm256_hadd_epi16(sum8, sum8);
  __m256i sum2 = _mm256_hadd_epi16(sum4, sum4);
  __m256i sum = _mm256_hadd_epi16(sum2, sum2);
  return _mm256_extract_epi16(sum, 0);
}
float sum8(__m256 x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}

inline double hsum_double_avx(__m256d v) {
  __m128d vlow = _mm256_castpd256_pd128(v);
  __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
  vlow = _mm_add_pd(vlow, vhigh);              // reduce down to 128

  __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
  return _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); // reduce to scalar
}

template <typename T, typename V> T sum_256(V v) {
  if constexpr (std::is_same<T, double>::value) {
    return hsum_double_avx(v);
  }
  if constexpr (std::is_same<T, uint32_t>::value) {
    return hsum_8x32(v);
  }
  if constexpr (std::is_same<T, uint16_t>::value) {
    return hsum_16x16(v);
  }
  if constexpr (std::is_same<T, float>::value) {
    return sum8(v);
  }
}

template <int index_size> __m256i get_indexes(uint8_t *start) {
  static_assert(index_size > 0 && index_size <= 4);
  if constexpr (index_size == 1) {
    __m128i small_indexes = _mm_loadu_si128((__m128i *)start);
    return _mm256_cvtepu8_epi32(small_indexes);
  }
  if constexpr (index_size == 2) {
    __m128i small_indexes = _mm_loadu_si128((__m128i *)start);
    return _mm256_cvtepu16_epi32(small_indexes);
  }
  if constexpr (index_size == 3) {
    const __m128i shuf =
        _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    // constexpr __m256i mask = _mm256_set1_epi32(0xFFFFFF);
    // constexpr __m256i offsets = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18,
    // 21);
    __m128i data_vector = _mm_loadu_si128((__m128i *)start);
    __m128i indexes1 = _mm_shuffle_epi8(data_vector, shuf);
    data_vector = _mm_loadu_si128((__m128i *)(start + 12));
    __m128i indexes2 = _mm_shuffle_epi8(data_vector, shuf);
    return _mm256_set_m128i(indexes2, indexes1);
    // __m256i indexes = _mm256_and_si256(
    //     mask,
    //     _mm256_i32gather_epi32(
    //         (int *)(((uint8_t *)array.p_data) + i * index_size), offsets,
    //         1));
  }
  if constexpr (index_size == 4) {
    return _mm256_load_si256((__m256i *)start);
  }
}

template <class T> inline void Log(const __m256i &value) {
  const size_t n = sizeof(__m256i) / sizeof(T);
  T buffer[n];
  _mm256_storeu_si256((__m256i *)buffer, value);
  for (size_t i = 0; i < n; i++) {
    std::cout << +buffer[i] << " ";
  }
  std::cout << std::endl;
}

template <class T> inline void Log(const __m256d &value) {
  const size_t n = sizeof(__m256d) / sizeof(T);
  T buffer[n];
  _mm256_store_pd((double *)buffer, value);
  for (size_t i = 0; i < n; i++) {
    std::cout << buffer[i] << " ";
  }
  std::cout << std::endl;
}

template <class T> inline void Log(const __m256 &value) {
  const size_t n = sizeof(__m256) / sizeof(T);
  T buffer[n];
  _mm256_store_ps((float *)buffer, value);
  for (size_t i = 0; i < n; i++) {
    std::cout << buffer[i] << " ";
  }
  std::cout << std::endl;
}
#endif

static constexpr uint32_t bsr_word_constexpr(uint32_t word) {
  if (word == 0) {
    return 0;
  }
  if (word & (1U << 31U)) {
    return 31;
  } else {
    return bsr_word_constexpr(word << 1U) - 1;
  }
}

template <class T> inline bool writeMin(T *a, T b) {
  T c;
  bool r = false;
  do {
    c = *a;
  } while (c > b && !(r = __sync_bool_compare_and_swap(a, c, b)));
  return r;
}

int isPowerOfTwo(uint32_t x) { return ((x != 0U) && !(x & (x - 1U))); }

// same as find_leaf, but does it for any level in the tree
// index: index in array
// len: length of sub-level.
uint32_t find_node(uint32_t index, uint32_t len) { return (index / len) * len; }

uint64_t get_usecs() {
  struct timeval st {};
  gettimeofday(&st, nullptr);
  return st.tv_sec * 1000000 + st.tv_usec;
}

inline void segfault() {
  uint64_t x = 0;
  uint64_t *y = (uint64_t *)x;
  *y = 0;
}

// A structure that keeps a sequence of strings all allocated from
// the same block of memory
struct words {
  char *Chars = nullptr; // array storing all strings
  int64_t n = 0;         // total number of characters
  char **Strings =
      nullptr;   // pointers to strings (all should be null terminated)
  int64_t m = 0; // number of substrings
  words() = default;
  words(char *C, int64_t nn, char **S, int64_t mm)
      : Chars(C), n(nn), Strings(S), m(mm) {}
  void del() const {
    free(Chars);
    free(Strings);
  }
};

inline bool isSpace(char c) {
  switch (c) {
  case '\r':
  case '\t':
  case '\n':
  case 0:
  case ' ':
    return true;
  default:
    return false;
  }
}
// parallel code for converting a string to words
words stringToWords(char *Str, uint64_t n) {
  ParallelTools::parallel_for(0, n, [&](size_t i) {
    if (isSpace(Str[i])) {
      Str[i] = 0;
    }
  });

  // mark start of words
  BitArray FL(n);
  if (Str[0]) {
    FL.set(0);
  }

  // line up with cache line to run efficiently in parallel without races
  for (uint64_t i = 1; i < std::min(256UL, n); i++) {
    if (Str[i] && !Str[i - 1]) {
      FL.set(i);
    }
  }
  ParallelTools::parallel_for(
      256, n,
      [&](size_t i) {
        if (Str[i] && !Str[i - 1]) {
          FL.set(i);
        }
      },
      256);

  uint32_t worker_count = ParallelTools::getWorkers();
  std::vector<uint64_t> sub_counts(worker_count, 0);
  uint64_t section_count = (n / worker_count) + 1;
  ParallelTools::parallel_for(0, worker_count, [&](size_t i) {
    uint64_t start = i * section_count;
    uint64_t end = std::min((i + 1) * section_count, n);
    uint64_t local_count = 0;
    for (uint64_t j = start; j < end; j++) {
      if (FL.get(j)) {
        local_count += 1;
      }
    }
    sub_counts[i] = local_count;
  });
  // count and prefix sum
  for (uint32_t i = 1; i < worker_count; i++) {
    sub_counts[i] += sub_counts[i - 1];
  }
  uint64_t m = sub_counts[worker_count - 1];
  if (m == 0) {
    printf("file has no data, exiting\n");
    exit(-1);
  }

  uint64_t *offsets = newA<uint64_t>(m);
  if (offsets == nullptr) {
    printf("out of memory\n");
    exit(-1);
  }
  ParallelTools::parallel_for(0, worker_count, [&](size_t i) {
    uint64_t start = i * section_count;
    uint64_t end = std::min((i + 1) * section_count, n);
    uint64_t offset = 0;
    if (i != 0) {
      offset = sub_counts[i - 1];
    }
    for (uint64_t j = start; j < end; j++) {
      if (FL.get(j) == 1) {
        offsets[offset++] = j;
      }
    }
  });
  // pointer to each start of word
  char **SA = newA<char *>(m);
  if (SA == nullptr) {
    printf("out of memory, SA\n");
    exit(-1);
  }
  ParallelTools::parallel_for(0, m,
                              [&](size_t j) { SA[j] = Str + offsets[j]; });

  free(offsets);
  return words(Str, n, SA, m);
}
char *readStringFromFile(const char *fileName, int64_t *length) {
  std::ifstream file(fileName, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cout << "Unable to open file: " << fileName << std::endl;
    abort();
  }
  int64_t end = file.tellg();
  file.seekg(0, std::ios::beg);
  int64_t n = end - file.tellg();
  char *bytes = newA<char>(n + 1);
  file.read(bytes, n);
  file.close();
  *length = n;
  return bytes;
}

template <typename value_type = bool>
typename std::conditional<std::is_same<value_type, bool>::value,
                          std::tuple<el_t, el_t>,
                          std::tuple<el_t, el_t, value_type>>::type *
get_edges_from_file_mtx(const std::string &filename, uint64_t *edge_count,
                        el_t *rows, el_t *cols,
                        [[maybe_unused]] bool print = true) {
  using edge_type =
      typename std::conditional<std::is_same<value_type, bool>::value,
                                std::tuple<el_t, el_t>,
                                std::tuple<el_t, el_t, value_type>>::type;
  int64_t length = 0;
  char *S = readStringFromFile(filename.c_str(), &length);
  if (length == 0) {
    printf("file has 0 length, exiting\n");
    exit(-1);
  }
  // remove comments
  // a line before the data that starts with % and ends with \n
  // don't remove the first line which starts with %%
  int64_t i = 0;
  // find the start of the second line
  while (i < length) {
    if (S[i] == '\n') {
      i += 1;
      break;
    }
    i += 1;
  }
  if (i == length) {
    std::cout << "Bad file format, didnt' find any line breaks" << std::endl;
    exit(-1);
  }
  // clearing each line which starts with %
  while (i < length) {
    if (S[i] != '%') {
      break;
    }
    while (S[i] != '\n') {
      S[i] = 0;
      i += 1;
    }
    i += 1;
  }
  words W = stringToWords(S, length);
  if (strcmp(W.Strings[0], "\%\%MatrixMarket")) {
    std::cout << "Bad input file: missing header: \%\%MatrixMarket"
              << std::endl;
    exit(-1);
  }
  if (strcmp(W.Strings[1], "matrix")) {
    std::cout << "can only read matrix files so far in mtx format" << std::endl;
    exit(-1);
  }
  if (strcmp(W.Strings[2], "coordinate")) {
    std::cout << "can only read coordinate files so far in mtx format"
              << std::endl;
    exit(-1);
  }
  bool pattern = false;
  if (!strcmp(W.Strings[3], "integer")) {
    if constexpr (std::is_same<value_type, bool>::value) {
      std::cerr
          << "reading in a weighted graph to binary format, ignoring weights"
          << std::endl;
    } else if (!std::is_integral_v<value_type>) {
      std::cout << "interger type files must be read in with integer types"
                << std::endl;
      exit(-1);
    }
  } else if (!strcmp(W.Strings[3], "pattern")) {
    pattern = true;
    if (!std::is_same_v<value_type, bool>) {
      std::cerr << "no weights in file, using 1 for all weights" << std::endl;
    }
  } else if (!strcmp(W.Strings[3], "real")) {
    if constexpr (std::is_same<value_type, bool>::value) {
      std::cerr
          << "reading in a weighted graph to binary format, ignoring weights"
          << std::endl;
    } else if (!std::is_floating_point<value_type>::value) {
      std::cout << "real type files must be read in with floating point values"
                << std::endl;
      exit(-1);
    }
  }
  if (strcmp(W.Strings[4], "symmetric") && strcmp(W.Strings[4], "general")) {
    std::cout << "can only read symmetric and general files so far in mtx "
                 "format, got "
              << W.Strings[4] << std::endl;
    exit(-1);
  }
  int64_t m = strtol(W.Strings[7], nullptr, 10);
  int64_t r = strtol(W.Strings[5], nullptr, 10);
  int64_t c = strtol(W.Strings[6], nullptr, 10);

  uint32_t elements_per_line = 3;
  if (pattern) {
    elements_per_line = 2;
  }

  if (W.m - 8 != m * elements_per_line) {
    std::cout << "Bad input file: length = " << W.m - 8 << ", "
              << "elements per line " << elements_per_line << ", "
              << "m = " << elements_per_line * m << std::endl;
    std::cout << W.Strings[0] << ", " << W.Strings[1] << ", " << W.Strings[2]
              << ", " << W.Strings[3] << ", " << W.Strings[4] << ", "
              << W.Strings[5] << ", " << W.Strings[6] << ", " << W.Strings[7]
              << ", " << W.Strings[8] << std::endl;

    exit(-1);
  }
  m *= 2; // to symetrize
  edge_type *edges_array = (edge_type *)malloc(m * sizeof(*edges_array));
  if (edges_array == nullptr) {
    printf("out of memory, edges_array\n");
    exit(-1);
  }

  ParallelTools::parallel_for(0, m / 2, [&](size_t i) {
    el_t src = strtol(W.Strings[8 + i * elements_per_line], nullptr, 10);
    el_t dest = strtol(W.Strings[9 + i * elements_per_line], nullptr, 10);

    if constexpr (!std::is_same_v<value_type, bool>) {
      value_type val;
      if (pattern) {
        val = 1;
      } else {
        if constexpr (std::is_integral_v<value_type>) {
          val = strtol(W.Strings[10 + i * elements_per_line], nullptr, 10);
        }
        if constexpr (std::is_floating_point_v<value_type>) {
          val = strtod(W.Strings[10 + i * elements_per_line], nullptr);
        }
      }
      // one indexed
      edges_array[2 * i] = {src - 1, dest - 1, val};
      edges_array[2 * i + 1] = {dest - 1, src - 1, val};
    } else {
      // one indexed
      edges_array[2 * i] = {src - 1, dest - 1};
      edges_array[2 * i + 1] = {dest - 1, src - 1};
    }
  });

  W.del();
  *edge_count = m;
  *rows = r;
  *cols = c;
  return edges_array;
}

template <typename value_type = bool>
typename std::conditional<std::is_same<value_type, bool>::value,
                          std::tuple<el_t, el_t>,
                          std::tuple<el_t, el_t, value_type>>::type *
get_edges_from_file_adj_sym(const std::string &filename, uint64_t *edge_count,
                            uint32_t *node_count,
                            [[maybe_unused]] bool print = true) {
  using edge_type =
      typename std::conditional<std::is_same<value_type, bool>::value,
                                std::tuple<el_t, el_t>,
                                std::tuple<el_t, el_t, value_type>>::type;
  if constexpr (!std::is_integral_v<value_type>) {
    printf("get_edges_from_file_adj_sym can only do integral weights\n");
    exit(-1);
  }
  int64_t length = 0;
  char *S = readStringFromFile(filename.c_str(), &length);
  if (length == 0) {
    printf("file has 0 length, exiting\n");
    exit(-1);
  }
  words W = stringToWords(S, length);
  if (strcmp(W.Strings[0], "AdjacencyGraph") &&
      strcmp(W.Strings[0], "WeightedAdjacencyGraph")) {
    std::cout << "Bad input file: missing header, got " << W.Strings[0]
              << std::endl;
    exit(-1);
  }
  if constexpr (std::is_same<value_type, bool>::value) {
    if (!strcmp(W.Strings[0], "WeightedAdjacencyGraph")) {
      std::cerr
          << "reading in a weighted graph to binary format, ignoring weights"
          << std::endl;
    }
  }
  bool pattern = false;
  if constexpr (!std::is_same<value_type, bool>::value) {
    if (!strcmp(W.Strings[0], "AdjacencyGraph")) {
      std::cerr << "trying reading in a binary graph to weighted format, using "
                   "1 for all weights"
                << std::endl;
      pattern = true;
    }
  }
  uint64_t len = W.m - 1;
  if (len == 0) {
    printf("the file appears to have no data, exiting\n");
    exit(-1);
  }
  uint64_t *In = newA<uint64_t>(len);
  ParallelTools::parallel_for(
      0, len, [&](size_t i) { In[i] = strtol(W.Strings[i + 1], nullptr, 10); });
  W.del();
  uint64_t n = In[0];
  uint64_t m = In[1];
  if (n == 0 || m == 0) {
    printf("the file says we have no edges or vertices, exiting\n");
    free(In);
    exit(-1);
  }

  if (len != n + m + 2 && len != n + 2 * m + 2) {
    std::cout << "n = " << n << " m = " << m << std::endl;
    std::cout << "Bad input file: length = " << len << " n+m+2 = " << n + m + 2
              << std::endl;
    std::cout << "or: length = " << len << " n+2*m+2 = " << n + 2 * m + 2
              << std::endl;
    free(In);
    exit(-1);
  }
  uint64_t *offsets = In + 2;
  uint64_t *edges = In + 2 + n;
  uint64_t *weights = In + 2 + n + m;
  edge_type *edges_array = (edge_type *)malloc(m * sizeof(*edges_array));
  ParallelTools::parallel_for(0, n, [&](size_t i) {
    uint64_t o = offsets[i];
    uint64_t l = ((i == n - 1) ? m : offsets[i + 1]) - offsets[i];
    for (uint64_t j = o; j < o + l; j++) {
      if constexpr (std::is_same<value_type, bool>::value) {
        edges_array[j] = {i, static_cast<el_t>(edges[j])};
      } else {
        if (pattern) {
          edges_array[j] = {i, static_cast<el_t>(edges[j]),
                            static_cast<value_type>(1)};
        } else {
          edges_array[j] = {i, static_cast<el_t>(edges[j]),
                            static_cast<value_type>(weights[j])};
        }
      }
    }
  });
  *edge_count = m;
  *node_count = n;
  free(In);
  return edges_array;
}

static inline bool endsWith(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() &&
         0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

template <typename value_type = bool>
typename std::conditional<std::is_same<value_type, bool>::value,
                          std::tuple<el_t, el_t>,
                          std::tuple<el_t, el_t, value_type>>::type *
get_edges_from_file(const std::string &filename, uint64_t *edge_count,
                    uint32_t *node_count, bool print = true) {
  if (endsWith(filename, ".adj") || endsWith(filename, ".adj.shuf")) {
    return get_edges_from_file_adj_sym<value_type>(filename, edge_count,
                                                   node_count, print);
  }
  if (endsWith(filename, ".mtx")) {
    uint32_t rows = 0;
    uint32_t cols = 0;
    auto ret = get_edges_from_file_mtx<value_type>(filename, edge_count, &rows,
                                                   &cols, print);
    if (rows != cols) {
      printf("not square\n");
      return nullptr;
    }
    *node_count = rows;
    return ret;
  }
  printf("can't read that format\n");
  return nullptr;
}

template <class ET> inline bool CAS(ET *ptr, ET oldv, ET newv) {
  if constexpr (sizeof(ET) == 1) {
    return __sync_bool_compare_and_swap((bool *)ptr, *((bool *)&oldv),
                                        *((bool *)&newv));
  } else if constexpr (sizeof(ET) == 4) {
    return __sync_bool_compare_and_swap((int *)ptr, *((int *)&oldv),
                                        *((int *)&newv));
  } else if constexpr (sizeof(ET) == 8) {
    return __sync_bool_compare_and_swap((long *)ptr, *((long *)&oldv),
                                        *((long *)&newv));
  } else {
    std::cout << "CAS bad length : " << sizeof(ET) << std::endl;
    abort();
  }
}

bool approximatelyEqual(double a, double b,
                        double epsilon = std::numeric_limits<float>::epsilon() *
                                         100) {
  // if they are both ing or nan ignore and don't bother checking
  if (!std::isfinite(a) && !std::isfinite(b)) {
    return true;
  }
  return std::fabs(a - b) <=
         ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) *
          epsilon);
}

inline std::string Join(std::vector<std::string> const &elements,
                        const char delimiter) {
  std::string out;
  for (size_t i = 0; i < elements.size() - 1; i++) {
    out += elements[i] + delimiter;
  }
  out += elements[elements.size() - 1];
  return out;
}

template <class T>
std::vector<T> create_random_data(size_t n, size_t max_val,
                                  std::seed_seq &seed) {
  std::mt19937 eng(seed); // a source of random data

  std::uniform_int_distribution<T> dist(1, max_val);
  std::vector<T> v(n);

  generate(begin(v), end(v), bind(dist, eng));
  return v;
}

template <typename T1, typename... Ts>
std::tuple<Ts...> leftshift_tuple(const std::tuple<T1, Ts...> &tuple) {
  return std::apply([](auto &&, auto &...args) { return std::tie(args...); },
                    tuple);
}

template <typename T1, typename T2, typename... Ts>
std::tuple<T1, Ts...>
remove_second_from_tuple(const std::tuple<T1, T2, Ts...> &tuple) {
  return std::apply([](auto &first, [[maybe_unused]] auto &&second,
                       auto &...args) { return std::tie(first, args...); },
                    tuple);
}

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t);
template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t);

// helper function to print a tuple of any size
template <class Tuple, std::size_t N> struct TuplePrinter {
  static void print(std::ostream &os, const Tuple &t) {
    using e_type = decltype(std::get<N - 1>(t));
    TuplePrinter<Tuple, N - 1>::print(os, t);
    if constexpr (sizeof(e_type) == 1) {
      os << static_cast<int64_t>(std::get<N - 1>(t));
    } else {
      os << ", " << std::get<N - 1>(t);
    }
  }
};

template <class Tuple> struct TuplePrinter<Tuple, 1> {
  static void print(std::ostream &os, const Tuple &t) {
    using e_type = decltype(std::get<0>(t));
    if constexpr (sizeof(e_type) == 1) {
      os << static_cast<int64_t>(std::get<0>(t));
    } else {
      os << std::get<0>(t);
    }
  }
};

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int>>
std::ostream &operator<<(std::ostream &os,
                         [[maybe_unused]] const std::tuple<Args...> &t) {
  os << "()";
  return os;
}

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int>>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t) {
  os << "(";
  TuplePrinter<decltype(t), sizeof...(Args)>::print(os, t);
  os << ")";
  return os;
}

template <std::integral T>
std::vector<T> create_random_data(size_t n,
                                  T max_val = std::numeric_limits<T>::max(),
                                  uint32_t seed = 0) {
  std::mt19937 rng(seed);

  std::uniform_int_distribution<T> dist(0, max_val);
  std::vector<T> v(n);

  generate(begin(v), end(v), bind(dist, rng));
  return v;
}

template <std::floating_point T>
std::vector<T> create_random_data(size_t n,
                                  T max_val = std::numeric_limits<T>::max(),
                                  uint32_t seed = 0) {
  std::mt19937 rng(seed);

  std::uniform_real_distribution<T> dist(0, max_val);
  std::vector<T> v(n);

  generate(begin(v), end(v), bind(dist, rng));
  return v;
}

template <class T>
void wrapArrayInVector(T *sourceArray, size_t arraySize,
                       std::vector<T, std::allocator<T>> &targetVector) {
  typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *vectorPtr =
      (typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *)((
          void *)&targetVector);
  vectorPtr->_M_start = sourceArray;
  vectorPtr->_M_finish = vectorPtr->_M_end_of_storage =
      vectorPtr->_M_start + arraySize;
}

template <class T>
void releaseVectorWrapper(std::vector<T, std::allocator<T>> &targetVector) {
  typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *vectorPtr =
      (typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *)((
          void *)&targetVector);
  vectorPtr->_M_start = vectorPtr->_M_finish = vectorPtr->_M_end_of_storage =
      nullptr;
}
