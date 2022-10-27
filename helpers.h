#ifndef HELPERS_H
#define HELPERS_H

#include "BitArray.hpp"
#include "parallel.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <sys/time.h>
#include <vector>

template <typename T> T *newA(size_t n) { return (T *)malloc(n * sizeof(T)); }

using el_t = uint32_t;

// default implementation
template <typename T> struct TypeName {
  static std::string Get() { return typeid(T).name(); }
};
template <> struct TypeName<bool> {
  static std::string Get() { return "bool"; }
};
template <> struct TypeName<uint8_t> {
  static std::string Get() { return "uint8_t"; }
};
template <> struct TypeName<uint16_t> {
  static std::string Get() { return "uint16_t"; }
};
template <> struct TypeName<uint32_t> {
  static std::string Get() { return "uint32_t"; }
};
template <> struct TypeName<uint64_t> {
  static std::string Get() { return "uint64_t"; }
};
template <> struct TypeName<int8_t> {
  static std::string Get() { return "int8_t"; }
};
template <> struct TypeName<int16_t> {
  static std::string Get() { return "int16_t"; }
};
template <> struct TypeName<int32_t> {
  static std::string Get() { return "int32_t"; }
};
template <> struct TypeName<int64_t> {
  static std::string Get() { return "int64_t"; }
};
template <> struct TypeName<float> {
  static std::string Get() { return "float"; }
};
template <> struct TypeName<double> {
  static std::string Get() { return "double"; }
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

uint64_t get_usecs() {
  struct timeval st {};
  gettimeofday(&st, nullptr);
  return st.tv_sec * 1000000 + st.tv_usec;
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
  parallel_for(uint64_t i = 0; i < n; i++) {
    if (isSpace(Str[i])) {
      Str[i] = 0;
    }
  }

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
  parallel_for_256(uint64_t i = 256; i < n; i++) {
    if (Str[i] && !Str[i - 1]) {
      FL.set(i);
    }
  }

  uint32_t worker_count = getWorkers();
  std::vector<uint64_t> sub_counts(worker_count, 0);
  uint64_t section_count = (n / worker_count) + 1;
  parallel_for_1(uint64_t i = 0; i < worker_count; i++) {
    uint64_t start = i * section_count;
    uint64_t end = std::min((i + 1) * section_count, n);
    uint64_t local_count = 0;
    for (uint64_t j = start; j < end; j++) {
      if (FL.get(j)) {
        local_count += 1;
      }
    }
    sub_counts[i] = local_count;
  }
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
  parallel_for_1(uint64_t i = 0; i < worker_count; i++) {
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
  }
  // pointer to each start of word
  char **SA = newA<char *>(m);
  if (SA == nullptr) {
    printf("out of memory, SA\n");
    exit(-1);
  }
  parallel_for(uint64_t j = 0; j < m; j++) { SA[j] = Str + offsets[j]; }

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
  //{parallel_for(uint64_t i=0; i < len; i++) In[i] = atol(W.Strings[i + 1]);}

  uint32_t elements_per_line = 3;
  if (pattern) {
    elements_per_line = 2;
  }

  if (W.m - 8 != m * elements_per_line) {
    std::cout << "Bad input file: length = " << W.m - 8 << elements_per_line << "m = " << elements_per_line * m
              << std::endl;
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


  parallel_for(int64_t i = 0; i < m / 2; i++) {
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
  }

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
  parallel_for(uint64_t i = 0; i < len; i++) {
    In[i] = strtol(W.Strings[i + 1], nullptr, 10);
  }
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
  parallel_for(uint32_t i = 0; i < n; i++) {
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
  }
  *edge_count = m;
  *node_count = n;
  free(In);
  return edges_array;
}

static bool endsWith(const std::string &str, const std::string &suffix) {
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

#endif
