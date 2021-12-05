#pragma once
// assumes endianess
#include "helpers.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <malloc.h>

static constexpr uint32_t MIN_SIZE = 16;
using block_t = uint64_t;

template <typename p>
p PackedArray_get(const block_t *const array, uint32_t i, uint32_t n) {
  uint8_t *loc1 = ((uint8_t *)array) + (p::i_size * i);
  uint8_t *loc2 = ((uint8_t *)array) + (p::i_size * n) + (p::v_size * i);
  return p(loc1, loc2);
}
template <typename p>
void *PackedArray_values_start(const block_t *const array, uint32_t n) {
  return (void *)(((uint8_t *)array) + (p::i_size * n));
}

template <int B>
uint32_t PackedArray_get(const block_t *const array, uint32_t i) {
  block_t *aligned_array = (block_t *)__builtin_assume_aligned(array, 16);
  uint32_t val;
  memcpy(&val, ((uint8_t *)aligned_array) + (B * i), B);
  return val;
}

template <typename p>
void PackedArray_set(const block_t *const array, uint32_t i, uint32_t n,
                     p pair) {
  uint8_t *loc1 = ((uint8_t *)array) + (p::i_size * i);
  uint8_t *loc2 = ((uint8_t *)array) + (p::i_size * n) + (p::v_size * i);
  pair.write(loc1, loc2);
}

template <typename pair_type>
void PackedArray_memset_0(block_t *const array, uint64_t start, uint64_t end,
                          uint32_t n) {
  std::memset(((uint8_t *)array) + (start * pair_type::i_size), 0,
              (end - start) * pair_type::i_size);
  std::memset(((uint8_t *)array) + (n * pair_type::i_size) +
                  (start * pair_type::v_size),
              0, (end - start) * pair_type::v_size);
}

template <typename pair_type> uint64_t PackedArray_get_size(uint32_t n) {
  uint32_t n_rounded = 1U << bsr_word(n);
  if (n_rounded < n) {
    n_rounded <<= 1U;
  }
  n = n_rounded;
  if (n < MIN_SIZE) {
    n = MIN_SIZE;
  }
  uint64_t size = pair_type::size * n;
  return size;
}

template <typename pair_type> block_t *create_PackedArray(uint32_t n) {
  uint32_t size = PackedArray_get_size<pair_type>(n);
  if constexpr (pair_type::i_size == 3) {
    size += 4; // to help not read of the end since I read 12 bytes with 16 byte
               // chunks
  }
  block_t *array = (block_t *)memalign(32, size);
  std::memset(array, 0, size);
  return array;
}

template <typename pair_type>
void PackedArray_memcpy_aligned(block_t *const dest,
                                const block_t *const source,
                                uint32_t length_to_copy, uint32_t old_n,
                                uint32_t new_n) {
  memcpy(dest, source, length_to_copy * pair_type::i_size);
  uint8_t *d = reinterpret_cast<uint8_t *>(dest) + new_n * pair_type::i_size;
  uint8_t const *s =
      reinterpret_cast<const uint8_t *>(source) + old_n * pair_type::i_size;
  memcpy(d, s, length_to_copy * pair_type::v_size);
}

// assumes old_length is a power of 2 greater than 32
template <typename pair_type>
block_t *PackedArray_double(block_t *array, uint32_t old_length) {
  uint32_t new_length = old_length * 2;
  uint32_t size = pair_type::size * new_length;
  if constexpr (!(pair_type::size == 1 || pair_type::size == 2 ||
                  pair_type::size == 4 || pair_type::size == 8)) {
    // when we can't read by machine sizes we might read some extra space off
    // the end so make sure we alloc it
    size += 8;
  }
  block_t *new_array = (block_t *)memalign(32, size);
  block_t *old_array = array;
  array = new_array;
  PackedArray_memcpy_aligned<pair_type>(array, old_array, old_length,
                                        old_length, new_length);
  free(old_array);
  PackedArray_memset_0<pair_type>(array, old_length, new_length, new_length);
  return array;
}

// assumes old_length is a power of 2 greater than 64
template <typename pair_type>
block_t *PackedArray_half(block_t *array, uint32_t new_length) {
  uint32_t old_length = new_length * 2;
  uint32_t size = (pair_type::size * new_length);
  if constexpr (!(pair_type::size == 1 || pair_type::size == 2 ||
                  pair_type::size == 4 || pair_type::size == 8)) {
    // when we can't read by machine sizes we might read some extra space off
    // the end so make sure we alloc it
    size += 8;
  }
  block_t *new_array = (block_t *)memalign(32, size);
  block_t *old_array = array;
  array = new_array;
  PackedArray_memcpy_aligned<pair_type>(array, old_array, old_length / 2,
                                        old_length, old_length / 2);
  free(old_array);
  return array;
}

// TODO(wheatman) fix bug when end == 0
template <typename pair_type>
void PackedArray_slide_right(block_t *const array, uint32_t index, uint32_t end,
                             uint32_t n) {

  memmove(&(((uint8_t *)array)[pair_type::i_size * (index + 1)]),
          &(((uint8_t *)array)[pair_type::i_size * (index)]),
          (pair_type::i_size * ((end - index) - 1)));

  memmove(
      &(((uint8_t *)
             array)[pair_type::i_size * n + pair_type::v_size * (index + 1)]),
      &(((uint8_t *)
             array)[pair_type::i_size * n + pair_type::v_size * (index)]),
      (pair_type::v_size * ((end - index) - 1)));

  PackedArray_set<pair_type>(array, index, n, pair_type());
}

// TODO(wheatman) fix bug when end == 0
template <typename pair_type>
void PackedArray_slide_left(block_t *const array, uint32_t index, uint32_t end,
                            uint32_t n) {

  memmove(&(((uint8_t *)array)[pair_type::i_size * (index)]),
          &(((uint8_t *)array)[pair_type::i_size * (index + 1)]),
          (pair_type::i_size * ((end - index) - 1)));

  memmove(
      &(((uint8_t *)
             array)[pair_type::i_size * n + pair_type::v_size * (index)]),
      &(((uint8_t *)
             array)[pair_type::i_size * n + pair_type::v_size * (index + 1)]),
      (pair_type::v_size * ((end - index) - 1)));

  PackedArray_set<pair_type>(array, end - 1, n, pair_type());
}

// assumes the allocated memory is a power of 2 in length
// num_elements is the number of elements that will be there after the insert
template <typename pair_type>
block_t *PackedArray_insert(block_t *array, pair_type e, uint32_t num_elements,
                            uint32_t index, uint32_t n,
                            bool double_pos = true) {
  if (num_elements >= MIN_SIZE && __builtin_popcount(num_elements - 1) == 1 &&
      double_pos) {
    // TODO(wheatman) could be more efficient if combine insert with move
    array = PackedArray_double<pair_type>(array, num_elements - 1);
    n *= 2;
  }
  if (index < num_elements) {
    PackedArray_slide_right<pair_type>(array, index, num_elements, n);
  }
  PackedArray_set<pair_type>(array, index, n, e);
  return array;
}

// num_elements is the number of elements that are there before the insert
template <typename pair_type>
block_t *PackedArray_insert_end(block_t *array, pair_type e,
                                uint32_t num_elements, uint32_t n) {
  if (num_elements >= MIN_SIZE && __builtin_popcount(num_elements) == 1) {
    array = PackedArray_double<pair_type>(array, num_elements);
    n *= 2;
  }
  PackedArray_set<pair_type>(array, num_elements, n, e);
  return array;
}

// num_elements is the number of elements that will be there after the remove
template <typename pair_type>
block_t *PackedArray_remove(block_t *array, uint32_t num_elements,
                            uint32_t index, uint32_t n, bool half_pos = true) {
  if (index < num_elements) {
    PackedArray_slide_left<pair_type>(array, index, num_elements + 1, n);
  } else if (index == num_elements) {
    PackedArray_set<pair_type>(array, index, n, pair_type());
  }
  if (num_elements + 1 >= MIN_SIZE && __builtin_popcount(num_elements) == 1 &&
      half_pos) {
    // TODO(wheatman) could be more efficient if combine insert with move
    array = PackedArray_half<pair_type>(array, num_elements);
  }
  return array;
}
