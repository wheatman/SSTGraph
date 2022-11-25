#pragma once
#include "helpers.h"

#include "TinySet_small.h"
#include <cassert>

template <typename value_type>
uint8_t TinySetV_small<value_type>::find_b_for_el_count(uint32_t element_count,
                                                        const extra_data &d) {
  if (element_count < d.thresh_24) {
    return 4;
  } else if (element_count < d.thresh_16) {
    return 3;
  } else if (element_count < d.thresh_8) {
    return 2;
  } else {
    return 1;
  }
}

template <typename value_type>
typename TinySetV_small<value_type>::extra_data
TinySetV_small<value_type>::get_thresholds(uint64_t element_count) const {
  extra_data d = {};
  d.max_el = element_count;
  d.thresh_24 = 2;
  uint32_t current_b = 32;
  for (uint32_t el_count = 2; el_count < UINT32_MAX / 2; el_count *= 2) {
    uint32_t best_b =
        find_best_b_for_given_element_count(el_count, {4, 3, 2, 1}, d.max_el);
    if (best_b < current_b) {
      current_b = best_b;
      switch (current_b) {
      case 3:
        d.thresh_24 = el_count * 2;
        break;
      case 2:
        d.thresh_16 = el_count * 2;
        break;
      case 1:
        d.thresh_8 = el_count * 2;
        break;
      }
    }
  }
#if TINYSET_32 == 1
  d.thresh_24 = UINT32_MAX / 4;
#endif
  if (d.thresh_16 <= d.thresh_24) {
    d.thresh_16 = d.thresh_24 * 2;
  }
  if (d.thresh_8 <= d.thresh_16) {
    d.thresh_8 = d.thresh_16 * 2;
  }
  return d;
}

template <typename value_type>
uint32_t TinySetV_small<value_type>::find_best_b_for_given_element_count(
    uint64_t element_count, const std::vector<uint32_t> &b_options,
    uint64_t max_val) const {
  if (b_options.empty()) {
    printf("find_best_b needs at least 1 option\n");
    exit(-1);
  }
  std::vector<uint64_t> size_guesses(b_options.size());
  for (uint64_t i = 0; i < b_options.size(); i++) {
    uint32_t b = b_options[i];
    uint64_t elements_space = (element_count * b) * 2; // mul 2 for extra spaces
    uint64_t num_pmas = calc_pma_count(max_val, b);
    // they are all the same size so just pick one
    uint64_t pma_space = num_pmas * sizeof(PMA<4, value_type>);
    if (num_pmas == 1) {
      pma_space = 0;
    }
    size_guesses[i] = elements_space + pma_space;
  }
  uint32_t best_b = b_options[0];
  uint64_t best_mem = size_guesses[0];
  for (uint64_t i = 1; i < b_options.size(); i++) {
    if (size_guesses[i] < best_mem) {
      best_mem = size_guesses[i];
      best_b = b_options[i];
    }
  }
  return best_b;
}

template <typename value_type>
uint32_t TinySetV_small<value_type>::calc_pma_count(uint64_t max_el,
                                                    uint32_t b) const {
  uint64_t n = max_el;
  uint64_t pma_count = std::max(1UL, n >> (b * 8));
  if ((pma_count << (b * 8)) < n) {
    pma_count += 1;
  }
  return pma_count;
}

template <typename value_type>
template <class F, typename p>
void TinySetV_small<value_type>::map_set_type(F &f, const extra_data &d,
                                              bool parallel) const {
  uint32_t iters = get_pma_count(d);
  const p *ps = reinterpret_cast<const p *>((iters == 1) ? &pmas.d[0] : pmas.p);
  uint64_t b_ = pmas.get_b();
  if (!parallel || iters == 1 || el_count <= 1000) {
    for (uint64_t i = 0; i < iters; i += 1) {
      uint64_t prefix = i << (b_ * 8);
      if constexpr (F::no_early_exit) {
        ps[i].template map<F>(f, prefix);
      } else {
        if (ps[i].template map<F>(f, prefix)) {
          break;
        }
      }
    }
  } else {
    parallel_for(uint64_t i = 0; i < iters; i += 1) {
      uint64_t prefix = i << (b_ * 8);
      ps[i].template map<F>(f, prefix);
    }
  }
}

template <typename value_type>
template <class F>
void TinySetV_small<value_type>::map(F &f, const extra_data &d,
                                     bool parallel) const {
  switch (pmas.get_b()) {
  case 1:
    return map_set_type<F, PMA<1, value_type>>(f, d, parallel);
  case 2:
    return map_set_type<F, PMA<2, value_type>>(f, d, parallel);
  case 3:
    return map_set_type<F, PMA<3, value_type>>(f, d, parallel);
  case 4:
    return map_set_type<F, PMA<4, value_type>>(f, d, parallel);
  }
}

enum which_done { FIRST, SECOND, BOTH, COMPLETE };

template <typename i1, typename i2>
static which_done set_intersection_count_body_intersection(
    uint64_t *count, i1 &it1, i2 &it2, i1 &it1_end, i2 &it2_end,
    uint64_t this_prefix, uint64_t other_prefix, uint64_t this_region_size,
    uint64_t other_region_size, uint32_t early_end_A, uint32_t early_end_B) {
  uint64_t local_count = 0;
  if ((it1 < it1_end) && (it2 < it2_end)) {
    uint64_t this_val = (*it1).first + this_prefix;
    uint64_t other_val = (*it2).first + other_prefix;
    if (this_val >= early_end_A || other_val >= early_end_B) {
      *count += local_count;
      return COMPLETE;
    }
    while (true) {
      local_count += (this_val == other_val);
      uint64_t original_this_val = this_val;
      if (this_val <= other_val) {
        ++it1;
        if (!(it1 < it1_end)) {
          break;
        }
        this_val = (*it1).first + this_prefix;
        if (this_val >= early_end_A) {
          *count += local_count;
          return COMPLETE;
        }
      }
      if (other_val <= original_this_val) {
        ++it2;
        if (!(it2 < it2_end)) {
          break;
        }
        other_val = (*it2).first + other_prefix;
        if (other_val >= early_end_B) {
          *count += local_count;
          return COMPLETE;
        }
      }
    }
  }
  *count += local_count;
  if (this_prefix + this_region_size == other_prefix + other_region_size) {
    return BOTH;
  }
  if (it1 < it1_end) {
    return SECOND;
  } else if (it2 < it2_end) {
    return FIRST;
  }
  return BOTH;
}

template <typename value_type>
template <typename p1, typename p2>
uint64_t TinySetV_small<value_type>::set_intersection_count_body(
    const extra_data &this_d, const TinySetV_small &other,
    const extra_data &other_d, uint32_t early_end_A,
    uint32_t early_end_B) const {
  static_assert(
      binary,
      "set_intersection_count only implemented for binary mode for now");
  uint64_t count = 0;
  uint64_t pma_count_this = get_pma_count(this_d);
  uint64_t pma_count_other = other.get_pma_count(other_d);
  uint64_t this_region_size = pma_region_size();
  uint64_t other_region_size = other.pma_region_size();
  uint64_t iter_this = 0;
  uint64_t iter_other = 0;
  const p1 *this_pmas = (pma_count_this == 1)
                            ? reinterpret_cast<const p1 *>(pmas.d)
                            : reinterpret_cast<const p1 *>(pmas.p);
  const p2 *other_pmas = (pma_count_other == 1)
                             ? reinterpret_cast<const p2 *>(other.pmas.d)
                             : reinterpret_cast<const p2 *>(other.pmas.p);

  bool this_fast = this_pmas[0].use_fast_iter();
  bool other_fast = other_pmas[0].use_fast_iter();

  typename p1::iterator pma_iterator_this = this_pmas[0].begin(true);
  typename p2::iterator pma_iterator_other = other_pmas[0].begin(true);
  typename p1::iterator pma_iterator_this_end = this_pmas[0].end();
  typename p2::iterator pma_iterator_other_end = other_pmas[0].end();
  typename p1::iterator_fast pma_iterator_fast_this = this_pmas[0].begin_fast();
  typename p2::iterator_fast pma_iterator_fast_other =
      other_pmas[0].begin_fast();
  typename p1::iterator_fast pma_iterator_fast_this_end =
      this_pmas[0].end_fast();
  typename p2::iterator_fast pma_iterator_fast_other_end =
      other_pmas[0].end_fast();

  uint64_t this_prefix = this_region_size * iter_this;
  uint64_t other_prefix = other_region_size * iter_other;
  uint64_t last_done = 0;

  if (this_pmas[0].has(0) && other_pmas[0].has(0)) {
    if (early_end_A > 0 && early_end_B > 0) {
      count++;
    } else {
      return count;
    }
  }

  which_done done;
  while (iter_this < pma_count_this && iter_other < pma_count_other) {
    if (this_fast) {
      if (other_fast) {
        done = set_intersection_count_body_intersection<
            typename p1::iterator_fast, typename p2::iterator_fast>(
            &count, pma_iterator_fast_this, pma_iterator_fast_other,
            pma_iterator_fast_this_end, pma_iterator_fast_other_end,
            this_prefix, other_prefix, this_region_size, other_region_size,
            early_end_A, early_end_B);
      } else {
        done =
            set_intersection_count_body_intersection<typename p1::iterator_fast,
                                                     typename p2::iterator>(
                &count, pma_iterator_fast_this, pma_iterator_other,
                pma_iterator_fast_this_end, pma_iterator_other_end, this_prefix,
                other_prefix, this_region_size, other_region_size, early_end_A,
                early_end_B);
      }
    } else {
      if (other_fast) {
        done = set_intersection_count_body_intersection<
            typename p1::iterator, typename p2::iterator_fast>(
            &count, pma_iterator_this, pma_iterator_fast_other,
            pma_iterator_this_end, pma_iterator_fast_other_end, this_prefix,
            other_prefix, this_region_size, other_region_size, early_end_A,
            early_end_B);
      } else {
        done = set_intersection_count_body_intersection<typename p1::iterator,
                                                        typename p2::iterator>(
            &count, pma_iterator_this, pma_iterator_other,
            pma_iterator_this_end, pma_iterator_other_end, this_prefix,
            other_prefix, this_region_size, other_region_size, early_end_A,
            early_end_B);
      }
    }
    switch (done) {
    case BOTH:
    case FIRST:
      iter_this += 1;
      this_prefix = this_region_size * iter_this;
      if (iter_this < pma_count_this) {
        pma_iterator_this = this_pmas[iter_this].begin(true);
        pma_iterator_this_end = this_pmas[iter_this].end();
        this_fast = this_pmas[iter_this].use_fast_iter();
        pma_iterator_fast_this = this_pmas[iter_this].begin_fast();
        pma_iterator_fast_this_end = this_pmas[iter_this].end_fast();
        // TODO(wheatman) this is technically a log n operation to do the other
        // lookup, maybe think of something smarter to do
        if (this_pmas[iter_this].has(0) && other.has(this_prefix, other_d) &&
            this_prefix != last_done) {
          if (early_end_A > this_prefix && early_end_B > this_prefix) {
            count++;
            last_done = this_prefix;
          } else {
            return count;
          }
        }
      } else {
        return count;
      }
      if (done == FIRST) {
        break;
      }
      [[fallthrough]];
    case SECOND:
      iter_other += 1;
      other_prefix = other_region_size * iter_other;
      if (iter_other < pma_count_other) {
        pma_iterator_other = other_pmas[iter_other].begin(true);
        pma_iterator_other_end = other_pmas[iter_other].end();
        other_fast = other_pmas[iter_other].use_fast_iter();
        pma_iterator_fast_other = other_pmas[iter_other].begin_fast();
        pma_iterator_fast_other_end = other_pmas[iter_other].end_fast();
        // TODO(wheatman) this is technically a log n operation to do the other
        // lookup, maybe think of something smarter to do
        if (other_pmas[iter_other].has(0) && has(other_prefix, this_d) &&
            other_prefix != last_done) {
          if (early_end_A > other_prefix && early_end_B > other_prefix) {
            count++;
            last_done = other_prefix;
          } else {
            return count;
          }
        }
      } else {
        return count;
      }
      break;
    case COMPLETE:
      return count;
    }
  }
  return count;
}

template <typename value_type>
uint64_t TinySetV_small<value_type>::Set_Intersection_Count(
    const extra_data &this_d, const TinySetV_small &other,
    const extra_data &other_d, uint32_t early_end_A,
    uint32_t early_end_B) const {
  static_assert(
      binary,
      "set_intersection_count only implemented for binary mode for now");
  switch (pmas.get_b()) {
  case 1:
    switch (other.pmas.get_b()) {
    case 1:
      return set_intersection_count_body<PMA<1, value_type>,
                                         PMA<1, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 2:
      return set_intersection_count_body<PMA<1, value_type>,
                                         PMA<2, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 3:
      return set_intersection_count_body<PMA<1, value_type>,
                                         PMA<3, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 4:
      return set_intersection_count_body<PMA<1, value_type>,
                                         PMA<4, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    }
    __builtin_unreachable();
    return 0;
  case 2:
    switch (other.pmas.get_b()) {
    case 1:
      return set_intersection_count_body<PMA<2, value_type>,
                                         PMA<1, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 2:
      return set_intersection_count_body<PMA<2, value_type>,
                                         PMA<2, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 3:
      return set_intersection_count_body<PMA<2, value_type>,
                                         PMA<3, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 4:
      return set_intersection_count_body<PMA<2, value_type>,
                                         PMA<4, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    }
    __builtin_unreachable();
    return 0;
  case 3:
    switch (other.pmas.get_b()) {
    case 1:
      return set_intersection_count_body<PMA<3, value_type>,
                                         PMA<1, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 2:
      return set_intersection_count_body<PMA<3, value_type>,
                                         PMA<2, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 3:
      return set_intersection_count_body<PMA<3, value_type>,
                                         PMA<3, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 4:
      return set_intersection_count_body<PMA<3, value_type>,
                                         PMA<4, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    }
    __builtin_unreachable();
    return 0;
  case 4:
    switch (other.pmas.get_b()) {
    case 1:
      return set_intersection_count_body<PMA<4, value_type>,
                                         PMA<1, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 2:
      return set_intersection_count_body<PMA<4, value_type>,
                                         PMA<2, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 3:
      return set_intersection_count_body<PMA<4, value_type>,
                                         PMA<3, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 4:
      return set_intersection_count_body<PMA<4, value_type>,
                                         PMA<4, value_type>>(
          this_d, other, other_d, early_end_A, early_end_B);
    }
    break;
  }
  __builtin_unreachable();
  return 0;
}

template <typename value_type>
void TinySetV_small<value_type>::print(const extra_data &d) const {
  printf("tinyset: element_count = %u, B = %u\n", el_count, pmas.get_b());
  if (pmas.get_b() > 4) {
    std::cout << "B is too big" << std::endl;
    return;
  }
  for (auto it = begin(d); it != end(d); ++it) {
    if constexpr (binary) {
      printf("%u, ", (*it).first);
    } else {
      std::cout << "( " << (*it).first << ", " << +(*it).second << "), ";
    }
  }
  printf("\n");
}

template <typename value_type>
template <int bytes>
void TinySetV_small<value_type>::print_pmas_internal(
    const extra_data &d) const {
  uint32_t pma_count = get_pma_count(d);
  if (pma_count == 1) {
    reinterpret_cast<const PMA<bytes, value_type> *>(&pmas.d[0])->print_pma(0);
  } else {
    for (uint32_t i = 0; i < get_pma_count(d); i++) {
      if constexpr (bytes < 4) {
        reinterpret_cast<const PMA<bytes, value_type> *>(&pmas.p[i])
            ->print_pma(i << (bytes * 8U));
      } else {
        reinterpret_cast<const PMA<bytes, value_type> *>(&pmas.p[i])
            ->print_pma(i);
      }
    }
  }
}

template <typename value_type>
void TinySetV_small<value_type>::print_pmas(const extra_data &d) const {
  std::cout << "B = " << pmas.get_b() << std::endl;
  switch (pmas.get_b()) {
  case 1:
    return print_pmas_internal<1>(d);
  case 2:
    return print_pmas_internal<2>(d);
  case 3:
    return print_pmas_internal<3>(d);
  case 4:
    return print_pmas_internal<4>(d);
  }
}

template <typename value_type>
uint64_t TinySetV_small<value_type>::get_n() const {
  return el_count;
}

struct SUM_KEYS_TS {
  static constexpr bool no_early_exit = true;
  uint64_t result = 0;
  inline bool update(el_t key) {
    result += key;
    return false;
  }
};

template <typename value_type>
uint64_t __attribute__((noinline))
TinySetV_small<value_type>::sum_keys(const extra_data &d) const {
  struct SUM_KEYS_TS v;
  map(v, d);
  return v.result;
}

template <typename value_type> struct SUM_VALUES {
  static constexpr bool no_early_exit = true;
  value_type result = 0;
  inline bool update([[maybe_unused]] el_t key, value_type val) {
    result += val;
    return false;
  }
};

template <typename value_type>
value_type __attribute__((noinline))
TinySetV_small<value_type>::sum_values(const extra_data &d) const {
  struct SUM_VALUES<value_type> v;
  map(v, d);
  return v.result;
}

template <typename value_type>
uint64_t TinySetV_small<value_type>::get_size(const extra_data &d) const {
  uint64_t size = 0;
  uint64_t pma_count = get_pma_count(d);
  switch (pmas.get_b()) {
  case 1: {
    if (pma_count == 1) {
      // printf("pma_size = %lu\n", pmas.d->pma.get_size());
#if NO_INLINE_TINYSET == 1
      size += pmas.d->pma1.get_size();
#else
      size += pmas.d[0].pma1.get_size() - sizeof(pmas.d[0].pma1);
#endif
    } else {
      for (uint32_t i = 0; i < pma_count; i++) {
        size += pmas.p[i].pma1.get_size();
      }
    }
    break;
  }
  case 2: {
    if (pma_count == 1) {
      // printf("pma_size = %lu\n", pmas.d->pma.get_size());
#if NO_INLINE_TINYSET == 1
      size += pmas.d->pma2.get_size();
#else
      size += pmas.d[0].pma2.get_size() - sizeof(pmas.d[0].pma2);
#endif
    } else {
      for (uint32_t i = 0; i < pma_count; i++) {
        size += pmas.p[i].pma2.get_size();
      }
    }
    break;
  }
  case 3: {
    if (pma_count == 1) {
      // printf("pma_size = %lu\n", pmas.d->pma.get_size());
#if NO_INLINE_TINYSET == 1
      size += pmas.d->pma3.get_size();
#else
      size += pmas.d[0].pma3.get_size() - sizeof(pmas.d[0].pma3);
#endif
    } else {
      for (uint32_t i = 0; i < pma_count; i++) {
        size += pmas.p[i].pma3.get_size();
      }
    }
    break;
  }
  case 4: {
    if (pma_count == 1) {
      // printf("pma_size = %lu\n", pmas.d->pma.get_size());
#if NO_INLINE_TINYSET == 1
      size += pmas.d->pma4.get_size();
#else
      size += pmas.d[0].pma4.get_size() - sizeof(pmas.d[0].pma4);
#endif
    } else {
      for (uint32_t i = 0; i < pma_count; i++) {
        size += pmas.p[i].pma4.get_size();
      }
    }
    break;
  }
  }
  return size + sizeof(TinySetV_small);
}

// returns number of buckets, max_count in any bucket, number of buckets that
// are empty
template <typename value_type>
std::tuple<uint64_t, uint64_t, uint64_t>
TinySetV_small<value_type>::statistics(const extra_data &d) const {
  uint32_t pma_count = get_pma_count(d);
  uint64_t max_count = 0;
  uint32_t empty = 0;
  switch (pmas.get_b()) {
  case 1: {
    const PMA<1, value_type> *ps = reinterpret_cast<const PMA<1, value_type> *>(
        (pma_count == 1) ? &pmas.d[0] : pmas.p);
    for (uint32_t i = 0; i < pma_count; i++) {
      max_count = std::max(max_count, ps[i].get_n());
      if (ps[i].get_n() == 0) {
        empty += 1;
      }
    }
    break;
  }
  case 2: {
    const PMA<2, value_type> *ps = reinterpret_cast<const PMA<2, value_type> *>(
        (pma_count == 1) ? &pmas.d[0] : pmas.p);
    for (uint32_t i = 0; i < pma_count; i++) {
      max_count = std::max(max_count, ps[i].get_n());
      if (ps[i].get_n() == 0) {
        empty += 1;
      }
    }
    break;
  }
  case 3: {
    const PMA<3, value_type> *ps = reinterpret_cast<const PMA<3, value_type> *>(
        (pma_count == 1) ? &pmas.d[0] : pmas.p);
    for (uint32_t i = 0; i < pma_count; i++) {
      max_count = std::max(max_count, ps[i].get_n());
      if (ps[i].get_n() == 0) {
        empty += 1;
      }
    }
    break;
  }
  case 4: {
    const PMA<4, value_type> *ps = reinterpret_cast<const PMA<4, value_type> *>(
        (pma_count == 1) ? &pmas.d[0] : pmas.p);
    for (uint32_t i = 0; i < pma_count; i++) {
      max_count = std::max(max_count, ps[i].get_n());
      if (ps[i].get_n() == 0) {
        empty += 1;
      }
    }
    break;
  }
  }
  if (empty > pma_count) {
    printf("empty = %u, pma_count = %u\n", empty, pma_count);
    print_pmas(d);
    exit(-1);
  }
  return {pma_count, max_count, empty};
}

template <typename value_type>
uint32_t TinySetV_small<value_type>::which_pma(el_t e) const {
  if constexpr (max_pma_index_size == sizeof(uint64_t)) {
    if (pmas.get_b() == sizeof(e)) {
      return 0;
    }
  }
  return static_cast<uint64_t>(e) >> (pmas.get_b() * 8);
}

template <typename value_type>
uint32_t TinySetV_small<value_type>::small_element(el_t e) const {
  if constexpr (max_pma_index_size == sizeof(uint64_t)) {
    if (pmas.get_b() == sizeof(e)) {
      return e;
    }
  }
  return e & ((1UL << (pmas.get_b() * 8)) - 1);
}

template <typename value_type>
uint64_t TinySetV_small<value_type>::pma_region_size() const {
  if constexpr (max_pma_index_size == sizeof(uint64_t)) {
    if (pmas.get_b() == sizeof(uint64_t)) {
      return 0; // this is incorrect but we can't express the number with the
                // given bits, since it is 1 bigger than the max size
    }
  }
  return 1UL << (pmas.get_b() * 8);
}

// TODO(wheatman) fails with new_b = 64
// TODO(wheatman) this can be made a lot faster since we know they are already
// in order
template <typename value_type>
template <uint32_t old_b, uint32_t new_b>
void TinySetV_small<value_type>::change_index(uint32_t old_pma_count,
                                              const extra_data &d) {
  const uint32_t new_pma_count = calc_pma_count(d.max_el, new_b);
  pma_types *new_pmas = nullptr;
  PMA<new_b, value_type> *new_pma_in_place = nullptr;
  if (new_pma_count > 1) {

    new_pmas = (pma_types *)malloc(new_pma_count * sizeof(pma_types));

    for (uint32_t i = 0; i < new_pma_count; i++) {
      new (&new_pmas[i]) PMA<new_b, value_type>();
    }
  } else {
    new_pma_in_place = new PMA<new_b, value_type>();
  }
  // TODO(wheatman) we know all of these inserts are to the end since the old
  // pma is sorted
  if (old_pma_count == 1) {
    auto it_end = reinterpret_cast<PMA<old_b, value_type> *>(&pmas.d[0])->end();
    auto it = reinterpret_cast<PMA<old_b, value_type> *>(&pmas.d[0])->begin();
    for (; it != it_end; ++it) {
      el_t val = (*it).first;
      if (new_pma_count == 1) {
        if constexpr (binary) {
          new_pma_in_place->insert(val);
        } else {
          new_pma_in_place->insert({val, (*it).second});
        }

        continue;
      }
      uint32_t pma_index = 0;
      uint32_t small_el = val;
      if constexpr (new_b < 4) {
        pma_index = val >> (new_b * 8);
        small_el &= (((1U << (new_b * 8)) - 1));
      }
      if constexpr (binary) {
        reinterpret_cast<PMA<new_b, value_type> *>(&new_pmas[pma_index])
            ->insert(small_el);
      } else {
        reinterpret_cast<PMA<new_b, value_type> *>(&new_pmas[pma_index])
            ->insert({small_el, (*it).second});
      }
    }
  } else {
    for (uint32_t i = 0; i < old_pma_count; i++) {
      auto it_end =
          reinterpret_cast<PMA<old_b, value_type> *>(&pmas.p[i])->end();
      auto it = reinterpret_cast<PMA<old_b, value_type> *>(&pmas.p[i])->begin();
      for (; it != it_end; ++it) {
        el_t val = (*it).first;
        if constexpr (old_b < 4) {
          val |= (i << (old_b * 8));
        }
        if (new_pma_count == 1) {
          if constexpr (binary) {
            new_pma_in_place->insert(val);
          } else {
            new_pma_in_place->insert({val, (*it).second});
          }
          continue;
        }
        uint32_t pma_index = 0;
        uint32_t small_el = val;
        if constexpr (new_b < 4) {
          pma_index = val >> (new_b * 8);
          small_el &= (((1U << (new_b * 8)) - 1));
        }
        if constexpr (binary) {
          reinterpret_cast<PMA<new_b, value_type> *>(&new_pmas[pma_index])
              ->insert(small_el);
        } else {
          reinterpret_cast<PMA<new_b, value_type> *>(&new_pmas[pma_index])
              ->insert({small_el, (*it).second});
        }
      }
    }
  }
  if (old_pma_count > 1) {
    for (uint32_t i = 0; i < old_pma_count; i++) {
      reinterpret_cast<PMA<old_b, value_type> *>(&pmas.p[i])
          ->~PMA<old_b, value_type>();
    }
    free(pmas.p);
  } else {
#if NO_INLINE_TINYSET == 1
    delete pmas.d;
#else
    reinterpret_cast<PMA<old_b, value_type> *>(&pmas.d[0])
        ->~PMA<old_b, value_type>();
#endif
  }
  if (new_pma_count == 1) {
    pmas.set_b(new_b);
#if NO_INLINE_TINYSET == 1
    pmas.d = (pma_types_ *)new_pma_in_place;
#else
    reinterpret_cast<PMA<new_b, value_type> *>(&pmas.d[0])
        ->shallow_copy(new_pma_in_place);
    // remove control of the elements from the temp pma
    new_pma_in_place->clean_no_free();
    delete new_pma_in_place;
#endif
  } else {
    pmas.set_b(new_b);
    pmas.p = new_pmas;
  }
  pmas.set_b(new_b);
}

template <typename value_type>
void TinySetV_small<value_type>::insert(item_type e, const extra_data &d) {
  uint32_t pma_count = get_pma_count(d);
  bool inserted = false;
  uint32_t which = which_pma(std::get<0>(e));
  std::get<0>(e) = small_element(std::get<0>(e));
  switch (pmas.get_b()) {
  case 1:
    inserted = ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma1.insert(e);
    break;
  case 2:
    inserted = ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma2.insert(e);
    break;
  case 3:
    inserted = ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma3.insert(e);
    break;
  case 4:
    inserted = ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma4.insert(e);
    break;
  }
  el_count += inserted;
  if (inserted) {
    if (el_count == d.thresh_24) {
      change_index<4, 3>(pma_count, d);
    } else if (el_count == d.thresh_16) {
      change_index<3, 2>(pma_count, d);
    } else if (el_count == d.thresh_8) {
      change_index<2, 1>(pma_count, d);
    }
  }
}

template <typename value_type>
void TinySetV_small<value_type>::insert_batch(item_type *els, uint64_t n,
                                              const extra_data &d) {
  for (uint64_t i = 0; i < n; i++) {
    insert(els[i], d);
  }
}

template <typename value_type>
void TinySetV_small<value_type>::remove(el_t e, const extra_data &d) {
  uint32_t pma_count = get_pma_count(d);
  bool removed = false;
  uint32_t which = which_pma(e);
  el_t small = small_element(e);
  switch (pmas.get_b()) {
  case 1:
    removed =
        ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma1.remove(small);

    break;
  case 2:
    removed =
        ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma2.remove(small);
    break;
  case 3:
    removed =
        ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma3.remove(small);
    break;
  case 4:
    removed =
        ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma4.remove(small);
    break;
  }
  el_count -= removed;
  if (removed) {
    if (el_count == d.thresh_24 - 1) {
      change_index<3, 4>(pma_count, d);
    } else if (el_count == d.thresh_16 - 1) {
      change_index<2, 3>(pma_count, d);
    } else if (el_count == d.thresh_8 - 1) {
      change_index<1, 2>(pma_count, d);
    }
  }
}

template <typename value_type>
bool TinySetV_small<value_type>::has(el_t e, const extra_data &d) const {
  uint32_t which = which_pma(e);
  uint32_t pma_count = get_pma_count(d);

  el_t small = small_element(e);
  switch (pmas.get_b()) {
  case 1:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma1.has(small);
  case 2:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma2.has(small);
  case 3:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma3.has(small);
  case 4:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma4.has(small);
  }
  // should never happen
  return false;
}

template <typename value_type>
value_type TinySetV_small<value_type>::value(el_t e,
                                             const extra_data &d) const {
  uint32_t which = which_pma(e);
  uint32_t pma_count = get_pma_count(d);
  el_t small = small_element(e);
  switch (pmas.get_b()) {
  case 1:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma1.value(small);
  case 2:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma2.value(small);
  case 3:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma3.value(small);
  case 4:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma4.value(small);
  }
  // should never happen
  return 0;
}

template <typename value_type> TinySetV_small<value_type>::TinySetV_small() {
#if NO_INLINE_TINYSET == 1
  pmas.d = (pma_types_ *)new PMA<4, value_type>(32);
#else
  new (&pmas.d[0]) PMA<4, value_type>();
#endif
  pmas.set_b(4);
}

template <typename value_type>
TinySetV_small<value_type>::TinySetV_small(const TinySetV_small &source,
                                           const extra_data &d) {
  el_count = source.el_count;
  uint32_t new_pma_count = source.get_pma_count(d);
  uint8_t b = source.pmas.get_b();
  pmas.set_b(b);
  if (new_pma_count > 1) {
    pmas.p = (pma_types *)malloc(new_pma_count * sizeof(pma_types));

    pmas.set_b(b);
    switch (pmas.get_b()) {
    case 1: {
      for (uint32_t i = 0; i < new_pma_count; i++) {
        new (&pmas.p[i]) PMA<1, value_type>(source.pmas.p[i].pma1);
      }
      break;
    }
    case 2: {
      for (uint32_t i = 0; i < new_pma_count; i++) {
        new (&pmas.p[i]) PMA<2, value_type>(source.pmas.p[i].pma2);
      }
      break;
    }
    case 3: {
      for (uint32_t i = 0; i < new_pma_count; i++) {
        new (&pmas.p[i]) PMA<3, value_type>(source.pmas.p[i].pma3);
      }
      break;
    }
    case 4: {
      for (uint32_t i = 0; i < new_pma_count; i++) {
        new (&pmas.p[i]) PMA<4, value_type>(source.pmas.p[i].pma4);
      }
      break;
    }
    }

  } else {
    switch (pmas.get_b()) {
    case 1:
      new (&pmas.d[0].pma1) PMA<1, value_type>(source.pmas.d[0].pma1);
      break;
    case 2:
      new (&pmas.d[0].pma2) PMA<2, value_type>(source.pmas.d[0].pma2);
      break;
    case 3:
      new (&pmas.d[0].pma3) PMA<3, value_type>(source.pmas.d[0].pma3);
      break;
    case 4:
      new (&pmas.d[0].pma4) PMA<4, value_type>(source.pmas.d[0].pma4);
      break;
    }
  }
  pmas.set_b(b);
}

template <typename value_type>
void TinySetV_small<value_type>::destroy(const extra_data &d) {
  if (get_pma_count(d) == 1) {
#if NO_INLINE_TINYSET == 1
    delete pmas.d;
#else
    switch (pmas.get_b()) {
    case 1:
      pmas.d[0].pma1.~PMA<1, value_type>();
      break;
    case 2:
      pmas.d[0].pma2.~PMA<2, value_type>();
      break;
    case 3:
      pmas.d[0].pma3.~PMA<3, value_type>();
      break;
    case 4:
      pmas.d[0].pma4.~PMA<4, value_type>();
      break;
    }

#endif
    return;
  }
  for (uint32_t i = 0; i < get_pma_count(d); i++) {
    switch (pmas.get_b()) {
    case 1:
      pmas.p[i].pma1.~PMA<1, value_type>();
      break;
    case 2:
      pmas.p[i].pma2.~PMA<2, value_type>();
      break;
    case 3:
      pmas.p[i].pma3.~PMA<3, value_type>();
      break;
    case 4:
      pmas.p[i].pma4.~PMA<4, value_type>();
      break;
    }
  }
  free(pmas.p);
}
