#include "SparseMatrix.hpp"
#include "cxxopts.hpp"
#include "helpers.h"

#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Sparse>

template <typename value_type>
[[nodiscard]] int matrix_values_add_remove_test_templated(uint32_t el_count,
                                                          uint32_t row_count,
                                                          bool check = false) {
  SparseMatrixV<true, value_type> mat(row_count, row_count);
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, value_type>>
      correct_matrix;

  auto rows = create_random_data(el_count, row_count - 1);
  auto cols = create_random_data(el_count, row_count - 1);
  auto values = create_random_data<value_type>(el_count, 100);

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    uint32_t r = rows[i];
    uint32_t c = cols[i];
    value_type value = values[i];

    correct_matrix[r][c] = value;

    mat.insert({r, c, value});
    if (check) {
      if (!mat.has(r, c)) {
        printf("FAILED: don't have something we inserted while inserting "
               "elements, %u, %u\n",
               r, c);
        mat.print_arrays();
        return -1;
      }
      if (std::get<0>(mat.value(r, c)) != value) {
        printf("FAILED: value doesn't match while inserting elements\n");
        std::cout << mat.value(r, c) << ", " << value << std::endl;
        mat.print_arrays();
        return -1;
      }
    }
  }
  uint64_t correct_sum = 0;
  for (auto &row : correct_matrix) {
    for (auto &pair : row.second) {
      correct_sum += pair.first;
      if (check) {
        if (!mat.has(row.first, pair.first)) {
          printf("FAILED: don't have something we inserted after inserting "
                 "elements, "
                 "row was %u, col was %u\n",
                 row.first, pair.first);
          return -1;
        }
        if (std::get<0>(mat.value(row.first, pair.first)) != pair.second) {
          printf("FAILED: value doesn't match after inserting elements\n");
          mat.print_arrays();
          return -1;
        }
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  printf("sum is %lu\n", mat.touch_all_sum());
  if (mat.touch_all_sum() != correct_sum) {
    printf("FAILED: sum didn't match after inserting elements\n");
    mat.print_arrays();
    return -1;
  }

  start = get_usecs();
  for (auto &row : correct_matrix) {
    for (auto &pair : row.second) {
      mat.remove(row.first, pair.first);
      if (check) {
        if (mat.has(row.first, pair.first)) {
          printf("have something we removed while removing elements\n");
          return -1;
        }
      }
    }
  }
  if (check && mat.M() != 0) {
    printf("still have elements when we shouldn't\n");
    return -1;
  }
  uint64_t remove_duration = get_usecs() - start;
  printf("insert duration = %lu, remove duration = %lu\n", insert_duration,
         remove_duration);
  return 0;
}

[[nodiscard]] int matrix_values_add_remove_test(
    uint32_t el_count,
    uint32_t row_count = std::numeric_limits<uint32_t>::max(),
    bool check = false) {
  int ret = 0;
  ret = matrix_values_add_remove_test_templated<uint8_t>(el_count, row_count,
                                                         check);
  if (ret) {
    return ret;
  }
  ret = matrix_values_add_remove_test_templated<uint32_t>(el_count, row_count,
                                                          check);
  if (ret) {
    return ret;
  }
  ret = matrix_values_add_remove_test_templated<float>(el_count, row_count,
                                                       check);
  if (ret) {
    return ret;
  }
  ret = matrix_values_add_remove_test_templated<double>(el_count, row_count,
                                                        check);
  return ret;
}

int main(int argc, char *argv[]) {

  cxxopts::Options options(
      "SparseMatrixTester",
      "allows testing diferent attributes of the SparseMatrix");

  options.positional_help("Help Text");
  // clang-format off
  options.add_options()
    ("r,rows", "how many rows", cxxopts::value<int>()->default_value("10000"))
    ("el_count", "how many values to insert", cxxopts::value<int>()->default_value( "100000"))
    ("v, verify", "verify the results of the test, might be much slower")
    ("matrix_values_add_remove_test","adding and removing from a matrix with values")
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  uint32_t el_count = result["el_count"].as<int>();
  bool verify = result["verify"].as<bool>();

  uint32_t rows = result["rows"].as<int>();

  if (result["matrix_values_add_remove_test"].as<bool>()) {
    return matrix_values_add_remove_test(el_count, rows, verify);
  }
}
