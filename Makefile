OPT?=3
VALGRIND?=0
SANITIZE?=0

OPENMP?=0
TINYSET_32?=0
NO_INLINE_TINYSET?=0
PARLAY?=0

CFLAGS := -Wall -Wno-address-of-packed-member -Wextra -O$(OPT) -g -gdwarf-4 -std=c++20 -IParallelTools/ -IStructOfArrays/include/ -Iparlaylib/include/ -Icxxopts/include/ -IEdgeMapVertexMap/include/ -Iinclude/
#-ferror-limit=1 -ftemplate-backtrace-limit=0

LDFLAGS := -lrt -lm -lm -ldl -lpthread


ifeq ($(PARLAY),1)
ONE_WORKER = PARLAY_NUM_THREADS=1
CILK=0
else
CILK?=0
endif

ifeq ($(CILK),1)
CFLAGS += -fopencilk -DPARLAY_CILK
LDFLAGS += -fopencilk
ONE_WORKER = CILK_NWORKERS=1
else
ifeq ($(PARLAY),0)
CFLAGS += -DPARLAY_SEQUENTIAL
endif
endif


ifeq ($(SANITIZE),1)
ifeq ($(OPENMP),1)
CFLAGS += -fsanitize=undefined,thread -fno-omit-frame-pointer
else
ifeq ($(CILK),1)
CFLAGS += -fsanitize=cilk,undefined -fno-omit-frame-pointer
else
CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
endif
endif
endif

ifeq ($(OPT),3)
CFLAGS += -fno-signed-zeros  -freciprocal-math -ffp-contract=fast -fno-trapping-math  -ffinite-math-only
ifeq ($(VALGRIND),0)
CFLAGS += -march=native #-static
endif
endif


VERIFY_COUNT ?= 10000
# -pg  bdver1 sandybridge haswell



DEFINES := -DOPENMP=$(OPENMP) -DCILK=$(CILK) -DTINYSET_32=$(TINYSET_32) -DNO_INLINE_TINYSET=$(NO_INLINE_TINYSET) -DPARLAY=$(PARLAY)

SRC := run.cpp

INCLUDES := include/SSTGraph/internal/helpers.hpp include/SSTGraph/internal/rmat_util.h include/SSTGraph/internal/BitArray.hpp  include/SSTGraph/PMA.hpp include/SSTGraph/SparseMatrix.hpp include/SSTGraph/TinySet.hpp include/SSTGraph/TinySet_small.hpp include/SSTGraph/internal/test.hpp  

.PHONY: all clean tidy

all:  basic 
#build_profile profile opt


basic: $(SRC) $(INCLUDES)
	$(CXX) $(CFLAGS) $(DEFINES) -DNDEBUG $(SRC) $(LDFLAGS) -o basic
	cp basic run
	objdump -S --disassemble basic > run_basic.dump &

pma_test: PMA_test.cpp include/SSTGraph/PMA.hpp include/SSTGraph/internal/helpers.hpp include/SSTGraph/internal/SizedInt.hpp
	$(CXX) $(CFLAGS) $(DEFINES) PMA_test.cpp $(LDFLAGS) -o pma_test
	@mkdir -p test_out
	@./pma_test --sizes > test_out/pma_sizes
	@./pma_test --pma_update_test --el_count $(VERIFY_COUNT) --verify  >test_out/pma_update_test || (echo "./pma_test --pma_update_test --el_count $(VERIFY_COUNT) --verify verification failed $$?"; exit 1)&
	@./pma_test --pma_map_test --el_count $(VERIFY_COUNT) --verify  >test_out/pma_map_test || (echo "./pma_test --pma_map_test --el_count $(VERIFY_COUNT) --verify verification failed $$?"; exit 1)&
	@./pma_test --pma_map_soa_test --el_count $(VERIFY_COUNT) --verify  >test_out/pma_map_soa_test || (echo "./pma_test --pma_map_soa_test --el_count $(VERIFY_COUNT) --verify verification failed $$?"; exit 1)&
	@./pma_test --pma_map_tuple_test --el_count $(VERIFY_COUNT) --verify  >test_out/pma_map_tuple_test || (echo "./pma_map_soa_test --pma_map_tuple_test --el_count $(VERIFY_COUNT) --verify verification failed $$?"; exit 1)&
	@wait
	@sleep 1
	@echo "PMA Tests Finished"

tinyset_test: Tinyset_test.cpp include/SSTGraph/TinySet.hpp include/SSTGraph/TinySet_small.hpp include/SSTGraph/internal/helpers.hpp include/SSTGraph/PMA.hpp
	$(CXX) $(CFLAGS) $(DEFINES) Tinyset_test.cpp $(LDFLAGS) -o tinyset_test
	@mkdir -p test_out
	@./tinyset_test --sizes > test_out/tinyset_sizes
	@./tinyset_test --sorting --max_val 1000000 --verify >test_out/sorting || (echo "./tinyset_test --sorting --max_val 1000000 --verify  verification failed $$?"; exit 1) &
	@./tinyset_test --tinyset_add_test --el_count 1000000 --verify >test_out/tinyset_add_test || (echo "./tinyset_test --tinyset_add_test --el_count 1000000 --verify verification failed $$?"; exit 1)&
	@./tinyset_test --tinyset_remove_test --el_count 10000 --verify >test_out/tinyset_remove_test|| (echo "./tinyset_test --tinyset_remove_test --el_count 10000 --verify verification failed $$?"; exit 1) &
	@./tinyset_test --tinyset_map_add_test --el_count 100000 --verify  >test_out/tinyset_map_add_test || (echo "./tinyset_test --tinyset_map_add_test --el_count 100000 --verify verification failed $$?"; exit 1)&
	@./tinyset_test --tinyset_map_remove_test --el_count 100000 --verify  >test_out/tinyset_map_remove_test || (echo "./tinyset_test --tinyset_map_remove_test --el_count 100000 --verify verification failed $$?"; exit 1)&
	@./tinyset_test --edge_case --max_val 1000000 >test_out/edge_case1|| (echo "./tinyset_test --edge_case --max_val 1000000 verification failed $$?"; exit 1)&
	@wait
	@sleep 1
	@echo "TinySet Tests Finished"

sparsematrix_test: SparseMatrix_test.cpp include/SSTGraph/PMA.hpp include/SSTGraph/internal/helpers.hpp include/SSTGraph/TinySet.hpp include/SSTGraph/TinySet_small.hpp include/SSTGraph/SparseMatrix.hpp
	$(CXX) $(CFLAGS) $(DEFINES) SparseMatrix_test.cpp $(LDFLAGS) -o sparsematrix_test
	@./sparsematrix_test --matrix_values_add_remove_test --el_count 100000 --rows 10000000 --verify  >test_out/matrix_values_add_remove_test || (echo "./sparsematrix_test --matrix_values_add_remove_test --el_count 100000 --rows 10000000 --verify verification failed $$?"; exit 1)&
	@wait
	@sleep 1
	@echo "SparseMatrix Tests Finished"

graph_utils: graph_utilities.cpp include/SSTGraph/PMA.hpp include/SSTGraph/internal/helpers.hpp include/SSTGraph/TinySet.hpp include/SSTGraph/TinySet_small.hpp include/SSTGraph/SparseMatrix.hpp
	$(CXX) $(CFLAGS) $(DEFINES) graph_utilities.cpp $(LDFLAGS) -o graph_utils

test_graph: $(SRC) $(INCLUDES)
	$(CXX) $(CFLAGS) $(DEFINES) $(SRC) $(LDFLAGS) -o test_graph
	@mkdir -p test_out
	@./test_graph --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 >test_out/real_graph_out 2>&1 || (echo "./test_graph --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 verification failed $$?"; exit 1)
	@diff bfs.out correct_output_files/bfs.out >test_out/bfs_diff|| (echo "bfs verification failed: ./test_graph --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 $$?";)
	@diff bc.out correct_output_files/bc.out >test_out/bc_diff|| (echo "bc verification failed: ./test_graph --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 $$?")
	@diff pr.out correct_output_files/pr.out >test_out/pr.diff|| (echo "pr verification failed: ./test_graph --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 $$?")
	@diff cc.out correct_output_files/cc.out >test_out/cc_diff|| (echo "cc verification failed: ./test_graph --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 $$?")
	@rm -f bfs.out bc.out pr.out cc.out bf.out
	@wait
	@sleep 1
	@echo "Graph Tests finished"

test: test_graph pma_test tinyset_test sparsematrix_test


build_profile: $(SRC)  $(INCLUDES)
ifeq ($(CXX),g++)
	 $(CXX) -fprofile-generate $(CFLAGS) $(DEFINES)  $(SRC) $(LDFLAGS) -o build_profile
else
	 $(CXX) -fprofile-arcs -fprofile-instr-generate=code-%p.profraw $(CFLAGS) $(DEFINES)  $(SRC) $(LDFLAGS) -o build_profile
endif

profile: build_profile 
	rm -f *.profdata *.profraw
	@$(ONE_WORKER) ./build_profile --sorting --max_val 1000000  >/dev/null  &
	@$(ONE_WORKER) ./build_profile --pma_add_test --el_count 1000000    >/dev/null &
	@$(ONE_WORKER) ./build_profile --pma_map_add_test --el_count 100000    >/dev/null &
	@$(ONE_WORKER) ./build_profile --pma_remove_test --el_count 10000   >/dev/null &
	@$(ONE_WORKER) ./build_profile --pma_map_remove_test --el_count 100000    >/dev/null  &
	@$(ONE_WORKER) ./build_profile --tinyset_add_test --el_count 1000000   >/dev/null  &
	@$(ONE_WORKER) ./build_profile --tinyset_remove_test --el_count 10000   >/dev/null  &
	@$(ONE_WORKER) ./build_profile --tinyset_map_add_test --el_count 100000    >/dev/null  &
	@$(ONE_WORKER) ./build_profile --tinyset_map_remove_test --el_count 100000    >/dev/null  &
	@$(ONE_WORKER) ./build_profile --matrix_values_add_remove_test --el_count 100000 --rows 10000000    >/dev/null  &
	@$(ONE_WORKER) ./build_profile --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0  >/dev/null 2>&1
	llvm-profdata-10 merge -output=default.profdata code-*.profraw
	rm -f bfs.out bc.out pr.out cc.out bf.out

opt: profile
ifeq ($(CXX),g++)
	$(CXX) -fprofile-use -fprofile-correction $(CFLAGS) $(DEFINES) $(SRC) $(LDFLAGS) -o opt
else	
	$(CXX) -fprofile-instr-use $(CFLAGS) $(DEFINES) $(SRC) $(LDFLAGS) -o opt
endif
	rm -f *.profdata *.profraw *.gcda
	cp opt run
	objdump -S --disassemble opt > opt.dump & 

tidy: $(SRC) $(INCLUDES)
	clang-tidy -header-filter=.*  --checks='clang-diagnostic-*,clang-analyzer-*,*,-hicpp-vararg,-cppcoreguidelines-pro-type-vararg,-fuchsia-default-arguments,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-fuchsia-overloaded-operator,-llvm-header-guard,-cppcoreguidelines-owning-memory,-readability-implicit-bool-conversion,-cppcoreguidelines-pro-type-cstyle-cast,-google-readability-casting,-misc-definitions-in-headers,-hicpp-no-malloc,-cppcoreguidelines-no-malloc,-*-use-auto,-readability-else-after-return,-cppcoreguidelines-pro-bounds-constant-array-index,-cert-err58-cpp,-cppcoreguidelines-pro-type-reinterpret-cast'   run.cpp > tidy

clean:
	rm -f run run_profile run.dump run_basic run.gcda run_basic.dump *.profdata *.profraw test_out/* test basic opt pma_test tinyset_test sparsematrix_test test_graph
	rm -f bfs.out bc.out pr.out cc.out bf.out
	rm -f coverage_* perf.data perf.data.old
