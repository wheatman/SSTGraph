OPT?=3
VALGRIND?=0
SANITIZE?=0

OPENMP?=0
TINYSET_32?=0
NO_INLINE_TINYSET?=0
PARALLEL=0

CFLAGS := -Wall -Wno-address-of-packed-member -Wextra -O$(OPT) -g  -std=c++17


ifeq ($(OPENMP),1)
PARALLEL=1
CILK=0
CFLAGS += -fopenmp
ONE_WORKER = OMP_NUM_THREADS=1
else
CILK?=0
endif

ifeq ($(CILK),1)
CFLAGS += -fopencilk
LDFLAGS += -Lx86_64/ -lopencilk
ONE_WORKER = CILK_NWORKERS=1
PARALLEL=1
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

LDFLAGS := -lrt -lm -lm -ldl 




DEFINES := -DOPENMP=$(OPENMP) -DCILK=$(CILK) -DTINYSET_32=$(TINYSET_32) -DNO_INLINE_TINYSET=$(NO_INLINE_TINYSET)

SRC := run.cpp

INCLUDES := TinySet_small.h  helpers.h  parallel.h  rmat_util.h  BitArray.hpp  PMA.hpp  SparseMatrix.hpp  TinySet.hpp  TinySet_small.hpp  VertexSubset.hpp  cxxopts.hpp  packedarray.hpp  test.hpp algorithms/BC.h  algorithms/BFS.h  algorithms/BellmanFord.h  algorithms/Components.h  algorithms/PageRank.h  algorithms/TC.h  algorithms/Touchall.h integerSort/blockRadixSort.h  integerSort/sequence.h  integerSort/transpose.h  integerSort/utils.h gap/sliding_queue.h

.PHONY: all clean tidy

all:  basic 
#build_profile profile opt


basic: $(SRC) $(INCLUDES)
	$(CXX) $(CFLAGS) $(DEFINES) -DNDEBUG $(SRC) $(LDFLAGS) -o basic
	cp basic run
	objdump -S --disassemble basic > run_basic.dump &

test: $(SRC) $(INCLUDES)
	$(CXX) $(CFLAGS) -fprofile-instr-generate=test_out/cov.%p.profraw -fcoverage-mapping $(DEFINES) $(SRC) $(LDFLAGS) -o test
	@mkdir -p test_out
	./test --sizes
	@./test --pma_add_test --el_count 1000000 --verify  >test_out/pma_add_test || (echo "./test --pma_add_test --el_count 1000000 --verify verification failed $$?"; exit 1)&
	@./test --pma_map_add_test --el_count 100000 --verify  >test_out/pma_map_add_test || (echo "./test --pma_map_add_test --el_count 100000 --verify verification failed $$?"; exit 1)&
	@./test --pma_remove_test --el_count 10000 --verify >test_out/pma_remove_test|| (echo "./test --pma_remove_test --el_count 10000 --verify verification failed $$?"; exit 1)&
	@./test --pma_map_remove_test --el_count 100000 --verify  >test_out/pma_map_remove_test || (echo "./test --pma_map_remove_test --el_count 100000 --verify  verification failed $$?"; exit 1)&
	@./test --tinyset_add_test --el_count 1000000 --verify >test_out/tinyset_add_test || (echo "./test --tinyset_add_test --el_count 1000000 --verify verification failed $$?"; exit 1)&
	@./test --tinyset_remove_test --el_count 10000 --verify >test_out/tinyset_remove_test|| (echo "./test --tinyset_remove_test --el_count 10000 --verify verification failed $$?"; exit 1) &
	@./test --tinyset_map_add_test --el_count 100000 --verify  >test_out/tinyset_map_add_test || (echo "./test --tinyset_map_add_test --el_count 100000 --verify verification failed $$?"; exit 1)&
	@./test --tinyset_map_remove_test --el_count 100000 --verify  >test_out/tinyset_map_remove_test || (echo "./test --tinyset_map_remove_test --el_count 100000 --verify verification failed $$?"; exit 1)&
	@./test --matrix_values_add_remove_test --el_count 100000 --rows 10000000 --verify  >test_out/matrix_values_add_remove_test || (echo "./test --matrix_values_add_remove_test --el_count 100000 --rows 10000000 --verify verification failed $$?"; exit 1)&

	@./test --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 >test_out/real_graph_out 2>&1 || (echo "./test --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 verification failed $$?"; exit 1)
	@diff bfs.out correct_output_files/bfs.out >test_out/bfs_diff|| (echo "bfs verification failed: ./run --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 $$?";)
	@diff bc.out correct_output_files/bc.out >test_out/bc_diff|| (echo "bc verification failed: ./run --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 $$?")
	@diff pr.out correct_output_files/pr.out >test_out/pr.diff|| (echo "pr verification failed: ./run --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 $$?")
	@diff cc.out correct_output_files/cc.out >test_out/cc_diff|| (echo "cc verification failed: ./run --real ~/graphs/soc-LiveJournal1_sym.adj --iters=1 --bfs_src=0 $$?")
	@rm -f bfs.out bc.out pr.out cc.out bf.out
	@wait
	llvm-profdata-12 merge -output=test_out/cov.profdata test_out/cov.*.profraw
	llvm-cov-12 show -Xdemangler llvm-cxxfilt-10 -Xdemangler -n ./test -instr-profile=test_out/cov.profdata > coverage_report
	llvm-cov-12 show -format=html -Xdemangler llvm-cxxfilt-10 -Xdemangler -n ./test -instr-profile=test_out/cov.profdata > coverage_report.html
	llvm-cov-12 report ./test -instr-profile=test_out/cov.profdata > coverage_summary
	@echo "Tests finished"





build_profile: $(SRC)  $(INCLUDES)
ifeq ($(CXX),g++)
	 $(CXX) -fprofile-generate $(CFLAGS) $(DEFINES) -DNDEBUG $(SRC) $(LDFLAGS) -o build_profile
else
	 $(CXX) -fprofile-arcs -fprofile-instr-generate=code-%p.profraw $(CFLAGS) $(DEFINES)  $(SRC) $(LDFLAGS) -o build_profile
endif

profile: build_profile 
	rm -f *.profdata *.profraw
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
	rm -f run run_profile run.dump run_basic run.gcda run_basic.dump *.profdata *.profraw test_out/* test basic opt
	rm -f bfs.out bc.out pr.out cc.out bf.out

