#!/bin/bash
ROUNDS=10

function ts_run {
#taskset -c 0 ./run --static=$1 --iters=$ROUNDS --run_info=1 --bfs_src=$2
#taskset -c 0-1 ./run --static=$1 --iters=$ROUNDS --run_info=2 --bfs_src=$2
#taskset -c 0-3 ./run --static=$1 --iters=$ROUNDS --run_info=4 --bfs_src=$2
#taskset -c 0-7 ./run --static=$1 --iters=$ROUNDS --run_info=8 --bfs_src=$2
#taskset -c 0-15 ./run --static=$1 --iters=$ROUNDS --run_info=16 --bfs_src=$2
  taskset -c 0-23 ./run --static=$1 --iters=$ROUNDS --run_info=24 --bfs_src=$2
  taskset -c 0-23,48-71 ./run --static=$1 --iters=$ROUNDS --run_info=48h --bfs_src=$2
  taskset -c 0-47 ./run --static=$1 --iters=$ROUNDS --run_info=48n --bfs_src=$2
  numactl -i all ./run --static=$1 --iters=$ROUNDS --run_info=96 --bfs_src=$2
}

ts_run ../graphs/soc-LiveJournal_shuf.adj 9
ts_run ../graphs/com-orkut.ungraph.adj.shuf 28
ts_run ../graphs/rmat_ligra.adj.shuf 19372
ts_run ../graphs/soc-LiveJournal1_sym.adj 0
ts_run ../graphs/com-orkut.ungraph.adj 1000
ts_run ../graphs/rmat_ligra.adj 0
ts_run ../graphs/twitter.adj 12
ts_run ../graphs/twitter.adj.shuf 6662945
ts_run ../graphs/protein.adj 35
ts_run ../graphs/protein.adj.shuf 5651565

