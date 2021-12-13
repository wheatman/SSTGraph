# SSTGraph

## Quick Start

compile with `make`

run on a graph with the command 
```
./run --real <file path to graph> --iters=<iteration per algorithm> --bfs_src=<source for algorithms which need one>
```

## Build with Parallelizm 

The recomended way to enable parallelizm to is build with [cilk](https://cilk.mit.edu/) 

To build with cilk compile with
```
make CILK=1
```

OpenMP is also minimially supported for testing purposes, but suffers from worse performance and less parallelism.

Any performance evaluation down against this code should use the cilk version and the clang compiler.

```
make OPENMP=1
```

## Compiler Support

Both clang++ and g++ have been tested and should work completly, but the clang++ was the main compiler used.

## Running

To run a performance evaluation of SSTGraph on a graph use 

```
./run --real <path to graph> --iters <iterations> --bfs_src <source node>
```

This will print out some statistics about the graph and then run a few different algorithms the specified number of iterations.  Lastly it will add batches of edges from an rmat distribution and measure the insertion throughput. The maximum batch size can be controlled with `--max_batch`

The algorithms are breadth first search, PageRank, Betweenness Centrality and Connected Components.


To add new algorithms define them in the algorithm's folder, then include them in test.cpp and call them in `real_graph`.

## Graph Format
The graphs can either be in the Adjencency graph format as described [here](http://www.cs.cmu.edu/~pbbs/benchmarks/graphIO.html) or the Matrix Market forat as described [here](https://networkrepository.com/mtx-matrix-market-format.html)
