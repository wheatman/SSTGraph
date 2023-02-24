# SSTGraph

SSTGraph is a shared-memory parallel framework for the storage and analysis of dynamic graphs originally described in [Wheatman and Burns., 2021](https://ieeexplore.ieee.org/abstract/document/9671836). SSTGraph builds on top of the tinyset parallel, dynamic set data structure. Tinyset implements set membership in a shallow hierarchy of sorted packed memory
arrays to achieve logarithmic time access and updates, and it scans in optimal linear time. Tinyset uses space comparable to that of systems that use data compression while avoiding compression’s computation and serialization overhead.

SSTGraph supports graphs in either th [Adjancency graph format](http://www.cs.cmu.edu/~pbbs/benchmarks/graphIO.html) or the [Matrix Market format](https://networkrepository.com/mtx-matrix-market-format.html). Graph algorithms are written in the EdgeMap/VertexMap programming interface first described in [Ligra](link). This repository includes implementations of standard algorithms, such as PageRank, connected components, breadth first search, betweenness centrality, and triangle counting.


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

[Parlaylib](https://cmuparlay.github.io/parlaylib/) is also supported with
```
make PARLAY=1
```


Any performance evaluation down against this code should use the cilk version and the clang compiler.


## Compiler Support

Both clang++ and g++ have been tested and should work completely, but the clang++ was the main compiler used.

## Running

To run a performance evaluation of SSTGraph on a graph use 

```
./run --real <path to graph> --iters <iterations> --bfs_src <source node>
```

This will print out some statistics about the graph and then run a few different algorithms the specified number of iterations.  Lastly it will add batches of edges from an rmat distribution and measure the insertion throughput. The maximum batch size can be controlled with `--max_batch`

The algorithms are breadth first search, PageRank, Betweenness Centrality and Connected Components.


## Graph Format
The graphs can either be in the Adjancency graph format as described [here](http://www.cs.cmu.edu/~pbbs/benchmarks/graphIO.html) or the Matrix Market format as described [here](https://networkrepository.com/mtx-matrix-market-format.html)
