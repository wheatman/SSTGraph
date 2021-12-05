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

```
make OPENMP=1
```

## Compiler Support

Both clang++ and g++ have been tested and should work completly, but the clang++ was the main compiler used.