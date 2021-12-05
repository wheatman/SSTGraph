#pragma once
#include "../SparseMatrix.hpp"
#include "../helpers.h"
#include <vector>

// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

using fType = double;
using uintE = uint32_t;

struct BC_F {
  fType *NumPaths;
  bool *Visited;
  BC_F(fType *_NumPaths, bool *_Visited)
      : NumPaths(_NumPaths), Visited(_Visited) {}
  inline bool update(uintE s, uintE d) { // Update function for forward phase
    fType oldV = NumPaths[d];
    NumPaths[d] += NumPaths[s];
    return oldV == 0.0;
  }
  inline bool updateAtomic(uintE s, uintE d) { // atomic Update, basically an
    // add
    volatile fType oldV, newV;
    do {
      oldV = NumPaths[d];
      newV = oldV + NumPaths[s];
    } while (!CAS(&NumPaths[d], oldV, newV));
    return oldV == 0.0;
  }
  inline bool cond(uintE d) { return Visited[d] == 0; } // check if visited
};

struct BC_Back_F {
  fType *Dependencies;
  bool *Visited;
  BC_Back_F(fType *_Dependencies, bool *_Visited)
      : Dependencies(_Dependencies), Visited(_Visited) {}
  inline bool update(uintE s, uintE d) { // Update function for backwards phase
    fType oldV = Dependencies[d];
    Dependencies[d] += Dependencies[s];
    return oldV == 0.0;
  }
  inline bool updateAtomic(uintE s, uintE d) { // atomic Update
    volatile fType oldV, newV;
    do {
      oldV = Dependencies[d];
      newV = oldV + Dependencies[s];
    } while (!CAS(&Dependencies[d], oldV, newV));
    return oldV == 0.0;
  }
  inline bool cond(uintE d) { return Visited[d] == 0; } // check if visited
};

// vertex map function to mark visited vertexSubset
struct BC_Vertex_F {
  bool *Visited;
  explicit BC_Vertex_F(bool *_Visited) : Visited(_Visited) {}
  inline bool operator()(uintE i) {
    Visited[i] = true;
    return true;
  }
};

// vertex map function (used on backwards phase) to mark visited vertexSubset
// and add to Dependencies score
struct BC_Back_Vertex_F {
  bool *Visited;
  fType *Dependencies, *inverseNumPaths;
  BC_Back_Vertex_F(bool *_Visited, fType *_Dependencies,
                   fType *_inverseNumPaths)
      : Visited(_Visited), Dependencies(_Dependencies),
        inverseNumPaths(_inverseNumPaths) {}
  inline bool operator()(uintE i) {
    Visited[i] = true;
    Dependencies[i] += inverseNumPaths[i];
    return true;
  }
};

template <typename SM>
fType *BC(const SM &G, const uintE &start,
          [[maybe_unused]] bool use_dense_forward = false) {
  const size_t n = G.get_rows();
  if (n == 0) {
    return nullptr;
  }
  fType *NumPaths = newA<fType>(n);
  {
    parallel_for(uint64_t i = 0; i < n; i++) { NumPaths[i] = 0.0; }
  }
  bool *Visited = newA<bool>(n);
  {
    parallel_for(uint64_t i = 0; i < n; i++) { Visited[i] = false; }
  }
  Visited[start] = true;
  NumPaths[start] = 1.0;
  VertexSubset Frontier = VertexSubset(start, n); // creates initial frontier

  std::vector<VertexSubset> Levels;
  Levels.push_back(Frontier);
  int64_t round = 0;
  while (Frontier.non_empty()) {
    round++;
    VertexSubset output =
        G.edgeMap(Frontier, BC_F(NumPaths, Visited), true, 20);
    Levels.push_back(output);
    Frontier = output;
    G.vertexMap(Frontier, BC_Vertex_F(Visited), false); // mark visited
  }

  fType *Dependencies = newA<fType>(n);
  parallel_for(uint64_t i = 0; i < n; i++) { Dependencies[i] = 0.0; }

  parallel_for(uint64_t i = 0; i < n; i++) { NumPaths[i] = 1 / NumPaths[i]; }
  Levels[round].del();

  parallel_for(uint64_t i = 0; i < n; i++) { Visited[i] = false; }

  G.vertexMap(Levels[round - 1],
              BC_Back_Vertex_F(Visited, Dependencies, NumPaths), false);
  for (int64_t r = round - 2; r >= 0; r--) {
    G.edgeMap(Levels[r + 1], BC_Back_F(Dependencies, Visited), false, 20);
    Levels[r + 1].del();
    G.vertexMap(Levels[r], BC_Back_Vertex_F(Visited, Dependencies, NumPaths),
                false);
  }
  parallel_for(uint32_t i = 0; i < n; i++) {
    Dependencies[i] = (Dependencies[i] - NumPaths[i]) / NumPaths[i];
  }
  Levels[0].del();
  free(NumPaths);
  free(Visited);

  return Dependencies;
}
