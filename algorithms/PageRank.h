#pragma once
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
#pragma once
#include "../SparseMatrix.hpp"

// template <class vertex>
template <typename T, typename SM> struct PR_F {
  T *p_curr, *p_next;
  // vertex* V;
  // PR_F(double* _p_curr, double* _p_next, vertex* _V) :
  const SM &G;
  PR_F(T *_p_curr, T *_p_next, const SM &_G)
      : p_curr(_p_curr), p_next(_p_next), G(_G) {}
  inline bool update(el_t s, el_t d) {
    p_next[d] += p_curr[s];

    return true;
  }
  inline bool updateAtomic([[maybe_unused]] el_t s,
                           [[maybe_unused]] el_t d) { // atomic Update
    printf("should never be called for now since its always dense\n");

    return true;
  }
  inline bool cond([[maybe_unused]] el_t d) { return true; }
}; // from ligra readme: for cond which always ret true, ret cond_true// return
   // cond_true(d); }};

template <typename T, typename SM> struct PR_Vertex {
  T *p_curr;
  const SM &G;
  PR_Vertex(T *_p_curr, const SM &_G) : p_curr(_p_curr), G(_G) {}
  inline bool operator()(uint32_t i) {
    p_curr[i] =
        p_curr[i] / G.getDegree(i); // damping*p_next[i] + addedConstant;
    return true;
  }
};

// resets p
template <typename T> struct PR_Vertex_Reset {
  T *p;
  explicit PR_Vertex_Reset(T *_p) : p(_p) {}
  inline bool operator()(el_t i) {
    p[i] = 0.0;
    return true;
  }
};

template <typename T, typename SM> T *PR_S(const SM &G, int64_t maxIters) {
  size_t n = G.get_rows();

  T one_over_n = 1 / (double)n;
  size_t size = n + (4 - (n % 4));
  T *p_curr = (T *)memalign(32, size * sizeof(T));
  T *p_next = (T *)memalign(32, size * sizeof(T));

  parallel_for(size_t i = 0; i < n; i++) { p_curr[i] = one_over_n; }
  VertexSubset Frontier = VertexSubset(0, n, true);

  int64_t iter = 0;
  // printf("max iters %lu\n", maxIters);
  while (iter++ < maxIters) {
    // using flat snapshot
    G.vertexMap(Frontier, PR_Vertex(p_curr, G), false);
    G.vertexMap(Frontier, PR_Vertex_Reset(p_next), false);
    G.edgeMap(Frontier, PR_F(p_curr, p_next, G), false, 20);

    swap(p_curr, p_next);
  }

  free(p_next);

  return p_curr;
}
