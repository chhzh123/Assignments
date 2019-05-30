/*
 * Copyright (c) Hongzheng Chen 2019
 *
 * This code implements the parallel Dijkstra algorithm using OpenMP + MPI,
 * and is designed for *Parallel & Distributed Computing* @ SYSU, Spring 2019.
 *
 * This file includes the input logic.
 *
 */

#ifndef IO_H
#define IO_H

#include <fstream>
#include "utils.h"
#include "blockRadixSort.h"
#include "graph.h"

// Modified from Ligra's undelying infrastructure

_seq<char> readStringFromFile(char *fileName) {
    ifstream file (fileName, ios::in | ios::binary | ios::ate);
    if (!file.is_open()) {
        std::cout << "Unable to open file: " << fileName << std::endl;
        abort();
    }
    long end = file.tellg();
    file.seekg (0, ios::beg);
    long n = end - file.tellg();
    char* bytes = newA(char,n+1);
    file.read (bytes,n);
    file.close();
    return _seq<char>(bytes,n);
}

  // A structure that keeps a sequence of strings all allocated from
  // the same block of memory
  struct words {
    long n; // total number of characters
    char* Chars;  // array storing all strings
    long m; // number of substrings
    char** Strings; // pointers to strings (all should be null terminated)
    words() {}
    words(char* C, long nn, char** S, long mm)
      : Chars(C), n(nn), Strings(S), m(mm) {}
    void del() {free(Chars); free(Strings);}
  };
 
  inline bool isSpace(char c) {
    switch (c)  {
    case '\r': 
    case '\t': 
    case '\n': 
    case 0:
    case ' ' : return true;
    default : return false;
    }
  }

  // parallel code for converting a string to words
  words stringToWords(char *Str, long n) {
    parallel_for (long i=0; i < n; i++) 
      if (isSpace(Str[i])) Str[i] = 0; 

    // mark start of words
    bool *FL = newA(bool,n);
    if (FL == NULL){
      cerr << "Error: NULL Pointer!" << endl;
      abort();
    }
    FL[0] = Str[0];
    parallel_for (long i=1; i < n; i++) FL[i] = Str[i] && !Str[i-1];
    
    // offset for each start of word
    _seq<long> Off = sequence::packIndex<long>(FL, n);
    long m = Off.n;
    long *offsets = Off.A;

    // pointer to each start of word
    char **SA = newA(char*, m);
    parallel_for (long j=0; j < m; j++) SA[j] = Str+offsets[j];

    free(offsets); free(FL);
    return words(Str,n,SA,m);
  }


template <class intT>
edgeArray<intT> readEdgeList(char* fname,int numVert) {
    _seq<char> S = readStringFromFile(fname);
    words W = stringToWords(S.A,S.n);
    long n = W.m / 3; // # of edges
    edge<intT> *E = newA(edge<intT>,n);
    parallel_for(long i = 0; i < n; i++)
        E[i] = edge<intT>(atol(W.Strings[3*i]),
                          atol(W.Strings[3*i + 1]),
                          atol(W.Strings[3*i + 2]));
    W.del();
    return edgeArray<intT>(E, numVert, numVert, n);
}

template <class intT>
struct getuF {intT operator() (edge<intT> e) {return e.u;} };

template <class intT>
graph<intT> graphFromEdges(edgeArray<intT> A) {
    intT m = A.nonZeros; // # of edges
    intT n = max<intT>(A.numCols,A.numRows); // # of vert
    intT* offsets = newA(intT,n);
    // sort by the src vertex
    // array A, length m, int range [0,n), offset length=n
    intSort::iSort(A.E,offsets,m,n,getuF<intT>());
#ifdef DEBUG
    for (int i = 0; i < m; ++i)
        printf("%d %d %d\n",A.E[i].u,A.E[i].v,A.E[i].w);
#endif
    intT *X = newA(intT,m*2); // edge array in CSR, with weights
    Vertex<intT> *v = newA(Vertex<intT>,n); // offset array in CSR
    parallel_for (intT i = 0; i < n; i++) {
        intT o = offsets[i];
        intT l = ((i == n-1) ? m : offsets[i+1]) - offsets[i];
        v[i].setOutDegree(l);
        v[i].setOutNeighbors(X+o*2);
        for (intT j = 0; j < l; j++) {
            v[i].outNeighbors[2*j] = A.E[o+j].v;
            v[i].outNeighbors[2*j+1] = A.E[o+j].w;
        }
    }
    free(offsets);
    return graph<intT>(v,n,m);
}

template <class intT>
void printWghGraph(graph<intT> G) {
    long m = G.m;
    long n = G.n;
    printf("%ld %ld\n", n, m);
    for (int i = 0; i < n; ++i){
        printf("Src: %d Degree: %d\n", i, G.V[i].getOutDegree());
        for (int j = 0; j < G.V[i].getOutDegree(); ++j)
            printf("%d w:%d ",G.V[i].getOutNeighbor(j),G.V[i].getOutWeight(j));
        printf("\n");
    }
}

#endif // IO_H