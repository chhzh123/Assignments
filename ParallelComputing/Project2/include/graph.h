// This code is part of the project "Krill"
// Copyright (c) 2019 Hongzheng Chen
// Original work Copyright (c) 2013 Julian Shun

#ifndef GRAPH_H
#define GRAPH_H

template <class intT>
struct Vertex
{
    Vertex(intT* oN, intT od):
        outNeighbors(oN), outDegree(od){}
    void setOutDegree(intT _d) { outDegree = _d; }
    intT getOutDegree() { return outDegree; }
    intT *getOutNeighbors() { return outNeighbors; }
    intT getOutNeighbor(intT j) { return outNeighbors[2 * j]; }
    intT getOutWeight(intT j) { return outNeighbors[2 * j + 1]; }
    void setOutNeighbors(intT *_i) { outNeighbors = _i; }
    intT outDegree;
    intT *outNeighbors; // the second position is used to store weights
};

template <class intT>
struct edge {
    intT u;
    intT v;
    intT w;
    edge(intT _u, intT _v, intT _w) : u(_u), v(_v), w(_w) {}
    void print(){printf("%d %d %d\n", u, v, w);}
};

template <class intT>
struct edgeArray {
    edge<intT>* E;
    intT numRows;
    intT numCols;
    intT nonZeros;
    edgeArray(edge<intT> *EE, intT r, intT c, intT nz) :
        E(EE), numRows(r), numCols(c), nonZeros(nz) {}
    edgeArray() {}
    void del() {free(E);}
};

template <class intT>
struct graph {
    Vertex<intT> *V;
    intT n;
    intT m;
    graph():V(NULL),n(0),m(0){}
    graph(Vertex<intT>* VV, intT nn, uintT mm) 
        : V(VV), n(nn), m(mm){}
};

#endif // GRAPH_H