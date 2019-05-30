// This code is part of the project "Krill"
// Copyright (c) 2019 Hongzheng Chen
// Original work Copyright (c) 2013 Julian Shun

#ifndef GRAPH_H
#define GRAPH_H

template <class intT>
struct Vertex
{
    Vertex(intT* iN, intT* oN, intT id, intT od):
        inNeighbors(iN), outNeighbors(oN), inDegree(id), outDegree(od){}
    void setInDegree(intT _d) { inDegree = _d; }
    void setOutDegree(intT _d) { outDegree = _d; }
    intT getInDegree() { return inDegree; }
    intT getOutDegree() { return outDegree; }
    intT *getInNeighbors() { return inNeighbors; }
    intT *getOutNeighbors() { return outNeighbors; }
    intT getInNeighbor(intT j) { return inNeighbors[2 * j]; }
    intT getOutNeighbor(intT j) { return outNeighbors[2 * j]; }
    intT getInWeight(intT j) { return inNeighbors[2 * j + 1]; }
    intT getOutWeight(intT j) { return outNeighbors[2 * j + 1]; }
    void setInNeighbors(intT *_i) { inNeighbors = _i; }
    void setOutNeighbors(intT *_i) { outNeighbors = _i; }
    intT inDegree, outDegree;
    intT *inNeighbors, *outNeighbors; // the second position is used to store weights
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
    graph(Vertex<intT>* VV, intT nn, uintT mm) 
        : V(VV), n(nn), m(mm){}
};

#endif // GRAPH_H