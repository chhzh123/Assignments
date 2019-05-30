/*
 * Copyright (c) Hongzheng Chen 2019
 *
 * This code implements the parallel Dijkstra algorithm using OpenMP + MPI,
 * and is designed for *Parallel & Distributed Computing* @ SYSU, Spring 2019.
 *
 * This file includes the main function.
 *
 */

#include <stdio.h>
#include <cstring>
#include "parallel.h"
#include "IO.h"
using namespace std;

void dijstra(Vertex<uintT>* V, uintT n, uintT start, uintT end);

int main(int argc, char* argv[])
{
	char* iFile = argv[1];
	uintT numVert = stoi(argv[2]);
	edgeArray<uintT> E = readEdgeList<uintT>(iFile,numVert);
	graph<uintT> G = graphFromEdges(E);

	dijstra(G.V,G.n,0,numVert-1);
	return 0;
}

void dijstra(Vertex<uintT>* V, uintT n, uintT start, uintT end)
{
	// initialization
	uintT* dist = newA(uintT,n);
	intT* prev = newA(intT,n);
	bool* flag = newA(bool,n);
	parallel_for(int i = 0; i < n; ++i){
		dist[i] = INT_T_MAX;
		prev[i] = UNDEFINED;
		flag[i] = 1; // in set Q
	}
	dist[start] = 0;

	for (int t = 0; t < n-1; ++t){ // except for start
		uintT src = start;
		uintT curr_min_dist = INT_T_MAX;
		for (int i = 0; i < n; ++i)
			if (flag[i] && dist[i] < curr_min_dist){
				curr_min_dist = dist[i];
				src = i;
			}
		flag[src] = 0;
		Vertex<uintT> v_src = V[src];
		parallel_for(int os = 0; os < v_src.getOutDegree(); ++os){
			uintT dst = v_src.getOutNeighbor(os);
			uintT alt = dist[src] + v_src.getOutWeight(os);
			if (alt < dist[dst]){
				dist[dst] = alt;
				prev[dst] = src;
			}
		}
	}

	// output
	printf("Dist: %d\n", dist[end]);
	printf("%d", end);
	if (prev[end] != UNDEFINED)
		for (int i = prev[end]; prev[i] != UNDEFINED; i = prev[i]){
			printf("<-%d", i);
		}
	printf("<-%d\n",start);
}