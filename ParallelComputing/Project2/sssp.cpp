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
#include <assert.h>
#include "parallel.h"
#include "IO.h"
using namespace std;

void dijstra(Vertex<uintT>* V, uintT n, uintT start, uintT end);
void dijstra_mpi(Vertex<uintT>* V, uintT n, uintT start, uintT end);

int main(int argc, char* argv[])
{
// Read graph
#ifdef TIME
	clock_t start_time = clock();
#endif

#ifndef MPIFLAG
	char* iFile = argv[1];
	uintT numVert = stoi(argv[2]);
	edgeArray<uintT> E = readEdgeList<uintT>(iFile,numVert);
	graph<uintT> G = graphFromEdges(E);
#else
	MPI_Init(NULL,NULL);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	uintT numVert = stoi(argv[2]);
	graph<uintT> G;
	if (rank == 0){
		char* iFile = argv[1];
		edgeArray<uintT> E = readEdgeList<uintT>(iFile,numVert);
		G = graphFromEdges(E);
	}
	MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef TIME
	clock_t end_time = clock();
	printf("Preprocessing time: %fs\n", ((double)(end_time-start_time))/CLOCKS_PER_SEC);
#endif

// Processing
#ifdef TIME
	start_time = clock();
#endif

#ifdef MPIFLAG
	dijstra_mpi(G.V,numVert,0,numVert-1);
#else
	dijstra(G.V,G.n,0,numVert-1);
#endif

#ifdef TIME
	end_time = clock();
	printf("Computing time: %fs\n", ((double)(end_time-start_time))/CLOCKS_PER_SEC);
#endif
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
		prev[i] = UNDEFINED_VERT;
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
	if (prev[end] != UNDEFINED_VERT)
		for (int i = prev[end]; prev[i] != UNDEFINED_VERT; i = prev[i]){
			printf("<-%d", i);
		}
	printf("<-%d\n",start);
}

void dijstra_mpi(Vertex<uintT>* V, uintT n, uintT start, uintT end)
{
	// n & start need to be passed to each process
	int rank, p;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&p);
	printf("Rank: %d/%d\n", rank, p);

	uintT glb_min_src, glb_min_dist;
	uintT* glb_dist = NULL;
	intT* glb_prev = NULL;
	uintT* glb_min_src_arr = NULL;
	uintT* glb_min_dist_arr = NULL;
	if (rank == 0){
		glb_dist = newA(uintT,n);
		glb_prev = newA(intT,n);
	}

	// initialization
	int loc_n = (rank != p-1) ? (n / p) : (n - n/p*(p-1));
	uintT* loc_dist = newA(uintT,loc_n);
	intT* loc_prev = newA(intT,loc_n);
	bool* loc_flag = newA(bool,loc_n);
	// data in [loc_start,loc_end)
	int loc_start = rank * loc_n;
	int loc_end = (rank != p-1) ? ((rank+1) * loc_n) : n;
	printf("n: %d start: %d end: %d\n", loc_n, loc_start, loc_end);
	parallel_for(int i = loc_start; i < loc_end; ++i){
		loc_dist[i] = INT_T_MAX;
		loc_prev[i] = UNDEFINED_VERT;
		loc_flag[i] = 1; // in set Q
	}
	if (start >= loc_start && start < loc_end)
		loc_dist[start-loc_start] = 0;

	for (int t = 0; t < n-1; ++t){ // except for start
		uintT loc_src = loc_start;
		uintT loc_min_dist = INT_T_MAX;
		for (int i = loc_start; i < loc_end; ++i)
			if (loc_flag[i] && loc_dist[i] < loc_min_dist){
				loc_min_dist = loc_dist[i];
				loc_src = i;
			}
		if (rank == 0){
			glb_min_src_arr = newA(uintT,p);
			glb_min_dist_arr = newA(uintT,p);
			assert(glb_min_src_arr != NULL);
			assert(glb_min_dist_arr != NULL);
		}
		MPI_Gather(&loc_src,1,MPI_UNSIGNED,glb_min_src_arr,1,MPI_UNSIGNED,0,comm);
		MPI_Gather(&loc_min_dist,1,MPI_UNSIGNED,glb_min_dist_arr,1,MPI_UNSIGNED,0,comm);
		if (rank == 0){
			for (int i = 0; i < p; ++i)
				if (i == 0){
					glb_min_src = glb_min_src_arr[i];
					glb_min_dist = glb_min_dist_arr[i];
				} else if (glb_min_dist_arr[i] < glb_min_dist){
					glb_min_src = glb_min_src_arr[i] + i*p;
					glb_min_dist = glb_min_dist_arr[i];
				}
		}
		MPI_Bcast(&glb_min_src,1,MPI_UNSIGNED,0,comm);
		MPI_Bcast(&glb_min_dist,1,MPI_UNSIGNED,0,comm);

		if (rank == glb_min_src/(n/p))
			loc_flag[glb_min_src-loc_start] = 0;

		uintT out_degree;
		int* send_count;
		int* displs;
		if (rank == glb_min_src/(n/p)){
			out_degree = V[glb_min_src].getOutDegree();
			send_count = newA(int,p);
			displs = newA(int,p);
			for (int i = 0; i < p-1; ++i){
				send_count[i] = (out_degree/p)*2; // with weights
				displs[i] = i*(out_degree/p)*2;
			}
			send_count[p-1] = out_degree*2-(out_degree/p*(p-1)*2);
			displs[p-1] = out_degree/p*(p-1)*2;
			// for (int i = 0; i < p; ++i)
				// printf("Send: %d %d\n", send_count[i], displs[i]);
		}
		MPI_Bcast(&out_degree,1,MPI_UNSIGNED,0,comm);

		uintT* loc_ngh;
		uintT loc_size = (rank != p - 1) ? (out_degree/p)*2 : (out_degree-(out_degree/p*(p-1)))*2;
		// printf("Locsize:%d\n", loc_size);
		loc_ngh = newA(uintT,loc_size);
		MPI_Scatterv(V[glb_min_src].outNeighbors,send_count,displs,MPI_UNSIGNED,loc_ngh,loc_size,MPI_UNSIGNED,0,comm);
		parallel_for(int os = 0; os < loc_size / 2; ++os){ // div 2!!!
			uintT dst = loc_ngh[os*2];
			if (dst >= loc_start && dst < loc_end){ // be careful!!!
				// printf("Dst: %d Rank: %d\n",dst,rank);
				uintT alt = glb_min_dist + loc_ngh[os*2+1];
				if (alt < loc_dist[dst-loc_start]){
					loc_dist[dst-loc_start] = alt;
					loc_prev[dst-loc_start] = glb_min_src;
				}
			}
		}
		MPI_Barrier(comm);
	}

	MPI_Gather(loc_dist,loc_n,MPI_UNSIGNED,glb_dist,loc_n,MPI_UNSIGNED,0,comm);
	MPI_Gather(loc_prev,loc_n,MPI_INT,glb_prev,loc_n,MPI_INT,0,comm);

	// output
	if (rank == 0){
		printf("Dist: %d\n", glb_dist[end]);
		printf("%d", end);
		if (glb_prev[end] != UNDEFINED_VERT)
			for (int i = glb_prev[end]; glb_prev[i] != UNDEFINED_VERT; i = glb_prev[i]){
				printf("<-%d", i);
			}
		printf("<-%d\n",start);
	}

	MPI_Finalize();
}