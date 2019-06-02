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

void dijkstra(Vertex<uintT>* V, uintT n, uintT start, uintT end);
#ifdef MPIFLAG
void dijkstra_mpi(Vertex<uintT>* V, uintT n, uintT start, uintT end);
#endif

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
	// MPI_Barrier(MPI_COMM_WORLD);
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
	dijkstra_mpi(G.V,numVert,0,numVert-1);
#else
	dijkstra(G.V,G.n,0,numVert-1);
#endif

#ifdef TIME
	end_time = clock();
	printf("Computing time: %fs\n", ((double)(end_time-start_time))/CLOCKS_PER_SEC);
#endif
	return 0;
}

struct DistPack
{
	int i;
	uintT dist;
	intT prev;
	bool flag;
	DistPack():
		i(UNDEFINED_VERT),dist(INT_T_MAX),prev(UNDEFINED_VERT),flag(0){}
};

struct minPackF
{
	DistPack operator() (const DistPack& a, const DistPack& b) const {
		if (a.flag && b.flag)
			return (a.dist < b.dist) ? a : b;
		else if (!a.flag && !b.flag)
			return DistPack();
		else if (!a.flag)
			return b;
		else
			return a;
	}
};

void dijkstra(Vertex<uintT>* V, uintT n, uintT start, uintT end)
{
	// initialization
	DistPack* res = newA(DistPack,n);
	parallel_for(int i = 0; i < n; ++i){
		res[i].i = i;
		res[i].dist = INT_T_MAX;
		res[i].prev = UNDEFINED_VERT;
		res[i].flag = 1; // in set Q
	}
	res[start].dist = 0;

	for (int t = 0; t < n-1; ++t){ // except for start
		DistPack min_pack = sequence::reduce(res,n,minPackF());
		uintT src = min_pack.i;
		res[src].flag = 0;
		if (src == end) // early break
			break;
		Vertex<uintT> v_src = V[src];
		parallel_for(int os = 0; os < v_src.getOutDegree(); ++os){
			uintT dst = v_src.getOutNeighbor(os);
			uintT alt = res[src].dist + v_src.getOutWeight(os);
			if (alt < res[dst].dist){
				res[dst].dist = alt;
				res[dst].prev = src;
			}
		}
	}

	// output
	printf("Dist: %d\n", res[end].dist);
	printf("%d", end);
	if (res[end].prev != UNDEFINED_VERT)
		for (int i = res[end].prev; res[i].prev != UNDEFINED_VERT; i = res[i].prev){
			printf("<-%d", i);
		}
	printf("<-%d\n",start);
}

#ifdef MPIFLAG
void dijkstra_mpi(Vertex<uintT>* V, uintT n, uintT start, uintT end)
{
	// n & start need to be passed to each process
	int rank, p;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&p);
#ifdef DEBUG
	printf("Rank: %d/%d\n", rank, p);
#endif

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
	int loc_start = rank * (n/p);
	int loc_end = (rank != p-1) ? ((rank+1) * loc_n) : n;
#ifdef DEBUG
	printf("n: %d start: %d end: %d\n", loc_n, loc_start, loc_end);
#endif
	parallel_for_4_threads(int i = 0; i < loc_n; ++i){
		loc_dist[i] = INT_T_MAX;
		loc_prev[i] = UNDEFINED_VERT;
		loc_flag[i] = 1; // in set Q
	}
	int plc_rank = start/(n/p);
	if (rank == ((plc_rank < p) ? plc_rank : p-1))
		loc_dist[start-loc_start] = 0;

	// begin iteration
	for (int t = 0; t < n-1; ++t){ // except for start
#ifdef TIME
	clock_t start_time = clock();
#endif
		// find minimum
		uintT loc_src = loc_start;
		uintT loc_min_dist = INT_T_MAX;
		for (int i = 0; i < loc_n; ++i)
			if (loc_flag[i] && loc_dist[i] < loc_min_dist){
				loc_min_dist = loc_dist[i];
				loc_src = i + loc_start;
			}
		int loc_min_pack[2], glb_min_pack[2];
		loc_min_pack[0] = loc_min_dist;
		loc_min_pack[1] = loc_src;
		MPI_Allreduce(loc_min_pack,glb_min_pack,1,MPI_2INT,MPI_MINLOC,comm);
		glb_min_dist = glb_min_pack[0];
		glb_min_src = glb_min_pack[1];
		assert(glb_min_src < n);
		if (glb_min_src == end) // early break!
			break;

		// set glb_min_src out of set Q
		plc_rank = glb_min_src/(n/p);
		if (rank == ((plc_rank < p) ? plc_rank : p-1))
			loc_flag[glb_min_src-loc_start] = 0;

		// scatter neighbor array
		uintT out_degree;
		uintT* loc_ngh = NULL;
		if (rank == 0){
			out_degree = V[glb_min_src].getOutDegree();
			loc_ngh = V[glb_min_src].outNeighbors;
		}
		MPI_Bcast(&out_degree,1,MPI_UNSIGNED,0,comm);
		if (rank != 0)
			loc_ngh = newA(uintT,out_degree*2);

		// not scatter!!!
		MPI_Bcast(loc_ngh,out_degree*2,MPI_INT,0,comm);

		parallel_for_4_threads(int os = 0; os < out_degree; ++os){
			uintT dst = loc_ngh[os*2];
			// printf("Src: %d Dst: %d Rank: %d\n",glb_min_src,dst,rank);
			if (dst >= loc_start && dst < loc_end){ // be careful!!!
				uintT alt = glb_min_dist + loc_ngh[os*2+1];
				if (alt < loc_dist[dst-loc_start]){
					loc_dist[dst-loc_start] = alt;
					loc_prev[dst-loc_start] = glb_min_src;
				}
			}
		}
		// MPI_Barrier(comm);
#ifdef DEBUG
		printf("Res-rank %d:\n", rank);
		for (int i = 0; i < loc_n; ++i)
			printf("%d ", loc_dist[i]);
		printf("\n");
		for (int i = 0; i < loc_n; ++i)
			printf("%d ", loc_prev[i]);
		printf("\n");
		for (int i = 0; i < loc_n; ++i)
			printf("%d ", loc_flag[i]);
		printf("\n");
#endif
#ifdef TIME
	clock_t end_time = clock();
	printf("Iter %d time: %fs\n", t,((double)(end_time-start_time))/CLOCKS_PER_SEC);
#endif
	}

	int* recvcounts = NULL;
	int* displs = NULL;
	if (rank == 0){
		recvcounts = newA(int,p);
		displs = newA(int,p);
		for (int i = 0; i < p; ++i){
			recvcounts[i] = n / p;
			displs[i] = n/p * i;
		}
		recvcounts[p-1] = n - n/p*(p-1);
		displs[p-1] = n/p*(p-1);
	}
	MPI_Gatherv(loc_dist,loc_n,MPI_UNSIGNED,glb_dist,recvcounts,displs,MPI_UNSIGNED,0,comm);
	MPI_Gatherv(loc_prev,loc_n,MPI_INT,glb_prev,recvcounts,displs,MPI_INT,0,comm);

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
#endif