// Hongzheng Chen 17341015
// chenhzh37@mail2.sysu.edu.cn

/********** matmul.cpp **********/

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <chrono> // timing
#include <omp.h>
using namespace std;
using Clock = chrono::high_resolution_clock;

#ifndef NWORKER
#define parallel_for _Pragma("omp parallel for") for
#else
#define parallel_for _Pragma("omp parallel for num_threads(2)") for
#endif

typedef float ValType;

/*
 * Compressed Row Format
 * 5 0 0 4
 * 0 3 2 0
 * 1 0 0 0
 *
 * row_offset 0 2 4
 * col_index  0 3 1 2 0
 * val        5 4 3 2 1
 */

template<typename T>
void read_mat(string file_name, int& m, int& n, int& k, int*& row_offset, int*& col_index, T*& val)
{
	cout << "Begin reading matrix..." << endl;

	string str;
	ifstream infile(file_name);
	getline(infile,str); // The first line is useless

	infile >> m >> n >> k;
	cout << "row: " << m << "  col: " << n << "  non-zeros: " << k << endl;
	row_offset = new int [m];
	col_index = new int [k];
	val = new T [k];

	T** mat = new T* [m];
	for (int i = 0; i < m; ++i){
		mat[i] = new T [n];
	}
	for (int l = 0; l < k; ++l){
		int a, b;
		T v;
		infile >> a >> b >> v;
		mat[a-1][b-1] = v; // remember to minus 1
	}
	int cnt = 0;
	for (int i = 0; i < m; ++i){
		row_offset[i] = cnt;
		for (int j = 0; j < n; ++j)
			if (mat[i][j] != 0){
				col_index[cnt] = j;
				val[cnt] = mat[i][j];
				cnt++;
			}
	}

	assert(cnt == k);

#ifdef DEBUG
	for (int i = 0; i < m; ++i)
		cout << row_offset[i] << " ";
	cout << endl;
	for (int i = 0; i < k; ++i)
		cout << col_index[i] << " ";
	cout << endl;
	for (int i = 0; i < k; ++i)
		cout << val[i] << " ";
	cout << endl;
#endif
}

template<typename T>
void seq_mul(const int m, const int n, const int k, const int* row_offset, const int* col_index, const T* val, const T* vec)
{
	T* res = new T [m];
	for (int i = 0; i < m; ++i)
		res[i] = 0;
	auto t1 = Clock::now();

	for (int i = 0; i < m; ++i){
		int offset = (i == m-1 ? k : row_offset[i+1]);
		for (int j = row_offset[i]; j < offset; ++j)
			res[i] += vec[col_index[j]] * val[j];
	}

	auto t2 = Clock::now();
#ifdef DEBUG
	for (int i = 0; i < m; ++i)
		cout << res[i] << " ";
	cout << endl;
#endif
	cout << "Sequential time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << " ns" << endl;
}

template<typename T>
void par_mul(const int m, const int n, const int k, const int* row_offset, const int* col_index, const T* val, const T* vec)
{
	T* res = new T [m];
	parallel_for (int i = 0; i < m; ++i)
		res[i] = 0;
	auto t1 = Clock::now();

	parallel_for (int i = 0; i < m; ++i){
		int offset = (i == m-1 ? k : row_offset[i+1]);
		for (int j = row_offset[i]; j < offset; ++j)
			res[i] += vec[col_index[j]] * val[j];
	}

	auto t2 = Clock::now();
#ifdef DEBUG
	for (int i = 0; i < m; ++i)
		cout << res[i] << " ";
	cout << endl;
#endif
	cout << "Parallel time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << " ns" << endl;
}

int main(int argc, char* argv[])
{
	if (argc >= 3)
		omp_set_num_threads(atoi(argv[2]));
	int m, n, k; // m*n matrix with k non-zero elements
	int* row_offset = nullptr;
	int* col_index = nullptr;
	ValType* val = nullptr;
	read_mat<ValType>(string(argv[1]),m,n,k,row_offset,col_index,val);

	ValType* v = new ValType [n];
	for (int i = 0; i < n; ++i)
		v[i] = i+1;

	seq_mul(m,n,k,row_offset,col_index,val,v);
	par_mul(m,n,k,row_offset,col_index,val,v);

	delete row_offset;
	delete col_index;
	delete val;
	delete v;

	return 0;
}