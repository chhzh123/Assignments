// Hongzheng Chen 17341015
// chenhzh37@mail2.sysu.edu.cn

/********** prod-coms.cpp **********/

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <time.h>
#include <cstring>
#include <chrono> // timing
#include <omp.h>
using namespace std;
using Clock = chrono::high_resolution_clock;

#define MAX_SIZE 100
#define MAX_SUM 100000

int queue[MAX_SIZE];
int cnt = 0;
int nextpush = 0;
int nextpop = -1;
int pushsum = 0;
int popsum = 0;
bool done = 0;

void next(int* ptr)
{
	(*ptr)++;
	if (*ptr >= MAX_SIZE)
		*ptr = 0;
}

void push(int num)
{
	#pragma omp critical
	{
		queue[nextpush] = num;
		if (cnt == 0)
			nextpop = nextpush;
		next(&nextpush);
		cnt++;
#ifdef DEBUG
		cout << "Producing " << num << "..." << endl;
#endif
	}
}

int pop()
{
	int num;
	#pragma omp critical
	{
		if (cnt > 0){
			num = queue[nextpop];
			next(&nextpop);
			cnt--;
			if (cnt == 0)
				nextpop = -1;
#ifdef DEBUG
			cout << "Consuming " << num << "..." << endl;
#endif
		} else
			num = -1;
	}
	return num;
}

void producer()
{
	while (pushsum < MAX_SUM){
		int num = rand() % 73;
		push(num);
		#pragma omp critical
		{
		if (pushsum < MAX_SUM)
			pushsum += num;
		}
	}
}

void consumer()
{
	while (popsum < MAX_SUM){
		int num = pop();
		if (num != -1){
			#pragma omp critical
			{
			if (popsum < MAX_SUM)
				popsum += num;
			}
		}
	}
}

int main(int argc, char* argv[])
{
	srand((int)time(0));

	auto t1 = Clock::now();
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			producer();
		}
		#pragma omp section
		{
			producer();
		}
		#pragma omp section
		{
			consumer();
		}
		#pragma omp section
		{
			consumer();
		}
	}
	auto t2 = Clock::now();

	cout << "Running time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << " ns" << endl;
	cout << "Producer sum: " << pushsum << endl;
	cout << "Consumer sum: " << popsum << endl;

	return 0;
}