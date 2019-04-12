#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono> // timing
using namespace std;

using Clock = chrono::high_resolution_clock;

#define MAX_THREAD 64
#define MAX_NUM 1000

mutex push_mutex;
mutex pop_mutex;

template <class T>
void Producer(queue<T>& q, T& val)
{
	while (1){
		lock_guard<mutex> guard(push_mutex);
		if (val < MAX_NUM){
			q.push(val);
			val++;
		} else
			break;
	}
}

template <class T>
void Consumer(queue<T>& q)
{
	while (1){
		lock_guard<mutex> guard(pop_mutex);
		if (!q.empty()){
#ifdef DEBUG
			cout << q.front() << " ";
#endif
			q.pop();
		} else
			break;
	}
}

int main()
{
	queue<int> q;
	thread t[MAX_THREAD];
	int cnt = 0;

	auto t1 = Clock::now();

	for (int i = 0; i < MAX_THREAD; ++i)
		t[i] = thread(Producer<int>,ref(q),ref(cnt));
	for (int i = 0; i < MAX_THREAD; ++i)
		t[i].join();

	for (int i = 0; i < MAX_THREAD; ++i)
		t[i] = thread(Consumer<int>,ref(q));
	for (int i = 0; i < MAX_THREAD; ++i)
		t[i].join();
	cout << endl;

	auto t2 = Clock::now();
	cout << "Multithreaded time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << " ns" << endl;

	t1 = Clock::now();
	for (int i = 0; i < MAX_NUM; ++i)
		q.push(i);
	for (int i = 0; i < MAX_NUM; ++i)
		if (!q.empty()){
#ifdef DEBUG
			cout << q.front() << " ";
#endif
			q.pop();
		}
	t2 = Clock::now();
	cout << "Single thread time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << " ns" << endl;
	
	return 0;
}