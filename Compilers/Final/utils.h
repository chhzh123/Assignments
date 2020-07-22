#ifndef UTILS_H
#define UTILS_H

#include <set>
#include <vector>
using namespace std;

template<typename T>
void print_set(const set<T>& s, bool newline=true) {
	cout << "Set: ";
	for (auto x : s)
		cout << x << " ";
	if (newline)
		cout << endl;
}

template<typename T>
void print_vector(const vector<T> v) {
	cout << "Vector: ";
	for (auto x : v)
		cout << x << " ";
	cout << endl;
}

#endif // UTILS_H