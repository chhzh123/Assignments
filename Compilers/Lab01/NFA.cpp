#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <algorithm>
using namespace std;

vector<vector<set<int>>> transition;

set<int> parse_num(const string str){
	set<int> res;
	int n = str.size();
	int front = 0;
	for (int i = 0; i < n; ++i){
		if (str[i] == ',' or i == n-1) {
			string s = str.substr(front,((i != n-1) ? (i-front) : (i-front+1)));
			res.insert(stoi(s));
			front = i+1;
		}
	}
	return res;
}

set<int> move(const set<int>& states, const int c){
	set<int> res;
	for (auto s : states)
		for (auto new_state : transition[s][c])
			res.insert(new_state);
	return res;
}

set<int> epsilon_closure(set<int> states){
	stack<int> stk;
	for (auto s : states)
		stk.push(s);
	while (!stk.empty()) {
		int top = stk.top();
		stk.pop();
		for (auto u : transition[top][0])
			if (states.find(u) == states.end()){ // u not in closure
				states.insert(u);
				stk.push(u);
			}
	}
	return states;
}

void print_set(const set<int>& states){
	cout << "S: ";
	for (auto s : states)
		cout << s << " ";
	cout << endl;
}

int main(){
	while (true){
		int n, m;
		cin >> n >> m;
		if (n == 0 && m == 0)
			break;
		cin.ignore();
		// transition matrix
		transition.clear();
		for (int i = 0; i < n; ++i) {
			string str;
			vector<set<int>> tmp;
			transition.push_back(tmp);
			for (int j = 0; j < m; ++j) { // including epsilon
				cin >> str;
				str = str.substr(1,str.size()-2);
				if (str == "") {
					set<int> empty;
					transition[i].push_back(empty);
				} else
					transition[i].push_back(parse_num(str));
			}
			cin.ignore();
		}
		// accepting state
		set<int> accepting;
		for (int i = 0; i < n; ++i) {
			int num;
			cin >> num;
			if (num == -1)
				break;
			else
				accepting.insert(num);
		}
		// queries
		while (true) {
			string str;
			cin >> str;
			if (str[0] == '#')
				break;
			set<int> initial;
			initial.insert(0);
			set<int> S = epsilon_closure(initial);
			// print_set(S);
			for (auto c : str) {
				S = epsilon_closure(move(S,c-'a'+1));
				// print_set(S);
			}
			vector<int> c;
			set_intersection(S.begin(),S.end(),accepting.begin(),accepting.end(),back_inserter(c));
			if (c.size() > 0)
				cout << "YES" << endl;
			else
				cout << "NO" << endl;
		}
	}
}

// http://soj.acmm.club/show_problem.php?pid=1001&cid=2834
// Input:
// 4 3
// {} {0,1} {0}
// {} {} {2}
// {} {} {3}
// {} {} {}
// 3 -1
// aaabb
// abbab
// abbaaabb
// abbb
// #
// 0 0

// Output:
// YES
// NO
// YES
// NO