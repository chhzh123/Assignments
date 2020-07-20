#include <iostream>
#include <string>
#include <cstring>
#include <stack>
#include <map>
#include <vector>
#include <queue>
#include <set>
#include <utility>
using namespace std;

/*
 * a-z E
 * ( )
 * * ? +
 * |
 */
// Construct DFA directly from a regex
// Minimize # of states of DFA, O(nlogn)

string insert_concat(string str) {
	string res = "";
	int i = 0;
	int len = str.size();
	for (auto c1 : str) {
		res += c1;
		if (i + 1 < len) {
			char c2 = str[i + 1];
			if (c1 != '(' && c1 != '|' &&
				c2 != '*' && c2 != '?' && c2 != '+' &&
				c2 != '|' && c2 != ')')
				res += '.';
		}
		i++;
	}
	return res;
}

int prec(char c) {
	switch (c) {
		case '(':
		case ')': return 3;
		case '*':
		case '?':
		case '+': return 2;
		case '.': return 1;
		case '|': return 0;
		default: return -1;
	}
}

// https://www.geeksforgeeks.org/stack-set-2-infix-to-postfix/
string infix2postfix(string str) {
	string res = "";
	stack<char> opstk;
	for (auto c : str) {
		// cout << c << " " << res << endl;
		if (isalpha(c)) // operand
			res += c;
		else if (c == '(')
			opstk.push('(');
		else if (c == ')') {
			char top = opstk.top();
			while (top != '(') {
				res += top;
				opstk.pop();
				top = opstk.top();
			}
			opstk.pop(); // discard '('
		} else { // operator
			while (!opstk.empty() && opstk.top() != '(' && prec(c) <= prec(opstk.top())) {
				char top = opstk.top();
				res += top;
				opstk.pop();
			}
			opstk.push(c);
		}
	}
	// pop all remaining ops in the stack
	while (!opstk.empty()) {
		char top = opstk.top();
		res += top;
		opstk.pop();
	}
	return res;
}

class NFA_Node{
public:
	NFA_Node() : id(cnt), accepting(false) {
		cnt++;
	}
	int id;
	static int cnt;
	bool accepting;
	map<char,int> out;
	vector<int> e;
};

// 3.7.4 Construction of an NFA from a Regular Expression
// McNaughton-Yamada-Thompson algorithm
pair<NFA_Node*,NFA_Node*> regex2nfa(const string str, vector<NFA_Node*>& nfa) {
	stack<NFA_Node*> autostk;
	for (auto c : str) {
		if (isalpha(c)) {
			NFA_Node* begin = new NFA_Node();
			NFA_Node* end = new NFA_Node();
			nfa.push_back(begin);
			nfa.push_back(end);
			// cout << c << " " << begin->id << " " << end->id << endl;
			if (c != 'E')
				begin->out[c] = end->id;
			else
				begin->e.push_back(end->id);
			end->accepting = true;
			autostk.push(begin);
			autostk.push(end);
		} else if (c == '|') { // union
			NFA_Node* begin = new NFA_Node();
			NFA_Node* end = new NFA_Node();
			nfa.push_back(begin);
			nfa.push_back(end);
			NFA_Node* d = autostk.top(); autostk.pop();
			NFA_Node* c = autostk.top(); autostk.pop();
			NFA_Node* b = autostk.top(); autostk.pop();
			NFA_Node* a = autostk.top(); autostk.pop();
			begin->e.push_back(a->id);
			begin->e.push_back(c->id);
			b->e.push_back(end->id);
			d->e.push_back(end->id);
			b->accepting = false;
			d->accepting = false;
			end->accepting = true;
			autostk.push(begin);
			autostk.push(end);
		} else if (c == '.') { // concatenation
			NFA_Node* d = autostk.top(); autostk.pop();
			NFA_Node* c = autostk.top(); autostk.pop();
			NFA_Node* b = autostk.top(); autostk.pop();
			NFA_Node* a = autostk.top(); autostk.pop();
			b->e.push_back(c->id);
			b->accepting = false;
			d->accepting = true;
			autostk.push(a);
			autostk.push(d);
		} else if (c == '*') { // Kleen closure
			NFA_Node* begin = new NFA_Node();
			NFA_Node* end = new NFA_Node();
			nfa.push_back(begin);
			nfa.push_back(end);
			NFA_Node* b = autostk.top(); autostk.pop();
			NFA_Node* a = autostk.top(); autostk.pop();
			b->e.push_back(a->id);
			begin->e.push_back(end->id);
			begin->e.push_back(a->id);
			b->e.push_back(end->id);
			b->accepting = false;
			end->accepting = true;
			autostk.push(begin);
			autostk.push(end);
		} else if (c == '?') {
		} else if (c == '+') {}
	}
	for (auto state : nfa) {
		cout << state->id;
		if (state->accepting)
			cout << "(A)";
		cout << ": ";
		for (auto edge : state->e)
			cout << edge << " ";
		for (auto& c : state->out)
			cout << c.first << "(" << c.second << ")";
		cout << endl;
	}
	NFA_Node* end = autostk.top(); autostk.pop();
	NFA_Node* begin = autostk.top(); autostk.pop();
	pair<NFA_Node*, NFA_Node*> res(begin,end);
	return res;
}

void traverse_e(NFA_Node* s, const vector<NFA_Node*>& nfa, set<int>& res) {
	for (auto neigh : s->e) {
		res.insert(neigh);
		traverse_e(nfa[neigh], nfa, res);
	}
}

set<int> epsilon_closure(NFA_Node* s, const vector<NFA_Node*>& nfa) {
	set<int> res;
	res.insert(s->id);
	traverse_e(s,nfa,res);
	return res;
}

set<int> epsilon_closure(const set<int>& T, const vector<NFA_Node*>& nfa) {
	set<int> res;
	for (auto s : T) {
		set<int> tmp = epsilon_closure(nfa[s],nfa);
		res.insert(tmp.begin(),tmp.end());
	}
	return res;
}

void traverse(NFA_Node* s, const vector<NFA_Node*>& nfa, const char a, set<int>& res) {
	if (s->out.count(a) != 0)
		res.insert(s->out[a]);
}

set<int> move_to(const set<int>& T, const vector<NFA_Node*>& nfa, const char a) {
	set<int> res;
	for (auto s : T)
		traverse(nfa[s],nfa,a,res);
	return res;
}

class DFA_Node{
public:
	DFA_Node() : id(cnt), start(false), accepting(false), group(1) {
		cnt++;
	}
	int id;
	static int cnt;
	bool start;
	bool accepting;
	map<char,int> out;
	int group;
};

template<typename T>
void print_set(const set<T>& s, bool newline=true) {
	cout << "Set: ";
	for (auto x : s)
		cout << x << " ";
	if (newline)
		cout << endl;
}

// 3.7.1 Conversion of an NFA to a DFA
void nfa2dfa(const pair<NFA_Node*,NFA_Node*>& p,
			 const vector<NFA_Node*>& nfa,
			 const set<char>& input_symbol,
			 vector<DFA_Node*>& dfa) {
	NFA_Node* start = p.first;
	NFA_Node* end = p.second;
	cout << "start: " << start->id << " end: " << end->id << endl;
	queue<set<int>> q;
	set<int> start_closure = epsilon_closure(start,nfa);
	q.push(start_closure);
	DFA_Node* node = new DFA_Node();
	dfa.push_back(node);
	set<set<int>> marked;
	map<set<int>,int> lut;
	lut[q.front()] = 0;
	while (!q.empty()) {
		set<int> s = q.front();
		// marked.insert(s);
		int idx = lut[s];
		for (auto sym : input_symbol) { // only alphas
			set<int> move = move_to(s,nfa,sym);
			if (move.empty())
				continue;
			set<int> U = epsilon_closure(move,nfa);
			cout << sym << " ";
			print_set<int>(move,false);
			cout << " | ";
			print_set<int>(U);
			if (marked.find(U) == marked.end()) { // U not in Dstates
				q.push(U);
				marked.insert(U);
				DFA_Node* node = new DFA_Node();
				dfa.push_back(node);
				lut[U] = dfa.size() - 1;
				cout << "Add state: ^ " << dfa.size() - 1 << endl;
			}
			dfa[idx]->out[sym] = lut[U];
			cout << ">" << idx << " " << sym << " " << lut[U] << endl;
		}
		q.pop();
	}
	for (auto& item : lut) {
		int idx = item.second;
		if (item.first.count(end->id) != 0) {
			cout << end->id << " " << item.second << endl;
			dfa[idx]->accepting = true;
			dfa[idx]->group = 0;
		} else if (item.first.count(start->id) != 0) {
			dfa[idx]->start = true;
		}
	}
	for (auto& item : lut) {
		print_set<int>(item.first,false);
		cout << " -> " << item.second;
		if (dfa[item.second]->start)
			cout << "(S)";
		else if (dfa[item.second]->accepting)
			cout << "(A)";
		cout << endl;
	}
}

// 3.9.6 Minimizing the Number of States of a DFA
int minimize_dfa(vector<DFA_Node*>& dfa,
				  const set<char>& input_symbol,
				  vector<DFA_Node*>& min_dfa) {
	queue<vector<int>> partition;
	vector<int> s1, s2;
	for (auto state : dfa) {
		if (state->group)
			s1.push_back(state->id);
		else // accepting state
			s2.push_back(state->id);
	}
	int group_id = 1; // count from 0
	partition.push(s2);
	partition.push(s1);
	vector<int> state_map;
	while (!partition.empty()) {
		vector<int> p = partition.front();
		partition.pop();
		int size = p.size();
		if (size < 2)
			continue;
		for (auto c : input_symbol) {
			map<int,vector<int>> groups;
			for (int i = 0; i < size; ++i) {
				int out = dfa[p[i]]->out[c];
				groups[dfa[out]->group].push_back(i);
			}
			int g_size = groups.size();
			if (g_size >= 2) {
				for (auto& item : groups) {
					for (auto idx : item.second)
						dfa[idx]->group = group_id;
					partition.push(item.second);
					state_map.push_back(item.second[0]);
					group_id++;
				}
				group_id--; // reduce the initial group
				break;
			}
		}
	}
	int len = dfa.size();
	vector<int> start_end_flag(group_id+1,0); // 1 start, 2 end
	int res_start;
	for (int i = 0; i < len; ++i) {
		int group = dfa[i]->group;
		if (dfa[i]->start) {
			start_end_flag[group] = 1;
			res_start = group;
		} else if (dfa[i]->accepting) {
			start_end_flag[group] = 2;
		}
	}
	for (int i = 0; i <= group_id; ++i) {
		DFA_Node* node = new DFA_Node();
		for (auto c : input_symbol) {
			int out = dfa[state_map[i]]->out[c];
			node->out[c] = dfa[out]->group;
		}
		if (start_end_flag[i] == 1)
			node->start = true;
		else if (start_end_flag[i] == 2)
			node->accepting = true;
		min_dfa.push_back(node);
	}
	return res_start;
}

void print_dfa(vector<DFA_Node*>& dfa, set<char>& input_symbol) {
	for (auto c : input_symbol)
		cout << "\t" << c;
	cout << endl;
	for (auto node : dfa) {
		cout << node->id;
		if (node->start)
			cout << "S";
		else if (node->accepting)
			cout << "*";
		cout << "\t";
		for (auto c : input_symbol)
			cout << node->out[c] << "\t";
		cout << endl;
	}
}

string get_postfix(string str) {
	str = insert_concat(str);
	cout << str << endl;
	str = infix2postfix(str);
	cout << str << endl;
	return str;
}

set<char> get_input_symbol(string str) {
	set<char> input_symbol(str.begin(),str.end());
	input_symbol.erase('.');
	input_symbol.erase('*');
	input_symbol.erase('?');
	input_symbol.erase('+');
	input_symbol.erase('|');
	print_set<char>(input_symbol);
	return input_symbol;
}

int build_dfa(string str, set<char>& input_symbol,vector<DFA_Node*>& min_dfa) {
	NFA_Node::cnt = 0;
	vector<NFA_Node*> nfa;
	pair<NFA_Node*, NFA_Node*> p;
	p = regex2nfa(str,nfa);
	DFA_Node::cnt = 0;
	vector<DFA_Node*> dfa;
	nfa2dfa(p,nfa,input_symbol,dfa);
	print_dfa(dfa,input_symbol);
	DFA_Node::cnt = 0;
	int start = minimize_dfa(dfa,input_symbol,min_dfa);
	print_dfa(min_dfa,input_symbol);
	return start;
}

bool contain(int s1, int s2,
			 vector<DFA_Node*>& dfa1,
			 vector<DFA_Node*>& dfa2,
			 set<char>& input_symbol,
			 vector<vector<bool>>& visited) {
	if (visited[s1][s2])
		return true;
	visited[s1][s2] = true;
	bool acc1 = dfa1[s1]->accepting;
	bool acc2 = dfa2[s2]->accepting;
	if (acc1 && acc2)
		return true;
	else if (acc1 || acc2)
		return false;
	for (auto c : input_symbol) {
		if (dfa2[s2]->out.find(c) == dfa2[s2]->out.end())
			return false;
		int o1 = dfa1[s1]->out[c];
		int o2 = dfa2[s2]->out[c];
		if (!contain(o1,o2,dfa1,dfa2,input_symbol,visited))
			return false;
	}
	return true;
}

int judge(vector<DFA_Node*>& dfa1, vector<DFA_Node*>& dfa2,
		  set<char>& symbol1, set<char>& symbol2,
		  int s1, int s2) {
	cout << "start: " << s1 << " " << s2 << endl;
	vector<vector<bool>> visited;
	int len1 = dfa1.size();
	int len2 = dfa2.size();
	for (int i = 0; i < len1; ++i) {
		vector<bool> tmp(len2,false);
		visited.push_back(tmp);
	}
	bool a_in_b = contain(s1,s2,dfa1,dfa2,symbol1,visited);
	visited.clear();
	for (int i = 0; i < len1; ++i) {
		vector<bool> tmp(len2,false);
		visited.push_back(tmp);
	}
	bool b_in_a = contain(s2,s1,dfa2,dfa1,symbol2,visited);
	if (a_in_b && b_in_a)
		return 0;
	else if (a_in_b && !b_in_a)
		return 1;
	else if (!a_in_b && b_in_a)
		return 2;
	else
		return 3;
}

int NFA_Node::cnt = 0;
int DFA_Node::cnt = 0;

// 3.9 Optimization of DFA-Based Pattern Matchers
int main() {
	string str1, str2;
	// cin >> str;
	// str = "(ab)*c+(d|e)?";
	// str = "(a|b)*cd";
	str1 = "(a|b)*abb";
	str2 = "(a|b)*abb";
	// str += "#";
	str1 = get_postfix(str1);
	str2 = get_postfix(str2);
	set<char> symbol1 = get_input_symbol(str1);
	set<char> symbol2 = get_input_symbol(str2);
	vector<DFA_Node*> dfa1;
	int s1 = build_dfa(str1,symbol1,dfa1);
	vector<DFA_Node*> dfa2;
	int s2 = build_dfa(str2,symbol2,dfa2);
	int res = judge(dfa1,dfa2,symbol1,symbol2,s1,s2);
	if (res == 0)
		cout << "=" << endl;
	else if (res == 1)
		cout << "<" << endl;
	else if (res == 2)
		cout << ">" << endl;
	else
		cout << "!" << endl;
	return 0;
}

// https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html
// http://www.cppblog.com/woaidongmao/archive/2010/09/05/97541.html