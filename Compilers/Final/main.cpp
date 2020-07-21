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
// https://github.com/Ronan-H/regex-nfa-builder/blob/master/nfa_utils.py
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
			// N(t)
			NFA_Node* d = autostk.top(); autostk.pop();
			NFA_Node* c = autostk.top(); autostk.pop();
			// N(s)
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
		} else if (c == '?') { // zero or one, E|N(t)
			NFA_Node* begin = new NFA_Node();
			NFA_Node* end = new NFA_Node();
			nfa.push_back(begin);
			nfa.push_back(end);
			NFA_Node* b = autostk.top(); autostk.pop();
			NFA_Node* a = autostk.top(); autostk.pop();
			begin->e.push_back(a->id);
			begin->e.push_back(end->id);
			b->e.push_back(end->id);
			b->accepting = false;
			end->accepting = true;
			autostk.push(begin);
			autostk.push(end);
		} else if (c == '+') {
		}
	}
	// for (auto state : nfa) {
	// 	cout << state->id;
	// 	if (state->accepting)
	// 		cout << "(A)";
	// 	cout << ": ";
	// 	for (auto edge : state->e)
	// 		cout << edge << " ";
	// 	for (auto& c : state->out)
	// 		cout << c.first << "(" << c.second << ")";
	// 	cout << endl;
	// }
	NFA_Node* end = autostk.top(); autostk.pop();
	NFA_Node* begin = autostk.top(); autostk.pop();
	pair<NFA_Node*, NFA_Node*> res(begin,end);
	return res;
}

void traverse_e(NFA_Node* s, const vector<NFA_Node*>& nfa, set<int>& res) {
	for (auto neigh : s->e) {
		if (res.find(neigh) != res.end())
			break;
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
	map<char,int> out; // be careful of non-existed keys
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

template<typename T>
void print_vector(const vector<T> v) {
	cout << "Vector: ";
	for (auto x : v)
		cout << x << " ";
	cout << endl;
}

// 3.7.1 Conversion of an NFA to a DFA
int nfa2dfa(const pair<NFA_Node*,NFA_Node*>& p,
			 const vector<NFA_Node*>& nfa,
			 const set<char>& input_symbol,
			 vector<DFA_Node*>& dfa) {
	NFA_Node* start = p.first;
	NFA_Node* end = p.second;
	// cout << "start: " << start->id << " end: " << end->id << endl;
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
			// cout << sym << " ";
			// print_set<int>(move,false);
			// cout << " | ";
			// print_set<int>(U);
			if (marked.find(U) == marked.end()) { // U not in Dstates
				q.push(U);
				marked.insert(U);
				DFA_Node* node = new DFA_Node();
				dfa.push_back(node);
				lut[U] = dfa.size() - 1;
				// cout << "Add state: ^ " << dfa.size() - 1 << endl;
			}
			dfa[idx]->out[sym] = lut[U];
			// cout << ">" << idx << " " << sym << " " << lut[U] << endl;
		}
		q.pop();
	}
	for (auto& item : lut) {
		int idx = item.second;
		if (item.first.count(end->id) != 0) {
			dfa[idx]->accepting = true;
			dfa[idx]->group = 0;
		}
		if (item.first.count(start->id) != 0) {
			dfa[idx]->start = true;
			dfa[idx]->group = 1;
		}
	}
	int res_start;
	for (auto& item : lut) {
		// print_set<int>(item.first,false);
		// cout << " -> " << item.second;
		if (dfa[item.second]->start) {
			// cout << "(S)";
			res_start = item.second;
		}
		if (dfa[item.second]->accepting) {
			// cout << "(A)";
		}
		// cout << endl;
	}
	return res_start;
}

// 3.9.6 Minimizing the Number of States of a DFA
int minimize_dfa(vector<DFA_Node*>& dfa,
				  const set<char>& input_symbol,
				  vector<DFA_Node*>& min_dfa) {
	queue<vector<int>> partition;
	vector<int> s1, s2;
	for (auto state : dfa) {
		// if (state->group)
		// 	s1.push_back(state->id);
		// else // accepting state
		// 	s2.push_back(state->id);
		if (state->start)
			s1.push_back(state->id);
		else
			s2.push_back(state->id);
	}
	int n_group = 2;
	partition.push(s2);
	partition.push(s1);
	map<int,int> state_map;
	set<int> group_id = {0,1};
	while (!partition.empty()) {
		vector<int> p = partition.front();
		print_vector<int>(p);
		partition.pop();
		int size = p.size();
		if (size < 2)
			continue;
		int origin_group = dfa[p[0]]->group;
		for (auto c : input_symbol) {
			map<int,vector<int>> groups; // gid, idx in group
			for (int i = 0; i < size; ++i) {
				int out = dfa[p[i]]->out.at(c);
				groups[dfa[out]->group].push_back(i);
			}
			int g_size = groups.size();
			if (g_size >= 2) {
				group_id.erase(origin_group);
				for (auto& item : groups) {
					n_group++;
					group_id.insert(n_group);
					for (auto idx : item.second)
						dfa[idx]->group = n_group;
					partition.push(item.second);
				}
				break;
			}
		}
	}
	// print_set<int>(group_id);
	int len = dfa.size();
	// be careful that start and end may overlap
	vector<bool> start_flag(n_group,false);
	vector<bool> end_flag(n_group,false);
	int res_start;
	map<int,int> group_map;
	int cnt = 0;
	for (auto id : group_id) {
		group_map[id] = cnt;
		cnt++;
	}
	n_group = group_id.size();
	for (int i = 0; i < len; ++i) {
		dfa[i]->group = group_map[dfa[i]->group];
		int group = dfa[i]->group;
		state_map[group] = i;
		if (dfa[i]->start) {
			start_flag[group] = true;
			res_start = group;
		}
		if (dfa[i]->accepting) {
			end_flag[group] = true;
		}
	}
	for (int i = 0; i < n_group; ++i) {
		DFA_Node* node = new DFA_Node();
		for (auto c : input_symbol) {
			int out = dfa[state_map[i]]->out.at(c);
			node->out[c] = dfa[out]->group;
		}
		if (start_flag[i])
			node->start = true;
		if (end_flag[i])
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
		// cout << (char)(node->id+'A');
		cout << node->id;
		if (node->start)
			cout << "S";
		if (node->accepting)
			cout << "*";
		cout << "\t";
		for (auto c : input_symbol)
			// cout << (char)(node->out[c]+'A') << "\t";
			if (node->out.count(c) == 0)
				cout << (-1) << "\t";
			else
				cout << node->out.at(c) << "\t";
		cout << endl;
	}
}

string get_postfix(string str) {
	str = insert_concat(str);
	// cout << str << endl;
	str = infix2postfix(str);
	// cout << str << endl;
	return str;
}

set<char> get_input_symbol(string str) {
	set<char> input_symbol(str.begin(),str.end());
	input_symbol.erase('E');
	input_symbol.erase('.');
	input_symbol.erase('*');
	input_symbol.erase('?');
	input_symbol.erase('+');
	input_symbol.erase('|');
	// print_set<char>(input_symbol);
	return input_symbol;
}

int build_dfa(string str, set<char>& input_symbol,vector<DFA_Node*>& min_dfa) {
	NFA_Node::cnt = 0;
	vector<NFA_Node*> nfa;
	pair<NFA_Node*, NFA_Node*> p;
	p = regex2nfa(str,nfa);
	DFA_Node::cnt = 0;
	vector<DFA_Node*> dfa;
	int start_dfa = nfa2dfa(p,nfa,input_symbol,dfa);
	// print_dfa(dfa,input_symbol);
	min_dfa = dfa;
	return start_dfa;
	DFA_Node::cnt = 0;
	int start_mindfa = minimize_dfa(dfa,input_symbol,min_dfa);
	print_dfa(min_dfa,input_symbol);
	return start_mindfa;
}

bool intersection(int s1, int s2,
			 vector<DFA_Node*>& dfa1,
			 vector<DFA_Node*>& dfa2,
			 int len1, int len2,
			 set<char>& input_symbol,
			 vector<vector<bool>>& visited) {
	visited[s1][s2] = true;
	bool acc1 = (s1 == len1) ? false : dfa1[s1]->accepting;
	bool acc2 = (s2 == len2) ? true : dfa2[s2]->accepting;
	// cout << s1 << " " << acc1 << " " << s2 << " " << acc2 << endl;
	if (acc1 && acc2)
		return true;
	for (auto c : input_symbol) {
		int o1 = (s1 == len1 || dfa1[s1]->out.count(c) == 0) ?
				  len1 : dfa1[s1]->out.at(c);
		int o2 = (s2 == len2 || dfa2[s2]->out.count(c) == 0) ?
				  len2 : dfa2[s2]->out.at(c);
		if (!visited[o1][o2])
			if (intersection(o1,o2,dfa1,dfa2,len1,len2,input_symbol,visited))
				return true;
	}
	return false;
}

void complement(vector<DFA_Node*>& dfa) {
	for (auto node : dfa) {
		node->accepting = !node->accepting;
	}
}

int judge(vector<DFA_Node*>& dfa1, vector<DFA_Node*>& dfa2,
		  set<char>& symbol1, set<char>& symbol2,
		  int s1, int s2) {
	// cout << "start: " << s1 << " " << s2 << endl;
	vector<vector<bool>> visited;
	// add a dummy state
	int len1 = dfa1.size();
	int len2 = dfa2.size();
	for (int i = 0; i < len1+1; ++i) {
		vector<bool> tmp(len2+1,false);
		visited.push_back(tmp);
	}
	complement(dfa2);
	// print_dfa(dfa1,symbol1);
	// print_dfa(dfa2,symbol2);
	bool a_in_cb = intersection(s1,s2,dfa1,dfa2,len1,len2,symbol1,visited);
	// cout << endl;
	visited.clear();
	for (int i = 0; i < len2+1; ++i) {
		vector<bool> tmp(len1+1,false);
		visited.push_back(tmp);
	}
	complement(dfa2);
	complement(dfa1);
	// print_dfa(dfa1,symbol1);
	bool b_in_ca = intersection(s2,s1,dfa2,dfa1,len2,len1,symbol2,visited);
	if (!a_in_cb && !b_in_ca)
		return 0;
	else if (!a_in_cb && b_in_ca)
		return 1;
	else if (a_in_cb && !b_in_ca)
		return 2;
	else
		return 3;
}

string insert_plus(const string str) {
	// cannot handle "((x|y)+)+"
	if (str.find("+") == string::npos)
		return str;
	string res = "";
	stack<int> stk;
	int len = str.size();
	for (int i = 0; i < len; ++i) {
		if (str[i] == '(') {
			stk.push(i);
			res += "(";
		} else if (str[i] == ')') {
			if (i + 1 < len && str[i+1] == '+') {
				int front = stk.top();
				string new_str = str.substr(front,i-front+1);
				res += ")" + new_str + "*";
				stk.pop();
			} else
				res += ")";
		} else if (str[i] == '+' && str[i-1] != ')') {
			char c = str[i-1];
			res += c;
			res += "*";
		} else if (str[i] != '+')
			res += str[i];
	}
	return res;
}

int NFA_Node::cnt = 0;
int DFA_Node::cnt = 0;

#ifdef NO_STDIN
const vector<vector<string>> input_str = {
	{"((E|a)b*)*", "(a|b)*"}, // =
	{"b*a*b?a*", "b*a*ba*|b*a*"}, // =
	{"b*a*b?a*", "(b*|a*)(b|E)a*"}, // >
	{"(c|d)*c(c|d)(c|d)", "(c|d)*d(c|d)(c|d)"}, // !
	{"x+y+z+", "x*y*z*"} // <
};
#endif

// 3.9 Optimization of DFA-Based Pattern Matchers
int main() {
	int n_case;
#ifdef NO_STDIN
	n_case = input_str.size();
#else
	cin >> n_case;
#endif
	for (int case_id = 0; case_id < n_case; ++case_id) {
		string str1, str2;
#ifdef NO_STDIN
		str1 = input_str[case_id][0];
		str2 = input_str[case_id][1];
#else
		cin >> str1 >> str2;
#endif
		str1 = insert_plus(str1);
		str2 = insert_plus(str2);
		// cout << str1 << endl << str2 << endl;
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
	}
	return 0;
}

// str = "(ab)*c+(d|e)?";
// str = "(a|b)*cd";
// str1 = "(a|b)*abb";
// str2 = "(a|b)*abb";
// str1 = "a";
// str2 = "a+";
// str1 = "(x|y)+(y+b*)*";
// str2 = "((x|y)+)+";

// https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html
// http://www.cppblog.com/woaidongmao/archive/2010/09/05/97541.html