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
	DFA_Node() : id(cnt), accepting(false) {
		cnt++;
	}
	int id;
	static int cnt;
	bool accepting;
	map<char,int> out;
};

template<typename T>
void print_set(const set<T>& s, bool newline=true) {
	cout << "Set: ";
	for (auto x : s)
		cout << x << " ";
	if (newline)
		cout << endl;
}

void nfa2dfa(const pair<NFA_Node*,NFA_Node*>& p,
			 const vector<NFA_Node*>& nfa,
			 const set<char>& input_symbol,
			 vector<DFA_Node*>& dfa) {
	NFA_Node* start = p.first;
	NFA_Node* end = p.second;
	// cout << start->id << " " << end->id << endl;
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
		marked.insert(s);
		int len = dfa.size();
		for (auto sym : input_symbol) {
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
				DFA_Node* node = new DFA_Node();
				dfa.push_back(node);
				lut[U] = dfa.size() - 1;
				cout << "Add state: ^ " << dfa.size() - 1 << endl;
			}
			dfa[len-1]->out[sym] = lut[U];
		}
		q.pop();
	}
	for (auto& item : lut) {
		print_set<int>(item.first,false);
		cout << " -> " << item.second << endl;
	}
}

int NFA_Node::cnt = 0;
int DFA_Node::cnt = 0;

// 3.9 Optimization of DFA-Based Pattern Matchers
int main() {
	string str;
	// cin >> str;
	// str = "(ab)*c+(d|e)?";
	// str = "(a|b)*cd";
	str = "(a|b)*abb";
	// str += "#";
	str = insert_concat(str);
	cout << str << endl;
	str = infix2postfix(str);
	cout << str << endl;
	vector<NFA_Node*> nfa;
	pair<NFA_Node*, NFA_Node*> p;
	p = regex2nfa(str,nfa);
	set<char> input_symbol(str.begin(),str.end());
	input_symbol.erase('.');
	print_set<char>(input_symbol);
	vector<DFA_Node*> dfa;
	nfa2dfa(p,nfa,input_symbol,dfa);
	return 0;
}

// https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html
// http://www.cppblog.com/woaidongmao/archive/2010/09/05/97541.html