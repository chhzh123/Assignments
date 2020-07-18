#include <iostream>
#include <string>
#include <cstring>
#include <stack>
#include <map>
#include <vector>
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
void regex2nfa(string str) {
	stack<NFA_Node*> autostk;
	vector<NFA_Node*> nfa;
	for (auto c : str) {
		if (isalpha(c)) {
			NFA_Node* begin = new NFA_Node();
			NFA_Node* end = new NFA_Node();
			nfa.push_back(begin);
			nfa.push_back(end);
			cout << c << " " << begin->id << " " << end->id << endl;
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
		cout << state->id << ": ";
		for (auto edge : state->e)
			cout << edge << " ";
		cout << endl << "\t";
		for (auto& c : state->out)
			cout << c.first << "(" << c.second << ")";
		cout << endl;
	}
}

int NFA_Node::cnt = 0;

// 3.9 Optimization of DFA-Based Pattern Matchers
int main() {
	string str;
	// cin >> str;
	// str = "(ab)*c+(d|e)?";
	str = "(a|b)*cd";
	// str += "#";
	str = insert_concat(str);
	cout << str << endl;
	str = infix2postfix(str);
	cout << str << endl;
	regex2nfa(str);
	return 0;
}

// https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html
// http://www.cppblog.com/woaidongmao/archive/2010/09/05/97541.html