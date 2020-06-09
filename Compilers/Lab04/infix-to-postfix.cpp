#include <iostream>
#include <string>
#include <stack>
#include <map>
#include <cctype>
using namespace std;

bool is_operator(char c)
{
	return (c == '+' || c == '-' || c == '*' || c == '/');
}

string to_postfix(string infix_str)
{
	map<char,int> prec;
	prec['*'] = 3;
	prec['/'] = 3;
	prec['+'] = 2;
	prec['-'] = 2;
	prec['('] = 1;
	stack<char> opstack;
	string res = "";
	for (auto c : infix_str) {
		if (isdigit(c))
			res += c;
		else if (c == '(')
			opstack.push(c);
		else if (c == ')')
			while (true) {
				char top = opstack.top();
				if (top != '(') {
					res += top;
					opstack.pop();
				} else {
					opstack.pop();
					break;
				}
			}
		else if (is_operator(c)) {
			while (true) {
				if (!opstack.empty()) {
					char top = opstack.top();
					if (prec[top] >= prec[c]) {
						res += top;
						opstack.pop();
					} else {
						opstack.push(c);
						break;
					}
				} else {
					opstack.push(c);
					break;
				}
			}
		}
	}
	while (!opstack.empty()) {
		res += opstack.top();
		opstack.pop();
	}
	return res;
}

int main(){
	while (true) {
		string str;
		getline(cin,str);
		if (str == "#")
			break;
		cout << to_postfix(str) << endl;
	}
	return 0;
}

// https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html