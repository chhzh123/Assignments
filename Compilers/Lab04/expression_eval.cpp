#include <iostream>
#include <string>
#include <stack>
#include <map>
#include <vector>
#include <cctype>
using namespace std;

bool is_operator(char c)
{
	return (c == '+' || c == '-' || c == '*' || c == '/');
}

vector<string> to_postfix(string infix_str)
{
	map<char,int> prec;
	prec['*'] = 3;
	prec['/'] = 3;
	prec['+'] = 2;
	prec['-'] = 2;
	prec['('] = 1;
	stack<char> opstack;
	vector<string> res;
	string num;
	for (auto c : infix_str) {
		if (isdigit(c)){
			num += c;
			continue;
		} else if (num != "") {
			res.push_back(num);
			num = "";
		}
		// ( ) + - * /
		if (c == '(') {
			opstack.push(c);
		} else if (c == ')')
			while (true) {
				char top = opstack.top();
				if (top != '(') {
					string top_str;
					top_str += top;
					res.push_back(top_str);
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
						string top_str;
						top_str += top;
						res.push_back(top_str);
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
	if (num != "") {
		res.push_back(num);
		num = "";
	}
	while (!opstack.empty()) {
		char top = opstack.top();
		string top_str;
		top_str += top;
		res.push_back(top_str);
		opstack.pop();
	}
	return res;
}

int eval(vector<string>& v) {
	stack<int> operandStack;
	for (auto item : v) {
		if (isdigit(item[0])) { // operand
			int operand = stoi(item);
			operandStack.push(operand);
		} else if (is_operator(item[0])) {
			int op2 = operandStack.top();
			operandStack.pop();
			int op1 = operandStack.top();
			operandStack.pop();
			int res;
			switch (item[0]) {
				case '+': res = op1 + op2;break;
				case '-': res = op1 - op2;break;
				case '*': res = op1 * op2;break;
				case '/': res = op1 / op2;break;
			}
			operandStack.push(res);
		}
	}
	return operandStack.top();
}

int main(){
	while (true) {
		string str;
		getline(cin,str);
		if (str == "#")
			break;
		vector<string> vec = to_postfix(str);
		// for (auto s : vec)
		// 	cout << s << " ";
		// cout << endl;
		cout << eval(vec) << endl;
	}
	return 0;
}

// https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html