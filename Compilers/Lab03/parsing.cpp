#include <iostream>
#include <stack>
#include <map>
#include <vector>
#include <cctype>
#include <cstdio>
using namespace std;

/*
 * E  -> T E'
 * E' -> + T E' | eps
 * T  -> F T'
 * T' -> * F T' | eps
 * F  -> (E) | id
 */

// Table-driven predictive parsing
// Input: A string w and a parsing table M for grammar G
// Output: If w \in L(G), a leftmost derivation of w;
//         otherwise, an error indication.

bool is_terminal(char c) {
    // return (c == '+' || c == '-' || isdigit(c));
    return !(c == 'E' || c == 'e' || c == 'T' || c == 't' || c == 'F');
}

int main() {

    map<char, map<char,string> > M;
    M['E']['i'] = "Te"; // E' = e, i = id
    M['E']['('] = "Te";
    M['e']['+'] = "+Te";
    M['e'][')'] = "p";
    M['e']['$'] = "p";
    M['T']['i'] = "Ft";
    M['T']['('] = "Ft";
    M['t']['+'] = "p";
    M['t']['*'] = "*Ft";
    M['t'][')'] = "p";
    M['t']['$'] = "p";
    M['F']['i'] = "i";
    M['F']['('] = "(E)";

    while (true) {
        string str;
        getline(cin,str);
        if (str == "#")
            break;
        str += "$";

        int ip = 0;
        stack<char> stk;
        stk.push('$');
        stk.push('E');
        char a;
        char X = stk.top();
        bool flag = false;
        string matched = "";
        vector<string> stack_str;
        stack_str.push_back("E");
        string last_str;
        vector<string> results;
        results.push_back("E");
        while (X != '$') { // stack not empty
            a = str[ip];
            // cout << ":" << a << ":" << X << endl;
            if (isdigit(a))
                a = 'i';
            if (X == a) {
                if (isdigit(str[ip]))
                    matched += str[ip];
                else
                    matched += a;
                stk.pop();
                last_str = stack_str[stack_str.size()-1];
                stack_str.push_back(last_str.substr(1,stack_str.size()-1));
                ip++;
            } else if (is_terminal(X) ||
                       M.count(X) == 0 ||
                       M[X].count(a) == 0) {
                flag = true;
                break;
            } else { // M[X][a] = X -> Y1Y2...Yk
                // cout << X << "->";
                // for (auto it = M[X][a].begin(); it != M[X][a].end(); ++it)
                //     if (*it == 'e')
                //         cout << "E'";
                //     else if (*it == 't')
                //         cout << "T'";
                //     else if (*it == 'i')
                //         cout << str[ip];
                //     else if (*it == 'p')
                //         cout << "[e]";
                //     else
                //         cout << *it;
                // cout << endl;
                last_str = stack_str[stack_str.size()-1];
                stk.pop();
                last_str = last_str.substr(1,stack_str.size()-1);
                for (auto it = M[X][a].rbegin(); it != M[X][a].rend(); ++it)
                    if (*it != 'p') { // epsilon
                        stk.push(*it);
                        last_str = (*it == 'i' ? str[ip] : *it) + last_str;
                    }
                stack_str.push_back(last_str);
                results.push_back(matched + last_str);
            }
            X = stk.top();
        }
        if (a != '$')
            flag = true;
        if (flag)
            cout << "Syntax Error" << endl;
        else {
            for (auto s : results) {
                for (auto it = s.begin(); it != s.end(); ++it)
                    if (*it == 'e')
                        cout << "E'";
                    else if (*it == 't')
                        cout << "T'";
                    else
                        cout << *it;
                cout << endl;
            }
        }
        cout << endl;
    }
    return 0;
}