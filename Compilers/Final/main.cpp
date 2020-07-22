#include <iostream>
#include <string>
#include <cstring>
#include <stack>
#include <map>
#include <vector>
#include <queue>
#include <set>
#include <utility>
// #include "utils.h"
using namespace std;

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

void print_nfa(const vector<NFA_Node*>& nfa) {
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
}

void print_dfa(const vector<DFA_Node*>& dfa, const set<char>& input_symbol) {
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

template<typename T>
void free_node(vector<T*> v) {
    for (auto it = v.begin(); it != v.end(); ++it)
        delete *it;
    v.clear();
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

/*
 * R+ is equivalent to RR* (one or more)
 * Since NFA engine cannot recognize +,
 * substitute + using * first.
 */
string substitute_plus(const string str) {
    if (str.find("+") == string::npos)
        return str;
    string res = "";
    stack<int> out_stk;
    int len = str.size();
    // do parentheses matching & make duplication
    for (int i = 0; i < len; ++i) {
        if (str[i] == '(') {
            res += "(";
            out_stk.push(res.size()-1);
        } else if (str[i] == ')') {
            if (i + 1 < len && str[i+1] == '+') {
                int front = out_stk.top();
                res += ")";
                string new_str = res.substr(front,res.size()-front);
                res += new_str + "*";
            } else {
                res += ")";
            }
            out_stk.pop();
        } else if (str[i] == '+' && str[i-1] != ')') {
            char c = str[i-1];
            res += c;
            res += "*";
        } else if (str[i] != '+')
            res += str[i];
    }
    return res;
}

/*
 * To easier parse the regex and change it
 * to postfix format, need to insert concatenation
 * sign first. Here use "." to represent.
 */
string insert_concat(const string str) {
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

/*
 * Change the regex to postfix format
 * Use a stack to maintain operators
 */
string infix2postfix(const string str) {
    string res = "";
    stack<char> op_stk;
    for (auto c : str) {
        if (isalpha(c)) // operand
            res += c;
        else if (c == '(')
            op_stk.push('(');
        else if (c == ')') {
            char top = op_stk.top();
            while (top != '(') {
                res += top;
                op_stk.pop();
                top = op_stk.top();
            }
            op_stk.pop(); // discard '('
        } else { // operator
            while (!op_stk.empty() && op_stk.top() != '('
                   && prec(c) <= prec(op_stk.top())) {
                char top = op_stk.top();
                res += top;
                op_stk.pop();
            }
            op_stk.push(c);
        }
    }
    // pop all remaining ops in the stack
    while (!op_stk.empty()) {
        char top = op_stk.top();
        res += top;
        op_stk.pop();
    }
    return res;
}

string get_postfix(const string str) {
    string res;
    cout << str << endl;
    res = substitute_plus(str);
    // cout << res << endl;
    res = insert_concat(res);
    cout << res << endl;
    res = infix2postfix(res);
    // cout << res << endl;
    return res;
}

set<char> get_input_symbol(const string str) {
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

/*
 * 3.7.4 Construction of an NFA from a Regular Expression
 * McNaughton-Yamada-Thompson algorithm
 *
 * Return:
 * pair<NFA_Node*,NFA_Node*>: start & accepting state of nfa
 * vector<NFA_Node*>: The built NFA
 *
 */
pair<int,int> regex2nfa(const string postfix_str,
                        vector<NFA_Node*>& nfa) {
    stack<NFA_Node*> nfa_stk;
    for (auto c : postfix_str) {
        if (isalpha(c)) { // a-z E
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
            nfa_stk.push(begin);
            nfa_stk.push(end);
        } else if (c == '|') { // union
            NFA_Node* begin = new NFA_Node();
            NFA_Node* end = new NFA_Node();
            nfa.push_back(begin);
            nfa.push_back(end);
            // N(t)
            NFA_Node* d = nfa_stk.top(); nfa_stk.pop();
            NFA_Node* c = nfa_stk.top(); nfa_stk.pop();
            // N(s)
            NFA_Node* b = nfa_stk.top(); nfa_stk.pop();
            NFA_Node* a = nfa_stk.top(); nfa_stk.pop();
            begin->e.push_back(a->id);
            begin->e.push_back(c->id);
            b->e.push_back(end->id);
            d->e.push_back(end->id);
            b->accepting = false;
            d->accepting = false;
            end->accepting = true;
            nfa_stk.push(begin);
            nfa_stk.push(end);
        } else if (c == '.') { // concatenation
            // N(t)
            NFA_Node* d = nfa_stk.top(); nfa_stk.pop();
            NFA_Node* c = nfa_stk.top(); nfa_stk.pop();
            // N(s)
            NFA_Node* b = nfa_stk.top(); nfa_stk.pop();
            NFA_Node* a = nfa_stk.top(); nfa_stk.pop();
            b->e.push_back(c->id);
            b->accepting = false;
            d->accepting = true;
            nfa_stk.push(a);
            nfa_stk.push(d);
        } else if (c == '*') { // Kleen closure
            NFA_Node* begin = new NFA_Node();
            NFA_Node* end = new NFA_Node();
            nfa.push_back(begin);
            nfa.push_back(end);
            NFA_Node* b = nfa_stk.top(); nfa_stk.pop();
            NFA_Node* a = nfa_stk.top(); nfa_stk.pop();
            b->e.push_back(a->id);
            begin->e.push_back(end->id);
            begin->e.push_back(a->id);
            b->e.push_back(end->id);
            b->accepting = false;
            end->accepting = true;
            nfa_stk.push(begin);
            nfa_stk.push(end);
        } else if (c == '?') { // zero or one, E|N(t)
            NFA_Node* begin = new NFA_Node();
            NFA_Node* end = new NFA_Node();
            nfa.push_back(begin);
            nfa.push_back(end);
            NFA_Node* b = nfa_stk.top(); nfa_stk.pop();
            NFA_Node* a = nfa_stk.top(); nfa_stk.pop();
            begin->e.push_back(a->id);
            begin->e.push_back(end->id);
            b->e.push_back(end->id);
            b->accepting = false;
            end->accepting = true;
            nfa_stk.push(begin);
            nfa_stk.push(end);
        } else if (c == '+') {
            // preprocess in input string
        }
    }
    // print_nfa(nfa);
    NFA_Node* end = nfa_stk.top(); nfa_stk.pop();
    NFA_Node* begin = nfa_stk.top(); nfa_stk.pop();
    pair<int,int> res(begin->id,end->id);
    return res;
}

// helper function for calculating e-closure of a state
void traverse_e(const NFA_Node* s, const vector<NFA_Node*>& nfa, set<int>& res) {
    for (auto neigh : s->e) {
        if (res.find(neigh) != res.end())
            break;
        res.insert(neigh);
        traverse_e(nfa[neigh], nfa, res);
    }
}

// e-closure of a state
set<int> epsilon_closure(const NFA_Node* s, const vector<NFA_Node*>& nfa) {
    set<int> res;
    res.insert(s->id);
    traverse_e(s,nfa,res);
    return res;
}

// e-closure of a set
set<int> epsilon_closure(const set<int>& T, const vector<NFA_Node*>& nfa) {
    set<int> res;
    for (auto s : T) {
        set<int> tmp = epsilon_closure(nfa[s],nfa);
        res.insert(tmp.begin(),tmp.end());
    }
    return res;
}

// helper function for calculating transition
void traverse(NFA_Node* s, const vector<NFA_Node*>& nfa,
              const char a, set<int>& res) {
    if (s->out.count(a) != 0)
        res.insert(s->out[a]);
}

// transition of a set
set<int> move_to(const set<int>& T, const vector<NFA_Node*>& nfa, const char a) {
    set<int> res;
    for (auto s : T)
        traverse(nfa[s],nfa,a,res);
    return res;
}

/*
 * 3.7.1 Conversion of an NFA to a DFA
 *
 * Return:
 * int: start state id of the DFA (only one entrance)
 * vector<DFA_Node*>: The built DFA
 *
 */
int nfa2dfa(const pair<int,int>& p,
            const vector<NFA_Node*>& nfa,
            const set<char>& input_symbol,
            vector<DFA_Node*>& dfa) {
    int start = p.first;
    int end = p.second;
    queue<set<int>> q;
    set<int> start_closure = epsilon_closure(nfa[start],nfa);
    q.push(start_closure);
    DFA_Node* node = new DFA_Node();
    dfa.push_back(node);
    set<set<int>> marked; // used to record visited states
    // loop up table
    // used to record mapping from NFA states to DFA
    // set of NFA states -> DFA id
    map<set<int>,int> lut;
    lut[q.front()] = 0;
    int start_id;
    if (start_closure.count(end) != 0) {
        node->accepting = true;
        node->group = 0;
    }
    if (start_closure.count(start) != 0) {
        node->start = true;
        start_id = dfa.size() - 1;
    }
    while (!q.empty()) {
        set<int> s = q.front();
        int idx = lut[s];
        for (auto sym : input_symbol) { // only alphas
            set<int> move = move_to(s,nfa,sym);
            if (move.empty())
                continue;
            set<int> U = epsilon_closure(move,nfa);
            if (marked.find(U) == marked.end()) { // U not in Dstates
                q.push(U);
                marked.insert(U);
                DFA_Node* node = new DFA_Node();
                dfa.push_back(node);
                lut[U] = dfa.size() - 1;
                // record the start & accepting states
                if (U.count(end) != 0) {
                    node->accepting = true;
                    node->group = 0;
                }
                if (U.count(start) != 0) {
                    node->start = true;
                    start_id = dfa.size() - 1;
                }
            }
            dfa[idx]->out[sym] = lut[U];
        }
        q.pop();
    }
    return start_id;
}

/*
 * 3.9.6 Minimizing the Number of States of a DFA
 * Hopcroft's algorithm
 *
 * Return:
 * int: start state id
 * vector<DFA_Node*>: Minimum DFA
 */
int minimize_dfa(vector<DFA_Node*>& dfa,
                 const set<char>& input_symbol,
                 vector<DFA_Node*>& min_dfa) {
    queue<vector<int>> partition;
    vector<int> s1, s2;
    set<int> group_id;
    for (auto state : dfa) {
        if (state->accepting)
            s1.push_back(state->id);
        else
            s2.push_back(state->id);
        group_id.insert(state->group);
    }
    int n_group = 2;
    partition.push(s2);
    partition.push(s1);
    map<int,int> state_map;
    // do partition
    while (!partition.empty()) {
        vector<int> p = partition.front();
        partition.pop();
        int size = p.size();
        // only one state, need not to be partitioned
        if (size < 2)
            continue;
        // more than 2 states
        int origin_group = dfa[p[0]]->group;
        for (auto c : input_symbol) {
            map<int,vector<int>> groups; // gid, idx in group
            for (int i = 0; i < size; ++i) {
                if (dfa[p[i]]->out.count(c) != 0) {
                    int out = dfa[p[i]]->out.at(c);
                    groups[dfa[out]->group].push_back(p[i]);
                } else {
                    // be careful of this case!
                    // no available transition in DFA!
                    groups[-1].push_back(p[i]);
                }
            }
            // if jump to different groups
            // can be partitioned
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
    // be careful that start and end may overlap
    vector<bool> start_flag(n_group,false);
    vector<bool> end_flag(n_group,false);
    map<int,int> group_map;
    // reallocate group id
    int cnt = 0;
    for (auto id : group_id) {
        group_map[id] = cnt;
        cnt++;
    }
    // remap group id
    int start_id;
    int len = dfa.size();
    for (int i = 0; i < len; ++i) {
        dfa[i]->group = group_map[dfa[i]->group];
        int group = dfa[i]->group;
        state_map[group] = i;
        if (dfa[i]->start) {
            start_flag[group] = true;
            start_id = group;
        }
        if (dfa[i]->accepting) {
            end_flag[group] = true;
        }
    }
    // generate new DFA
    n_group = group_id.size();
    for (int i = 0; i < n_group; ++i) {
        DFA_Node* node = new DFA_Node();
        for (auto c : input_symbol) {
            if (dfa[state_map[i]]->out.count(c) != 0) {
                int out = dfa[state_map[i]]->out.at(c);
                node->out[c] = dfa[out]->group;
            }
        }
        if (start_flag[i])
            node->start = true;
        if (end_flag[i])
            node->accepting = true;
        min_dfa.push_back(node);
    }
    return start_id;
}

int build_dfa(const string postfix_str,
              const set<char>& input_symbol,
              vector<DFA_Node*>& min_dfa) {
    NFA_Node::cnt = 0;
    vector<NFA_Node*> nfa;
    pair<int,int> p = regex2nfa(postfix_str,nfa);

    DFA_Node::cnt = 0;
    vector<DFA_Node*> dfa;
    int start_dfa = nfa2dfa(p,nfa,input_symbol,dfa);
    free_node<NFA_Node>(nfa);
    // print_dfa(dfa,input_symbol);
    // min_dfa = dfa;
    // return start_dfa;

    DFA_Node::cnt = 0;
    int start_mindfa = minimize_dfa(dfa,input_symbol,min_dfa);
    free_node<DFA_Node>(dfa);
    // print_dfa(min_dfa,input_symbol);
    return start_mindfa;
}

bool intersection(const int s1, const int s2,
                  const vector<DFA_Node*>& dfa1,
                  const vector<DFA_Node*>& dfa2,
                  const int len1, const int len2,
                  const set<char>& input_symbol,
                  vector<vector<bool>>& visited) {
    visited[s1][s2] = true;
    bool acc1 = (s1 == len1) ? false : dfa1[s1]->accepting;
    bool acc2 = (s2 == len2) ? true : dfa2[s2]->accepting;
    if (acc1 && acc2)
        return true;
    for (auto c : input_symbol) {
        // be careful of unavailable states of DFA
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

int judge(vector<DFA_Node*>& dfa1,
          vector<DFA_Node*>& dfa2,
          const set<char>& symbol1,
          const set<char>& symbol2,
          const int s1, const int s2) {
    // add a dummy state
    int len1 = dfa1.size();
    int len2 = dfa2.size();

    vector<vector<bool>> visited;
    for (int i = 0; i < len1+1; ++i) {
        vector<bool> tmp(len2+1,false);
        visited.push_back(tmp);
    }
    complement(dfa2);
    bool a_in_cb = intersection(s1,s2,dfa1,dfa2,len1,len2,symbol1,visited);

    visited.clear();
    for (int i = 0; i < len2+1; ++i) {
        vector<bool> tmp(len1+1,false);
        visited.push_back(tmp);
    }
    complement(dfa2);
    complement(dfa1);
    bool b_in_ca = intersection(s2,s1,dfa2,dfa1,len2,len1,symbol2,visited);

    if (!a_in_cb && !b_in_ca)
        return 0; // dfa1 = dfa2
    else if (!a_in_cb && b_in_ca)
        return 1; // dfa1 in dfa2
    else if (a_in_cb && !b_in_ca)
        return 2; // dfa2 in dfa1
    else
        return 3; // none
}

int NFA_Node::cnt = 0;
int DFA_Node::cnt = 0;

#ifdef NO_STDIN
const vector<vector<string>> input_str = {
    {"((E|a)b*)*", "(a|b)*"}, // =
    {"b*a*b?a*", "b*a*ba*|b*a*"}, // =
    {"b*a*b?a*", "(b*|a*)(b|E)a*"}, // >
    {"(c|d)*c(c|d)(c|d)", "(c|d)*d(c|d)(c|d)"}, // !
    {"x+y+z+", "x*y*z*"}, // <
    {"a", "a+"}, // <
    {"(a|b)*abb", "(a|b)*abbb*"}, // <
    {"(a|b)*c+(d|e)?", "(a|b)*cd"}, // >
    {"((x|y)+)+","(x|y)+(y+b*)*"} // <
};
#endif

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
        string postfix_str1 = get_postfix(str1);
        string postfix_str2 = get_postfix(str2);
        set<char> symbol1 = get_input_symbol(str1);
        set<char> symbol2 = get_input_symbol(str2);
        vector<DFA_Node*> dfa1;
        int s1 = build_dfa(postfix_str1,symbol1,dfa1);
        vector<DFA_Node*> dfa2;
        int s2 = build_dfa(postfix_str2,symbol2,dfa2);
        int res = judge(dfa1,dfa2,symbol1,symbol2,s1,s2);
        if (res == 0)
            cout << "=" << endl;
        else if (res == 1)
            cout << "<" << endl;
        else if (res == 2)
            cout << ">" << endl;
        else
            cout << "!" << endl;
        free_node<DFA_Node>(dfa1);
        free_node<DFA_Node>(dfa2);
    }
    return 0;
}