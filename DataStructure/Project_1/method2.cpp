#include <iostream>
#include <stack>
#include <vector>
#include<cstdlib>
using namespace std;

vector<stack<int>> stk;
int outmin = 1;

void goTo(int temp, bool dest, int num){
    if (dest) {
        cout<<"Train "<<temp<<" go to dest. "<<endl;
        outmin += 1;
    }else{
        cout<<"Train "<<temp<<" go to stack "<<num+1<<". "<<endl;
    }
}

void moveToStack(int temp, bool createNew){
    int index = 0;
    int min = 999999;
    if (createNew) {
        bool flag1 = false, flag2 = false;
        for (int i = 0; i<stk.size(); i++) {
            if (stk.at(i).empty()) {
                index = i;
                flag1 = true;
            }
            else if (stk.at(i).top()>temp) {
                if (min<stk.at(i).top()) {
                    //todo
                }else{
                    min = stk.at(i).top();
                    index = i;
                    flag2 = true;
                }
            }
        }
        if (flag1&&!flag2) {
            stk.at(index).push(temp);
        }else if (flag2){
            stk.at(index).push(temp);
        }
        else{
            stack<int> sk;
            sk.push(temp);
            stk.push_back(sk);
            index = (int)stk.size()-1;
        }
    }else{
        
        index = 0;
        for (int i = 0; i<stk.size(); i++) {
            if (stk.at(i).top()>temp) {
                if (min<stk.at(i).top()) {
                    //todo
                }else{
                    min = stk.at(i).top();
                    index = i;
                }
            }
        }
        stk.at(index).push(temp);
    }
    goTo(temp, false, index);
    //cout<<stk.size()<<endl;
}

int main(){
    int count;
    cout<<"Input the number of total_train"<<endl;
    cin>>count;
    int series[count], delta[count];
    cout<<"Input the sequence of start "<<endl;
    for (int i = 0; i<count; i++) {
        cin>>series[count-1-i];
    }
    cout<<endl;
    for (int i = 0; i<count-1; i++) {
        delta[i] = series[i+1]-series[i];
    }
    cout<<endl;
    for (int i = 0; i<count; i++) {
        for (int k = 0; k<stk.size(); k++) {
            if (stk.at(k).empty()) {
                continue;
            }
            if (stk.at(k).top()==outmin) {
                goTo(stk.at(k).top(), true, 0);
                stk.at(k).pop();
                if (!stk.at(k).empty()) {
                    k--;
                }
            }
        }
        if(series[i] == 1 || series[i] == outmin){
            //go to dest.
            goTo(series[i], true, 0);
            continue;
        }
        if (delta[i-1]<0) {
            moveToStack(series[i], i==0);
        }else{
            moveToStack(series[i], true);
        }
        for (int k = 0; k<stk.size(); k++) {
            if (stk.at(k).empty()) {
                stk.erase(stk.begin()+k);
            }
        }
        //cout<<"**"<<outmin<<"**\n";
    }
    for (int i = outmin; i<=count; i++) {
        for (int k = 0; k<stk.size(); k++) {
            if (stk.at(k).empty()) {
                continue;
            }
            if (stk.at(k).top() == i) {
                goTo(stk.at(k).top(), true, 0);
                stk.at(k).pop();
                break;
            }
        }
    }
    cout<<endl<<"Use stack count "<<stk.size()<<endl;
    return 0;
}