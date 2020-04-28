#include <iostream>
#include <cstdio>
#include <string>
using namespace std;

int main(){
	while (true){
		int a[55][50];
		int n, m;
		scanf("%d %d",&n,&m);
		if (n == 0 && m == 0)
			break;
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j)
				scanf("%d",&a[i][j]);
		int end[55], cnt_end;
		for (int i = 0; i < n+1; ++i){
			int num;
			scanf("%d",&num);
			if (num == -1){
				cnt_end = i;
				break;
			} else
				end[i] = num;
		}
		// dfa
		cin.ignore(); // ignore \n!
		string s;
		while (true){
			getline(cin,s);
			if (s[0] == '#')
				break;
			int curr = 0;
			for (auto c : s)
				curr = a[curr][c-'a'];
			bool flag = false;
			for (int i = 0; i < cnt_end; ++i)
				if (curr == end[i]){
					printf("YES\n");
					flag = true;
					break;
				}
			if (!flag)
				printf("NO\n");
		}
	}
	return 0;
}

// http://soj.acmm.club/show_problem.php?pid=1000&cid=2834
// Input:
// 4 2
// 1 0
// 1 2
// 1 3
// 1 0
// 3 -1
// aaabb
// abbab
// abbaaabb
// abbb
// #
// 1 3
// 0 0 0
// 0 -1
// cacba
// #
// 0 0

// Output:
// YES
// NO
// YES
// NO
// YES