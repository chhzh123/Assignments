/*
Copyright 2018 Chen Hongzheng, Huang Yangjun, Zeng Tianyu

This is the implementation of train rearrangement system.
*/

#include <iostream>
#include <stack>
#include <vector>
using namespace std;

int main()
{
	// read in sequence
	int n;
	vector<int> trainLst;
	cin >> n;
	for (int i = 0; i < n; ++i)
	{
		int num;
		cin >> num;
		trainLst.push_back(num);
	}
	vector<stack<int>> station;

	// rearrangement
	int res = 0;
	for (auto train : trainLst)
	{
		// directly move out the train satisfies the order
		if (train == res + 1)
		{
			cout << "Train " << train << " move out" << endl;
			res++;
			while (true)
			{
				bool flag_in = false;
				for (auto psrcst = station.begin(); psrcst < station.end(); ++psrcst)
					while (!psrcst->empty() && psrcst->top() == res + 1)
					{
						cout << "Train " << psrcst->top() << " move out from stack " << (psrcst-station.begin()) << endl;
						psrcst->pop();
						res++;
						flag_in = true;
					}
				if (flag_in)
					continue;
				else
					break;
			}
			continue;
		}

		// push the current train into the previous stack
		int cnt = 0;
		bool flag = false;
		for (auto ptmpst = station.begin(); ptmpst < station.end(); ++ptmpst)
			if ((!ptmpst->empty() && train < ptmpst->top()) || ptmpst->empty())
			{
				ptmpst->push(train);
				cout << "Train " << train << " push into stack " << cnt << endl;
				flag = true;
				break;
			}
			else
				cnt++;
		if (!flag)
		{
			stack<int> tmpst;
			tmpst.push(train);
			cout << "Train " << train << " push into stack " << cnt << endl;
			station.push_back(tmpst);
		}
	}
	cout << "Total stacks used: " << station.size() << endl;
	return 0;
}