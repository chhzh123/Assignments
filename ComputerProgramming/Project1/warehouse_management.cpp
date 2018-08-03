// Chen Hongzheng(Reddie) 2018.3

#include <iostream>
#include <string>
#include <map>
#include <fstream> //文件流
#include <iomanip> //输出格式
using namespace std;

#define MAXINT 100000000 //防止数据溢出

class Warehouse
{
public:
	Warehouse(){};
	bool import_goods(string name, int count);
	bool export_goods(string name, int count);
	int find_goods(string name);
	void show_goods();
	void save();
private:
	map<string,int> data;
	bool increase_count(string name, int count);
	bool add_to_list(string name, int count);
	bool decrease_count(string name, int count);
	bool delete_from_list(string name);
};

void import_process(Warehouse& house); //封装进货、出货、询问过程
void export_process(Warehouse& house);
void query_process(Warehouse& house);
void warehouse_interface();
void read_in_data(Warehouse& house); //从文件中读入数据
void show_manual();
bool validname_test(string x);

int main()
{
	warehouse_interface();
	return 0;
}

void warehouse_interface()
{
	string x;
	Warehouse house;
	int flag = 1;
	system("cls");
	cout << setw(60) << "================================================" << endl << endl;
	cout << setw(60) << "-    Reddie's Repository Management System!    -" << endl << endl;
	cout << setw(60) << "================================================" << endl;
	read_in_data(house);
	show_manual();
	while(flag)
	{
		cout << "Please enter the number of operation. Entering \"5\" can show the manual.\n>>> ";
		cin.clear();cin.sync();
		getline(cin,x);
		if(x.length() > 1 || x=="\n") //防止读入错误信息或单一换行符
			cout << "Error: Undefined command. Please enter an integer between 1 to 5." << endl;
		else switch(x[0] - '0')
			{
				case 1:import_process(house);break;
				case 2:export_process(house);break;
				case 3:query_process(house);break;
				case 4:house.save();cout << "Auto saved." << endl;flag = 0;break;
				case 5:show_manual();break;
				default: cout << "Error: Undefined command. Please enter an integer between 1 to 5." << endl;break;
			}
	}
}

bool Warehouse::increase_count(string name, int count)
{
	if(data[name]+count < MAXINT)
		data[name] += count;
	else
		return false;
	return true;
}

bool Warehouse::add_to_list(string name, int count)
{
	if(count < MAXINT)
		data[name] = count;
	else
		return false;
	return true;
}

bool Warehouse::import_goods(string name, int count)
{
	if (count <= 0)
		return false;
	if (find_goods(name))
		return increase_count(name,count);
	else
		return add_to_list(name,count);
}

void import_process(Warehouse& house)
{
	string name;
	int number;
	cout << "Please enter the name and number of the goods you want to import. Seperate them by space.\n>>> ";
	cin.clear();cin.sync(); //清空缓冲区
	cin >> name >> number;
	if(!validname_test(name))
		cout << "Error: Invalid name." << endl;
	else if(!house.import_goods(name,number))
			cout << "Error: Invalid number." << endl;
		else
			cout << "Import successfully." << endl;
	cin.clear();cin.sync();
}

bool Warehouse::decrease_count(string name, int count)
{
	data[name] -= count;
	return true;
}

bool Warehouse::delete_from_list(string name)
{
	data.erase(data.find(name));
	return true;
}

bool Warehouse::export_goods(string name, int count)
{
	if(data[name] < count)
		return false;
	else
		if(data[name] == count)
			return delete_from_list(name);
		else
			return decrease_count(name,count);
}

void export_process(Warehouse& house)
{
	string name;
	int number;
	cout << "Please enter the name and number of the goods you want to export. Seperate them by space.\n>>> ";
	cin.clear();cin.sync();
	cin >> name >> number;
	if (!house.find_goods(name))
		cout << "Error: There's no \"" << name << "\" in the warehouse." << endl;
	else
		if (number <= 0 || number > MAXINT)
			cout << "Error: Invalid number." << endl;
		else if (!house.export_goods(name,number))
				cout << "Error: The inventory is not enough to export." << endl;
			else
				cout << "Export successfully." << endl;
	cin.clear();cin.sync();
}

int Warehouse::find_goods(string name)
{
	map<string,int>::iterator find_it;
	find_it = data.find(name);
	if (find_it != data.end())
		return find_it->second;
	else
		return 0;
}

void Warehouse::show_goods()
{
	int cnt = 0;
	cout << endl;
	if(data.begin() != data.end())
	{
		cout << setw(22) << "  name" << "  " << setw(10) << "amount" << endl;
		cout << setw(34) << setfill('-') << " " << endl;
		cout.fill(' ');
		for(map<string,int>::iterator find_it = data.begin(); find_it != data.end(); find_it++)
		{
			cout << "| " << setw(20) << find_it->first << "| " << setw(10) << find_it->second << "|" << endl;
			cnt++;
		}
		cout << setw(34) << setfill('-') << " " << endl;
		cout.fill(' ');
		cout << "Total: " << cnt << " goods" << endl;
	}
	else
		cout << "Oops...There's no inventory." << endl;
	cout << endl;
}

void query_process(Warehouse& house)
{
	string name;
	cout << "Please enter the name of the goods you want to query. Entering \"Show\" will show all the goods' information.\n>>> ";
	cin >> name;
	if(name == "Show")
		house.show_goods();
	else
		cout << setw(20) << name << setw(10) << house.find_goods(name) << endl;
}

void show_manual()
{
	cout << "Command List:\n" <<
		"\t1. Import\n" <<
		"\t2. Export\n" <<
		"\t3. Query\n" <<
		"\t4. Exit" << endl;
	cout << setw(8) << setiosflags(ios::left) << "Note:";
	cout << "1. A name should only contain English letters, hyphens and underscores." << endl;
	cout << setw(8) << setiosflags(ios::left) << " ";
	cout << "2. The amount of importing and exporting must be non-negative integers and cannot exceed MAXINT." << endl;
	cout << setw(8) << setiosflags(ios::left) << " ";
	cout << "3. You must exit this program by entering \"4\", or the data will be lost." << endl;
	cout << endl;
}

void read_in_data(Warehouse& house)
{
	ifstream infile("GoodsData.dat");
	string name;
	int number;
	if (infile)
	{
		while (!infile.eof())
		{
			infile >> name >> number;
			if(!validname_test(name)) //空文件
			{
				cout << "There's no inventory." << endl; //Something wrong
				return;
			}
			house.import_goods(name,number);
		}
		cout << "Original data inputed." << endl;
	}
	else
		cout << "No original data." << endl;
}

void Warehouse::save()
{
	ofstream outfile("GoodsData.dat");
	map<string,int>::iterator find_it;
	if (outfile)
		for(find_it = data.begin(); find_it != data.end(); find_it++){
			outfile << find_it->first << " " << find_it->second;
			if((++find_it) != data.end())
				outfile << endl;
			find_it--;
		}
	outfile.close();
}

bool validname_test(string x)
{
	for(int i=0;i<x.length();i++)
		if(!(isdigit(x[i]) || x[i]=='-' || x[i]=='_' || (x[i]>='a'&&x[i]<='z') || (x[i]>='A'&&x[i]<='Z')))
			return false;
	return true;
}