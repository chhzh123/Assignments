// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is a polynomial calculator made by Reddie, which is able to do simple calculation such as addtion,
// subtraction, multiplication and derivative on polynomials. It also has other functional modules like management and storement.

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <regex>
#include <iterator>
#include "polynomial.hpp"
#include "polynomials.hpp"
using namespace std;

// Module
void managePolynomials(Polynomials_& PolyAll);
void calPolynomials(Polynomials_& PolyAll);
void readPolynomials(Polynomials_& PolyAll);
void savePolynomials(Polynomials_& PolyAll);
// Magagement
void appendPolynomials(Polynomials_& PolyAll);
void deletePolynomials(Polynomials_& PolyAll);
void showPolynomials(Polynomials_& PolyAll);
// Calculation
void addPolynomials(Polynomials_& PolyAll);
void subtractPolynomials(Polynomials_& PolyAll);
void multipyPolynomials(Polynomials_& PolyAll);
void derivativeOfPolynomials(Polynomials_& PolyAll);
void equalQPolynomials(Polynomials_& PolyAll);
void assignPolynomials(Polynomials_& PolyAll);
// Data process
void setPolynomial(Polynomial& polyin, string str);
void getNameOfPolynomials(Polynomials_& PolyAll);
// Deal with the input polynomials without names
int Polynomials_::cntVir = 0;
void appendVirtualPoly(Polynomials_& PolyAll, vector<string>& name);
void deleteVirtualPoly(Polynomials_& PolyAll);
// Ask whether to save the result of calculation
void askToSave(Polynomials_& PolyAll, vector<Polynomial>& vecPoly);
// Others
void show_main_manual();
void calculator_interface();
bool validname_test(const string& name);
vector<string> split(const string& input, const string& regex);

int main()
{
	calculator_interface();
	return 0;
}

void show_main_manual()
{
	cout << "Command List:\n" <<
		"\t1. Management\n" <<
		"\t2. Calculation\n" <<
		"\t3. Save\n" <<
		"\t4. Exit" << endl;
	cout << setw(8) << setiosflags(ios::left) << "Note:";
	cout << "1. To append, delete, show the polynomials, please enter \"1\"." << endl;
	cout << setw(8) << setiosflags(ios::left) << " ";
	cout << "2. To do calculations on existed polynomials, please enter \"2\"." << endl;
	cout << setw(8) << setiosflags(ios::left) << " ";
	cout << "3. Remember to save polynomials by entering \"3\" before exit." << endl;
	cout << endl;
}

void calculator_interface()
{
	Polynomials_ PolyAll;
	int flag = 1;
	readPolynomials(PolyAll);
	while(flag)
	{
		if (flag == 1)
		{
			system("cls");
			cout << setiosflags(ios::right) << setw(60) << "================================================" << endl << endl;
			cout << setiosflags(ios::right) << setw(60) << "-        Reddie's Polynomial Calculator !      -" << endl << endl;
			cout << setiosflags(ios::right) << setw(60) << "================================================" << endl;
			show_main_manual();
		}
		cout << "Please enter the number of operation. Enter \"4\" to quit." << endl;
		cin.clear();cin.sync();
		string str;
		do{
			if (str.size() > 1)
				cout << "Error: Undefined command. Please enter an integer between 1 to 4." << endl;
			cout << ">>> ";
			getline(cin,str);
		} while (str.size() == 0 || str.size() > 1);
		switch(str[0] - '0')
		{
			case 1: managePolynomials(PolyAll);
					flag = 1;
					break;
			case 2: calPolynomials(PolyAll);
					flag = 1;
					break;
			case 3: savePolynomials(PolyAll);
					flag = 2;
					break;
			case 4: return;
			default: cout << "Error: Undefined command. Please enter an integer between 1 to 4." << endl;
					flag = 2;
					break;
		}
	}
}

void managePolynomials(Polynomials_& PolyAll)
{
	int flag = 1;
	while(flag)
	{
		if (flag == 1)
		{
			system("cls");
			cout << "Command List:\n" <<
				"\t1. Append\n" <<
				"\t2. Delete\n" <<
				"\t3. Show all polynomials\n" <<
				"\t4. Back" << endl;
			cout << endl;
		}
		cout << "Please enter the number of operation. Enter \"4\" to return." << endl;
		cin.clear();cin.sync();
		string str;
		do{
			if (str.size() > 1)
				cout << "Error: Undefined command. Please enter an integer between 1 to 4." << endl;
			cout << ">>> ";
			getline(cin,str);
		} while (str.size() == 0 || str.size() > 1);
		switch(str[0] - '0')
			{
				case 1: appendPolynomials(PolyAll);flag = 1;break;
				case 2: deletePolynomials(PolyAll);flag = 1;break;
				case 3: showPolynomials(PolyAll);flag = 2;break;
				case 4: return;
				default: cout << "Error: Undefined command. Please enter an integer between 1 to 4." << endl;flag = 2;break;
			}
	}
}

void calPolynomials(Polynomials_& PolyAll)
{
	int flag = 1;
	while(flag)
	{
		if (flag == 1)
		{
			system("cls");
			cout << "Command List:\n" <<
				"\t1. Add\n" <<
				"\t2. Subtraction\n" <<
				"\t3. Multiplication\n" <<
				"\t4. Derivative\n" <<
				"\t5. EqualQ\n" <<
				"\t6. Assignment\n" <<
				"\t7. Back" << endl;
			cout << endl;
		}
		cout << "Please enter the number of operation. Enter \"7\" to return." << endl;
		cin.clear();cin.sync();
		string str;
		do{
			if (str.size() > 1)
				cout << "Error: Undefined command. Please enter an integer between 1 to 4." << endl;
			cout << ">>> ";
			getline(cin,str);
		} while (str.size() == 0 || str.size() > 1);
		switch(str[0] - '0')
			{
				case 1: addPolynomials(PolyAll);flag = 1;break;
				case 2: subtractPolynomials(PolyAll);flag = 1;break;
				case 3: multipyPolynomials(PolyAll);flag = 1;break;
				case 4: derivativeOfPolynomials(PolyAll);flag = 1;break;
				case 5: equalQPolynomials(PolyAll);flag = 1;break;
				case 6: assignPolynomials(PolyAll);flag = 1;break;
				case 7: return;
				default: cout << "Error: Undefined command. Please enter an integer between 1 to 7." << endl;flag = 2;break;
			}
	}
}

void readPolynomials(Polynomials_& PolyAll)
{
	ifstream infile("PolynomialsData.dat");
	if (infile)
	{
		while (!infile.eof())
		{
			string str;
			getline(infile,str);
			if(str.size() == 0) // empty file
			{
				cout << "There's no polynomials in the file." << endl; //Something wrong
				return;
			}
			vector<string> name_and_terms = split(str,"=");
			Polynomial temp;
			setPolynomial(temp,name_and_terms[1]);
			PolyAll.append(name_and_terms[0],temp);
		}
		cout << "Original data inputed." << endl;
	}
	else
		cout << "No original data." << endl;
}

void savePolynomials(Polynomials_& PolyAll)
{
	ofstream outfile("PolynomialsData.dat");
	map<string,Polynomial> data = PolyAll.getAllData();
	if (outfile)
		for(auto pdata = data.begin(); pdata != data.end(); ++pdata)
		{
			vector<int> temp = (pdata->second).getCoefficients();
			outfile << pdata->first << "=";
			int cnt = 0;
			for (auto ptemp = temp.begin(); ptemp != temp.end(); ++ptemp, ++cnt)
				outfile << "(" << *ptemp << "," << cnt << ")";
			if((++pdata) != data.end())
				outfile << endl;
			pdata--;
		}
	outfile.close();
	cout << "Save successfully!" << endl;
}

void setPolynomial(Polynomial& polyin, string str)
{
	vector<string> strPoly = split(str,"\\)\\(|\\)|\\(");
	if (strPoly.empty())
		return;
	strPoly.erase(strPoly.begin());
	vector<std::pair<int,int>> poly;
	for (auto pterm = strPoly.begin(); pterm != strPoly.end(); ++pterm)
	{
		vector<string> tempstr = split(*pterm,",");
		pair<int,int> temp(stoi(*(tempstr.begin())),stoi(*(tempstr.end()-1)));
		poly.push_back(temp);
	}
	polyin = Polynomial(poly);
}

void appendPolynomials(Polynomials_& PolyAll)
{
	cout << endl;
	cout << "Please enter the polynomials and their name, using \"=\" to assign. Do not add any space!" << endl
		<< "Names should only contain letters, hyphen or underline.\n"
		<< "The polynomials should be in form of ordered pairs, i.e. the first element is coefficient, and the second is the degree of the term." << endl
		<< "Notice the ordered pairs are not necessarily in decreasing order, and there's no space between two pairs." << endl
		<< "You can enter several of polynomials in one time. One line one polynomial. End appending by typing \"END\" in a new line." << endl
		<< "A valid example input is shown below: \n\n>>> polyA=(1,0)(2,5)(3,4) \n>>> polyB=(-3,5)(2,4)(2,2)(1,1)(0,0)\n>>> END\n" << endl;
	cin.clear();cin.sync(); // clear buffer
	int flag = 0;
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> name_and_terms = split(str,"=");
		if(!validname_test(name_and_terms[0]))
		{
			cout << "Error: Invalid name." << endl;
			continue;
		}
		Polynomial temp;
		setPolynomial(temp,name_and_terms[1]);
		cout << temp.PolyToString() << endl;
		if(!PolyAll.append(name_and_terms[0],temp))
		{
			cout << "Error: Some names conflict." << endl;
			continue;
		}
	}
	if (flag == 0)
		cout << "Append successfully." << endl;
	else
		cout << "Append Failed." << endl;
	cin.clear();cin.sync();
}

void deletePolynomials(Polynomials_& PolyAll)
{
	cout << endl;
	cout << "Please enter the polynomials and their name, seperating them by space." << endl
		<< "You can enter several of polynomials in one time. Also seperate them by space. Enter \"END\" to return." << endl;
	getNameOfPolynomials(PolyAll);
	cin.clear();cin.sync(); // clear buffer
	int flag = 0;
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> names = split(str," ");
		for (auto name : names)
			if (!PolyAll.erase(name))
			{
				cout << "Error: Invalid name." << endl;
				flag = 1;
				break;
			}
		break;
	}
	if (flag == 0)
		cout << "Delete successfully." << endl;
	else
		cout << "Delete Failed." << endl;
	cin.clear();cin.sync();
}

void showPolynomials(Polynomials_& PolyAll)
{
	int cnt = 0;
	map<string,Polynomial> data = PolyAll.getAllData();
	cout << endl;
	if(data.begin() != data.end())
	{
		cout << setiosflags(ios::left) << setw(22) << "  Name" << "  "  << setw(50) << "Terms" << endl;
		cout << setw(74) << setfill('-') << " " << endl;
		cout.fill(' ');
		for(auto pdata = data.begin(); pdata != data.end(); ++pdata)
		{
			cout << "| " << setiosflags(ios::left) << setw(20) << pdata->first << "| " << setw(50) << (pdata->second).PolyToString() << "|" << endl;
			cnt++;
		}
		cout << setw(74) << setfill('-') << " " << endl;
		cout.fill(' ');
		cout << "Total: " << cnt << " polynomials" << endl;
	}
	else
		cout << "Oops...There's no polynomials." << endl;
	cout << endl;
}

void getNameOfPolynomials(Polynomials_& PolyAll)
{
	vector<string> names(PolyAll.getNames());
	cout << endl;
	cout << "The names of polynomials which are stored are listed below:" << endl;
	for (int i = 0; i < names.size(); ++i)
	{
		cout << setiosflags(ios::right) << setw(15) << names[i];
		if ((i+1) % 4 == 0)
			cout << endl;
	}
	cout << endl;
}

void addPolynomials(Polynomials_& PolyAll)
{
	cout << endl;
	cout << "Please enter the names of polynomials or terms of unnamed polynomials you want to add.\n"
			"You can enter several cases. Every case a line. End by typing \"END\".\n" 
			"The examples listed below are all valid.\n"
			">>> polyA polyB polyC\n>>> polyA (1,0)(0,1)\n>>> (1,1)(2,3) (3,1)" << endl;
	getNameOfPolynomials(PolyAll);
	cin.clear();cin.sync(); // clear buffer
	string str;
	int flag = 0;
	vector<Polynomial> results;
	while (1)
	{
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> names = split(str," ");
		appendVirtualPoly(PolyAll,names);
		vector<Polynomial> polys = PolyAll.getPolys(names);
		if (polys.empty())
		{
			cout << "Error: Match no polynomials." << endl;
			continue;
		}
		if (polys.size() < names.size())
			cout << "Warning: Some names match no polynomials. Sum of the rest is shown below." << endl;
		Polynomial resAdd = PolyAll.add(polys);
		results.push_back(resAdd);
		cout << resAdd.PolyToString() << endl;
		deleteVirtualPoly(PolyAll);
	}
	askToSave(PolyAll,results);
	cin.clear();cin.sync();
}

void multipyPolynomials(Polynomials_& PolyAll)
{
	cout << endl;
	cout << "Please enter the names of polynomials or terms of unnamed polynomials you want to multipy. \n"
			" You can enter several cases. Every case a line. End by typing \"END\"."
			"The examples listed below are all valid.\n"
			">>> polyA polyB polyC\n>>> polyA (1,0)(0,1)\n>>> (1,1)(2,3) (3,1)" << endl;
	getNameOfPolynomials(PolyAll);
	cin.clear();cin.sync(); // clear buffer
	string str;
	int flag = 0;
	vector<Polynomial> results;
	while (1)
	{
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> names = split(str," ");
		appendVirtualPoly(PolyAll,names);
		vector<Polynomial> polys = PolyAll.getPolys(names);
		if (polys.empty())
		{
			cout << "Error: Match no polynomials." << endl;
			continue;
		}
		if (polys.size() < names.size())
			cout << "Warning: Some names match no polynomials. Product of the rest is shown below." << endl;
		Polynomial resProduct = PolyAll.multipy(polys);
		results.push_back(resProduct);
		cout << resProduct.PolyToString() << endl;
		deleteVirtualPoly(PolyAll);
	}
	askToSave(PolyAll,results);
	cin.clear();cin.sync();
}

void subtractPolynomials(Polynomials_& PolyAll)
{
	cout << endl;
	cout << "Please enter the names or terms of TWO polynomials you want to subtract. \n"
			"You can enter several cases. Every case a line. End by typing \"END\"."
			"The examples listed below are all valid.\n"
			">>> polyA polyB\n>>> polyA (1,0)(0,1)\n>>> (1,1)(2,3) (3,1)" << endl;
	getNameOfPolynomials(PolyAll);
	cin.clear();cin.sync(); // clear buffer
	string str;
	int flag = 0;
	vector<Polynomial> results;
	while (1)
	{
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> names = split(str," ");
		appendVirtualPoly(PolyAll,names);
		vector<Polynomial> polys = PolyAll.getPolys(names);
		switch (polys.size())
		{
			case 0: cout << "Error: Match no polynomials." << endl; continue;
			case 1: cout << "Error: Only one valid polynomial." << endl; continue;
			case 2: if (polys.size() < names.size())
						cout << "Warning: Entered more than two polynomials and some names match no polynomials." << endl;
					break;
			default: if (polys.size() < names.size())
						cout << "Warning: Entered more than two polynomials and some names match no polynomials.\n"
								"Only the difference of the first two valid polynomials will be calculated." << endl;
					else
						cout << "Warning: Entered more than two polynomials. "
								"Only the difference of the first two valid polynomials will be calculated." << endl;
					break;
		}
		Polynomial resSub = PolyAll.subtract(polys[0],polys[1]);
		results.push_back(resSub);
		cout << resSub.PolyToString() << endl;
		deleteVirtualPoly(PolyAll);
	}
	askToSave(PolyAll,results);
	cin.clear();cin.sync();
}

void derivativeOfPolynomials(Polynomials_& PolyAll)
{
	cout << endl;
	cout << "Please enter the name or terms of ONE polynomial you want to differentiate. \n"
			"You can enter several cases. Every case a line. End by typing \"END\"."
			"The examples listed below are all valid.\n"
			">>> polyA\n>>> (1,0)(0,1)" << endl;
	getNameOfPolynomials(PolyAll);
	cin.clear();cin.sync(); // clear buffer
	string str;
	int flag = 0;
	vector<Polynomial> results;
	while (1)
	{
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> names = split(str," ");
		appendVirtualPoly(PolyAll,names);
		vector<Polynomial> polys = PolyAll.getPolys(names);
		switch (polys.size())
		{
			case 0: cout << "Error: Match no polynomials." << endl; continue;
			case 1: if (polys.size() < names.size())
						cout << "Warning: Entered more than two polynomials and some names match no polynomials." << endl;
					break;
			default: if (polys.size() < names.size())
						cout << "Warning: Entered more than two polynomials and some names match no polynomials.\n"
								"Only the first valid polynomial will be calculated." << endl;
					else
						cout << "Warning: Entered more than two polynomials. "
								"Only the first valid polynomial will be calculated." << endl;
					break;
		}
		Polynomial resDiff = PolyAll.derivative(polys[0]);
		results.push_back(resDiff);
		cout << resDiff.PolyToString() << endl;
		deleteVirtualPoly(PolyAll);
	}
	askToSave(PolyAll,results);
	cin.clear();cin.sync();
}

void assignPolynomials(Polynomials_& PolyAll)
{
	cout << endl;
	cout << "Please enter the name of the polynomial and the value you want to assigned, seperating them by space. \n"
			"You can enter several cases. Every case a line. End by typing \"END\"." << endl;
	getNameOfPolynomials(PolyAll);
	cin.clear();cin.sync(); // clear buffer
	int flag = 0;
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> name_and_value = split(str," ");
		if(!PolyAll.findName(name_and_value[0]))
		{
			cout << "Error: Match no polynomials." << endl;
			flag = 1;
			break;
		}
		vector<string> simu;
		simu.push_back(name_and_value[0]);
		vector<Polynomial> temp = PolyAll.getPolys(simu);;
		cout << PolyAll.assign(temp[0],stoi(name_and_value[1])) << endl;
	}
	if (flag == 0)
		cout << "Assign successfully." << endl;
	else
		cout << "Assign Failed." << endl;
	cin.clear();cin.sync();
}

void equalQPolynomials(Polynomials_& PolyAll)
{
	cout << endl;
	cout << "Please enter the names of polynomials or terms of unnamed polynomials you want to test if they are equal."
			" You can enter several cases. Every case a line. End by typing \"END\"."
			"The examples listed below are all valid.\n"
			">>> polyA polyB polyC\n>>> polyA (1,0)(0,1)\n>>> (1,1)(2,3) (3,1)" << endl;
	getNameOfPolynomials(PolyAll);
	cin.clear();cin.sync(); // clear buffer
	string str;
	int flag = 0;
	while (1)
	{
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> names = split(str," ");
		appendVirtualPoly(PolyAll,names);
		vector<Polynomial> polys = PolyAll.getPolys(names);
		if (polys.empty())
		{
			cout << "Error: Match no polynomials." << endl;
			continue;
		}
		if (polys.size() < names.size())
			cout << "Warning: Some names match no polynomials. The result of the rest is shown below." << endl;
		bool res = PolyAll.equalQ(polys);
		if (res)
			cout << "True!" << endl;
		else
			cout << "False!" << endl;
		deleteVirtualPoly(PolyAll);
	}
	cin.clear();cin.sync();
}

void askToSave(Polynomials_& PolyAll, vector<Polynomial>& vecPoly)
{
	cout << endl;
	cout << "Do you want to save the results? (Y/N)" << endl;
	cin.clear();cin.sync(); // clear buffer
	int flag = 0;
	string str;
	do{
		cout << ">>> ";
		getline(cin,str);
	} while (str.size() == 0 || (str != "Y" && str != "N"));
	if (str == "N")
		return;
	cout << "Enter number of the output line and name to save, seperating them by space.\n" 
			"Notice the name should only contain letters, hyphen or underline.\n"
			"You can enter several cases. Every case a line. End by typing \"END\".\n" << endl;
	cout << "A valid example is shown below:\n"
			">>> polyA polyB\n3x^2+4\n"
			">>> polyA\n2x^2\n"
			">>> END\n>>> Y\n>>> 2 quadratic\n"
			"2x^2\n>>> END\n" << endl;
	cin.clear();cin.sync(); // clear buffer
	while (1)
	{
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		if (str.substr(0,3) == "END")
			break;
		vector<string> num_and_names = split(str," ");
		if (stoi(num_and_names[0]) > vecPoly.size())
		{
			cout << "Error: Number of line exceeds." << endl;
			continue;
		}
		if(!validname_test(num_and_names[1]))
		{
			cout << "Error: Invalid name." << endl;
			continue;
		}
		if(!PolyAll.append(num_and_names[1],vecPoly[stoi(num_and_names[0])-1]))
		{
			cout << "Error: Some names conflict." << endl;
			flag = 1;
			continue;
		}
		else
			cout << num_and_names[1] << "=" << vecPoly[stoi(num_and_names[0])-1].PolyToString() << " saved!" <<endl;
	}
	cout << "Save successfully." << endl;
	cin.clear();cin.sync();
}

void appendVirtualPoly(Polynomials_& PolyAll, vector<string>& names)
{
	for (auto &name : names)
		if (name[0] == '(')
		{
			Polynomial temp;
			setPolynomial(temp,name);
			Polynomials_::cntVir++;
			name = to_string(Polynomials_::cntVir);
			PolyAll.append(name,temp);
		}
}

void deleteVirtualPoly(Polynomials_& PolyAll)
{
	for (int i = 1; i < Polynomials_::cntVir + 1; ++i)
		PolyAll.erase(to_string(i));
}

bool validname_test(const string& name)
{
	for(int i = 0; i < name.length(); ++i)
		if(!(isalpha(name[i]) || name[i]=='-' || name[i]=='_'))
			return false;
	return true;
}

vector<string> split(const string& input, const string& regex) // split the string
{
	// passing -1 as the submatch index parameter performs splitting
	try{
		std::regex re(regex);
		std::sregex_token_iterator
			first{input.begin(), input.end(), re, -1},
			last;
		return {first, last};
	}
	catch (...)
	{
		cout << "Unknown Error!!!" << endl;
		vector<string> temp;
		return temp;
	}
}