// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is Campus Card Management System made by Reddie, which is able to manage campus cards and deposit cards.

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <regex>
#include <iterator>

#include "card_repository.hpp"

using namespace std;

// Module
void manageCards(card_repository& repo);
void tranCards(card_repository& repo);
void queryCards(card_repository& repo);
void readAllCards(card_repository& repo);
void saveAllCards(card_repository& repo);
// Management
void appendCampusCard(card_repository& repo);
void appendDepositCard(card_repository& repo);
void appendBindingCard(card_repository& repo);
void deleteCard(card_repository& repo);
// Transaction
void withdrawMoney(card_repository& repo);
void depositMoney(card_repository& repo);
void transferMoney(card_repository& repo);
// Query
void queryCardInfo(card_repository& repo);
void queryFlowRecord(card_repository& repo);
// Others
void show_main_manual();
void interface();
vector<string> split(const string& input, const string& regex);
bool validNameTest(const string& name);
bool vecLenTest(vector<string> vec,int len);

int main()
{
	interface();
	return 0;
}

void show_main_manual()
{
	cout << "Command List:\n" <<
		"\t1. Card management\n" <<
		"\t2. Transaction\n" <<
		"\t3. Query\n" <<
		"\t4. Exit" << endl;
	cout << setw(8) << setiosflags(ios::left) << "Note:";
	cout << "1. To append, delete a card, please enter \"1\"." << endl;
	cout << setw(8) << setiosflags(ios::left) << " ";
	cout << "2. To withdraw, transfer, deposit money, please enter \"2\"." << endl;
	cout << setw(8) << setiosflags(ios::left) << " ";
	cout << "3. To query card information and flow record, please enter \"3\"." << endl;
	cout << setw(8) << setiosflags(ios::left) << " ";
	cout << "4. To save records, you SHOULD exit by entering \"4\"." << endl;
	cout << endl;
}

void interface()
{
	card_repository repo;
	int flag = 1;
	//readAllCards(repo);
	while(flag)
	{
		if (flag == 1)
		{
			system("cls");
			cout << setiosflags(ios::right) << setw(60) << "================================================" << endl << endl;
			cout << setiosflags(ios::right) << setw(60) << "-    Reddie's Campus Card Management System!   -" << endl << endl;
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
			case 1: manageCards(repo);
					flag = 1;
					break;
			case 2: tranCards(repo);
					flag = 1;
					break;
			case 3: queryCards(repo);
					flag = 1;
					break;
			case 4: saveAllCards(repo);
					return;
			default: cout << "Error: Undefined command. Please enter an integer between 1 to 4." << endl;
					flag = 2;
					break;
		}
	}
}

void manageCards(card_repository& repo)
{
	int flag = 1;
	while(flag)
	{
		if (flag == 1)
		{
			system("cls");
			cout << "Command List:\n" <<
				"\t1. Create a new campus card\n" <<
				"\t2. Create a new deposit card\n" <<
				"\t3. Create a new binding card\n" <<
				"\t4. Cancel a card\n" <<
				"\t5. Back" << endl;
			cout << endl;
		}
		cout << "Please enter the number of operation. Enter \"5\" to return." << endl;
		cin.clear();cin.sync();
		string str;
		do{
			if (str.size() > 1)
				cout << "Error: Undefined command. Please enter an integer between 1 to 5." << endl;
			cout << ">>> ";
			getline(cin,str);
		} while (str.size() == 0 || str.size() > 1);
		switch(str[0] - '0')
			{
				case 1: appendCampusCard(repo);flag = 2;break;
				case 2: appendDepositCard(repo);flag = 2;break;
				case 3: appendBindingCard(repo);flag = 2;break;
				case 4: deleteCard(repo);flag = 2;break;
				case 5: return;
				default: cout << "Error: Undefined command. Please enter an integer between 1 to 5." << endl;flag = 2;break;
			}
	}
}

void tranCards(card_repository& repo)
{
	int flag = 1;
	while(flag)
	{
		if (flag == 1)
		{
			system("cls");
			cout << "Command List:\n" <<
				"\t1. Transfer\n" <<
				"\t2. Withdraw\n" <<
				"\t3. Deposit\n" <<
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
				case 1: transferMoney(repo);flag = 2;break;
				case 2: withdrawMoney(repo);flag = 2;break;
				case 3: depositMoney(repo);flag = 2;break;
				case 4: return;
				default: cout << "Error: Undefined command. Please enter an integer between 1 to 4." << endl;flag = 2;break;
			}
	}
}

void queryCards(card_repository& repo)
{
	int flag = 1;
	while(flag)
	{
		if (flag == 1)
		{
			system("cls");
			cout << "Command List:\n" <<
				"\t1. Card info\n" <<
				"\t2. Flow record\n" <<
				"\t3. Back" << endl;
			cout << endl;
		}
		cout << "Please enter the number of operation. Enter \"3\" to return." << endl;
		cin.clear();cin.sync();
		string str;
		do{
			if (str.size() > 1)
				cout << "Error: Undefined command. Please enter an integer between 1 to 3." << endl;
			cout << ">>> ";
			getline(cin,str);
		} while (str.size() == 0 || str.size() > 1);
		switch(str[0] - '0')
			{
				case 1: queryCardInfo(repo);flag = 2;break;
				case 2: queryFlowRecord(repo);flag = 2;break;
				case 3: return;
				default: cout << "Error: Undefined command. Please enter an integer between 1 to 3." << endl;flag = 2;break;
			}
	}
}

void readFlows(card& _card,ifstream& infile)
{
	int numData;
	infile >> numData;
	string str;
	for (int j = 0; j < numData; ++j)
	{
		getline(infile,str);
		vector<string> temp = split(str," ");
		Date dateTemp;
		dateTemp.Year = stoi(temp[0]);
		dateTemp.Month = stoi(temp[1]);
		dateTemp.Day = stoi(temp[2]);
		flow_record fr;
		fr.date = dateTemp;
		fr.place = temp[3];
		fr.moneyIO = stoi(temp[4]);
		fr.moneyCurr = stoi(temp[5]);
		_card.pushFlowRecord(fr);
	}
}

void readCampusCards(card_repository& repo, ifstream& infile)
{
	int numCard;
	infile >> numCard;
	for (int i = 0; i < numCard; ++i)
	{
		string str;
		getline(infile,str);
		vector<string> temp = split(str," ");
		Date dateTemp;
		dateTemp.Year = stoi(temp[1]);
		dateTemp.Month = stoi(temp[2]);
		dateTemp.Day = stoi(temp[3]);
		campus_card cc(stoi(temp[0]),dateTemp,temp[4],temp[5]);
		readFlows(cc,infile);
		repo.appendCampusCard(cc);
	}
}

void readDepositCards(card_repository& repo, ifstream& infile)
{
	int numCard;
	infile >> numCard;
	for (int i = 0; i < numCard; ++i)
	{
		string str;
		getline(infile,str);
		vector<string> temp = split(str," ");
		Date dateTemp;
		dateTemp.Year = stoi(temp[1]);
		dateTemp.Month = stoi(temp[2]);
		dateTemp.Day = stoi(temp[3]);
		deposit_card dc(stoi(temp[0]),dateTemp,temp[4]);
		readFlows(dc,infile);
		repo.appendDepositCard(dc);
	}
}

void readBindingCards(card_repository& repo, ifstream& infile)
{
	int numCard;
	infile >> numCard;
	for (int i = 0; i < numCard; ++i)
	{
		string str;
		getline(infile,str);
		vector<string> temp = split(str," ");
		Date dateTemp;
		dateTemp.Year = stoi(temp[1]);
		dateTemp.Month = stoi(temp[2]);
		dateTemp.Day = stoi(temp[3]);
		binding_card bc(stoi(temp[0]),dateTemp,temp[4],temp[5]);
		readFlows(bc,infile);
		int numBind;
		infile >> numBind;
		vector<int> tempBind;
		for (int j = 0; j < numBind; ++j)
		{
			int numb;
			infile >> numb;
			tempBind.push_back(numb);
		}
		repo.appendBindingCard(bc,tempBind);
	}
}

void readAllCards(card_repository& repo)
{
	ifstream infiled("Deposit_card.dat");
	ifstream infilec("Campus_card.dat");
	ifstream infileb("Binding_card.dat");
	if (infiled)
	{
		readDepositCards(repo,infiled); // use of deleted function?
		infiled.close();
		cout << "Read deposit cards successfully!" << endl;
	}
	else
		cout << "No origin deposit cards data." << endl;
	if (infilec)
	{
		readCampusCards(repo,infilec);
		infilec.close();
		cout << "Read campus cards successfully!" << endl;
	}
	else
		cout << "No origin campus cards data." << endl;
	if (infileb)
	{
		readBindingCards(repo,infileb);
		infileb.close();
		cout << "Read binding cards successfully!" << endl;
	}
	else
		cout << "No origin binding cards data." << endl;
	cout << "Read all cards successfully!" << endl;
}

template <class Data>
void saveCards(Data& data, ofstream& outfile)
{
	outfile << data.size() << endl;
	for(auto pdata = data.begin(); pdata != data.end(); ++pdata)
	{
		vector<string> temp = pdata->getAllInfo();
		for (auto ptemp = temp.begin(); ptemp != temp.end(); ++ptemp)
			{ outfile << *ptemp << " ";}
		outfile << endl;
		vector<flow_record> fr = pdata->queryFlow();
		outfile << fr.size() << endl;
		for (auto ptemp = fr.begin(); ptemp != fr.end(); ++ptemp)
			outfile << (ptemp->date).Year << " " << (ptemp->date).Month << " " << (ptemp->date).Day << " "
					<< ptemp->place << " " << ptemp->moneyIO << " " << ptemp->moneyCurr << endl;
	}
}

void saveBindCards(vector<binding_card>& data, ofstream& outfile)
{
	outfile << data.size() << endl;
	for(auto pdata = data.begin(); pdata != data.end(); ++pdata)
	{
		vector<string> temp = pdata->getAllInfo();
		for (auto ptemp = temp.begin(); ptemp != temp.end(); ++ptemp)
			{ outfile << *ptemp << " ";}
		outfile << endl;
		vector<flow_record> fr = pdata->queryFlow();
		outfile << fr.size() << endl;
		for (auto ptemp = fr.begin(); ptemp != fr.end(); ++ptemp)
			outfile << (ptemp->date).Year << " " << (ptemp->date).Month << " " << (ptemp->date).Day << " "
					<< ptemp->place << " " << ptemp->moneyIO << " " << ptemp->moneyCurr << endl;
		vector<int> bindNum = pdata->getBindRecord();
		outfile << bindNum.size() << endl;
		for (auto ptemp = bindNum.begin(); ptemp != bindNum.end(); ++ptemp)
			outfile << (*ptemp) << " ";
		outfile << endl;
	}
}

void saveAllCards(card_repository& repo)
{
	ofstream outfiled("Deposit_card.dat");
	ofstream outfilec("Campus_card.dat");
	ofstream outfileb("Binding_card.dat");
	vector<deposit_card> datad = repo.getDepoCard();
	if (outfiled)
		saveCards(datad,outfiled); // use of deleted function?
	outfiled.close();
	cout << "Save deposit cards successfully!" << endl;
	vector<campus_card> datac = repo.getCampCard();
	if (outfilec)
		saveCards(datac,outfilec);
	outfilec.close();
	cout << "Save campus cards successfully!" << endl;
	vector<binding_card> datab = repo.getBindCard();
	if (outfileb)
		saveBindCards(datab,outfileb);
	outfileb.close();
	cout << "Save binding cards successfully!" << endl;
	cout << "Save all cards successfully!" << endl;
}

void appendCampusCard(card_repository& repo)
{
	cout << endl;
	cout << "Please enter your name and school, sperating them by space." << endl
		<< "Note: Your name should not contain space, which means you need to use caption to distinguish your first name and family name." << endl
		<< "So is school name. You can only contain letters, hyphen or underline." << endl;
	cin.clear();cin.sync(); // clear buffer
	int flag = 0;
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		vector<string> name_and_school = split(str," +");
		if(!validNameTest(name_and_school[0]))
		{
			cout << "Error: Invalid name." << endl;
			flag = 1;
			break;
		}
		if(!validNameTest(name_and_school[1]))
		{
			cout << "Error: Invalid school name." << endl;
			flag = 1;
			break;
		}
		if(repo.findName(name_and_school[0]))
		{
			cout << "Error: Some names conflict." << endl;
			flag = 1;
			break;
		}
		repo.appendCampusCard(name_and_school[0],name_and_school[1]);
		break;
	}
	if (flag == 0)
		cout << "Append successfully. Your card number is #" << repo.getTotal() << "." << endl;
	else
		cout << "Append Failed." << endl;
	cin.clear();cin.sync();
}

void appendDepositCard(card_repository& repo)
{
	cout << endl;
	cout << "Please enter your name, which should not contain space, meaning you need to use caption to distinguish your first name and family name." << endl
		<< "Note: Default overdraw limit is 1000 yuan. You can change it by calling the administrator." << endl;
	cin.clear();cin.sync(); // clear buffer
	int flag = 0;
	while (1)
	{
		string name;
		cout << ">>> ";
		getline(cin,name);
		if (name.size() == 0)
			continue;
		if(!validNameTest(name))
		{
			cout << "Error: Invalid name." << endl;
			flag = 1;
			break;
		}
		repo.appendDepositCard(name);
		break;
	}
	if (flag == 0)
		cout << "Append successfully. Your card number is #" << repo.getTotal() << "." << endl;
	else
		cout << "Append Failed." << endl;
	cin.clear();cin.sync();
}

void appendBindingCard(card_repository& repo)
{
	cout << endl;
	cout << "Please enter your name and school, sperating them by space." << endl
		<< "Note: Your name should not contain space, which means you need to use caption to distinguish your first name and family name." << endl
		<< "So is school name. You can only contain letters, hyphen or underline." << endl;
	cin.clear();cin.sync(); // clear buffer
	int flag = 0;
	vector<string> name_and_school;
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		name_and_school = split(str," +");
		if(!validNameTest(name_and_school[0]))
		{
			cout << "Error: Invalid name." << endl;
			flag = 1;
			break;
		}
		if(!validNameTest(name_and_school[1]))
		{
			cout << "Error: Invalid school name." << endl;
			flag = 1;
			break;
		}
		if(repo.findName(name_and_school[0]))
		{
			cout << "Error: Some names conflict." << endl;
			flag = 1;
			break;
		}
		break;
	}
	if (flag == 0)
		cout << "Append successfully. Your card number is #" << repo.getTotal() << "." << endl;
	else
	{
		cout << "Append Failed." << endl;
		return;
	}
	cout << endl;
	cout << "Now Please enter the numbers of your deposit cards you want to bind, sperating them by space." << endl
		<< "If you don't want to bind now, please enter 0." << endl;
	cin.clear();cin.sync(); // clear buffer
	vector<int> numb;
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		vector<string> numbstr = split(str," +");
		for (auto num : numbstr)
			numb.push_back(stoi(num));
		break;
	}
	repo.appendBindingCard(name_and_school[0],name_and_school[1],numb);
	cout << "Done!" << endl;
	cin.clear();cin.sync();
}

void deleteCard(card_repository& repo)
{
	cout << endl;
	cout << "Please enter the numbers of the cards you want to delete, sperating them by space." << endl
		<< "Be careful of this operation, all the money in the card will lost. Enter 0 to return." << endl;
	cin.clear();cin.sync(); // clear buffer
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		vector<string> numbs = split(str," +");
		for (auto pnum = numbs.cbegin(); pnum != numbs.cend(); ++pnum)
		{
			if (std::stoi(*pnum) == 0)
				break;
			if (repo.deleteCard(std::stoi(*pnum)))
				cout << "Card #" << *pnum << " has been deleted!" << endl;
			else
				cout << "Error: No card #" << *pnum << " !" << endl;
		}
		break;
	}
}

void transferMoney(card_repository& repo)
{
	cout << endl;
	cout << "Please enter your card number, the number of the card you want to transfer money to, and amount of money, sperating them by space." << endl;
	cin.clear();cin.sync(); // clear buffer
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		vector<string> numbs = split(str," +");
		if (!vecLenTest(numbs,3))
		{
			cout << "Error: You should enter three numbers." << endl;
			continue;
		}
		if (repo.transferMoney(std::stoi(numbs[0]),std::stoi(numbs[1]),std::stof(numbs[2]))) // stof!!!
			cout << "Error: Card num wrong or money is not enough!" << endl;
		else
			cout << "Transfer successfully!" << endl;
		break;
	}
}

void withdrawMoney(card_repository& repo)
{
	cout << endl;
	cout << "Please enter your card number and amount of money you want to withdraw, sperating them by space." << endl;
	cin.clear();cin.sync(); // clear buffer
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		vector<string> numbs = split(str," +");
		if (!vecLenTest(numbs,2))
		{
			cout << "Error: You should enter two numbers." << endl;
			continue;
		}
		if (repo.withdrawMoney(std::stoi(numbs[0]),std::stof(numbs[1]))) // stof!!!
			cout << "Error: Card num wrong or money is not enough!" << endl;
		else
			cout << "Withdraw successfully!" << endl;
		break;
	}
}

void depositMoney(card_repository& repo)
{
	cout << endl;
	cout << "Please enter your card number and amount of money you want to deposit, sperating them by space." << endl;
	cin.clear();cin.sync(); // clear buffer
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		vector<string> numbs = split(str," +");
		if (!vecLenTest(numbs,2))
		{
			cout << "Error: You should enter two numbers." << endl;
			continue;
		}
		if (repo.depositMoney(std::stoi(numbs[0]),std::stof(numbs[1]))) // stof!!!
			cout << "Error: Card num wrong!" << endl;
		else
			cout << "Deposit successfully!" << endl;
		break;
	}
}

template <class T>
void outputFlowRecord(T data)
{
	if(data.cbegin() != data.cend())
	{
		cout << setiosflags(ios::left) << setw(22) << "  Date" << "  "  << setw(20) << "Place" 
			<< setw(20) << "MoneyIO" << setw(20) << "Current Money" << endl;
		cout << setw(74) << setfill('-') << " " << endl;
		cout.fill(' ');
		int cnt = 0;
		for(auto pdata = data.cbegin(); pdata != data.cend(); ++pdata)
		{
			cout << "| " << setiosflags(ios::left) << setw(20) << (pdata->date).toString() << "| " 
				<< setw(20) << setprecision(2) << pdata->place << "|"
				<< setw(20) << setprecision(2) << pdata->moneyIO << "|"
				<< setw(20) << setprecision(2) << pdata->moneyCurr << "|" << endl;
			cnt++;
		}
		cout << setw(74) << setfill('-') << " " << endl;
		cout.fill(' ');
		cout << "Total: " << cnt << " flow records." << endl;
	}
	else
		cout << "Oops...There's no flow record." << endl;
	cout << endl;
}

void queryFlowRecord(card_repository& repo)
{
	cout << endl;
	cout << "Please enter your card number." << endl;
	cin.clear();cin.sync(); // clear buffer
	int num;
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		num = stoi(str);
		break;
	}
	switch (repo.QType(num))
	{
		case 1:
		outputFlowRecord((repo.findCamp(num))->queryFlow());
		return;
		case 2:
		outputFlowRecord((repo.findDepo(num))->queryFlow());
		return;
		case 3:
		outputFlowRecord((repo.findBind(num))->queryFlow());
		return;
		case 0:
		cout << "Error: Card num wrong!" << endl;
		return;
	}
}

void queryCardInfo(card_repository& repo)
{
	cout << endl;
	cout << "Please enter your card number." << endl;
	cin.clear();cin.sync(); // clear buffer
	int num;
	while (1)
	{
		string str;
		cout << ">>> ";
		getline(cin,str);
		if (str.size() == 0)
			continue;
		num = stoi(str);
		break;
	}
	switch (repo.QType(num))
	{
		case 1:{
		// genetic programming + lambda expression
		vector<string> temp = (repo.findCamp(num))->queryInfo();
		for_each(temp.begin(),temp.end(),[](const string &s) { cout << s << " ";});
		cout << endl;
		return;}
		case 2:{
		vector<string> temp = (repo.findDepo(num))->queryInfo();
		for_each(temp.begin(),temp.end(),[](const string &s) { cout << s << " ";});
		cout << endl;
		return;}
		case 3:{
		vector<string> temp = (repo.findBind(num))->queryInfo();
		for_each(temp.begin(),temp.end(),[](const string &s) { cout << s << " ";});
		cout << endl;
		return;}
		case 0:
		cout << "Error: Card num wrong!" << endl;
		return;
	}
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

bool validNameTest(const string& name)
{
	for(int i = 0; i < name.length(); ++i)
		if(!(isalpha(name[i]) || name[i]=='-' || name[i]=='_'))
			return false;
	return true;
}

bool vecLenTest(vector<string> vec,int len)
{
	if (vec.size() < len)
		return false;
	else
		return true;
}