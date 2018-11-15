//
//  initialization.cpp
//  GenealogyManagement
//
//  Copyright © 2018 曾天宇、陈鸿峥、黄杨峻. All rights reserved.
//

#include <iostream>
#include <fstream> 
#include <string>
#include <regex>
#include "node.hpp"
using namespace std;

/*
digraph genealogy{
    HYJ [gender="M", birth="1998", age="20", mate="NVREN"];
    LTK [gender="M"];
    DZA [gender="F"];
    HYJ -> NVREN [label="mate"];
    HYJ -> LTK [label="kid"];
    HYJ -> DZA [label="kid"];
}
*/

std::vector<std::string> split(const std::string& input, const std::string& regex) // split the string
{
	// passing -1 as the submatch index parameter performs splitting
	std::regex re(regex);
	std::sregex_token_iterator
		first{ input.begin(), input.end(), re, -1 },
		last;
	return { first, last };
}

Family* read(const string infile_name)
{
	ifstream infile(infile_name);
	if (!infile)
	{
		cout << "Error: No such files!" << endl;
		return nullptr;
	}
	string str;
	// The first lines in dot file are useless info
	getline(infile,str);
	cout << "Begin parsing..." << endl;
	// nodes info
	getline(infile,str);
	vector<string> info = split(str," *\\[ *gender *= *\" *|\" *\\];| +|\", *birth *= *\"|\", age *= *\""); // reg exp
	// the first is space
	Person* root = new Person(info[1],info[2],stoi(info[3]),stoi(info[4]));
	Family* family = new Family(root);
	while (getline(infile,str) && str.find("-") == std::string::npos)
	{
		vector<string> info = split(str," *\\[ *gender *= *\" *|\" *\\];| +|\", *birth *= *\"|\", age *= *\""); // reg exp
		Person* man = new Person(info[1],info[2],stoi(info[3]),stoi(info[4]));
		family->insert_map(man);
	}
	// edges info
	do {
		vector<string> arc = split(str," *\\[ *label *= *\"|\" *\\];| *-> *| +"); // reg exp
		if (arc[3] == "mate")
			family->insert_mate(arc[1],arc[2]);
		else
			family->insert_kid(arc[1],arc[2]);
	} while (getline(infile,str) && str.size() > 1);
	cout << "Parsed dot file successfully!" << endl;
	cout << endl;
	return family;
}
