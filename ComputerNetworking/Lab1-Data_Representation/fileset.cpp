// Chen Hongzheng 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 1: Data Representation - File copy

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
using namespace std;

#define FILE_NAME "FileSet.pak"

int main()
{
	ofstream output(FILE_NAME);
	cout << "----- USER INPUT -----" << endl;
	while (true){
		cout << "Enter the file name: ";
		string filename;
		getline(cin,filename);
		if (filename == "")
			break;

		FILE *p_file = fopen(filename.c_str(),"rb");
		fseek(p_file,0,SEEK_END);
		long size = ftell(p_file);
		fclose(p_file);

		output << filename << endl;
		output << size << endl;
		// output << "##### FILE BEGIN #####" << endl;
		ifstream input(filename);
		string str;
		while (getline(input,str))
			output << str << endl;
		// output << "##### FILE OVER #####" << endl;
	}
	cout << "----- USER INPUT OVER -----\n" << endl;

	// read in file
	ifstream input(FILE_NAME);
	cout << "----- OUTPUT FILES -----" << endl;
	for (int i = 1; true ; ++i){
		string str, filename;
		if (!getline(input,filename))
			break;
		getline(input,str); // size
		long exp_size = stol(str);
		// getline(input,str); // ##### FILE BEGIN #####
		ofstream output(to_string(i)+"_"+filename);
		while (getline(input,str)){
			// if (str.find("#####") != string::npos){ // ##### FILE OVER #####
			output << str << endl;
			FILE *p_file = fopen((to_string(i)+"_"+filename).c_str(),"rb");
			fseek(p_file,0,SEEK_END);
			long size = ftell(p_file);
			fclose(p_file);
			if (size >= exp_size){
				break;
			}
		}
		cout << "Generated file " << i << ": " << filename << endl;
	}
	cout << "----- OUTPUT FILES END -----" << endl;
	return 0;
}