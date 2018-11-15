//
//  main.cpp
//  GenealogyManagement
//
//  Copyright © 2018 曾天宇、陈鸿峥、黄杨峻. All rights reserved.
//

#include <iostream>
#include<stdlib.h>
#include "initialization.cpp"
#include "output.cpp"
using namespace std;

Family *myFamily;

void interface();
void showHelpInfo();

void interface(){
	cout<<"   Help                                     -h     (by file tree.in)"<<endl;
	cout<<"   Input / Refresh Info                     -r              "<<endl;
	cout<<"   Insert                                   -i              "<<endl;
	cout<<"   Summary                                  -S              "<<endl; 
	cout<<"   Search  Info                             -s     (by name)"<<endl;
	cout<<"   Delete  Info                             -d     (by name)"<<endl;
	cout<<"   Display Info                             -D"<<endl;
	cout<<"   Exit                                     -e"<<endl;
	cout<<">>> ";
	char op;
	string str;
	cin>>op;
	//system("clear");// Mac 下为 clear，Win 下为 cls
	switch (op){
		case ('h'):{
			showHelpInfo();
			break;}
		case ('r'):{
			myFamily = read("./tree.in");
			break;}
		case ('i'):{
			string name="",matename="",parentname="";
			int birthyear = 0;
			int age = 0;
			string gend="M";
			cout << "Please enter the name of the person: "<<endl;
			cin >> name;
			cout << "Please enter the gender of the person: (M)ale (F)emale"<<endl;
			cin >>gend;
			cout << "Please enter the age of the person" << endl;
			cin >> age;
			cout << "Please enter the birth year of the person" << endl;
			cin >> birthyear;
			cout << "Please enter the father's name of the person: "<<endl; 
			cin >> parentname;
			cout << "Please enter the mate's name of the person (If no mate just enter a 'null')"<<endl;
			cin >> matename;
			Person *p =  new Person(name,gend,birthyear,age);
			myFamily->insert_map(p);
			p -> setParent(parentname);
			myFamily->search(parentname)->addKid(p); 
			if (matename != "null"){
				Person *m = new Person(matename,(gend=="M")?"F":"M",birthyear,age);
				myFamily->insert_map(m); 
				p -> setMate(myFamily->search(matename));
				myFamily->insert_mate(name,matename);
			}
			//myFamily->insert_kid(parentname,name);
			cout << "Insert successfully!\n" << endl;
			break;}
		case ('S'):{
			cout <<"The total generation of your family is: "<<myFamily->generation()<<endl;
			cout <<"There are "<<myFamily->getNameMap().size()<<" person in your family.\n"<<endl;			
			break;
		}
		case ('s'):{
			cout << "Please enter the name of the person: ";
			cin >> str;
			 
			if(myFamily->search(str)){
				myFamily->search(str)->print();
			}
			break;}
		case ('d'):{
			cout << "Please enter the name of the person: ";
			string str2;
			cin >> str2;
			myFamily->delete_per(str2);
			cout << "Delete " << str2 << " successfully!\n" << endl;
			break;}
		case ('D'):{
			output(myFamily,"./tree.opml","./tree.csv");
			break;}
		case ('e'):
			exit(0);
		default:{
			cout<<"Invalid Operation!\n"<<endl;
			break;
		}
	}
	interface(); 
}

void showHelpInfo(){
	cout<<"To input / refresh the Info, you need to create a file named `tree.in`. Content of the file must follow the standard:"<<endl;
	cout<<"Starts with a few lines, recording information(Name, Age, Birth, gender) of all family members. ";
	cout<<"and the following lines shows the relations between each member.\n e.g.\n"<<endl; 
	cout<<"   HYJ [gender=\"M\", birthday=\"1998\", age=\"20\"]\n";
    cout<<"   DZA [gender=\"F\", birthday=\"2018\", age=\"0\"]\n";
    cout<<"   HYJ -> NVREN [label=\"mate\"]\n";
    cout<<"   HYJ -> DZA [label=\"kid\"]\n";
    cout<<endl;
    cout<<"To search someone or delete someone from the genealogy, you should input his/her name. \n";
    cout<<"The display function will show the Info in .csv form.\n "<<endl;
	cout<<endl;
	interface();
}

int main(int argc, const char * argv[]) {
	system("cls");
	cout<<"Welcome to the GenealogyManagement system.\n"<<endl;
	cout<<"Copyright (C) 2018 Tianyu Zeng, Hongzheng Chen, Yangjun Huang. All rights reserved.\n"<<endl;
	interface();
}
