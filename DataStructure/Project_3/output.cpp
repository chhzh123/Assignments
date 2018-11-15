//
//  output.cpp
//  GenealogyManagement
//
//  Copyright © 2018 曾天宇、陈鸿峥、黄杨峻. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "node.hpp"
using namespace std;

/*
<family_tree>
    <node>
        <name>HYJ</name>
        <birthyear>1998</birthyear>
        <age>20</age>
        <kids>
            <node>
                <name>LTK</name>
                <other_info>emmm</other_info>
            </node>
        </kids>
        <mate>NVREN</mate>
    </node>
</family_tree>
*/

string aligned_tab(int num)
{
	string res = "";
	for (int n = 0; n < num; ++n)
		res += "   ";
	return res;
}

void outputOneNode(Person* const& node,int depth,ofstream& out,ofstream& out2)
{
	if (node == nullptr)
		return;
	cout<<node->getName()<<"\t"<<node->getGender()<<"\t"<<node->getAge()
		<<"\t"<<node->getBirth()<<"\t"<<node->getParent()<<"\t"<<node->getMate()
		<<"\t"<<node->getGeneration()<<endl;
	out2<<node->getName()<<","<<node->getGender()<<","<<node->getAge()
		<<","<<node->getBirth()<<","<<node->getParent()<<","<<node->getMate()
		<<","<<node->getGeneration()<<endl;
	//out << aligned_tab(depth) << "<node>" << endl;
	out << aligned_tab(depth) << "<outline text =\"person\" >"<< endl;
	//out << aligned_tab(depth+1) << "<name>" << node->getName() << "</name>" << endl;
	out << aligned_tab(depth+1) << "<outline text =\"name : "<< node->getName()<<"\" />" << endl;
	//out << aligned_tab(depth+1) << "<birth>" << node->getBirth() << "</birth>" << endl;
	out << aligned_tab(depth+1) <<"<outline text =\"birthyear : "<< node->getBirth() <<"\" />" <<endl;
	//out << aligned_tab(depth+1) << "<age>" << node->getAge() << "</age>" << endl;
	out << aligned_tab(depth+1) << "<outline text =\"age : "<< node->getAge() <<"\" />"<< endl;
	if (node->getMate() != "")
		//out << aligned_tab(depth+1) << "<mate>" << node->getMate() << "</mate>" << endl;
		out << aligned_tab(depth+1) << "<outline text =\"mate : " << node->getMate() <<"\" />" <<  endl;
	for (auto pnode = node->getKids_const().cbegin(); pnode != node->getKids_const().cend(); ++pnode)
	{
		//out << aligned_tab(depth+1) << "<kids>" << endl;
		out << aligned_tab(depth+1) << "<outline text =\"kid :\" >" << endl;
		outputOneNode(*pnode,depth+2,out,out2);
		//out << aligned_tab(depth+1) << "</kids>" << endl;
		out <<aligned_tab(depth+1)<<"</outline>"<<endl;
	}
	//out << aligned_tab(depth) << "</node>" << endl;
	out <<aligned_tab(depth)<<"</outline>"<<endl;
}

void output(Family* family,const string outfile_name1,const string outfile_name2)
{
	ofstream out(outfile_name1);
	ofstream out2(outfile_name2);
	cout << "Starting output..." << endl;
	cout << "Name\tGender\tAge\tBirth\tParent\tMate\tGeneration"<<endl;
 	out2 << "Name,Gender,Age,Birth,Parent,Mate,Generation"<<endl;
	//out << "<family_tree>" << endl;
	out<<"<?xml version=\"1.0\"?>\n<opml version=\"2.0\">\n   <head>\n      <ownerEmail>1023198294@qq.com</ownerEmail>\n   </head>\n   <body>"<<endl; 
	out<<"      <outline text = \"family_tree\" >\n"<<endl;
	//outputOneNode(family->getRoot(),1,out,out2);
	outputOneNode(family->getRoot(),3,out,out2);
	//out << "</family_tree>" << endl;
	out<<"      </outline>\n   </body>\n</opml>"<<endl;
	cout << "Finish output. All the data have been outputted to " << outfile_name1<<"  and  "<<outfile_name2<< " !" << endl;
	cout << endl;
	// cout << family->getNameMap().size()<<endl;
}
