//
//  Node.hpp
//  GenealogyManagement
//
//  Copyright © 2018 曾天宇、陈鸿峥、黄杨峻. All rights reserved.
//

#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <map>
#include <iostream>
using namespace std;

enum gendertype{Female, Male};
    
class Person{
	/*
		个人信息分别为出生年份、目前年龄（或享年）、性别、名字、伴侣、孩子
		其中孩子用一个指针vector信息域存储，伴侣用一个指针信息域存储
		构造函数将录入除伴侣和子嗣之外的个人信息
	*/
	private:
		int birth = 1900;
		int age = 0;
		int gen = 1;
		gendertype gender = Male;
		string name = "";
		string parent = "";
		Person* mate = nullptr;
		vector<Person*> kids;
		
	public:
		Person(string name, string gender, int birth, int age);	
		~Person(){
			birth = 0;
			age = 0;
		}
		void setGeneration(const int gen_num);
		void setGender(const gendertype gender);
		void setName(const string name);
		void setParent(const string parent);
		void setBirth(const int birth);
		void setAge(const int age);
		int getAge() const;
		int getBirth() const;
		Person*& Mate();
		string getMate() const;
		string getName() const;
		string getParent() const;
		int getGeneration() const;
		string getGender() const;
		void setMate(Person* const& mate)
		{
			this -> mate = mate;
		};
		void addKid(Person* const& kid)
		{
			kids.push_back(kid);
		};
		vector<Person*>& getKids_const()
		{ return kids; };
		vector<Person*>& getKids()
		{ return kids; };
		void print() const;
		void delete_kid(string name){
			int index = -1, status = 0;
			for(int i = 0; i<kids.size(); i++){
				if(kids.at(i)->getName() == name){
					status = 1;
					index = i;
				}
			}
			if(status){
				kids.erase(kids.begin()+index);
			}
		}
};

void Person::setBirth(const int birth){
	this->birth = birth;
}

void Person::setAge(const int age){
	this->age = age;
}

int Person::getAge() const{
	return age;
}

int Person::getBirth() const{
	return birth;
}

Person*& Person::Mate(){
	return mate;
}

string Person::getMate() const{
	if (mate == nullptr)
		return "";
	else
		return mate->getName();
}

string Person::getName() const{
	return name;
}

void Person::setName(const string name){
	this->name = name;
}

void Person::setParent(const string parent){
	this->parent = parent;
} 
string Person::getParent() const{
	return parent;
}

int Person::getGeneration() const{
	return gen;
}

void Person::setGeneration(const int gen_num){
	this->gen = gen_num;
}

string Person::getGender() const{
	return gender==Male?"Male":"Female";
}

void Person::setGender(const gendertype gender){
	this->gender = gender;
}

Person::Person(string name, string gender, int birth, int age){
	this->age = age;
	this->birth = birth;
	this->gender = gender == "M" ? Male : Female;
	this->name = name;
}

void Person::print() const{
	cout<<"Name: "<<name<<endl
	    <<"Gender: "<<(gender?"Male":"Female")<<endl
	    <<"Age: "<<age<<endl
	    <<"Birth Year: "<<birth<<endl;
	if(mate != nullptr){
		cout<<"Mate: "<<mate->getName()<<endl;
	}
	cout << "Generation: #" << getGeneration() << endl;
	cout << endl;
}

class Family{
	private:
		map<string, Person*> name_map;
		void visit(Person* const& ptr);
		int counter = 0;
		Person* root = nullptr;//老祖宗

	public:
		Family();
		Family(Person* root);
		void insert_mate(const string parent_name, const string mate_name);
		void insert_kid(const string parent_name, const string kid_name);//只记录男性成员的子女
		void insert_map(Person* root);
		void delete_per(const string p_name);//删除其中某个人,同时删除子树所有信息
		Person* search(const string name);//查找其中某个人
		int count() const;//输出家族总人数(含异性伴侣),
		int generation() const; //输出家族总代数
		//void print();//打印家谱
		Person* getRoot() const { return root; };
		map<string, Person*>& getNameMap(){
			return name_map;
		}
};


Family::Family(){
	root = nullptr;
	counter = 0;
}

Family::Family(Person* root){
	this->root = root;
	name_map.insert(make_pair(root->getName(), root));
	counter++;
}

void Family::insert_mate(const string parent_name, const string mate_name){
	Person* parent = search(parent_name);
	Person* mate = search(mate_name);
	if (mate != nullptr && parent != nullptr){
		parent->setMate(mate);
	}
}

void Family::insert_kid(const string parent_name, const string kid_name){
	Person* parent = search(parent_name);
	Person* kid = search(kid_name);
	if (kid != nullptr && parent != nullptr){
		kid->setGeneration(1+parent->getGeneration());
		kid->setParent(parent_name);
		parent->addKid(kid);
	}
}

void Family::insert_map(Person* root){
	if(root == this->root){
		root->setGeneration(1);
	}
	name_map.insert(make_pair(root->getName(), root));
	counter++;
}

int Family::count() const{
	// cout<<"People Count: "<<counter<<"\n";
	return counter;
}

int Family::generation() const{
	int gen = 0;
	for(auto x:name_map){
		if(gen<x.second->getGeneration()){
			gen = x.second->getGeneration();
		}
	}
	// cout<<"Generation Count: "<<gen<<"\n";
	return gen;
}

Person* Family::search(string name){
	Person* target_per = nullptr;
	for(auto x: name_map){
		if(x.first==name){
			// x.second->print();
			target_per=x.second;
		}else if(x.second->getGender()=="Male" && x.second->getMate()==name){
			target_per=x.second->Mate();
		}
	}
	if(target_per){
		return target_per;
	}else{
		cout<<"404 NOT FOUND!\n";
		return nullptr;
	}
}

void Family::visit(Person* const& ptr){
	if(ptr!=nullptr){
		string name = ptr->getName();
		bool flage = false;
		if(ptr->getGender() == "Male"){
			vector<Person*> kids = ptr->getKids();
			int num = kids.size();
			for(int i = 0; i < num; i++){
				visit(kids.front());
				kids.erase(kids.begin());
			}
		}
		name_map.erase(name_map.find(name));
		if(ptr->getGender() == "Male" && ptr->getMate() != ""){
			ptr->Mate()->~Person();
			name_map.erase(name_map.find(ptr->getMate()));
		}
		if(ptr!=root && ptr->getGender() == "Male"){
			search(ptr->getParent())->delete_kid(ptr->getName());
		}
		if(ptr==root){
			flage = true;
		}
		delete ptr;
		if(flage) {
			root = nullptr;
		}
	}
}

void Family::delete_per(string p_name){
	Person* node = search(p_name);
	visit(node);
}

#endif // NODE_HPP
