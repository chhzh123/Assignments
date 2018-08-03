// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is the headfile of Campus Card Management System made by Reddie, which contains implement of Campus_card class.

#ifndef CAMPUS_CARD_HPP
#define CAMPUS_CARD_HPP

#include "card.hpp"

class campus_card : virtual public card{ // 由于要进行菱形继承，故这里采用虚继承
public:
	// 构造函数
	campus_card() = default;
	// 标准构造函数
	// 由于校园卡不可透支，故overflow设为0
	campus_card(int _cardNum, std::string _name, std::string _school):
		card(_cardNum, _name, 0), school(_school){};
	// 方便文件流读入
	campus_card(int _cardNum, Date _releaseDate, std::string _name, std::string _school):
		card(_cardNum, _releaseDate, _name, 0), school(_school){};

	// 析构函数
	~campus_card() = default;

	// 具体实施存储、支付、查询功能
	bool store(double _money, const std::string _place);
	bool pay(double _money, const std::string _place);
	std::vector<std::string> queryInfo() const;
	std::vector<std::string> getAllInfo() const;

protected:
	// 由于大部分成员已经在基类里实现，故这里只实现校园卡特有的成员，即学院名
	std::string school;
};

bool campus_card::store(double _money, const std::string _place)
{
	money += _money;
	flow_record temp;
	temp.date.getNowDate();
	temp.place = _place;
	temp.moneyIO = _money;
	temp.moneyCurr = money;
	fr.push_back(temp);
	return true;
}

bool campus_card::pay(double _money, const std::string _place)
{
	if (testMoneyValid(money - _money))
	{
		money -= _money;
		flow_record temp;
		temp.date.getNowDate();
		temp.place = _place;
		temp.moneyIO = (-1)*_money;
		temp.moneyCurr = money;
		fr.push_back(temp);
		return true;
	}
	else
		return false;
}

std::vector<std::string> campus_card::queryInfo() const
{
	std::vector<std::string> res;
	res.push_back(std::to_string(cardNum));
	res.push_back(school);
	res.push_back(name);
	return res;
}

std::vector<std::string> campus_card::getAllInfo() const
{
	std::vector<std::string> res = card::getAllInfo();
	res.push_back(school);
	return res;
}

#endif // CAMPUS_CARD_HPP

// 校园卡 支付 查询
//	1.不可透支
//	2a.流水
//	2b.time 收到转账
//	2c. 学号 姓名 学院
//	3. 转账<-现金