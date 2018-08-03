// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is the headfile of Campus Card Management System made by Reddie, which contains implement of Binding_card class.

#ifndef BINDING_CARD_HPP
#define BINDING_CARD_HPP

#include "campus_card.hpp"
#include "deposit_card.hpp"

class binding_card : public campus_card, public deposit_card{ // 菱形继承
public:
	// 构造函数
	binding_card() = default;
	// 标准构造函数
	// 由于主要功能为校园卡，故与校园卡的构造相同
	binding_card(int _cardNum, std::string _name, std::string _school):
		campus_card(_cardNum, _name, _school){};
	// 方便文件流读入
	binding_card(int _cardNum, Date _releaseDate, std::string _name, std::string _school):
		campus_card(_cardNum, _releaseDate, _name, _school){};

	// 析构函数
	~binding_card() = default;

	// 具体实施存储、支付、查询功能
	bool store(double _money, const std::string _place);
	bool pay(double _money, const std::string _place);
	std::vector<std::string> queryInfo() const;
	std::vector<std::string> getAllInfo() const;

	// 添加绑定的储蓄卡
	bool appendDepo(deposit_card* depo);
	// 获得绑定的储蓄卡号
	std::vector<int> getBindRecord() const;
	
private:
	// 由于本类在卡仓库类之前实施，故只能存储绑定储蓄卡的指针
	// 而不能储存其在仓库内的位置，否则会造成交叉引用错误
	std::vector<deposit_card*> bindDepo;
};

bool binding_card::store(double _money, const std::string _place)
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

bool binding_card::appendDepo(deposit_card* depo)
{
	bindDepo.push_back(depo);
	return true;
}

std::vector<std::string> binding_card::queryInfo() const
{
	std::vector<std::string> res;
	res.push_back(std::to_string(cardNum));
	res.push_back(school);
	res.push_back(name);
	return res;
}

bool binding_card::pay(double _money, const std::string _place)
{
	if (testMoneyValid(money - _money))
		money -= _money;
	else
	{
		double sum = money + overdraw;
		for (auto depo = bindDepo.cbegin(); depo != bindDepo.cend(); ++depo)
		{
			sum += (*depo)->getAvailableMoney();
			if (sum >= _money)
				break;
			else if (depo + 1 == bindDepo.cend())
				return false;
		}
		_money -= money;
		std::string info;
		info = "Depo #"+std::to_string(cardNum);
		for (auto depo = bindDepo.cbegin(); depo != bindDepo.cend(); ++depo)
		{
			if (_money <= 0)
				break;
			double AMon = (*depo)->getAvailableMoney();
			if (AMon >= _money)
				(*depo)->pay(money,info);
			else
			{
				(*depo)->pay(AMon,info);
				_money -= AMon;
			}
		}
	}
	flow_record temp;
	temp.date.getNowDate();
	temp.place = _place;
	temp.moneyIO = (-1)*_money;
	temp.moneyCurr = money;
	fr.push_back(temp);
	return true;
}

std::vector<std::string> binding_card::getAllInfo() const
{
	std::vector<std::string> res = card::getAllInfo();
	res.push_back(school);
	return res;
}

std::vector<int> binding_card::getBindRecord() const
{
	std::vector<int> res;
	for (auto pdepo : bindDepo)
		res.push_back(pdepo->getCardNum());
	return res;
}

#endif // BINDING_cARD_HPP

//	4. 绑定一张或多张储蓄卡 依次透支