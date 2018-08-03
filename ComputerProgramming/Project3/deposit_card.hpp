// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is the headfile of Campus Card Management System made by Reddie, which contains implement of Deposit_card class.

#ifndef DEPOSIT_CARD_HPP
#define DEPOSIT_CARD_HPP

#include "card.hpp"

// 预设储蓄卡透支金额，默认为1000元
#define OVERDRAW_LIMIT 1000

class deposit_card : virtual public card{ // 由于要进行菱形继承，故这里采用虚继承
public:
	// 构造函数
	deposit_card() = default;
	// 标准构造函数
	// 由于校园卡不可透支，故overflow设为0
	deposit_card(int _cardNum, std::string _name, double _overdraw = OVERDRAW_LIMIT):
		card(_cardNum, _name, _overdraw){};
	// 方便文件流读入
	deposit_card(int _cardNum, Date _releaseDate, std::string _name):
		card(_cardNum, _releaseDate, _name, 0){};

	// 析构函数
	~deposit_card() = default;

	// 具体实施存储、支付、查询功能
	bool store(double _money, const std::string _place);
	bool pay(double _money, const std::string _place);
	std::vector<std::string> queryInfo() const;

	// 转账功能
	// 由于转账可以转给三种不同的卡，故这里直接输入一个card类（泛型编程的思想）
	bool transfer(card& camp, double _money); // genetic

protected:
	// 由于基本信息已在基类中实施，故本类没有自己的成员
};

bool deposit_card::store(double _money, const std::string _place)
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

bool deposit_card::pay(double _money, const std::string _place)
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

bool deposit_card::transfer(card& camp, double _money)
{
	if (!testMoneyValid(money - _money))
		return false;
	this->pay(_money, "To_Card_#" + std::to_string(camp.getCardNum()));
	camp.store(_money, "From Depo #" + std::to_string(cardNum));
	return true;
}

std::vector<std::string> deposit_card::queryInfo() const
{
	std::vector<std::string> res;
	res.push_back(std::to_string(cardNum));
	res.push_back(releaseDate.toString());
	res.push_back(name);
	return res;
}

#endif // DEPOSIT_CARD_HPP

// 储蓄卡 支付 查询
//	1.支付可透支一定额度
//	2a.流水time place money
//	2b.time 收到转账
//	2c.本身信息 卡号 发卡日期 持卡人姓名
//	3. 转账->储蓄卡/校园卡 <-储蓄卡/现金
// 储蓄卡可以同名创建