// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is the headfile of Campus Card Management System made by Reddie, which contains implement of Card class and basic structure.

#ifndef CARD_HPP
#define CARD_HPP

#include <ctime>
#include <string>

struct Date
{
	int Year;
	int Month;
	int Day;
	void getNowDate()
	{
		time_t now = time(0);
		tm *ltm = localtime(&now);
		Year = 1900 + ltm->tm_year;
		Month = 1 + ltm->tm_mon;
		Day = ltm->tm_mday;
	}
	std::string toString() const
	{
		std::string resDate;
		resDate = std::to_string(Year) + "/"
			+ std::to_string(Month) + "/"
			+ std::to_string(Day);
		return resDate;
	}
	Date& operator=(const Date& other)
	{
		Year = other.Year;
		Month = other.Month;
		Day = other.Day;
		return *this;
	}
};

struct flow_record
{
	Date date;
	// 记录流水操作的地点
	// 如果是转账，则本条会显示对应的卡号；
	// 否则显示消费地点或取款地点
	std::string place;
	// 卡的金额流动
	// 如果值为正，即为存入或转入操作；
	// 如果值为负，即为提款或转出操作
	double moneyIO;
	// 存储当前越余额
	// 值为正，则卡内还有余额
	// 值为负，则卡为透支状态
	double moneyCurr;
};

class card{
public:
	// 构造函数
	card() = default;
	// 正常的构造函数
	card(int _cardNum, std::string _name, int _overdraw = 0):
		cardNum(_cardNum), name(_name), overdraw(_overdraw)
	{
		releaseDate.getNowDate();
	};
	// 为方便从文件流中读入数据，此处重新实现了一个构造函数，增添了直接读入日期的功能
	card(int _cardNum, Date _releaseDate, std::string _name, int _overdraw = 0):
		cardNum(_cardNum), releaseDate(_releaseDate), name(_name), overdraw(_overdraw){};

	//析构函数
	~card() = default;

	// 为子类提供存储、支付、查询接口
	virtual bool store(double _money, const std::string _place) = 0; // pure specifier
	virtual bool pay(double _money, const std::string _place) = 0;
	virtual std::vector<std::string> queryInfo() const = 0;
	// 由于流水信息结构一致，不需每个类单独实现，故本函数没有变为虚函数
	std::vector<flow_record> queryFlow() const;

	// 方便文件流输出信息
	virtual std::vector<std::string> getAllInfo() const;
	// 方便文件流读入历史流水
	inline void pushFlowRecord(flow_record& _fr);

	// 测试是否能从卡里继续提钱或支付
	// 主要看当前需要的金额是否大于透支金额与存储金额之和
	inline bool testMoneyValid(double money) const;
	// 获得当前可用的金额，包括可透支部分
	inline double getAvailableMoney() const;

	// 获得卡信息
	inline int getCardNum() const;
	inline std::string getName() const;

	// 判断两张卡的持有人是否一致
	inline bool operator==(const card& other) const;
	
protected:
	// 历史流水
	std::vector<flow_record> fr;
	// 基础卡信息
	int cardNum = 0;
	Date releaseDate;
	std::string name;
	// 当前余额，若为负，即透支
	double money = 0; // can be negative
	// 可透支金额，为常数，一旦确定不可修改
	// 预设可透支金额为0
	double overdraw = 0; // fixed number
};

bool card::testMoneyValid(double _money) const
{
	if (_money >= 0)
		return true;
	else
		if ((-1)*_money <= overdraw)
			return true;
		else
			false;
}

double card::getAvailableMoney() const
{
	return money + overdraw;
}

int card::getCardNum() const
{
	return cardNum;
}

std::string card::getName() const
{
	return name;
}

bool card::operator==(const card& other) const
{
	if (other.cardNum != cardNum)
		return false;
	else
		return true;
}

std::vector<flow_record> card::queryFlow() const
{
	return fr;
}

std::vector<std::string> card::getAllInfo() const
{
	std::vector<std::string> res;
	res.push_back(std::to_string(cardNum));
	res.push_back(std::to_string(releaseDate.Year));
	res.push_back(std::to_string(releaseDate.Month));
	res.push_back(std::to_string(releaseDate.Day));
	res.push_back(name);
	return res;
}

void card::pushFlowRecord(flow_record& _fr)
{ 
	fr.push_back(_fr);
}

#endif // CARD_HPP

// 0 depo 1 camp 2 binding

// 储蓄卡 支付 查询
//	1.支付可透支一定额度
//	2a.流水time place money
//	2b.time 收到转账
//	2c.本身信息 卡号 发卡日期 持卡人姓名
//	3. 转账->储蓄卡/校园卡 <-储蓄卡/现金
// 校园卡 支付 查询
//	1.不可透支
//	2a.流水
//	2b.time 收到转账
//	2c. 学号 姓名 学院
//	3. 转账<-现金
//	4. 绑定一张或多张储蓄卡 依次透支