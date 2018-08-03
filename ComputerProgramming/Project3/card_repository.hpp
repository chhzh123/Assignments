// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is the headfile of Campus Card Management System made by Reddie, which contains implement of Card_repository class.

#ifndef CARD_REPOSITORY_HPP
#define CARD_REPOSITORY_HPP

#include "binding_card.hpp"

class card_repository{
public:
	// 构造函数与析构函数
	card_repository() = default;
	~card_repository() = default;

	// 添加卡
	void appendDepositCard(const std::string name);
	void appendCampusCard(const std::string name, const std::string school);
	void appendBindingCard(const std::string name, const std::string school, std::vector<int> cardli);
	// 方便文件流读入，先在主函数中构建好卡信息后再读入仓库
	void appendDepositCard(deposit_card& dc);
	void appendCampusCard(campus_card& cc);
	void appendBindingCard(binding_card& bc, std::vector<int> cardli);

	// 转账、提取、存储
	bool transferMoney(int numFrom, int numTo, double money);
	bool withdrawMoney(int numCard, double money);
	bool depositMoney(int numCard, double money);

	// 删除卡
	bool deleteCard(int num);

	// 在仓库中查询卡
	bool findName(const std::string name);
	int QType(int num);
	campus_card* findCamp(int num);
	deposit_card* findDepo(int num);
	binding_card* findBind(int num);
	inline int getTotal() const { return totalNum; };
	inline std::vector<deposit_card> getDepoCard() const { return depoCard; };
	inline std::vector<campus_card> getCampCard() const { return campCard; };
	inline std::vector<binding_card> getBindCard() const { return bindCard; };

private:
	// 存储三类卡的全部信息
	std::vector<deposit_card> depoCard;
	std::vector<campus_card> campCard;
	std::vector<binding_card> bindCard;
	// 存储卡的类型
	// 由卡号映射到{1,2,3}
	// 其中，1为校园卡，2为储蓄卡，3为绑定卡
	std::map<int,int> cardType;
	// 开卡时的卡号，每次开一张卡，总数目就加1
	// 但注意删除卡不会减1，以防冲突
	int totalNum = 0;
};

int card_repository::QType(int num)
{
	if (cardType.find(num) != cardType.end())
		return cardType[num];
	else
		return 0;
}

void card_repository::appendDepositCard(const std::string name)
{
	totalNum++;
	deposit_card dc(totalNum,name);
	depoCard.push_back(dc);
	cardType[totalNum] = 2;
}

void card_repository::appendDepositCard(deposit_card& dc)
{
	totalNum++;
	depoCard.push_back(dc);
	cardType[dc.getCardNum()] = 2;
}

void card_repository::appendCampusCard(const std::string name, const std::string school)
{
	totalNum++;
	campus_card cc(totalNum,name,school);
	campCard.push_back(cc);
	cardType[totalNum] = 1;
}

void card_repository::appendCampusCard(campus_card& cc)
{
	totalNum++;
	campCard.push_back(cc);
	cardType[cc.getCardNum()] = 1;
}

void card_repository::appendBindingCard(const std::string name, const std::string school, std::vector<int> cardli)
{
	totalNum++;
	binding_card bc(totalNum,name,school);
	for (auto cardnum : cardli)
	{
		deposit_card* tempDepo = findDepo(cardnum);
		if (tempDepo != nullptr)
			bc.appendDepo(tempDepo);
	}
	bindCard.push_back(bc);
	cardType[totalNum] = 3;
}

void card_repository::appendBindingCard(binding_card& bc, std::vector<int> cardli)
{
	totalNum++;
	for (auto cardnum : cardli)
	{
		deposit_card* tempDepo = findDepo(cardnum);
		if (tempDepo != nullptr)
			bc.appendDepo(tempDepo);
	}
	bindCard.push_back(bc);
	cardType[bc.getCardNum()] = 3;
}

bool card_repository::deleteCard(int num)
{
	switch (QType(num))
	{
		case 1:
		for (auto camp = campCard.cbegin(); camp != campCard.cend(); ++camp)
			if (camp->getCardNum() == num)
			{
				campCard.erase(camp);
				return true;
			}
		case 2:
		for (auto depo = depoCard.cbegin(); depo != depoCard.cend(); ++depo)
			if (depo->getCardNum() == num)
			{
				depoCard.erase(depo);
				return true;
			}
		case 3:
		for (auto bind = bindCard.cbegin(); bind != bindCard.cend(); ++bind)
		if (bind->getCardNum() == num)
		{
			bindCard.erase(bind);
			return true;
		}
		default:
		return false;
	}
}

bool card_repository::transferMoney(int numFrom, int numTo, double money)
{
	deposit_card* depo = findDepo(numFrom);
	if (depo == nullptr || QType(numFrom) != 2)
		return false;
	switch (QType(numTo))
	{
		case 1:{
		campus_card* camp = findCamp(numTo);
		if (camp == nullptr)
			return false;
		return depo->transfer(*camp,money);}
		case 2:{
		deposit_card* depoc = findDepo(numTo);
		if (depoc == nullptr)
			return false;
		return depo->transfer(*depoc,money);}
		case 3:{
		binding_card* bind = findBind(numTo);
		if (bind == nullptr)
			return false;
		return depo->transfer(*bind,money);}
		case 0:
		return false;
	}
}

bool card_repository::withdrawMoney(int numCard, double money)
{
	deposit_card* depo = findDepo(numCard);
	if (depo == nullptr)
		return false;
	if (depo->pay(money,"Withdraw"))
		return true;
	else
		return false; 
}

bool card_repository::depositMoney(int numCard, double money)
{
	deposit_card* depo = findDepo(numCard);
	if (depo == nullptr)
		return false;
	if (depo->store(money,"Store"))
		return true;
	else
		return false; 
}

deposit_card* card_repository::findDepo(int num)
{
	for (auto depo = depoCard.cbegin(); depo != depoCard.cend(); ++depo)
		if (depo->getCardNum() == num)
			return const_cast<deposit_card*>(&(*depo));
	return nullptr;
}

campus_card* card_repository::findCamp(int num)
{
	for (auto camp = campCard.cbegin(); camp != campCard.cend(); ++camp)
		if (camp->getCardNum() == num)
			return const_cast<campus_card*>(&(*camp));
	return nullptr;
}

binding_card* card_repository::findBind(int num)
{
	for (auto bind = bindCard.cbegin(); bind != bindCard.cend(); ++bind)
		if (bind->getCardNum() == num)
			return const_cast<binding_card*>(&(*bind));
	return nullptr;
}

bool card_repository::findName(const std::string name)
{
	for (auto camp = campCard.cbegin(); camp != campCard.cend(); ++camp)
		if (camp->getName() == name)
			return true;
	return false;
}

#endif // CARD_REPOSITORY_HPP