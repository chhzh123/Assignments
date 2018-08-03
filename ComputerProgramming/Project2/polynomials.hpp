// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is a headfile of polynomial calculator made by Reddie, which contains implement of warehouse of polynomials.

#ifndef POLYNOMIALS_H
#define POLYNOMIALS_H

class Polynomials_
{
public:
	//default constructor
	Polynomials_(){};
	//append polynomials
	bool append(const std::string name, Polynomial poly);
	//delete polynomials
	bool erase(const std::string name);

	//basic calculation
	Polynomial add(const std::vector<Polynomial> Polys) const;//enable multi-polynomials input
	Polynomial subtract(const Polynomial& PolyA, const Polynomial& PolyB) const;
	Polynomial multipy(const std::vector<Polynomial> Polys) const;//enable multi-polynomials input
	Polynomial derivative(const Polynomial& Poly) const;
	int assign(const Polynomial& Poly, int value) const;
	bool equalQ(const std::vector<Polynomial> Polys) const;

	//get data
	size_t getSize() const;
	bool findName(const std::string name) const;
	std::vector<std::string> getNames() const;//get all names of polynomials
	std::vector<Polynomial> getPolys(std::vector<std::string> names);//get specific polynomials from names 
	std::map<std::string,Polynomial> getAllData() const;
	
	//count num of polynomials without names(temporary polynomials when calculation)
	static int cntVir;
private:
	//map names to polynomials
	std::map<std::string,Polynomial> data;
};

bool Polynomials_::append(const std::string name, Polynomial poly)
{
	if (!findName(name))
		data[name] = poly;
	else
		return false;
	return true;
}

bool Polynomials_::erase(const std::string name)
{
	if (findName(name))
		data.erase(name);
	else
		return false;
	return true;
}

Polynomial Polynomials_::add(const std::vector<Polynomial> Polys) const
{
	Polynomial sum;
	for (auto Poly : Polys)
		sum += Poly;
	return sum;
}

Polynomial Polynomials_::multipy(const std::vector<Polynomial> Polys) const
{
	Polynomial product;
	product.setCoefficient(0,1);
	for (auto Poly : Polys)
		product *= Poly;
	return product;
}

Polynomial Polynomials_::subtract(const Polynomial& PolyA, const Polynomial& PolyB) const
{
	return PolyA - PolyB;
}

Polynomial Polynomials_::derivative(const Polynomial& Poly) const
{
	return Poly.D();
}

int Polynomials_::assign(const Polynomial& Poly, int value) const
{
	return Poly(value);
}

bool Polynomials_::equalQ(const std::vector<Polynomial> Polys) const
{
	for (auto pPoly1 = Polys.begin(); pPoly1 != Polys.end(); ++pPoly1)
		for (auto pPoly2 = Polys.begin(); pPoly2 < pPoly1; ++pPoly2)
			if (!((*pPoly1) == (*pPoly2)))
				return false;
	return true;
}

size_t Polynomials_::getSize() const
{
	return data.size();
}

bool Polynomials_::findName(const std::string name) const
{
	return data.find(name) != data.end();
}

std::vector<std::string> Polynomials_::getNames() const
{
	std::vector<std::string> res;
	for (auto pele = data.begin(); pele != data.end(); ++pele)
		res.push_back(pele->first);
	return res;
}

std::vector<Polynomial> Polynomials_::getPolys(std::vector<std::string> names)
{
	std::vector<Polynomial> res;
	for (auto name : names)
		if (findName(name))
			res.push_back(data[name]);
	return res;
}

std::map<std::string,Polynomial> Polynomials_::getAllData() const
{
	return data;
}

#endif // POLYNOMIALS_H