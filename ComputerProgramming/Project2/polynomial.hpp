// Copyright 2018 CHZ
// Author: Chen Hongzheng(Reddie)
// E-mail: chenhzh37@mail2.sysu.edu.cn

// This is the headfile of polynomial calculator made by Reddie, which contains implement of Polynomial class.

#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

class Polynomial
{
public:
	// constructor
	// default
	Polynomial();
	// construct from pairs like (1,2)
	Polynomial(std::vector<std::pair<int,int>> poly);
	// construct from coefficients
	Polynomial(std::vector<int> poly);
	// construct from other polynomials (=)
	Polynomial(const Polynomial& other);
	// destructor
	~Polynomial();
	// clear all terms including 0
	void clear();

	// basic operations
	// derivative
	Polynomial D() const;
	// overload
	Polynomial operator+(const Polynomial& Poly) const;
	Polynomial operator-(const Polynomial& Poly) const;
	Polynomial operator*(const Polynomial& Poly) const;
	Polynomial& operator=(const Polynomial& other);
	Polynomial& operator+=(const Polynomial& Poly);
	Polynomial& operator*=(const Polynomial& Poly);
	bool operator==(const Polynomial& Poly) const;
	// assignment
	int operator()(int value) const;

	// set and get data
	void setCoefficient(int numTerm, int coeff);
	int getTermCoeff(int numTerm) const; // get coefficient of specific term
	int getHighestDegree() const;
	std::vector<int> getCoefficients() const;

	// output
	std::string PolyToString() const;
private:
	// a_0x_0^0+a_1x_1^1+\cdots+a_nx_n^n
	// only record coefficients
	// push term 0 into the vector while no corresponding terms were inputed
	std::vector<int> terms;
};

Polynomial::Polynomial() // set to 0
{
	if (terms.size() == 0)
		terms.push_back(0);
	else
		terms[0] = 0;
}

Polynomial::Polynomial(std::vector<std::pair<int,int>> poly)
{
	sort(poly.begin(),poly.end());
	for (auto term : poly)
		setCoefficient(term.second,term.first);
}

Polynomial::Polynomial(std::vector<int> poly)
{
	for (auto term : poly)
		terms.push_back(term);
}

Polynomial::Polynomial(const Polynomial& other)
{
	for (auto term : other.terms)
		terms.push_back(term);
}

void Polynomial::clear()
{
	terms.clear();
}

Polynomial::~Polynomial()
{
	clear();
}

Polynomial& Polynomial::operator=(const Polynomial& other)
{
	clear(); // important!!!
	for (auto term : other.terms)
		terms.push_back(term);
	return *this; // important!!!
}

Polynomial Polynomial::operator+(const Polynomial& other) const
{
	int m = getHighestDegree(), n = other.getHighestDegree();
	std::vector<int> a = getCoefficients(), b = other.getCoefficients();
	std::vector<int> resTerms;
	auto ptermA = a.begin();
	auto ptermB = b.begin();
	if (m > n)
	{
		for ( ; ptermB != b.end(); ++ptermA, ++ptermB)
			resTerms.push_back(*ptermA + *ptermB);
		for ( ; ptermA != a.end(); ++ptermA)
			resTerms.push_back(*ptermA);
	} else {
		for ( ; ptermA != a.end(); ++ptermA, ++ptermB)
			resTerms.push_back(*ptermA + *ptermB);
		for ( ; ptermB != b.end(); ++ptermB)
			resTerms.push_back(*ptermB);
	}
	return Polynomial(resTerms);
}

Polynomial Polynomial::operator-(const Polynomial& other) const
{
	int m = getHighestDegree(), n = other.getHighestDegree();
	std::vector<int> a = getCoefficients(), b = other.getCoefficients();
	std::vector<int> resTerms;
	auto ptermA = a.begin();
	auto ptermB = b.begin();
	if (m > n)
	{
		for ( ; ptermB != b.end(); ++ptermA, ++ptermB)
			resTerms.push_back(*ptermA - *ptermB);
		for ( ; ptermA != a.end(); ++ptermA)
			resTerms.push_back(*ptermA);
	} else {
		for ( ; ptermA != a.end(); ++ptermA, ++ptermB)
			resTerms.push_back(*ptermA - *ptermB);
		for ( ; ptermB != b.end(); ++ptermB)
			resTerms.push_back(*ptermB);
	}
	return Polynomial(resTerms);
}

Polynomial Polynomial::operator*(const Polynomial& other) const
{
	int m = getHighestDegree(), n = other.getHighestDegree();
	Polynomial resPoly;
	for (int i = 0; i <= m+n; ++i) // degree = i
	{
		int sum = 0;
		for (int j = 0; j <= i; ++j)
			sum += getTermCoeff(j)*other.getTermCoeff(i-j);
		resPoly.setCoefficient(i, sum);
	}
	return resPoly;
}

Polynomial Polynomial::D() const
{
	std::vector<int> c = getCoefficients();
	std::vector<int> resTerms;
	int currTerm = 0;
	for (auto pterm = c.begin(); pterm != c.end(); ++pterm, ++currTerm)
		resTerms.push_back((*pterm)*currTerm);
	resTerms.erase(resTerms.begin());
	return Polynomial(resTerms);
}
Polynomial& Polynomial::operator+=(const Polynomial& other)
{
	*this = *this + other;
	return *this;
}

Polynomial& Polynomial::operator*=(const Polynomial& other)
{
	*this = (*this) * (other);
	return *this;
}

bool Polynomial::operator==(const Polynomial& other) const
{
	int m = getHighestDegree(), n = other.getHighestDegree();
	if (m != n)
		return false;
	for (int i = 0; i < m; ++i)
		if (getTermCoeff(i) != other.getTermCoeff(i))
			return false;
	return true;
}

int pow(int base,int exp) // my pow function
{
	int res = 1;
	while (exp--)
		res *= base;
	return res;
}

int Polynomial::operator()(int value) const // assignment
{
	int numTerm = 0, sum = 0;
	for (auto pterm = terms.begin(); pterm < terms.end(); ++pterm, ++numTerm)
		sum += (*pterm) * pow(value,numTerm);
	return sum;
}

void Polynomial::setCoefficient(int numTerm, int coeff)
{
	if (numTerm > (int)(terms.size())-1) // terms.size unsigned int!!!
	{
		for (int i = terms.size(); i < numTerm; ++i)
			terms.push_back(0);
		terms.push_back(coeff);
	} else {
		int currTerm = 0;
		for (auto pterm = terms.begin(); pterm != terms.end(); ++pterm, ++currTerm)
			if (numTerm == currTerm)
			{
				*pterm = coeff;
				break;
			}
	}
}

std::string Polynomial::PolyToString() const
{
	if (terms.size() == 1 && terms[0] == 0)
		return "0";
	std::string res;
	int numTerm = (int)terms.size()-1, degree = getHighestDegree();
	for (auto pterm = terms.rbegin(); pterm != terms.rend(); ++pterm, --numTerm) // be careful of overflow
	{
		std::string temp;
		switch (numTerm)
		{
			case 0: break;
			case 1: temp = std::string("x"); break;
			default: temp = std::string("x^") + std::to_string(numTerm); break;
		}
		switch (*pterm){
			case 0: break;
			case 1: res += std::string("+") + (numTerm == 0 ? std::string("1") : temp); break;
			case -1: res += std::string("-") + (numTerm == 0 ? std::string("1") : temp); break;
			default: res += (*pterm > 0 ? std::string("+") : std::string("")) + std::to_string(*pterm) + temp;break;
		}
		if (numTerm == degree && *pterm > 0)
			res = res.substr(1);
	}
	return res;
}

int Polynomial::getTermCoeff(int numTerm) const
{
	if (numTerm >= terms.size())
		return 0;
	int currTerm = 0;
	for (auto pterm = terms.begin(); pterm != terms.end(); ++pterm, ++currTerm)
		if (currTerm == numTerm)
			return *pterm;
}

int Polynomial::getHighestDegree() const
{
	if(terms.size() == 0)
		return 0;
	int degree = (int)terms.size()-1;
	for (auto pterm = terms.rbegin(); pterm != terms.rend() && *pterm == 0; ++pterm, --degree); // rbegin
	return degree;
}

std::vector<int> Polynomial::getCoefficients() const
{
	return terms;
}

#endif // POLYNOMIAL_H