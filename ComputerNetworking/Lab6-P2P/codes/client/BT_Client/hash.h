#ifndef HASH_H
#define HASH_H

#include <cstdint>
#include <iostream>
#include <string>


class SHA1class
{
public:
	SHA1class();
	void update(const std::string &s);
	void update(std::istream &is);
	void update(const char buff[], int bufflen);
	std::string final();
	static std::string from_file(const std::string &filename);

private:
	uint32_t digest[5];
	std::string buffer;
	uint64_t transforms;
};

//对buff进行hash，返回20字节的哈希值
std::string SHA1(char* buff, int bufflen);


#endif // !HASH_H
