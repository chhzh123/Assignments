#ifndef TORRENT
#define TORRENT

#include<string>
#include<vector>

typedef struct {
	std::string announce;
	int length;
	std::string name;
	int piece_length;
	std::vector<std::string> pieces;
}torrent_file;

//在torrent.cpp中实现下面的函数

//从给定的种子文件中读取信息，返回一个torrent_file对象
torrent_file read_torrent(std::string filename);
//根据给定的文件和信息做种子，如果成功返回种子文件名，否则返回空字符串
std::string make_torrent(std::string filename, int piece_length, std::string announce);


#endif // !TORRENT

