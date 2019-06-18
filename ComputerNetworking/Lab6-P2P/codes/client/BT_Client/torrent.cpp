//
// Created by jijorbq on 6/14/19.
//

#include "torrent.h"
#include "hash.h"
#include <string>
#include<assert.h>
#include<fstream>
#include<cstdio>
#include<cstring>

using std::string;
using std::to_string;

string read_torrent_str(const char t[], int sz, int &i) {
	int strl = 0;
	assert(isdigit(t[i]));
	for (; i < sz && isdigit(t[i]); ++i)	strl = strl * 10 + t[i] - 48;
	assert(t[i++] == ':');
	assert(i + strl <= sz);
	//	string ret=t.substr(i, strl); i+=strl;
	string ret = "";
	for (int j = 0; j < strl; ++j, ++i) ret += t[i];
	return ret;
}
int read_torrent_int(const char t[], int sz, int &i) {
	assert(t[i++] == 'i');
	int num = 0, sgn = 1;
	if (t[i] == '-') sgn = -1, ++i;
	assert(isdigit(t[i]));
	for (; i < sz && isdigit(t[i]); ++i)num = num * 10 + t[i] - 48;
	assert(t[i++] == 'e');
	return num * sgn;
}
// vector<string> read_torrent_strlis(const string &t, int &i){
// 	vector<string> res;
// 	assert(t[i++] =='l');
// 	while (i+1<t.size() && t[i]!='e')
// 		res.push_back(read_torrent_str(t, i));
// 	assert(t[i++]=='e');
// 	return res;
// }
torrent_file read_torrent(string filename) {
	//	ifstream fin(filename);
	//	ostringstream finbuf; finbuf<<fin.rdbuf();
	//	string s(finbuf.str());fin.close();
	//	torrent_file res;
	//	assert(s.front()=='d' && s.back()=='e');
	//	s=s.substr(1, s.size()-2);
	FILE *fin;
	fopen_s(&fin, filename.c_str(), "rb"); int filesz = 0;
	for (char tmpc; fread(&tmpc, 1, 1, fin); ++filesz);
	fclose(fin); fopen_s(&fin, filename.c_str(), "rb");
	char *s = new char[filesz + 1];
	fread(s, 1, filesz, fin); fclose(fin);
	assert(s[0] == 'd' && s[--filesz] == 'e');

	torrent_file res;
	for (int i = 1; i < filesz;) {
		string itname = read_torrent_str(s, filesz, i);

		if (itname == "announce") {
			res.announce = read_torrent_str(s, filesz, i);
		}
		else if (itname == "length") {
			res.length = read_torrent_int(s, filesz, i);
		}
		else if (itname == "name") {
			res.name = read_torrent_str(s, filesz, i);
		}
		else if (itname == "piece_length") {
			res.piece_length = read_torrent_int(s, filesz, i);
		}
		else if (itname == "pieces") {
			string totpiece = read_torrent_str(s, filesz, i);
			for (int j = 0; j < totpiece.size(); j += 20)
				res.pieces.push_back(totpiece.substr(j, 20));
		}
		else {
			fprintf(stderr, "Wrong on item name");
		}
	}

	delete[]s;
	return res;
}


string make_torrent_int(int num) {
	string ret = "";
	int sgn = 1;
	if (num < 0)num = -num, sgn = -1;
	if (num == 0) ret = "0";
	else for (; num; num /= 10) ret += char(num % 10 + 48);
	reverse(ret.begin(), ret.end());
	if (sgn == -1) ret = "-" + ret;
	return "i" + ret + "e";
}

string make_torrent_str(string t) {
	return to_string(t.size()) + ":" + t;
}


void write_to_file(string str, FILE *fout) {
	for (auto c : str) fwrite(&c, 1, 1, fout);
}

string make_torrent(string filename, int piece_length, string announce) {

	char *buf = new char[piece_length + 1];
	string torrname, s;
	if (filename.find('.') == string::npos)torrname = filename + ".torrent";
	else torrname = filename.substr(0, filename.find('.')) + ".torrent";
	FILE *fin, *fout;
	fopen_s(&fin, filename.c_str(), "rb");
	fopen_s(&fout, torrname.c_str(), "wb");
	fseek(fin, 0, SEEK_END);
	int filesz = ftell(fin);
	fclose(fin); fopen_s(&fin, filename.c_str(), "rb");
	write_to_file("d", fout);
	write_to_file(make_torrent_str("announce") + make_torrent_str(announce),
		fout);
	write_to_file(make_torrent_str("length") + make_torrent_int(filesz),
		fout);
	write_to_file(make_torrent_str("name") + make_torrent_str(filename),
		fout);
	write_to_file(make_torrent_str("piece_length") + make_torrent_int(piece_length),
		fout);
	write_to_file(make_torrent_str("pieces"), fout);

	//	string pieces="";
	//	for (int cur=fread(buf, 1,piece_length, fin);cur
	//			;cur=fread(buf, 1,piece_length, fin))
	//		pieces+=SHA1(buf, cur);
	//	make_torrent_str(pieces);

		// return "d"+s+"e";


	write_to_file(to_string((filesz + piece_length - 1) / piece_length * 20) + ":",
		fout);
	for (int cur = fread(buf, 1, piece_length, fin); cur
		; cur = fread(buf, 1, piece_length, fin)) {
		string hashstr = SHA1(buf, cur);
		assert(hashstr.size() == 20);
		write_to_file(SHA1(buf, cur), fout);
	}
	write_to_file("e", fout);
	fclose(fout);
	delete[]buf;
	return torrname;
}