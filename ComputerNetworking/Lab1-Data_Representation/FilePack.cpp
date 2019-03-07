// 陈鸿峥 17341015
// chenhzh37@mail2.sysu.edu.cn
// Exp 1: Data Representation - File copy

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <string>
using namespace std;

#define PAK_FILE_PATH "FileSet.pak"

class FileClass
{
public:
	FileClass(const string _filepath):
		filepath(_filepath){};

	string getFileName()
	{
		if (filename != "")
			return filename;
		for (int i = filepath.size() - 1; i >= 0; --i)
			if (filepath[i] != '\\')
				filename = filepath[i] + filename;
			else
				break;
		return filename;
	}

	ifstream::pos_type getSize()
	{
	    ifstream in(filepath, ifstream::ate | ifstream::binary); // at the end of file
	    return in.tellg();
	}

	ifstream getStream()
	{
		ifstream input(filepath, ios::binary);
		return input;
	}

private:
	string filepath;
	string filename;
};

bool packFile(const string src, const string dst)
{
	FileClass infile(src);
	ofstream outfile(dst, ios::app|ios::binary); // append

	outfile << infile.getFileName() << endl;
	outfile << infile.getSize() << endl;
	ifstream input = infile.getStream();
	string str;
	while (getline(input,str))
		outfile << str << endl;

	outfile.close();
	return true;
}

bool unpackFile(const string srcFile, const string dstPath)
{
	ifstream input(srcFile,ios::binary);
	string dst = dstPath;
	for (int i = 1; true ; ++i){
		string str, filename;
		if (!getline(input,filename))
			break;
		if (filename == "")
			if (!getline(input,filename))
				break;
		if (filename.find("/") != -1 || filename.find("\\") != -1)
			for (int i = filename.length() -1; i >= 0; --i)
				if (filename[i] == '/' || filename[i] == '\\'){
					filename = filename.substr(i+1,filename.length()-i);
					break;
				}
		cout << "正在解包第" << i << "个文件：" << filename << " ..." << endl;
		getline(input,str); // size
		streampos size = stol(str);

		// cout << "FileName: " << filename << endl;
		// cout << "Size: " << exp_size << endl;

		fstream output_file;
		// get output file name
		string path;
		if (dst[dst.length()-1] != '\\')
			dst += "\\";
		int cnt = 1;
		while (true){
			if (cnt == 1)
				path = dst + filename;
			else{
				int index = filename.find(".");
				if (index != -1){
					string suffix = filename.substr(index,filename.size()-index);
					path = dst + filename.substr(0,index)
						 + "(" + to_string(cnt) + ")" + suffix;
				} else {
					path = dst + filename + "(" + to_string(cnt) + ")";
				}
			}
			if (access(path.c_str(), F_OK ) == -1)
				break;
			cnt++;
		}

		ofstream output(path, ios::out|ios::binary);
		char* memblock = new char [size];
		input.read(memblock,size);
		output.write(memblock,size);
		output.close();

		delete [] memblock;
	}
}

int main(int argc, char *argv[])
{
	if (argc == 1){
	ofstream output(PAK_FILE_PATH); // initialization
	output.close();
	while (true){
		cout << "输入打包文件（含路径）：";
		string src_file_path;
		getline(cin,src_file_path);
		if (src_file_path == "exit")
			break;

		packFile(src_file_path,PAK_FILE_PATH);
	}
	cout << "打包结束！\n" << endl;
	}

	cout << "输入解包文件夹：";
	string output_path;
	cin >> output_path;
	
	unpackFile(PAK_FILE_PATH,output_path);

	cout << "解包结束！";
	return 0;
}