// Copyright (c) 2019 Hongzheng Chen
// chenhzh37@mail2.sysu.edu.cn
// Ubuntu 18.04 + gcc 7.3.0

/****** interpreter.h ******/

#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "sysio.h"

const char* HELLO_INTER_INFO = "Welcome to C/Python Interpreter!\n";
const char* PROMPT_INTER_INFO = ">>> ";
const char* EXIT_INTER_STR = "exit";
const char* RETURN_STR = "return";
const char* PRINTF_STR = "test";

void ini_inter()
{
	puts(newline);
	set_color(CYAN,BLACK);
	puts(HELLO_INTER_INFO);
	set_color(WHITE,BLACK);
}

void name_error(char* str)
{
	if (strcmp(str,newline) == 0 || strlen(str) == 0)
		return;
	char* NAME_ERROR = "NameError: name '";
	char buf[MAX_BUF_LEN];
	strcpy(buf,NAME_ERROR);
	strcat(buf,str);
	NAME_ERROR = "' is not defined";
	strcat(buf,NAME_ERROR);
	puts(buf);
	puts(newline);
}

void io_test()
{
	char* str = "This is a scanf test:\n";
	char* format = "%s";
	printf(format, str);
	str = "Please enter two numbers (a and b): ";
	printf(format, str);
	int a, b;
	format = "%d %d";
	scanf(format,&a,&b);
	str = "This is a printf test: ";
	int c = a + b;
	char d = '!';
	format = "%s\nThe sum of a and b is: %d + %d = %d%c\n";
	printf(format, str, a, b, c, d);
}

bool isop(char c)
{
	return (c == '+' || c == '-' || c == '*' || c == '/');
}

int splitop(const char* str, int* offset)
{
	// str: the input string
	// c: the split char (where to split)
	// offset: the position of c in str
	int length = strlen(str);
	int i= 0;
	int cnt = 0;
	for (i = 0; i < length; ++i)
		if (isop(str[i]))
			offset[cnt++] = i;
	offset[cnt++] = length;
	return cnt;
}

typedef struct data
{
	char name[100];
	int n;
} data;

int findvar(char* str,data* var,int num)
{
	for (int j = 0; j < num; ++j)
		if (strcmp(var[j].name,str) == 0)
			return j;
	return -1;
}

int calculator(const char* str,data* var,int num)
{
	int offset[MAX_BUF_LEN];
	int opnum = splitop(str,offset);
	int i = 0;
	int res = 0;
	for (i = 0; i < opnum; ++i){
		char s1[MAX_BUF_LEN];
		strmcpy(s1,str,(i == 0 ? 0 : offset[i-1]+1),offset[i]);
		int n1;
		if (!isnum(s1)){
			int varn = findvar(s1,var,num);
			if (varn != -1)
				n1 = var[varn].n;
		} else
			n1 = atoi(s1);
		if (i == 0){
			res = n1;
			continue;
		} else switch (str[offset[i-1]]) {
			case '+': res += n1;break;
			case '-': res -= n1;break;
			case '*': res *= n1;break;
			case '/': res /= n1;break;
			default: break;
		}
	}
	return res;
}

void store(char* str,data* var,int* num)
{
	int offset[MAX_BUF_LEN];
	split(str,'=',offset);
	char name[100];
	strmcpy(name,str,0,offset[0]);
	strcpy(var[*num].name,name);
	char number[100];
	strmcpy(number,str,offset[0]+1,offset[1]);
	int n = atoi(number);
	var[*num].n = n;
	*num += 1;
}

void interpreter(){
	ini_inter();
	data var[100];
	int cntvar = 0;
	while (1){
		put_info(PROMPT_INTER_INFO);
		char str[MAX_BUF_LEN];
		getline(str);
		if (strcmp(str,newline) == 0 || strlen(str) == 0)
			continue;
		if (isnum(str)){
			puts(str);
			puts(newline);
			continue;
		}
		int varn = findvar(str,var,cntvar);
		if (varn != -1){
			puti(var[varn].n);
			puts(newline);
			continue;
		}
		if (isin(str,'=')){
			store(str,var,&cntvar);
			continue;
		}
		if (isin(str,'+') || isin(str,'-') || isin(str,'*') || isin(str,'/')){
			puti(calculator(str,var,cntvar));
			puts(newline);
			continue;
		}
		char buf[MAX_BUF_LEN];
		strncpy(buf,str,6);
		if (strcmp(buf,PRINTF_STR) == 0){
			io_test();
		} else if (strcmp(str,EXIT_INTER_STR) == 0){
			puts(newline);
			return;
		} else
			name_error(str);
	}
}

#endif