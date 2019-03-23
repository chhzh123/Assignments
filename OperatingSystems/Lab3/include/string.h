// Copyright (c) 2019 Hongzheng Chen
// chenhzh37@mail2.sysu.edu.cn
// Lab 3 - Individual OS kernel
// Ubuntu 18.04 + gcc 7.3.0

/****** string.h ******/

#ifndef STRING_H
#define STRING_H

#include "type.h"

#define NEWLINE "\n"
#define SPACE " "
#define TAB "    "

size_os strlen(const char* str) 
{
	size_os len = 0;
	for (size_os i = 0; i < MAX_BUF_LEN; i++)
		if (str[len] != '\0')
			len++;
		else
			break;
	return len;
}

int strcmp(const char *s1, const char *s2){
	// = 0, < -1, > 1
	while(*s1 && (*s1 == *s2))
	{
		s1++;
		s2++;
	}
	return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}

char* strcpy(char *strDest, const char *strSrc)
{
	char *temp = strDest;
	while((*strDest++=*strSrc++) != '\0');
	return temp;
}

char* strncpy(char *dest, const char *src, size_t n)
{
	size_t i;
	for (i = 0; i < n && src[i] != '\0'; i++)
		dest[i] = src[i];
	for ( ; i < n; i++)
		dest[i] = '\0';
	return dest;
}

char* strcat(char* desy, const char* src)
{
	char* ptr = desy + strlen(desy);
	while (*src != '\0')
		*ptr++ = *src++;
	*ptr = '\0';
	return desy;
}

int atoi(const char *str) 
{ 
	int res = 0;
	for (int i = 0; str[i] != '\0'; ++i) 
		res = res*10 + str[i] - '0';
	return res; 
}

void reverse(char* s)
{
	int i, j;
	char c;
	for (i = 0, j = strlen(s)-1; i < j; i++, j--) {
		c = s[i];
		s[i] = s[j];
		s[j] = c;
	}
}

// K&R
void itoa(int n, char* s)
{
	int i, sign;
	if ((sign = n) < 0)
		n = -n;
	i = 0;
	do { // generate digits in reverse order
		s[i++] = n % 10 + '0';
	} while ((n /= 10) > 0);
	if (sign < 0)
		s[i++] = '-';
	s[i] = '\0';
	reverse(s);
}

#endif // STRING_H