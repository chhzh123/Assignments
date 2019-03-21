// Copyright (c) 2019 Hongzheng Chen
// chenhzh37@mail2.sysu.edu.cn
// Lab 3 - Individual OS kernel
// Ubuntu 18.04 + gcc 7.3.0

/****** kernel.c ******/

#include "sysio.h"

const char* PROMPT_INFO = "chzos> ";
const char* HELLO_WORLD = "Hello,world!";

uint16_t main(){
	clear();
	show_string(PROMPT_INFO);
	set_color(WHITE,BLACK);
	show_string(HELLO_WORLD);
	return 0;
}