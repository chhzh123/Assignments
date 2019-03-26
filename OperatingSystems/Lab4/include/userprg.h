// Copyright (c) 2019 Hongzheng Chen
// chenhzh37@mail2.sysu.edu.cn
// Ubuntu 18.04 + gcc 7.3.0

/****** userprg.h ******/

#ifndef USERPRG_H
#define USERPRG_H

#include "type.h"
#include "sysio.h"

typedef struct Program{
	char name[8];
	size_os space;
	char pos[8];
	char description[50];
} Program;

#define PRG_NUM 5
#define PrgSectorOffset 0
Program prgs[PRG_NUM];

const char* tab = TAB;
const char* newline = NEWLINE;

// assembly function
extern void load_program(int num);
// asm volatile ("jmp load_program\n\t"
// 			:
// 			:"a"(0x000C));

static inline void set_one_prg_info(int i, char* _name, size_os _space, char* _pos, char* _description)
{
	strcpy(prgs[i].name,_name);
	prgs[i].space = _space;
	strcpy(prgs[i].pos,_pos);
	strcpy(prgs[i].description,_description);
}

static inline void set_prg_info()
{
	char* name = "1", *pos = "/", *buf = "Quadrant 1: Flying single char";
	set_one_prg_info(0,name,434,pos,buf);
	name = "2"; pos = "/"; char* buf2 = "Quadrant 2: Flying two chars - V shape";
	set_one_prg_info(1,name,508,pos,buf2);
	name = "3"; pos = "/"; char* buf3 = "Quadrant 3: Flying two chars - OS";
	set_one_prg_info(2,name,458,pos,buf3);
	name = "4"; pos = "/"; char* buf4 = "Quadrant 4: Flying two chars - parallelogram";
	set_one_prg_info(3,name,508,pos,buf4);
	name = "5"; pos = "/"; char* buf5 = "Interrupt test program";
	set_one_prg_info(4,name,512,pos,buf5);
}

static inline void show_prg_info()
{
	char* str = "Name  Size  Pos  Description";
	show_string(str);
	show_string(newline);
	set_prg_info();
	for (size_os i = 0; i < PRG_NUM; ++i){
		char buf[MAX_BUF_LEN];
		show_string(prgs[i].name);
		show_string(tab);
		itoa(prgs[i].space,buf);
		show_string(buf);
		show_string(tab);
		show_string(prgs[i].pos);
		show_string(tab);
		show_string(prgs[i].description);
		show_string(newline);
	}
}

static inline void execute(char c)
{
	int num = c - '0';
	if (num > 0 && num < PRG_NUM+1)
		load_program(num + PrgSectorOffset);
	else{
		char* err = "Error: No this program!\n";
		put_error(err);
	}
}

#endif // USERPRG_H