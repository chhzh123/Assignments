// Copyright (c) 2019 Hongzheng Chen
// chenhzh37@mail2.sysu.edu.cn
// Ubuntu 18.04 + gcc 7.3.0

/****** userprg.h ******/

#ifndef USERPRG_H
#define USERPRG_H

#include "type.h"
#include "sysio.h"

#define FLOPPY_144_SECTORS_PER_TRACK 18

void lba_2_chs(uint32_t lba, uint16_t* cyl, uint16_t* head, uint16_t* sector)
{
    *cyl    = lba / (2 * FLOPPY_144_SECTORS_PER_TRACK);
    *head   = ((lba % (2 * FLOPPY_144_SECTORS_PER_TRACK)) / FLOPPY_144_SECTORS_PER_TRACK);
    *sector = ((lba % (2 * FLOPPY_144_SECTORS_PER_TRACK)) % FLOPPY_144_SECTORS_PER_TRACK + 1);
}

typedef struct Program{
	char name[8];
	size_os space;
	char pos[8];
	char description[50];
} Program;

#define PRG_NUM 6
#define PrgSectorOffset 0
Program prgs[PRG_NUM];

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
	name = "6"; pos = "/"; char* buf6 = "Draw the box";
	set_one_prg_info(5,name,512,pos,buf6);
}

static inline void show_prg_info()
{
	char* str = "Name  Size  Pos  Description";
	puts(str);
	puts(newline);
	set_prg_info();
	for (size_os i = 0; i < PRG_NUM; ++i){
		char buf[MAX_BUF_LEN];
		puts(prgs[i].name);
		puts(tab);
		itoa(prgs[i].space,buf);
		puts(buf);
		puts(tab);
		puts(prgs[i].pos);
		puts(tab);
		puts(prgs[i].description);
		puts(newline);
	}
}

static inline void execute(char c)
{
	int num = c - '0';
	if (num > 0 && num < PRG_NUM+1)
		// load_program(num + PrgSectorOffset);
		load_program((num + PrgSectorOffset)*2-1);
	else{
		char* err = "Error: No this program!\n";
		put_error(err);
	}
	// clear();
}

#endif // USERPRG_H