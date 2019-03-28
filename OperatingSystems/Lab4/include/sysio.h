// Copyright (c) 2019 Hongzheng Chen
// chenhzh37@mail2.sysu.edu.cn
// Ubuntu 18.04 + gcc 7.3.0

/****** sysio.h ******/

#ifndef SYSIO_H
#define SYSIO_H

#include "type.h"
#include "string.h"

static const size_wd WD_WIDTH = 80; // window width
static const size_wd WD_HEIGHT = 25; // window height

#define VGA_ADDRESS ((volatile uint16_t *) 0xB8000)
#define NEWLINE "\n"
#define PAGE_UP_CONST 2

const char* TEST = "Hello,world!";

volatile size_wd terminal_row;
volatile size_wd terminal_col;
volatile uint8_t color_code;
char buf[MAX_BUF_LEN];

enum Color{
	BLACK = 0,
	BLUE = 1,
	GREEN = 2,
	CYAN = 3,
	RED = 4,
	MAGENTA = 5,
	BROWN = 6,
	LIGHT_GREY = 7,
	DARK_GREY = 8,
	LIGHT_BLUE = 9,
	LIGHT_GREEN = 10, //a
	LIGHT_CYAN = 11, // b
	LIGHT_RED = 12, // c
	LIGHT_MAGENTA = 13, // d
	LIGHT_BROWN = 14, // e
	WHITE = 15 // f
};

size_wd get_cursor(){
	// AH = 0x03
	// BH = display page (usually, if not always 0)
	// The return values:
	// CH = start scanline
	// CL = end scanline
	// DH = row
	// DL = column
	size_wd p;
	asm volatile("int 0x10\n\t"
				:"=d"(p)
				:"a"(0x0300), "b"(0)
				);
	return p;
}

void set_cursor(size_wd row, size_wd col){
	// AH = 0x02
	// BH = display page (usually, if not always 0)
	// DH = row
	// DL = column
	asm volatile("int 0x10\n\t"
				:
				:"a"(0x0200), "b"(0), "d"((row << 8) | col)
				);
}

void page_up(){
	// ah=06h page up
	// al= page up lines
	// dx= row & col
	size_wd p = 1536 + PAGE_UP_CONST; // 0x0602
	asm volatile("int 0x10\n\t"
				:
				:"a"(p), "c"(0x0000), "d"(0x184F)
				);
}

void put_cursor(){
	if (terminal_col + 1 == WD_WIDTH) { // newline
		set_cursor(terminal_row+1, 0);
		if (terminal_row + 1 == WD_HEIGHT) // roll back
			set_cursor(0, 0);
	} else
		set_cursor(terminal_row, terminal_col+1);
}

static inline uint8_t entry_color(enum Color fg, enum Color bg) 
{
	// higher 4 bits: front color
	// lower 4 bits: background color
	return ((uint8_t) fg) | (((uint8_t) bg) << 4);
}
 
static inline uint16_t entry_code(unsigned char uc, uint8_t color) 
{
	// higher 8 bits: color
	// lower 8 bits (1B): ascii
	return ((uint16_t) uc) | ((uint16_t) color) << 8;
}

static inline void set_color(enum Color fg, enum Color bg)
{
	color_code = entry_color(fg,bg);
}

static inline void draw_char(char c, size_wd row, size_wd col, uint8_t color){
	size_wd pos = (row * 80 + col) * 2;
	asm volatile(
				"push es\n\t"
				"mov es, ax\n\t"
				"mov es:[bx],cx\n\t"
				"pop es\n\t"
				:
				:"a"(0xB800), "b"(pos), "c"((color << 8) | c)
				:
			);
	put_cursor();
}

void clear()
{
	// initialization
	terminal_row = 0;
	terminal_col = 0;
	color_code = entry_color(GREEN, BLACK);
	// clear
	for (size_wd y = 0; y < WD_HEIGHT; ++y) // terminal_row
		for (size_wd x = 0; x < WD_WIDTH; ++x) // terminal_col
			draw_char(' ', y, x, 0x07);
}

char getchar(){
	char ch;
	asm volatile("int 0x16\n\t"
				:"=a"(ch)
				:"a"(0x1000)
				);
	return ch;
}

void putchar(char c)
{
	if (c == '\r' || c == '\n') { // newline
		if (++terminal_row == WD_HEIGHT){
			// clear();
			page_up();
			terminal_row -= PAGE_UP_CONST;
		}
		terminal_col = 0;
		return;
	}
	draw_char(c, terminal_row, terminal_col, color_code);
	if (++terminal_col == WD_WIDTH) { // newline
		terminal_col = 0;
		if (++terminal_row == WD_HEIGHT){ // roll back
			// clear();
			page_up();
			terminal_row -= PAGE_UP_CONST;
		}
	}
}

void show_string(const char* data)
{
	size_wd size = strlen(data);
	for (size_wd i = 0; i < size; i++)
		putchar(data[i]);
}

void put_error(const char* data)
{
	set_color(RED,BLACK);
	show_string(data);
	set_color(WHITE,BLACK);
}

void put_info(const char* data)
{
	set_color(GREEN,BLACK);
	show_string(data);
	set_color(WHITE,BLACK);
}

void getline(char* res)
{
	int i = 0;
	while(1){
		char ch = getchar();
		if (ch == '\b'){
			if (i == 0)
				continue;
			res[--i] = '\0';
			if (terminal_col == 0){
				terminal_col = WD_WIDTH-1;
				terminal_row--;
			} else
				terminal_col--;
			putchar(' ');
			if (terminal_col == 0){
				terminal_col = WD_WIDTH-1;
				terminal_row--;
			} else
				terminal_col--;
			set_cursor(terminal_row,terminal_col);
			continue;
		}
		res[i++] = ch;
		putchar(ch);
		if (ch == '\r' || ch == '\n'){
			res[--i] = '\0';
			break;
		}
	}
}

#endif // SYSIO_H