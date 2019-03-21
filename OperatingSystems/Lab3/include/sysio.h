// Copyright (c) 2019 Hongzheng Chen
// chenhzh37@mail2.sysu.edu.cn
// Lab 3 - Individual OS kernel
// Ubuntu 18.04 + gcc 7.3.0

/****** sysio.h ******/

#ifndef SYSIO_H
#define SYSIO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h> // machine independent

typedef uint16_t size_wd;

static const size_wd WD_WIDTH = 80; // window width
static const size_wd WD_HEIGHT = 25; // window height

#define VGA_ADDRESS ((volatile uint16_t *) 0xB8000)

volatile size_wd termial_row;
volatile size_wd terminal_col;
volatile uint8_t color_code;

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

void set_color(enum Color fg, enum Color bg)
{
	color_code = entry_color(fg,bg);
}

size_wd strlen(const char* str) 
{
	size_wd len = 0;
	while (str[len])
		len++;
	return len;
}

void draw_char(char c, size_wd row, size_wd col, uint8_t color){
	size_wd pos = (row * 80 + col) * 2;
	asm volatile(
				"push es\n\t"
				"mov es, ax\n\t"
				"mov es:[bx],cx\n\t"
				"pop es\n\t"
				:
				:"a"(0xB800),"b"(pos), "c"((color << 8) | c)
				:
			);
}

void clear()
{
	// initialization
	termial_row = 0;
	terminal_col = 0;
	color_code = entry_color(GREEN, BLACK);
	// clear
	for (size_wd y = 0; y < WD_HEIGHT; ++y) // termial_row
		for (size_wd x = 0; x < WD_WIDTH; ++x) // terminal_col
			draw_char(' ', y, x, 0x07);
}
 
void putchar(char c)
{
	draw_char(c, termial_row, terminal_col, color_code);
	if (++terminal_col == WD_WIDTH) { // newline
		terminal_col = 0;
		if (++termial_row == WD_HEIGHT) // roll back
			termial_row = 0;
	}
}
 
void show_string(const char* data)
{
	size_wd size = strlen(data);
	for (size_wd i = 0; i < size; i++)
		if (data[i] == '\n') { // newline
			if (++termial_row == WD_WIDTH)
				termial_row = 0;
			continue;
		} else
			putchar(data[i]);
}

#endif // SYSIO_H