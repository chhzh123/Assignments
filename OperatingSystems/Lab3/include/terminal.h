// Copyright (c) 2019 Hongzheng Chen
// chenhzh37@mail2.sysu.edu.cn
// Lab 3 - Individual OS kernel
// Ubuntu 18.04 + gcc 7.3.0

/****** terminal.h ******/

#ifndef TERMINAL_H
#define TERMINAL_H

#include "sysio.h"
#include "userprg.h"

const char* PROMPT_INFO = "chzos> ";
const char* HELLO_INFO = "Welcome to CHZOS!\n";
const char* HELP_INFO =
"CHZ OS Shell version 0.1\n\
These shell commands are defined internally. Type 'help' to see this list.\n\
\n\
 help       -- Show this list\n\
 show       -- Show existing programs\n\
 exec       -- Execute all the user programs\n\
 exec [num] -- Execute the num-th program\n\
 exit       -- Exit OS\n";
const char* HELP_STR = "help";
const char* SHOW_STR = "show";
const char* EXE_STR = "exec";
const char* EXIT_STR = "exit";
const char* CMD_NOT_FOUND = ": command not found";

void initialize()
{
	clear();
	set_color(CYAN,BLACK);
	show_string(HELLO_INFO);
	set_color(WHITE,BLACK);
}

void command_not_found(char* str)
{
	char* newline = "\n";
	if (strcmp(str,newline) == 0 || strlen(str) == 0)
		return;
	strcat(str,CMD_NOT_FOUND);
	show_string(str);
	show_string(newline);
}

void terminal()
{
	while (1){
		put_info(PROMPT_INFO);
		char str[MAX_BUF_LEN];
		getline(str);
		if (strcmp(str,HELP_STR) == 0)
			show_string(HELP_INFO);
		else if (strcmp(str,SHOW_STR) == 0)
			show_prg_info();
		else if (strcmp(str,EXIT_STR) == 0)
			break;
		else if (strlen(str) >= 4){
			char cpystr[MAX_BUF_LEN];
			strncpy(cpystr,str,4);
			if (strcmp(cpystr,EXE_STR) == 0){
				if (strlen(str) == 4){
					for (size_os i = 1; i < 5; ++i)
						execute(i+'0'); // batch execution
				} else
					execute(str[5]);
			} else
				command_not_found(str);
		} else
			command_not_found(str);
	}
}

#endif // TERMINAL_H