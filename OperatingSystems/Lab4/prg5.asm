; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn

; Graphic memory: 25*80 Text Mode

; Constants
MAX_X equ 80
MAX_Y equ 25

; Disk Initialization
;;; ***** REMEMBER TO MODIFY ***** ;;;
	org 0A100h              ; ORG (origin) is used to set the assembler location counter

start:
	int 33h
	int 34h
	int 35h
	int 36h
	int 21h
end:
	jmp $