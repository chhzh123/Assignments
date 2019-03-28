; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn

; Disk Initialization
;;; ***** REMEMBER TO MODIFY ***** ;;;
	org 0A100h              ; ORG (origin) is used to set the assembler location counter

start:
	int 33h
	int 34h
	int 35h
	int 36h
	ret
end:
	jmp $