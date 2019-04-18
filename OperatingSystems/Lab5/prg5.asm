; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn

; Disk Initialization
;;; ***** REMEMBER TO MODIFY ***** ;;;
	org 0F100h              ; ORG (origin) is used to set the assembler location counter

start:
	mov ax, 1
	int 21h
	mov ax, 2
	int 21h
	mov ax, 3
	int 21h
	mov ax, 4
	int 21h
	ret
end:
	jmp $