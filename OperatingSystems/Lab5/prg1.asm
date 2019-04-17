; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn

; Graphic memory: 25*80 Text Mode

bits 16

; Constants
Dn_Rt equ 1     ; D-Down, U-Up, R-right, L-Left
Up_Rt equ 2
Up_Lt equ 3
Dn_Lt equ 4
MAX_X equ 40
MAX_Y equ 12
msgX equ 0
msgY equ 0
in_delay equ 60000 ; control the speed
out_delay equ 6000  ; outer loop

; Disk Initialization
;;; ***** REMEMBER TO MODIFY ***** ;;;
	org 0A100h              ; ORG (origin) is used to set the assembler location counter
; start:
	mov ax, 0B800h          ; graphic memory start address, ax is GPR
	mov gs, ax              ; GS = B800h segment register

%macro OneChar 5
	; px: position x
	; py: position y
	; drt: direction
	; char: char ascii
	; color
	mov ax, [%1]
	mov [px], ax
	mov ax, [%2]
	mov [py], ax
	mov al, [%3]
	mov [drt], al
	mov al, [%4]
	mov [char], al
	mov al, [%5]
	; inc al
	; mov [color], al         ; change colors

	call onemove

	;;; Store back to memory ;;;
	mov dx, [px]
	mov [%1], dx
	mov dx, [py]
	mov [%2], dx
	mov dl, [drt]           ; bytes, be careful!
	mov [%3], dl
	mov dl, [color]
	mov [%5], dl
%endmacro

mainloop:
	call delayloop
	OneChar px, py, drt, char, color
	call showinfo
	int 20h                 ; INTERRUPT!!!
	mov ax, [cnt]
	dec ax
	mov [cnt], ax
	cmp ax, 0
	jne mainloop
	ret
	; int 21h

showinfo:
	mov esi, msg            ; move msg's address into si (GPR)
	mov di, (msgY*80+msgX)*2; middle of the 10th line
	mov cx, msglen          ; loop index
	infoloop:
		mov bl, byte [esi]
		inc si                  ; next char
		mov bh, 03h             ; property
		mov word [gs:di], bx
		add di, 2               ; 2 Bytes
		loop infoloop           ; automatically minus 1 from cx
	ret

delayloop:
	mov ecx, out_delay
	outloop:
		mov eax, in_delay
		inloop:
			dec eax
			jg inloop
	loop outloop
	ret

onemove:
	cmp byte[drt], 1
	jz DnRt
	cmp byte[drt], 2
	jz UpRt
	cmp byte[drt], 3
	jz UpLt
	cmp byte[drt], 4
	jz DnLt
onemoveret:
	ret

;;;;; Down Right ;;;;;
DnRt:
	inc word[px]
	inc word[py]
	mov ax, word[py]
	cmp ax, MAX_Y            ; px = MAX_Y?
	je dr2ur                 ; jump if the result of the last arithmetic operation was zero
	mov ax, word[px]         ; PX = MAX_X?
	cmp ax, MAX_X
	je dr2dl
	jmp show
dr2ur:
	mov word[py], MAX_Y-2
	mov byte[drt], Up_Rt
	jmp show
dr2dl:
	mov word[px], MAX_X-2
	mov byte[drt], Dn_Lt
	jmp show

;;;;; Up Right ;;;;;
UpRt:
	inc word[px]
	dec word[py]
	mov ax, word[px]
	cmp ax, MAX_X
	je ur2ul
	mov ax, word[py]
	cmp ax, -1
	je ur2dr
	jmp show
ur2ul:
	mov word[px], MAX_X-2
	mov byte[drt], Up_Lt
	jmp show
ur2dr:
	mov word[py], 1
	mov byte[drt], Dn_Rt
	jmp show

;;;;; Up Left ;;;;;
UpLt:
	dec word[px]
	dec word[py]
	mov ax, word[py]
	cmp ax, -1
	je ul2dl
	mov ax, word[px]
	cmp ax, -1
	je ul2ur
	jmp show
ul2dl:
	mov word[py], 1
	mov byte[drt], Dn_Lt
	jmp show
ul2ur:
	mov word[px], 1
	mov byte[drt], Up_Rt
	jmp show

;;;;; Down Left ;;;;;
DnLt:
	dec word[px]
	inc word[py]
	mov ax, word[px]
	cmp ax, -1
	je dl2dr
	mov ax, word[py]
	cmp ax, MAX_Y
	je dl2ul
	jmp show
dl2dr:
	mov word[px], 1
	mov byte[drt], Dn_Rt
	jmp show
dl2ul:
	mov word[py], MAX_Y-2
	mov byte[drt], Up_Lt
	jmp show

;;;;; Show words ;;;;;
show:
	push ax
	push bx
	push gs
	push bp
	
	xor ax, ax              ; Compute memory address, ax = 0
	mov ax, word [py]       ; ax = y
	mov bx, 80              ; bx = 80
	mul bx                  ; dstop: ax = 80*y, srcop: parameter (bx)
	add ax, word [px]       ; ax = 80*y + x
	mov bx, 2               ; bx = 2
	mul bx                  ; ax = (80*y + x) * 2
	mov bp, ax              ; bp = ax, position
	mov ah, [color]         ; AH = char property, 0000：Black, 1111：White, high bits
	mov al, byte [char]     ; AL = char value (default 20h=space), low bits
	mov word [gs:bp], ax    ; AX = (AH,AL)
	
	pop bp
	pop gs
	pop bx
	pop ax
	jmp onemoveret

;;;;; Data Segment ;;;;;
datadef:
	px dw 0
	py dw 0
	drt db Dn_Rt                    ; Down Right
	char db '*'
	color db 00000111b

	msg db 'This is Prg1!'
	msglen equ ($-msg)

	cnt db 20               ; maximum iteration