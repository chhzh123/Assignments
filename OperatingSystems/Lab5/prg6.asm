; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn
; Ubuntu 18.04 + nasm 2.13.02

	bits 16

    MAX_X equ 79
    MAX_Y equ 24
    MIN_X equ 0
    MIN_Y equ 0
    DEFAULT_COLOR equ 0007h

    in_delay equ 6000 ; control the speed
    out_delay equ 6000  ; outer loop

	org 0F100h              ; ORG (origin) is used to set the assembler location counter

%macro showchar 4
    ; x/col, y/row, char, property 
    push ax
    push bx
    push gs
    push bp
    mov ax, 0B800h          ; VGA
    mov gs, ax
    xor ax, ax              ; Compute memory address, ax = 0
    mov ax, word [%2]       ; ax = y
    mov bx, 80              ; bx = 80
    mul bx                  ; dstop: ax = 80*y, srcop: parameter (bx)
    add ax, word [%1]       ; ax = 80*y + x
    mov bx, 2               ; bx = 2
    mul bx                  ; ax = (80*y + x) * 2
    mov bp, ax              ; bp = ax, position
    mov ah, byte [%4]       ; AH = char property, 0000：Black, 1111：White, high bits
    mov al, byte [%3]       ; AL = char value (default 20h=space), low bits
    mov word [gs:bp], ax    ; AX = (AH,AL)
    pop bp
    pop gs
    pop bx
    pop ax
%endmacro

start:
    call delayloop
    int 20h
draw_box:
    push ax
    push bx
    push cx
    push dx
    push bp
    showchar posx, posy, char, color
showchar_finished:
    mov ah, byte [posx]
    mov al, byte [posy]
    mov cl, byte [char]
    mov dl, byte [drt]
    cmp dl, 1
    je deal_lr
    cmp dl, 2
    je deal_ud
    cmp dl, 3
    je deal_rl
    cmp dl, 4
    je deal_du
draw_box_finished:
    mov bl, byte [color]
    inc bl ; change color
    mov bh, 08h
    cmp bl, bh
    jne no_change_back_color
    mov bl, 01h
no_change_back_color:
    mov byte [color], bl
    mov byte [drt], dl
    mov byte [char], cl
    mov byte [posy], al
    mov byte [posx], ah
    pop bp
    pop dx
    pop cx
    pop bx
    pop ax
    jmp start
    ret

deal_lr:
    inc ah
    cmp ah, MAX_X
    jne draw_box_finished
    mov dl, 2 ; change direction
    jmp draw_box_finished
deal_ud:
    inc al
    cmp al, MAX_Y
    jne draw_box_finished
    mov dl, 3
    jmp draw_box_finished
deal_rl:
    dec ah
    cmp ah, MIN_X
    jne draw_box_finished
    mov dl, 4
    jmp draw_box_finished
deal_du:
    dec al
    cmp al, MIN_Y
    jne draw_box_finished
    mov dl, 1
    inc cl                   ; change char
    cmp cl, 'Z'
    jne draw_box_finished
    mov cl, 'A'
    jmp draw_box_finished

delayloop:
    mov ecx, out_delay
    outloop:
        mov eax, in_delay
        inloop:
            dec eax
            jg inloop
    loop outloop
    ret

end:
	jmp $

datadef:
    posx dw 0
    posy dw 0
    char db 'A'
    color db 07h
    drt db 1                    ; direction: 1 lr, 2 ud, 3 rl, 4 lu