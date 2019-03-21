; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn
; Lab 3 - Individual OS kernel
; Ubuntu 18.04 + nasm 2.13.02

; second stage loader
;;;;; os.asm ;;;;;
    bits 16
    extern main
    global _start

%macro showstring 4
    ; msg, msglen, row, col
    mov ax, cs               ; as = cs
    mov ds, ax               ; data segment
    mov bp, %1               ; bp = string index
    mov ax, ds               ; (BIOS) es:bp = string address
    mov es, ax               ; es = ax
    mov cx, %2               ; (BIOS) set string length
    mov ax, 1301h            ; (BIOS) ah = 13h (BIOS: function code) al = 01h (only char)
    mov bx, 0007h            ; (BIOS) page/bh = 00h property/bl = 07h
    mov dh, %3               ; (BIOS) row
    mov dl, %4               ; (BIOS) col
    int 10h                  ; (BIOS) 10h: show one string
%endmacro

; section .text
_start:
    mov ax, cs
    mov ds, ax
    showstring msg, msglen, 0, 0
    mov ah, 0                ; (BIOS) function code
    int 16h                  ; (BIOS) read keyboard
    call dword main

end:
    jmp $

;;;;; Data Segment ;;;;;
datadef:
    msg db 'Loading main function... Enter to get in...'
    msglen equ ($-msg)

    msg2 db 'DEBUG!'
    msglen2 equ ($-msg2)