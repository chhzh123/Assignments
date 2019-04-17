; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn
; Ubuntu 18.04 + nasm 2.13.02

;;;;; bootloader.asm ;;;;;

    bits 16
; Constants
    OSOffset equ 7e00h       ; OS kernel location (2nd sector)
    OSSectorNum equ 2
    NumKernelSectors equ 36

;;;;; Initialization ;;;;;
    org  7c00h

;;;;; Initializeion ;;;;;
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

start:
    showstring msg, msglen, 0, 0

input:
    mov ah, 0                ; (BIOS) function code
    int 16h                  ; (BIOS) read keyboard

;;;;; Load OS kernel ;;;;;
loadOS:                      ; load [es:bx]
    call clear               ; clear the screen
    mov ax, 0                ; segment address (store data), cs?
    mov es, ax               ; set segment
    mov bx, OSOffset         ; user program address
    mov ah, 2                ; (BIOS) function code
    mov al, NumKernelSectors ; (BIOS) # of sector that program used
    mov dl, 0                ; (BIOS) driver: floppy disk (0)
    mov dh, 0                ; (BIOS) magnetic head
    mov ch, 0                ; (BIOS) cylinder
    mov cl, OSSectorNum      ; (BIOS) start sector
    int 13H                  ; (BIOS) 13h: read disk
    jmp OSOffset             ; OS kernel has been loaded into memory

end:
    jmp $

clear:
    mov ax, 0B800h           ; video memory
    mov es, ax
    mov si, 0
    mov cx, 80*25
    mov dx, 0
    clearloop:
        mov [es:si], dx
        add si, 2
    loop clearloop
    ret

;;;;; Data Segment ;;;;;
datadef:
    msg db 'Welcome to CHZOS! Press any key to enter!'
    msglen equ ($-msg)