; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn
; Lab 3 - Individual OS kernel
; Ubuntu 18.04 + nasm 2.13.02

; second stage loader
;;;;; os.asm ;;;;;
    bits 16
    extern main
    global _start
    global load_program

    UserPrgOffset equ 0A100h

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

%macro writeIVT 2            ; write interrupt vector table (IVT)
    ; num, function address
    mov ax, 0000h            ; physical address
    mov es, ax
    mov ax, %1
    mov bx, 4
    mul bx                   ; calculate the IVT address (ax*4)
    mov si, ax
    mov ax, %2
    mov [es:si], ax          ; write segment
    add si, 2
    mov ax, cs
    mov [es:si], ax          ; write offset
%endmacro

; section .text
_start:
    writeIVT 20h, INT20H
    writeIVT 21h, INT21H

    ; mov ax, cs
    ; mov ds, ax
    ; showstring msg, msglen, 0, 0
    ; mov ah, 0                ; (BIOS) function code
    ; int 16h                  ; (BIOS) read keyboard
    ; push word 0              ; use for align
    call dword main

end:
    call clear
    showstring msg4, msglen4, 0, 0
    jmp $

load_program:                ; load [es:bx]
    push ebp
    mov ebp, esp

    mov ecx, [ebp+8]         ; argument 1, (BIOS) start sector
    ; mov cx, ax
    mov ax, cs               ; segment address (store data)
    mov es, ax               ; set segment
    mov bx, UserPrgOffset    ; user program address
    mov ah, 2                ; (BIOS) function code
    mov al, 1                ; (BIOS) # of sector that program used
    mov dl, 0                ; (BIOS) driver: floppy disk (0)
    mov dh, 1                ; (BIOS) magnetic head
    mov ch, 0                ; (BIOS) cylinder
    int 13H                  ; (BIOS) 13h: read disk
    call clear
    jmp UserPrgOffset        ; X.com has been loaded into memory

    mov esp, ebp
    pop ebp
    ret

INT20H:                      ; get click and return
    showstring msg3, msglen3, 24, 3
    mov ah, 01h
    int 16h
    jz noclick
    mov ah, 00h              ; click the button
    int 16h
    cmp ax, 2e03h            ; click Ctrl + C
    jne noclick
    int 21h
    noclick:
    iret                     ; interrupt return

INT21H:                      ; directly return
    call clear
    ret                 ; return back to kernel
    iret

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
    msg db 'Loading main function... Enter to get in...'
    msglen equ ($-msg)

    msg2 db 'DEBUG!'
    msglen2 equ ($-msg2)

    msg3 db 'Input Ctrl + C to return!'
    msglen3 equ ($-msg3)

    msg4 db 'See you again in CHZOS next time! Byebye!'
    msglen4 equ ($-msg4)