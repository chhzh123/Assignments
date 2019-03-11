; 17341015 陈鸿峥
; chenhzh37@mail2.sysu.edu.cn
; Lab 2 - 加载用户程序的监控程序
; Ubuntu 18.04 + nasm 2.13.02

; Constants
OffSetOfUserPrg equ 0A100h

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

writeIVT 20h, INT20H

begin:
    showstring msg, msglen, 0, 0
    showstring msg2, msglen2, 1, 0

input:
    mov ah, 0                ; (BIOS) function code
    int 16h                  ; (BIOS) read keyboard
    sub al, '0'              ; (BIOS) return al = ASCII
    cmp al, 1
    jl input                 ; not from 1~4
    cmp al, 4
    jg input                 ; not from 1~4
    inc al                   ; start from 2
    mov [sectorNum], al      ; put it into memory

;;;;; Load Program ;;;;;
load:                        ; load [es:bx]
    call clear               ; clear the screen
    mov ax, cs               ; segment address (store data)
    mov es, ax               ; set segment
    mov bx, OffSetOfUserPrg  ; user program address
    mov ah, 2                ; (BIOS) function code
    mov al, 1                ; (BIOS) # of sector that program used
    mov dl, 0                ; (BIOS) driver: floppy disk (0)
    mov dh, 0                ; (BIOS) magnetic head
    mov ch, 0                ; (BIOS) cylinder
    mov cl, [sectorNum]      ; (BIOS) start sector
    int 13H                  ; (BIOS) 13h: read disk
    jmp OffSetOfUserPrg      ; a.com has been loaded into memory

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

INT20H:
    showstring msg3, msglen3, 24, 3
    mov ah, 01h
    int 16h
    jz noclick
    mov ah, 00h               ; click the button
    int 16h
    cmp ax, 2e03h             ; click Ctrl + C
    jne noclick
    call clear                ; clear the screen
    jmp begin                 ; return to monitor program
    noclick:
    iret                      ; interrupt return

end:
    jmp $

;;;;; Data Segment ;;;;;
datadef:
    msg db 'Welcome to CHZOS!'
    msglen equ ($-msg)

    msg2 db 'Input number 1~4 to select a program: '
    msglen2 equ ($-msg2)

    msg3 db 'Input Ctrl + C to return!'
    msglen3 equ ($-msg3)

    sectorNum db '1'