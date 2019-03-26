; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn
; Ubuntu 18.04 + nasm 2.13.02

; second stage loader
;;;;; os.asm ;;;;;
    bits 16
    extern main
    global _start
    global load_program

    UserPrgOffset equ 0A100h

    MAX_X equ 79
    MAX_Y equ 24
    MIN_X equ 0
    MIN_Y equ 0
    DEFAULT_COLOR equ 0007h

%macro showchar 4
    ; x/col, y/row, char, property 
    push ax
    push bx
    push gs
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
    pop gs
    pop bx
    pop ax
%endmacro

%macro showstring 5
    ; msg, msglen, row, col, color
    push ax
    push bx
    push cx
    push dx
    push bp
    push ds
    push es
    mov ax, cs               ; as = cs
    mov ds, ax               ; data segment
    mov bp, %1               ; bp = string index
    mov ax, ds               ; (BIOS) es:bp = string address
    mov es, ax               ; es = ax
    mov cx, %2               ; (BIOS) set string length
    mov ax, 1301h            ; (BIOS) ah = 13h (BIOS: function code) al = 01h (only char)
    mov bx, %5               ; (BIOS) page/bh = 00h property/bl = 07h
    mov dh, %3               ; (BIOS) row
    mov dl, %4               ; (BIOS) col
    int 10h                  ; (BIOS) 10h: show one string
    pop es
    pop ds
    pop bp
    pop dx
    pop cx
    pop bx
    pop ax
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

%macro loadPrg 1
    mov ax, cs               ; segment address (store data)
    mov es, ax               ; set segment
    mov bx, UserPrgOffset    ; user program address
    mov ah, 2                ; (BIOS) function code
    mov al, 1                ; (BIOS) # of sector that program used
    mov dl, 0                ; (BIOS) driver: floppy disk (0)
    mov dh, 1                ; (BIOS) magnetic head
    mov ch, 0                ; (BIOS) cylinder
    mov cl, %1               ; start sector
    int 13H                  ; (BIOS) 13h: read disk
    call clear
    jmp UserPrgOffset        ; X.com has been loaded into memory
%endmacro

; section .text
_start:
    ; writeIVT 08h, Timer        ; Timer
    writeIVT 20h, INT20H       ; Ctrl+C return
    writeIVT 21h, INT21H       ; directly return
    writeIVT 33h, INT33H       ; load user prg1
    writeIVT 34h, INT34H       ; load user prg2
    writeIVT 35h, INT35H       ; load user prg3
    writeIVT 36h, INT36H       ; load user prg4

    call dword main

end:
    call clear
    showstring msg4, msglen4, 0, 0, DEFAULT_COLOR
    jmp $

load_program:                ; load [es:bx]
    push ebp                 ; push one more
    mov ebp, esp

    loadPrg byte [ebp+8]     ; argument 1, (BIOS) start sector (cl)

    mov esp, ebp
    pop ebp
    ret

Timer:
    dec byte [count]
    jnz TimerRet
    call draw_box
    mov byte [count], delay
TimerRet:
    push eax
    mov al, 20h              ; al = EOI
    out 20h, al              ; master PIC
    out 0A0h, al             ; slave PIC
    pop eax
    iret

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
    inc bl
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

INT20H:                      ; get click and return
    showstring msg3, msglen3, 24, 3, DEFAULT_COLOR
    mov ah, 01h
    int 16h
    jz noclick
    mov ah, 00h              ; click the button
    int 16h
    showstring msgOuch, msgOuchlen, 12, 33, 0004h
    cmp ax, 2e03h            ; click Ctrl + C
    jne noclick
    int 21h
    noclick:
    iret                     ; interrupt return

INT21H:                      ; directly return
    call clear
    jmp main                 ; return back to kernel
    iret

INT33H:
    showstring msgOuch, msgOuchlen, 13, 30, 0004h
    ; push dword 1
    loadPrg 1
    iret
INT34H:
    push dword 2
    call load_program
    iret
INT35H:
    push dword 3
    call load_program
    iret
INT36H:
    push dword 4
    call load_program
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

    msgOuch db 'OUCH! OUCH!'
    msgOuchlen equ ($-msgOuch)

    delay equ 1
    count db delay

    posx dw 0
    posy dw 0
    char db 'A'
    color db 07h
    drt db 1                    ; direction: 1 lr, 2 ud, 3 rl, 4 lu