; Copyright (c) 2019 Hongzheng Chen
; chenhzh37@mail2.sysu.edu.cn
; Ubuntu 18.04 + nasm 2.13.02

; second stage loader
;;;;; os.asm ;;;;;
    bits 16
    extern main
    global _start
    global load_program

    UserPrgOffset equ 0F100h

    MAX_X equ 79
    MAX_Y equ 24
    MIN_X equ 0
    MIN_Y equ 0
    DEFAULT_COLOR equ 0007h
    PrgSize equ 2
    CylinderNum  equ 1
    HeadNum equ 0

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

%macro saveIVT 3             ; save interrupt vector table (IVT)
    ; num, function address, offset
    mov ax, 0000h            ; physical address
    mov es, ax
    mov ax, %1
    mov bx, 4
    mul bx                   ; calculate the IVT address (ax*4)
    mov si, ax
    mov %2, word [es:si]     ; write segment
    add si, 2
    mov %3, word [es:si]     ; write offset
%endmacro

%macro writeIVT 2            ; write interrupt vector table (IVT)
    ; num, function address
    push es
    push ds
    push si
    pusha
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
    popa
    pop si
    pop ds
    pop es
%endmacro

%macro loadPrg 1
    push ebp                 ; push one more
    mov ebp, esp
    push ax
    push bx
    push cx
    push dx
    push es

    mov ax, cs               ; segment address (store data)
    mov es, ax               ; set segment
    mov bx, UserPrgOffset    ; user program address
    mov ah, 2                ; (BIOS) function code
    mov al, PrgSize          ; (BIOS) # of sector that program used
    mov dl, 0                ; (BIOS) driver: floppy disk (0)
    mov dh, HeadNum          ; (BIOS) magnetic head
    mov ch, CylinderNum      ; (BIOS) cylinder / side num
    mov cl, %1               ; (BIOS) start sector num
    int 13H                  ; (BIOS) 13h: read disk
    ; call clear
    call UserPrgOffset        ; X.com has been loaded into memory
    ; jmp UserPrgOffset        ; X.com has been loaded into memory

    pop es
    pop dx
    pop cx
    pop bx
    pop ax
    mov esp, ebp
    pop ebp
%endmacro

_start:
    ; saveIVT 16h, [IVTseg], [IVToffset]
    ; writeIVT 08h, Timer        ; Programmable Timer
    writeIVT 20h, INT20H       ; Ctrl+C return
    writeIVT 21h, INT21H       ; all the system interrupts

    call dword main

end:
    call clear
    showstring msg4, msglen4, 0, 0, DEFAULT_COLOR
    jmp $

load_program:                ; load [es:bx]
    loadPrg byte [ebp+8]     ; argument 1, (BIOS) start sector (cl)
    ret

Timer:
    cli
    pusha
    dec byte [count]
    jnz TimerRet
    jmp draw_slash
slash_ret:
    mov byte [count], delay
TimerRet:
    push eax
    ; sending a message to the PIC confirming
    ; that the interrupt has been handled.
    ; If this isn’t done the PIC won’t generate any more interrupts.
    ; Acknowledging a PIC interrupt is done by
    ; sending the byte 0x20 to the PIC that raised the interrupt
    mov al, 20h              ; al = End of Interrupt (EOI)
    out 20h, al              ; master PIC
    out 0A0h, al             ; slave PIC
    pop eax
    popa
    sti
    iret

draw_slash:
    showchar pos_slash_x, pos_slash_y, bar, color
    inc byte [cnt]
    cmp byte [cnt], 4
    jne no_change_cnt
    mov byte [cnt], 1
no_change_cnt:
    cmp byte [cnt], 1
    je to_vert
    cmp byte [cnt], 2
    je to_lslash
    cmp byte [cnt], 3
    je to_rslash
    jmp slash_ret

to_vert:
    mov byte [bar], '|'
    jmp slash_ret
to_lslash:
    mov byte [bar], '/'
    jmp slash_ret
to_rslash:
    mov byte [bar], '\'
    jmp slash_ret

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
    mov ax, 0
    int 21h
    noclick:
    iret                     ; interrupt return

INT21H:
    cmp ax, 0
    je backtokernel
    cmp ax, 1
    je loadprg1
    cmp ax, 2
    je loadprg2
    cmp ax, 3
    je loadprg3
    cmp ax, 4
    je loadprg4
    iret

backtokernel:
    call clear
    jmp main                 ; return back to kernel
    iret
loadprg1:
    loadPrg 1
    iret
loadprg2:
    loadPrg 2
    iret
loadprg3:
    loadPrg 3
    iret
loadprg4:
    loadPrg 4
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

    bar db '|'
    cnt db 2
    pos_slash_x dw 79
    pos_slash_y dw 24

    color db 07h

    IVTseg dw 0
    IVToffset dw 0