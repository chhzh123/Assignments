# Copyright 2018, SYSU
# Author: Chen Hongzheng, Major in Computer Science (17), SDCS, SYSU
# E-mail: chenhzh37@mail2.sysu.edu.cn
# Date: 2018/12/08
# This is the testing program for multi-cycle CPU. The program is written in MIPS.

.text
.global main

# ----------- main function segment -------------
main:
	addiu  $1,  $0,  8    # 0x00
	ori    $2,  $0,  2    # 0x04
	xori   $3,  $2,  8    # 0x08
	sub    $4,  $3,  $1   # 0x0C
	and    $5,  $4,  $2   # 0x10
	sll    $5,  $5,  2    # 0x14
	beq    $5,  $1,  -2   # 0x18
	jal    0x0000050      # 0x1C
	slt    $8,  $13, $1   # 0x20
	addiu  $14, $0,  -2   # 0x24
	slt    $9,  $8,  $14  # 0x28
	slti   $10, $9,  2    # 0x2C
	slti   $11, $10, 0    # 0x30
	add    $11, $11, $10  # 0x34
	bne    $11, $2,  -2   # 0x38
	addiu  $12, $0,  -2   # 0x3C
	addiu  $12, $12, 1    # 0x40
	bltz   $12, -2        # 0x44
	andi   $12, $2,  2    # 0x48
	j      0x000005C      # 0x4C
	sw     $2,  4($1)     # 0x50
	lw     $13, 4($1)     # 0x54
	jr     $31            # 0x58
	halt                  # 0x5C