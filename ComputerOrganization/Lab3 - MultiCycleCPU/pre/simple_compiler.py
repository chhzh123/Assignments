# Copyright 2018, SYSU
# Author: Chen Hongzheng, Major in Computer Science (17), SDCS, SYSU
# E-mail: chenhzh37@mail2.sysu.edu.cn
# Date: 2018/12/08
# This is a simple compiler for MIPS program.

import re

opcode = {
	"add":   "000000",
	"sub":   "000001",
	"addiu": "000010",
	"and":   "010000",
	"andi":  "010001",
	"ori":   "010010",
	"xori":  "010011",
	"sll":   "011000",
	"slti":  "100110",
	"slt":   "100111",
	"sw":    "110000",
	"lw":    "110001",
	"beq":   "110100",
	"bne":   "110101",
	"bltz":  "110110",
	"j":     "111000",
	"jr":    "111001",
	"jal":   "111010",
	"halt":  "111111"
}

def twos(x):
	res = eval(x)
	if res >= 0:
		return str(bin(res)[2:]).zfill(16)
	else:
		return str(bin(2**16 + res)[2:]).zfill(16)

def parse(ins):
	res = ""
	elems = list(filter(bool,re.split("\t| +#.*|, +| +",ins)))
	op = elems[0]
	res += opcode[op]
	if op in ["add","sub","and","slt"]:
		rd, rs, rt = elems[1][1:], elems[2][1:], elems[3][1:]
		res += str(bin(eval(rs))[2:]).zfill(5)
		res += str(bin(eval(rt))[2:]).zfill(5)
		res += str(bin(eval(rd))[2:]).zfill(5)
		res += "".zfill(11)
	elif op in ["addiu","andi","ori","xori","slti"]:
		rt, rs, imm = elems[1][1:], elems[2][1:], elems[3]
		res += str(bin(eval(rs))[2:]).zfill(5)
		res += str(bin(eval(rt))[2:]).zfill(5)
		res += twos(imm)
	elif op in ["sw","lw"]:
		rt = elems[1][1:]
		lst = list(filter(bool,re.split("\(\$|\)",elems[2])))
		imm, rs = lst[0], lst[1]
		res += str(bin(eval(rs))[2:]).zfill(5)
		res += str(bin(eval(rt))[2:]).zfill(5)
		res += twos(imm)
	elif op in ["beq","bne"]:
		rs, rt, imm = elems[1][1:], elems[2][1:], elems[3]
		res += str(bin(eval(rs))[2:]).zfill(5)
		res += str(bin(eval(rt))[2:]).zfill(5)
		res += twos(imm)
	elif op == "bltz":
		rs, imm = elems[1][1:], elems[2]
		res += str(bin(eval(rs))[2:]).zfill(5)
		res += "".zfill(5)
		res += twos(imm)
	elif op in ["j","jal"]:
		address = elems[1] # 0x...
		res += str(bin(eval(address))[2:-2]).zfill(26)
	elif op == "jr":
		rs = elems[1][1:]
		res += str(bin(eval(rs))[2:]).zfill(5)
		res += "".zfill(21)
	elif op == "sll":
		rd, rt, sa = elems[1][1:], elems[2][1:], elems[3]
		res += "".zfill(5)
		res += str(bin(eval(rt))[2:]).zfill(5)
		res += str(bin(eval(rd))[2:]).zfill(5)
		res += str(bin(eval(sa))[2:]).zfill(5)
		res += "".zfill(6)
	else: # halt
		res += "".zfill(26)
	return res

instructions = []
infile = open("./test.asm","r")
for line in infile:
	if line[0] in [".","#","m","\n"]:
		continue
	else:
		instructions.append(parse(line))

for x in instructions:
	print("0x" + hex(eval("0b" + x))[2:].zfill(8),end="    ")
	print(x[:6],end=" ")
	print(x[6:11],end=" ")
	print(x[11:16],end=" ")
	print(x[16:20],end=" ")
	print(x[20:24],end=" ")
	print(x[24:28],end=" ")
	print(x[28:32],end="\n")

outfile = open("./instruction.data","w")
for (n,x) in enumerate(instructions):
	# little endian!
	outfile.write("// #%d\n" % n)
	outfile.write(x[24:32])
	outfile.write("\n")
	outfile.write(x[16:24])
	outfile.write("\n")
	outfile.write(x[8:16])
	outfile.write("\n")
	outfile.write(x[:8])
	outfile.write("\n")