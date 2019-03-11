#!/bin/bash

rm mydisk.img
/sbin/mkfs.msdos -C mydisk.img 1440
nasm os.asm -o os.com
dd if=os.com of=mydisk.img conv=notrunc
nasm prg1.asm -o prg1.com
dd if=prg1.com of=mydisk.img seek=1 conv=notrunc
nasm prg2.asm -o prg2.com
dd if=prg2.com of=mydisk.img seek=2 conv=notrunc
nasm prg3.asm -o prg3.com
dd if=prg3.com of=mydisk.img seek=3 conv=notrunc
nasm prg4.asm -o prg4.com
dd if=prg4.com of=mydisk.img seek=4 conv=notrunc