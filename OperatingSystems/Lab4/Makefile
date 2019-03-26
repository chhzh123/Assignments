CC = gcc
CCFLAGS := -c -march=i386 -nostdlib -ffreestanding -Wall -Wextra
CCFLAGS += -Iinclude -lgcc -fno-PIE -nostartfiles
CCFLAGS += -m16 -masm=intel -mpreferred-stack-boundary=2
# CCFLAGS += -m32 -masm=intel

AS = nasm
ASFLAGS = -felf32

LD = ld
LDFLAGS = -m elf_i386 -N --oformat binary

IMG = mydisk.img
KERNEL = bootloader.bin kernel.bin
PROGS = prg1.com prg2.com prg3.com prg4.com prg5.com

DEBUG = kernel.s

all: programs
	-rm -f $(IMG)
	/sbin/mkfs.msdos -C $(IMG) 1440
	dd if=bootloader.bin of=$(IMG) conv=notrunc
	dd if=kernel.bin of=$(IMG) seek=1 conv=notrunc # 2nd sector
	dd if=prg1.com of=$(IMG) seek=18 conv=notrunc # 1st side
	dd if=prg2.com of=$(IMG) seek=19 conv=notrunc
	dd if=prg3.com of=$(IMG) seek=20 conv=notrunc
	dd if=prg4.com of=$(IMG) seek=21 conv=notrunc
	dd if=prg5.com of=$(IMG) seek=22 conv=notrunc

programs: $(KERNEL) $(PROGS) $(DEBUG)

bootloader.bin: bootloader.asm
	$(AS) -fbin $< -o $@

kernel.bin: os.o kernel.o
	$(LD) $(LDFLAGS) -Ttext 0x7e00 $^ -o $@
# kernel.bin: os.o kernel.o
# 	$(LD) $(LDFLAGS) -T link.ld $^ -o $@

%.o : %.asm
	$(AS) $(ASFLAGS) $< -o $@
%.o : %.c
	$(CC) $(CCFLAGS) $< -o $@
%.com : %.asm
	$(AS) $< -o $@

# debug
%.s : %.c
	$(CC) $(CCFLAGS) -S $< -o $@

.PHONY : clean
clean :
	-rm -f *.o *.bin *.com *.elf *.s $(IMG) $(PROGS)