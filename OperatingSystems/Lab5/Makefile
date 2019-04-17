CC = gcc
CCFLAGS := -c -march=i386 -nostdlib -ffreestanding -Wall -Wextra
CCFLAGS += -Iinclude -lgcc -fno-PIE -nostartfiles
CCFLAGS += -m16 -masm=intel -mpreferred-stack-boundary=2
# CCFLAGS += -m32 -masm=intel

AS = nasm
ASFLAGS = -felf32

LD = ld
LDFLAGS = -m elf_i386 -N --oformat binary

BUILD = build

IMG = mydisk.img
KERNEL = bootloader.bin kernel.bin
PROGS = prg1.com prg2.com prg3.com prg4.com prg5.com prg6.com

DEBUG = kernel.s

all: build programs
	-rm -f $(IMG)
	/sbin/mkfs.msdos -C $(IMG) 1440
	dd if=$(BUILD)/bootloader.bin of=$(IMG) conv=notrunc
	dd if=$(BUILD)/kernel.bin of=$(IMG) seek=1 conv=notrunc # 2nd sector
	dd if=$(BUILD)/prg1.com of=$(IMG) seek=36 conv=notrunc # 1st side
	dd if=$(BUILD)/prg2.com of=$(IMG) seek=38 conv=notrunc
	dd if=$(BUILD)/prg3.com of=$(IMG) seek=40 conv=notrunc
	dd if=$(BUILD)/prg4.com of=$(IMG) seek=42 conv=notrunc
	dd if=$(BUILD)/prg5.com of=$(IMG) seek=44 conv=notrunc
	dd if=$(BUILD)/prg6.com of=$(IMG) seek=46 conv=notrunc

build:
	-mkdir $(BUILD)

programs: $(KERNEL) $(PROGS) $(DEBUG)

bootloader.bin: bootloader.asm
	$(AS) -fbin $< -o $(BUILD)/$@

kernel.bin: os.o kernel.o
	$(LD) $(LDFLAGS) -Ttext 0x7e00 $(BUILD)/os.o $(BUILD)/kernel.o -o $(BUILD)/$@
# kernel.bin: os.o kernel.o
# 	$(LD) $(LDFLAGS) -T link.ld $^ -o $@

%.o : %.asm
	$(AS) $(ASFLAGS) $< -o $(BUILD)/$@
%.o : %.c
	$(CC) $(CCFLAGS) $< -o $(BUILD)/$@
%.com : %.asm
	$(AS) $< -o $(BUILD)/$@

# debug
%.s : %.c
	$(CC) $(CCFLAGS) -S $< -o $@

.PHONY : clean
clean :
	-rm -rf $(BUILD)
	-rm -f *.o *.bin *.com *.elf *.s $(IMG) $(PROGS)