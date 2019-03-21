/*
* link.ld
*/

OUTPUT_FORMAT("elf32-i386")

ENTRY(_start)

SECTIONS
{
	. = 0x7E00; /* second sector of the disk */

	.text ALIGN (0x00) :
	{
		*(.text);
	}
	
	.rodata : {
		*(.rodata*);
	}

	.data : {
		*(.data*);
	}

	.bss : {
		*(COMMON);
		*(.bss*);
	}
}