/* QEMU riscv32 `virt` with -bios none: RAM at 0x8000_0000, entry `_start`. */
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M
}

SECTIONS
{
  .text : { KEEP(*(.text.start)) *(.text .text.*) } > RAM
  .rodata : ALIGN(16) { *(.rodata .rodata.*) *(.srodata .srodata.*) } > RAM
  .data : ALIGN(16) { *(.data .data.*) *(.sdata .sdata.*) } > RAM
  .bss : ALIGN(16) { *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON) } > RAM
  .stack (NOLOAD) : ALIGN(16)
  {
    __stack_bottom = .;
    . += 1M;
    __stack_top = .;
  } > RAM
}
