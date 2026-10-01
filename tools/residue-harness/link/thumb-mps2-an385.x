/* QEMU mps2-an385 (Cortex-M3): vector table at 0x0, data RAM at 0x2000_0000.
   QEMU loads every segment at its address and starts RAM zeroed, so no
   startup copy or clear is needed. */
ENTRY(reset)

MEMORY
{
  CODE (rx) : ORIGIN = 0x00000000, LENGTH = 4M
  RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

SECTIONS
{
  .vectors ORIGIN(CODE) : { KEEP(*(.vectors)) } > CODE
  .text : { *(.text .text.*) } > CODE
  .rodata : ALIGN(16) { *(.rodata .rodata.*) } > CODE
  .ARM.exidx : { *(.ARM.exidx .ARM.exidx.*) } > CODE
  .data : ALIGN(16) { *(.data .data.*) } > RAM
  .bss : ALIGN(16) { *(.bss .bss.*) *(COMMON) } > RAM
  .stack (NOLOAD) : ALIGN(16)
  {
    __stack_bottom = .;
    . += 1M;
    __stack_top = .;
  } > RAM
}
