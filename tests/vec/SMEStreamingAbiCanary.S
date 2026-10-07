// Ordinary AAPCS64 caller. Scalars below are ABI canaries, not Vec storage.
// Save the caller's d8--d15, seed all eight lower halves, call the probe,
// record the bits, then restore the original registers before returning.
.text
.p2align 2
.global vecops_streaming_abi_canary
.type vecops_streaming_abi_canary, %function
vecops_streaming_abi_canary:
  stp x29, x30, [sp, #-112]!
  mov x29, sp
  stp x19, x20, [sp, #16]
  stp x21, x22, [sp, #32]
  stp d8, d9, [sp, #48]
  stp d10, d11, [sp, #64]
  stp d12, d13, [sp, #80]
  stp d14, d15, [sp, #96]
  mov x19, x0
  mov x20, x1
  mov x9, #0x0808080808080808
  fmov d8, x9
  movz x9, #0x0909
  movk x9, #0x0909, lsl #16
  movk x9, #0x0909, lsl #32
  movk x9, #0x0909, lsl #48
  fmov d9, x9
  movz x9, #0x0a0a
  movk x9, #0x0a0a, lsl #16
  movk x9, #0x0a0a, lsl #32
  movk x9, #0x0a0a, lsl #48
  fmov d10, x9
  movz x9, #0x0b0b
  movk x9, #0x0b0b, lsl #16
  movk x9, #0x0b0b, lsl #32
  movk x9, #0x0b0b, lsl #48
  fmov d11, x9
  mov x9, #0x0c0c0c0c0c0c0c0c
  fmov d12, x9
  movz x9, #0x0d0d
  movk x9, #0x0d0d, lsl #16
  movk x9, #0x0d0d, lsl #32
  movk x9, #0x0d0d, lsl #48
  fmov d13, x9
  mov x9, #0x0e0e0e0e0e0e0e0e
  fmov d14, x9
  mov x9, #0x0f0f0f0f0f0f0f0f
  fmov d15, x9
  blr x19
  mov w21, w0
  fmov x9, d8
  str x9, [x20]
  fmov x9, d9
  str x9, [x20, #8]
  fmov x9, d10
  str x9, [x20, #16]
  fmov x9, d11
  str x9, [x20, #24]
  fmov x9, d12
  str x9, [x20, #32]
  fmov x9, d13
  str x9, [x20, #40]
  fmov x9, d14
  str x9, [x20, #48]
  fmov x9, d15
  str x9, [x20, #56]
  mov w0, w21
  ldp d14, d15, [sp, #96]
  ldp d12, d13, [sp, #80]
  ldp d10, d11, [sp, #64]
  ldp d8, d9, [sp, #48]
  ldp x21, x22, [sp, #32]
  ldp x19, x20, [sp, #16]
  ldp x29, x30, [sp], #112
  ret
.size vecops_streaming_abi_canary, .-vecops_streaming_abi_canary
.section .note.GNU-stack,"",%progbits
