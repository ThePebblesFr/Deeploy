
/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(BatchNorm_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                      Size     VMA      Type
  0                           00000000 00000000 
  1 .strtab                   0000013e 00000000 
  2 .text                     00000000 00000000 TEXT
  3 .sdata                    00000008 00000000 DATA
  4 .text.BatchNorm_fp32      000000e8 00000000 TEXT
  5 .rela.text.BatchNorm_fp32 00000030 00000000 
  6 .debug_loclists           0000026c 00000000 DEBUG
  7 .debug_abbrev             0000008c 00000000 DEBUG
  8 .debug_info               00000136 00000000 DEBUG
  9 .rela.debug_info          000000e4 00000000 
 10 .debug_str_offsets        00000078 00000000 DEBUG
 11 .rela.debug_str_offsets   00000150 00000000 
 12 .debug_str                00000184 00000000 DEBUG
 13 .debug_addr               00000018 00000000 DEBUG
 14 .rela.debug_addr          00000030 00000000 
 15 .comment                  00000073 00000000 
 16 .note.GNU-stack           00000000 00000000 
 17 .riscv.attributes         00000030 00000000 
 18 .debug_frame              00000030 00000000 DEBUG
 19 .rela.debug_frame         00000060 00000000 
 20 .debug_line               00000142 00000000 DEBUG
 21 .rela.debug_line          000002a0 00000000 
 22 .debug_line_str           000000e9 00000000 DEBUG
 23 .llvm_addrsig             00000000 00000000 
 24 .symtab                   000004e0 00000000 

Disassembly of section .text.BatchNorm_fp32:

00000000 <BatchNorm_fp32>:
;                     int L) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
       8: 23 24 91 00  	sw	s1, 8(sp)
       c: 23 22 21 01  	sw	s2, 4(sp)
      10: 83 22 01 01  	lw	t0, 16(sp)
;   for (int c = 0; c < C; ++c) {
      14: 13 a3 18 00  	slti	t1, a7, 1
      18: 93 23 18 00  	slti	t2, a6, 1
      1c: 33 63 73 00  	or	t1, t1, t2
      20: 93 a3 12 00  	slti	t2, t0, 1
      24: 33 63 73 00  	or	t1, t1, t2
      28: 63 16 03 0a  	bnez	t1, 0xd4 <BatchNorm_fp32+0xd4>
      2c: b7 03 00 00  	lui	t2, 0
      30: 07 a0 03 00  	flw	ft0, 0(t2)
      34: b7 03 00 00  	lui	t2, 0
      38: 87 a0 03 00  	flw	ft1, 0(t2)
;   for (int c = 0; c < C; ++c) {
      3c: 93 93 22 00  	slli	t2, t0, 2
      40: 33 8e 12 03  	mul	t3, t0, a7
      44: 13 1e 2e 00  	slli	t3, t3, 2
      48: 93 0e 00 00  	li	t4, 0
;     float32_t c_mean = mean[c];
      4c: 13 1f 23 00  	slli	t5, t1, 2
      50: b3 8f e6 01  	add	t6, a3, t5
      54: 07 a1 0f 00  	flw	ft2, 0(t6)
;     float32_t c_var = var[c];
      58: b3 0f e7 01  	add	t6, a4, t5
      5c: 87 a2 0f 00  	flw	ft5, 0(t6)
;     float32_t c_gamma = gamma[c];
      60: b3 8f e5 01  	add	t6, a1, t5
      64: 87 a1 0f 00  	flw	ft3, 0(t6)
;     float32_t c_beta = beta[c];
      68: 33 0f e6 01  	add	t5, a2, t5
      6c: 07 22 0f 00  	flw	ft4, 0(t5)
;     float32_t denom = sqrtf(c_var + epsilon);
      70: d3 f2 02 00  	fadd.s	ft5, ft5, ft0
      74: d3 f2 02 58  	fsqrt.s	ft5, ft5
      78: d3 f2 50 18  	fdiv.s	ft5, ft1, ft5
      7c: 13 0f 05 00  	mv	t5, a0
      80: 93 8f 07 00  	mv	t6, a5
      84: fb 40 e8 01  	<unknown>
      88: 13 04 0f 00  	mv	s0, t5
      8c: 93 84 0f 00  	mv	s1, t6
      90: 13 89 02 00  	mv	s2, t0
      94: 7b c0 02 01  	<unknown>
;         float32_t x = input[index];
      98: 07 23 04 00  	flw	ft6, 0(s0)
;         float32_t norm = (x - c_mean) / denom;
      9c: 53 73 23 08  	fsub.s	ft6, ft6, ft2
;         output[index] = c_gamma * norm + c_beta;
      a0: 53 73 33 10  	fmul.s	ft6, ft6, ft3
      a4: 43 73 53 20  	fmadd.s	ft6, ft6, ft5, ft4
      a8: 27 a0 64 00  	fsw	ft6, 0(s1)
;       for (int l = 0; l < L; ++l) {
      ac: 13 09 f9 ff  	addi	s2, s2, -1
      b0: 93 84 44 00  	addi	s1, s1, 4
      b4: 13 04 44 00  	addi	s0, s0, 4
;     for (int n = 0; n < N; ++n) {
      b8: 93 8e 1e 00  	addi	t4, t4, 1
      bc: b3 8f cf 01  	add	t6, t6, t3
      c0: 33 0f cf 01  	add	t5, t5, t3
;   for (int c = 0; c < C; ++c) {
      c4: 13 03 13 00  	addi	t1, t1, 1
      c8: b3 87 77 00  	add	a5, a5, t2
      cc: 33 05 75 00  	add	a0, a0, t2
      d0: e3 1c 13 f7  	bne	t1, a7, 0x48 <BatchNorm_fp32+0x48>
; }
      d4: 03 24 c1 00  	lw	s0, 12(sp)
      d8: 83 24 81 00  	lw	s1, 8(sp)
      dc: 03 29 41 00  	lw	s2, 4(sp)
      e0: 13 01 01 01  	addi	sp, sp, 16
      e4: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(ConvTranspose1d_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                       Size     VMA      Type
  0                            00000000 00000000 
  1 .strtab                    0000013c 00000000 
  2 .text                      00000000 00000000 TEXT
  3 .text.ConvTranspose1d_fp32 000001e4 00000000 TEXT
  4 .debug_loclists            000004fd 00000000 DEBUG
  5 .debug_abbrev              00000095 00000000 DEBUG
  6 .debug_info                0000016d 00000000 DEBUG
  7 .rela.debug_info           00000138 00000000 
  8 .debug_rnglists            0000002a 00000000 DEBUG
  9 .debug_str_offsets         00000088 00000000 DEBUG
 10 .rela.debug_str_offsets    00000180 00000000 
 11 .debug_str                 000001d6 00000000 DEBUG
 12 .debug_addr                00000028 00000000 DEBUG
 13 .rela.debug_addr           00000060 00000000 
 14 .comment                   00000073 00000000 
 15 .note.GNU-stack            00000000 00000000 
 16 .riscv.attributes          00000030 00000000 
 17 .debug_frame               00000040 00000000 DEBUG
 18 .rela.debug_frame          00000060 00000000 
 19 .debug_line                0000021e 00000000 DEBUG
 20 .rela.debug_line           000004a4 00000000 
 21 .debug_line_str            00000179 00000000 DEBUG
 22 .llvm_addrsig              00000000 00000000 
 23 .symtab                    000006a0 00000000 

Disassembly of section .text.ConvTranspose1d_fp32:

00000000 <ConvTranspose1d_fp32>:
;                           float32_t *output, uint32_t W_out) {
       0: 13 01 01 fd  	addi	sp, sp, -48
       4: 23 26 81 02  	sw	s0, 44(sp)
       8: 23 24 91 02  	sw	s1, 40(sp)
       c: 23 22 21 03  	sw	s2, 36(sp)
      10: 23 20 31 03  	sw	s3, 32(sp)
      14: 23 2e 41 01  	sw	s4, 28(sp)
      18: 23 2c 51 01  	sw	s5, 24(sp)
      1c: 23 2a 61 01  	sw	s6, 20(sp)
      20: 23 28 71 01  	sw	s7, 16(sp)
      24: 23 26 81 01  	sw	s8, 12(sp)
      28: 23 24 91 01  	sw	s9, 8(sp)
      2c: 23 22 a1 01  	sw	s10, 4(sp)
      30: 23 20 b1 01  	sw	s11, 0(sp)
;   for (uint32_t c = 0; c < C_out; ++c) {
      34: 63 0c 07 16  	beqz	a4, 0x1ac <ConvTranspose1d_fp32+0x1ac>
      38: 83 22 81 03  	lw	t0, 56(sp)
      3c: 03 23 41 03  	lw	t1, 52(sp)
      40: 93 93 22 00  	slli	t2, t0, 2
;     for (uint32_t w = 0; w < W_out; ++w) {
      44: 63 88 02 02  	beqz	t0, 0x74 <ConvTranspose1d_fp32+0x74>
      48: 13 0e 00 00  	li	t3, 0
      4c: 93 0e 03 00  	mv	t4, t1
      50: fb 40 e7 00  	<unknown>
      54: 13 8f 0e 00  	mv	t5, t4
      58: 93 8f 02 00  	mv	t6, t0
      5c: 7b c0 42 00  	<unknown>
;     for (uint32_t w = 0; w < W_out; ++w) {
      60: 93 8f ff ff  	addi	t6, t6, -1
;       output[c * W_out + w] = 0.0f;
      64: 2b 22 0f 00  	<unknown>
;   for (uint32_t c = 0; c < C_out; ++c) {
      68: 13 0e 1e 00  	addi	t3, t3, 1
      6c: b3 8e 7e 00  	add	t4, t4, t2
;   for (uint32_t cout = 0; cout < C_out; ++cout) {
      70: 63 0e 07 12  	beqz	a4, 0x1ac <ConvTranspose1d_fp32+0x1ac>
      74: 03 2e 01 03  	lw	t3, 48(sp)
      78: b3 3e 50 00  	snez	t4, t0
      7c: 33 fe ce 01  	and	t3, t4, t3
;     for (uint32_t cin = 0; cin < C_in; ++cin) {
      80: 63 86 05 0e  	beqz	a1, 0x16c <ConvTranspose1d_fp32+0x16c>
      84: 93 0e 00 00  	li	t4, 0
      88: 13 bf 17 00  	seqz	t5, a5
      8c: 93 3f 16 00  	seqz	t6, a2
      90: 33 ef ef 01  	or	t5, t6, t5
;   for (uint32_t cout = 0; cout < C_out; ++cout) {
      94: 93 9f 27 00  	slli	t6, a5, 2
      98: 33 84 e7 02  	mul	s0, a5, a4
      9c: 13 14 24 00  	slli	s0, s0, 2
      a0: 93 14 28 00  	slli	s1, a6, 2
      a4: 6f 00 40 01  	j	0xb8 <ConvTranspose1d_fp32+0xb8>
      a8: 93 8e 1e 00  	addi	t4, t4, 1
      ac: b3 86 f6 01  	add	a3, a3, t6
      b0: 33 03 73 00  	add	t1, t1, t2
      b4: 63 8c ee 0e  	beq	t4, a4, 0x1ac <ConvTranspose1d_fp32+0x1ac>
;       for (uint32_t w_in = 0; w_in < W_in; ++w_in) {
      b8: 63 10 0f 08  	bnez	t5, 0x138 <ConvTranspose1d_fp32+0x138>
      bc: 13 09 00 00  	li	s2, 0
      c0: 93 89 06 00  	mv	s3, a3
      c4: 13 0a 00 00  	li	s4, 0
      c8: 93 0a 00 00  	li	s5, 0
      cc: 33 0b c9 02  	mul	s6, s2, a2
      d0: 93 0b 03 00  	mv	s7, t1
      d4: fb 40 a6 02  	<unknown>
;         float32_t val = input[cin * W_in + w_in];
      d8: 33 8c 6a 01  	add	s8, s5, s6
      dc: 13 1c 2c 00  	slli	s8, s8, 2
      e0: 33 0c 85 01  	add	s8, a0, s8
      e4: 07 20 0c 00  	flw	ft0, 0(s8)
      e8: 13 0c 0a 00  	mv	s8, s4
      ec: 93 8c 0b 00  	mv	s9, s7
      f0: 13 8d 09 00  	mv	s10, s3
      f4: 93 8d 07 00  	mv	s11, a5
      f8: 7b c0 27 01  	<unknown>
;           if (w_out < W_out) {
      fc: 63 7a 5c 00  	bgeu	s8, t0, 0x110 <ConvTranspose1d_fp32+0x110>
;             float32_t wgt = weight[cin * (C_out * K) + cout * K + k];
     100: 87 20 0d 00  	flw	ft1, 0(s10)
;             output[cout * W_out + w_out] += val * wgt;
     104: 07 a1 0c 00  	flw	ft2, 0(s9)
     108: c3 f0 00 10  	fmadd.s	ft1, ft1, ft0, ft2
     10c: 27 a0 1c 00  	fsw	ft1, 0(s9)
;         for (uint32_t k = 0; k < K; ++k) {
     110: 93 8d fd ff  	addi	s11, s11, -1
     114: 13 0d 4d 00  	addi	s10, s10, 4
     118: 93 8c 4c 00  	addi	s9, s9, 4
     11c: 13 0c 1c 00  	addi	s8, s8, 1
;       for (uint32_t w_in = 0; w_in < W_in; ++w_in) {
     120: 93 8a 1a 00  	addi	s5, s5, 1
     124: b3 8b 9b 00  	add	s7, s7, s1
     128: 33 0a 0a 01  	add	s4, s4, a6
;     for (uint32_t cin = 0; cin < C_in; ++cin) {
     12c: 13 09 19 00  	addi	s2, s2, 1
     130: b3 89 89 00  	add	s3, s3, s0
     134: e3 18 b9 f8  	bne	s2, a1, 0xc4 <ConvTranspose1d_fp32+0xc4>
;     if (has_bias) {
     138: e3 08 0e f6  	beqz	t3, 0xa8 <ConvTranspose1d_fp32+0xa8>
     13c: 13 99 2e 00  	slli	s2, t4, 2
     140: 33 89 28 01  	add	s2, a7, s2
     144: 93 09 03 00  	mv	s3, t1
     148: 13 8a 02 00  	mv	s4, t0
;       for (uint32_t w = 0; w < W_out; ++w) {
     14c: 7b c0 c2 00  	<unknown>
;         output[cout * W_out + w] += bias[cout];
     150: 07 20 09 00  	flw	ft0, 0(s2)
     154: 87 a0 09 00  	flw	ft1, 0(s3)
     158: 53 f0 00 00  	fadd.s	ft0, ft1, ft0
     15c: 27 a0 09 00  	fsw	ft0, 0(s3)
;       for (uint32_t w = 0; w < W_out; ++w) {
     160: 13 0a fa ff  	addi	s4, s4, -1
     164: 93 89 49 00  	addi	s3, s3, 4
     168: 6f f0 1f f4  	j	0xa8 <ConvTranspose1d_fp32+0xa8>
;     if (has_bias) {
     16c: 63 00 0e 04  	beqz	t3, 0x1ac <ConvTranspose1d_fp32+0x1ac>
     170: 13 05 00 00  	li	a0, 0
     174: fb 40 a7 01  	<unknown>
     178: 93 15 25 00  	slli	a1, a0, 2
     17c: b3 85 b8 00  	add	a1, a7, a1
     180: 13 06 03 00  	mv	a2, t1
     184: 93 86 02 00  	mv	a3, t0
     188: 7b c0 c2 00  	<unknown>
;         output[cout * W_out + w] += bias[cout];
     18c: 07 a0 05 00  	flw	ft0, 0(a1)
     190: 87 20 06 00  	flw	ft1, 0(a2)
     194: 53 f0 00 00  	fadd.s	ft0, ft1, ft0
     198: 27 20 06 00  	fsw	ft0, 0(a2)
;       for (uint32_t w = 0; w < W_out; ++w) {
     19c: 93 86 f6 ff  	addi	a3, a3, -1
     1a0: 13 06 46 00  	addi	a2, a2, 4
;   for (uint32_t cout = 0; cout < C_out; ++cout) {
     1a4: 13 05 15 00  	addi	a0, a0, 1
     1a8: 33 03 73 00  	add	t1, t1, t2
; }
     1ac: 03 24 c1 02  	lw	s0, 44(sp)
     1b0: 83 24 81 02  	lw	s1, 40(sp)
     1b4: 03 29 41 02  	lw	s2, 36(sp)
     1b8: 83 29 01 02  	lw	s3, 32(sp)
     1bc: 03 2a c1 01  	lw	s4, 28(sp)
     1c0: 83 2a 81 01  	lw	s5, 24(sp)
     1c4: 03 2b 41 01  	lw	s6, 20(sp)
     1c8: 83 2b 01 01  	lw	s7, 16(sp)
     1cc: 03 2c c1 00  	lw	s8, 12(sp)
     1d0: 83 2c 81 00  	lw	s9, 8(sp)
     1d4: 03 2d 41 00  	lw	s10, 4(sp)
     1d8: 83 2d 01 00  	lw	s11, 0(sp)
     1dc: 13 01 01 03  	addi	sp, sp, 48
     1e0: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Convolution_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                             Size     VMA      Type
  0                                  00000000 00000000 
  1 .strtab                          0000015a 00000000 
  2 .text                            00000000 00000000 TEXT
  3 .text.Conv2d_fp32_fp32_fp32_NCHW 00000538 00000000 TEXT
  4 .text.Conv1d_fp32_fp32_fp32      000002cc 00000000 TEXT
  5 .debug_loclists                  00001a2b 00000000 DEBUG
  6 .debug_abbrev                    00000093 00000000 DEBUG
  7 .debug_info                      000001fe 00000000 DEBUG
  8 .rela.debug_info                 00000078 00000000 
  9 .debug_rnglists                  0000008b 00000000 DEBUG
 10 .debug_str_offsets               000000bc 00000000 DEBUG
 11 .rela.debug_str_offsets          0000021c 00000000 
 12 .debug_str                       0000021c 00000000 DEBUG
 13 .debug_addr                      00000010 00000000 DEBUG
 14 .rela.debug_addr                 00000018 00000000 
 15 .comment                         00000073 00000000 
 16 .note.GNU-stack                  00000000 00000000 
 17 .riscv.attributes                00000030 00000000 
 18 .debug_frame                     00000070 00000000 DEBUG
 19 .rela.debug_frame                000000c0 00000000 
 20 .debug_line                      0000073e 00000000 DEBUG
 21 .rela.debug_line                 00001368 00000000 
 22 .debug_line_str                  00000171 00000000 DEBUG
 23 .llvm_addrsig                    00000000 00000000 
 24 .symtab                          000010f0 00000000 

Disassembly of section .text.Conv2d_fp32_fp32_fp32_NCHW:

00000000 <Conv2d_fp32_fp32_fp32_NCHW>:
;                                 float32_t *__restrict__ pDstC) {
       0: 13 01 01 fb  	addi	sp, sp, -80
       4: 23 26 11 04  	sw	ra, 76(sp)
       8: 23 24 81 04  	sw	s0, 72(sp)
       c: 23 22 91 04  	sw	s1, 68(sp)
      10: 23 20 21 05  	sw	s2, 64(sp)
      14: 23 2e 31 03  	sw	s3, 60(sp)
      18: 23 2c 41 03  	sw	s4, 56(sp)
      1c: 23 2a 51 03  	sw	s5, 52(sp)
      20: 23 28 61 03  	sw	s6, 48(sp)
      24: 23 26 71 03  	sw	s7, 44(sp)
      28: 23 24 81 03  	sw	s8, 40(sp)
      2c: 23 22 91 03  	sw	s9, 36(sp)
      30: 23 20 a1 03  	sw	s10, 32(sp)
      34: 23 2e b1 01  	sw	s11, 28(sp)
      38: 83 22 c1 05  	lw	t0, 92(sp)
      3c: 93 8e 08 00  	mv	t4, a7
      40: 13 8f 07 00  	mv	t5, a5
      44: 23 2c e1 00  	sw	a4, 24(sp)
      48: 23 26 a1 00  	sw	a0, 12(sp)
      4c: 13 f7 12 00  	andi	a4, t0, 1
      50: 83 29 01 05  	lw	s3, 80(sp)
      54: 83 22 01 06  	lw	t0, 96(sp)
      58: 03 2a 41 05  	lw	s4, 84(sp)
;   uint32_t H_out = (H_padded - P) / SP + 1;
      5c: 33 05 06 41  	sub	a0, a2, a6
      60: 33 53 35 03  	divu	t1, a0, s3
      64: 93 03 13 00  	addi	t2, t1, 1
;   uint32_t W_out = (W_padded - Q) / SQ + 1;
      68: 33 85 16 41  	sub	a0, a3, a7
      6c: 33 5e 45 03  	divu	t3, a0, s4
      70: 13 05 1e 00  	addi	a0, t3, 1
      74: 23 2a 61 00  	sw	t1, 20(sp)
      78: 23 28 71 00  	sw	t2, 16(sp)
;   if (has_bias) {
      7c: 63 0a 07 14  	beqz	a4, 0x1d0 <Conv2d_fp32_fp32_fp32_NCHW+0x1d0>
;     for (f = 0; f < F; ++f) {
      80: 63 0e 0f 46  	beqz	t5, 0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
      84: 13 b7 13 00  	seqz	a4, t2
      88: 93 37 15 00  	seqz	a5, a0
;       for (h = 0; h < H_out; ++h) {
      8c: 33 67 f7 00  	or	a4, a4, a5
      90: 63 16 07 46  	bnez	a4, 0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
      94: 03 24 81 05  	lw	s0, 88(sp)
;           for (c = 0; c < C; ++c) {
      98: 63 86 05 26  	beqz	a1, 0x304 <Conv2d_fp32_fp32_fp32_NCHW+0x304>
;             for (p = 0; p < P; ++p) {
      9c: 63 06 08 30  	beqz	a6, 0x3a8 <Conv2d_fp32_fp32_fp32_NCHW+0x3a8>
;               for (q = 0; q < Q; ++q) {
      a0: 63 8a 0e 3a  	beqz	t4, 0x454 <Conv2d_fp32_fp32_fp32_NCHW+0x454>
      a4: 93 0f 00 00  	li	t6, 0
;     for (f = 0; f < F; ++f) {
      a8: 33 87 0e 03  	mul	a4, t4, a6
      ac: b3 07 b7 02  	mul	a5, a4, a1
      b0: 93 97 27 00  	slli	a5, a5, 2
      b4: 23 20 f1 00  	sw	a5, 0(sp)
      b8: 93 14 27 00  	slli	s1, a4, 2
      bc: 13 99 2e 00  	slli	s2, t4, 2
      c0: 33 87 d9 02  	mul	a4, s3, a3
      c4: 93 19 27 00  	slli	s3, a4, 2
      c8: 13 1a 2a 00  	slli	s4, s4, 2
      cc: 33 86 c6 02  	mul	a2, a3, a2
      d0: 13 16 26 00  	slli	a2, a2, 2
      d4: 93 96 26 00  	slli	a3, a3, 2
      d8: 53 00 00 f0  	fmv.w.x	ft0, zero
;     for (f = 0; f < F; ++f) {
      dc: 23 22 e1 01  	sw	t5, 4(sp)
      e0: 23 24 81 00  	sw	s0, 8(sp)
      e4: 13 97 2f 00  	slli	a4, t6, 2
      e8: 33 07 e4 00  	add	a4, s0, a4
      ec: 87 20 07 00  	flw	ft1, 0(a4)
      f0: 13 07 00 00  	li	a4, 0
      f4: b3 8a 7f 02  	mul	s5, t6, t2
      f8: 83 28 c1 00  	lw	a7, 12(sp)
      fc: 93 07 00 00  	li	a5, 0
     100: 93 0b 07 00  	mv	s7, a4
     104: 33 07 57 01  	add	a4, a4, s5
     108: 33 0c a7 02  	mul	s8, a4, a0
     10c: 13 8b 08 00  	mv	s6, a7
     110: 93 0d 00 00  	li	s11, 0
     114: 13 8d 07 00  	mv	s10, a5
     118: 93 0c 0b 00  	mv	s9, s6
     11c: 03 27 81 01  	lw	a4, 24(sp)
     120: 53 01 00 20  	fmv.s	ft2, ft0
     124: 13 04 00 00  	li	s0, 0
     128: 93 80 0c 00  	mv	ra, s9
     12c: 93 03 07 00  	mv	t2, a4
     130: fb 40 a8 01  	<unknown>
     134: 93 87 00 00  	mv	a5, ra
     138: 13 83 03 00  	mv	t1, t2
     13c: 13 8f 0e 00  	mv	t5, t4
     140: 7b c0 ce 00  	<unknown>
;                 sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
     144: 87 a1 07 00  	flw	ft3, 0(a5)
;                        pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
     148: 07 22 03 00  	flw	ft4, 0(t1)
;                 sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
     14c: 43 71 32 10  	fmadd.s	ft2, ft4, ft3, ft2
;               for (q = 0; q < Q; ++q) {
     150: 13 0f ff ff  	addi	t5, t5, -1
     154: 13 03 43 00  	addi	t1, t1, 4
     158: 93 87 47 00  	addi	a5, a5, 4
;             for (p = 0; p < P; ++p) {
     15c: 13 04 14 00  	addi	s0, s0, 1
     160: b3 83 23 01  	add	t2, t2, s2
     164: b3 80 d0 00  	add	ra, ra, a3
;           for (c = 0; c < C; ++c) {
     168: 93 8d 1d 00  	addi	s11, s11, 1
     16c: 33 07 97 00  	add	a4, a4, s1
     170: b3 8c cc 00  	add	s9, s9, a2
     174: e3 98 bd fa  	bne	s11, a1, 0x124 <Conv2d_fp32_fp32_fp32_NCHW+0x124>
;           pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     178: 53 f1 20 00  	fadd.s	ft2, ft1, ft2
     17c: 33 07 8d 01  	add	a4, s10, s8
     180: 13 17 27 00  	slli	a4, a4, 2
     184: 33 87 e2 00  	add	a4, t0, a4
     188: 27 20 27 00  	fsw	ft2, 0(a4)
;         for (w = 0; w < W_out; ++w) {
     18c: 93 07 1d 00  	addi	a5, s10, 1
     190: 33 0b 4b 01  	add	s6, s6, s4
     194: e3 1e cd f7  	bne	s10, t3, 0x110 <Conv2d_fp32_fp32_fp32_NCHW+0x110>
;       for (h = 0; h < H_out; ++h) {
     198: 13 87 1b 00  	addi	a4, s7, 1
     19c: b3 88 38 01  	add	a7, a7, s3
     1a0: 83 27 41 01  	lw	a5, 20(sp)
     1a4: e3 9c fb f4  	bne	s7, a5, 0xfc <Conv2d_fp32_fp32_fp32_NCHW+0xfc>
;     for (f = 0; f < F; ++f) {
     1a8: 93 8f 1f 00  	addi	t6, t6, 1
     1ac: 03 27 81 01  	lw	a4, 24(sp)
     1b0: 83 27 01 00  	lw	a5, 0(sp)
     1b4: 33 07 f7 00  	add	a4, a4, a5
     1b8: 23 2c e1 00  	sw	a4, 24(sp)
     1bc: 03 2f 41 00  	lw	t5, 4(sp)
     1c0: 83 23 01 01  	lw	t2, 16(sp)
     1c4: 03 24 81 00  	lw	s0, 8(sp)
     1c8: e3 9e ef f1  	bne	t6, t5, 0xe4 <Conv2d_fp32_fp32_fp32_NCHW+0xe4>
     1cc: 6f 00 00 33  	j	0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
;     for (f = 0; f < F; ++f) {
     1d0: 63 06 0f 32  	beqz	t5, 0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
     1d4: 13 b7 13 00  	seqz	a4, t2
     1d8: 93 37 15 00  	seqz	a5, a0
;       for (h = 0; h < H_out; ++h) {
     1dc: 33 67 f7 00  	or	a4, a4, a5
     1e0: 63 1e 07 30  	bnez	a4, 0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
;           for (c = 0; c < C; ++c) {
     1e4: 63 8c 05 16  	beqz	a1, 0x35c <Conv2d_fp32_fp32_fp32_NCHW+0x35c>
;             for (p = 0; p < P; ++p) {
     1e8: 63 0e 08 20  	beqz	a6, 0x404 <Conv2d_fp32_fp32_fp32_NCHW+0x404>
;               for (q = 0; q < Q; ++q) {
     1ec: 63 82 0e 2c  	beqz	t4, 0x4b0 <Conv2d_fp32_fp32_fp32_NCHW+0x4b0>
     1f0: 93 08 00 00  	li	a7, 0
;     for (f = 0; f < F; ++f) {
     1f4: 33 87 0e 03  	mul	a4, t4, a6
     1f8: b3 07 b7 02  	mul	a5, a4, a1
     1fc: 93 97 27 00  	slli	a5, a5, 2
     200: 23 20 f1 00  	sw	a5, 0(sp)
     204: 13 14 27 00  	slli	s0, a4, 2
     208: 93 94 2e 00  	slli	s1, t4, 2
     20c: 33 87 d9 02  	mul	a4, s3, a3
     210: 13 19 27 00  	slli	s2, a4, 2
     214: 93 19 2a 00  	slli	s3, s4, 2
     218: 33 86 c6 02  	mul	a2, a3, a2
     21c: 13 16 26 00  	slli	a2, a2, 2
     220: 93 96 26 00  	slli	a3, a3, 2
     224: 53 00 00 f0  	fmv.w.x	ft0, zero
     228: 13 07 00 00  	li	a4, 0
     22c: 23 24 11 01  	sw	a7, 8(sp)
     230: 33 8a 78 02  	mul	s4, a7, t2
     234: 83 27 c1 00  	lw	a5, 12(sp)
     238: 93 08 00 00  	li	a7, 0
     23c: 13 0b 07 00  	mv	s6, a4
     240: 33 07 47 01  	add	a4, a4, s4
     244: b3 0b a7 02  	mul	s7, a4, a0
     248: 93 8f 07 00  	mv	t6, a5
     24c: 13 0d 00 00  	li	s10, 0
     250: 93 8c 08 00  	mv	s9, a7
     254: 93 8a 0f 00  	mv	s5, t6
     258: 03 23 81 01  	lw	t1, 24(sp)
     25c: d3 00 00 20  	fmv.s	ft1, ft0
     260: 13 07 00 00  	li	a4, 0
     264: 13 8c 0a 00  	mv	s8, s5
     268: 93 03 03 00  	mv	t2, t1
     26c: fb 40 a8 01  	<unknown>
     270: 93 0d 0c 00  	mv	s11, s8
     274: 93 80 03 00  	mv	ra, t2
     278: 93 88 0e 00  	mv	a7, t4
     27c: 7b c0 ce 00  	<unknown>
;                 sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
     280: 07 a1 0d 00  	flw	ft2, 0(s11)
;                        pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
     284: 87 a1 00 00  	flw	ft3, 0(ra)
;                 sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
     288: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;               for (q = 0; q < Q; ++q) {
     28c: 93 88 f8 ff  	addi	a7, a7, -1
     290: 93 80 40 00  	addi	ra, ra, 4
     294: 93 8d 4d 00  	addi	s11, s11, 4
;             for (p = 0; p < P; ++p) {
     298: 13 07 17 00  	addi	a4, a4, 1
     29c: b3 83 93 00  	add	t2, t2, s1
     2a0: 33 0c dc 00  	add	s8, s8, a3
;           for (c = 0; c < C; ++c) {
     2a4: 13 0d 1d 00  	addi	s10, s10, 1
     2a8: 33 03 83 00  	add	t1, t1, s0
     2ac: b3 8a ca 00  	add	s5, s5, a2
     2b0: e3 18 bd fa  	bne	s10, a1, 0x260 <Conv2d_fp32_fp32_fp32_NCHW+0x260>
;           pDstC[f * H_out * W_out + h * W_out + w] = sum;
     2b4: 33 87 7c 01  	add	a4, s9, s7
     2b8: 13 17 27 00  	slli	a4, a4, 2
     2bc: 33 87 e2 00  	add	a4, t0, a4
     2c0: 27 20 17 00  	fsw	ft1, 0(a4)
;         for (w = 0; w < W_out; ++w) {
     2c4: 93 88 1c 00  	addi	a7, s9, 1
     2c8: b3 8f 3f 01  	add	t6, t6, s3
     2cc: e3 90 cc f9  	bne	s9, t3, 0x24c <Conv2d_fp32_fp32_fp32_NCHW+0x24c>
;       for (h = 0; h < H_out; ++h) {
     2d0: 13 07 1b 00  	addi	a4, s6, 1
     2d4: b3 87 27 01  	add	a5, a5, s2
     2d8: 83 28 41 01  	lw	a7, 20(sp)
     2dc: e3 1e 1b f5  	bne	s6, a7, 0x238 <Conv2d_fp32_fp32_fp32_NCHW+0x238>
     2e0: 83 28 81 00  	lw	a7, 8(sp)
;     for (f = 0; f < F; ++f) {
     2e4: 93 88 18 00  	addi	a7, a7, 1
     2e8: 03 27 81 01  	lw	a4, 24(sp)
     2ec: 83 27 01 00  	lw	a5, 0(sp)
     2f0: 33 07 f7 00  	add	a4, a4, a5
     2f4: 23 2c e1 00  	sw	a4, 24(sp)
     2f8: 83 23 01 01  	lw	t2, 16(sp)
     2fc: e3 96 e8 f3  	bne	a7, t5, 0x228 <Conv2d_fp32_fp32_fp32_NCHW+0x228>
     300: 6f 00 c0 1f  	j	0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
;     for (f = 0; f < F; ++f) {
     304: fb 40 8f 02  	<unknown>
     308: 13 96 25 00  	slli	a2, a1, 2
     30c: 33 06 c4 00  	add	a2, s0, a2
     310: 07 20 06 00  	flw	ft0, 0(a2)
     314: 13 07 00 00  	li	a4, 0
     318: 33 86 75 02  	mul	a2, a1, t2
     31c: 93 07 00 00  	li	a5, 0
     320: 93 06 07 00  	mv	a3, a4
     324: 33 08 c7 00  	add	a6, a4, a2
     328: 13 07 f0 ff  	li	a4, -1
     32c: b3 08 ee 40  	sub	a7, t3, a4
     330: 33 08 a8 02  	mul	a6, a6, a0
     334: 7b c0 a8 00  	<unknown>
;           pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     338: b3 88 07 01  	add	a7, a5, a6
     33c: 93 98 28 00  	slli	a7, a7, 2
     340: b3 88 12 01  	add	a7, t0, a7
     344: 27 a0 08 00  	fsw	ft0, 0(a7)
;         for (w = 0; w < W_out; ++w) {
     348: 93 87 17 00  	addi	a5, a5, 1
;       for (h = 0; h < H_out; ++h) {
     34c: 13 87 16 00  	addi	a4, a3, 1
     350: e3 96 66 fc  	bne	a3, t1, 0x31c <Conv2d_fp32_fp32_fp32_NCHW+0x31c>
;     for (f = 0; f < F; ++f) {
     354: 93 85 15 00  	addi	a1, a1, 1
     358: 6f 00 40 1a  	j	0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
;     for (f = 0; f < F; ++f) {
     35c: fb 40 2f 02  	<unknown>
     360: 93 07 00 00  	li	a5, 0
     364: 33 86 75 02  	mul	a2, a1, t2
     368: 13 07 00 00  	li	a4, 0
     36c: 93 86 07 00  	mv	a3, a5
     370: 33 88 c7 00  	add	a6, a5, a2
     374: 93 07 f0 ff  	li	a5, -1
     378: b3 08 fe 40  	sub	a7, t3, a5
     37c: 33 08 a8 02  	mul	a6, a6, a0
     380: 7b c0 a8 00  	<unknown>
;           pDstC[f * H_out * W_out + h * W_out + w] = sum;
     384: b3 08 07 01  	add	a7, a4, a6
     388: 93 98 28 00  	slli	a7, a7, 2
     38c: b3 88 12 01  	add	a7, t0, a7
     390: 23 a0 08 00  	sw	zero, 0(a7)
;         for (w = 0; w < W_out; ++w) {
     394: 13 07 17 00  	addi	a4, a4, 1
;       for (h = 0; h < H_out; ++h) {
     398: 93 87 16 00  	addi	a5, a3, 1
     39c: e3 96 66 fc  	bne	a3, t1, 0x368 <Conv2d_fp32_fp32_fp32_NCHW+0x368>
;     for (f = 0; f < F; ++f) {
     3a0: 93 85 15 00  	addi	a1, a1, 1
     3a4: 6f 00 80 15  	j	0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
     3a8: 93 05 00 00  	li	a1, 0
;     for (f = 0; f < F; ++f) {
     3ac: fb 40 8f 02  	<unknown>
     3b0: 13 96 25 00  	slli	a2, a1, 2
     3b4: 33 06 c4 00  	add	a2, s0, a2
     3b8: 07 20 06 00  	flw	ft0, 0(a2)
     3bc: 13 07 00 00  	li	a4, 0
     3c0: 33 86 75 02  	mul	a2, a1, t2
     3c4: 93 07 00 00  	li	a5, 0
     3c8: 93 06 07 00  	mv	a3, a4
     3cc: 33 08 c7 00  	add	a6, a4, a2
     3d0: 13 07 f0 ff  	li	a4, -1
     3d4: b3 08 ee 40  	sub	a7, t3, a4
     3d8: 33 08 a8 02  	mul	a6, a6, a0
     3dc: 7b c0 a8 00  	<unknown>
;           pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     3e0: b3 88 07 01  	add	a7, a5, a6
     3e4: 93 98 28 00  	slli	a7, a7, 2
     3e8: b3 88 12 01  	add	a7, t0, a7
     3ec: 27 a0 08 00  	fsw	ft0, 0(a7)
;         for (w = 0; w < W_out; ++w) {
     3f0: 93 87 17 00  	addi	a5, a5, 1
;       for (h = 0; h < H_out; ++h) {
     3f4: 13 87 16 00  	addi	a4, a3, 1
     3f8: e3 96 66 fc  	bne	a3, t1, 0x3c4 <Conv2d_fp32_fp32_fp32_NCHW+0x3c4>
;     for (f = 0; f < F; ++f) {
     3fc: 93 85 15 00  	addi	a1, a1, 1
     400: 6f 00 c0 0f  	j	0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
     404: 93 05 00 00  	li	a1, 0
;     for (f = 0; f < F; ++f) {
     408: fb 40 2f 02  	<unknown>
     40c: 93 07 00 00  	li	a5, 0
     410: 33 86 75 02  	mul	a2, a1, t2
     414: 13 07 00 00  	li	a4, 0
     418: 93 86 07 00  	mv	a3, a5
     41c: 33 88 c7 00  	add	a6, a5, a2
     420: 93 07 f0 ff  	li	a5, -1
     424: b3 08 fe 40  	sub	a7, t3, a5
     428: 33 08 a8 02  	mul	a6, a6, a0
     42c: 7b c0 a8 00  	<unknown>
;           pDstC[f * H_out * W_out + h * W_out + w] = sum;
     430: b3 08 07 01  	add	a7, a4, a6
     434: 93 98 28 00  	slli	a7, a7, 2
     438: b3 88 12 01  	add	a7, t0, a7
     43c: 23 a0 08 00  	sw	zero, 0(a7)
;         for (w = 0; w < W_out; ++w) {
     440: 13 07 17 00  	addi	a4, a4, 1
;       for (h = 0; h < H_out; ++h) {
     444: 93 87 16 00  	addi	a5, a3, 1
     448: e3 96 66 fc  	bne	a3, t1, 0x414 <Conv2d_fp32_fp32_fp32_NCHW+0x414>
;     for (f = 0; f < F; ++f) {
     44c: 93 85 15 00  	addi	a1, a1, 1
     450: 6f 00 c0 0a  	j	0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
     454: 93 05 00 00  	li	a1, 0
;     for (f = 0; f < F; ++f) {
     458: fb 40 8f 02  	<unknown>
     45c: 13 96 25 00  	slli	a2, a1, 2
     460: 33 06 c4 00  	add	a2, s0, a2
     464: 07 20 06 00  	flw	ft0, 0(a2)
     468: 13 07 00 00  	li	a4, 0
     46c: 33 86 75 02  	mul	a2, a1, t2
     470: 93 07 00 00  	li	a5, 0
     474: 93 06 07 00  	mv	a3, a4
     478: 33 08 c7 00  	add	a6, a4, a2
     47c: 13 07 f0 ff  	li	a4, -1
     480: b3 08 ee 40  	sub	a7, t3, a4
     484: 33 08 a8 02  	mul	a6, a6, a0
     488: 7b c0 a8 00  	<unknown>
;           pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     48c: b3 88 07 01  	add	a7, a5, a6
     490: 93 98 28 00  	slli	a7, a7, 2
     494: b3 88 12 01  	add	a7, t0, a7
     498: 27 a0 08 00  	fsw	ft0, 0(a7)
;         for (w = 0; w < W_out; ++w) {
     49c: 93 87 17 00  	addi	a5, a5, 1
;       for (h = 0; h < H_out; ++h) {
     4a0: 13 87 16 00  	addi	a4, a3, 1
     4a4: e3 96 66 fc  	bne	a3, t1, 0x470 <Conv2d_fp32_fp32_fp32_NCHW+0x470>
;     for (f = 0; f < F; ++f) {
     4a8: 93 85 15 00  	addi	a1, a1, 1
     4ac: 6f 00 00 05  	j	0x4fc <Conv2d_fp32_fp32_fp32_NCHW+0x4fc>
     4b0: 93 05 00 00  	li	a1, 0
;     for (f = 0; f < F; ++f) {
     4b4: fb 40 2f 02  	<unknown>
     4b8: 93 07 00 00  	li	a5, 0
     4bc: 33 86 75 02  	mul	a2, a1, t2
     4c0: 13 07 00 00  	li	a4, 0
     4c4: 93 86 07 00  	mv	a3, a5
     4c8: 33 88 c7 00  	add	a6, a5, a2
     4cc: 93 07 f0 ff  	li	a5, -1
     4d0: b3 08 fe 40  	sub	a7, t3, a5
     4d4: 33 08 a8 02  	mul	a6, a6, a0
     4d8: 7b c0 a8 00  	<unknown>
;           pDstC[f * H_out * W_out + h * W_out + w] = sum;
     4dc: b3 08 07 01  	add	a7, a4, a6
     4e0: 93 98 28 00  	slli	a7, a7, 2
     4e4: b3 88 12 01  	add	a7, t0, a7
     4e8: 23 a0 08 00  	sw	zero, 0(a7)
;         for (w = 0; w < W_out; ++w) {
     4ec: 13 07 17 00  	addi	a4, a4, 1
;       for (h = 0; h < H_out; ++h) {
     4f0: 93 87 16 00  	addi	a5, a3, 1
     4f4: e3 96 66 fc  	bne	a3, t1, 0x4c0 <Conv2d_fp32_fp32_fp32_NCHW+0x4c0>
;     for (f = 0; f < F; ++f) {
     4f8: 93 85 15 00  	addi	a1, a1, 1
; }
     4fc: 83 20 c1 04  	lw	ra, 76(sp)
     500: 03 24 81 04  	lw	s0, 72(sp)
     504: 83 24 41 04  	lw	s1, 68(sp)
     508: 03 29 01 04  	lw	s2, 64(sp)
     50c: 83 29 c1 03  	lw	s3, 60(sp)
     510: 03 2a 81 03  	lw	s4, 56(sp)
     514: 83 2a 41 03  	lw	s5, 52(sp)
     518: 03 2b 01 03  	lw	s6, 48(sp)
     51c: 83 2b c1 02  	lw	s7, 44(sp)
     520: 03 2c 81 02  	lw	s8, 40(sp)
     524: 83 2c 41 02  	lw	s9, 36(sp)
     528: 03 2d 01 02  	lw	s10, 32(sp)
     52c: 83 2d c1 01  	lw	s11, 28(sp)
     530: 13 01 01 05  	addi	sp, sp, 80
     534: 67 80 00 00  	ret

Disassembly of section .text.Conv1d_fp32_fp32_fp32:

00000000 <Conv1d_fp32_fp32_fp32>:
;     uint32_t W_out) {
       0: 13 01 01 fd  	addi	sp, sp, -48
       4: 23 26 81 02  	sw	s0, 44(sp)
       8: 23 24 91 02  	sw	s1, 40(sp)
       c: 23 22 21 03  	sw	s2, 36(sp)
      10: 23 20 31 03  	sw	s3, 32(sp)
      14: 23 2e 41 01  	sw	s4, 28(sp)
      18: 23 2c 51 01  	sw	s5, 24(sp)
      1c: 23 2a 61 01  	sw	s6, 20(sp)
      20: 23 28 71 01  	sw	s7, 16(sp)
      24: 23 26 81 01  	sw	s8, 12(sp)
      28: 23 24 91 01  	sw	s9, 8(sp)
      2c: 23 22 a1 01  	sw	s10, 4(sp)
;   for (c_out = 0; c_out < C_out; ++c_out) {
      30: 63 04 07 26  	beqz	a4, 0x298 <Conv1d_fp32_fp32_fp32+0x298>
      34: 83 22 81 03  	lw	t0, 56(sp)
;     for (w_out = 0; w_out < W_out; ++w_out) {
      38: 63 80 02 26  	beqz	t0, 0x298 <Conv1d_fp32_fp32_fp32+0x298>
      3c: 03 23 41 03  	lw	t1, 52(sp)
      40: 83 23 01 03  	lw	t2, 48(sp)
;       for (c_in = 0; c_in < C_in; ++c_in) {
      44: 63 88 05 0c  	beqz	a1, 0x114 <Conv1d_fp32_fp32_fp32+0x114>
;         for (k = 0; k < K; ++k) {
      48: 63 88 07 10  	beqz	a5, 0x158 <Conv1d_fp32_fp32_fp32+0x158>
;       if (has_bias) {
      4c: 93 f3 13 00  	andi	t2, t2, 1
      50: 63 86 03 14  	beqz	t2, 0x19c <Conv1d_fp32_fp32_fp32+0x19c>
      54: 93 03 00 00  	li	t2, 0
;   for (c_out = 0; c_out < C_out; ++c_out) {
      58: 33 8e b7 02  	mul	t3, a5, a1
      5c: 13 1e 2e 00  	slli	t3, t3, 2
      60: 93 9e 27 00  	slli	t4, a5, 2
      64: 13 1f 28 00  	slli	t5, a6, 2
      68: 93 1f 26 00  	slli	t6, a2, 2
      6c: 53 00 00 f0  	fmv.w.x	ft0, zero
      70: 13 94 23 00  	slli	s0, t2, 2
      74: 33 84 88 00  	add	s0, a7, s0
      78: 87 20 04 00  	flw	ft1, 0(s0)
      7c: 13 04 00 00  	li	s0, 0
      80: 93 04 00 00  	li	s1, 0
      84: 33 89 53 02  	mul	s2, t2, t0
      88: 93 09 05 00  	mv	s3, a0
      8c: 13 0a 00 00  	li	s4, 0
      90: 93 8a 09 00  	mv	s5, s3
      94: 13 8b 06 00  	mv	s6, a3
      98: 53 01 00 20  	fmv.s	ft2, ft0
      9c: fb c0 05 02  	<unknown>
      a0: 93 0b 04 00  	mv	s7, s0
      a4: 13 8c 0a 00  	mv	s8, s5
      a8: 93 0c 0b 00  	mv	s9, s6
      ac: 13 8d 07 00  	mv	s10, a5
      b0: 7b c0 07 01  	<unknown>
;           if (w_in < W_in) {
      b4: 63 f8 cb 00  	bgeu	s7, a2, 0xc4 <Conv1d_fp32_fp32_fp32+0xc4>
;             sum += pSrcA[c_in * W_in + w_in] *
      b8: 87 21 0c 00  	flw	ft3, 0(s8)
;                    pSrcB[c_out * C_in * K + c_in * K + k];
      bc: 07 a2 0c 00  	flw	ft4, 0(s9)
;             sum += pSrcA[c_in * W_in + w_in] *
      c0: 43 71 32 10  	fmadd.s	ft2, ft4, ft3, ft2
;         for (k = 0; k < K; ++k) {
      c4: 13 0d fd ff  	addi	s10, s10, -1
      c8: 93 8c 4c 00  	addi	s9, s9, 4
      cc: 13 0c 4c 00  	addi	s8, s8, 4
      d0: 93 8b 1b 00  	addi	s7, s7, 1
;       for (c_in = 0; c_in < C_in; ++c_in) {
      d4: 13 0a 1a 00  	addi	s4, s4, 1
      d8: 33 0b db 01  	add	s6, s6, t4
      dc: b3 8a fa 01  	add	s5, s5, t6
;         sum += pSrcBias[c_out];
      e0: 53 f1 20 00  	fadd.s	ft2, ft1, ft2
;       pDstC[c_out * W_out + w_out] = sum;
      e4: 33 8a 24 01  	add	s4, s1, s2
      e8: 13 1a 2a 00  	slli	s4, s4, 2
      ec: 33 0a 43 01  	add	s4, t1, s4
      f0: 27 20 2a 00  	fsw	ft2, 0(s4)
;     for (w_out = 0; w_out < W_out; ++w_out) {
      f4: 93 84 14 00  	addi	s1, s1, 1
      f8: b3 89 e9 01  	add	s3, s3, t5
      fc: 33 04 04 01  	add	s0, s0, a6
     100: e3 96 54 f8  	bne	s1, t0, 0x8c <Conv1d_fp32_fp32_fp32+0x8c>
;   for (c_out = 0; c_out < C_out; ++c_out) {
     104: 93 83 13 00  	addi	t2, t2, 1
     108: b3 86 c6 01  	add	a3, a3, t3
     10c: e3 92 e3 f6  	bne	t2, a4, 0x70 <Conv1d_fp32_fp32_fp32+0x70>
     110: 6f 00 80 18  	j	0x298 <Conv1d_fp32_fp32_fp32+0x298>
;       if (has_bias) {
     114: 13 f5 13 00  	andi	a0, t2, 1
     118: 63 0a 05 12  	beqz	a0, 0x24c <Conv1d_fp32_fp32_fp32+0x24c>
     11c: 13 05 00 00  	li	a0, 0
;   for (c_out = 0; c_out < C_out; ++c_out) {
     120: 93 95 22 00  	slli	a1, t0, 2
     124: fb 40 67 01  	<unknown>
     128: 13 16 25 00  	slli	a2, a0, 2
     12c: 33 86 c8 00  	add	a2, a7, a2
     130: 07 20 06 00  	flw	ft0, 0(a2)
     134: 13 06 03 00  	mv	a2, t1
     138: 93 86 02 00  	mv	a3, t0
     13c: 7b c0 62 00  	<unknown>
;       pDstC[c_out * W_out + w_out] = sum;
     140: 27 20 06 00  	fsw	ft0, 0(a2)
;     for (w_out = 0; w_out < W_out; ++w_out) {
     144: 93 86 f6 ff  	addi	a3, a3, -1
     148: 13 06 46 00  	addi	a2, a2, 4
;   for (c_out = 0; c_out < C_out; ++c_out) {
     14c: 13 05 15 00  	addi	a0, a0, 1
     150: 33 03 b3 00  	add	t1, t1, a1
     154: 6f 00 40 14  	j	0x298 <Conv1d_fp32_fp32_fp32+0x298>
;       if (has_bias) {
     158: 13 f5 13 00  	andi	a0, t2, 1
     15c: 63 0c 05 10  	beqz	a0, 0x274 <Conv1d_fp32_fp32_fp32+0x274>
     160: 13 05 00 00  	li	a0, 0
;   for (c_out = 0; c_out < C_out; ++c_out) {
     164: 93 95 22 00  	slli	a1, t0, 2
     168: fb 40 67 01  	<unknown>
     16c: 13 16 25 00  	slli	a2, a0, 2
     170: 33 86 c8 00  	add	a2, a7, a2
     174: 07 20 06 00  	flw	ft0, 0(a2)
     178: 13 06 03 00  	mv	a2, t1
     17c: 93 86 02 00  	mv	a3, t0
     180: 7b c0 62 00  	<unknown>
;       pDstC[c_out * W_out + w_out] = sum;
     184: 27 20 06 00  	fsw	ft0, 0(a2)
;     for (w_out = 0; w_out < W_out; ++w_out) {
     188: 93 86 f6 ff  	addi	a3, a3, -1
     18c: 13 06 46 00  	addi	a2, a2, 4
;   for (c_out = 0; c_out < C_out; ++c_out) {
     190: 13 05 15 00  	addi	a0, a0, 1
     194: 33 03 b3 00  	add	t1, t1, a1
     198: 6f 00 00 10  	j	0x298 <Conv1d_fp32_fp32_fp32+0x298>
     19c: 93 08 00 00  	li	a7, 0
;   for (c_out = 0; c_out < C_out; ++c_out) {
     1a0: b3 83 b7 02  	mul	t2, a5, a1
     1a4: 93 93 23 00  	slli	t2, t2, 2
     1a8: 13 9e 27 00  	slli	t3, a5, 2
     1ac: 93 1e 28 00  	slli	t4, a6, 2
     1b0: 13 1f 26 00  	slli	t5, a2, 2
     1b4: 53 00 00 f0  	fmv.w.x	ft0, zero
     1b8: 93 0f 00 00  	li	t6, 0
     1bc: 13 04 00 00  	li	s0, 0
     1c0: b3 84 58 02  	mul	s1, a7, t0
     1c4: 13 09 05 00  	mv	s2, a0
     1c8: 93 09 00 00  	li	s3, 0
     1cc: 13 0a 09 00  	mv	s4, s2
     1d0: 93 8a 06 00  	mv	s5, a3
     1d4: d3 00 00 20  	fmv.s	ft1, ft0
     1d8: fb c0 05 02  	<unknown>
     1dc: 13 8b 0f 00  	mv	s6, t6
     1e0: 93 0b 0a 00  	mv	s7, s4
     1e4: 13 8c 0a 00  	mv	s8, s5
     1e8: 93 8c 07 00  	mv	s9, a5
     1ec: 7b c0 07 01  	<unknown>
;           if (w_in < W_in) {
     1f0: 63 78 cb 00  	bgeu	s6, a2, 0x200 <Conv1d_fp32_fp32_fp32+0x200>
;             sum += pSrcA[c_in * W_in + w_in] *
     1f4: 07 a1 0b 00  	flw	ft2, 0(s7)
;                    pSrcB[c_out * C_in * K + c_in * K + k];
     1f8: 87 21 0c 00  	flw	ft3, 0(s8)
;             sum += pSrcA[c_in * W_in + w_in] *
     1fc: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;         for (k = 0; k < K; ++k) {
     200: 93 8c fc ff  	addi	s9, s9, -1
     204: 13 0c 4c 00  	addi	s8, s8, 4
     208: 93 8b 4b 00  	addi	s7, s7, 4
     20c: 13 0b 1b 00  	addi	s6, s6, 1
;       for (c_in = 0; c_in < C_in; ++c_in) {
     210: 93 89 19 00  	addi	s3, s3, 1
     214: b3 8a ca 01  	add	s5, s5, t3
     218: 33 0a ea 01  	add	s4, s4, t5
;       pDstC[c_out * W_out + w_out] = sum;
     21c: b3 09 94 00  	add	s3, s0, s1
     220: 93 99 29 00  	slli	s3, s3, 2
     224: b3 09 33 01  	add	s3, t1, s3
     228: 27 a0 19 00  	fsw	ft1, 0(s3)
;     for (w_out = 0; w_out < W_out; ++w_out) {
     22c: 13 04 14 00  	addi	s0, s0, 1
     230: 33 09 d9 01  	add	s2, s2, t4
     234: b3 8f 0f 01  	add	t6, t6, a6
     238: e3 18 54 f8  	bne	s0, t0, 0x1c8 <Conv1d_fp32_fp32_fp32+0x1c8>
;   for (c_out = 0; c_out < C_out; ++c_out) {
     23c: 93 88 18 00  	addi	a7, a7, 1
     240: b3 86 76 00  	add	a3, a3, t2
     244: e3 9a e8 f6  	bne	a7, a4, 0x1b8 <Conv1d_fp32_fp32_fp32+0x1b8>
     248: 6f 00 00 05  	j	0x298 <Conv1d_fp32_fp32_fp32+0x298>
     24c: 93 95 22 00  	slli	a1, t0, 2
     250: fb 40 e7 00  	<unknown>
     254: 13 06 03 00  	mv	a2, t1
     258: 93 86 02 00  	mv	a3, t0
     25c: 7b c0 42 00  	<unknown>
;     for (w_out = 0; w_out < W_out; ++w_out) {
     260: 93 86 f6 ff  	addi	a3, a3, -1
;       pDstC[c_out * W_out + w_out] = sum;
     264: 2b 22 06 00  	<unknown>
;   for (c_out = 0; c_out < C_out; ++c_out) {
     268: 13 05 15 00  	addi	a0, a0, 1
     26c: 33 03 b3 00  	add	t1, t1, a1
     270: 6f 00 80 02  	j	0x298 <Conv1d_fp32_fp32_fp32+0x298>
     274: 93 95 22 00  	slli	a1, t0, 2
     278: fb 40 e7 00  	<unknown>
     27c: 13 06 03 00  	mv	a2, t1
     280: 93 86 02 00  	mv	a3, t0
     284: 7b c0 42 00  	<unknown>
;     for (w_out = 0; w_out < W_out; ++w_out) {
     288: 93 86 f6 ff  	addi	a3, a3, -1
;       pDstC[c_out * W_out + w_out] = sum;
     28c: 2b 22 06 00  	<unknown>
;   for (c_out = 0; c_out < C_out; ++c_out) {
     290: 13 05 15 00  	addi	a0, a0, 1
     294: 33 03 b3 00  	add	t1, t1, a1
; }
     298: 03 24 c1 02  	lw	s0, 44(sp)
     29c: 83 24 81 02  	lw	s1, 40(sp)
     2a0: 03 29 41 02  	lw	s2, 36(sp)
     2a4: 83 29 01 02  	lw	s3, 32(sp)
     2a8: 03 2a c1 01  	lw	s4, 28(sp)
     2ac: 83 2a 81 01  	lw	s5, 24(sp)
     2b0: 03 2b 41 01  	lw	s6, 20(sp)
     2b4: 83 2b 01 01  	lw	s7, 16(sp)
     2b8: 03 2c c1 00  	lw	s8, 12(sp)
     2bc: 83 2c 81 00  	lw	s9, 8(sp)
     2c0: 03 2d 41 00  	lw	s10, 4(sp)
     2c4: 13 01 01 03  	addi	sp, sp, 48
     2c8: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Convolution_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                        Size     VMA      Type
  0                             00000000 00000000 
  1 .strtab                     00000127 00000000 
  2 .text                       00000000 00000000 TEXT
  3 .text.Conv2d_s8_s8_s32_NCHW 000002d4 00000000 TEXT
  4 .debug_loclists             00000a23 00000000 DEBUG
  5 .debug_abbrev               0000007d 00000000 DEBUG
  6 .debug_info                 0000014f 00000000 DEBUG
  7 .rela.debug_info            0000006c 00000000 
  8 .debug_str_offsets          00000094 00000000 DEBUG
  9 .rela.debug_str_offsets     000001a4 00000000 
 10 .debug_str                  000001c1 00000000 DEBUG
 11 .debug_addr                 0000000c 00000000 DEBUG
 12 .rela.debug_addr            0000000c 00000000 
 13 .comment                    00000073 00000000 
 14 .note.GNU-stack             00000000 00000000 
 15 .riscv.attributes           00000030 00000000 
 16 .debug_frame                00000044 00000000 DEBUG
 17 .rela.debug_frame           00000060 00000000 
 18 .debug_line                 000002b5 00000000 DEBUG
 19 .rela.debug_line            000006c0 00000000 
 20 .debug_line_str             00000149 00000000 DEBUG
 21 .llvm_addrsig               00000000 00000000 
 22 .symtab                     00000770 00000000 

Disassembly of section .text.Conv2d_s8_s8_s32_NCHW:

00000000 <Conv2d_s8_s8_s32_NCHW>:
;                            int32_t output_offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 28 a1 00  	sw	a0, 16(sp)
;   for (f = 0; f < F; ++f) {
      3c: 63 8e 07 24  	beqz	a5, 0x298 <Conv2d_s8_s8_s32_NCHW+0x298>
      40: 93 82 08 00  	mv	t0, a7
      44: 93 0a 07 00  	mv	s5, a4
      48: 03 25 01 06  	lw	a0, 96(sp)
      4c: 83 2f 41 06  	lw	t6, 100(sp)
      50: 33 07 06 41  	sub	a4, a2, a6
      54: 33 53 a7 02  	divu	t1, a4, a0
      58: 13 0e 13 00  	addi	t3, t1, 1
      5c: 33 87 16 41  	sub	a4, a3, a7
      60: b3 53 f7 03  	divu	t2, a4, t6
      64: 13 84 13 00  	addi	s0, t2, 1
      68: 13 37 1e 00  	seqz	a4, t3
      6c: 93 38 14 00  	seqz	a7, s0
;     for (h = 0; h < H_out; ++h) {
      70: 33 67 17 01  	or	a4, a4, a7
      74: 63 12 07 22  	bnez	a4, 0x298 <Conv2d_s8_s8_s32_NCHW+0x298>
      78: 83 2e 01 07  	lw	t4, 112(sp)
      7c: 03 2f 81 06  	lw	t5, 104(sp)
;         for (c = 0; c < C; ++c) {
      80: 63 86 05 12  	beqz	a1, 0x1ac <Conv2d_s8_s8_s32_NCHW+0x1ac>
;           for (p = 0; p < P; ++p) {
      84: 63 0c 08 16  	beqz	a6, 0x1fc <Conv2d_s8_s8_s32_NCHW+0x1fc>
;             for (q = 0; q < Q; ++q) {
      88: 63 82 02 1c  	beqz	t0, 0x24c <Conv2d_s8_s8_s32_NCHW+0x24c>
      8c: 93 08 00 00  	li	a7, 0
      90: 83 24 c1 06  	lw	s1, 108(sp)
;   for (f = 0; f < F; ++f) {
      94: 33 89 02 03  	mul	s2, t0, a6
      98: 33 07 b9 02  	mul	a4, s2, a1
      9c: 23 22 e1 00  	sw	a4, 4(sp)
      a0: 33 05 d5 02  	mul	a0, a0, a3
      a4: 23 2c a1 00  	sw	a0, 24(sp)
      a8: 33 86 c6 02  	mul	a2, a3, a2
      ac: 23 26 f1 00  	sw	a5, 12(sp)
      b0: 23 20 61 02  	sw	t1, 32(sp)
      b4: 23 24 c1 01  	sw	t3, 8(sp)
      b8: 23 2e 81 00  	sw	s0, 28(sp)
      bc: 13 07 00 00  	li	a4, 0
      c0: 23 2a 11 01  	sw	a7, 20(sp)
      c4: 33 85 c8 03  	mul	a0, a7, t3
      c8: 23 22 a1 02  	sw	a0, 36(sp)
      cc: 03 2a 01 01  	lw	s4, 16(sp)
      d0: 13 05 00 00  	li	a0, 0
      d4: 93 07 07 00  	mv	a5, a4
      d8: 03 27 41 02  	lw	a4, 36(sp)
      dc: 23 24 f1 02  	sw	a5, 40(sp)
      e0: 33 87 e7 00  	add	a4, a5, a4
      e4: 33 0c 87 02  	mul	s8, a4, s0
      e8: 13 04 0a 00  	mv	s0, s4
      ec: 93 0d 00 00  	li	s11, 0
      f0: 93 00 00 00  	li	ra, 0
      f4: 13 0d 05 00  	mv	s10, a0
      f8: 13 05 04 00  	mv	a0, s0
      fc: 93 87 0a 00  	mv	a5, s5
     100: 13 03 00 00  	li	t1, 0
     104: 13 0b 05 00  	mv	s6, a0
     108: 93 89 07 00  	mv	s3, a5
     10c: fb 40 88 01  	<unknown>
     110: 93 0c 0b 00  	mv	s9, s6
     114: 13 87 09 00  	mv	a4, s3
     118: 93 88 02 00  	mv	a7, t0
     11c: 7b c0 a2 00  	<unknown>
;               sum += (pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
     120: 0b 8e 1c 00  	<unknown>
;                      pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
     124: 8b 0b 17 00  	<unknown>
;               sum += (pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
     128: 33 0e 9e 00  	add	t3, t3, s1
;             for (q = 0; q < Q; ++q) {
     12c: 93 88 f8 ff  	addi	a7, a7, -1
;               sum += (pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
     130: b3 0d 7e 43  	<unknown>
;           for (p = 0; p < P; ++p) {
     134: 13 03 13 00  	addi	t1, t1, 1
     138: b3 89 59 00  	add	s3, s3, t0
     13c: 33 0b db 00  	add	s6, s6, a3
;         for (c = 0; c < C; ++c) {
     140: 93 80 10 00  	addi	ra, ra, 1
     144: b3 87 27 01  	add	a5, a5, s2
     148: 33 05 c5 00  	add	a0, a0, a2
     14c: e3 9a b0 fa  	bne	ra, a1, 0x100 <Conv2d_s8_s8_s32_NCHW+0x100>
;         pDstC[f * H_out * W_out + h * W_out + w] = sum + output_offset;
     150: 33 85 dd 01  	add	a0, s11, t4
     154: 33 07 8d 01  	add	a4, s10, s8
     158: 13 17 27 00  	slli	a4, a4, 2
     15c: 33 07 ef 00  	add	a4, t5, a4
     160: 23 20 a7 00  	sw	a0, 0(a4)
;       for (w = 0; w < W_out; ++w) {
     164: 13 05 1d 00  	addi	a0, s10, 1
     168: 33 04 f4 01  	add	s0, s0, t6
     16c: e3 10 7d f8  	bne	s10, t2, 0xec <Conv2d_s8_s8_s32_NCHW+0xec>
     170: 83 27 81 02  	lw	a5, 40(sp)
;     for (h = 0; h < H_out; ++h) {
     174: 13 87 17 00  	addi	a4, a5, 1
     178: 03 25 81 01  	lw	a0, 24(sp)
     17c: 33 0a aa 00  	add	s4, s4, a0
     180: 03 23 01 02  	lw	t1, 32(sp)
     184: 03 24 c1 01  	lw	s0, 28(sp)
;     for (h = 0; h < H_out; ++h) {
     188: e3 94 67 f4  	bne	a5, t1, 0xd0 <Conv2d_s8_s8_s32_NCHW+0xd0>
     18c: 83 28 41 01  	lw	a7, 20(sp)
;   for (f = 0; f < F; ++f) {
     190: 93 88 18 00  	addi	a7, a7, 1
     194: 03 25 41 00  	lw	a0, 4(sp)
     198: b3 8a aa 00  	add	s5, s5, a0
     19c: 83 27 c1 00  	lw	a5, 12(sp)
     1a0: 03 2e 81 00  	lw	t3, 8(sp)
     1a4: e3 9c f8 f0  	bne	a7, a5, 0xbc <Conv2d_s8_s8_s32_NCHW+0xbc>
     1a8: 6f 00 00 0f  	j	0x298 <Conv2d_s8_s8_s32_NCHW+0x298>
     1ac: 13 05 00 00  	li	a0, 0
;   for (f = 0; f < F; ++f) {
     1b0: fb c0 27 02  	<unknown>
     1b4: 93 06 00 00  	li	a3, 0
     1b8: b3 05 c5 03  	mul	a1, a0, t3
     1bc: 13 07 00 00  	li	a4, 0
     1c0: 13 86 06 00  	mv	a2, a3
     1c4: b3 88 b6 00  	add	a7, a3, a1
     1c8: 93 06 f0 ff  	li	a3, -1
     1cc: 33 88 d3 40  	sub	a6, t2, a3
     1d0: b3 88 88 02  	mul	a7, a7, s0
     1d4: 7b 40 a8 00  	<unknown>
;         pDstC[f * H_out * W_out + h * W_out + w] = sum + output_offset;
     1d8: 33 08 17 01  	add	a6, a4, a7
     1dc: 13 18 28 00  	slli	a6, a6, 2
     1e0: 33 08 0f 01  	add	a6, t5, a6
     1e4: 23 20 d8 01  	sw	t4, 0(a6)
;       for (w = 0; w < W_out; ++w) {
     1e8: 13 07 17 00  	addi	a4, a4, 1
;     for (h = 0; h < H_out; ++h) {
     1ec: 93 06 16 00  	addi	a3, a2, 1
     1f0: e3 16 66 fc  	bne	a2, t1, 0x1bc <Conv2d_s8_s8_s32_NCHW+0x1bc>
;   for (f = 0; f < F; ++f) {
     1f4: 13 05 15 00  	addi	a0, a0, 1
     1f8: 6f 00 00 0a  	j	0x298 <Conv2d_s8_s8_s32_NCHW+0x298>
     1fc: 13 05 00 00  	li	a0, 0
;   for (f = 0; f < F; ++f) {
     200: fb c0 27 02  	<unknown>
     204: 93 06 00 00  	li	a3, 0
     208: b3 05 c5 03  	mul	a1, a0, t3
     20c: 13 07 00 00  	li	a4, 0
     210: 13 86 06 00  	mv	a2, a3
     214: b3 88 b6 00  	add	a7, a3, a1
     218: 93 06 f0 ff  	li	a3, -1
     21c: 33 88 d3 40  	sub	a6, t2, a3
     220: b3 88 88 02  	mul	a7, a7, s0
     224: 7b 40 a8 00  	<unknown>
;         pDstC[f * H_out * W_out + h * W_out + w] = sum + output_offset;
     228: 33 08 17 01  	add	a6, a4, a7
     22c: 13 18 28 00  	slli	a6, a6, 2
     230: 33 08 0f 01  	add	a6, t5, a6
     234: 23 20 d8 01  	sw	t4, 0(a6)
;       for (w = 0; w < W_out; ++w) {
     238: 13 07 17 00  	addi	a4, a4, 1
;     for (h = 0; h < H_out; ++h) {
     23c: 93 06 16 00  	addi	a3, a2, 1
     240: e3 16 66 fc  	bne	a2, t1, 0x20c <Conv2d_s8_s8_s32_NCHW+0x20c>
;   for (f = 0; f < F; ++f) {
     244: 13 05 15 00  	addi	a0, a0, 1
     248: 6f 00 00 05  	j	0x298 <Conv2d_s8_s8_s32_NCHW+0x298>
     24c: 13 05 00 00  	li	a0, 0
;   for (f = 0; f < F; ++f) {
     250: fb c0 27 02  	<unknown>
     254: 93 06 00 00  	li	a3, 0
     258: b3 05 c5 03  	mul	a1, a0, t3
     25c: 13 07 00 00  	li	a4, 0
     260: 13 86 06 00  	mv	a2, a3
     264: b3 88 b6 00  	add	a7, a3, a1
     268: 93 06 f0 ff  	li	a3, -1
     26c: 33 88 d3 40  	sub	a6, t2, a3
     270: b3 88 88 02  	mul	a7, a7, s0
     274: 7b 40 a8 00  	<unknown>
;         pDstC[f * H_out * W_out + h * W_out + w] = sum + output_offset;
     278: 33 08 17 01  	add	a6, a4, a7
     27c: 13 18 28 00  	slli	a6, a6, 2
     280: 33 08 0f 01  	add	a6, t5, a6
     284: 23 20 d8 01  	sw	t4, 0(a6)
;       for (w = 0; w < W_out; ++w) {
     288: 13 07 17 00  	addi	a4, a4, 1
;     for (h = 0; h < H_out; ++h) {
     28c: 93 06 16 00  	addi	a3, a2, 1
     290: e3 16 66 fc  	bne	a2, t1, 0x25c <Conv2d_s8_s8_s32_NCHW+0x25c>
;   for (f = 0; f < F; ++f) {
     294: 13 05 15 00  	addi	a0, a0, 1
; }
     298: 83 20 c1 05  	lw	ra, 92(sp)
     29c: 03 24 81 05  	lw	s0, 88(sp)
     2a0: 83 24 41 05  	lw	s1, 84(sp)
     2a4: 03 29 01 05  	lw	s2, 80(sp)
     2a8: 83 29 c1 04  	lw	s3, 76(sp)
     2ac: 03 2a 81 04  	lw	s4, 72(sp)
     2b0: 83 2a 41 04  	lw	s5, 68(sp)
     2b4: 03 2b 01 04  	lw	s6, 64(sp)
     2b8: 83 2b c1 03  	lw	s7, 60(sp)
     2bc: 03 2c 81 03  	lw	s8, 56(sp)
     2c0: 83 2c 41 03  	lw	s9, 52(sp)
     2c4: 03 2d 01 03  	lw	s10, 48(sp)
     2c8: 83 2d c1 02  	lw	s11, 44(sp)
     2cc: 13 01 01 06  	addi	sp, sp, 96
     2d0: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(DWConvolution_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                               Size     VMA      Type
  0                                    00000000 00000000 
  1 .strtab                            00000142 00000000 
  2 .text                              00000000 00000000 TEXT
  3 .text.DWConv2d_fp32_fp32_fp32_NCHW 00000464 00000000 TEXT
  4 .debug_loclists                    00000a2b 00000000 DEBUG
  5 .debug_abbrev                      00000086 00000000 DEBUG
  6 .debug_info                        00000157 00000000 DEBUG
  7 .rela.debug_info                   00000078 00000000 
  8 .debug_rnglists                    00000052 00000000 DEBUG
  9 .debug_str_offsets                 00000090 00000000 DEBUG
 10 .rela.debug_str_offsets            00000198 00000000 
 11 .debug_str                         000001d9 00000000 DEBUG
 12 .debug_addr                        0000000c 00000000 DEBUG
 13 .rela.debug_addr                   0000000c 00000000 
 14 .comment                           00000073 00000000 
 15 .note.GNU-stack                    00000000 00000000 
 16 .riscv.attributes                  00000030 00000000 
 17 .debug_frame                       00000044 00000000 DEBUG
 18 .rela.debug_frame                  00000060 00000000 
 19 .debug_line                        0000044d 00000000 DEBUG
 20 .rela.debug_line                   00000b04 00000000 
 21 .debug_line_str                    00000175 00000000 DEBUG
 22 .llvm_addrsig                      00000000 00000000 
 23 .symtab                            00000a40 00000000 

Disassembly of section .text.DWConv2d_fp32_fp32_fp32_NCHW:

00000000 <DWConv2d_fp32_fp32_fp32_NCHW>:
;     float32_t *__restrict__ pDstC) {
       0: 13 01 01 fb  	addi	sp, sp, -80
       4: 23 26 11 04  	sw	ra, 76(sp)
       8: 23 24 81 04  	sw	s0, 72(sp)
       c: 23 22 91 04  	sw	s1, 68(sp)
      10: 23 20 21 05  	sw	s2, 64(sp)
      14: 23 2e 31 03  	sw	s3, 60(sp)
      18: 23 2c 41 03  	sw	s4, 56(sp)
      1c: 23 2a 51 03  	sw	s5, 52(sp)
      20: 23 28 61 03  	sw	s6, 48(sp)
      24: 23 26 71 03  	sw	s7, 44(sp)
      28: 23 24 81 03  	sw	s8, 40(sp)
      2c: 23 22 91 03  	sw	s9, 36(sp)
      30: 23 20 a1 03  	sw	s10, 32(sp)
      34: 23 2e b1 01  	sw	s11, 28(sp)
      38: 83 22 c1 05  	lw	t0, 92(sp)
      3c: 13 83 08 00  	mv	t1, a7
      40: 23 2c f1 00  	sw	a5, 24(sp)
      44: 13 09 06 00  	mv	s2, a2
      48: 93 f8 12 00  	andi	a7, t0, 1
      4c: 03 24 01 05  	lw	s0, 80(sp)
      50: 83 22 01 06  	lw	t0, 96(sp)
      54: 83 24 41 05  	lw	s1, 84(sp)
;   uint32_t H_out = (H_padded - P) / SP + 1;
      58: 33 06 06 41  	sub	a2, a2, a6
      5c: b3 57 86 02  	divu	a5, a2, s0
      60: 93 83 17 00  	addi	t2, a5, 1
;   uint32_t W_out = (W_padded - Q) / SQ + 1;
      64: 33 86 66 40  	sub	a2, a3, t1
      68: 33 5e 96 02  	divu	t3, a2, s1
      6c: 93 0e 1e 00  	addi	t4, t3, 1
      70: 23 2a b1 00  	sw	a1, 20(sp)
;   if (has_bias) {
      74: 63 8a 08 1e  	beqz	a7, 0x268 <DWConv2d_fp32_fp32_fp32_NCHW+0x268>
;     for (c = 0; c < C; ++c) {
      78: 63 82 05 20  	beqz	a1, 0x27c <DWConv2d_fp32_fp32_fp32_NCHW+0x27c>
      7c: 93 b8 13 00  	seqz	a7, t2
      80: 13 b6 1e 00  	seqz	a2, t4
;       for (h = 0; h < H_out; ++h) {
      84: 33 e6 c8 00  	or	a2, a7, a2
      88: 63 1a 06 1e  	bnez	a2, 0x27c <DWConv2d_fp32_fp32_fp32_NCHW+0x27c>
      8c: 13 0f 00 00  	li	t5, 0
      90: 83 2f 81 05  	lw	t6, 88(sp)
;     for (c = 0; c < C; ++c) {
      94: 33 86 26 03  	mul	a2, a3, s2
      98: 13 16 26 00  	slli	a2, a2, 2
      9c: 23 26 c1 00  	sw	a2, 12(sp)
      a0: 33 06 d4 02  	mul	a2, s0, a3
      a4: 13 14 26 00  	slli	s0, a2, 2
      a8: 93 94 24 00  	slli	s1, s1, 2
      ac: 93 96 26 00  	slli	a3, a3, 2
      b0: 53 00 00 f0  	fmv.w.x	ft0, zero
      b4: 6f 00 80 01  	j	0xcc <DWConv2d_fp32_fp32_fp32_NCHW+0xcc>
;     for (c = 0; c < C; ++c) {
      b8: 03 25 c1 00  	lw	a0, 12(sp)
      bc: 33 85 a5 00  	add	a0, a1, a0
      c0: 83 25 41 01  	lw	a1, 20(sp)
      c4: 03 2f 01 01  	lw	t5, 16(sp)
      c8: 63 0a bf 1a  	beq	t5, a1, 0x27c <DWConv2d_fp32_fp32_fp32_NCHW+0x27c>
      cc: 83 28 81 01  	lw	a7, 24(sp)
      d0: 33 06 1f 03  	mul	a2, t5, a7
      d4: 13 0f 1f 00  	addi	t5, t5, 1
      d8: 23 28 e1 01  	sw	t5, 16(sp)
      dc: b3 08 1f 03  	mul	a7, t5, a7
      e0: 33 59 b6 02  	divu	s2, a2, a1
      e4: b3 d9 b8 02  	divu	s3, a7, a1
      e8: 93 05 05 00  	mv	a1, a0
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
      ec: e3 76 39 fd  	bgeu	s2, s3, 0xb8 <DWConv2d_fp32_fp32_fp32_NCHW+0xb8>
      f0: 13 06 00 00  	li	a2, 0
;             for (p = 0; p < P; ++p) {
      f4: 63 02 08 0c  	beqz	a6, 0x1b8 <DWConv2d_fp32_fp32_fp32_NCHW+0x1b8>
;               for (q = 0; q < Q; ++q) {
      f8: 63 0c 03 10  	beqz	t1, 0x210 <DWConv2d_fp32_fp32_fp32_NCHW+0x210>
      fc: 13 8f 05 00  	mv	t5, a1
     100: 93 08 00 00  	li	a7, 0
     104: 93 0a 06 00  	mv	s5, a2
     108: 13 0a 0f 00  	mv	s4, t5
     10c: 93 8b 08 00  	mv	s7, a7
     110: 13 0c 09 00  	mv	s8, s2
     114: 93 0c 00 00  	li	s9, 0
     118: 33 0d 0c 03  	mul	s10, s8, a6
     11c: 13 0b 0a 00  	mv	s6, s4
     120: d3 00 00 20  	fmv.s	ft1, ft0
     124: fb 40 08 02  	<unknown>
     128: 93 00 00 00  	li	ra, 0
     12c: 33 86 ac 01  	add	a2, s9, s10
     130: 33 06 66 02  	mul	a2, a2, t1
     134: 93 0d 0b 00  	mv	s11, s6
     138: 93 08 03 00  	mv	a7, t1
     13c: 7b 40 03 01  	<unknown>
;                 sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
     140: 07 a1 0d 00  	flw	ft2, 0(s11)
;                        pSrcB[f * P * Q + p * Q + q];
     144: 33 85 c0 00  	add	a0, ra, a2
     148: 13 15 25 00  	slli	a0, a0, 2
     14c: 33 05 a7 00  	add	a0, a4, a0
     150: 87 21 05 00  	flw	ft3, 0(a0)
;                 sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
     154: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;               for (q = 0; q < Q; ++q) {
     158: 93 80 10 00  	addi	ra, ra, 1
     15c: 93 8d 4d 00  	addi	s11, s11, 4
;             for (p = 0; p < P; ++p) {
     160: 93 8c 1c 00  	addi	s9, s9, 1
     164: 33 0b db 00  	add	s6, s6, a3
;             pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     168: 13 15 2c 00  	slli	a0, s8, 2
     16c: 33 85 af 00  	add	a0, t6, a0
     170: 07 21 05 00  	flw	ft2, 0(a0)
     174: d3 70 11 00  	fadd.s	ft1, ft2, ft1
     178: 13 85 0a 00  	mv	a0, s5
     17c: 33 05 7c 42  	<unknown>
     180: 13 86 0b 00  	mv	a2, s7
     184: 33 06 d5 43  	<unknown>
     188: 13 15 26 00  	slli	a0, a2, 2
     18c: 33 85 a2 00  	add	a0, t0, a0
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     190: 13 0c 1c 00  	addi	s8, s8, 1
;             pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     194: 27 20 15 00  	fsw	ft1, 0(a0)
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     198: e3 6e 3c f7  	bltu	s8, s3, 0x114 <DWConv2d_fp32_fp32_fp32_NCHW+0x114>
;         for (w = 0; w < W_out; ++w) {
     19c: 93 88 1b 00  	addi	a7, s7, 1
     1a0: 33 0a 9a 00  	add	s4, s4, s1
     1a4: e3 94 cb f7  	bne	s7, t3, 0x10c <DWConv2d_fp32_fp32_fp32_NCHW+0x10c>
;       for (h = 0; h < H_out; ++h) {
     1a8: 13 86 1a 00  	addi	a2, s5, 1
     1ac: 33 0f 8f 00  	add	t5, t5, s0
     1b0: e3 98 fa f4  	bne	s5, a5, 0x100 <DWConv2d_fp32_fp32_fp32_NCHW+0x100>
     1b4: 6f f0 5f f0  	j	0xb8 <DWConv2d_fp32_fp32_fp32_NCHW+0xb8>
     1b8: 93 08 00 00  	li	a7, 0
     1bc: 13 0a 06 00  	mv	s4, a2
     1c0: 33 8f 29 41  	sub	t5, s3, s2
     1c4: 93 8a 08 00  	mv	s5, a7
     1c8: 13 06 09 00  	mv	a2, s2
     1cc: 7b 40 6f 01  	<unknown>
;             pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     1d0: 93 18 26 00  	slli	a7, a2, 2
     1d4: b3 88 1f 01  	add	a7, t6, a7
     1d8: 87 a0 08 00  	flw	ft1, 0(a7)
     1dc: 93 08 0a 00  	mv	a7, s4
     1e0: b3 08 76 42  	<unknown>
     1e4: 13 8f 0a 00  	mv	t5, s5
     1e8: 33 8f d8 43  	<unknown>
     1ec: 93 18 2f 00  	slli	a7, t5, 2
     1f0: b3 88 12 01  	add	a7, t0, a7
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     1f4: 13 06 16 00  	addi	a2, a2, 1
;             pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     1f8: 27 a0 18 00  	fsw	ft1, 0(a7)
;         for (w = 0; w < W_out; ++w) {
     1fc: 93 88 1a 00  	addi	a7, s5, 1
     200: e3 90 ca fd  	bne	s5, t3, 0x1c0 <DWConv2d_fp32_fp32_fp32_NCHW+0x1c0>
;       for (h = 0; h < H_out; ++h) {
     204: 13 06 1a 00  	addi	a2, s4, 1
     208: e3 18 fa fa  	bne	s4, a5, 0x1b8 <DWConv2d_fp32_fp32_fp32_NCHW+0x1b8>
     20c: 6f f0 df ea  	j	0xb8 <DWConv2d_fp32_fp32_fp32_NCHW+0xb8>
     210: 93 08 00 00  	li	a7, 0
     214: 13 0a 06 00  	mv	s4, a2
     218: 33 8f 29 41  	sub	t5, s3, s2
     21c: 93 8a 08 00  	mv	s5, a7
     220: 13 06 09 00  	mv	a2, s2
     224: 7b 40 6f 01  	<unknown>
;             pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     228: 93 18 26 00  	slli	a7, a2, 2
     22c: b3 88 1f 01  	add	a7, t6, a7
     230: 87 a0 08 00  	flw	ft1, 0(a7)
     234: 93 08 0a 00  	mv	a7, s4
     238: b3 08 76 42  	<unknown>
     23c: 13 8f 0a 00  	mv	t5, s5
     240: 33 8f d8 43  	<unknown>
     244: 93 18 2f 00  	slli	a7, t5, 2
     248: b3 88 12 01  	add	a7, t0, a7
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     24c: 13 06 16 00  	addi	a2, a2, 1
;             pDstC[f * H_out * W_out + h * W_out + w] = sum + pSrcBias[f];
     250: 27 a0 18 00  	fsw	ft1, 0(a7)
;         for (w = 0; w < W_out; ++w) {
     254: 93 88 1a 00  	addi	a7, s5, 1
     258: e3 90 ca fd  	bne	s5, t3, 0x218 <DWConv2d_fp32_fp32_fp32_NCHW+0x218>
;       for (h = 0; h < H_out; ++h) {
     25c: 13 06 1a 00  	addi	a2, s4, 1
     260: e3 18 fa fa  	bne	s4, a5, 0x210 <DWConv2d_fp32_fp32_fp32_NCHW+0x210>
     264: 6f f0 5f e5  	j	0xb8 <DWConv2d_fp32_fp32_fp32_NCHW+0xb8>
;     for (c = 0; c < C; ++c) {
     268: 63 8a 05 00  	beqz	a1, 0x27c <DWConv2d_fp32_fp32_fp32_NCHW+0x27c>
     26c: 93 b8 13 00  	seqz	a7, t2
     270: 13 b6 1e 00  	seqz	a2, t4
;       for (h = 0; h < H_out; ++h) {
     274: 33 e6 c8 00  	or	a2, a7, a2
     278: 63 00 06 04  	beqz	a2, 0x2b8 <DWConv2d_fp32_fp32_fp32_NCHW+0x2b8>
; }
     27c: 83 20 c1 04  	lw	ra, 76(sp)
     280: 03 24 81 04  	lw	s0, 72(sp)
     284: 83 24 41 04  	lw	s1, 68(sp)
     288: 03 29 01 04  	lw	s2, 64(sp)
     28c: 83 29 c1 03  	lw	s3, 60(sp)
     290: 03 2a 81 03  	lw	s4, 56(sp)
     294: 83 2a 41 03  	lw	s5, 52(sp)
     298: 03 2b 01 03  	lw	s6, 48(sp)
     29c: 83 2b c1 02  	lw	s7, 44(sp)
     2a0: 03 2c 81 02  	lw	s8, 40(sp)
     2a4: 83 2c 41 02  	lw	s9, 36(sp)
     2a8: 03 2d 01 02  	lw	s10, 32(sp)
     2ac: 83 2d c1 01  	lw	s11, 28(sp)
     2b0: 13 01 01 05  	addi	sp, sp, 80
     2b4: 67 80 00 00  	ret
;     for (c = 0; c < C; ++c) {
     2b8: b3 88 26 03  	mul	a7, a3, s2
     2bc: 93 98 28 00  	slli	a7, a7, 2
     2c0: 23 28 11 01  	sw	a7, 16(sp)
     2c4: b3 08 d4 02  	mul	a7, s0, a3
     2c8: 93 9f 28 00  	slli	t6, a7, 2
     2cc: 13 94 24 00  	slli	s0, s1, 2
     2d0: 93 96 26 00  	slli	a3, a3, 2
     2d4: 53 00 00 f0  	fmv.w.x	ft0, zero
     2d8: 6f 00 40 01  	j	0x2ec <DWConv2d_fp32_fp32_fp32_NCHW+0x2ec>
;     for (c = 0; c < C; ++c) {
     2dc: 03 25 01 01  	lw	a0, 16(sp)
     2e0: 33 85 a5 00  	add	a0, a1, a0
     2e4: 83 25 41 01  	lw	a1, 20(sp)
     2e8: e3 0a b6 f8  	beq	a2, a1, 0x27c <DWConv2d_fp32_fp32_fp32_NCHW+0x27c>
     2ec: 83 28 81 01  	lw	a7, 24(sp)
     2f0: 33 0f 16 03  	mul	t5, a2, a7
     2f4: 13 06 16 00  	addi	a2, a2, 1
     2f8: b3 08 16 03  	mul	a7, a2, a7
     2fc: b3 54 bf 02  	divu	s1, t5, a1
     300: 33 d9 b8 02  	divu	s2, a7, a1
     304: 93 05 05 00  	mv	a1, a0
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     308: e3 fa 24 fd  	bgeu	s1, s2, 0x2dc <DWConv2d_fp32_fp32_fp32_NCHW+0x2dc>
;             for (p = 0; p < P; ++p) {
     30c: 63 0c 08 0a  	beqz	a6, 0x3c4 <DWConv2d_fp32_fp32_fp32_NCHW+0x3c4>
;               for (q = 0; q < Q; ++q) {
     310: 63 02 03 10  	beqz	t1, 0x414 <DWConv2d_fp32_fp32_fp32_NCHW+0x414>
     314: 93 08 00 00  	li	a7, 0
     318: 13 85 05 00  	mv	a0, a1
     31c: 13 0f 00 00  	li	t5, 0
     320: 13 8a 08 00  	mv	s4, a7
     324: 93 09 05 00  	mv	s3, a0
     328: 13 0b 0f 00  	mv	s6, t5
     32c: 93 8b 04 00  	mv	s7, s1
     330: 13 0c 00 00  	li	s8, 0
     334: b3 8c 0b 03  	mul	s9, s7, a6
     338: 93 8a 09 00  	mv	s5, s3
     33c: d3 00 00 20  	fmv.s	ft1, ft0
     340: fb 40 08 02  	<unknown>
     344: 93 0d 00 00  	li	s11, 0
     348: b3 08 9c 01  	add	a7, s8, s9
     34c: b3 80 68 02  	mul	ra, a7, t1
     350: 13 8d 0a 00  	mv	s10, s5
     354: 13 0f 03 00  	mv	t5, t1
     358: 7b 40 03 01  	<unknown>
;                 sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
     35c: 07 21 0d 00  	flw	ft2, 0(s10)
;                        pSrcB[f * P * Q + p * Q + q];
     360: b3 88 1d 00  	add	a7, s11, ra
     364: 93 98 28 00  	slli	a7, a7, 2
     368: b3 08 17 01  	add	a7, a4, a7
     36c: 87 a1 08 00  	flw	ft3, 0(a7)
;                 sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
     370: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;               for (q = 0; q < Q; ++q) {
     374: 93 8d 1d 00  	addi	s11, s11, 1
     378: 13 0d 4d 00  	addi	s10, s10, 4
;             for (p = 0; p < P; ++p) {
     37c: 13 0c 1c 00  	addi	s8, s8, 1
     380: b3 8a da 00  	add	s5, s5, a3
;             pDstC[f * H_out * W_out + h * W_out + w] = sum;
     384: 93 08 0a 00  	mv	a7, s4
     388: b3 88 7b 42  	<unknown>
     38c: 13 0f 0b 00  	mv	t5, s6
     390: 33 8f d8 43  	<unknown>
     394: 93 18 2f 00  	slli	a7, t5, 2
     398: b3 88 12 01  	add	a7, t0, a7
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     39c: 93 8b 1b 00  	addi	s7, s7, 1
;             pDstC[f * H_out * W_out + h * W_out + w] = sum;
     3a0: 27 a0 18 00  	fsw	ft1, 0(a7)
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     3a4: e3 e6 2b f9  	bltu	s7, s2, 0x330 <DWConv2d_fp32_fp32_fp32_NCHW+0x330>
;         for (w = 0; w < W_out; ++w) {
     3a8: 13 0f 1b 00  	addi	t5, s6, 1
     3ac: b3 89 89 00  	add	s3, s3, s0
     3b0: e3 1c cb f7  	bne	s6, t3, 0x328 <DWConv2d_fp32_fp32_fp32_NCHW+0x328>
;       for (h = 0; h < H_out; ++h) {
     3b4: 93 08 1a 00  	addi	a7, s4, 1
     3b8: 33 05 f5 01  	add	a0, a0, t6
     3bc: e3 10 fa f6  	bne	s4, a5, 0x31c <DWConv2d_fp32_fp32_fp32_NCHW+0x31c>
     3c0: 6f f0 df f1  	j	0x2dc <DWConv2d_fp32_fp32_fp32_NCHW+0x2dc>
     3c4: 13 05 00 00  	li	a0, 0
     3c8: 93 08 00 00  	li	a7, 0
     3cc: 93 09 05 00  	mv	s3, a0
     3d0: 33 0f 99 40  	sub	t5, s2, s1
     3d4: 13 85 08 00  	mv	a0, a7
     3d8: 93 88 04 00  	mv	a7, s1
     3dc: 7b 40 0f 01  	<unknown>
;             pDstC[f * H_out * W_out + h * W_out + w] = sum;
     3e0: 13 8f 09 00  	mv	t5, s3
     3e4: 33 8f 78 42  	<unknown>
     3e8: 13 0a 05 00  	mv	s4, a0
     3ec: 33 0a df 43  	<unknown>
     3f0: 13 1f 2a 00  	slli	t5, s4, 2
     3f4: 33 8f e2 01  	add	t5, t0, t5
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     3f8: 93 88 18 00  	addi	a7, a7, 1
;             pDstC[f * H_out * W_out + h * W_out + w] = sum;
     3fc: 23 20 0f 00  	sw	zero, 0(t5)
;         for (w = 0; w < W_out; ++w) {
     400: 93 08 15 00  	addi	a7, a0, 1
     404: e3 16 c5 fd  	bne	a0, t3, 0x3d0 <DWConv2d_fp32_fp32_fp32_NCHW+0x3d0>
;       for (h = 0; h < H_out; ++h) {
     408: 13 85 19 00  	addi	a0, s3, 1
     40c: e3 9e f9 fa  	bne	s3, a5, 0x3c8 <DWConv2d_fp32_fp32_fp32_NCHW+0x3c8>
     410: 6f f0 df ec  	j	0x2dc <DWConv2d_fp32_fp32_fp32_NCHW+0x2dc>
     414: 13 05 00 00  	li	a0, 0
     418: 93 08 00 00  	li	a7, 0
     41c: 93 09 05 00  	mv	s3, a0
     420: 33 0f 99 40  	sub	t5, s2, s1
     424: 13 85 08 00  	mv	a0, a7
     428: 93 88 04 00  	mv	a7, s1
     42c: 7b 40 0f 01  	<unknown>
;             pDstC[f * H_out * W_out + h * W_out + w] = sum;
     430: 13 8f 09 00  	mv	t5, s3
     434: 33 8f 78 42  	<unknown>
     438: 13 0a 05 00  	mv	s4, a0
     43c: 33 0a df 43  	<unknown>
     440: 13 1f 2a 00  	slli	t5, s4, 2
     444: 33 8f e2 01  	add	t5, t0, t5
;           for (f = c * F / C; f < (c + 1) * F / C; ++f) {
     448: 93 88 18 00  	addi	a7, a7, 1
;             pDstC[f * H_out * W_out + h * W_out + w] = sum;
     44c: 23 20 0f 00  	sw	zero, 0(t5)
;         for (w = 0; w < W_out; ++w) {
     450: 93 08 15 00  	addi	a7, a0, 1
     454: e3 16 c5 fd  	bne	a0, t3, 0x420 <DWConv2d_fp32_fp32_fp32_NCHW+0x420>
;       for (h = 0; h < H_out; ++h) {
     458: 13 85 19 00  	addi	a0, s3, 1
     45c: e3 9e f9 fa  	bne	s3, a5, 0x418 <DWConv2d_fp32_fp32_fp32_NCHW+0x418>
     460: 6f f0 df e7  	j	0x2dc <DWConv2d_fp32_fp32_fp32_NCHW+0x2dc>

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(DWConvolution_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                          Size     VMA      Type
  0                               00000000 00000000 
  1 .strtab                       0000012b 00000000 
  2 .text                         00000000 00000000 TEXT
  3 .text.DWConv2d_s8_s8_s32_NCHW 0000023c 00000000 TEXT
  4 .debug_loclists               00000724 00000000 DEBUG
  5 .debug_abbrev                 0000007d 00000000 DEBUG
  6 .debug_info                   00000146 00000000 DEBUG
  7 .rela.debug_info              0000006c 00000000 
  8 .debug_str_offsets            00000090 00000000 DEBUG
  9 .rela.debug_str_offsets       00000198 00000000 
 10 .debug_str                    000001c3 00000000 DEBUG
 11 .debug_addr                   0000000c 00000000 DEBUG
 12 .rela.debug_addr              0000000c 00000000 
 13 .comment                      00000073 00000000 
 14 .note.GNU-stack               00000000 00000000 
 15 .riscv.attributes             00000030 00000000 
 16 .debug_frame                  00000044 00000000 DEBUG
 17 .rela.debug_frame             00000060 00000000 
 18 .debug_line                   0000022b 00000000 DEBUG
 19 .rela.debug_line              00000510 00000000 
 20 .debug_line_str               0000014d 00000000 DEBUG
 21 .llvm_addrsig                 00000000 00000000 
 22 .symtab                       00000640 00000000 

Disassembly of section .text.DWConv2d_s8_s8_s32_NCHW:

00000000 <DWConv2d_s8_s8_s32_NCHW>:
;                              int32_t output_offset) {
       0: 13 01 01 fb  	addi	sp, sp, -80
       4: 23 26 11 04  	sw	ra, 76(sp)
       8: 23 24 81 04  	sw	s0, 72(sp)
       c: 23 22 91 04  	sw	s1, 68(sp)
      10: 23 20 21 05  	sw	s2, 64(sp)
      14: 23 2e 31 03  	sw	s3, 60(sp)
      18: 23 2c 41 03  	sw	s4, 56(sp)
      1c: 23 2a 51 03  	sw	s5, 52(sp)
      20: 23 28 61 03  	sw	s6, 48(sp)
      24: 23 26 71 03  	sw	s7, 44(sp)
      28: 23 24 81 03  	sw	s8, 40(sp)
      2c: 23 22 91 03  	sw	s9, 36(sp)
      30: 23 20 a1 03  	sw	s10, 32(sp)
      34: 23 2e b1 01  	sw	s11, 28(sp)
;   for (c = 0; c < C; ++c) {
      38: 63 84 05 1c  	beqz	a1, 0x200 <DWConv2d_s8_s8_s32_NCHW+0x200>
      3c: 13 09 08 00  	mv	s2, a6
      40: 13 03 07 00  	mv	t1, a4
      44: 13 07 06 00  	mv	a4, a2
      48: 83 2f 01 05  	lw	t6, 80(sp)
      4c: 33 06 f6 40  	sub	a2, a2, a5
      50: b3 52 16 03  	divu	t0, a2, a7
      54: 93 89 12 00  	addi	s3, t0, 1
      58: 33 86 06 41  	sub	a2, a3, a6
      5c: b3 53 f6 03  	divu	t2, a2, t6
      60: 13 8e 13 00  	addi	t3, t2, 1
      64: 13 b6 19 00  	seqz	a2, s3
      68: 13 38 1e 00  	seqz	a6, t3
;     for (h = 0; h < H_out; ++h) {
      6c: 33 66 06 01  	or	a2, a2, a6
      70: 63 18 06 18  	bnez	a2, 0x200 <DWConv2d_s8_s8_s32_NCHW+0x200>
      74: 83 2e c1 05  	lw	t4, 92(sp)
      78: 03 2f 41 05  	lw	t5, 84(sp)
;         for (p = 0; p < P; ++p) {
      7c: 63 84 07 0e  	beqz	a5, 0x164 <DWConv2d_s8_s8_s32_NCHW+0x164>
;           for (q = 0; q < Q; ++q) {
      80: 63 0a 09 12  	beqz	s2, 0x1b4 <DWConv2d_s8_s8_s32_NCHW+0x1b4>
      84: 13 04 00 00  	li	s0, 0
      88: 83 24 81 05  	lw	s1, 88(sp)
;   for (c = 0; c < C; ++c) {
      8c: 33 06 f9 02  	mul	a2, s2, a5
      90: 23 26 c1 00  	sw	a2, 12(sp)
      94: 33 86 e6 02  	mul	a2, a3, a4
      98: 23 24 c1 00  	sw	a2, 8(sp)
      9c: b3 88 d8 02  	mul	a7, a7, a3
      a0: 23 2a b1 00  	sw	a1, 20(sp)
      a4: 23 28 31 01  	sw	s3, 16(sp)
      a8: 13 08 00 00  	li	a6, 0
      ac: b3 09 34 03  	mul	s3, s0, s3
      b0: 23 2c a1 00  	sw	a0, 24(sp)
      b4: 13 07 00 00  	li	a4, 0
      b8: 93 0a 08 00  	mv	s5, a6
      bc: 33 08 38 01  	add	a6, a6, s3
      c0: 33 0b c8 03  	mul	s6, a6, t3
      c4: 13 0a 05 00  	mv	s4, a0
      c8: 93 0c 00 00  	li	s9, 0
      cc: 13 0d 00 00  	li	s10, 0
      d0: 13 0c 07 00  	mv	s8, a4
      d4: 93 0b 0a 00  	mv	s7, s4
      d8: 13 07 03 00  	mv	a4, t1
      dc: fb c0 87 01  	<unknown>
      e0: 93 8d 0b 00  	mv	s11, s7
      e4: 93 00 07 00  	mv	ra, a4
      e8: 13 08 09 00  	mv	a6, s2
      ec: 7b 40 a9 00  	<unknown>
;             sum += (pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
      f0: 0b 86 1d 00  	<unknown>
;                    pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
      f4: 8b 85 10 00  	<unknown>
;             sum += (pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
      f8: 33 06 96 00  	add	a2, a2, s1
;           for (q = 0; q < Q; ++q) {
      fc: 13 08 f8 ff  	addi	a6, a6, -1
;             sum += (pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
     100: b3 0c b6 42  	<unknown>
;         for (p = 0; p < P; ++p) {
     104: 13 0d 1d 00  	addi	s10, s10, 1
     108: 33 07 27 01  	add	a4, a4, s2
     10c: b3 8b db 00  	add	s7, s7, a3
;         pDstC[c * H_out * W_out + h * W_out + w] = sum + output_offset;
     110: b3 85 dc 01  	add	a1, s9, t4
     114: 33 06 6c 01  	add	a2, s8, s6
     118: 13 16 26 00  	slli	a2, a2, 2
     11c: 33 06 cf 00  	add	a2, t5, a2
     120: 23 20 b6 00  	sw	a1, 0(a2)
;       for (w = 0; w < W_out; ++w) {
     124: 13 07 1c 00  	addi	a4, s8, 1
     128: 33 0a fa 01  	add	s4, s4, t6
     12c: e3 1e 7c f8  	bne	s8, t2, 0xc8 <DWConv2d_s8_s8_s32_NCHW+0xc8>
;     for (h = 0; h < H_out; ++h) {
     130: 13 88 1a 00  	addi	a6, s5, 1
     134: 33 05 15 01  	add	a0, a0, a7
     138: e3 9e 5a f6  	bne	s5, t0, 0xb4 <DWConv2d_s8_s8_s32_NCHW+0xb4>
;   for (c = 0; c < C; ++c) {
     13c: 13 04 14 00  	addi	s0, s0, 1
     140: 03 25 c1 00  	lw	a0, 12(sp)
     144: 33 03 a3 00  	add	t1, t1, a0
     148: 03 25 81 01  	lw	a0, 24(sp)
     14c: 83 25 81 00  	lw	a1, 8(sp)
     150: 33 05 b5 00  	add	a0, a0, a1
     154: 83 25 41 01  	lw	a1, 20(sp)
     158: 83 29 01 01  	lw	s3, 16(sp)
     15c: e3 16 b4 f4  	bne	s0, a1, 0xa8 <DWConv2d_s8_s8_s32_NCHW+0xa8>
     160: 6f 00 00 0a  	j	0x200 <DWConv2d_s8_s8_s32_NCHW+0x200>
     164: 13 05 00 00  	li	a0, 0
;   for (c = 0; c < C; ++c) {
     168: fb c0 25 02  	<unknown>
     16c: 13 07 00 00  	li	a4, 0
     170: 33 06 35 03  	mul	a2, a0, s3
     174: 93 07 00 00  	li	a5, 0
     178: 93 06 07 00  	mv	a3, a4
     17c: 33 08 c7 00  	add	a6, a4, a2
     180: 13 07 f0 ff  	li	a4, -1
     184: b3 88 e3 40  	sub	a7, t2, a4
     188: 33 08 c8 03  	mul	a6, a6, t3
     18c: 7b c0 a8 00  	<unknown>
;         pDstC[c * H_out * W_out + h * W_out + w] = sum + output_offset;
     190: b3 88 07 01  	add	a7, a5, a6
     194: 93 98 28 00  	slli	a7, a7, 2
     198: b3 08 1f 01  	add	a7, t5, a7
     19c: 23 a0 d8 01  	sw	t4, 0(a7)
;       for (w = 0; w < W_out; ++w) {
     1a0: 93 87 17 00  	addi	a5, a5, 1
;     for (h = 0; h < H_out; ++h) {
     1a4: 13 87 16 00  	addi	a4, a3, 1
     1a8: e3 96 56 fc  	bne	a3, t0, 0x174 <DWConv2d_s8_s8_s32_NCHW+0x174>
;   for (c = 0; c < C; ++c) {
     1ac: 13 05 15 00  	addi	a0, a0, 1
     1b0: 6f 00 00 05  	j	0x200 <DWConv2d_s8_s8_s32_NCHW+0x200>
     1b4: 13 05 00 00  	li	a0, 0
;   for (c = 0; c < C; ++c) {
     1b8: fb c0 25 02  	<unknown>
     1bc: 13 07 00 00  	li	a4, 0
     1c0: 33 06 35 03  	mul	a2, a0, s3
     1c4: 93 07 00 00  	li	a5, 0
     1c8: 93 06 07 00  	mv	a3, a4
     1cc: 33 08 c7 00  	add	a6, a4, a2
     1d0: 13 07 f0 ff  	li	a4, -1
     1d4: b3 88 e3 40  	sub	a7, t2, a4
     1d8: 33 08 c8 03  	mul	a6, a6, t3
     1dc: 7b c0 a8 00  	<unknown>
;         pDstC[c * H_out * W_out + h * W_out + w] = sum + output_offset;
     1e0: b3 88 07 01  	add	a7, a5, a6
     1e4: 93 98 28 00  	slli	a7, a7, 2
     1e8: b3 08 1f 01  	add	a7, t5, a7
     1ec: 23 a0 d8 01  	sw	t4, 0(a7)
;       for (w = 0; w < W_out; ++w) {
     1f0: 93 87 17 00  	addi	a5, a5, 1
;     for (h = 0; h < H_out; ++h) {
     1f4: 13 87 16 00  	addi	a4, a3, 1
     1f8: e3 96 56 fc  	bne	a3, t0, 0x1c4 <DWConv2d_s8_s8_s32_NCHW+0x1c4>
;   for (c = 0; c < C; ++c) {
     1fc: 13 05 15 00  	addi	a0, a0, 1
; }
     200: 83 20 c1 04  	lw	ra, 76(sp)
     204: 03 24 81 04  	lw	s0, 72(sp)
     208: 83 24 41 04  	lw	s1, 68(sp)
     20c: 03 29 01 04  	lw	s2, 64(sp)
     210: 83 29 c1 03  	lw	s3, 60(sp)
     214: 03 2a 81 03  	lw	s4, 56(sp)
     218: 83 2a 41 03  	lw	s5, 52(sp)
     21c: 03 2b 01 03  	lw	s6, 48(sp)
     220: 83 2b c1 02  	lw	s7, 44(sp)
     224: 03 2c 81 02  	lw	s8, 40(sp)
     228: 83 2c 41 02  	lw	s9, 36(sp)
     22c: 03 2d 01 02  	lw	s10, 32(sp)
     230: 83 2d c1 01  	lw	s11, 28(sp)
     234: 13 01 01 05  	addi	sp, sp, 80
     238: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Div_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                     Size     VMA      Type
  0                          00000000 00000000 
  1 .strtab                  0000011e 00000000 
  2 .text                    00000000 00000000 TEXT
  3 .text.Div_fp32_fp32_fp32 00000028 00000000 TEXT
  4 .debug_loclists          00000040 00000000 DEBUG
  5 .debug_abbrev            00000078 00000000 DEBUG
  6 .debug_info              0000008d 00000000 DEBUG
  7 .rela.debug_info         00000084 00000000 
  8 .debug_str_offsets       00000040 00000000 DEBUG
  9 .rela.debug_str_offsets  000000a8 00000000 
 10 .debug_str               00000154 00000000 DEBUG
 11 .debug_addr              0000000c 00000000 DEBUG
 12 .rela.debug_addr         0000000c 00000000 
 13 .comment                 00000073 00000000 
 14 .note.GNU-stack          00000000 00000000 
 15 .riscv.attributes        00000030 00000000 
 16 .debug_frame             00000024 00000000 DEBUG
 17 .rela.debug_frame        00000030 00000000 
 18 .debug_line              000000f2 00000000 DEBUG
 19 .rela.debug_line         00000144 00000000 
 20 .debug_line_str          00000161 00000000 DEBUG
 21 .llvm_addrsig            00000000 00000000 
 22 .symtab                  00000260 00000000 

Disassembly of section .text.Div_fp32_fp32_fp32:

00000000 <Div_fp32_fp32_fp32>:
;   for (int i = 0; i < size; i++) {
       0: 63 52 d0 02  	blez	a3, 0x24 <Div_fp32_fp32_fp32+0x24>
       4: 7b c0 e6 00  	<unknown>
;     data_out[i] = data_in_1[i] / data_in_2[i];
       8: 07 20 05 00  	flw	ft0, 0(a0)
       c: 87 a0 05 00  	flw	ft1, 0(a1)
      10: 53 70 10 18  	fdiv.s	ft0, ft0, ft1
      14: 27 20 06 00  	fsw	ft0, 0(a2)
;   for (int i = 0; i < size; i++) {
      18: 13 06 46 00  	addi	a2, a2, 4
      1c: 93 85 45 00  	addi	a1, a1, 4
      20: 13 05 45 00  	addi	a0, a0, 4
; }
      24: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Div_s32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 00000134 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text.Div_s32_s32       000001f4 00000000 TEXT
  4 .rela.text.Div_s32_s32  0000000c 00000000 
  5 .debug_loclists         00000151 00000000 DEBUG
  6 .debug_abbrev           00000090 00000000 DEBUG
  7 .debug_info             00000127 00000000 DEBUG
  8 .rela.debug_info        000000a8 00000000 
  9 .debug_rnglists         00000019 00000000 DEBUG
 10 .debug_str_offsets      0000007c 00000000 DEBUG
 11 .rela.debug_str_offsets 0000015c 00000000 
 12 .debug_str              000001c5 00000000 DEBUG
 13 .debug_addr             00000014 00000000 DEBUG
 14 .rela.debug_addr        00000024 00000000 
 15 .comment                00000073 00000000 
 16 .note.GNU-stack         00000000 00000000 
 17 .riscv.attributes       00000030 00000000 
 18 .debug_frame            00000044 00000000 DEBUG
 19 .rela.debug_frame       00000060 00000000 
 20 .debug_line             000001c1 00000000 DEBUG
 21 .rela.debug_line        000003d8 00000000 
 22 .debug_line_str         0000013b 00000000 DEBUG
 23 .llvm_addrsig           00000000 00000000 
 24 .symtab                 00000580 00000000 

Disassembly of section .text.Div_s32_s32:

00000000 <Div_s32_s32>:
;                  int32_t eps, int32_t eta) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 24 f1 02  	sw	a5, 40(sp)
;   int32_t secondIter = nomStep / innerMostIter;
      3c: b3 46 f7 02  	div	a3, a4, a5
      40: 23 20 d1 02  	sw	a3, 32(sp)
;   int32_t thirdIter = size_nom / secondIter;
      44: 33 46 d6 02  	div	a2, a2, a3
      48: 23 2e b1 00  	sw	a1, 28(sp)
      4c: 23 22 c1 00  	sw	a2, 4(sp)
;   for (int i = 0; i < thirdIter; i++) {
      50: 63 54 c0 16  	blez	a2, 0x1b8 <Div_s32_s32+0x1b8>
      54: 13 04 05 00  	mv	s0, a0
      58: 03 25 81 02  	lw	a0, 40(sp)
      5c: 13 25 15 00  	slti	a0, a0, 1
      60: 83 25 01 02  	lw	a1, 32(sp)
      64: 93 a5 15 00  	slti	a1, a1, 1
;     for (int k = 0; k < innerMostIter; k++) {
      68: 33 65 b5 00  	or	a0, a0, a1
      6c: 63 16 05 14  	bnez	a0, 0x1b8 <Div_s32_s32+0x1b8>
      70: 13 0b 08 00  	mv	s6, a6
      74: 03 2c 41 06  	lw	s8, 100(sp)
      78: 13 06 00 00  	li	a2, 0
      7c: 03 27 01 06  	lw	a4, 96(sp)
      80: b3 0d 1c 03  	mul	s11, s8, a7
      84: 83 25 81 02  	lw	a1, 40(sp)
;   for (int i = 0; i < thirdIter; i++) {
      88: 03 25 01 02  	lw	a0, 32(sp)
      8c: 33 05 b5 02  	mul	a0, a0, a1
      90: 13 15 25 00  	slli	a0, a0, 2
      94: 23 20 a1 00  	sw	a0, 0(sp)
      98: 93 9b 25 00  	slli	s7, a1, 2
      9c: 23 2c e1 00  	sw	a4, 24(sp)
      a0: 13 55 f7 41  	srai	a0, a4, 31
      a4: 23 2a a1 00  	sw	a0, 20(sp)
      a8: 93 09 00 00  	li	s3, 0
      ac: 03 25 81 02  	lw	a0, 40(sp)
      b0: 23 24 c1 00  	sw	a2, 8(sp)
      b4: 33 05 a6 02  	mul	a0, a2, a0
      b8: 23 22 a1 02  	sw	a0, 36(sp)
      bc: 23 26 81 00  	sw	s0, 12(sp)
      c0: 23 28 61 01  	sw	s6, 16(sp)
;       denom = data_in_denom[i * innerMostIter + k];
      c4: 03 25 41 02  	lw	a0, 36(sp)
      c8: 33 85 a9 00  	add	a0, s3, a0
      cc: 13 15 25 00  	slli	a0, a0, 2
      d0: 83 25 c1 01  	lw	a1, 28(sp)
      d4: 33 85 a5 00  	add	a0, a1, a0
      d8: 03 25 05 00  	lw	a0, 0(a0)
      dc: 13 09 00 00  	li	s2, 0
;       denom = ((eta * denom) + eps);
      e0: b3 15 85 03  	mulh	a1, a0, s8
      e4: 33 06 85 03  	mul	a2, a0, s8
      e8: 83 26 81 01  	lw	a3, 24(sp)
      ec: 13 8a 06 00  	mv	s4, a3
      f0: 33 0a 85 43  	<unknown>
      f4: 33 35 ca 00  	sltu	a0, s4, a2
      f8: 03 27 41 01  	lw	a4, 20(sp)
      fc: b3 85 e5 00  	add	a1, a1, a4
     100: b3 8a a5 00  	add	s5, a1, a0
     104: db a4 a5 02  	<unknown>
     108: 13 95 fa 01  	slli	a0, s5, 31
     10c: db 25 d6 82  	<unknown>
     110: b3 ec a5 00  	or	s9, a1, a0
     114: 03 2d 01 02  	lw	s10, 32(sp)
;             data_in_nom[i * secondIter * innerMostIter + j * innerMostIter + k];
     118: 33 05 24 01  	add	a0, s0, s2
     11c: 83 25 05 00  	lw	a1, 0(a0)
;         nom = (Delta * eta * nom);
     120: 33 96 b5 03  	mulh	a2, a1, s11
;         sgnNom = (nom >= 0) - (nom < 0);
     124: 13 45 f6 ff  	not	a0, a2
     128: 13 55 f5 01  	srli	a0, a0, 31
     12c: 93 56 f6 41  	srai	a3, a2, 31
     130: 33 07 d5 00  	add	a4, a0, a3
;             (int32_t)((nom + sgnNom * (denom >> 1)) / denom);
     134: 5b 25 d5 3e  	<unknown>
     138: b3 b6 ec 02  	mulhu	a3, s9, a4
     13c: b3 86 ac 42  	<unknown>
     140: b3 87 ec 02  	mul	a5, s9, a4
     144: b3 86 e4 42  	<unknown>
     148: 13 85 07 00  	mv	a0, a5
     14c: 33 85 b5 43  	<unknown>
     150: b3 35 f5 00  	sltu	a1, a0, a5
     154: 33 86 c6 00  	add	a2, a3, a2
     158: b3 05 b6 00  	add	a1, a2, a1
     15c: 13 06 0a 00  	mv	a2, s4
     160: 93 86 0a 00  	mv	a3, s5
     164: 97 00 00 00  	auipc	ra, 0
     168: e7 80 00 00  	jalr	ra
;         data_out[i * secondIter * innerMostIter + j * innerMostIter + k] =
     16c: b3 05 2b 01  	add	a1, s6, s2
     170: 23 a0 a5 00  	sw	a0, 0(a1)
;       for (int j = 0; j < secondIter; j++) {
     174: 13 0d fd ff  	addi	s10, s10, -1
     178: 33 09 79 01  	add	s2, s2, s7
     17c: e3 1e 0d f8  	bnez	s10, 0x118 <Div_s32_s32+0x118>
;     for (int k = 0; k < innerMostIter; k++) {
     180: 93 89 19 00  	addi	s3, s3, 1
     184: 13 0b 4b 00  	addi	s6, s6, 4
     188: 13 04 44 00  	addi	s0, s0, 4
     18c: 03 25 81 02  	lw	a0, 40(sp)
     190: e3 9a a9 f2  	bne	s3, a0, 0xc4 <Div_s32_s32+0xc4>
     194: 03 26 81 00  	lw	a2, 8(sp)
;   for (int i = 0; i < thirdIter; i++) {
     198: 13 06 16 00  	addi	a2, a2, 1
     19c: 03 2b 01 01  	lw	s6, 16(sp)
     1a0: 03 25 01 00  	lw	a0, 0(sp)
;   for (int i = 0; i < thirdIter; i++) {
     1a4: 33 0b ab 00  	add	s6, s6, a0
     1a8: 03 24 c1 00  	lw	s0, 12(sp)
     1ac: 33 04 a4 00  	add	s0, s0, a0
     1b0: 03 25 41 00  	lw	a0, 4(sp)
     1b4: e3 1a a6 ee  	bne	a2, a0, 0xa8 <Div_s32_s32+0xa8>
; }
     1b8: 83 20 c1 05  	lw	ra, 92(sp)
     1bc: 03 24 81 05  	lw	s0, 88(sp)
     1c0: 83 24 41 05  	lw	s1, 84(sp)
     1c4: 03 29 01 05  	lw	s2, 80(sp)
     1c8: 83 29 c1 04  	lw	s3, 76(sp)
     1cc: 03 2a 81 04  	lw	s4, 72(sp)
     1d0: 83 2a 41 04  	lw	s5, 68(sp)
     1d4: 03 2b 01 04  	lw	s6, 64(sp)
     1d8: 83 2b c1 03  	lw	s7, 60(sp)
     1dc: 03 2c 81 03  	lw	s8, 56(sp)
     1e0: 83 2c 41 03  	lw	s9, 52(sp)
     1e4: 03 2d 01 03  	lw	s10, 48(sp)
     1e8: 83 2d c1 02  	lw	s11, 44(sp)
     1ec: 13 01 01 06  	addi	sp, sp, 96
     1f0: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(GELU_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                              Size     VMA      Type
  0                                   00000000 00000000 
  1 .strtab                           00000191 00000000 
  2 .text                             00000000 00000000 TEXT
  3 .sdata                            00000014 00000000 DATA
  4 .text.GELU_fp32_fp32              000000b0 00000000 TEXT
  5 .rela.text.GELU_fp32_fp32         00000054 00000000 
  6 .text.GELU_fp32_fp32_sigmoid      00000090 00000000 TEXT
  7 .rela.text.GELU_fp32_fp32_sigmoid 0000003c 00000000 
  8 .debug_loclists                   000000b6 00000000 DEBUG
  9 .debug_abbrev                     000000e3 00000000 DEBUG
 10 .debug_info                       0000013e 00000000 DEBUG
 11 .rela.debug_info                  000000f0 00000000 
 12 .debug_rnglists                   00000019 00000000 DEBUG
 13 .debug_str_offsets                0000005c 00000000 DEBUG
 14 .rela.debug_str_offsets           000000fc 00000000 
 15 .debug_str                        0000018c 00000000 DEBUG
 16 .debug_addr                       00000024 00000000 DEBUG
 17 .rela.debug_addr                  00000054 00000000 
 18 .comment                          00000073 00000000 
 19 .note.GNU-stack                   00000000 00000000 
 20 .riscv.attributes                 00000030 00000000 
 21 .debug_frame                      0000005c 00000000 DEBUG
 22 .rela.debug_frame                 000000c0 00000000 
 23 .debug_line                       000001c2 00000000 DEBUG
 24 .rela.debug_line                  0000033c 00000000 
 25 .debug_line_str                   0000019e 00000000 DEBUG
 26 .llvm_addrsig                     00000000 00000000 
 27 .symtab                           000005b0 00000000 

Disassembly of section .text.GELU_fp32_fp32:

00000000 <GELU_fp32_fp32>:
; void GELU_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 27 26 81 00  	fsw	fs0, 12(sp)
      18: 27 24 91 00  	fsw	fs1, 8(sp)
      1c: 27 22 21 01  	fsw	fs2, 4(sp)
      20: 27 20 31 01  	fsw	fs3, 0(sp)
;   for (int i = 0; i < dataSize; i++) {
      24: 63 52 c0 06  	blez	a2, 0x88 <GELU_fp32_fp32+0x88>
      28: 13 04 06 00  	mv	s0, a2
      2c: 93 84 05 00  	mv	s1, a1
      30: 13 09 05 00  	mv	s2, a0
      34: 37 05 00 00  	lui	a0, 0
      38: 07 24 05 00  	flw	fs0, 0(a0)
      3c: 37 05 00 00  	lui	a0, 0
      40: 87 24 05 00  	flw	fs1, 0(a0)
      44: 37 05 00 00  	lui	a0, 0
      48: 07 29 05 00  	flw	fs2, 0(a0)
;     float32_t x = data_in[i];
      4c: 87 29 09 00  	flw	fs3, 0(s2)
; 	__asm__ volatile("fsqrt.s %0, %1" : "=f" (result) : "f" (x));
      50: 53 70 04 58  	fsqrt.s	ft0, fs0
;                                           (x + 0.044715f * powf(x, 3.0f)))));
      54: d3 f0 39 11  	fmul.s	ft1, fs3, fs3
      58: d3 f0 90 10  	fmul.s	ft1, ft1, fs1
      5c: c3 f0 30 99  	fmadd.s	ft1, ft1, fs3, fs3
;     float32_t cdf = 0.5f * (1.0f + tanhf((sqrtf(2.0f / (float)M_PI) *
      60: 53 f5 00 10  	fmul.s	fa0, ft1, ft0
      64: 97 00 00 00  	auipc	ra, 0
      68: e7 80 00 00  	jalr	ra
      6c: 43 70 25 91  	fmadd.s	ft0, fa0, fs2, fs2
;     data_out[i] = x * cdf;
      70: 53 70 30 11  	fmul.s	ft0, ft0, fs3
      74: 27 a0 04 00  	fsw	ft0, 0(s1)
;   for (int i = 0; i < dataSize; i++) {
      78: 13 04 f4 ff  	addi	s0, s0, -1
      7c: 93 84 44 00  	addi	s1, s1, 4
      80: 13 09 49 00  	addi	s2, s2, 4
      84: e3 14 04 fc  	bnez	s0, 0x4c <GELU_fp32_fp32+0x4c>
; }
      88: 83 20 c1 01  	lw	ra, 28(sp)
      8c: 03 24 81 01  	lw	s0, 24(sp)
      90: 83 24 41 01  	lw	s1, 20(sp)
      94: 03 29 01 01  	lw	s2, 16(sp)
      98: 07 24 c1 00  	flw	fs0, 12(sp)
      9c: 87 24 81 00  	flw	fs1, 8(sp)
      a0: 07 29 41 00  	flw	fs2, 4(sp)
      a4: 87 29 01 00  	flw	fs3, 0(sp)
      a8: 13 01 01 02  	addi	sp, sp, 32
      ac: 67 80 00 00  	ret

Disassembly of section .text.GELU_fp32_fp32_sigmoid:

00000000 <GELU_fp32_fp32_sigmoid>:
;                             int32_t dataSize) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 27 26 81 00  	fsw	fs0, 12(sp)
      18: 27 24 91 00  	fsw	fs1, 8(sp)
      1c: 27 22 21 01  	fsw	fs2, 4(sp)
;   for (int i = 0; i < dataSize; i++) {
      20: 63 56 c0 04  	blez	a2, 0x6c <GELU_fp32_fp32_sigmoid+0x6c>
      24: 13 04 06 00  	mv	s0, a2
      28: 93 84 05 00  	mv	s1, a1
      2c: 13 09 05 00  	mv	s2, a0
      30: 37 05 00 00  	lui	a0, 0
      34: 07 24 05 00  	flw	fs0, 0(a0)
      38: 37 05 00 00  	lui	a0, 0
      3c: 87 24 05 00  	flw	fs1, 0(a0)
;     float32_t x = data_in[i];
      40: 07 29 09 00  	flw	fs2, 0(s2)
;     float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
      44: 53 75 89 10  	fmul.s	fa0, fs2, fs0
      48: 97 00 00 00  	auipc	ra, 0
      4c: e7 80 00 00  	jalr	ra
      50: 53 70 95 00  	fadd.s	ft0, fa0, fs1
;     data_out[i] = x * sigmoid;
      54: 53 70 09 18  	fdiv.s	ft0, fs2, ft0
      58: 27 a0 04 00  	fsw	ft0, 0(s1)
;   for (int i = 0; i < dataSize; i++) {
      5c: 13 04 f4 ff  	addi	s0, s0, -1
      60: 93 84 44 00  	addi	s1, s1, 4
      64: 13 09 49 00  	addi	s2, s2, 4
      68: e3 1c 04 fc  	bnez	s0, 0x40 <GELU_fp32_fp32_sigmoid+0x40>
; }
      6c: 83 20 c1 01  	lw	ra, 28(sp)
      70: 03 24 81 01  	lw	s0, 24(sp)
      74: 83 24 41 01  	lw	s1, 20(sp)
      78: 03 29 01 01  	lw	s2, 16(sp)
      7c: 07 24 c1 00  	flw	fs0, 12(sp)
      80: 87 24 81 00  	flw	fs1, 8(sp)
      84: 07 29 41 00  	flw	fs2, 4(sp)
      88: 13 01 01 02  	addi	sp, sp, 32
      8c: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(GELU_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 00000116 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text.GELU_s8_s32       00000048 00000000 TEXT
  4 .debug_loclists         000000ae 00000000 DEBUG
  5 .debug_abbrev           00000085 00000000 DEBUG
  6 .debug_info             000000fe 00000000 DEBUG
  7 .rela.debug_info        00000084 00000000 
  8 .debug_str_offsets      00000074 00000000 DEBUG
  9 .rela.debug_str_offsets 00000144 00000000 
 10 .debug_str              00000190 00000000 DEBUG
 11 .debug_addr             0000000c 00000000 DEBUG
 12 .rela.debug_addr        0000000c 00000000 
 13 .comment                00000073 00000000 
 14 .note.GNU-stack         00000000 00000000 
 15 .riscv.attributes       00000030 00000000 
 16 .debug_frame            00000024 00000000 DEBUG
 17 .rela.debug_frame       00000030 00000000 
 18 .debug_line             00000129 00000000 DEBUG
 19 .rela.debug_line        00000210 00000000 
 20 .debug_line_str         0000013b 00000000 DEBUG
 21 .llvm_addrsig           00000000 00000000 
 22 .symtab                 000003c0 00000000 

Disassembly of section .text.GELU_s8_s32:

00000000 <GELU_s8_s32>:
;   for (int i = 0; i < dataSize; i++) {
       0: 63 52 c0 04  	blez	a2, 0x44 <GELU_s8_s32+0x44>
       4: 33 08 d0 40  	neg	a6, a3
;   for (int i = 0; i < dataSize; i++) {
       8: 7b 40 c6 01  	<unknown>
;     x = data_in[i] + input_offset;
       c: 8b 08 15 00  	<unknown>
      10: b3 82 f8 00  	add	t0, a7, a5
;     sign = (x > 0) - (x < 0); // sgn(x)
      14: 33 23 50 00  	sgtz	t1, t0
      18: db a8 f8 3e  	<unknown>
      1c: b3 88 68 00  	add	a7, a7, t1
;     x_abs = sign * x;         // abs(x)
      20: 33 83 58 02  	mul	t1, a7, t0
      24: 33 43 03 05  	<unknown>
;     d = q + b;
      28: 33 03 d3 00  	add	t1, t1, a3
;     L = sign * (-(d * d) + one);
      2c: 93 03 07 00  	mv	t2, a4
      30: b3 13 63 42  	<unknown>
      34: b3 88 13 03  	mul	a7, t2, a7
;     y = x * (((one + L)) >> 1);
      38: db a8 e8 02  	<unknown>
      3c: b3 88 58 02  	mul	a7, a7, t0
;     data_out[i] = y;
      40: 2b a2 15 01  	<unknown>
; }
      44: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Gemm_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                           Size     VMA      Type
  0                                00000000 00000000 
  1 .strtab                        00000135 00000000 
  2 .text                          00000000 00000000 TEXT
  3 .text.Gemm_fp32_fp32_fp32_fp32 0000013c 00000000 TEXT
  4 .debug_loclists                00000293 00000000 DEBUG
  5 .debug_abbrev                  0000008f 00000000 DEBUG
  6 .debug_info                    00000127 00000000 DEBUG
  7 .rela.debug_info               000000c0 00000000 
  8 .debug_rnglists                00000028 00000000 DEBUG
  9 .debug_str_offsets             00000074 00000000 DEBUG
 10 .rela.debug_str_offsets        00000144 00000000 
 11 .debug_str                     0000019a 00000000 DEBUG
 12 .debug_addr                    00000018 00000000 DEBUG
 13 .rela.debug_addr               00000030 00000000 
 14 .comment                       00000073 00000000 
 15 .note.GNU-stack                00000000 00000000 
 16 .riscv.attributes              00000030 00000000 
 17 .debug_frame                   00000034 00000000 DEBUG
 18 .rela.debug_frame              00000060 00000000 
 19 .debug_line                    000001ac 00000000 DEBUG
 20 .rela.debug_line               0000036c 00000000 
 21 .debug_line_str                00000163 00000000 DEBUG
 22 .llvm_addrsig                  00000000 00000000 
 23 .symtab                        00000520 00000000 

Disassembly of section .text.Gemm_fp32_fp32_fp32_fp32:

00000000 <Gemm_fp32_fp32_fp32_fp32>:
;                               int32_t transB) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 81 00  	sw	s0, 28(sp)
       8: 23 2c 91 00  	sw	s1, 24(sp)
       c: 23 2a 21 01  	sw	s2, 20(sp)
      10: 23 28 31 01  	sw	s3, 16(sp)
      14: 23 26 41 01  	sw	s4, 12(sp)
      18: 23 24 51 01  	sw	s5, 8(sp)
;   for (uint32_t i = 0; i < M; ++i) {
      1c: 63 00 07 10  	beqz	a4, 0x11c <Gemm_fp32_fp32_fp32_fp32+0x11c>
;     for (uint32_t j = 0; j < O; ++j) {
      20: 63 0e 08 0e  	beqz	a6, 0x11c <Gemm_fp32_fp32_fp32_fp32+0x11c>
;       for (uint32_t k = 0; k < N; ++k) {
      24: 63 8e 07 0a  	beqz	a5, 0xe0 <Gemm_fp32_fp32_fp32_fp32+0xe0>
      28: 83 22 01 02  	lw	t0, 32(sp)
      2c: 13 03 00 00  	li	t1, 0
      30: 93 03 00 00  	li	t2, 0
      34: 53 00 00 f0  	fmv.w.x	ft0, zero
      38: 13 0e 00 00  	li	t3, 0
      3c: 93 0e 00 00  	li	t4, 0
      40: 33 8f 03 03  	mul	t5, t2, a6
      44: fb 40 48 04  	<unknown>
      48: 93 8f 03 00  	mv	t6, t2
      4c: 13 04 03 00  	mv	s0, t1
      50: 93 04 0e 00  	mv	s1, t3
      54: 13 89 0e 00  	mv	s2, t4
      58: 93 89 07 00  	mv	s3, a5
      5c: d3 00 00 20  	fmv.s	ft1, ft0
      60: 7b c0 47 02  	<unknown>
      64: 13 0a 04 00  	mv	s4, s0
;         uint32_t a_idx = transA ? (k * M + i) : (i * N + k);
      68: 63 84 08 00  	beqz	a7, 0x70 <Gemm_fp32_fp32_fp32_fp32+0x70>
      6c: 13 8a 0f 00  	mv	s4, t6
      70: 93 0a 09 00  	mv	s5, s2
;         uint32_t b_idx = transB ? (j * N + k) : (k * O + j);
      74: 63 84 02 00  	beqz	t0, 0x7c <Gemm_fp32_fp32_fp32_fp32+0x7c>
      78: 93 8a 04 00  	mv	s5, s1
;         sum += pSrcA[a_idx] * pSrcB[b_idx];
      7c: 13 1a 2a 00  	slli	s4, s4, 2
      80: 33 0a 45 01  	add	s4, a0, s4
      84: 07 21 0a 00  	flw	ft2, 0(s4)
      88: 13 9a 2a 00  	slli	s4, s5, 2
      8c: 33 8a 45 01  	add	s4, a1, s4
      90: 87 21 0a 00  	flw	ft3, 0(s4)
      94: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;       for (uint32_t k = 0; k < N; ++k) {
      98: 93 89 f9 ff  	addi	s3, s3, -1
      9c: 33 09 09 01  	add	s2, s2, a6
      a0: 93 84 14 00  	addi	s1, s1, 1
      a4: 13 04 14 00  	addi	s0, s0, 1
      a8: b3 8f ef 00  	add	t6, t6, a4
;       pDstY[i * O + j] = sum + pDstC[i * O + j];
      ac: b3 8f ee 01  	add	t6, t4, t5
      b0: 93 9f 2f 00  	slli	t6, t6, 2
      b4: 33 04 f6 01  	add	s0, a2, t6
      b8: 07 21 04 00  	flw	ft2, 0(s0)
      bc: d3 70 11 00  	fadd.s	ft1, ft2, ft1
      c0: b3 8f f6 01  	add	t6, a3, t6
      c4: 27 a0 1f 00  	fsw	ft1, 0(t6)
;     for (uint32_t j = 0; j < O; ++j) {
      c8: 93 8e 1e 00  	addi	t4, t4, 1
      cc: 33 0e fe 00  	add	t3, t3, a5
;   for (uint32_t i = 0; i < M; ++i) {
      d0: 93 83 13 00  	addi	t2, t2, 1
      d4: 33 03 f3 00  	add	t1, t1, a5
      d8: e3 90 e3 f6  	bne	t2, a4, 0x38 <Gemm_fp32_fp32_fp32_fp32+0x38>
      dc: 6f 00 00 04  	j	0x11c <Gemm_fp32_fp32_fp32_fp32+0x11c>
      e0: 13 05 00 00  	li	a0, 0
;   for (uint32_t i = 0; i < M; ++i) {
      e4: 93 15 28 00  	slli	a1, a6, 2
      e8: fb 40 87 01  	<unknown>
      ec: 93 07 06 00  	mv	a5, a2
      f0: 93 88 06 00  	mv	a7, a3
      f4: 93 02 08 00  	mv	t0, a6
      f8: 7b 40 a8 00  	<unknown>
;       pDstY[i * O + j] = sum + pDstC[i * O + j];
      fc: 07 a0 07 00  	flw	ft0, 0(a5)
     100: 27 a0 08 00  	fsw	ft0, 0(a7)
;     for (uint32_t j = 0; j < O; ++j) {
     104: 93 82 f2 ff  	addi	t0, t0, -1
     108: 93 88 48 00  	addi	a7, a7, 4
     10c: 93 87 47 00  	addi	a5, a5, 4
;   for (uint32_t i = 0; i < M; ++i) {
     110: 13 05 15 00  	addi	a0, a0, 1
     114: b3 86 b6 00  	add	a3, a3, a1
     118: 33 06 b6 00  	add	a2, a2, a1
; }
     11c: 03 24 c1 01  	lw	s0, 28(sp)
     120: 83 24 81 01  	lw	s1, 24(sp)
     124: 03 29 41 01  	lw	s2, 20(sp)
     128: 83 29 01 01  	lw	s3, 16(sp)
     12c: 03 2a c1 00  	lw	s4, 12(sp)
     130: 83 2a 81 00  	lw	s5, 8(sp)
     134: 13 01 01 02  	addi	sp, sp, 32
     138: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Gemm_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                     Size     VMA      Type
  0                          00000000 00000000 
  1 .strtab                  0000012d 00000000 
  2 .text                    00000000 00000000 TEXT
  3 .text.Gemm_s8_s8_s32_s32 000003d8 00000000 TEXT
  4 .debug_loclists          00001138 00000000 DEBUG
  5 .debug_abbrev            00000086 00000000 DEBUG
  6 .debug_info              00000192 00000000 DEBUG
  7 .rela.debug_info         00000078 00000000 
  8 .debug_rnglists          00000087 00000000 DEBUG
  9 .debug_str_offsets       00000088 00000000 DEBUG
 10 .rela.debug_str_offsets  00000180 00000000 
 11 .debug_str               000001c1 00000000 DEBUG
 12 .debug_addr              0000000c 00000000 DEBUG
 13 .rela.debug_addr         0000000c 00000000 
 14 .comment                 00000073 00000000 
 15 .note.GNU-stack          00000000 00000000 
 16 .riscv.attributes        00000030 00000000 
 17 .debug_frame             00000038 00000000 DEBUG
 18 .rela.debug_frame        00000060 00000000 
 19 .debug_line              000004da 00000000 DEBUG
 20 .rela.debug_line         00000cc0 00000000 
 21 .debug_line_str          0000013b 00000000 DEBUG
 22 .llvm_addrsig            00000000 00000000 
 23 .symtab                  00000b50 00000000 

Disassembly of section .text.Gemm_s8_s8_s32_s32:

00000000 <Gemm_s8_s8_s32_s32>:
;                         int32_t C_offset, int32_t Y_offset) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 81 00  	sw	s0, 28(sp)
       8: 23 2c 91 00  	sw	s1, 24(sp)
       c: 23 2a 21 01  	sw	s2, 20(sp)
      10: 23 28 31 01  	sw	s3, 16(sp)
      14: 23 26 41 01  	sw	s4, 12(sp)
      18: 23 24 51 01  	sw	s5, 8(sp)
      1c: 23 22 61 01  	sw	s6, 4(sp)
      20: 23 20 71 01  	sw	s7, 0(sp)
      24: 83 22 81 03  	lw	t0, 56(sp)
      28: 03 23 41 03  	lw	t1, 52(sp)
      2c: 03 2e 01 03  	lw	t3, 48(sp)
      30: 83 2f 81 02  	lw	t6, 40(sp)
      34: 03 2f 41 02  	lw	t5, 36(sp)
      38: 83 2e c1 02  	lw	t4, 44(sp)
      3c: 83 23 01 02  	lw	t2, 32(sp)
;   if (transA == 0 && transB == 0) {
      40: 33 e4 ef 01  	or	s0, t6, t5
      44: 63 02 04 0a  	beqz	s0, 0xe8 <Gemm_s8_s8_s32_s32+0xe8>
;   } else if (transA == 1 && transB == 0) {
      48: 13 44 1f 00  	xori	s0, t5, 1
      4c: 33 64 f4 01  	or	s0, s0, t6
      50: 63 16 04 12  	bnez	s0, 0x17c <Gemm_s8_s8_s32_s32+0x17c>
;     for (uint32_t m = 0; m < M; ++m) {
      54: 13 3f 17 00  	seqz	t5, a4
      58: 93 3f 18 00  	seqz	t6, a6
      5c: 33 6f ff 01  	or	t5, t5, t6
      60: 63 18 0f 34  	bnez	t5, 0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
;         for (n = 0; n < N; n++) {
      64: 63 82 07 28  	beqz	a5, 0x2e8 <Gemm_s8_s8_s32_s32+0x2e8>
      68: 13 0f 00 00  	li	t5, 0
      6c: 93 0f 00 00  	li	t6, 0
      70: 33 04 0f 03  	mul	s0, t5, a6
      74: 93 84 05 00  	mv	s1, a1
      78: fb 40 e8 02  	<unknown>
      7c: 13 09 00 00  	li	s2, 0
      80: 93 09 05 00  	mv	s3, a0
      84: 13 8a 04 00  	mv	s4, s1
      88: 93 8a 07 00  	mv	s5, a5
      8c: 7b c0 c7 00  	<unknown>
;           sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
      90: 0b fb e9 00  	<unknown>
;                  (pSrcB[n * P + p] + B_offset);
      94: 8b 7b 0a 01  	<unknown>
;           sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
      98: 33 0b db 01  	add	s6, s6, t4
;                  (pSrcB[n * P + p] + B_offset);
      9c: b3 8b cb 01  	add	s7, s7, t3
;         for (n = 0; n < N; n++) {
      a0: 93 8a fa ff  	addi	s5, s5, -1
;           sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
      a4: 33 89 6b 43  	<unknown>
;             alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
      a8: b3 89 8f 00  	add	s3, t6, s0
      ac: 93 99 29 00  	slli	s3, s3, 2
      b0: 33 0a 36 01  	add	s4, a2, s3
      b4: 03 2a 0a 00  	lw	s4, 0(s4)
      b8: 33 0a 6a 00  	add	s4, s4, t1
      bc: 93 8a 02 00  	mv	s5, t0
      c0: b3 0a 19 43  	<unknown>
      c4: b3 0a 7a 42  	<unknown>
;         pDstY[m * P + p] =
      c8: 33 89 36 01  	add	s2, a3, s3
      cc: 23 20 59 01  	sw	s5, 0(s2)
;       for (p = 0; p < P; p++) {
      d0: 93 8f 1f 00  	addi	t6, t6, 1
      d4: 93 84 14 00  	addi	s1, s1, 1
;     for (uint32_t m = 0; m < M; ++m) {
      d8: 13 0f 1f 00  	addi	t5, t5, 1
      dc: 13 05 15 00  	addi	a0, a0, 1
      e0: e3 16 ef f8  	bne	t5, a4, 0x6c <Gemm_s8_s8_s32_s32+0x6c>
      e4: 6f 00 c0 2c  	j	0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
;     for (m = 0; m < M; ++m) {
      e8: 13 3f 17 00  	seqz	t5, a4
      ec: 93 3f 18 00  	seqz	t6, a6
      f0: 33 6f ff 01  	or	t5, t5, t6
      f4: 63 1e 0f 2a  	bnez	t5, 0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
;         for (n = 0; n < N; n++) {
      f8: 63 86 07 1a  	beqz	a5, 0x2a4 <Gemm_s8_s8_s32_s32+0x2a4>
      fc: 13 0f 00 00  	li	t5, 0
     100: 93 0f 00 00  	li	t6, 0
     104: 33 04 0f 03  	mul	s0, t5, a6
     108: 93 84 05 00  	mv	s1, a1
     10c: fb 40 e8 02  	<unknown>
     110: 13 09 00 00  	li	s2, 0
     114: 93 09 05 00  	mv	s3, a0
     118: 13 8a 04 00  	mv	s4, s1
     11c: 93 8a 07 00  	mv	s5, a5
     120: 7b c0 c7 00  	<unknown>
;           sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
     124: 0b 8b 19 00  	<unknown>
;                  (pSrcB[n * P + p] + B_offset);
     128: 8b 7b 0a 01  	<unknown>
;           sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
     12c: 33 0b db 01  	add	s6, s6, t4
;                  (pSrcB[n * P + p] + B_offset);
     130: b3 8b cb 01  	add	s7, s7, t3
;         for (n = 0; n < N; n++) {
     134: 93 8a fa ff  	addi	s5, s5, -1
;           sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
     138: 33 89 6b 43  	<unknown>
;             alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
     13c: b3 89 8f 00  	add	s3, t6, s0
     140: 93 99 29 00  	slli	s3, s3, 2
     144: 33 0a 36 01  	add	s4, a2, s3
     148: 03 2a 0a 00  	lw	s4, 0(s4)
     14c: 33 0a 6a 00  	add	s4, s4, t1
     150: 93 8a 02 00  	mv	s5, t0
     154: b3 0a 19 43  	<unknown>
     158: b3 0a 7a 42  	<unknown>
;         pDstY[m * P + p] =
     15c: 33 89 36 01  	add	s2, a3, s3
     160: 23 20 59 01  	sw	s5, 0(s2)
;       for (p = 0; p < P; p++) {
     164: 93 8f 1f 00  	addi	t6, t6, 1
     168: 93 84 14 00  	addi	s1, s1, 1
;     for (m = 0; m < M; ++m) {
     16c: 13 0f 1f 00  	addi	t5, t5, 1
     170: 33 05 f5 00  	add	a0, a0, a5
     174: e3 16 ef f8  	bne	t5, a4, 0x100 <Gemm_s8_s8_s32_s32+0x100>
     178: 6f 00 80 23  	j	0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
     17c: 93 cf 1f 00  	xori	t6, t6, 1
;   } else if (transA == 0 && transB == 1) {
     180: b3 6f ff 01  	or	t6, t5, t6
     184: 13 3f 17 00  	seqz	t5, a4
     188: 13 34 18 00  	seqz	s0, a6
     18c: 33 6f 8f 00  	or	t5, t5, s0
;   } else if (transA == 0 && transB == 1) {
     190: 63 96 0f 08  	bnez	t6, 0x21c <Gemm_s8_s8_s32_s32+0x21c>
;     for (uint32_t m = 0; m < M; ++m) {
     194: 63 1e 0f 20  	bnez	t5, 0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
;         for (n = 0; n < N; n++) {
     198: 63 8a 07 18  	beqz	a5, 0x32c <Gemm_s8_s8_s32_s32+0x32c>
     19c: 13 0f 00 00  	li	t5, 0
     1a0: 93 0f 00 00  	li	t6, 0
     1a4: 33 04 0f 03  	mul	s0, t5, a6
     1a8: 93 84 05 00  	mv	s1, a1
     1ac: fb 40 e8 02  	<unknown>
     1b0: 13 09 00 00  	li	s2, 0
     1b4: 93 09 05 00  	mv	s3, a0
     1b8: 13 8a 04 00  	mv	s4, s1
     1bc: 93 8a 07 00  	mv	s5, a5
     1c0: 7b c0 c7 00  	<unknown>
;           sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
     1c4: 0b 8b 19 00  	<unknown>
;                  (pSrcB[p * N + n] + B_offset);
     1c8: 8b 0b 1a 00  	<unknown>
;           sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
     1cc: 33 0b db 01  	add	s6, s6, t4
;                  (pSrcB[p * N + n] + B_offset);
     1d0: b3 8b cb 01  	add	s7, s7, t3
;         for (n = 0; n < N; n++) {
     1d4: 93 8a fa ff  	addi	s5, s5, -1
;           sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
     1d8: 33 89 6b 43  	<unknown>
;             alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
     1dc: b3 89 8f 00  	add	s3, t6, s0
     1e0: 93 99 29 00  	slli	s3, s3, 2
     1e4: 33 0a 36 01  	add	s4, a2, s3
     1e8: 03 2a 0a 00  	lw	s4, 0(s4)
     1ec: 33 0a 6a 00  	add	s4, s4, t1
     1f0: 93 8a 02 00  	mv	s5, t0
     1f4: b3 0a 19 43  	<unknown>
     1f8: b3 0a 7a 42  	<unknown>
;         pDstY[m * P + p] =
     1fc: 33 89 36 01  	add	s2, a3, s3
     200: 23 20 59 01  	sw	s5, 0(s2)
;       for (p = 0; p < P; p++) {
     204: 93 8f 1f 00  	addi	t6, t6, 1
     208: b3 84 f4 00  	add	s1, s1, a5
;     for (uint32_t m = 0; m < M; ++m) {
     20c: 13 0f 1f 00  	addi	t5, t5, 1
     210: 33 05 f5 00  	add	a0, a0, a5
     214: e3 16 ef f8  	bne	t5, a4, 0x1a0 <Gemm_s8_s8_s32_s32+0x1a0>
     218: 6f 00 80 19  	j	0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
;     for (uint32_t m = 0; m < M; ++m) {
     21c: 63 1a 0f 18  	bnez	t5, 0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
;         for (n = 0; n < N; n++) {
     220: 63 88 07 14  	beqz	a5, 0x370 <Gemm_s8_s8_s32_s32+0x370>
     224: 13 0f 00 00  	li	t5, 0
     228: 93 0f 00 00  	li	t6, 0
     22c: 33 04 0f 03  	mul	s0, t5, a6
     230: 93 84 05 00  	mv	s1, a1
     234: fb 40 e8 02  	<unknown>
     238: 13 09 00 00  	li	s2, 0
     23c: 93 09 05 00  	mv	s3, a0
     240: 13 8a 04 00  	mv	s4, s1
     244: 93 8a 07 00  	mv	s5, a5
     248: 7b c0 c7 00  	<unknown>
;           sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
     24c: 0b fb e9 00  	<unknown>
;                  (pSrcB[p * N + n] + B_offset);
     250: 8b 0b 1a 00  	<unknown>
;           sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
     254: 33 0b db 01  	add	s6, s6, t4
;                  (pSrcB[p * N + n] + B_offset);
     258: b3 8b cb 01  	add	s7, s7, t3
;         for (n = 0; n < N; n++) {
     25c: 93 8a fa ff  	addi	s5, s5, -1
;           sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
     260: 33 89 6b 43  	<unknown>
;             alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
     264: b3 89 8f 00  	add	s3, t6, s0
     268: 93 99 29 00  	slli	s3, s3, 2
     26c: 33 0a 36 01  	add	s4, a2, s3
     270: 03 2a 0a 00  	lw	s4, 0(s4)
     274: 33 0a 6a 00  	add	s4, s4, t1
     278: 93 8a 02 00  	mv	s5, t0
     27c: b3 0a 19 43  	<unknown>
     280: b3 0a 7a 42  	<unknown>
;         pDstY[m * P + p] =
     284: 33 89 36 01  	add	s2, a3, s3
     288: 23 20 59 01  	sw	s5, 0(s2)
;       for (p = 0; p < P; p++) {
     28c: 93 8f 1f 00  	addi	t6, t6, 1
     290: b3 84 f4 00  	add	s1, s1, a5
;     for (uint32_t m = 0; m < M; ++m) {
     294: 13 0f 1f 00  	addi	t5, t5, 1
     298: 13 05 15 00  	addi	a0, a0, 1
     29c: e3 16 ef f8  	bne	t5, a4, 0x228 <Gemm_s8_s8_s32_s32+0x228>
     2a0: 6f 00 00 11  	j	0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
     2a4: 13 05 00 00  	li	a0, 0
;     for (m = 0; m < M; ++m) {
     2a8: 93 15 28 00  	slli	a1, a6, 2
     2ac: fb 40 a7 01  	<unknown>
     2b0: 93 07 06 00  	mv	a5, a2
     2b4: 93 88 06 00  	mv	a7, a3
     2b8: 13 0e 08 00  	mv	t3, a6
     2bc: 7b 40 c8 00  	<unknown>
;             alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
     2c0: 8b ae 47 00  	<unknown>
     2c4: b3 8e 6e 00  	add	t4, t4, t1
     2c8: 13 8f 02 00  	mv	t5, t0
     2cc: 33 8f 7e 42  	<unknown>
;       for (p = 0; p < P; p++) {
     2d0: 13 0e fe ff  	addi	t3, t3, -1
;         pDstY[m * P + p] =
     2d4: 2b a2 e8 01  	<unknown>
;     for (m = 0; m < M; ++m) {
     2d8: 13 05 15 00  	addi	a0, a0, 1
     2dc: b3 86 b6 00  	add	a3, a3, a1
     2e0: 33 06 b6 00  	add	a2, a2, a1
     2e4: 6f 00 c0 0c  	j	0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
     2e8: 13 05 00 00  	li	a0, 0
;     for (uint32_t m = 0; m < M; ++m) {
     2ec: 93 15 28 00  	slli	a1, a6, 2
     2f0: fb 40 a7 01  	<unknown>
     2f4: 93 07 06 00  	mv	a5, a2
     2f8: 93 88 06 00  	mv	a7, a3
     2fc: 13 0e 08 00  	mv	t3, a6
     300: 7b 40 c8 00  	<unknown>
;             alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
     304: 8b ae 47 00  	<unknown>
     308: b3 8e 6e 00  	add	t4, t4, t1
     30c: 13 8f 02 00  	mv	t5, t0
     310: 33 8f 7e 42  	<unknown>
;       for (p = 0; p < P; p++) {
     314: 13 0e fe ff  	addi	t3, t3, -1
;         pDstY[m * P + p] =
     318: 2b a2 e8 01  	<unknown>
;     for (uint32_t m = 0; m < M; ++m) {
     31c: 13 05 15 00  	addi	a0, a0, 1
     320: b3 86 b6 00  	add	a3, a3, a1
     324: 33 06 b6 00  	add	a2, a2, a1
     328: 6f 00 80 08  	j	0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
     32c: 13 05 00 00  	li	a0, 0
;     for (uint32_t m = 0; m < M; ++m) {
     330: 93 15 28 00  	slli	a1, a6, 2
     334: fb 40 a7 01  	<unknown>
     338: 93 07 06 00  	mv	a5, a2
     33c: 93 88 06 00  	mv	a7, a3
     340: 13 0e 08 00  	mv	t3, a6
     344: 7b 40 c8 00  	<unknown>
;             alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
     348: 8b ae 47 00  	<unknown>
     34c: b3 8e 6e 00  	add	t4, t4, t1
     350: 13 8f 02 00  	mv	t5, t0
     354: 33 8f 7e 42  	<unknown>
;       for (p = 0; p < P; p++) {
     358: 13 0e fe ff  	addi	t3, t3, -1
;         pDstY[m * P + p] =
     35c: 2b a2 e8 01  	<unknown>
;     for (uint32_t m = 0; m < M; ++m) {
     360: 13 05 15 00  	addi	a0, a0, 1
     364: b3 86 b6 00  	add	a3, a3, a1
     368: 33 06 b6 00  	add	a2, a2, a1
     36c: 6f 00 40 04  	j	0x3b0 <Gemm_s8_s8_s32_s32+0x3b0>
     370: 13 05 00 00  	li	a0, 0
;     for (uint32_t m = 0; m < M; ++m) {
     374: 93 15 28 00  	slli	a1, a6, 2
     378: fb 40 a7 01  	<unknown>
     37c: 93 07 06 00  	mv	a5, a2
     380: 93 88 06 00  	mv	a7, a3
     384: 13 0e 08 00  	mv	t3, a6
     388: 7b 40 c8 00  	<unknown>
;             alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
     38c: 8b ae 47 00  	<unknown>
     390: b3 8e 6e 00  	add	t4, t4, t1
     394: 13 8f 02 00  	mv	t5, t0
     398: 33 8f 7e 42  	<unknown>
;       for (p = 0; p < P; p++) {
     39c: 13 0e fe ff  	addi	t3, t3, -1
;         pDstY[m * P + p] =
     3a0: 2b a2 e8 01  	<unknown>
;     for (uint32_t m = 0; m < M; ++m) {
     3a4: 13 05 15 00  	addi	a0, a0, 1
     3a8: b3 86 b6 00  	add	a3, a3, a1
     3ac: 33 06 b6 00  	add	a2, a2, a1
; }
     3b0: 03 24 c1 01  	lw	s0, 28(sp)
     3b4: 83 24 81 01  	lw	s1, 24(sp)
     3b8: 03 29 41 01  	lw	s2, 20(sp)
     3bc: 83 29 01 01  	lw	s3, 16(sp)
     3c0: 03 2a c1 00  	lw	s4, 12(sp)
     3c4: 83 2a 81 00  	lw	s5, 8(sp)
     3c8: 03 2b 41 00  	lw	s6, 4(sp)
     3cc: 83 2b 01 00  	lw	s7, 0(sp)
     3d0: 13 01 01 02  	addi	sp, sp, 32
     3d4: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Hardswish_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 00000121 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text.iHardswish_s8_s32 0000002c 00000000 TEXT
  4 .debug_loclists         0000007c 00000000 DEBUG
  5 .debug_abbrev           00000078 00000000 DEBUG
  6 .debug_info             000000be 00000000 DEBUG
  7 .rela.debug_info        00000084 00000000 
  8 .debug_str_offsets      00000054 00000000 DEBUG
  9 .rela.debug_str_offsets 000000e4 00000000 
 10 .debug_str              0000017c 00000000 DEBUG
 11 .debug_addr             0000000c 00000000 DEBUG
 12 .rela.debug_addr        0000000c 00000000 
 13 .comment                00000073 00000000 
 14 .note.GNU-stack         00000000 00000000 
 15 .riscv.attributes       00000030 00000000 
 16 .debug_frame            00000024 00000000 DEBUG
 17 .rela.debug_frame       00000030 00000000 
 18 .debug_line             000000f1 00000000 DEBUG
 19 .rela.debug_line        00000180 00000000 
 20 .debug_line_str         00000145 00000000 DEBUG
 21 .llvm_addrsig           00000000 00000000 
 22 .symtab                 000002e0 00000000 

Disassembly of section .text.iHardswish_s8_s32:

00000000 <iHardswish_s8_s32>:
;   for (int i = 0; i < size; i++) {
       0: 63 54 c0 02  	blez	a2, 0x28 <iHardswish_s8_s32+0x28>
       4: 33 07 e8 00  	add	a4, a6, a4
;   for (int i = 0; i < size; i++) {
       8: 7b 40 e6 00  	<unknown>
;     temp = input[i] + input_offset + three;
       c: 0b 08 15 00  	<unknown>
      10: b3 08 07 01  	add	a7, a4, a6
;     if (temp < 0) {
      14: b3 e8 08 04  	<unknown>
;     if (temp > six) {
      18: b3 c8 f8 04  	<unknown>
;     temp = temp * one_over_six;
      1c: 33 08 d8 02  	mul	a6, a6, a3
;     temp = input[i] * temp;
      20: 33 08 18 03  	mul	a6, a6, a7
;     output[i] = temp;
      24: 2b a2 05 01  	<unknown>
; }
      28: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Layernorm_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                           Size     VMA      Type
  0                                00000000 00000000 
  1 .strtab                        0000014a 00000000 
  2 .text                          00000000 00000000 TEXT
  3 .sdata                         00000004 00000000 DATA
  4 .text.Layernorm_fp32_fp32      000000e0 00000000 TEXT
  5 .rela.text.Layernorm_fp32_fp32 00000018 00000000 
  6 .debug_loclists                00000228 00000000 DEBUG
  7 .debug_abbrev                  000000ce 00000000 DEBUG
  8 .debug_info                    0000012e 00000000 DEBUG
  9 .rela.debug_info               000000c0 00000000 
 10 .debug_rnglists                00000027 00000000 DEBUG
 11 .debug_str_offsets             0000006c 00000000 DEBUG
 12 .rela.debug_str_offsets        0000012c 00000000 
 13 .debug_str                     00000193 00000000 DEBUG
 14 .debug_addr                    00000018 00000000 DEBUG
 15 .rela.debug_addr               00000030 00000000 
 16 .comment                       00000073 00000000 
 17 .note.GNU-stack                00000000 00000000 
 18 .riscv.attributes              00000030 00000000 
 19 .debug_frame                   00000024 00000000 DEBUG
 20 .rela.debug_frame              00000030 00000000 
 21 .debug_line                    000001d9 00000000 DEBUG
 22 .rela.debug_line               00000390 00000000 
 23 .debug_line_str                000001a8 00000000 DEBUG
 24 .llvm_addrsig                  00000000 00000000 
 25 .symtab                        00000500 00000000 

Disassembly of section .text.Layernorm_fp32_fp32:

00000000 <Layernorm_fp32_fp32>:
;                          int32_t size, int32_t lastDimLength) {
       0: 33 47 f7 02  	div	a4, a4, a5
;   for (int i = 0; i < (size / lastDimLength); i++) {
       4: 63 5c e0 0c  	blez	a4, 0xdc <Layernorm_fp32_fp32+0xdc>
;     for (int j = 0; j < lastDimLength; j++) {
       8: 63 54 f0 0c  	blez	a5, 0xd0 <Layernorm_fp32_fp32+0xd0>
       c: 37 08 00 00  	lui	a6, 0
      10: 07 20 08 00  	flw	ft0, 0(a6)
      14: 13 08 00 00  	li	a6, 0
      18: d3 f0 07 d0  	fcvt.s.w	ft1, a5
      1c: d3 70 10 18  	fdiv.s	ft1, ft0, ft1
;   for (int i = 0; i < (size / lastDimLength); i++) {
      20: 93 98 27 00  	slli	a7, a5, 2
      24: 53 01 00 f0  	fmv.w.x	ft2, zero
      28: 93 02 05 00  	mv	t0, a0
      2c: 13 83 07 00  	mv	t1, a5
      30: d3 01 21 20  	fmv.s	ft3, ft2
;     for (int j = 0; j < lastDimLength; j++) {
      34: 7b c0 87 00  	<unknown>
;       mean += data_in[j + i * lastDimLength];
      38: 07 a2 02 00  	flw	ft4, 0(t0)
      3c: d3 71 32 00  	fadd.s	ft3, ft4, ft3
;     for (int j = 0; j < lastDimLength; j++) {
      40: 13 03 f3 ff  	addi	t1, t1, -1
      44: 93 82 42 00  	addi	t0, t0, 4
      48: d3 f1 11 10  	fmul.s	ft3, ft3, ft1
      4c: 93 02 05 00  	mv	t0, a0
      50: 13 83 07 00  	mv	t1, a5
      54: 53 02 21 20  	fmv.s	ft4, ft2
;     for (int j = 0; j < lastDimLength; j++) {
      58: 7b c0 a7 00  	<unknown>
;       temp = data_in[j + i * lastDimLength] - mean;
      5c: 87 a2 02 00  	flw	ft5, 0(t0)
      60: d3 f2 32 08  	fsub.s	ft5, ft5, ft3
;       sum += temp * temp;
      64: 43 f2 52 20  	fmadd.s	ft4, ft5, ft5, ft4
;     for (int j = 0; j < lastDimLength; j++) {
      68: 13 03 f3 ff  	addi	t1, t1, -1
      6c: 93 82 42 00  	addi	t0, t0, 4
      70: 93 02 00 00  	li	t0, 0
;     sum += epsilon;
      74: 43 72 12 50  	fmadd.s	ft4, ft4, ft1, fa0
; 	__asm__ volatile("fsqrt.s %0, %1" : "=f" (result) : "f" (x));
      78: 53 72 02 58  	fsqrt.s	ft4, ft4
      7c: 53 72 40 18  	fdiv.s	ft4, ft0, ft4
      80: 13 83 07 00  	mv	t1, a5
      84: 7b c0 a7 01  	<unknown>
;           ((data_in[j + i * lastDimLength] - mean) / std) * scale[j] + bias[j];
      88: b3 03 55 00  	add	t2, a0, t0
      8c: 87 a2 03 00  	flw	ft5, 0(t2)
      90: b3 03 56 00  	add	t2, a2, t0
      94: 07 a3 03 00  	flw	ft6, 0(t2)
      98: b3 83 56 00  	add	t2, a3, t0
      9c: 87 a3 03 00  	flw	ft7, 0(t2)
      a0: d3 f2 32 08  	fsub.s	ft5, ft5, ft3
      a4: d3 f2 62 10  	fmul.s	ft5, ft5, ft6
      a8: c3 f2 42 38  	fmadd.s	ft5, ft5, ft4, ft7
;       data_out[j + i * lastDimLength] =
      ac: b3 83 55 00  	add	t2, a1, t0
      b0: 27 a0 53 00  	fsw	ft5, 0(t2)
;     for (int j = 0; j < lastDimLength; j++) {
      b4: 13 03 f3 ff  	addi	t1, t1, -1
      b8: 93 82 42 00  	addi	t0, t0, 4
;   for (int i = 0; i < (size / lastDimLength); i++) {
      bc: 13 08 18 00  	addi	a6, a6, 1
      c0: 33 05 15 01  	add	a0, a0, a7
      c4: b3 85 15 01  	add	a1, a1, a7
      c8: e3 10 e8 f6  	bne	a6, a4, 0x28 <Layernorm_fp32_fp32+0x28>
      cc: 6f 00 00 01  	j	0xdc <Layernorm_fp32_fp32+0xdc>
      d0: 7b 40 47 00  	<unknown>
      d4: 13 00 00 00  	nop
; 	__asm__ volatile("fsqrt.s %0, %1" : "=f" (result) : "f" (x));
      d8: 53 70 05 58  	fsqrt.s	ft0, fa0
; }
      dc: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Layernorm_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                       Size     VMA      Type
  0                            00000000 00000000 
  1 .strtab                    00000151 00000000 
  2 .text                      00000000 00000000 TEXT
  3 .text._plp_sqrt_q32        00000058 00000000 TEXT
  4 .text.Layernorm_s8_s8      00000228 00000000 TEXT
  5 .rela.text.Layernorm_s8_s8 0000000c 00000000 
  6 .debug_loclists            000002f4 00000000 DEBUG
  7 .debug_abbrev              00000101 00000000 DEBUG
  8 .debug_info                0000022d 00000000 DEBUG
  9 .rela.debug_info           000000f0 00000000 
 10 .debug_rnglists            00000018 00000000 DEBUG
 11 .debug_str_offsets         000000b8 00000000 DEBUG
 12 .rela.debug_str_offsets    00000210 00000000 
 13 .debug_str                 00000257 00000000 DEBUG
 14 .debug_addr                00000024 00000000 DEBUG
 15 .rela.debug_addr           00000054 00000000 
 16 .comment                   00000073 00000000 
 17 .note.GNU-stack            00000000 00000000 
 18 .riscv.attributes          00000030 00000000 
 19 .debug_frame               00000054 00000000 DEBUG
 20 .rela.debug_frame          00000090 00000000 
 21 .debug_line                000002bf 00000000 DEBUG
 22 .rela.debug_line           0000069c 00000000 
 23 .debug_line_str            00000145 00000000 DEBUG
 24 .llvm_addrsig              00000000 00000000 
 25 .symtab                    000008e0 00000000 

Disassembly of section .text._plp_sqrt_q32:

00000000 <_plp_sqrt_q32>:
;   int32_t number = *pSrc;
       0: 03 25 05 00  	lw	a0, 0(a0)
;   if (number > 0) {
       4: 63 56 a0 04  	blez	a0, 0x50 <_plp_sqrt_q32+0x50>
       8: 93 06 00 00  	li	a3, 0
       c: 93 07 00 00  	li	a5, 0
      10: 37 b7 00 00  	lui	a4, 11
      14: 13 08 67 50  	addi	a6, a4, 1286
      18: 6f 00 00 01  	j	0x28 <_plp_sqrt_q32+0x28>
      1c: 13 08 f7 ff  	addi	a6, a4, -1
      20: 13 87 06 00  	mv	a4, a3
;     while (start <= end) {
      24: 63 42 f8 02  	blt	a6, a5, 0x48 <_plp_sqrt_q32+0x48>
;       mid = (start + end) >> 1;
      28: 5b 27 f8 02  	<unknown>
;       if (((mid * mid) >> fracBits) == number) {
      2c: b3 08 e7 02  	mul	a7, a4, a4
      30: b3 d8 b8 00  	srl	a7, a7, a1
      34: 63 8a a8 00  	beq	a7, a0, 0x48 <_plp_sqrt_q32+0x48>
;       if (((mid * mid) >> fracBits) < number) {
      38: e3 d2 a8 fe  	bge	a7, a0, 0x1c <_plp_sqrt_q32+0x1c>
      3c: 93 07 17 00  	addi	a5, a4, 1
      40: 93 06 07 00  	mv	a3, a4
;     while (start <= end) {
      44: e3 52 f8 fe  	bge	a6, a5, 0x28 <_plp_sqrt_q32+0x28>
      48: 23 20 e6 00  	sw	a4, 0(a2)
; }
      4c: 67 80 00 00  	ret
      50: 23 20 06 00  	sw	zero, 0(a2)
; }
      54: 67 80 00 00  	ret

Disassembly of section .text.Layernorm_s8_s8:

00000000 <Layernorm_s8_s8>:
;                      int32_t lastDimLength, int32_t log2D) {
       0: 13 01 01 fb  	addi	sp, sp, -80
       4: 23 26 11 04  	sw	ra, 76(sp)
       8: 23 24 81 04  	sw	s0, 72(sp)
       c: 23 22 91 04  	sw	s1, 68(sp)
      10: 23 20 21 05  	sw	s2, 64(sp)
      14: 23 2e 31 03  	sw	s3, 60(sp)
      18: 23 2c 41 03  	sw	s4, 56(sp)
      1c: 23 2a 51 03  	sw	s5, 52(sp)
      20: 23 28 61 03  	sw	s6, 48(sp)
      24: 23 26 71 03  	sw	s7, 44(sp)
      28: 23 24 81 03  	sw	s8, 40(sp)
      2c: 23 22 91 03  	sw	s9, 36(sp)
      30: 23 20 a1 03  	sw	s10, 32(sp)
      34: 23 2e b1 01  	sw	s11, 28(sp)
      38: 23 2a d1 00  	sw	a3, 20(sp)
      3c: 23 28 c1 00  	sw	a2, 16(sp)
      40: 23 2c f1 00  	sw	a5, 24(sp)
;   for (int lastDimStart = 0; lastDimStart < size;
      44: 63 54 f0 1a  	blez	a5, 0x1ec <Layernorm_s8_s8+0x1ec>
      48: 13 84 08 00  	mv	s0, a7
      4c: 93 04 08 00  	mv	s1, a6
      50: 93 09 07 00  	mv	s3, a4
      54: 13 8b 05 00  	mv	s6, a1
      58: 93 0b 05 00  	mv	s7, a0
      5c: 93 0c 00 00  	li	s9, 0
      60: 37 b5 00 00  	lui	a0, 11
      64: 13 05 65 50  	addi	a0, a0, 1286
      68: 23 26 a1 00  	sw	a0, 12(sp)
      6c: 6f 00 80 01  	j	0x84 <Layernorm_s8_s8+0x84>
;        lastDimStart += lastDimLength) {
      70: b3 8c 9c 00  	add	s9, s9, s1
;   for (int lastDimStart = 0; lastDimStart < size;
      74: b3 8b 9b 00  	add	s7, s7, s1
      78: 33 0b 9b 00  	add	s6, s6, s1
      7c: 03 25 81 01  	lw	a0, 24(sp)
      80: 63 d6 ac 16  	bge	s9, a0, 0x1ec <Layernorm_s8_s8+0x1ec>
      84: 13 05 00 00  	li	a0, 0
      88: 93 05 00 00  	li	a1, 0
;     for (int j = 0; j < lastDimLength; j++) {
      8c: 63 54 90 04  	blez	s1, 0xd4 <Layernorm_s8_s8+0xd4>
      90: 7b c0 a4 00  	<unknown>
;       mean += data_in[lastDimStart + j] + input_offset;
      94: 33 86 ab 00  	add	a2, s7, a0
      98: 03 06 06 00  	lb	a2, 0(a2)
      9c: b3 85 35 01  	add	a1, a1, s3
;     for (int j = 0; j < lastDimLength; j++) {
      a0: 13 05 15 00  	addi	a0, a0, 1
;       mean += data_in[lastDimStart + j] + input_offset;
      a4: b3 85 c5 00  	add	a1, a1, a2
;     mean = mean / lastDimLength;
      a8: 33 c5 95 02  	div	a0, a1, s1
;     for (int j = 0; j < lastDimLength; j++) {
      ac: 63 5a 90 06  	blez	s1, 0x120 <Layernorm_s8_s8+0x120>
      b0: 13 06 00 00  	li	a2, 0
      b4: 93 05 00 00  	li	a1, 0
      b8: b3 86 a9 40  	sub	a3, s3, a0
;     for (int j = 0; j < lastDimLength; j++) {
      bc: 7b c0 a4 00  	<unknown>
;       temp = (int16_t)(data_in[lastDimStart + j] + input_offset - mean);
      c0: 33 87 cb 00  	add	a4, s7, a2
      c4: 03 07 07 00  	lb	a4, 0(a4)
      c8: 33 87 e6 00  	add	a4, a3, a4
;     for (int j = 0; j < lastDimLength; j++) {
      cc: 13 06 16 00  	addi	a2, a2, 1
;       sum += temp * temp;
      d0: db 15 e7 80  	<unknown>
      d4: 13 0c 00 00  	li	s8, 0
;     sum = sum / lastDimLength;
      d8: b3 c5 95 02  	div	a1, a1, s1
      dc: 63 ca 05 04  	bltz	a1, 0x130 <Layernorm_s8_s8+0x130>
      e0: 93 06 00 00  	li	a3, 0
      e4: 13 86 15 00  	addi	a2, a1, 1
      e8: 03 27 c1 00  	lw	a4, 12(sp)
      ec: 6f 00 00 01  	j	0xfc <Layernorm_s8_s8+0xfc>
      f0: 93 86 17 00  	addi	a3, a5, 1
      f4: 13 8c 07 00  	mv	s8, a5
;     while (start <= end) {
      f8: 63 4c d7 02  	blt	a4, a3, 0x130 <Layernorm_s8_s8+0x130>
;       mid = (start + end) >> 1;
      fc: db a7 e6 02  	<unknown>
;       if (((mid * mid) >> fracBits) == number) {
     100: 33 88 f7 02  	mul	a6, a5, a5
     104: 63 0a c8 00  	beq	a6, a2, 0x118 <Layernorm_s8_s8+0x118>
;       if (((mid * mid) >> fracBits) < number) {
     108: e3 d4 05 ff  	bge	a1, a6, 0xf0 <Layernorm_s8_s8+0xf0>
     10c: 13 87 f7 ff  	addi	a4, a5, -1
;     while (start <= end) {
     110: e3 56 d7 fe  	bge	a4, a3, 0xfc <Layernorm_s8_s8+0xfc>
     114: 6f 00 c0 01  	j	0x130 <Layernorm_s8_s8+0x130>
     118: 13 8c 07 00  	mv	s8, a5
     11c: 6f 00 40 01  	j	0x130 <Layernorm_s8_s8+0x130>
     120: 93 05 00 00  	li	a1, 0
     124: 13 0c 00 00  	li	s8, 0
;     sum = sum / lastDimLength;
     128: b3 c5 95 02  	div	a1, a1, s1
     12c: e3 da 05 fa  	bgez	a1, 0xe0 <Layernorm_s8_s8+0xe0>
;     for (int j = 0; j < lastDimLength; j++) {
     130: e3 50 90 f4  	blez	s1, 0x70 <Layernorm_s8_s8+0x70>
     134: 93 0d 00 00  	li	s11, 0
     138: 93 d5 f9 41  	srai	a1, s3, 31
     13c: 13 56 f5 41  	srai	a2, a0, 31
     140: b3 b6 a9 00  	sltu	a3, s3, a0
     144: b3 85 c5 40  	sub	a1, a1, a2
     148: 33 8a d5 40  	sub	s4, a1, a3
     14c: b3 8a a9 40  	sub	s5, s3, a0
     150: 03 2d 01 01  	lw	s10, 16(sp)
     154: 03 29 41 01  	lw	s2, 20(sp)
     158: 6f 00 80 01  	j	0x170 <Layernorm_s8_s8+0x170>
     15c: db a5 e6 40  	<unknown>
;       data_out[lastDimStart + j] =
     160: 33 05 bb 01  	add	a0, s6, s11
;     for (int j = 0; j < lastDimLength; j++) {
     164: 93 8d 1d 00  	addi	s11, s11, 1
;       data_out[lastDimStart + j] =
     168: 23 00 b5 00  	sb	a1, 0(a0)
;     for (int j = 0; j < lastDimLength; j++) {
     16c: e3 82 b4 f1  	beq	s1, s11, 0x70 <Layernorm_s8_s8+0x70>
;           (int8_t)((((((int64_t)data_in[lastDimStart + j]) + input_offset -
     170: 33 85 bb 01  	add	a0, s7, s11
     174: 03 05 05 00  	lb	a0, 0(a0)
     178: 93 56 fc 41  	srai	a3, s8, 31
     17c: 93 55 f5 41  	srai	a1, a0, 31
     180: 33 85 aa 00  	add	a0, s5, a0
;                      weight[j]) /
     184: 0b 26 4d 00  	<unknown>
;           (int8_t)((((((int64_t)data_in[lastDimStart + j]) + input_offset -
     188: 33 37 55 01  	sltu	a4, a0, s5
     18c: b3 05 ba 00  	add	a1, s4, a1
     190: 33 87 e5 00  	add	a4, a1, a4
;                      weight[j]) /
     194: 93 57 f6 41  	srai	a5, a2, 31
;                       mean) *
     198: b3 35 c5 02  	mulhu	a1, a0, a2
     19c: b3 05 f5 42  	<unknown>
     1a0: b3 05 c7 42  	<unknown>
     1a4: 33 05 c5 02  	mul	a0, a0, a2
;                      weight[j]) /
     1a8: 13 06 0c 00  	mv	a2, s8
     1ac: 97 00 00 00  	auipc	ra, 0
     1b0: e7 80 00 00  	jalr	ra
;                     bias[j]) >>
     1b4: 0b 26 49 00  	<unknown>
     1b8: 93 57 f6 41  	srai	a5, a2, 31
;                         (std) +
     1bc: b3 06 c5 00  	add	a3, a0, a2
     1c0: b3 b6 a6 00  	sltu	a3, a3, a0
;                     bias[j]) >>
     1c4: 13 07 04 fe  	addi	a4, s0, -32
;                         (std) +
     1c8: b3 85 f5 00  	add	a1, a1, a5
     1cc: e3 58 07 f8  	bgez	a4, 0x15c <Layernorm_s8_s8+0x15c>
     1d0: b3 85 d5 00  	add	a1, a1, a3
     1d4: 93 95 15 00  	slli	a1, a1, 1
     1d8: 93 46 f4 01  	xori	a3, s0, 31
     1dc: b3 95 d5 00  	sll	a1, a1, a3
     1e0: 5b 25 86 c0  	<unknown>
     1e4: b3 65 b5 00  	or	a1, a0, a1
     1e8: 6f f0 9f f7  	j	0x160 <Layernorm_s8_s8+0x160>
; }
     1ec: 83 20 c1 04  	lw	ra, 76(sp)
     1f0: 03 24 81 04  	lw	s0, 72(sp)
     1f4: 83 24 41 04  	lw	s1, 68(sp)
     1f8: 03 29 01 04  	lw	s2, 64(sp)
     1fc: 83 29 c1 03  	lw	s3, 60(sp)
     200: 03 2a 81 03  	lw	s4, 56(sp)
     204: 83 2a 41 03  	lw	s5, 52(sp)
     208: 03 2b 01 03  	lw	s6, 48(sp)
     20c: 83 2b c1 02  	lw	s7, 44(sp)
     210: 03 2c 81 02  	lw	s8, 40(sp)
     214: 83 2c 41 02  	lw	s9, 36(sp)
     218: 03 2d 01 02  	lw	s10, 32(sp)
     21c: 83 2d c1 01  	lw	s11, 28(sp)
     220: 13 01 01 05  	addi	sp, sp, 80
     224: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(MatMul_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                  Size     VMA      Type
  0                                       00000000 00000000 
  1 .strtab                               0000015a 00000000 
  2 .text                                 00000000 00000000 TEXT
  3 .text.MatMul_fp32_fp32_fp32           000000b4 00000000 TEXT
  4 .text.MatMul_fp32_fp32_fp32_unroll1x7 0000028c 00000000 TEXT
  5 .debug_loclists                       000006bb 00000000 DEBUG
  6 .debug_abbrev                         0000009c 00000000 DEBUG
  7 .debug_info                           000001fb 00000000 DEBUG
  8 .rela.debug_info                      000000a8 00000000 
  9 .debug_rnglists                       0000008a 00000000 DEBUG
 10 .debug_str_offsets                    000000a4 00000000 DEBUG
 11 .rela.debug_str_offsets               000001d4 00000000 
 12 .debug_str                            000001d7 00000000 DEBUG
 13 .debug_addr                           00000014 00000000 DEBUG
 14 .rela.debug_addr                      00000024 00000000 
 15 .comment                              00000073 00000000 
 16 .note.GNU-stack                       00000000 00000000 
 17 .riscv.attributes                     00000030 00000000 
 18 .debug_frame                          00000040 00000000 DEBUG
 19 .rela.debug_frame                     00000090 00000000 
 20 .debug_line                           000003f4 00000000 DEBUG
 21 .rela.debug_line                      00000a80 00000000 
 22 .debug_line_str                       00000167 00000000 DEBUG
 23 .llvm_addrsig                         00000000 00000000 
 24 .symtab                               00000ab0 00000000 

Disassembly of section .text.MatMul_fp32_fp32_fp32:

00000000 <MatMul_fp32_fp32_fp32>:
;   for (uint32_t i = 0; i < M; ++i) {
       0: 13 b8 16 00  	seqz	a6, a3
       4: 93 b8 17 00  	seqz	a7, a5
       8: 33 68 18 01  	or	a6, a6, a7
       c: 63 12 08 0a  	bnez	a6, 0xb0 <MatMul_fp32_fp32_fp32+0xb0>
;       for (uint32_t k = 0; k < N; ++k) {
      10: 63 0c 07 06  	beqz	a4, 0x88 <MatMul_fp32_fp32_fp32+0x88>
      14: 13 08 00 00  	li	a6, 0
;   for (uint32_t i = 0; i < M; ++i) {
      18: 93 98 27 00  	slli	a7, a5, 2
      1c: 93 12 27 00  	slli	t0, a4, 2
      20: 53 00 00 f0  	fmv.w.x	ft0, zero
      24: 13 03 00 00  	li	t1, 0
      28: b3 03 f8 02  	mul	t2, a6, a5
      2c: 13 8e 05 00  	mv	t3, a1
      30: fb c0 27 02  	<unknown>
      34: 93 0e 05 00  	mv	t4, a0
      38: 13 0f 0e 00  	mv	t5, t3
      3c: 93 0f 07 00  	mv	t6, a4
      40: d3 00 00 20  	fmv.s	ft1, ft0
      44: 7b 40 c7 00  	<unknown>
;         sum += pSrcA[i * N + k] * pSrcB[k * O + j];
      48: 07 a1 0e 00  	flw	ft2, 0(t4)
      4c: 87 21 0f 00  	flw	ft3, 0(t5)
      50: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;       for (uint32_t k = 0; k < N; ++k) {
      54: 93 8f ff ff  	addi	t6, t6, -1
      58: 33 0f 1f 01  	add	t5, t5, a7
      5c: 93 8e 4e 00  	addi	t4, t4, 4
;       pDstY[i * O + j] = sum;
      60: b3 0e 73 00  	add	t4, t1, t2
      64: 93 9e 2e 00  	slli	t4, t4, 2
      68: b3 0e d6 01  	add	t4, a2, t4
      6c: 27 a0 1e 00  	fsw	ft1, 0(t4)
;     for (uint32_t j = 0; j < O; ++j) {
      70: 13 03 13 00  	addi	t1, t1, 1
      74: 13 0e 4e 00  	addi	t3, t3, 4
;   for (uint32_t i = 0; i < M; ++i) {
      78: 13 08 18 00  	addi	a6, a6, 1
      7c: 33 05 55 00  	add	a0, a0, t0
      80: e3 12 d8 fa  	bne	a6, a3, 0x24 <MatMul_fp32_fp32_fp32+0x24>
      84: 6f 00 c0 02  	j	0xb0 <MatMul_fp32_fp32_fp32+0xb0>
      88: 13 05 00 00  	li	a0, 0
;   for (uint32_t i = 0; i < M; ++i) {
      8c: 93 95 27 00  	slli	a1, a5, 2
      90: fb c0 e6 00  	<unknown>
      94: 13 07 06 00  	mv	a4, a2
      98: 13 88 07 00  	mv	a6, a5
      9c: 7b c0 47 00  	<unknown>
;     for (uint32_t j = 0; j < O; ++j) {
      a0: 13 08 f8 ff  	addi	a6, a6, -1
;       pDstY[i * O + j] = sum;
      a4: 2b 22 07 00  	<unknown>
;   for (uint32_t i = 0; i < M; ++i) {
      a8: 13 05 15 00  	addi	a0, a0, 1
      ac: 33 06 b6 00  	add	a2, a2, a1
; }
      b0: 67 80 00 00  	ret

Disassembly of section .text.MatMul_fp32_fp32_fp32_unroll1x7:

00000000 <MatMul_fp32_fp32_fp32_unroll1x7>:
;                                      uint32_t N, uint32_t O) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
       8: 23 24 91 00  	sw	s1, 8(sp)
       c: 23 22 21 01  	sw	s2, 4(sp)
      10: 23 20 31 01  	sw	s3, 0(sp)
;   for (i = 0; i < M; i++) {
      14: 63 86 06 22  	beqz	a3, 0x240 <MatMul_fp32_fp32_fp32_unroll1x7+0x240>
      18: 37 58 92 24  	lui	a6, 149797
      1c: 13 08 58 92  	addi	a6, a6, -1755
      20: 33 b8 07 03  	mulhu	a6, a5, a6
      24: db b8 07 83  	<unknown>
      28: 5b a8 08 85  	<unknown>
      2c: 93 18 38 00  	slli	a7, a6, 3
      30: 33 88 08 41  	sub	a6, a7, a6
      34: b3 88 07 41  	sub	a7, a5, a6
;     for (j = 0; j < O_block; j += 7) {
      38: 63 96 f8 08  	bne	a7, a5, 0xc4 <MatMul_fp32_fp32_fp32_unroll1x7+0xc4>
;     for (j = O_block; j < O; j++) {
      3c: 63 72 f8 20  	bgeu	a6, a5, 0x240 <MatMul_fp32_fp32_fp32_unroll1x7+0x240>
;       for (k = 0; k < N; k++) {
      40: 63 0c 07 20  	beqz	a4, 0x258 <MatMul_fp32_fp32_fp32_unroll1x7+0x258>
      44: 93 08 00 00  	li	a7, 0
;   for (i = 0; i < M; i++) {
      48: 93 12 28 00  	slli	t0, a6, 2
      4c: b3 85 55 00  	add	a1, a1, t0
      50: 93 92 27 00  	slli	t0, a5, 2
      54: 13 13 27 00  	slli	t1, a4, 2
      58: 53 00 00 f0  	fmv.w.x	ft0, zero
      5c: 33 8f 07 41  	sub	t5, a5, a6
      60: b3 83 f8 02  	mul	t2, a7, a5
      64: 13 8e 05 00  	mv	t3, a1
      68: 93 0e 08 00  	mv	t4, a6
      6c: fb 40 2f 02  	<unknown>
      70: 13 0f 05 00  	mv	t5, a0
      74: 93 0f 0e 00  	mv	t6, t3
      78: 13 04 07 00  	mv	s0, a4
      7c: d3 00 00 20  	fmv.s	ft1, ft0
      80: 7b 40 c7 00  	<unknown>
;         float32_t a_val = pSrcA[i * N + k];
      84: 07 21 0f 00  	flw	ft2, 0(t5)
;         float32_t b_val = pSrcB[k * O + j];
      88: 87 a1 0f 00  	flw	ft3, 0(t6)
;         sum += prod;
      8c: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;       for (k = 0; k < N; k++) {
      90: 13 04 f4 ff  	addi	s0, s0, -1
      94: b3 8f 5f 00  	add	t6, t6, t0
      98: 13 0f 4f 00  	addi	t5, t5, 4
;       pDstY[i * O + j] = sum;
      9c: 33 8f 7e 00  	add	t5, t4, t2
      a0: 13 1f 2f 00  	slli	t5, t5, 2
      a4: 33 0f e6 01  	add	t5, a2, t5
      a8: 27 20 1f 00  	fsw	ft1, 0(t5)
;     for (j = O_block; j < O; j++) {
      ac: 93 8e 1e 00  	addi	t4, t4, 1
      b0: 13 0e 4e 00  	addi	t3, t3, 4
;   for (i = 0; i < M; i++) {
      b4: 93 88 18 00  	addi	a7, a7, 1
      b8: 33 05 65 00  	add	a0, a0, t1
      bc: e3 90 d8 fa  	bne	a7, a3, 0x5c <MatMul_fp32_fp32_fp32_unroll1x7+0x5c>
      c0: 6f 00 00 18  	j	0x240 <MatMul_fp32_fp32_fp32_unroll1x7+0x240>
      c4: 93 02 00 00  	li	t0, 0
;   for (i = 0; i < M; i++) {
      c8: 13 13 27 00  	slli	t1, a4, 2
      cc: 93 83 c5 00  	addi	t2, a1, 12
      d0: 13 9e 27 00  	slli	t3, a5, 2
      d4: 93 1e 28 00  	slli	t4, a6, 2
      d8: b3 85 d5 01  	add	a1, a1, t4
      dc: b3 0e d6 01  	add	t4, a2, t4
      e0: 53 00 00 f0  	fmv.w.x	ft0, zero
      e4: fb c0 c6 0a  	<unknown>
      e8: 93 0f 00 00  	li	t6, 0
      ec: 33 8f f2 02  	mul	t5, t0, a5
      f0: 13 84 03 00  	mv	s0, t2
      f4: 6f 00 80 03  	j	0x12c <MatMul_fp32_fp32_fp32_unroll1x7+0x12c>
;       pDstY[i * O + (j + 0)] = sum0;
      f8: b3 84 ef 01  	add	s1, t6, t5
      fc: 93 94 24 00  	slli	s1, s1, 2
     100: b3 04 96 00  	add	s1, a2, s1
     104: 27 a0 74 00  	fsw	ft7, 0(s1)
;       pDstY[i * O + (j + 1)] = sum1;
     108: 27 a2 64 00  	fsw	ft6, 4(s1)
;       pDstY[i * O + (j + 2)] = sum2;
     10c: 27 a4 54 00  	fsw	ft5, 8(s1)
;       pDstY[i * O + (j + 3)] = sum3;
     110: 27 a6 44 00  	fsw	ft4, 12(s1)
;       pDstY[i * O + (j + 4)] = sum4;
     114: 27 a8 34 00  	fsw	ft3, 16(s1)
;       pDstY[i * O + (j + 5)] = sum5;
     118: 27 aa 24 00  	fsw	ft2, 20(s1)
;       pDstY[i * O + (j + 6)] = sum6;
     11c: 27 ac 14 00  	fsw	ft1, 24(s1)
;     for (j = 0; j < O_block; j += 7) {
     120: 93 8f 7f 00  	addi	t6, t6, 7
     124: 13 04 c4 01  	addi	s0, s0, 28
     128: 63 fe 0f 09  	bgeu	t6, a6, 0x1c4 <MatMul_fp32_fp32_fp32_unroll1x7+0x1c4>
     12c: d3 00 00 20  	fmv.s	ft1, ft0
     130: 53 01 00 20  	fmv.s	ft2, ft0
     134: d3 01 00 20  	fmv.s	ft3, ft0
     138: 53 02 00 20  	fmv.s	ft4, ft0
     13c: d3 02 00 20  	fmv.s	ft5, ft0
     140: 53 03 00 20  	fmv.s	ft6, ft0
     144: d3 03 00 20  	fmv.s	ft7, ft0
;       for (k = 0; k < N; k++) {
     148: e3 08 07 fa  	beqz	a4, 0xf8 <MatMul_fp32_fp32_fp32_unroll1x7+0xf8>
     14c: 93 04 04 00  	mv	s1, s0
     150: 13 09 05 00  	mv	s2, a0
     154: 93 09 07 00  	mv	s3, a4
     158: d3 03 00 20  	fmv.s	ft7, ft0
     15c: 53 03 00 20  	fmv.s	ft6, ft0
     160: d3 02 00 20  	fmv.s	ft5, ft0
     164: 53 02 00 20  	fmv.s	ft4, ft0
     168: d3 01 00 20  	fmv.s	ft3, ft0
     16c: 53 01 00 20  	fmv.s	ft2, ft0
     170: d3 00 00 20  	fmv.s	ft1, ft0
;       for (k = 0; k < N; k++) {
     174: 7b 40 47 02  	<unknown>
;         float32_t a0 = pSrcA[i * N + k];
     178: 07 25 09 00  	flw	fa0, 0(s2)
;         float32_t b0 = pSrcB[k * O + (j + 0)];
     17c: 87 a5 44 ff  	flw	fa1, -12(s1)
;         float32_t b1 = pSrcB[k * O + (j + 1)];
     180: 07 a6 84 ff  	flw	fa2, -8(s1)
;         float32_t b2 = pSrcB[k * O + (j + 2)];
     184: 87 a6 c4 ff  	flw	fa3, -4(s1)
;         float32_t b3 = pSrcB[k * O + (j + 3)];
     188: 07 a7 04 00  	flw	fa4, 0(s1)
;         float32_t b4 = pSrcB[k * O + (j + 4)];
     18c: 87 a7 44 00  	flw	fa5, 4(s1)
;         float32_t b5 = pSrcB[k * O + (j + 5)];
     190: 07 a8 84 00  	flw	fa6, 8(s1)
;         float32_t b6 = pSrcB[k * O + (j + 6)];
     194: 87 a8 c4 00  	flw	fa7, 12(s1)
;         sum0 += a0 * b0;
     198: c3 f3 a5 38  	fmadd.s	ft7, fa1, fa0, ft7
;         sum1 += a0 * b1;
     19c: 43 73 a6 30  	fmadd.s	ft6, fa2, fa0, ft6
;         sum2 += a0 * b2;
     1a0: c3 f2 a6 28  	fmadd.s	ft5, fa3, fa0, ft5
;         sum3 += a0 * b3;
     1a4: 43 72 a7 20  	fmadd.s	ft4, fa4, fa0, ft4
;         sum4 += a0 * b4;
     1a8: c3 f1 a7 18  	fmadd.s	ft3, fa5, fa0, ft3
;         sum5 += a0 * b5;
     1ac: 43 71 a8 10  	fmadd.s	ft2, fa6, fa0, ft2
;         sum6 += a0 * b6;
     1b0: c3 f0 a8 08  	fmadd.s	ft1, fa7, fa0, ft1
;       for (k = 0; k < N; k++) {
     1b4: 93 89 f9 ff  	addi	s3, s3, -1
     1b8: 13 09 49 00  	addi	s2, s2, 4
     1bc: b3 84 c4 01  	add	s1, s1, t3
     1c0: 6f f0 9f f3  	j	0xf8 <MatMul_fp32_fp32_fp32_unroll1x7+0xf8>
;     for (j = O_block; j < O; j++) {
     1c4: 63 78 f8 06  	bgeu	a6, a5, 0x234 <MatMul_fp32_fp32_fp32_unroll1x7+0x234>
     1c8: 93 84 0e 00  	mv	s1, t4
     1cc: 13 89 08 00  	mv	s2, a7
     1d0: 93 8f 05 00  	mv	t6, a1
     1d4: 13 04 08 00  	mv	s0, a6
;       for (k = 0; k < N; k++) {
     1d8: 63 08 07 04  	beqz	a4, 0x228 <MatMul_fp32_fp32_fp32_unroll1x7+0x228>
     1dc: 93 04 05 00  	mv	s1, a0
     1e0: 13 89 0f 00  	mv	s2, t6
     1e4: 93 09 07 00  	mv	s3, a4
     1e8: d3 00 00 20  	fmv.s	ft1, ft0
     1ec: 7b 40 c7 00  	<unknown>
;         float32_t a_val = pSrcA[i * N + k];
     1f0: 07 a1 04 00  	flw	ft2, 0(s1)
;         float32_t b_val = pSrcB[k * O + j];
     1f4: 87 21 09 00  	flw	ft3, 0(s2)
;         sum += prod;
     1f8: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;       for (k = 0; k < N; k++) {
     1fc: 93 89 f9 ff  	addi	s3, s3, -1
     200: 33 09 c9 01  	add	s2, s2, t3
     204: 93 84 44 00  	addi	s1, s1, 4
;       pDstY[i * O + j] = sum;
     208: b3 04 e4 01  	add	s1, s0, t5
     20c: 93 94 24 00  	slli	s1, s1, 2
     210: b3 04 96 00  	add	s1, a2, s1
     214: 27 a0 14 00  	fsw	ft1, 0(s1)
;     for (j = O_block; j < O; j++) {
     218: 13 04 14 00  	addi	s0, s0, 1
     21c: 93 8f 4f 00  	addi	t6, t6, 4
     220: e3 1e f4 fa  	bne	s0, a5, 0x1dc <MatMul_fp32_fp32_fp32_unroll1x7+0x1dc>
     224: 6f 00 00 01  	j	0x234 <MatMul_fp32_fp32_fp32_unroll1x7+0x234>
     228: 13 09 f9 ff  	addi	s2, s2, -1
;       pDstY[i * O + j] = sum;
     22c: 2b a2 04 00  	<unknown>
;     for (j = O_block; j < O; j++) {
     230: e3 1c 09 fe  	bnez	s2, 0x228 <MatMul_fp32_fp32_fp32_unroll1x7+0x228>
;   for (i = 0; i < M; i++) {
     234: 93 82 12 00  	addi	t0, t0, 1
     238: 33 05 65 00  	add	a0, a0, t1
     23c: b3 8e ce 01  	add	t4, t4, t3
; }
     240: 03 24 c1 00  	lw	s0, 12(sp)
     244: 83 24 81 00  	lw	s1, 8(sp)
     248: 03 29 41 00  	lw	s2, 4(sp)
     24c: 83 29 01 00  	lw	s3, 0(sp)
     250: 13 01 01 01  	addi	sp, sp, 16
     254: 67 80 00 00  	ret
     258: 13 05 00 00  	li	a0, 0
;   for (i = 0; i < M; i++) {
     25c: 93 15 28 00  	slli	a1, a6, 2
     260: b3 05 b6 00  	add	a1, a2, a1
     264: 13 96 27 00  	slli	a2, a5, 2
     268: fb c0 e6 00  	<unknown>
     26c: 13 87 05 00  	mv	a4, a1
     270: 93 87 08 00  	mv	a5, a7
     274: 7b c0 48 00  	<unknown>
     278: 13 00 00 00  	nop
;       pDstY[i * O + j] = sum;
     27c: 2b 22 07 00  	<unknown>
;   for (i = 0; i < M; i++) {
     280: 13 05 15 00  	addi	a0, a0, 1
     284: b3 85 c5 00  	add	a1, a1, a2
     288: 6f f0 9f fb  	j	0x240 <MatMul_fp32_fp32_fp32_unroll1x7+0x240>

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(MatMul_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 0000012d 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text.MatMul_s8_s8_s32  000004c4 00000000 TEXT
  4 .debug_loclists         00000969 00000000 DEBUG
  5 .debug_abbrev           0000008f 00000000 DEBUG
  6 .debug_info             000001b9 00000000 DEBUG
  7 .rela.debug_info        000000c0 00000000 
  8 .debug_rnglists         00000048 00000000 DEBUG
  9 .debug_str_offsets      000000ac 00000000 DEBUG
 10 .rela.debug_str_offsets 000001ec 00000000 
 11 .debug_str              000001f8 00000000 DEBUG
 12 .debug_addr             00000018 00000000 DEBUG
 13 .rela.debug_addr        00000030 00000000 
 14 .comment                00000073 00000000 
 15 .note.GNU-stack         00000000 00000000 
 16 .riscv.attributes       00000030 00000000 
 17 .debug_frame            00000044 00000000 DEBUG
 18 .rela.debug_frame       00000060 00000000 
 19 .debug_line             000004dc 00000000 DEBUG
 20 .rela.debug_line        00000d38 00000000 
 21 .debug_line_str         0000013f 00000000 DEBUG
 22 .llvm_addrsig           00000000 00000000 
 23 .symtab                 00000c90 00000000 

Disassembly of section .text.MatMul_s8_s8_s32:

00000000 <MatMul_s8_s8_s32>:
;                       int32_t C_offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 83 22 01 06  	lw	t0, 96(sp)
      3c: 13 03 20 00  	li	t1, 2
      40: 93 8c 05 00  	mv	s9, a1
      44: 13 0d 05 00  	mv	s10, a0
;   for (i = 0; i < M / 2; i++) {
      48: 63 fa 66 00  	bgeu	a3, t1, 0x5c <MatMul_s8_s8_s32+0x5c>
      4c: 13 09 00 00  	li	s2, 0
      50: 13 03 00 00  	li	t1, 0
      54: 93 0d 00 00  	li	s11, 0
      58: 6f 00 00 20  	j	0x258 <MatMul_s8_s8_s32+0x258>
      5c: 93 03 10 00  	li	t2, 1
      60: 13 d5 16 00  	srli	a0, a3, 1
;     for (k = 0; k < P / 2; k++) {
      64: 63 ea f3 00  	bltu	t2, a5, 0x78 <MatMul_s8_s8_s32+0x78>
      68: 13 09 00 00  	li	s2, 0
      6c: 13 03 00 00  	li	t1, 0
;   for (i = 0; i < M / 2; i++) {
      70: b3 7d 75 04  	<unknown>
      74: 6f 00 40 1e  	j	0x258 <MatMul_s8_s8_s32+0x258>
      78: 13 03 20 00  	li	t1, 2
      7c: 93 d3 17 00  	srli	t2, a5, 1
;       for (j = 0; j < N / 2; j++) {
      80: 63 7c 67 04  	bgeu	a4, t1, 0xd8 <MatMul_s8_s8_s32+0xd8>
      84: 13 03 00 00  	li	t1, 0
      88: 93 0e 10 00  	li	t4, 1
;   for (i = 0; i < M / 2; i++) {
      8c: 33 f9 d3 05  	<unknown>
      90: b3 7d d5 05  	<unknown>
      94: 93 93 27 00  	slli	t2, a5, 2
      98: 93 9e 37 00  	slli	t4, a5, 3
      9c: 13 0f 06 00  	mv	t5, a2
      a0: 93 0f 0f 00  	mv	t6, t5
      a4: 13 04 09 00  	mv	s0, s2
      a8: 7b 40 c9 00  	<unknown>
;       pDstC[(i * 2) * P + (k * 2)] = sum00;
      ac: 23 a0 5f 00  	sw	t0, 0(t6)
;       pDstC[(i * 2) * P + (k * 2 + 1)] = sum01;
      b0: 23 a2 5f 00  	sw	t0, 4(t6)
;       pDstC[(i * 2 + 1) * P + (k * 2)] = sum10;
      b4: 33 85 7f 00  	add	a0, t6, t2
      b8: 23 20 55 00  	sw	t0, 0(a0)
;       pDstC[(i * 2 + 1) * P + (k * 2 + 1)] = sum11;
      bc: 23 22 55 00  	sw	t0, 4(a0)
;     for (k = 0; k < P / 2; k++) {
      c0: 93 8f 8f 00  	addi	t6, t6, 8
;   for (i = 0; i < M / 2; i++) {
      c4: 13 03 13 00  	addi	t1, t1, 1
      c8: 33 0f df 01  	add	t5, t5, t4
      cc: e3 1a b3 fd  	bne	t1, s11, 0xa0 <MatMul_s8_s8_s32+0xa0>
      d0: 13 03 00 00  	li	t1, 0
      d4: 6f 00 40 18  	j	0x258 <MatMul_s8_s8_s32+0x258>
      d8: 23 26 d1 00  	sw	a3, 12(sp)
      dc: 93 06 00 00  	li	a3, 0
      e0: 13 53 17 00  	srli	t1, a4, 1
      e4: 13 0f 10 00  	li	t5, 1
;   for (i = 0; i < M / 2; i++) {
      e8: b3 75 e3 05  	<unknown>
      ec: 23 20 b1 02  	sw	a1, 32(sp)
      f0: 33 f9 e3 05  	<unknown>
      f4: b3 7d e5 05  	<unknown>
      f8: 23 24 a1 01  	sw	s10, 8(sp)
      fc: 13 05 1d 00  	addi	a0, s10, 1
     100: 23 24 a1 02  	sw	a0, 40(sp)
     104: 13 15 17 00  	slli	a0, a4, 1
     108: 23 28 a1 00  	sw	a0, 16(sp)
     10c: 13 94 17 00  	slli	s0, a5, 1
     110: 23 2c 91 01  	sw	s9, 24(sp)
     114: 23 2a b1 01  	sw	s11, 20(sp)
     118: 93 04 00 00  	li	s1, 0
     11c: 23 2e d1 00  	sw	a3, 28(sp)
     120: 93 93 16 00  	slli	t2, a3, 1
     124: 93 e9 13 00  	ori	s3, t2, 1
     128: 33 85 f3 02  	mul	a0, t2, a5
     12c: 23 22 a1 02  	sw	a0, 36(sp)
     130: b3 89 f9 02  	mul	s3, s3, a5
     134: 13 8a 0c 00  	mv	s4, s9
     138: fb 40 49 07  	<unknown>
     13c: 13 9b 14 00  	slli	s6, s1, 1
     140: 93 6a 1b 00  	ori	s5, s6, 1
     144: 93 0d 0a 00  	mv	s11, s4
     148: 83 20 81 02  	lw	ra, 40(sp)
     14c: 03 25 01 02  	lw	a0, 32(sp)
     150: 93 03 05 00  	mv	t2, a0
     154: 13 8d 02 00  	mv	s10, t0
     158: 93 8c 02 00  	mv	s9, t0
     15c: 13 8c 02 00  	mv	s8, t0
     160: 93 8b 02 00  	mv	s7, t0
     164: 7b 40 85 03  	<unknown>
;         int32_t AVal00 = pSrcA[(i * 2) * N + j * 2] + A_offset;
     168: 03 85 f0 ff  	lb	a0, -1(ra)
;         int32_t AVal10 = pSrcA[(i * 2 + 1) * N + j * 2] + A_offset;
     16c: b3 85 e0 00  	add	a1, ra, a4
     170: 83 86 f5 ff  	lb	a3, -1(a1)
;         int32_t AVal01 = pSrcA[(i * 2) * N + j * 2 + 1] + A_offset;
     174: 83 8f 00 00  	lb	t6, 0(ra)
;         int32_t AVal11 = pSrcA[(i * 2 + 1) * N + j * 2 + 1] + A_offset;
     178: 83 85 05 00  	lb	a1, 0(a1)
;         int32_t AVal00 = pSrcA[(i * 2) * N + j * 2] + A_offset;
     17c: 33 05 05 01  	add	a0, a0, a6
;         int32_t AVal10 = pSrcA[(i * 2 + 1) * N + j * 2] + A_offset;
     180: b3 86 06 01  	add	a3, a3, a6
;         int32_t AVal01 = pSrcA[(i * 2) * N + j * 2 + 1] + A_offset;
     184: b3 8f 0f 01  	add	t6, t6, a6
;         int32_t AVal11 = pSrcA[(i * 2 + 1) * N + j * 2 + 1] + A_offset;
     188: b3 85 05 01  	add	a1, a1, a6
;         int32_t BVal00 = pSrcB[(j * 2) * P + (k * 2)] + B_offset;
     18c: 83 8e 0d 00  	lb	t4, 0(s11)
;         int32_t BVal01 = pSrcB[(j * 2) * P + (k * 2 + 1)] + B_offset;
     190: 03 8e 1d 00  	lb	t3, 1(s11)
;         int32_t BVal10 = pSrcB[(j * 2 + 1) * P + (k * 2)] + B_offset;
     194: 33 8f fd 00  	add	t5, s11, a5
     198: 03 03 0f 00  	lb	t1, 0(t5)
;         int32_t BVal11 = pSrcB[(j * 2 + 1) * P + (k * 2 + 1)] + B_offset;
     19c: 03 0f 1f 00  	lb	t5, 1(t5)
;         int32_t BVal00 = pSrcB[(j * 2) * P + (k * 2)] + B_offset;
     1a0: b3 8e 1e 01  	add	t4, t4, a7
;         int32_t BVal01 = pSrcB[(j * 2) * P + (k * 2 + 1)] + B_offset;
     1a4: 33 0e 1e 01  	add	t3, t3, a7
;         int32_t BVal10 = pSrcB[(j * 2 + 1) * P + (k * 2)] + B_offset;
     1a8: 33 03 13 01  	add	t1, t1, a7
;         int32_t BVal11 = pSrcB[(j * 2 + 1) * P + (k * 2 + 1)] + B_offset;
     1ac: 33 0f 1f 01  	add	t5, t5, a7
;         sum00 = sum00 + AVal00 * BVal00;
     1b0: 33 8d ae 42  	<unknown>
;         sum00 = sum00 + AVal01 * BVal10;
     1b4: 33 0d f3 43  	<unknown>
;         sum01 = sum01 + AVal00 * BVal01;
     1b8: b3 0c ae 42  	<unknown>
;         sum01 = sum01 + AVal01 * BVal11;
     1bc: b3 0c ff 43  	<unknown>
;         sum10 = sum10 + AVal10 * BVal00;
     1c0: 33 8c de 42  	<unknown>
;         sum10 = sum10 + AVal11 * BVal10;
     1c4: 33 0c b3 42  	<unknown>
;         sum11 = sum11 + AVal10 * BVal01;
     1c8: b3 0b de 42  	<unknown>
;         sum11 = sum11 + AVal11 * BVal11;
     1cc: b3 0b bf 42  	<unknown>
;       for (j = 0; j < N / 2; j++) {
     1d0: 93 80 20 00  	addi	ra, ra, 2
     1d4: b3 8d 8d 00  	add	s11, s11, s0
     1d8: 83 25 41 02  	lw	a1, 36(sp)
;       pDstC[(i * 2) * P + (k * 2)] = sum00;
     1dc: 33 05 bb 00  	add	a0, s6, a1
     1e0: 13 15 25 00  	slli	a0, a0, 2
     1e4: 33 05 a6 00  	add	a0, a2, a0
     1e8: 23 20 a5 01  	sw	s10, 0(a0)
;       pDstC[(i * 2) * P + (k * 2 + 1)] = sum01;
     1ec: 33 85 ba 00  	add	a0, s5, a1
     1f0: 13 15 25 00  	slli	a0, a0, 2
     1f4: 33 05 a6 00  	add	a0, a2, a0
     1f8: 23 20 95 01  	sw	s9, 0(a0)
;       pDstC[(i * 2 + 1) * P + (k * 2)] = sum10;
     1fc: 33 05 3b 01  	add	a0, s6, s3
     200: 13 15 25 00  	slli	a0, a0, 2
     204: 33 05 a6 00  	add	a0, a2, a0
     208: 23 20 85 01  	sw	s8, 0(a0)
;       pDstC[(i * 2 + 1) * P + (k * 2 + 1)] = sum11;
     20c: 33 85 3a 01  	add	a0, s5, s3
     210: 13 15 25 00  	slli	a0, a0, 2
     214: 33 05 a6 00  	add	a0, a2, a0
     218: 23 20 75 01  	sw	s7, 0(a0)
;     for (k = 0; k < P / 2; k++) {
     21c: 93 84 14 00  	addi	s1, s1, 1
     220: 13 0a 2a 00  	addi	s4, s4, 2
     224: 83 26 c1 01  	lw	a3, 28(sp)
;   for (i = 0; i < M / 2; i++) {
     228: 93 86 16 00  	addi	a3, a3, 1
     22c: 03 25 81 02  	lw	a0, 40(sp)
     230: 83 25 01 01  	lw	a1, 16(sp)
     234: 33 05 b5 00  	add	a0, a0, a1
     238: 23 24 a1 02  	sw	a0, 40(sp)
     23c: 83 2c 81 01  	lw	s9, 24(sp)
     240: 83 2d 41 01  	lw	s11, 20(sp)
     244: e3 9a b6 ed  	bne	a3, s11, 0x118 <MatMul_s8_s8_s32+0x118>
;   i = i * 2;
     248: 03 25 01 02  	lw	a0, 32(sp)
     24c: 13 13 15 00  	slli	t1, a0, 1
     250: 83 26 c1 00  	lw	a3, 12(sp)
     254: 03 2d 81 00  	lw	s10, 8(sp)
;   i = i * 2;
     258: 93 93 1d 00  	slli	t2, s11, 1
;   k = k * 2;
     25c: 13 1e 19 00  	slli	t3, s2, 1
;   if (i == M && j == N && k == P) {
     260: 33 c5 d3 00  	xor	a0, t2, a3
     264: 33 4f e3 00  	xor	t5, t1, a4
     268: 33 65 e5 01  	or	a0, a0, t5
     26c: b3 4e fe 00  	xor	t4, t3, a5
     270: 33 65 d5 01  	or	a0, a0, t4
     274: 63 0a 05 20  	beqz	a0, 0x488 <MatMul_s8_s8_s32+0x488>
     278: 33 35 e0 01  	snez	a0, t5
     27c: 33 3f 70 00  	snez	t5, t2
     280: b3 35 c0 01  	snez	a1, t3
;     if (jEnd != N) {
     284: 33 75 e5 01  	and	a0, a0, t5
     288: 33 75 b5 00  	and	a0, a0, a1
     28c: b3 35 e3 00  	sltu	a1, t1, a4
     290: 33 75 b5 00  	and	a0, a0, a1
     294: 63 0e 05 06  	beqz	a0, 0x310 <MatMul_s8_s8_s32+0x310>
     298: 93 0f 00 00  	li	t6, 0
;       for (i = 0; i < iEnd; i++) {
     29c: 33 04 67 40  	sub	s0, a4, t1
     2a0: 93 84 0c 00  	mv	s1, s9
     2a4: b3 04 f3 42  	<unknown>
     2a8: 33 09 6d 00  	add	s2, s10, t1
     2ac: 13 03 00 00  	li	t1, 0
     2b0: b3 89 ff 02  	mul	s3, t6, a5
     2b4: 13 8a 04 00  	mv	s4, s1
     2b8: fb 40 4e 02  	<unknown>
     2bc: 93 0a 00 00  	li	s5, 0
     2c0: 13 0b 09 00  	mv	s6, s2
     2c4: 93 0b 0a 00  	mv	s7, s4
     2c8: 13 0c 04 00  	mv	s8, s0
     2cc: 7b 40 a4 00  	<unknown>
;                   (pSrcA[i * N + j] + A_offset) * (pSrcB[j * P + k] + B_offset);
     2d0: 0b 05 1b 00  	<unknown>
     2d4: 8b f5 fb 00  	<unknown>
     2d8: 33 05 05 01  	add	a0, a0, a6
     2dc: b3 85 15 01  	add	a1, a1, a7
;             sum = sum +
     2e0: b3 8a a5 42  	<unknown>
;           pDstC[i * P + k] += sum;
     2e4: 33 05 33 01  	add	a0, t1, s3
     2e8: 13 15 25 00  	slli	a0, a0, 2
     2ec: 33 05 a6 00  	add	a0, a2, a0
     2f0: 83 25 05 00  	lw	a1, 0(a0)
     2f4: b3 85 55 01  	add	a1, a1, s5
     2f8: 23 20 b5 00  	sw	a1, 0(a0)
;         for (k = 0; k < kEnd; k++) {
     2fc: 13 03 13 00  	addi	t1, t1, 1
     300: 13 0a 1a 00  	addi	s4, s4, 1
;       for (i = 0; i < iEnd; i++) {
     304: 93 8f 1f 00  	addi	t6, t6, 1
     308: 33 09 e9 00  	add	s2, s2, a4
     30c: e3 90 7f fa  	bne	t6, t2, 0x2ac <MatMul_s8_s8_s32+0x2ac>
     310: 33 35 d0 01  	snez	a0, t4
;     if (kEnd != P) {
     314: 33 75 e5 01  	and	a0, a0, t5
     318: b3 35 fe 00  	sltu	a1, t3, a5
;     if (kEnd != P) {
     31c: 33 75 b5 00  	and	a0, a0, a1
     320: 63 06 05 0a  	beqz	a0, 0x3cc <MatMul_s8_s8_s32+0x3cc>
     324: 13 03 00 00  	li	t1, 0
;           for (j = 0; j < N; j++) {
     328: 63 0a 07 06  	beqz	a4, 0x39c <MatMul_s8_s8_s32+0x39c>
;       for (i = 0; i < iEnd; i++) {
     32c: b3 8e cc 01  	add	t4, s9, t3
     330: 13 0f 0d 00  	mv	t5, s10
     334: 33 85 c7 41  	sub	a0, a5, t3
     338: b3 0f f3 02  	mul	t6, t1, a5
     33c: 13 84 0e 00  	mv	s0, t4
     340: 93 04 0e 00  	mv	s1, t3
     344: fb 40 25 02  	<unknown>
     348: 93 09 0f 00  	mv	s3, t5
     34c: 13 0a 04 00  	mv	s4, s0
     350: 93 0a 07 00  	mv	s5, a4
     354: 13 89 02 00  	mv	s2, t0
     358: 7b 40 c7 00  	<unknown>
;                   (pSrcA[i * N + j] + A_offset) * (pSrcB[j * P + k] + B_offset);
     35c: 0b 85 19 00  	<unknown>
     360: 8b 75 fa 00  	<unknown>
     364: 33 05 05 01  	add	a0, a0, a6
     368: b3 85 15 01  	add	a1, a1, a7
;           for (j = 0; j < N; j++) {
     36c: 93 8a fa ff  	addi	s5, s5, -1
;             sum = sum +
     370: 33 89 a5 42  	<unknown>
;           pDstC[i * P + k] = sum;
     374: 33 85 f4 01  	add	a0, s1, t6
     378: 13 15 25 00  	slli	a0, a0, 2
     37c: 33 05 a6 00  	add	a0, a2, a0
     380: 23 20 25 01  	sw	s2, 0(a0)
;         for (k = kEnd; k < P; k++) {
     384: 93 84 14 00  	addi	s1, s1, 1
     388: 13 04 14 00  	addi	s0, s0, 1
;       for (i = 0; i < iEnd; i++) {
     38c: 13 03 13 00  	addi	t1, t1, 1
     390: 33 0f ef 00  	add	t5, t5, a4
     394: e3 10 73 fa  	bne	t1, t2, 0x334 <MatMul_s8_s8_s32+0x334>
     398: 6f 00 40 03  	j	0x3cc <MatMul_s8_s8_s32+0x3cc>
     39c: b3 8e c7 41  	sub	t4, a5, t3
     3a0: 13 15 2e 00  	slli	a0, t3, 2
     3a4: 33 0e a6 00  	add	t3, a2, a0
     3a8: 13 9f 27 00  	slli	t5, a5, 2
     3ac: fb c0 e3 00  	<unknown>
     3b0: 93 0f 0e 00  	mv	t6, t3
     3b4: 13 84 0e 00  	mv	s0, t4
     3b8: 7b c0 4e 00  	<unknown>
     3bc: 13 00 00 00  	nop
;           pDstC[i * P + k] = sum;
     3c0: 2b a2 5f 00  	<unknown>
;       for (i = 0; i < iEnd; i++) {
     3c4: 13 03 13 00  	addi	t1, t1, 1
     3c8: 33 0e ee 01  	add	t3, t3, t5
;     for (i = iEnd; i < M; i++) {
     3cc: 33 b5 d3 00  	sltu	a0, t2, a3
     3d0: 13 45 15 00  	xori	a0, a0, 1
     3d4: 93 b5 17 00  	seqz	a1, a5
     3d8: 33 65 b5 00  	or	a0, a0, a1
     3dc: 63 16 05 0a  	bnez	a0, 0x488 <MatMul_s8_s8_s32+0x488>
;         for (j = 0; j < N; j++) {
     3e0: 63 0a 07 06  	beqz	a4, 0x454 <MatMul_s8_s8_s32+0x454>
;     for (i = iEnd; i < M; i++) {
     3e4: 33 85 ed 02  	mul	a0, s11, a4
     3e8: 13 15 15 00  	slli	a0, a0, 1
     3ec: 33 05 ad 00  	add	a0, s10, a0
     3f0: 13 03 00 00  	li	t1, 0
     3f4: 33 8e f3 02  	mul	t3, t2, a5
     3f8: 93 8e 0c 00  	mv	t4, s9
     3fc: fb c0 27 02  	<unknown>
     400: 93 0f 05 00  	mv	t6, a0
     404: 13 84 0e 00  	mv	s0, t4
     408: 93 04 07 00  	mv	s1, a4
     40c: 13 8f 02 00  	mv	t5, t0
     410: 7b 40 c7 00  	<unknown>
;                 (pSrcA[i * N + j] + A_offset) * (pSrcB[j * P + k] + B_offset);
     414: 8b 85 1f 00  	<unknown>
     418: 0b 79 f4 00  	<unknown>
     41c: b3 85 05 01  	add	a1, a1, a6
     420: 33 09 19 01  	add	s2, s2, a7
;         for (j = 0; j < N; j++) {
     424: 93 84 f4 ff  	addi	s1, s1, -1
;           sum = sum +
     428: 33 0f b9 42  	<unknown>
;         pDstC[i * P + k] = sum;
     42c: b3 05 c3 01  	add	a1, t1, t3
     430: 93 95 25 00  	slli	a1, a1, 2
     434: b3 05 b6 00  	add	a1, a2, a1
     438: 23 a0 e5 01  	sw	t5, 0(a1)
;       for (k = 0; k < P; k++) {
     43c: 13 03 13 00  	addi	t1, t1, 1
     440: 93 8e 1e 00  	addi	t4, t4, 1
;     for (i = iEnd; i < M; i++) {
     444: 93 83 13 00  	addi	t2, t2, 1
     448: 33 05 e5 00  	add	a0, a0, a4
     44c: e3 92 d3 fa  	bne	t2, a3, 0x3f0 <MatMul_s8_s8_s32+0x3f0>
     450: 6f 00 80 03  	j	0x488 <MatMul_s8_s8_s32+0x488>
     454: 33 85 fd 02  	mul	a0, s11, a5
     458: 13 15 35 00  	slli	a0, a0, 3
     45c: 33 05 a6 00  	add	a0, a2, a0
     460: 33 86 76 40  	sub	a2, a3, t2
     464: 93 95 27 00  	slli	a1, a5, 2
     468: fb 40 e6 00  	<unknown>
     46c: 13 06 05 00  	mv	a2, a0
     470: 13 87 07 00  	mv	a4, a5
     474: 7b c0 47 00  	<unknown>
;       for (k = 0; k < P; k++) {
     478: 13 07 f7 ff  	addi	a4, a4, -1
;         pDstC[i * P + k] = sum;
     47c: 2b 22 56 00  	<unknown>
;     for (i = iEnd; i < M; i++) {
     480: 93 83 13 00  	addi	t2, t2, 1
     484: 33 05 b5 00  	add	a0, a0, a1
; }
     488: 83 20 c1 05  	lw	ra, 92(sp)
     48c: 03 24 81 05  	lw	s0, 88(sp)
     490: 83 24 41 05  	lw	s1, 84(sp)
     494: 03 29 01 05  	lw	s2, 80(sp)
     498: 83 29 c1 04  	lw	s3, 76(sp)
     49c: 03 2a 81 04  	lw	s4, 72(sp)
     4a0: 83 2a 41 04  	lw	s5, 68(sp)
     4a4: 03 2b 01 04  	lw	s6, 64(sp)
     4a8: 83 2b c1 03  	lw	s7, 60(sp)
     4ac: 03 2c 81 03  	lw	s8, 56(sp)
     4b0: 83 2c 41 03  	lw	s9, 52(sp)
     4b4: 03 2d 01 03  	lw	s10, 48(sp)
     4b8: 83 2d c1 02  	lw	s11, 44(sp)
     4bc: 13 01 01 06  	addi	sp, sp, 96
     4c0: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(MaxPool1D_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                           Size     VMA      Type
  0                                00000000 00000000 
  1 .strtab                        0000014a 00000000 
  2 .text                          00000000 00000000 TEXT
  3 .sdata                         00000004 00000000 DATA
  4 .text.MaxPool1d_fp32_fp32      00000100 00000000 TEXT
  5 .rela.text.MaxPool1d_fp32_fp32 00000018 00000000 
  6 .debug_loclists                000001cf 00000000 DEBUG
  7 .debug_abbrev                  0000009c 00000000 DEBUG
  8 .debug_info                    000000fc 00000000 DEBUG
  9 .rela.debug_info               000000a8 00000000 
 10 .debug_rnglists                0000003a 00000000 DEBUG
 11 .debug_str_offsets             00000060 00000000 DEBUG
 12 .rela.debug_str_offsets        00000108 00000000 
 13 .debug_str                     00000173 00000000 DEBUG
 14 .debug_addr                    00000014 00000000 DEBUG
 15 .rela.debug_addr               00000024 00000000 
 16 .comment                       00000073 00000000 
 17 .note.GNU-stack                00000000 00000000 
 18 .riscv.attributes              00000030 00000000 
 19 .debug_frame                   00000030 00000000 DEBUG
 20 .rela.debug_frame              00000060 00000000 
 21 .debug_line                    0000019e 00000000 DEBUG
 22 .rela.debug_line               0000033c 00000000 
 23 .debug_line_str                0000016d 00000000 DEBUG
 24 .llvm_addrsig                  00000000 00000000 
 25 .symtab                        000004a0 00000000 

Disassembly of section .text.MaxPool1d_fp32_fp32:

00000000 <MaxPool1d_fp32_fp32>:
;                          float32_t *__restrict__ pDstC) {
       0: 13 01 01 ff  	addi	sp, sp, -16
;   uint32_t W_out = (W - K) / S + 1;
       4: 23 26 81 00  	sw	s0, 12(sp)
       8: 23 24 91 00  	sw	s1, 8(sp)
       c: 23 22 21 01  	sw	s2, 4(sp)
      10: 33 08 d6 40  	sub	a6, a2, a3
      14: 33 58 e8 02  	divu	a6, a6, a4
      18: 93 08 18 00  	addi	a7, a6, 1
      1c: b3 b2 08 01  	sltu	t0, a7, a6
;   for (uint32_t c = 0; c < C; ++c) {
      20: 13 b3 15 00  	seqz	t1, a1
      24: b3 62 53 00  	or	t0, t1, t0
      28: 63 92 02 0c  	bnez	t0, 0xec <MaxPool1d_fp32_fp32+0xec>
;       for (uint32_t k = 0; k < K; ++k) {
      2c: 63 84 06 08  	beqz	a3, 0xb4 <MaxPool1d_fp32_fp32+0xb4>
      30: b7 02 00 00  	lui	t0, 0
      34: 07 a0 02 00  	flw	ft0, 0(t0)
      38: 93 02 00 00  	li	t0, 0
;   for (uint32_t c = 0; c < C; ++c) {
      3c: 13 13 26 00  	slli	t1, a2, 2
      40: 93 13 27 00  	slli	t2, a4, 2
      44: fb c0 45 03  	<unknown>
      48: 13 0e 00 00  	li	t3, 0
      4c: 13 04 00 00  	li	s0, 0
      50: b3 8e 12 03  	mul	t4, t0, a7
      54: 13 0f 05 00  	mv	t5, a0
      58: 93 0f 04 00  	mv	t6, s0
      5c: 13 04 0e 00  	mv	s0, t3
      60: 93 04 0f 00  	mv	s1, t5
      64: 13 89 06 00  	mv	s2, a3
      68: d3 00 00 20  	fmv.s	ft1, ft0
      6c: 7b c0 c6 00  	<unknown>
;         if (w_in >= W)
      70: 63 76 c4 00  	bgeu	s0, a2, 0x7c <MaxPool1d_fp32_fp32+0x7c>
;         float32_t tmp = pSrcA[c * W + w_in];
      74: 07 a1 04 00  	flw	ft2, 0(s1)
;         if (tmp > max) {
      78: d3 10 11 28  	fmax.s	ft1, ft2, ft1
;       for (uint32_t k = 0; k < K; ++k) {
      7c: 13 09 f9 ff  	addi	s2, s2, -1
      80: 93 84 44 00  	addi	s1, s1, 4
      84: 13 04 14 00  	addi	s0, s0, 1
;       pDstC[c * W_out + w_out] = max;
      88: 33 84 df 01  	add	s0, t6, t4
      8c: 13 14 24 00  	slli	s0, s0, 2
      90: 33 84 87 00  	add	s0, a5, s0
      94: 27 20 14 00  	fsw	ft1, 0(s0)
;     for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      98: 13 84 1f 00  	addi	s0, t6, 1
      9c: 33 0f 7f 00  	add	t5, t5, t2
      a0: 33 0e ee 00  	add	t3, t3, a4
      a4: e3 9a 0f fb  	bne	t6, a6, 0x58 <MaxPool1d_fp32_fp32+0x58>
;   for (uint32_t c = 0; c < C; ++c) {
      a8: 93 82 12 00  	addi	t0, t0, 1
      ac: 33 05 65 00  	add	a0, a0, t1
      b0: 6f 00 c0 03  	j	0xec <MaxPool1d_fp32_fp32+0xec>
      b4: 13 05 00 00  	li	a0, 0
      b8: 37 06 80 ff  	lui	a2, 1046528
;   for (uint32_t c = 0; c < C; ++c) {
      bc: fb c0 65 01  	<unknown>
      c0: 93 06 00 00  	li	a3, 0
      c4: 13 07 f0 ff  	li	a4, -1
      c8: 33 03 e8 40  	sub	t1, a6, a4
      cc: b3 02 15 03  	mul	t0, a0, a7
      d0: 7b 40 a3 00  	<unknown>
;       pDstC[c * W_out + w_out] = max;
      d4: 33 83 56 00  	add	t1, a3, t0
      d8: 13 13 23 00  	slli	t1, t1, 2
      dc: 33 83 67 00  	add	t1, a5, t1
      e0: 23 20 c3 00  	sw	a2, 0(t1)
;     for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      e4: 93 86 16 00  	addi	a3, a3, 1
;   for (uint32_t c = 0; c < C; ++c) {
      e8: 13 05 15 00  	addi	a0, a0, 1
; }
      ec: 03 24 c1 00  	lw	s0, 12(sp)
      f0: 83 24 81 00  	lw	s1, 8(sp)
      f4: 03 29 41 00  	lw	s2, 4(sp)
      f8: 13 01 01 01  	addi	sp, sp, 16
      fc: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(MaxPool_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                Size     VMA      Type
  0                                     00000000 00000000 
  1 .strtab                             0000014d 00000000 
  2 .text                               00000000 00000000 TEXT
  3 .sdata                              00000004 00000000 DATA
  4 .text.MaxPool2d_fp32_fp32_NCHW      0000025c 00000000 TEXT
  5 .rela.text.MaxPool2d_fp32_fp32_NCHW 00000018 00000000 
  6 .debug_loclists                     0000039e 00000000 DEBUG
  7 .debug_abbrev                       0000009c 00000000 DEBUG
  8 .debug_info                         00000143 00000000 DEBUG
  9 .rela.debug_info                    000000a8 00000000 
 10 .debug_rnglists                     00000082 00000000 DEBUG
 11 .debug_str_offsets                  0000007c 00000000 DEBUG
 12 .rela.debug_str_offsets             0000015c 00000000 
 13 .debug_str                          00000191 00000000 DEBUG
 14 .debug_addr                         00000014 00000000 DEBUG
 15 .rela.debug_addr                    00000024 00000000 
 16 .comment                            00000073 00000000 
 17 .note.GNU-stack                     00000000 00000000 
 18 .riscv.attributes                   00000030 00000000 
 19 .debug_frame                        00000044 00000000 DEBUG
 20 .rela.debug_frame                   00000060 00000000 
 21 .debug_line                         00000268 00000000 DEBUG
 22 .rela.debug_line                    00000594 00000000 
 23 .debug_line_str                     00000169 00000000 DEBUG
 24 .llvm_addrsig                       00000000 00000000 
 25 .symtab                             000006a0 00000000 

Disassembly of section .text.MaxPool2d_fp32_fp32_NCHW:

00000000 <MaxPool2d_fp32_fp32_NCHW>:
;                               float32_t *__restrict__ pDstC) {
       0: 13 01 01 fb  	addi	sp, sp, -80
       4: 23 26 11 04  	sw	ra, 76(sp)
       8: 23 24 81 04  	sw	s0, 72(sp)
       c: 23 22 91 04  	sw	s1, 68(sp)
      10: 23 20 21 05  	sw	s2, 64(sp)
      14: 23 2e 31 03  	sw	s3, 60(sp)
      18: 23 2c 41 03  	sw	s4, 56(sp)
      1c: 23 2a 51 03  	sw	s5, 52(sp)
      20: 23 28 61 03  	sw	s6, 48(sp)
      24: 23 26 71 03  	sw	s7, 44(sp)
      28: 23 24 81 03  	sw	s8, 40(sp)
      2c: 23 22 91 03  	sw	s9, 36(sp)
      30: 23 20 a1 03  	sw	s10, 32(sp)
      34: 23 2e b1 01  	sw	s11, 28(sp)
;   for (uint32_t c = 0; c < C; ++c) {
      38: 63 84 05 1e  	beqz	a1, 0x220 <MaxPool2d_fp32_fp32_NCHW+0x220>
      3c: 13 83 07 00  	mv	t1, a5
      40: b3 07 e6 40  	sub	a5, a2, a4
      44: b3 d2 07 03  	divu	t0, a5, a6
      48: 93 8f 12 00  	addi	t6, t0, 1
      4c: b3 87 66 40  	sub	a5, a3, t1
      50: b3 d3 17 03  	divu	t2, a5, a7
      54: 13 8e 13 00  	addi	t3, t2, 1
      58: 93 be 1f 00  	seqz	t4, t6
      5c: 93 37 1e 00  	seqz	a5, t3
;     for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      60: b3 e7 fe 00  	or	a5, t4, a5
      64: 63 9e 07 1a  	bnez	a5, 0x220 <MaxPool2d_fp32_fp32_NCHW+0x220>
      68: 83 2e 01 05  	lw	t4, 80(sp)
;         for (uint32_t p = 0; p < P; ++p) {
      6c: 63 08 07 10  	beqz	a4, 0x17c <MaxPool2d_fp32_fp32_NCHW+0x17c>
;           for (uint32_t q = 0; q < Q; ++q) {
      70: 63 00 03 16  	beqz	t1, 0x1d0 <MaxPool2d_fp32_fp32_NCHW+0x1d0>
      74: 13 0f 00 00  	li	t5, 0
;   for (uint32_t c = 0; c < C; ++c) {
      78: b3 87 c6 02  	mul	a5, a3, a2
      7c: 93 97 27 00  	slli	a5, a5, 2
      80: 23 26 f1 00  	sw	a5, 12(sp)
      84: 33 04 d8 02  	mul	s0, a6, a3
      88: b7 07 00 00  	lui	a5, 0
      8c: 07 a0 07 00  	flw	ft0, 0(a5)
      90: 13 14 24 00  	slli	s0, s0, 2
      94: 93 94 28 00  	slli	s1, a7, 2
      98: 13 99 26 00  	slli	s2, a3, 2
      9c: 23 2a b1 00  	sw	a1, 20(sp)
      a0: 23 28 f1 01  	sw	t6, 16(sp)
      a4: fb c0 85 06  	<unknown>
      a8: 6f 00 00 02  	j	0xc8 <MaxPool2d_fp32_fp32_NCHW+0xc8>
      ac: 13 0f 1f 00  	addi	t5, t5, 1
      b0: 03 25 81 01  	lw	a0, 24(sp)
;   for (uint32_t c = 0; c < C; ++c) {
      b4: 83 25 c1 00  	lw	a1, 12(sp)
      b8: 33 05 b5 00  	add	a0, a0, a1
      bc: 83 25 41 01  	lw	a1, 20(sp)
      c0: 83 2f 01 01  	lw	t6, 16(sp)
      c4: 63 0e bf 14  	beq	t5, a1, 0x220 <MaxPool2d_fp32_fp32_NCHW+0x220>
      c8: 93 07 00 00  	li	a5, 0
      cc: b3 09 ff 03  	mul	s3, t5, t6
      d0: 23 2c a1 00  	sw	a0, 24(sp)
      d4: 6f 00 00 01  	j	0xe4 <MaxPool2d_fp32_fp32_NCHW+0xe4>
;     for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      d8: 93 87 1a 00  	addi	a5, s5, 1
      dc: 33 05 85 00  	add	a0, a0, s0
      e0: e3 86 5a fc  	beq	s5, t0, 0xac <MaxPool2d_fp32_fp32_NCHW+0xac>
      e4: 93 0f 00 00  	li	t6, 0
      e8: 13 0b 00 00  	li	s6, 0
      ec: 93 8a 07 00  	mv	s5, a5
      f0: b3 8b 07 03  	mul	s7, a5, a6
      f4: b3 87 37 01  	add	a5, a5, s3
      f8: 33 8c c7 03  	mul	s8, a5, t3
      fc: 13 0a 05 00  	mv	s4, a0
     100: 6f 00 40 02  	j	0x124 <MaxPool2d_fp32_fp32_NCHW+0x124>
;         pDstC[c * H_out * W_out + h_out * W_out + w_out] = max;
     104: b3 05 8d 01  	add	a1, s10, s8
     108: 93 95 25 00  	slli	a1, a1, 2
     10c: b3 85 be 00  	add	a1, t4, a1
     110: 27 a0 15 00  	fsw	ft1, 0(a1)
;       for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
     114: 13 0b 1d 00  	addi	s6, s10, 1
     118: 33 0a 9a 00  	add	s4, s4, s1
     11c: b3 8f 1f 01  	add	t6, t6, a7
     120: e3 0c 7d fa  	beq	s10, t2, 0xd8 <MaxPool2d_fp32_fp32_NCHW+0xd8>
     124: 93 0d 00 00  	li	s11, 0
     128: 13 0d 0b 00  	mv	s10, s6
     12c: 93 0c 0a 00  	mv	s9, s4
     130: d3 00 00 20  	fmv.s	ft1, ft0
     134: 7b 40 e7 01  	<unknown>
;           uint32_t h_in = h_out * SP + p;
     138: b3 85 7d 01  	add	a1, s11, s7
     13c: 13 8b 0f 00  	mv	s6, t6
     140: 93 80 0c 00  	mv	ra, s9
     144: 93 07 03 00  	mv	a5, t1
;           if (h_in >= H)
     148: 63 e2 c5 02  	bltu	a1, a2, 0x16c <MaxPool2d_fp32_fp32_NCHW+0x16c>
;         for (uint32_t p = 0; p < P; ++p) {
     14c: 93 8d 1d 00  	addi	s11, s11, 1
     150: b3 8c 2c 01  	add	s9, s9, s2
     154: e3 92 ed fe  	bne	s11, a4, 0x138 <MaxPool2d_fp32_fp32_NCHW+0x138>
     158: 6f f0 df fa  	j	0x104 <MaxPool2d_fp32_fp32_NCHW+0x104>
;           for (uint32_t q = 0; q < Q; ++q) {
     15c: 93 87 f7 ff  	addi	a5, a5, -1
     160: 93 80 40 00  	addi	ra, ra, 4
     164: 13 0b 1b 00  	addi	s6, s6, 1
     168: e3 82 07 fe  	beqz	a5, 0x14c <MaxPool2d_fp32_fp32_NCHW+0x14c>
;             if (w_in >= W)
     16c: e3 78 db fe  	bgeu	s6, a3, 0x15c <MaxPool2d_fp32_fp32_NCHW+0x15c>
;             float32_t tmp = pSrcA[c * H * W + h_in * W + w_in];
     170: 07 a1 00 00  	flw	ft2, 0(ra)
;             if (tmp > max) {
     174: d3 10 11 28  	fmax.s	ft1, ft2, ft1
     178: 6f f0 5f fe  	j	0x15c <MaxPool2d_fp32_fp32_NCHW+0x15c>
     17c: 13 05 00 00  	li	a0, 0
     180: 37 06 80 ff  	lui	a2, 1046528
;   for (uint32_t c = 0; c < C; ++c) {
     184: fb c0 25 02  	<unknown>
     188: 13 07 00 00  	li	a4, 0
     18c: b3 06 f5 03  	mul	a3, a0, t6
     190: 93 07 00 00  	li	a5, 0
     194: 13 03 07 00  	mv	t1, a4
     198: 33 08 d7 00  	add	a6, a4, a3
     19c: 13 07 f0 ff  	li	a4, -1
     1a0: b3 88 e3 40  	sub	a7, t2, a4
     1a4: 33 08 c8 03  	mul	a6, a6, t3
     1a8: 7b c0 a8 00  	<unknown>
;         pDstC[c * H_out * W_out + h_out * W_out + w_out] = max;
     1ac: b3 88 07 01  	add	a7, a5, a6
     1b0: 93 98 28 00  	slli	a7, a7, 2
     1b4: b3 88 1e 01  	add	a7, t4, a7
     1b8: 23 a0 c8 00  	sw	a2, 0(a7)
;       for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
     1bc: 93 87 17 00  	addi	a5, a5, 1
;     for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
     1c0: 13 07 13 00  	addi	a4, t1, 1
     1c4: e3 16 53 fc  	bne	t1, t0, 0x190 <MaxPool2d_fp32_fp32_NCHW+0x190>
;   for (uint32_t c = 0; c < C; ++c) {
     1c8: 13 05 15 00  	addi	a0, a0, 1
     1cc: 6f 00 40 05  	j	0x220 <MaxPool2d_fp32_fp32_NCHW+0x220>
     1d0: 13 05 00 00  	li	a0, 0
     1d4: 37 06 80 ff  	lui	a2, 1046528
;   for (uint32_t c = 0; c < C; ++c) {
     1d8: fb c0 25 02  	<unknown>
     1dc: 13 07 00 00  	li	a4, 0
     1e0: b3 06 f5 03  	mul	a3, a0, t6
     1e4: 93 07 00 00  	li	a5, 0
     1e8: 13 03 07 00  	mv	t1, a4
     1ec: 33 08 d7 00  	add	a6, a4, a3
     1f0: 13 07 f0 ff  	li	a4, -1
     1f4: b3 88 e3 40  	sub	a7, t2, a4
     1f8: 33 08 c8 03  	mul	a6, a6, t3
     1fc: 7b c0 a8 00  	<unknown>
;         pDstC[c * H_out * W_out + h_out * W_out + w_out] = max;
     200: b3 88 07 01  	add	a7, a5, a6
     204: 93 98 28 00  	slli	a7, a7, 2
     208: b3 88 1e 01  	add	a7, t4, a7
     20c: 23 a0 c8 00  	sw	a2, 0(a7)
;       for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
     210: 93 87 17 00  	addi	a5, a5, 1
;     for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
     214: 13 07 13 00  	addi	a4, t1, 1
     218: e3 16 53 fc  	bne	t1, t0, 0x1e4 <MaxPool2d_fp32_fp32_NCHW+0x1e4>
;   for (uint32_t c = 0; c < C; ++c) {
     21c: 13 05 15 00  	addi	a0, a0, 1
; }
     220: 83 20 c1 04  	lw	ra, 76(sp)
     224: 03 24 81 04  	lw	s0, 72(sp)
     228: 83 24 41 04  	lw	s1, 68(sp)
     22c: 03 29 01 04  	lw	s2, 64(sp)
     230: 83 29 c1 03  	lw	s3, 60(sp)
     234: 03 2a 81 03  	lw	s4, 56(sp)
     238: 83 2a 41 03  	lw	s5, 52(sp)
     23c: 03 2b 01 03  	lw	s6, 48(sp)
     240: 83 2b c1 02  	lw	s7, 44(sp)
     244: 03 2c 81 02  	lw	s8, 40(sp)
     248: 83 2c 41 02  	lw	s9, 36(sp)
     24c: 03 2d 01 02  	lw	s10, 32(sp)
     250: 83 2d c1 01  	lw	s11, 28(sp)
     254: 13 01 01 05  	addi	sp, sp, 80
     258: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(MaxPool_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                       Size     VMA      Type
  0                            00000000 00000000 
  1 .strtab                    00000122 00000000 
  2 .text                      00000000 00000000 TEXT
  3 .text.MaxPool2d_s8_s8_NCHW 000001f8 00000000 TEXT
  4 .debug_loclists            00000606 00000000 DEBUG
  5 .debug_abbrev              00000093 00000000 DEBUG
  6 .debug_info                00000144 00000000 DEBUG
  7 .rela.debug_info           0000006c 00000000 
  8 .debug_str_offsets         0000008c 00000000 DEBUG
  9 .rela.debug_str_offsets    0000018c 00000000 
 10 .debug_str                 000001b6 00000000 DEBUG
 11 .debug_addr                0000000c 00000000 DEBUG
 12 .rela.debug_addr           0000000c 00000000 
 13 .comment                   00000073 00000000 
 14 .note.GNU-stack            00000000 00000000 
 15 .riscv.attributes          00000030 00000000 
 16 .debug_frame               00000044 00000000 DEBUG
 17 .rela.debug_frame          00000060 00000000 
 18 .debug_line                0000023d 00000000 DEBUG
 19 .rela.debug_line           00000540 00000000 
 20 .debug_line_str            00000141 00000000 DEBUG
 21 .llvm_addrsig              00000000 00000000 
 22 .symtab                    00000650 00000000 

Disassembly of section .text.MaxPool2d_s8_s8_NCHW:

00000000 <MaxPool2d_s8_s8_NCHW>:
;                           int32_t input_offset, int32_t output_offset) {
       0: 13 01 01 fc  	addi	sp, sp, -64
       4: 23 2e 11 02  	sw	ra, 60(sp)
       8: 23 2c 81 02  	sw	s0, 56(sp)
       c: 23 2a 91 02  	sw	s1, 52(sp)
      10: 23 28 21 03  	sw	s2, 48(sp)
      14: 23 26 31 03  	sw	s3, 44(sp)
      18: 23 24 41 03  	sw	s4, 40(sp)
      1c: 23 22 51 03  	sw	s5, 36(sp)
      20: 23 20 61 03  	sw	s6, 32(sp)
      24: 23 2e 71 01  	sw	s7, 28(sp)
      28: 23 2c 81 01  	sw	s8, 24(sp)
      2c: 23 2a 91 01  	sw	s9, 20(sp)
      30: 23 28 a1 01  	sw	s10, 16(sp)
      34: 23 26 b1 01  	sw	s11, 12(sp)
;   for (c = 0; c < C; ++c) {
      38: 63 82 05 18  	beqz	a1, 0x1bc <MaxPool2d_s8_s8_NCHW+0x1bc>
      3c: b3 02 e6 40  	sub	t0, a2, a4
      40: b3 d2 02 03  	divu	t0, t0, a6
      44: 13 83 12 00  	addi	t1, t0, 1
      48: b3 83 f6 40  	sub	t2, a3, a5
      4c: b3 d3 13 03  	divu	t2, t2, a7
      50: 13 8e 13 00  	addi	t3, t2, 1
      54: 93 3e 13 00  	seqz	t4, t1
      58: 13 3f 1e 00  	seqz	t5, t3
;     for (h = 0; h < H_out; ++h) {
      5c: b3 ee ee 01  	or	t4, t4, t5
      60: 63 9e 0e 14  	bnez	t4, 0x1bc <MaxPool2d_s8_s8_NCHW+0x1bc>
      64: 03 2f 81 04  	lw	t5, 72(sp)
      68: 83 2e 01 04  	lw	t4, 64(sp)
      6c: 93 4f 0f f8  	xori	t6, t5, -128
;         for (p = 0; p < P; ++p) {
      70: 63 0c 07 0a  	beqz	a4, 0x128 <MaxPool2d_s8_s8_NCHW+0x128>
;           for (q = 0; q < Q; ++q) {
      74: 63 80 07 10  	beqz	a5, 0x174 <MaxPool2d_s8_s8_NCHW+0x174>
      78: 83 2f 41 04  	lw	t6, 68(sp)
      7c: 13 04 00 00  	li	s0, 0
;   for (c = 0; c < C; ++c) {
      80: 33 86 c6 02  	mul	a2, a3, a2
      84: 33 08 d8 02  	mul	a6, a6, a3
      88: 13 0a 00 00  	li	s4, 0
      8c: b3 04 64 02  	mul	s1, s0, t1
      90: 13 09 05 00  	mv	s2, a0
      94: 13 0c 00 00  	li	s8, 0
      98: 93 09 0a 00  	mv	s3, s4
      9c: 33 0a 9a 00  	add	s4, s4, s1
      a0: 33 0a ca 03  	mul	s4, s4, t3
      a4: 93 0a 09 00  	mv	s5, s2
      a8: 93 0b 00 00  	li	s7, 0
      ac: 13 0b 0c 00  	mv	s6, s8
      b0: 13 0c 00 f8  	li	s8, -128
      b4: 93 8c 0a 00  	mv	s9, s5
      b8: fb 40 a7 01  	<unknown>
      bc: 13 8d 0c 00  	mv	s10, s9
      c0: 93 8d 07 00  	mv	s11, a5
      c4: 7b c0 07 01  	<unknown>
;             tmp = (int32_t)(pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
      c8: 83 00 0d 00  	lb	ra, 0(s10)
      cc: b3 80 f0 01  	add	ra, ra, t6
      d0: 23 24 11 00  	sw	ra, 8(sp)
;             if (tmp > max) {
      d4: 83 20 81 00  	lw	ra, 8(sp)
      d8: 63 54 1c 00  	bge	s8, ra, 0xe0 <MaxPool2d_s8_s8_NCHW+0xe0>
;               max = tmp;
      dc: 03 2c 81 00  	lw	s8, 8(sp)
;           for (q = 0; q < Q; ++q) {
      e0: 93 8d fd ff  	addi	s11, s11, -1
      e4: 13 0d 1d 00  	addi	s10, s10, 1
;         for (p = 0; p < P; ++p) {
      e8: 93 8b 1b 00  	addi	s7, s7, 1
      ec: b3 8c dc 00  	add	s9, s9, a3
;             (int8_t)(max + output_offset);
      f0: b3 0b ec 01  	add	s7, s8, t5
;         pDstC[c * H_out * W_out + h * W_out + w] =
      f4: 33 0c 4b 01  	add	s8, s6, s4
      f8: 33 8c 8e 01  	add	s8, t4, s8
      fc: 23 00 7c 01  	sb	s7, 0(s8)
;       for (w = 0; w < W_out; ++w) {
     100: 13 0c 1b 00  	addi	s8, s6, 1
     104: b3 8a 1a 01  	add	s5, s5, a7
     108: e3 10 7b fa  	bne	s6, t2, 0xa8 <MaxPool2d_s8_s8_NCHW+0xa8>
;     for (h = 0; h < H_out; ++h) {
     10c: 13 8a 19 00  	addi	s4, s3, 1
     110: 33 09 09 01  	add	s2, s2, a6
     114: e3 90 59 f8  	bne	s3, t0, 0x94 <MaxPool2d_s8_s8_NCHW+0x94>
;   for (c = 0; c < C; ++c) {
     118: 13 04 14 00  	addi	s0, s0, 1
     11c: 33 05 c5 00  	add	a0, a0, a2
     120: e3 14 b4 f6  	bne	s0, a1, 0x88 <MaxPool2d_s8_s8_NCHW+0x88>
     124: 6f 00 80 09  	j	0x1bc <MaxPool2d_s8_s8_NCHW+0x1bc>
     128: 13 05 00 00  	li	a0, 0
;   for (c = 0; c < C; ++c) {
     12c: fb c0 05 02  	<unknown>
     130: 13 07 00 00  	li	a4, 0
     134: 33 06 65 02  	mul	a2, a0, t1
     138: 93 07 00 00  	li	a5, 0
     13c: 93 06 07 00  	mv	a3, a4
     140: 33 08 c7 00  	add	a6, a4, a2
     144: 13 07 f0 ff  	li	a4, -1
     148: b3 88 e3 40  	sub	a7, t2, a4
     14c: 33 08 c8 03  	mul	a6, a6, t3
     150: 7b c0 88 00  	<unknown>
;         pDstC[c * H_out * W_out + h * W_out + w] =
     154: b3 88 07 01  	add	a7, a5, a6
     158: b3 88 1e 01  	add	a7, t4, a7
     15c: 23 80 f8 01  	sb	t6, 0(a7)
;       for (w = 0; w < W_out; ++w) {
     160: 93 87 17 00  	addi	a5, a5, 1
;     for (h = 0; h < H_out; ++h) {
     164: 13 87 16 00  	addi	a4, a3, 1
     168: e3 98 56 fc  	bne	a3, t0, 0x138 <MaxPool2d_s8_s8_NCHW+0x138>
;   for (c = 0; c < C; ++c) {
     16c: 13 05 15 00  	addi	a0, a0, 1
     170: 6f 00 c0 04  	j	0x1bc <MaxPool2d_s8_s8_NCHW+0x1bc>
     174: 13 05 00 00  	li	a0, 0
;   for (c = 0; c < C; ++c) {
     178: fb c0 05 02  	<unknown>
     17c: 13 07 00 00  	li	a4, 0
     180: 33 06 65 02  	mul	a2, a0, t1
     184: 93 07 00 00  	li	a5, 0
     188: 93 06 07 00  	mv	a3, a4
     18c: 33 08 c7 00  	add	a6, a4, a2
     190: 13 07 f0 ff  	li	a4, -1
     194: b3 88 e3 40  	sub	a7, t2, a4
     198: 33 08 c8 03  	mul	a6, a6, t3
     19c: 7b c0 88 00  	<unknown>
;         pDstC[c * H_out * W_out + h * W_out + w] =
     1a0: b3 88 07 01  	add	a7, a5, a6
     1a4: b3 88 1e 01  	add	a7, t4, a7
     1a8: 23 80 f8 01  	sb	t6, 0(a7)
;       for (w = 0; w < W_out; ++w) {
     1ac: 93 87 17 00  	addi	a5, a5, 1
;     for (h = 0; h < H_out; ++h) {
     1b0: 13 87 16 00  	addi	a4, a3, 1
     1b4: e3 98 56 fc  	bne	a3, t0, 0x184 <MaxPool2d_s8_s8_NCHW+0x184>
;   for (c = 0; c < C; ++c) {
     1b8: 13 05 15 00  	addi	a0, a0, 1
; }
     1bc: 83 20 c1 03  	lw	ra, 60(sp)
     1c0: 03 24 81 03  	lw	s0, 56(sp)
     1c4: 83 24 41 03  	lw	s1, 52(sp)
     1c8: 03 29 01 03  	lw	s2, 48(sp)
     1cc: 83 29 c1 02  	lw	s3, 44(sp)
     1d0: 03 2a 81 02  	lw	s4, 40(sp)
     1d4: 83 2a 41 02  	lw	s5, 36(sp)
     1d8: 03 2b 01 02  	lw	s6, 32(sp)
     1dc: 83 2b c1 01  	lw	s7, 28(sp)
     1e0: 03 2c 81 01  	lw	s8, 24(sp)
     1e4: 83 2c 41 01  	lw	s9, 20(sp)
     1e8: 03 2d 01 01  	lw	s10, 16(sp)
     1ec: 83 2d c1 00  	lw	s11, 12(sp)
     1f0: 13 01 01 04  	addi	sp, sp, 64
     1f4: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Pow_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                            Size     VMA      Type
  0                                 00000000 00000000 
  1 .strtab                         00000158 00000000 
  2 .text                           00000000 00000000 TEXT
  3 .text.Pow_fp32_fp32_fp32        00000070 00000000 TEXT
  4 .rela.text.Pow_fp32_fp32_fp32   0000000c 00000000 
  5 .text.Pow_fp32_scalar_fp32      0000006c 00000000 TEXT
  6 .rela.text.Pow_fp32_scalar_fp32 0000000c 00000000 
  7 .debug_loclists                 000000a4 00000000 DEBUG
  8 .debug_abbrev                   00000088 00000000 DEBUG
  9 .debug_info                     000000e5 00000000 DEBUG
 10 .rela.debug_info                000000a8 00000000 
 11 .debug_rnglists                 00000017 00000000 DEBUG
 12 .debug_str_offsets              00000044 00000000 DEBUG
 13 .rela.debug_str_offsets         000000b4 00000000 
 14 .debug_str                      00000166 00000000 DEBUG
 15 .debug_addr                     00000018 00000000 DEBUG
 16 .rela.debug_addr                00000030 00000000 
 17 .comment                        00000073 00000000 
 18 .note.GNU-stack                 00000000 00000000 
 19 .riscv.attributes               00000030 00000000 
 20 .debug_frame                    00000054 00000000 DEBUG
 21 .rela.debug_frame               000000c0 00000000 
 22 .debug_line                     00000161 00000000 DEBUG
 23 .rela.debug_line                00000270 00000000 
 24 .debug_line_str                 00000161 00000000 DEBUG
 25 .llvm_addrsig                   00000000 00000000 
 26 .symtab                         00000410 00000000 

Disassembly of section .text.Pow_fp32_fp32_fp32:

00000000 <Pow_fp32_fp32_fp32>:
;                         float32_t *__restrict__ data_out, int32_t size) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 23 26 31 01  	sw	s3, 12(sp)
;   for (int i = 0; i < size; i++) {
      18: 63 5e d0 02  	blez	a3, 0x54 <Pow_fp32_fp32_fp32+0x54>
      1c: 13 84 06 00  	mv	s0, a3
      20: 93 04 06 00  	mv	s1, a2
      24: 13 89 05 00  	mv	s2, a1
      28: 93 09 05 00  	mv	s3, a0
;     data_out[i] = powf(data_in[i], exponent[i]);
      2c: 07 a5 09 00  	flw	fa0, 0(s3)
      30: 87 25 09 00  	flw	fa1, 0(s2)
      34: 97 00 00 00  	auipc	ra, 0
      38: e7 80 00 00  	jalr	ra
      3c: 27 a0 a4 00  	fsw	fa0, 0(s1)
;   for (int i = 0; i < size; i++) {
      40: 13 04 f4 ff  	addi	s0, s0, -1
      44: 93 84 44 00  	addi	s1, s1, 4
      48: 13 09 49 00  	addi	s2, s2, 4
      4c: 93 89 49 00  	addi	s3, s3, 4
      50: e3 1e 04 fc  	bnez	s0, 0x2c <Pow_fp32_fp32_fp32+0x2c>
; }
      54: 83 20 c1 01  	lw	ra, 28(sp)
      58: 03 24 81 01  	lw	s0, 24(sp)
      5c: 83 24 41 01  	lw	s1, 20(sp)
      60: 03 29 01 01  	lw	s2, 16(sp)
      64: 83 29 c1 00  	lw	s3, 12(sp)
      68: 13 01 01 02  	addi	sp, sp, 32
      6c: 67 80 00 00  	ret

Disassembly of section .text.Pow_fp32_scalar_fp32:

00000000 <Pow_fp32_scalar_fp32>:
;                           int32_t size) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 27 26 81 00  	fsw	fs0, 12(sp)
;   for (int i = 0; i < size; i++) {
      18: 63 5c c0 02  	blez	a2, 0x50 <Pow_fp32_scalar_fp32+0x50>
      1c: 13 04 06 00  	mv	s0, a2
      20: 93 84 05 00  	mv	s1, a1
      24: 53 04 a5 20  	fmv.s	fs0, fa0
      28: 13 09 05 00  	mv	s2, a0
;     data_out[i] = powf(data_in[i], exponent);
      2c: 07 25 09 00  	flw	fa0, 0(s2)
      30: d3 05 84 20  	fmv.s	fa1, fs0
      34: 97 00 00 00  	auipc	ra, 0
      38: e7 80 00 00  	jalr	ra
      3c: 27 a0 a4 00  	fsw	fa0, 0(s1)
;   for (int i = 0; i < size; i++) {
      40: 13 04 f4 ff  	addi	s0, s0, -1
      44: 93 84 44 00  	addi	s1, s1, 4
      48: 13 09 49 00  	addi	s2, s2, 4
      4c: e3 10 04 fe  	bnez	s0, 0x2c <Pow_fp32_scalar_fp32+0x2c>
; }
      50: 83 20 c1 01  	lw	ra, 28(sp)
      54: 03 24 81 01  	lw	s0, 24(sp)
      58: 83 24 41 01  	lw	s1, 20(sp)
      5c: 03 29 01 01  	lw	s2, 16(sp)
      60: 07 24 c1 00  	flw	fs0, 12(sp)
      64: 13 01 01 02  	addi	sp, sp, 32
      68: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(RQDiv_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 00000136 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text.RQDiv_s32_s8      00000224 00000000 TEXT
  4 .rela.text.RQDiv_s32_s8 0000000c 00000000 
  5 .debug_loclists         00000170 00000000 DEBUG
  6 .debug_abbrev           00000090 00000000 DEBUG
  7 .debug_info             00000176 00000000 DEBUG
  8 .rela.debug_info        000000a8 00000000 
  9 .debug_rnglists         00000019 00000000 DEBUG
 10 .debug_str_offsets      0000009c 00000000 DEBUG
 11 .rela.debug_str_offsets 000001bc 00000000 
 12 .debug_str              00000218 00000000 DEBUG
 13 .debug_addr             00000014 00000000 DEBUG
 14 .rela.debug_addr        00000024 00000000 
 15 .comment                00000073 00000000 
 16 .note.GNU-stack         00000000 00000000 
 17 .riscv.attributes       00000030 00000000 
 18 .debug_frame            00000044 00000000 DEBUG
 19 .rela.debug_frame       00000060 00000000 
 20 .debug_line             000001eb 00000000 DEBUG
 21 .rela.debug_line        00000468 00000000 
 22 .debug_line_str         0000013d 00000000 DEBUG
 23 .llvm_addrsig           00000000 00000000 
 24 .symtab                 00000660 00000000 

Disassembly of section .text.RQDiv_s32_s8:

00000000 <RQDiv_s32_s8>:
;                   int32_t requant_add, int32_t requant_shift) {
       0: 13 01 01 f9  	addi	sp, sp, -112
       4: 23 26 11 06  	sw	ra, 108(sp)
       8: 23 24 81 06  	sw	s0, 104(sp)
       c: 23 22 91 06  	sw	s1, 100(sp)
      10: 23 20 21 07  	sw	s2, 96(sp)
      14: 23 2e 31 05  	sw	s3, 92(sp)
      18: 23 2c 41 05  	sw	s4, 88(sp)
      1c: 23 2a 51 05  	sw	s5, 84(sp)
      20: 23 28 61 05  	sw	s6, 80(sp)
      24: 23 26 71 05  	sw	s7, 76(sp)
      28: 23 24 81 05  	sw	s8, 72(sp)
      2c: 23 22 91 05  	sw	s9, 68(sp)
      30: 23 20 a1 05  	sw	s10, 64(sp)
      34: 23 2e b1 03  	sw	s11, 60(sp)
      38: 23 28 f1 02  	sw	a5, 48(sp)
;   int32_t secondIter = nomStep / innerMostIter;
      3c: b3 46 f7 02  	div	a3, a4, a5
      40: 23 24 d1 02  	sw	a3, 40(sp)
;   int32_t thirdIter = size_nom / secondIter;
      44: 33 46 d6 02  	div	a2, a2, a3
      48: 23 22 b1 02  	sw	a1, 36(sp)
      4c: 23 26 c1 00  	sw	a2, 12(sp)
;   for (int i = 0; i < thirdIter; i++) {
      50: 63 5c c0 18  	blez	a2, 0x1e8 <RQDiv_s32_s8+0x1e8>
      54: 13 09 05 00  	mv	s2, a0
      58: 03 25 01 03  	lw	a0, 48(sp)
      5c: 13 25 15 00  	slti	a0, a0, 1
      60: 83 25 81 02  	lw	a1, 40(sp)
      64: 93 a5 15 00  	slti	a1, a1, 1
;     for (int k = 0; k < innerMostIter; k++) {
      68: 33 65 b5 00  	or	a0, a0, a1
      6c: 63 1e 05 16  	bnez	a0, 0x1e8 <RQDiv_s32_s8+0x1e8>
      70: 93 04 08 00  	mv	s1, a6
      74: 13 07 00 00  	li	a4, 0
      78: 83 2c 01 08  	lw	s9, 128(sp)
      7c: 83 25 41 07  	lw	a1, 116(sp)
      80: 03 25 c1 07  	lw	a0, 124(sp)
      84: 83 2d 81 07  	lw	s11, 120(sp)
      88: 83 27 01 07  	lw	a5, 112(sp)
      8c: 23 20 b1 02  	sw	a1, 32(sp)
      90: b3 89 15 03  	mul	s3, a1, a7
      94: 93 85 fc ff  	addi	a1, s9, -1
      98: 13 06 10 00  	li	a2, 1
      9c: b3 15 b6 00  	sll	a1, a2, a1
      a0: 33 8c a5 00  	add	s8, a1, a0
      a4: 83 25 01 03  	lw	a1, 48(sp)
;   for (int i = 0; i < thirdIter; i++) {
      a8: 03 25 81 02  	lw	a0, 40(sp)
      ac: 33 05 b5 02  	mul	a0, a0, a1
      b0: 13 15 25 00  	slli	a0, a0, 2
      b4: 23 24 a1 00  	sw	a0, 8(sp)
      b8: 13 94 25 00  	slli	s0, a1, 2
      bc: 23 2e f1 00  	sw	a5, 28(sp)
      c0: 13 d5 f7 41  	srai	a0, a5, 31
      c4: 23 2c a1 00  	sw	a0, 24(sp)
      c8: 93 05 00 00  	li	a1, 0
      cc: 03 25 01 03  	lw	a0, 48(sp)
      d0: 23 28 e1 00  	sw	a4, 16(sp)
      d4: 33 05 a7 02  	mul	a0, a4, a0
      d8: 23 26 a1 02  	sw	a0, 44(sp)
      dc: 23 2a 21 01  	sw	s2, 20(sp)
      e0: 23 2c b1 02  	sw	a1, 56(sp)
;       denom = data_in_denom[i * innerMostIter + k];
      e4: 03 25 c1 02  	lw	a0, 44(sp)
      e8: 33 85 a5 00  	add	a0, a1, a0
      ec: 13 15 25 00  	slli	a0, a0, 2
      f0: 83 25 41 02  	lw	a1, 36(sp)
      f4: 33 85 a5 00  	add	a0, a1, a0
      f8: 03 25 05 00  	lw	a0, 0(a0)
      fc: 83 26 01 02  	lw	a3, 32(sp)
;       denom = ((eta * denom) + eps);
     100: b3 15 d5 02  	mulh	a1, a0, a3
     104: 33 06 d5 02  	mul	a2, a0, a3
     108: 03 27 c1 01  	lw	a4, 28(sp)
     10c: 13 0a 07 00  	mv	s4, a4
     110: 33 0a d5 42  	<unknown>
     114: 33 35 ca 00  	sltu	a0, s4, a2
     118: 83 26 81 01  	lw	a3, 24(sp)
     11c: b3 85 d5 00  	add	a1, a1, a3
     120: b3 8a a5 00  	add	s5, a1, a0
     124: 5b ad a5 02  	<unknown>
     128: 13 95 fa 01  	slli	a0, s5, 31
     12c: db 25 e6 82  	<unknown>
     130: b3 eb a5 00  	or	s7, a1, a0
     134: 23 2a 21 03  	sw	s2, 52(sp)
     138: 03 2b 81 02  	lw	s6, 40(sp)
;             data_in_nom[i * secondIter * innerMostIter + j * innerMostIter + k];
     13c: 8b 75 89 20  	<unknown>
;         nom = (Delta * eta * nom);
     140: 33 96 35 03  	mulh	a2, a1, s3
;         sgnNom = (nom >= 0) - (nom < 0);
     144: 13 45 f6 ff  	not	a0, a2
     148: 13 55 f5 01  	srli	a0, a0, 31
     14c: 93 56 f6 41  	srai	a3, a2, 31
     150: 33 07 d5 00  	add	a4, a0, a3
;         y = (int32_t)((nom + sgnNom * (denom >> 1)) / denom);
     154: 5b 25 d5 3e  	<unknown>
     158: b3 b6 eb 02  	mulhu	a3, s7, a4
     15c: b3 86 ab 42  	<unknown>
     160: b3 87 eb 02  	mul	a5, s7, a4
     164: b3 06 ed 42  	<unknown>
     168: 13 85 07 00  	mv	a0, a5
     16c: 33 85 35 43  	<unknown>
     170: b3 35 f5 00  	sltu	a1, a0, a5
     174: 33 86 c6 00  	add	a2, a3, a2
     178: b3 05 b6 00  	add	a1, a2, a1
     17c: 13 06 0a 00  	mv	a2, s4
     180: 93 86 0a 00  	mv	a3, s5
     184: 97 00 00 00  	auipc	ra, 0
     188: e7 80 00 00  	jalr	ra
     18c: 93 06 f0 07  	li	a3, 127
     190: 13 06 00 f8  	li	a2, -128
;         intermediate = (int32_t)(y)*requant_mul + requant_add;
     194: 33 05 b5 03  	mul	a0, a0, s11
;             ((intermediate + ((1 << (requant_shift - 1)))) >> requant_shift);
     198: 93 05 0c 00  	mv	a1, s8
     19c: db 25 95 41  	<unknown>
;         *data_out++ = (int8_t)CLAMP(intermediate, -128, 127);
     1a0: 33 e5 c5 04  	<unknown>
     1a4: 33 45 d5 04  	<unknown>
;       for (int j = 0; j < secondIter; j++) {
     1a8: 13 0b fb ff  	addi	s6, s6, -1
;         *data_out++ = (int8_t)CLAMP(intermediate, -128, 127);
     1ac: ab 80 a4 00  	<unknown>
;       for (int j = 0; j < secondIter; j++) {
     1b0: e3 16 0b f8  	bnez	s6, 0x13c <RQDiv_s32_s8+0x13c>
     1b4: 83 25 81 03  	lw	a1, 56(sp)
;     for (int k = 0; k < innerMostIter; k++) {
     1b8: 93 85 15 00  	addi	a1, a1, 1
     1bc: 03 29 41 03  	lw	s2, 52(sp)
;     for (int k = 0; k < innerMostIter; k++) {
     1c0: 13 09 49 00  	addi	s2, s2, 4
     1c4: 03 25 01 03  	lw	a0, 48(sp)
     1c8: e3 9c a5 f0  	bne	a1, a0, 0xe0 <RQDiv_s32_s8+0xe0>
     1cc: 03 27 01 01  	lw	a4, 16(sp)
;   for (int i = 0; i < thirdIter; i++) {
     1d0: 13 07 17 00  	addi	a4, a4, 1
     1d4: 03 29 41 01  	lw	s2, 20(sp)
;   for (int i = 0; i < thirdIter; i++) {
     1d8: 03 25 81 00  	lw	a0, 8(sp)
     1dc: 33 09 a9 00  	add	s2, s2, a0
     1e0: 03 25 c1 00  	lw	a0, 12(sp)
     1e4: e3 12 a7 ee  	bne	a4, a0, 0xc8 <RQDiv_s32_s8+0xc8>
; }
     1e8: 83 20 c1 06  	lw	ra, 108(sp)
     1ec: 03 24 81 06  	lw	s0, 104(sp)
     1f0: 83 24 41 06  	lw	s1, 100(sp)
     1f4: 03 29 01 06  	lw	s2, 96(sp)
     1f8: 83 29 c1 05  	lw	s3, 92(sp)
     1fc: 03 2a 81 05  	lw	s4, 88(sp)
     200: 83 2a 41 05  	lw	s5, 84(sp)
     204: 03 2b 01 05  	lw	s6, 80(sp)
     208: 83 2b c1 04  	lw	s7, 76(sp)
     20c: 03 2c 81 04  	lw	s8, 72(sp)
     210: 83 2c 41 04  	lw	s9, 68(sp)
     214: 03 2d 01 04  	lw	s10, 64(sp)
     218: 83 2d c1 03  	lw	s11, 60(sp)
     21c: 13 01 01 07  	addi	sp, sp, 112
     220: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(RQGELU_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 00000119 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text.RQGELU_s8_s8      000000a8 00000000 TEXT
  4 .debug_loclists         00000125 00000000 DEBUG
  5 .debug_abbrev           00000085 00000000 DEBUG
  6 .debug_info             0000012b 00000000 DEBUG
  7 .rela.debug_info        00000084 00000000 
  8 .debug_str_offsets      00000088 00000000 DEBUG
  9 .rela.debug_str_offsets 00000180 00000000 
 10 .debug_str              000001bc 00000000 DEBUG
 11 .debug_addr             00000010 00000000 DEBUG
 12 .rela.debug_addr        00000018 00000000 
 13 .comment                00000073 00000000 
 14 .note.GNU-stack         00000000 00000000 
 15 .riscv.attributes       00000030 00000000 
 16 .debug_frame            00000030 00000000 DEBUG
 17 .rela.debug_frame       00000060 00000000 
 18 .debug_line             00000187 00000000 DEBUG
 19 .rela.debug_line        00000318 00000000 
 20 .debug_line_str         0000013f 00000000 DEBUG
 21 .llvm_addrsig           00000000 00000000 
 22 .symtab                 000004f0 00000000 

Disassembly of section .text.RQGELU_s8_s8:

00000000 <RQGELU_s8_s8>:
;                   int32_t *mul, int32_t *add, int32_t *shift) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
       8: 23 24 91 00  	sw	s1, 8(sp)
       c: 23 22 21 01  	sw	s2, 4(sp)
;   for (int i = 0; i < dataSize; i++) {
      10: 63 52 c0 08  	blez	a2, 0x94 <RQGELU_s8_s8+0x94>
      14: 83 22 41 01  	lw	t0, 20(sp)
      18: 03 23 01 01  	lw	t1, 16(sp)
      1c: b3 03 d0 40  	neg	t2, a3
      20: 13 0e 10 00  	li	t3, 1
      24: 93 0e 00 f8  	li	t4, -128
      28: 13 0f f0 07  	li	t5, 127
;   for (int i = 0; i < dataSize; i++) {
      2c: 7b 40 26 03  	<unknown>
;     x = data_in[i] + input_offset;
      30: 8b 0f 15 00  	<unknown>
      34: 33 84 ff 00  	add	s0, t6, a5
;     sign = (x > 0) - (x < 0); // sgn(x)
      38: b3 24 80 00  	sgtz	s1, s0
      3c: db af ff 3e  	<unknown>
      40: b3 8f 9f 00  	add	t6, t6, s1
;     x_abs = sign * x;         // abs(x)
      44: b3 84 8f 02  	mul	s1, t6, s0
      48: b3 c4 74 04  	<unknown>
;     d = q + b;
      4c: b3 84 d4 00  	add	s1, s1, a3
;     L = sign * (-(d * d) + one);
      50: 13 09 07 00  	mv	s2, a4
      54: 33 99 94 42  	<unknown>
;     intermediate = ((int32_t)y) * (*mul) + (*add);
      58: 83 a4 08 00  	lw	s1, 0(a7)
;     L = sign * (-(d * d) + one);
      5c: b3 0f f9 03  	mul	t6, s2, t6
;     y = x * (((one + L)) >> 1);
      60: db af ef 02  	<unknown>
;         ((intermediate + ((1 << ((*shift) - 1)))) >> (*shift)) + output_offset;
      64: 03 a9 02 00  	lw	s2, 0(t0)
;     y = x * (((one + L)) >> 1);
      68: 33 04 94 02  	mul	s0, s0, s1
;     intermediate = ((int32_t)y) * (*mul) + (*add);
      6c: 83 24 03 00  	lw	s1, 0(t1)
      70: b3 0f f4 03  	mul	t6, s0, t6
;         ((intermediate + ((1 << ((*shift) - 1)))) >> (*shift)) + output_offset;
      74: 13 04 f9 ff  	addi	s0, s2, -1
      78: 33 14 8e 00  	sll	s0, t3, s0
;     intermediate = ((int32_t)y) * (*mul) + (*add);
      7c: 33 04 94 00  	add	s0, s0, s1
;         ((intermediate + ((1 << ((*shift) - 1)))) >> (*shift)) + output_offset;
      80: 5b a4 2f 41  	<unknown>
      84: b3 0f 04 01  	add	t6, s0, a6
;     data_out[i] = (int8_t)CLAMP(intermediate, -128, 127);
      88: b3 ef df 05  	<unknown>
      8c: b3 cf ef 05  	<unknown>
      90: ab 80 f5 01  	<unknown>
; }
      94: 03 24 c1 00  	lw	s0, 12(sp)
      98: 83 24 81 00  	lw	s1, 8(sp)
      9c: 03 29 41 00  	lw	s2, 4(sp)
      a0: 13 01 01 01  	addi	sp, sp, 16
      a4: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(RQHardswish.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                     Size     VMA      Type
  0                          00000000 00000000 
  1 .strtab                  00000121 00000000 
  2 .text                    00000000 00000000 TEXT
  3 .text.RQiHardswish_s8_s8 00000068 00000000 TEXT
  4 .debug_loclists          000000d0 00000000 DEBUG
  5 .debug_abbrev            00000078 00000000 DEBUG
  6 .debug_info              000000dd 00000000 DEBUG
  7 .rela.debug_info         00000084 00000000 
  8 .debug_str_offsets       00000064 00000000 DEBUG
  9 .rela.debug_str_offsets  00000114 00000000 
 10 .debug_str               00000198 00000000 DEBUG
 11 .debug_addr              0000000c 00000000 DEBUG
 12 .rela.debug_addr         0000000c 00000000 
 13 .comment                 00000073 00000000 
 14 .note.GNU-stack          00000000 00000000 
 15 .riscv.attributes        00000030 00000000 
 16 .debug_frame             00000024 00000000 DEBUG
 17 .rela.debug_frame        00000030 00000000 
 18 .debug_line              0000010f 00000000 DEBUG
 19 .rela.debug_line         000001c8 00000000 
 20 .debug_line_str          00000143 00000000 DEBUG
 21 .llvm_addrsig            00000000 00000000 
 22 .symtab                  00000350 00000000 

Disassembly of section .text.RQiHardswish_s8_s8:

00000000 <RQiHardswish_s8_s8>:
;   for (int i = 0; i < size; i++) {
       0: 63 52 c0 06  	blez	a2, 0x64 <RQiHardswish_s8_s8+0x64>
       4: 03 23 01 00  	lw	t1, 0(sp)
       8: 83 22 81 00  	lw	t0, 8(sp)
       c: 83 23 41 00  	lw	t2, 4(sp)
      10: 33 07 e8 00  	add	a4, a6, a4
      14: b3 06 d3 02  	mul	a3, t1, a3
      18: 13 88 f2 ff  	addi	a6, t0, -1
      1c: 13 03 10 00  	li	t1, 1
      20: 33 18 03 01  	sll	a6, t1, a6
      24: 33 08 78 00  	add	a6, a6, t2
      28: 13 03 00 f8  	li	t1, -128
      2c: 93 03 f0 07  	li	t2, 127
;   for (int i = 0; i < size; i++) {
      30: 7b 40 86 01  	<unknown>
;     temp = input[i] + input_offset + three;
      34: 0b 0e 15 00  	<unknown>
      38: b3 0e c7 01  	add	t4, a4, t3
;     if (temp < 0) {
      3c: b3 ee 0e 04  	<unknown>
;     if (temp > six) {
      40: b3 ce fe 04  	<unknown>
;     temp = input[i] * temp;
      44: 33 8e c6 03  	mul	t3, a3, t3
;     temp = temp * (mul) + (add);
      48: 33 0e de 03  	mul	t3, t3, t4
;     temp = ((temp + ((1 << ((shift)-1)))) >> (shift)) + output_offset;
      4c: 93 0e 08 00  	mv	t4, a6
      50: db 2e 5e 40  	<unknown>
      54: 33 8e 1e 01  	add	t3, t4, a7
;     output[i] = (int8_t)CLAMP(temp, -128, 127);
      58: 33 6e 6e 04  	<unknown>
      5c: 33 4e 7e 04  	<unknown>
      60: ab 80 c5 01  	<unknown>
; }
      64: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Relu_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 0000011b 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text.Relu_fp32_fp32    00000024 00000000 TEXT
  4 .debug_loclists         00000050 00000000 DEBUG
  5 .debug_abbrev           00000078 00000000 DEBUG
  6 .debug_info             0000009d 00000000 DEBUG
  7 .rela.debug_info        0000009c 00000000 
  8 .debug_str_offsets      00000044 00000000 DEBUG
  9 .rela.debug_str_offsets 000000b4 00000000 
 10 .debug_str              00000147 00000000 DEBUG
 11 .debug_addr             00000010 00000000 DEBUG
 12 .rela.debug_addr        00000018 00000000 
 13 .comment                00000073 00000000 
 14 .note.GNU-stack         00000000 00000000 
 15 .riscv.attributes       00000030 00000000 
 16 .debug_frame            00000024 00000000 DEBUG
 17 .rela.debug_frame       00000030 00000000 
 18 .debug_line             000000e0 00000000 DEBUG
 19 .rela.debug_line        00000114 00000000 
 20 .debug_line_str         00000163 00000000 DEBUG
 21 .llvm_addrsig           00000000 00000000 
 22 .symtab                 00000270 00000000 

Disassembly of section .text.Relu_fp32_fp32:

00000000 <Relu_fp32_fp32>:
;   for (int i = 0; i < size; i++) {
       0: 63 50 c0 02  	blez	a2, 0x20 <Relu_fp32_fp32+0x20>
       4: 53 00 00 f0  	fmv.w.x	ft0, zero
       8: 7b 40 a6 00  	<unknown>
;     output[i] = MAX(input[i], 0.0f);
       c: 87 20 05 00  	flw	ft1, 0(a0)
      10: d3 90 00 28  	fmax.s	ft1, ft1, ft0
      14: 27 a0 15 00  	fsw	ft1, 0(a1)
;   for (int i = 0; i < size; i++) {
      18: 93 85 45 00  	addi	a1, a1, 4
      1c: 13 05 45 00  	addi	a0, a0, 4
; }
      20: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(RequantShift_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                           Size     VMA      Type
  0                                00000000 00000000 
  1 .strtab                        000001d4 00000000 
  2 .text                          00000000 00000000 TEXT
  3 .text.RequantShift_s8_s8_NHWC  000000a8 00000000 TEXT
  4 .text.RequantShift_s16_s8_NHWC 000000a4 00000000 TEXT
  5 .text.RequantShift_s32_s8_NHWC 000000a4 00000000 TEXT
  6 .text.RequantShift_s8_s8_NCHW  000000a8 00000000 TEXT
  7 .text.RequantShift_s16_s8_NCHW 000000bc 00000000 TEXT
  8 .text.RequantShift_s32_s8_NCHW 000000bc 00000000 TEXT
  9 .debug_loclists                0000085f 00000000 DEBUG
 10 .debug_abbrev                  00000087 00000000 DEBUG
 11 .debug_info                    00000439 00000000 DEBUG
 12 .rela.debug_info               00000198 00000000 
 13 .debug_rnglists                00000029 00000000 DEBUG
 14 .debug_str_offsets             000000a0 00000000 DEBUG
 15 .rela.debug_str_offsets        000001c8 00000000 
 16 .debug_str                     00000285 00000000 DEBUG
 17 .debug_addr                    00000040 00000000 DEBUG
 18 .rela.debug_addr               000000a8 00000000 
 19 .comment                       00000073 00000000 
 20 .note.GNU-stack                00000000 00000000 
 21 .riscv.attributes              00000030 00000000 
 22 .debug_frame                   000000a4 00000000 DEBUG
 23 .rela.debug_frame              00000240 00000000 
 24 .debug_line                    0000054b 00000000 DEBUG
 25 .rela.debug_line               00000da4 00000000 
 26 .debug_line_str                0000014b 00000000 DEBUG
 27 .llvm_addrsig                  00000000 00000000 
 28 .symtab                        00000f50 00000000 

Disassembly of section .text.RequantShift_s8_s8_NHWC:

00000000 <RequantShift_s8_s8_NHWC>:
;                              int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 5a b0 08  	blez	a1, 0x9c <RequantShift_s8_s8_NHWC+0x9c>
       c: 83 22 c1 01  	lw	t0, 28(sp)
      10: 03 2e 81 01  	lw	t3, 24(sp)
      14: 83 2e 41 01  	lw	t4, 20(sp)
      18: 33 23 f0 00  	sgtz	t1, a5
      1c: b3 72 53 00  	and	t0, t1, t0
      20: 63 88 02 00  	beqz	t0, 0x30 <RequantShift_s8_s8_NHWC+0x30>
      24: 93 82 f7 ff  	addi	t0, a5, -1
      28: 13 03 10 00  	li	t1, 1
      2c: b3 12 53 00  	sll	t0, t1, t0
      30: 03 23 01 01  	lw	t1, 16(sp)
      34: 93 03 00 00  	li	t2, 0
      38: 33 6e 0e 10  	<unknown>
      3c: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      40: 7b c0 a5 02  	<unknown>
      44: 6f 00 40 01  	j	0x58 <RequantShift_s8_s8_NHWC+0x58>
;     data_out[i] = out;
      48: b3 0f 77 00  	add	t6, a4, t2
;   for (int i = 0; i < size; i++) {
      4c: 93 83 13 00  	addi	t2, t2, 1
;     data_out[i] = out;
      50: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      54: 63 84 75 04  	beq	a1, t2, 0x9c <RequantShift_s8_s8_NHWC+0x9c>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      58: 33 0f 75 00  	add	t5, a0, t2
      5c: 03 0f 0f 00  	lb	t5, 0(t5)
      60: b3 ef 03 03  	rem	t6, t2, a6
      64: 93 9f 2f 00  	slli	t6, t6, 2
      68: 33 04 f6 01  	add	s0, a2, t6
      6c: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      70: b3 8f f6 01  	add	t6, a3, t6
      74: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      78: 33 0f 1f 01  	add	t5, t5, a7
      7c: 33 0f 8f 02  	mul	t5, t5, s0
      80: b3 8f 5f 00  	add	t6, t6, t0
;     intermediate = (intermediate >> log2D) + output_offset;
      84: db 2f ff 40  	<unknown>
      88: b3 8f 6f 00  	add	t6, t6, t1
      8c: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      90: e3 4c fe fb  	blt	t3, t6, 0x48 <RequantShift_s8_s8_NHWC+0x48>
      94: 33 ef df 05  	<unknown>
      98: 6f f0 1f fb  	j	0x48 <RequantShift_s8_s8_NHWC+0x48>
; }
      9c: 03 24 c1 00  	lw	s0, 12(sp)
      a0: 13 01 01 01  	addi	sp, sp, 16
      a4: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s16_s8_NHWC:

00000000 <RequantShift_s16_s8_NHWC>:
;                               int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 58 b0 08  	blez	a1, 0x98 <RequantShift_s16_s8_NHWC+0x98>
       c: 83 22 c1 01  	lw	t0, 28(sp)
      10: 03 2e 81 01  	lw	t3, 24(sp)
      14: 83 2e 41 01  	lw	t4, 20(sp)
      18: 33 23 f0 00  	sgtz	t1, a5
      1c: b3 72 53 00  	and	t0, t1, t0
      20: 63 88 02 00  	beqz	t0, 0x30 <RequantShift_s16_s8_NHWC+0x30>
      24: 93 82 f7 ff  	addi	t0, a5, -1
      28: 13 03 10 00  	li	t1, 1
      2c: b3 12 53 00  	sll	t0, t1, t0
      30: 03 23 01 01  	lw	t1, 16(sp)
      34: 93 03 00 00  	li	t2, 0
      38: 33 6e 0e 10  	<unknown>
      3c: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      40: 7b c0 85 02  	<unknown>
      44: 6f 00 40 01  	j	0x58 <RequantShift_s16_s8_NHWC+0x58>
;     data_out[i] = out;
      48: b3 0f 77 00  	add	t6, a4, t2
;   for (int i = 0; i < size; i++) {
      4c: 93 83 13 00  	addi	t2, t2, 1
;     data_out[i] = out;
      50: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      54: 63 82 75 04  	beq	a1, t2, 0x98 <RequantShift_s16_s8_NHWC+0x98>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      58: 0b 1f 25 00  	<unknown>
      5c: b3 ef 03 03  	rem	t6, t2, a6
      60: 93 9f 2f 00  	slli	t6, t6, 2
      64: 33 04 f6 01  	add	s0, a2, t6
      68: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      6c: b3 8f f6 01  	add	t6, a3, t6
      70: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      74: 33 0f 1f 01  	add	t5, t5, a7
      78: 33 0f 8f 02  	mul	t5, t5, s0
      7c: b3 8f 5f 00  	add	t6, t6, t0
;     intermediate = (intermediate >> log2D) + output_offset;
      80: db 2f ff 40  	<unknown>
      84: b3 8f 6f 00  	add	t6, t6, t1
      88: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      8c: e3 4e fe fb  	blt	t3, t6, 0x48 <RequantShift_s16_s8_NHWC+0x48>
      90: 33 ef df 05  	<unknown>
      94: 6f f0 5f fb  	j	0x48 <RequantShift_s16_s8_NHWC+0x48>
; }
      98: 03 24 c1 00  	lw	s0, 12(sp)
      9c: 13 01 01 01  	addi	sp, sp, 16
      a0: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s32_s8_NHWC:

00000000 <RequantShift_s32_s8_NHWC>:
;                               int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 58 b0 08  	blez	a1, 0x98 <RequantShift_s32_s8_NHWC+0x98>
       c: 83 22 c1 01  	lw	t0, 28(sp)
      10: 03 2e 81 01  	lw	t3, 24(sp)
      14: 83 2e 41 01  	lw	t4, 20(sp)
      18: 33 23 f0 00  	sgtz	t1, a5
      1c: b3 72 53 00  	and	t0, t1, t0
      20: 63 88 02 00  	beqz	t0, 0x30 <RequantShift_s32_s8_NHWC+0x30>
      24: 93 82 f7 ff  	addi	t0, a5, -1
      28: 13 03 10 00  	li	t1, 1
      2c: b3 12 53 00  	sll	t0, t1, t0
      30: 03 23 01 01  	lw	t1, 16(sp)
      34: 93 03 00 00  	li	t2, 0
      38: 33 6e 0e 10  	<unknown>
      3c: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      40: 7b c0 85 02  	<unknown>
      44: 6f 00 40 01  	j	0x58 <RequantShift_s32_s8_NHWC+0x58>
;     data_out[i] = out;
      48: b3 0f 77 00  	add	t6, a4, t2
;   for (int i = 0; i < size; i++) {
      4c: 93 83 13 00  	addi	t2, t2, 1
;     data_out[i] = out;
      50: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      54: 63 82 75 04  	beq	a1, t2, 0x98 <RequantShift_s32_s8_NHWC+0x98>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      58: 0b 2f 45 00  	<unknown>
      5c: b3 ef 03 03  	rem	t6, t2, a6
      60: 93 9f 2f 00  	slli	t6, t6, 2
      64: 33 04 f6 01  	add	s0, a2, t6
      68: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      6c: b3 8f f6 01  	add	t6, a3, t6
      70: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      74: 33 0f 1f 01  	add	t5, t5, a7
      78: 33 0f e4 03  	mul	t5, s0, t5
      7c: b3 8f 5f 00  	add	t6, t6, t0
;     intermediate = (intermediate >> log2D) + output_offset;
      80: db 2f ff 40  	<unknown>
      84: b3 8f 6f 00  	add	t6, t6, t1
      88: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      8c: e3 4e fe fb  	blt	t3, t6, 0x48 <RequantShift_s32_s8_NHWC+0x48>
      90: 33 ef df 05  	<unknown>
      94: 6f f0 5f fb  	j	0x48 <RequantShift_s32_s8_NHWC+0x48>
; }
      98: 03 24 c1 00  	lw	s0, 12(sp)
      9c: 13 01 01 01  	addi	sp, sp, 16
      a0: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s8_s8_NCHW:

00000000 <RequantShift_s8_s8_NCHW>:
;                              int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 5a b0 08  	blez	a1, 0x9c <RequantShift_s8_s8_NCHW+0x9c>
       c: 83 22 c1 01  	lw	t0, 28(sp)
      10: 03 2e 81 01  	lw	t3, 24(sp)
      14: 83 2e 41 01  	lw	t4, 20(sp)
      18: 33 23 f0 00  	sgtz	t1, a5
      1c: b3 72 53 00  	and	t0, t1, t0
      20: 63 88 02 00  	beqz	t0, 0x30 <RequantShift_s8_s8_NCHW+0x30>
      24: 93 82 f7 ff  	addi	t0, a5, -1
      28: 13 03 10 00  	li	t1, 1
      2c: b3 12 53 00  	sll	t0, t1, t0
      30: 03 23 01 01  	lw	t1, 16(sp)
      34: 93 03 00 00  	li	t2, 0
      38: 33 6e 0e 10  	<unknown>
      3c: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      40: 7b c0 a5 02  	<unknown>
      44: 6f 00 40 01  	j	0x58 <RequantShift_s8_s8_NCHW+0x58>
;     data_out[i] = out;
      48: b3 0f 77 00  	add	t6, a4, t2
;   for (int i = 0; i < size; i++) {
      4c: 93 83 13 00  	addi	t2, t2, 1
;     data_out[i] = out;
      50: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      54: 63 84 75 04  	beq	a1, t2, 0x9c <RequantShift_s8_s8_NCHW+0x9c>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      58: 33 0f 75 00  	add	t5, a0, t2
      5c: 03 0f 0f 00  	lb	t5, 0(t5)
      60: b3 cf 03 03  	div	t6, t2, a6
      64: 93 9f 2f 00  	slli	t6, t6, 2
      68: 33 04 f6 01  	add	s0, a2, t6
      6c: 03 24 04 00  	lw	s0, 0(s0)
      70: b3 8f f6 01  	add	t6, a3, t6
      74: 83 af 0f 00  	lw	t6, 0(t6)
      78: 33 0f 1f 01  	add	t5, t5, a7
      7c: 33 0f 8f 02  	mul	t5, t5, s0
      80: b3 8f 5f 00  	add	t6, t6, t0
;     intermediate = (intermediate >> log2D) + output_offset;
      84: db 2f ff 40  	<unknown>
      88: b3 8f 6f 00  	add	t6, t6, t1
      8c: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      90: e3 4c fe fb  	blt	t3, t6, 0x48 <RequantShift_s8_s8_NCHW+0x48>
      94: 33 ef df 05  	<unknown>
      98: 6f f0 1f fb  	j	0x48 <RequantShift_s8_s8_NCHW+0x48>
; }
      9c: 03 24 c1 00  	lw	s0, 12(sp)
      a0: 13 01 01 01  	addi	sp, sp, 16
      a4: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s16_s8_NCHW:

00000000 <RequantShift_s16_s8_NCHW>:
;                               int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 54 b0 0a  	blez	a1, 0xb0 <RequantShift_s16_s8_NCHW+0xb0>
       c: 83 22 c1 01  	lw	t0, 28(sp)
      10: 03 2e 81 01  	lw	t3, 24(sp)
      14: 83 2e 41 01  	lw	t4, 20(sp)
      18: 33 23 f0 00  	sgtz	t1, a5
      1c: b3 72 53 00  	and	t0, t1, t0
      20: 63 88 02 00  	beqz	t0, 0x30 <RequantShift_s16_s8_NCHW+0x30>
      24: 93 82 f7 ff  	addi	t0, a5, -1
      28: 13 03 10 00  	li	t1, 1
      2c: b3 12 53 00  	sll	t0, t1, t0
      30: 03 23 01 01  	lw	t1, 16(sp)
      34: 93 03 00 00  	li	t2, 0
      38: 33 6e 0e 10  	<unknown>
      3c: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      40: fb c0 45 03  	<unknown>
      44: 6f 00 40 01  	j	0x58 <RequantShift_s16_s8_NCHW+0x58>
;     data_out[i] = out;
      48: b3 0f 77 00  	add	t6, a4, t2
;   for (int i = 0; i < size; i++) {
      4c: 93 83 13 00  	addi	t2, t2, 1
;     data_out[i] = out;
      50: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      54: 63 8e b3 04  	beq	t2, a1, 0xb0 <RequantShift_s16_s8_NCHW+0xb0>
;     intermediate = (int32_t)data_in[i];
      58: 13 9f 13 00  	slli	t5, t2, 1
      5c: 33 0f e5 01  	add	t5, a0, t5
      60: 03 1f 0f 00  	lh	t5, 0(t5)
      64: 93 0f 30 00  	li	t6, 3
;     for (int j = 0; j < 3; j++) {
      68: 7b 50 32 00  	<unknown>
      6c: 93 8f ff ff  	addi	t6, t6, -1
;       asm volatile("nop" ::);
      70: 13 00 00 00  	nop
;     intermediate = (intermediate + input_offset) * mul[i / HW] + add[i / HW];
      74: b3 cf 03 03  	div	t6, t2, a6
      78: 93 9f 2f 00  	slli	t6, t6, 2
      7c: 33 04 f6 01  	add	s0, a2, t6
      80: 03 24 04 00  	lw	s0, 0(s0)
      84: b3 8f f6 01  	add	t6, a3, t6
      88: 83 af 0f 00  	lw	t6, 0(t6)
      8c: 33 0f 1f 01  	add	t5, t5, a7
      90: 33 0f e4 03  	mul	t5, s0, t5
      94: b3 8f 5f 00  	add	t6, t6, t0
;     intermediate = (intermediate >> log2D) + output_offset;
      98: db 2f ff 40  	<unknown>
      9c: b3 8f 6f 00  	add	t6, t6, t1
      a0: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      a4: e3 42 fe fb  	blt	t3, t6, 0x48 <RequantShift_s16_s8_NCHW+0x48>
      a8: 33 ef df 05  	<unknown>
      ac: 6f f0 df f9  	j	0x48 <RequantShift_s16_s8_NCHW+0x48>
; }
      b0: 03 24 c1 00  	lw	s0, 12(sp)
      b4: 13 01 01 01  	addi	sp, sp, 16
      b8: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s32_s8_NCHW:

00000000 <RequantShift_s32_s8_NCHW>:
;                               int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 54 b0 0a  	blez	a1, 0xb0 <RequantShift_s32_s8_NCHW+0xb0>
       c: 83 22 c1 01  	lw	t0, 28(sp)
      10: 03 2e 81 01  	lw	t3, 24(sp)
      14: 83 2e 41 01  	lw	t4, 20(sp)
      18: 33 23 f0 00  	sgtz	t1, a5
      1c: b3 72 53 00  	and	t0, t1, t0
      20: 63 88 02 00  	beqz	t0, 0x30 <RequantShift_s32_s8_NCHW+0x30>
      24: 93 82 f7 ff  	addi	t0, a5, -1
      28: 13 03 10 00  	li	t1, 1
      2c: b3 12 53 00  	sll	t0, t1, t0
      30: 03 23 01 01  	lw	t1, 16(sp)
      34: 93 03 00 00  	li	t2, 0
      38: 33 6e 0e 10  	<unknown>
      3c: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      40: fb c0 45 03  	<unknown>
      44: 6f 00 40 01  	j	0x58 <RequantShift_s32_s8_NCHW+0x58>
;     data_out[i] = out;
      48: b3 0f 77 00  	add	t6, a4, t2
;   for (int i = 0; i < size; i++) {
      4c: 93 83 13 00  	addi	t2, t2, 1
;     data_out[i] = out;
      50: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      54: 63 8e b3 04  	beq	t2, a1, 0xb0 <RequantShift_s32_s8_NCHW+0xb0>
;     intermediate = (int32_t)data_in[i];
      58: 13 9f 23 00  	slli	t5, t2, 2
      5c: 33 0f e5 01  	add	t5, a0, t5
      60: 03 2f 0f 00  	lw	t5, 0(t5)
      64: 93 0f 30 00  	li	t6, 3
;     for (int j = 0; j < 3; j++) {
      68: 7b 50 32 00  	<unknown>
      6c: 93 8f ff ff  	addi	t6, t6, -1
;       asm volatile("nop" ::);
      70: 13 00 00 00  	nop
;     intermediate = (intermediate + input_offset) * mul[i / HW] + add[i / HW];
      74: b3 cf 03 03  	div	t6, t2, a6
      78: 93 9f 2f 00  	slli	t6, t6, 2
      7c: 33 04 f6 01  	add	s0, a2, t6
      80: 03 24 04 00  	lw	s0, 0(s0)
      84: b3 8f f6 01  	add	t6, a3, t6
      88: 83 af 0f 00  	lw	t6, 0(t6)
      8c: 33 0f 1f 01  	add	t5, t5, a7
      90: 33 0f e4 03  	mul	t5, s0, t5
      94: b3 8f 5f 00  	add	t6, t6, t0
;     intermediate = (intermediate >> log2D) + output_offset;
      98: db 2f ff 40  	<unknown>
      9c: b3 8f 6f 00  	add	t6, t6, t1
      a0: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      a4: e3 42 fe fb  	blt	t3, t6, 0x48 <RequantShift_s32_s8_NCHW+0x48>
      a8: 33 ef df 05  	<unknown>
      ac: 6f f0 df f9  	j	0x48 <RequantShift_s32_s8_NCHW+0x48>
; }
      b0: 03 24 c1 00  	lw	s0, 12(sp)
      b4: 13 01 01 01  	addi	sp, sp, 16
      b8: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Softmax_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                             Size     VMA      Type
  0                                  00000000 00000000 
  1 .strtab                          00000175 00000000 
  2 .text                            00000000 00000000 TEXT
  3 .sdata                           00000008 00000000 DATA
  4 .text.Softmax_fp32_fp32          00000144 00000000 TEXT
  5 .rela.text.Softmax_fp32_fp32     0000003c 00000000 
  6 .text.SoftmaxGrad_fp32_fp32_fp32 00000094 00000000 TEXT
  7 .debug_loclists                  0000028a 00000000 DEBUG
  8 .debug_abbrev                    00000087 00000000 DEBUG
  9 .debug_info                      000001a2 00000000 DEBUG
 10 .rela.debug_info                 000001b0 00000000 
 11 .debug_rnglists                  00000019 00000000 DEBUG
 12 .debug_str_offsets               00000070 00000000 DEBUG
 13 .rela.debug_str_offsets          00000138 00000000 
 14 .debug_str                       000001d8 00000000 DEBUG
 15 .debug_addr                      00000038 00000000 DEBUG
 16 .rela.debug_addr                 00000090 00000000 
 17 .comment                         00000073 00000000 
 18 .note.GNU-stack                  00000000 00000000 
 19 .riscv.attributes                00000030 00000000 
 20 .debug_frame                     00000054 00000000 DEBUG
 21 .rela.debug_frame                00000090 00000000 
 22 .debug_line                      00000247 00000000 DEBUG
 23 .rela.debug_line                 00000528 00000000 
 24 .debug_line_str                  00000169 00000000 DEBUG
 25 .llvm_addrsig                    00000000 00000000 
 26 .symtab                          000007a0 00000000 

Disassembly of section .text.Softmax_fp32_fp32:

00000000 <Softmax_fp32_fp32>:
;                        int32_t last_dim_length) {
       0: 13 01 01 fc  	addi	sp, sp, -64
       4: 23 2e 11 02  	sw	ra, 60(sp)
       8: 23 2c 81 02  	sw	s0, 56(sp)
       c: 23 2a 91 02  	sw	s1, 52(sp)
      10: 23 28 21 03  	sw	s2, 48(sp)
      14: 23 26 31 03  	sw	s3, 44(sp)
      18: 23 24 41 03  	sw	s4, 40(sp)
      1c: 23 22 51 03  	sw	s5, 36(sp)
      20: 23 20 61 03  	sw	s6, 32(sp)
      24: 23 2e 71 01  	sw	s7, 28(sp)
      28: 27 2c 81 00  	fsw	fs0, 24(sp)
      2c: 27 2a 91 00  	fsw	fs1, 20(sp)
      30: 27 28 21 01  	fsw	fs2, 16(sp)
      34: 27 26 31 01  	fsw	fs3, 12(sp)
      38: 27 24 41 01  	fsw	fs4, 8(sp)
      3c: 13 84 06 00  	mv	s0, a3
;   int32_t batch_size = size / last_dim_length;
      40: b3 49 d6 02  	div	s3, a2, a3
;   for (int b = 0; b < batch_size; b++) {
      44: 13 a6 19 00  	slti	a2, s3, 1
      48: 93 a6 16 00  	slti	a3, a3, 1
;   for (int b = 0; b < batch_size; b++) {
      4c: 33 66 d6 00  	or	a2, a2, a3
      50: 63 1a 06 0a  	bnez	a2, 0x104 <Softmax_fp32_fp32+0x104>
      54: 93 84 05 00  	mv	s1, a1
      58: 13 09 05 00  	mv	s2, a0
      5c: 37 05 00 00  	lui	a0, 0
      60: 07 24 05 00  	flw	fs0, 0(a0)
      64: 37 05 00 00  	lui	a0, 0
      68: 87 24 05 00  	flw	fs1, 0(a0)
      6c: 13 0a 00 00  	li	s4, 0
;   for (int b = 0; b < batch_size; b++) {
      70: 93 1a 24 00  	slli	s5, s0, 2
      74: 53 09 00 f0  	fmv.w.x	fs2, zero
      78: 13 05 09 00  	mv	a0, s2
      7c: 93 05 04 00  	mv	a1, s0
      80: d3 09 84 20  	fmv.s	fs3, fs0
;     for (int i = 0; i < last_dim_length; i++) {
      84: 7b 40 84 00  	<unknown>
;       if (input[b * last_dim_length + i] > max_val) {
      88: 07 20 05 00  	flw	ft0, 0(a0)
      8c: d3 19 30 29  	fmax.s	fs3, ft0, fs3
;     for (int i = 0; i < last_dim_length; i++) {
      90: 93 85 f5 ff  	addi	a1, a1, -1
      94: 13 05 45 00  	addi	a0, a0, 4
      98: 13 0b 00 00  	li	s6, 0
      9c: 93 0b 04 00  	mv	s7, s0
      a0: 53 0a 29 21  	fmv.s	fs4, fs2
;       float32_t exp_val = input[b * last_dim_length + i] - max_val;
      a4: 33 05 69 01  	add	a0, s2, s6
      a8: 07 20 05 00  	flw	ft0, 0(a0)
      ac: 53 75 30 09  	fsub.s	fa0, ft0, fs3
;       output[b * last_dim_length + i] = expf(exp_val);
      b0: 97 00 00 00  	auipc	ra, 0
      b4: e7 80 00 00  	jalr	ra
      b8: 33 85 64 01  	add	a0, s1, s6
      bc: 27 20 a5 00  	fsw	fa0, 0(a0)
;       sum += output[b * last_dim_length + i];
      c0: 53 7a 45 01  	fadd.s	fs4, fa0, fs4
;     for (int i = 0; i < last_dim_length; i++) {
      c4: 93 8b fb ff  	addi	s7, s7, -1
      c8: 13 0b 4b 00  	addi	s6, s6, 4
      cc: e3 9c 0b fc  	bnez	s7, 0xa4 <Softmax_fp32_fp32+0xa4>
      d0: 53 f0 44 19  	fdiv.s	ft0, fs1, fs4
      d4: 13 85 04 00  	mv	a0, s1
      d8: 93 05 04 00  	mv	a1, s0
;     for (int i = 0; i < last_dim_length; i++) {
      dc: 7b 40 a4 00  	<unknown>
;       output[b * last_dim_length + i] = output[b * last_dim_length + i] * sum_1;
      e0: 87 20 05 00  	flw	ft1, 0(a0)
      e4: d3 f0 00 10  	fmul.s	ft1, ft1, ft0
      e8: 27 20 15 00  	fsw	ft1, 0(a0)
;     for (int i = 0; i < last_dim_length; i++) {
      ec: 93 85 f5 ff  	addi	a1, a1, -1
      f0: 13 05 45 00  	addi	a0, a0, 4
;   for (int b = 0; b < batch_size; b++) {
      f4: 13 0a 1a 00  	addi	s4, s4, 1
      f8: 33 09 59 01  	add	s2, s2, s5
      fc: b3 84 54 01  	add	s1, s1, s5
     100: e3 1c 3a f7  	bne	s4, s3, 0x78 <Softmax_fp32_fp32+0x78>
; }
     104: 83 20 c1 03  	lw	ra, 60(sp)
     108: 03 24 81 03  	lw	s0, 56(sp)
     10c: 83 24 41 03  	lw	s1, 52(sp)
     110: 03 29 01 03  	lw	s2, 48(sp)
     114: 83 29 c1 02  	lw	s3, 44(sp)
     118: 03 2a 81 02  	lw	s4, 40(sp)
     11c: 83 2a 41 02  	lw	s5, 36(sp)
     120: 03 2b 01 02  	lw	s6, 32(sp)
     124: 83 2b c1 01  	lw	s7, 28(sp)
     128: 07 24 81 01  	flw	fs0, 24(sp)
     12c: 87 24 41 01  	flw	fs1, 20(sp)
     130: 07 29 01 01  	flw	fs2, 16(sp)
     134: 87 29 c1 00  	flw	fs3, 12(sp)
     138: 07 2a 81 00  	flw	fs4, 8(sp)
     13c: 13 01 01 04  	addi	sp, sp, 64
     140: 67 80 00 00  	ret

Disassembly of section .text.SoftmaxGrad_fp32_fp32_fp32:

00000000 <SoftmaxGrad_fp32_fp32_fp32>:
;   int32_t batch_size = size / last_dim_length;
       0: b3 c6 e6 02  	div	a3, a3, a4
;   for (int b = 0; b < batch_size; b++) {
       4: 93 a7 16 00  	slti	a5, a3, 1
       8: 13 28 17 00  	slti	a6, a4, 1
       c: b3 e7 07 01  	or	a5, a5, a6
      10: 63 90 07 08  	bnez	a5, 0x90 <SoftmaxGrad_fp32_fp32_fp32+0x90>
      14: 13 18 27 00  	slli	a6, a4, 2
      18: 53 00 00 f0  	fmv.w.x	ft0, zero
      1c: fb c0 86 03  	<unknown>
      20: 93 08 05 00  	mv	a7, a0
      24: 93 82 05 00  	mv	t0, a1
      28: 13 03 07 00  	mv	t1, a4
      2c: d3 00 00 20  	fmv.s	ft1, ft0
;     for (int i = 0; i < last_dim_length; i++) {
      30: 7b 40 c7 00  	<unknown>
;       weighted_sum += upstream_grad[idx] * softmax_output[idx];
      34: 07 a1 08 00  	flw	ft2, 0(a7)
      38: 87 a1 02 00  	flw	ft3, 0(t0)
      3c: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;     for (int i = 0; i < last_dim_length; i++) {
      40: 13 03 f3 ff  	addi	t1, t1, -1
      44: 93 82 42 00  	addi	t0, t0, 4
      48: 93 88 48 00  	addi	a7, a7, 4
      4c: 93 08 00 00  	li	a7, 0
      50: 93 02 07 00  	mv	t0, a4
;     for (int i = 0; i < last_dim_length; i++) {
      54: 7b 40 47 01  	<unknown>
;           softmax_output[idx] * (upstream_grad[idx] - weighted_sum);
      58: 33 83 15 01  	add	t1, a1, a7
      5c: b3 03 15 01  	add	t2, a0, a7
      60: 07 a1 03 00  	flw	ft2, 0(t2)
      64: 87 21 03 00  	flw	ft3, 0(t1)
      68: 53 71 11 08  	fsub.s	ft2, ft2, ft1
      6c: 53 71 31 10  	fmul.s	ft2, ft2, ft3
;       softmax_gradient[idx] =
      70: 33 03 16 01  	add	t1, a2, a7
      74: 27 20 23 00  	fsw	ft2, 0(t1)
;     for (int i = 0; i < last_dim_length; i++) {
      78: 93 82 f2 ff  	addi	t0, t0, -1
      7c: 93 88 48 00  	addi	a7, a7, 4
;   for (int b = 0; b < batch_size; b++) {
      80: 93 87 17 00  	addi	a5, a5, 1
      84: b3 85 05 01  	add	a1, a1, a6
      88: 33 05 05 01  	add	a0, a0, a6
      8c: 33 06 06 01  	add	a2, a2, a6
; }
      90: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Softmax_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                     Size     VMA      Type
  0                          00000000 00000000 
  1 .strtab                  00000173 00000000 
  2 .text                    00000000 00000000 TEXT
  3 .text.Softmax_s8_s8      00000194 00000000 TEXT
  4 .rela.text.Softmax_s8_s8 00000018 00000000 
  5 .text.ITAMax_s8          000000b4 00000000 TEXT
  6 .text.ITAPartialMax_s8   00000158 00000000 TEXT
  7 .debug_loclists          00000671 00000000 DEBUG
  8 .debug_abbrev            000000e2 00000000 DEBUG
  9 .debug_info              00000363 00000000 DEBUG
 10 .rela.debug_info         000000f0 00000000 
 11 .debug_rnglists          000000a6 00000000 DEBUG
 12 .debug_str_offsets       00000118 00000000 DEBUG
 13 .rela.debug_str_offsets  00000330 00000000 
 14 .debug_str               0000034d 00000000 DEBUG
 15 .debug_addr              0000002c 00000000 DEBUG
 16 .rela.debug_addr         0000006c 00000000 
 17 .comment                 00000073 00000000 
 18 .note.GNU-stack          00000000 00000000 
 19 .riscv.attributes        00000030 00000000 
 20 .debug_frame             00000070 00000000 DEBUG
 21 .rela.debug_frame        000000f0 00000000 
 22 .debug_line              00000469 00000000 DEBUG
 23 .rela.debug_line         00000af8 00000000 
 24 .debug_line_str          00000194 00000000 DEBUG
 25 .llvm_addrsig            00000000 00000000 
 26 .symtab                  00000dc0 00000000 

Disassembly of section .text.Softmax_s8_s8:

00000000 <Softmax_s8_s8>:
;                    int64_t coeffC, int32_t log2, uint32_t n_levels) {
       0: 13 01 01 fd  	addi	sp, sp, -48
       4: 23 26 11 02  	sw	ra, 44(sp)
       8: 23 24 81 02  	sw	s0, 40(sp)
       c: 23 22 91 02  	sw	s1, 36(sp)
      10: 23 20 21 03  	sw	s2, 32(sp)
      14: 23 2e 31 01  	sw	s3, 28(sp)
      18: 23 2c 41 01  	sw	s4, 24(sp)
      1c: 23 2a 51 01  	sw	s5, 20(sp)
      20: 23 28 61 01  	sw	s6, 16(sp)
      24: 23 26 71 01  	sw	s7, 12(sp)
      28: 23 24 81 01  	sw	s8, 8(sp)
      2c: 23 22 91 01  	sw	s9, 4(sp)
      30: 83 2c 41 03  	lw	s9, 52(sp)
      34: 03 2c 01 03  	lw	s8, 48(sp)
      38: 13 84 08 00  	mv	s0, a7
      3c: 93 04 08 00  	mv	s1, a6
      40: 13 89 07 00  	mv	s2, a5
      44: 93 09 07 00  	mv	s3, a4
      48: 13 8a 06 00  	mv	s4, a3
      4c: 93 0b 06 00  	mv	s7, a2
      50: 93 8a 05 00  	mv	s5, a1
      54: 13 0b 05 00  	mv	s6, a0
;   uint32_t *y = (uint32_t *)deeploy_malloc(sizeof(int32_t) * lastDimLength);
      58: 13 95 26 00  	slli	a0, a3, 2
      5c: 97 00 00 00  	auipc	ra, 0
      60: e7 80 00 00  	jalr	ra
;   for (uint32_t i = 0; i < size / lastDimLength; i++) {
      64: 63 ec 4b 0f  	bltu	s7, s4, 0x15c <Softmax_s8_s8+0x15c>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      68: 63 0a 0a 0e  	beqz	s4, 0x15c <Softmax_s8_s8+0x15c>
      6c: 93 05 00 00  	li	a1, 0
      70: 33 d7 4b 03  	divu	a4, s7, s4
      74: 13 86 fc ff  	addi	a2, s9, -1
      78: 93 d6 1c 00  	srli	a3, s9, 1
      7c: 93 07 10 00  	li	a5, 1
      80: 33 77 f7 04  	<unknown>
      84: 93 07 f0 01  	li	a5, 31
      88: 13 08 00 00  	li	a6, 0
      8c: 93 08 00 08  	li	a7, 128
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      90: 7b 40 aa 00  	<unknown>
;       if (data_in[j + i * lastDimLength] > x_max) {
      94: b3 02 0b 01  	add	t0, s6, a6
      98: 83 82 02 00  	lb	t0, 0(t0)
      9c: b3 e8 08 10  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      a0: 13 08 18 00  	addi	a6, a6, 1
;       if (data_in[j + i * lastDimLength] > x_max) {
      a4: b3 e8 12 05  	<unknown>
      a8: 93 02 00 00  	li	t0, 0
      ac: 13 08 00 00  	li	a6, 0
      b0: 13 03 05 00  	mv	t1, a0
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      b4: 7b 40 6a 03  	<unknown>
;       xTilde = (data_in[j + i * lastDimLength] - x_max);
      b8: b3 03 5b 00  	add	t2, s6, t0
      bc: 83 83 03 00  	lb	t2, 0(t2)
      c0: b3 83 13 41  	sub	t2, t2, a7
;       z = (int8_t)(-(xTilde / log2));
      c4: 33 ce 83 03  	div	t3, t2, s8
;       z = CLAMP(z, 0, 31);
      c8: 13 1e 8e 01  	slli	t3, t3, 24
      cc: 5b 3e c0 31  	<unknown>
      d0: 33 6e 0e 04  	<unknown>
      d4: 33 4e fe 04  	<unknown>
;       p = (int16_t)(xTilde + z * log2);
      d8: b3 03 8e 43  	<unknown>
;       y[j] = (uint32_t)((uint64_t)(coeffA * ((p + coeffB) * (p + coeffB)) +
      dc: b3 c3 03 10  	<unknown>
      e0: b3 83 23 01  	add	t2, t2, s2
      e4: b3 8e 33 03  	mul	t4, t2, s3
      e8: 33 8f 7e 02  	mul	t5, t4, t2
      ec: 93 5f ff 41  	srai	t6, t5, 31
      f0: 93 8b 04 00  	mv	s7, s1
      f4: b3 8b 7e 42  	<unknown>
      f8: b3 b3 eb 01  	sltu	t2, s7, t5
      fc: b3 8e 8f 00  	add	t4, t6, s0
     100: b3 83 7e 00  	add	t2, t4, t2
;                                    coeffC) >>
     104: 93 93 13 00  	slli	t2, t2, 1
     108: 93 4e fe 01  	xori	t4, t3, 31
     10c: b3 93 d3 01  	sll	t2, t2, t4
     110: 5b af c4 c1  	<unknown>
     114: b3 63 7f 00  	or	t2, t5, t2
;       y[j] = (uint32_t)((uint64_t)(coeffA * ((p + coeffB) * (p + coeffB)) +
     118: 2b 22 73 00  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
     11c: 93 82 12 00  	addi	t0, t0, 1
;       y_sum += y[j];
     120: 33 08 78 00  	add	a6, a6, t2
     124: 93 08 05 00  	mv	a7, a0
     128: 93 82 0a 00  	mv	t0, s5
     12c: 13 03 0a 00  	mv	t1, s4
;           (int8_t)((y[j] * (n_levels - 1)) / (y_sum)-n_levels / 2);
     130: 8b a3 48 00  	<unknown>
     134: b3 83 c3 02  	mul	t2, t2, a2
     138: b3 d3 03 03  	divu	t2, t2, a6
     13c: b3 83 d3 40  	sub	t2, t2, a3
;     for (uint32_t j = 0; j < lastDimLength; j++) {
     140: 13 03 f3 ff  	addi	t1, t1, -1
;       data_out[j + i * lastDimLength] =
     144: ab 80 72 00  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
     148: e3 14 03 fe  	bnez	t1, 0x130 <Softmax_s8_s8+0x130>
;   for (uint32_t i = 0; i < size / lastDimLength; i++) {
     14c: 93 85 15 00  	addi	a1, a1, 1
     150: 33 0b 4b 01  	add	s6, s6, s4
     154: b3 8a 4a 01  	add	s5, s5, s4
     158: e3 98 e5 f2  	bne	a1, a4, 0x88 <Softmax_s8_s8+0x88>
;   deeploy_free(y);
     15c: 83 20 c1 02  	lw	ra, 44(sp)
     160: 03 24 81 02  	lw	s0, 40(sp)
     164: 83 24 41 02  	lw	s1, 36(sp)
     168: 03 29 01 02  	lw	s2, 32(sp)
     16c: 83 29 c1 01  	lw	s3, 28(sp)
     170: 03 2a 81 01  	lw	s4, 24(sp)
     174: 83 2a 41 01  	lw	s5, 20(sp)
     178: 03 2b 01 01  	lw	s6, 16(sp)
     17c: 83 2b c1 00  	lw	s7, 12(sp)
     180: 03 2c 81 00  	lw	s8, 8(sp)
     184: 83 2c 41 00  	lw	s9, 4(sp)
     188: 13 01 01 03  	addi	sp, sp, 48
     18c: 17 03 00 00  	auipc	t1, 0
     190: 67 00 03 00  	jr	t1

Disassembly of section .text.ITAMax_s8:

00000000 <ITAMax_s8>:
;   for (i = 0; i < size / lastDimLength; ++i) {
       0: 63 e8 e6 0a  	bltu	a3, a4, 0xb0 <ITAMax_s8+0xb0>
;     for (j = 0; j < lastDimLength; ++j) {
       4: 63 06 07 0a  	beqz	a4, 0xb0 <ITAMax_s8+0xb0>
       8: 13 08 00 00  	li	a6, 0
       c: b3 d8 e6 02  	divu	a7, a3, a4
      10: 93 96 87 00  	slli	a3, a5, 8
      14: 93 86 06 f0  	addi	a3, a3, -256
      18: 93 d7 17 00  	srli	a5, a5, 1
      1c: 93 02 10 00  	li	t0, 1
      20: b3 f8 58 04  	<unknown>
      24: 93 02 00 10  	li	t0, 256
      28: 93 03 00 00  	li	t2, 0
      2c: 13 03 00 08  	li	t1, 128
;     for (j = 0; j < lastDimLength; ++j) {
      30: 7b 40 a7 00  	<unknown>
;       if (pSrcA[i * lastDimLength + j] > max) {
      34: 33 0e 75 00  	add	t3, a0, t2
      38: 03 0e 0e 00  	lb	t3, 0(t3)
      3c: 33 63 03 10  	<unknown>
;     for (j = 0; j < lastDimLength; ++j) {
      40: 93 83 13 00  	addi	t2, t2, 1
;       if (pSrcA[i * lastDimLength + j] > max) {
      44: 33 63 6e 04  	<unknown>
      48: 13 0e 00 00  	li	t3, 0
      4c: 93 03 00 00  	li	t2, 0
      50: 13 03 03 01  	addi	t1, t1, 16
;     for (j = 0; j < lastDimLength; ++j) {
      54: 7b 40 07 01  	<unknown>
;       int32_t diff = max - pSrcA[i * lastDimLength + j];
      58: b3 0e c5 01  	add	t4, a0, t3
      5c: 83 8e 0e 00  	lb	t4, 0(t4)
;       shift[j] = (uint8_t)((diff + 16) >> 5);
      60: db 3e d3 0b  	<unknown>
      64: 33 0f c6 01  	add	t5, a2, t3
      68: 23 00 df 01  	sb	t4, 0(t5)
;       exp_sum += (256U >> shift[j]);
      6c: b3 de d2 01  	srl	t4, t0, t4
;     for (j = 0; j < lastDimLength; ++j) {
      70: 13 0e 1e 00  	addi	t3, t3, 1
;       exp_sum += (256U >> shift[j]);
      74: b3 83 7e 00  	add	t2, t4, t2
;     uint32_t exp_sum_inv = ((n_levels - 1) * 256U) / exp_sum;
      78: 33 d3 76 02  	divu	t1, a3, t2
      7c: 93 03 06 00  	mv	t2, a2
      80: 13 8e 05 00  	mv	t3, a1
      84: 93 0e 07 00  	mv	t4, a4
;     for (j = 0; j < lastDimLength; ++j) {
      88: 7b 40 a7 00  	<unknown>
;           (int8_t)((exp_sum_inv >> shift[j]) - (n_levels / 2));
      8c: 0b cf 13 00  	<unknown>
      90: 33 5f e3 01  	srl	t5, t1, t5
      94: 33 0f ff 40  	sub	t5, t5, a5
;     for (j = 0; j < lastDimLength; ++j) {
      98: 93 8e fe ff  	addi	t4, t4, -1
;       pDstB[i * lastDimLength + j] =
      9c: ab 00 ee 01  	<unknown>
;   for (i = 0; i < size / lastDimLength; ++i) {
      a0: 13 08 18 00  	addi	a6, a6, 1
      a4: 33 05 e5 00  	add	a0, a0, a4
      a8: b3 85 e5 00  	add	a1, a1, a4
      ac: e3 1e 18 f7  	bne	a6, a7, 0x28 <ITAMax_s8+0x28>
; }
      b0: 67 80 00 00  	ret

Disassembly of section .text.ITAPartialMax_s8:

00000000 <ITAPartialMax_s8>:
;                       uint32_t n_levels) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 81 00  	sw	s0, 28(sp)
       8: 23 2c 91 00  	sw	s1, 24(sp)
       c: 23 2a 21 01  	sw	s2, 20(sp)
      10: 23 28 31 01  	sw	s3, 16(sp)
      14: 23 26 41 01  	sw	s4, 12(sp)
;   for (i = 0; i < size / lastDimLength; ++i) {
      18: 63 62 d6 12  	bltu	a2, a3, 0x13c <ITAPartialMax_s8+0x13c>
;     for (g = 0; g < lastDimLength / group_width; ++g) {
      1c: 63 e0 e6 12  	bltu	a3, a4, 0x13c <ITAPartialMax_s8+0x13c>
      20: 13 08 00 00  	li	a6, 0
      24: b3 52 d6 02  	divu	t0, a2, a3
      28: 13 d6 17 00  	srli	a2, a5, 1
      2c: 93 17 86 00  	slli	a5, a2, 8
      30: 93 87 07 f0  	addi	a5, a5, -256
      34: b3 d8 e6 02  	divu	a7, a3, a4
      38: 13 03 10 00  	li	t1, 1
      3c: b3 f8 68 04  	<unknown>
      40: b3 f2 62 04  	<unknown>
      44: 13 03 00 01  	li	t1, 16
      48: 93 03 00 10  	li	t2, 256
      4c: fb c0 62 07  	<unknown>
      50: 93 0e 00 00  	li	t4, 0
      54: 13 0e 00 00  	li	t3, 0
      58: 93 0f 00 08  	li	t6, 128
      5c: 13 0f 05 00  	mv	t5, a0
      60: 6f 00 c0 01  	j	0x7c <ITAPartialMax_s8+0x7c>
      64: 93 04 00 00  	li	s1, 0
;       exp_partial_sum = (exp_partial_sum >> shift_sum) + exp_sum;
      68: 33 5e 8e 00  	srl	t3, t3, s0
      6c: 33 8e c4 01  	add	t3, s1, t3
;     for (g = 0; g < lastDimLength / group_width; ++g) {
      70: 93 8e 1e 00  	addi	t4, t4, 1
      74: 33 0f ef 00  	add	t5, t5, a4
      78: 63 82 1e 09  	beq	t4, a7, 0xfc <ITAPartialMax_s8+0xfc>
      7c: 13 04 00 08  	li	s0, 128
      80: 63 02 07 02  	beqz	a4, 0xa4 <ITAPartialMax_s8+0xa4>
      84: 13 04 00 08  	li	s0, 128
      88: 93 04 0f 00  	mv	s1, t5
      8c: 13 09 07 00  	mv	s2, a4
;       for (uint32_t k = 0; k < group_width; ++k) {
      90: 7b 40 87 00  	<unknown>
;         int8_t value = pSrcA[i * lastDimLength + g * group_width + k];
      94: 8b 89 14 00  	<unknown>
;         if (value > current_max) {
      98: 33 64 04 10  	<unknown>
;       for (uint32_t k = 0; k < group_width; ++k) {
      9c: 13 09 f9 ff  	addi	s2, s2, -1
;         if (value > current_max) {
      a0: 33 e4 89 04  	<unknown>
;       int32_t max_shift = (current_max - global_max + 16) >> 5;
      a4: b3 64 04 10  	<unknown>
      a8: b3 ef 0f 10  	<unknown>
;       int32_t shift_sum = (current_max > global_max) ? max_shift : 0;
      ac: 63 ca 9f 00  	blt	t6, s1, 0xc0 <ITAPartialMax_s8+0xc0>
      b0: 13 04 00 00  	li	s0, 0
;       global_max = (current_max > global_max) ? current_max : global_max;
      b4: b3 ef f4 05  	<unknown>
      b8: 63 1c 07 00  	bnez	a4, 0xd0 <ITAPartialMax_s8+0xd0>
      bc: 6f f0 9f fa  	j	0x64 <ITAPartialMax_s8+0x64>
      c0: 33 84 f4 41  	sub	s0, s1, t6
      c4: 5b 24 64 0a  	<unknown>
;       global_max = (current_max > global_max) ? current_max : global_max;
      c8: b3 ef f4 05  	<unknown>
;       for (uint32_t k = 0; k < group_width; ++k) {
      cc: e3 0c 07 f8  	beqz	a4, 0x64 <ITAPartialMax_s8+0x64>
      d0: 13 09 00 00  	li	s2, 0
      d4: 93 04 00 00  	li	s1, 0
      d8: 93 89 0f 01  	addi	s3, t6, 16
;       for (uint32_t k = 0; k < group_width; ++k) {
      dc: 7b 40 c7 00  	<unknown>
;             global_max - pSrcA[i * lastDimLength + g * group_width + k];
      e0: 33 0a 2f 01  	add	s4, t5, s2
      e4: 03 0a 0a 00  	lb	s4, 0(s4)
;         uint8_t shift = (uint8_t)((diff + 16) >> 5);
      e8: 5b ba 49 8b  	<unknown>
;         exp_sum += (256U >> shift);
      ec: 33 da 43 01  	srl	s4, t2, s4
;       for (uint32_t k = 0; k < group_width; ++k) {
      f0: 13 09 19 00  	addi	s2, s2, 1
;         exp_sum += (256U >> shift);
      f4: b3 04 9a 00  	add	s1, s4, s1
;       for (uint32_t k = 0; k < group_width; ++k) {
      f8: 6f f0 1f f7  	j	0x68 <ITAPartialMax_s8+0x68>
      fc: 63 8a 06 02  	beqz	a3, 0x130 <ITAPartialMax_s8+0x130>
     100: 93 0e 00 00  	li	t4, 0
     104: 33 de c7 03  	divu	t3, a5, t3
     108: 13 8f 0f 01  	addi	t5, t6, 16
;     for (j = 0; j < lastDimLength; ++j) {
     10c: 7b c0 06 01  	<unknown>
;       int32_t diff = global_max - pSrcA[i * lastDimLength + j];
     110: b3 0f d5 01  	add	t6, a0, t4
     114: 83 8f 0f 00  	lb	t6, 0(t6)
;       uint8_t shift = (uint8_t)((diff + 16) >> 5);
     118: db 3f ff 8b  	<unknown>
;           (int8_t)((exp_partial_sum_inverse >> shift) - (n_levels / 2));
     11c: b3 5f fe 01  	srl	t6, t3, t6
     120: b3 8f cf 40  	sub	t6, t6, a2
;       pDstB[i * lastDimLength + j] =
     124: 33 84 d5 01  	add	s0, a1, t4
;     for (j = 0; j < lastDimLength; ++j) {
     128: 93 8e 1e 00  	addi	t4, t4, 1
;       pDstB[i * lastDimLength + j] =
     12c: 23 00 f4 01  	sb	t6, 0(s0)
;   for (i = 0; i < size / lastDimLength; ++i) {
     130: 13 08 18 00  	addi	a6, a6, 1
     134: 33 05 d5 00  	add	a0, a0, a3
     138: b3 85 d5 00  	add	a1, a1, a3
; }
     13c: 03 24 c1 01  	lw	s0, 28(sp)
     140: 83 24 81 01  	lw	s1, 24(sp)
     144: 03 29 41 01  	lw	s2, 20(sp)
     148: 83 29 01 01  	lw	s3, 16(sp)
     14c: 03 2a c1 00  	lw	s4, 12(sp)
     150: 13 01 01 02  	addi	sp, sp, 32
     154: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Sqrt_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 0000011b 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text.Sqrt_fp32_fp32    00000020 00000000 TEXT
  4 .debug_loclists         00000036 00000000 DEBUG
  5 .debug_abbrev           00000078 00000000 DEBUG
  6 .debug_info             00000084 00000000 DEBUG
  7 .rela.debug_info        00000084 00000000 
  8 .debug_str_offsets      0000003c 00000000 DEBUG
  9 .rela.debug_str_offsets 0000009c 00000000 
 10 .debug_str              00000145 00000000 DEBUG
 11 .debug_addr             0000000c 00000000 DEBUG
 12 .rela.debug_addr        0000000c 00000000 
 13 .comment                00000073 00000000 
 14 .note.GNU-stack         00000000 00000000 
 15 .riscv.attributes       00000030 00000000 
 16 .debug_frame            00000024 00000000 DEBUG
 17 .rela.debug_frame       00000030 00000000 
 18 .debug_line             000000e8 00000000 DEBUG
 19 .rela.debug_line        0000012c 00000000 
 20 .debug_line_str         00000163 00000000 DEBUG
 21 .llvm_addrsig           00000000 00000000 
 22 .symtab                 00000240 00000000 

Disassembly of section .text.Sqrt_fp32_fp32:

00000000 <Sqrt_fp32_fp32>:
;   for (int i = 0; i < size; i++) {
       0: 63 5e c0 00  	blez	a2, 0x1c <Sqrt_fp32_fp32+0x1c>
       4: 7b 40 a6 00  	<unknown>
;     data_out[i] = sqrtf(data_in[i]);
       8: 07 20 05 00  	flw	ft0, 0(a0)
       c: 53 70 00 58  	fsqrt.s	ft0, ft0
      10: 27 a0 05 00  	fsw	ft0, 0(a1)
;   for (int i = 0; i < size; i++) {
      14: 93 85 45 00  	addi	a1, a1, 4
      18: 13 05 45 00  	addi	a0, a0, 4
; }
      1c: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(Util.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                            Size     VMA      Type
  0                                 00000000 00000000 
  1 .strtab                         000003a4 00000000 
  2 .text                           00000000 00000000 TEXT
  3 .text.PrintMatrix_s8_NCHW       000002c0 00000000 TEXT
  4 .rela.text.PrintMatrix_s8_NCHW  0000021c 00000000 
  5 .text.PrintMatrix_s8_NHWC       000002bc 00000000 TEXT
  6 .rela.text.PrintMatrix_s8_NHWC  0000021c 00000000 
  7 .text.PrintMatrix_s16_NCHW      000002d0 00000000 TEXT
  8 .rela.text.PrintMatrix_s16_NCHW 0000021c 00000000 
  9 .text.PrintMatrix_s16_NHWC      000002d4 00000000 TEXT
 10 .rela.text.PrintMatrix_s16_NHWC 0000021c 00000000 
 11 .text.PrintMatrix_s32_NCHW      000002cc 00000000 TEXT
 12 .rela.text.PrintMatrix_s32_NCHW 0000021c 00000000 
 13 .text.PrintMatrix_s32_NHWC      000002d0 00000000 TEXT
 14 .rela.text.PrintMatrix_s32_NHWC 0000021c 00000000 
 15 .text.PrintArray_s8             00000078 00000000 TEXT
 16 .rela.text.PrintArray_s8        00000048 00000000 
 17 .text.PrintArray_s16            00000078 00000000 TEXT
 18 .rela.text.PrintArray_s16       00000048 00000000 
 19 .text.PrintArray_s32            00000074 00000000 TEXT
 20 .rela.text.PrintArray_s32       00000048 00000000 
 21 .text.PrintMatrix_u8_NCHW       000002c0 00000000 TEXT
 22 .rela.text.PrintMatrix_u8_NCHW  0000021c 00000000 
 23 .text.PrintMatrix_u8_NHWC       000002bc 00000000 TEXT
 24 .rela.text.PrintMatrix_u8_NHWC  0000021c 00000000 
 25 .text.PrintMatrix_u16_NCHW      000002d0 00000000 TEXT
 26 .rela.text.PrintMatrix_u16_NCHW 0000021c 00000000 
 27 .text.PrintMatrix_u16_NHWC      000002d4 00000000 TEXT
 28 .rela.text.PrintMatrix_u16_NHWC 0000021c 00000000 
 29 .text.PrintMatrix_u32_NCHW      000002cc 00000000 TEXT
 30 .rela.text.PrintMatrix_u32_NCHW 0000021c 00000000 
 31 .text.PrintMatrix_u32_NHWC      000002d0 00000000 TEXT
 32 .rela.text.PrintMatrix_u32_NHWC 0000021c 00000000 
 33 .text.PrintArray_u8             00000078 00000000 TEXT
 34 .rela.text.PrintArray_u8        00000048 00000000 
 35 .text.PrintArray_u16            00000078 00000000 TEXT
 36 .rela.text.PrintArray_u16       00000048 00000000 
 37 .text.PrintArray_u32            00000074 00000000 TEXT
 38 .rela.text.PrintArray_u32       00000048 00000000 
 39 .rodata.str1.1                  0000003a 00000000 DATA
 40 .debug_loclists                 0000161e 00000000 DEBUG
 41 .debug_abbrev                   000000f9 00000000 DEBUG
 42 .debug_info                     00000ddf 00000000 DEBUG
 43 .rela.debug_info                000003a8 00000000 
 44 .debug_rnglists                 0000033a 00000000 DEBUG
 45 .debug_str_offsets              000000d4 00000000 DEBUG
 46 .rela.debug_str_offsets         00000264 00000000 
 47 .debug_str                      0000031f 00000000 DEBUG
 48 .debug_addr                     000003f8 00000000 DEBUG
 49 .rela.debug_addr                00000bd0 00000000 
 50 .comment                        00000073 00000000 
 51 .note.GNU-stack                 00000000 00000000 
 52 .riscv.attributes               00000030 00000000 
 53 .debug_frame                    00000314 00000000 DEBUG
 54 .rela.debug_frame               000006c0 00000000 
 55 .debug_line                     00001bf5 00000000 DEBUG
 56 .rela.debug_line                00004b0c 00000000 
 57 .debug_line_str                 00000135 00000000 DEBUG
 58 .llvm_addrsig                   00000000 00000000 
 59 .symtab                         00004d70 00000000 

Disassembly of section .text.PrintMatrix_s8_NCHW:

00000000 <PrintMatrix_s8_NCHW>:
;                          uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 24 c1 02  	sw	a2, 40(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 84 05 24  	beqz	a1, 0x284 <PrintMatrix_s8_NCHW+0x284>
      40: 93 8d 05 00  	mv	s11, a1
      44: 13 0b 05 00  	mv	s6, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 81 02  	lw	a0, 40(sp)
      4c: 63 0e 05 10  	beqz	a0, 0x168 <PrintMatrix_s8_NCHW+0x168>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 84 06 14  	beqz	a3, 0x19c <PrintMatrix_s8_NCHW+0x19c>
      58: 93 0b 07 00  	mv	s7, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 02 07 1a  	beqz	a4, 0x200 <PrintMatrix_s8_NCHW+0x200>
      60: 93 89 07 00  	mv	s3, a5
      64: 93 05 00 00  	li	a1, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 86 2b 03  	mul	a2, s7, s2
      6c: 03 25 81 02  	lw	a0, 40(sp)
      70: 23 22 c1 02  	sw	a2, 36(sp)
      74: 33 05 a6 02  	mul	a0, a2, a0
      78: 23 28 a1 00  	sw	a0, 16(sp)
      7c: 37 05 00 00  	lui	a0, 0
      80: 13 05 05 00  	mv	a0, a0
      84: 23 26 a1 00  	sw	a0, 12(sp)
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 20 a1 02  	sw	a0, 32(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 0c 05 00  	mv	s8, a0
      9c: 37 05 00 00  	lui	a0, 0
      a0: 93 0c 05 00  	mv	s9, a0
      a4: 37 05 00 00  	lui	a0, 0
      a8: 13 0d 05 00  	mv	s10, a0
      ac: 23 2a b1 01  	sw	s11, 20(sp)
      b0: 23 2c b1 00  	sw	a1, 24(sp)
;       deeploy_log("[\r\n");
      b4: 03 25 c1 00  	lw	a0, 12(sp)
      b8: 97 00 00 00  	auipc	ra, 0
      bc: e7 80 00 00  	jalr	ra
      c0: 93 0a 00 00  	li	s5, 0
      c4: 23 2e 61 01  	sw	s6, 28(sp)
;         deeploy_log("  [\r\n  ");
      c8: 03 25 01 02  	lw	a0, 32(sp)
      cc: 97 00 00 00  	auipc	ra, 0
      d0: e7 80 00 00  	jalr	ra
      d4: 93 0d 00 00  	li	s11, 0
      d8: 93 04 0b 00  	mv	s1, s6
      dc: 13 84 04 00  	mv	s0, s1
      e0: 13 8a 0b 00  	mv	s4, s7
;               (int8_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] + offset));
      e4: 0b 45 14 00  	<unknown>
      e8: 33 05 35 01  	add	a0, a0, s3
      ec: b3 65 05 10  	<unknown>
;           deeploy_log(
      f0: 13 05 0c 00  	mv	a0, s8
      f4: 97 00 00 00  	auipc	ra, 0
      f8: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
      fc: 13 0a fa ff  	addi	s4, s4, -1
     100: e3 12 0a fe  	bnez	s4, 0xe4 <PrintMatrix_s8_NCHW+0xe4>
;           deeploy_log("\r\n  ");
     104: 13 85 0c 00  	mv	a0, s9
     108: 97 00 00 00  	auipc	ra, 0
     10c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     110: 93 8d 1d 00  	addi	s11, s11, 1
     114: b3 84 74 01  	add	s1, s1, s7
     118: e3 92 2d fd  	bne	s11, s2, 0xdc <PrintMatrix_s8_NCHW+0xdc>
;         deeploy_log("]\r\n");
     11c: 13 05 0d 00  	mv	a0, s10
     120: 97 00 00 00  	auipc	ra, 0
     124: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     128: 93 8a 1a 00  	addi	s5, s5, 1
     12c: 03 25 41 02  	lw	a0, 36(sp)
     130: 33 0b ab 00  	add	s6, s6, a0
     134: 03 25 81 02  	lw	a0, 40(sp)
     138: e3 98 aa f8  	bne	s5, a0, 0xc8 <PrintMatrix_s8_NCHW+0xc8>
;       deeploy_log("]\r\n");
     13c: 13 05 0d 00  	mv	a0, s10
     140: 97 00 00 00  	auipc	ra, 0
     144: e7 80 00 00  	jalr	ra
     148: 83 25 81 01  	lw	a1, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     14c: 93 85 15 00  	addi	a1, a1, 1
     150: 03 2b c1 01  	lw	s6, 28(sp)
;   for (uint32_t n = 0; n < N; n++) {
     154: 03 25 01 01  	lw	a0, 16(sp)
     158: 33 0b ab 00  	add	s6, s6, a0
     15c: 83 2d 41 01  	lw	s11, 20(sp)
     160: e3 98 b5 f5  	bne	a1, s11, 0xb0 <PrintMatrix_s8_NCHW+0xb0>
     164: 6f 00 00 12  	j	0x284 <PrintMatrix_s8_NCHW+0x284>
     168: 37 05 00 00  	lui	a0, 0
     16c: 93 04 05 00  	mv	s1, a0
     170: 37 05 00 00  	lui	a0, 0
     174: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     178: 13 85 04 00  	mv	a0, s1
     17c: 97 00 00 00  	auipc	ra, 0
     180: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     184: 13 05 09 00  	mv	a0, s2
     188: 97 00 00 00  	auipc	ra, 0
     18c: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     190: 93 8d fd ff  	addi	s11, s11, -1
     194: e3 92 0d fe  	bnez	s11, 0x178 <PrintMatrix_s8_NCHW+0x178>
     198: 6f 00 c0 0e  	j	0x284 <PrintMatrix_s8_NCHW+0x284>
     19c: 13 04 00 00  	li	s0, 0
     1a0: 37 05 00 00  	lui	a0, 0
     1a4: 13 09 05 00  	mv	s2, a0
     1a8: 37 05 00 00  	lui	a0, 0
     1ac: 93 09 05 00  	mv	s3, a0
     1b0: 37 05 00 00  	lui	a0, 0
     1b4: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1b8: 13 05 09 00  	mv	a0, s2
     1bc: 97 00 00 00  	auipc	ra, 0
     1c0: e7 80 00 00  	jalr	ra
     1c4: 83 24 81 02  	lw	s1, 40(sp)
;         deeploy_log("  [\r\n  ");
     1c8: 13 85 09 00  	mv	a0, s3
     1cc: 97 00 00 00  	auipc	ra, 0
     1d0: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1d4: 13 05 0a 00  	mv	a0, s4
     1d8: 97 00 00 00  	auipc	ra, 0
     1dc: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1e0: 93 84 f4 ff  	addi	s1, s1, -1
     1e4: e3 92 04 fe  	bnez	s1, 0x1c8 <PrintMatrix_s8_NCHW+0x1c8>
;       deeploy_log("]\r\n");
     1e8: 13 05 0a 00  	mv	a0, s4
     1ec: 97 00 00 00  	auipc	ra, 0
     1f0: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1f4: 13 04 14 00  	addi	s0, s0, 1
     1f8: e3 10 b4 fd  	bne	s0, s11, 0x1b8 <PrintMatrix_s8_NCHW+0x1b8>
     1fc: 6f 00 80 08  	j	0x284 <PrintMatrix_s8_NCHW+0x284>
     200: 13 04 00 00  	li	s0, 0
     204: 37 05 00 00  	lui	a0, 0
     208: 93 09 05 00  	mv	s3, a0
     20c: 37 05 00 00  	lui	a0, 0
     210: 13 0a 05 00  	mv	s4, a0
     214: 37 05 00 00  	lui	a0, 0
     218: 93 0a 05 00  	mv	s5, a0
     21c: 37 05 00 00  	lui	a0, 0
     220: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     224: 13 85 09 00  	mv	a0, s3
     228: 97 00 00 00  	auipc	ra, 0
     22c: e7 80 00 00  	jalr	ra
     230: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     234: 13 05 0a 00  	mv	a0, s4
     238: 97 00 00 00  	auipc	ra, 0
     23c: e7 80 00 00  	jalr	ra
     240: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     244: 13 85 0a 00  	mv	a0, s5
     248: 97 00 00 00  	auipc	ra, 0
     24c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     250: 93 8b fb ff  	addi	s7, s7, -1
     254: e3 98 0b fe  	bnez	s7, 0x244 <PrintMatrix_s8_NCHW+0x244>
;         deeploy_log("]\r\n");
     258: 13 05 0b 00  	mv	a0, s6
     25c: 97 00 00 00  	auipc	ra, 0
     260: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     264: 93 84 14 00  	addi	s1, s1, 1
     268: 03 25 81 02  	lw	a0, 40(sp)
     26c: e3 94 a4 fc  	bne	s1, a0, 0x234 <PrintMatrix_s8_NCHW+0x234>
;       deeploy_log("]\r\n");
     270: 13 05 0b 00  	mv	a0, s6
     274: 97 00 00 00  	auipc	ra, 0
     278: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     27c: 13 04 14 00  	addi	s0, s0, 1
     280: e3 12 b4 fb  	bne	s0, s11, 0x224 <PrintMatrix_s8_NCHW+0x224>
; }
     284: 83 20 c1 05  	lw	ra, 92(sp)
     288: 03 24 81 05  	lw	s0, 88(sp)
     28c: 83 24 41 05  	lw	s1, 84(sp)
     290: 03 29 01 05  	lw	s2, 80(sp)
     294: 83 29 c1 04  	lw	s3, 76(sp)
     298: 03 2a 81 04  	lw	s4, 72(sp)
     29c: 83 2a 41 04  	lw	s5, 68(sp)
     2a0: 03 2b 01 04  	lw	s6, 64(sp)
     2a4: 83 2b c1 03  	lw	s7, 60(sp)
     2a8: 03 2c 81 03  	lw	s8, 56(sp)
     2ac: 83 2c 41 03  	lw	s9, 52(sp)
     2b0: 03 2d 01 03  	lw	s10, 48(sp)
     2b4: 83 2d c1 02  	lw	s11, 44(sp)
     2b8: 13 01 01 06  	addi	sp, sp, 96
     2bc: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_s8_NHWC:

00000000 <PrintMatrix_s8_NHWC>:
;                          uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
;   for (uint32_t n = 0; n < N; n++) {
      38: 63 84 05 24  	beqz	a1, 0x280 <PrintMatrix_s8_NHWC+0x280>
      3c: 93 04 06 00  	mv	s1, a2
      40: 13 8d 05 00  	mv	s10, a1
;     for (uint32_t c = 0; c < C; c++) {
      44: 63 02 06 12  	beqz	a2, 0x168 <PrintMatrix_s8_NHWC+0x168>
      48: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      4c: 63 88 06 14  	beqz	a3, 0x19c <PrintMatrix_s8_NHWC+0x19c>
      50: 93 0a 07 00  	mv	s5, a4
;         for (uint32_t w = 0; w < W; w++) {
      54: 63 06 07 1a  	beqz	a4, 0x200 <PrintMatrix_s8_NHWC+0x200>
      58: 93 89 07 00  	mv	s3, a5
      5c: 13 04 05 00  	mv	s0, a0
      60: 93 05 00 00  	li	a1, 0
;   for (uint32_t n = 0; n < N; n++) {
      64: 33 85 2a 03  	mul	a0, s5, s2
      68: 33 05 95 02  	mul	a0, a0, s1
      6c: 23 28 a1 00  	sw	a0, 16(sp)
      70: 33 8b 9a 02  	mul	s6, s5, s1
      74: 37 05 00 00  	lui	a0, 0
      78: 13 05 05 00  	mv	a0, a0
      7c: 23 26 a1 00  	sw	a0, 12(sp)
      80: 37 05 00 00  	lui	a0, 0
      84: 13 05 05 00  	mv	a0, a0
      88: 23 20 a1 02  	sw	a0, 32(sp)
      8c: 37 05 00 00  	lui	a0, 0
      90: 13 0c 05 00  	mv	s8, a0
      94: 37 05 00 00  	lui	a0, 0
      98: 93 0c 05 00  	mv	s9, a0
      9c: 37 05 00 00  	lui	a0, 0
      a0: 13 05 05 00  	mv	a0, a0
      a4: 23 22 a1 02  	sw	a0, 36(sp)
      a8: 23 2a a1 01  	sw	s10, 20(sp)
      ac: 23 2c b1 00  	sw	a1, 24(sp)
;       deeploy_log("[\r\n");
      b0: 03 25 c1 00  	lw	a0, 12(sp)
      b4: 97 00 00 00  	auipc	ra, 0
      b8: e7 80 00 00  	jalr	ra
      bc: 13 05 00 00  	li	a0, 0
      c0: 23 2e 81 00  	sw	s0, 28(sp)
      c4: 13 0d 04 00  	mv	s10, s0
      c8: 23 24 a1 02  	sw	a0, 40(sp)
;         deeploy_log("  [\r\n  ");
      cc: 03 25 01 02  	lw	a0, 32(sp)
      d0: 97 00 00 00  	auipc	ra, 0
      d4: e7 80 00 00  	jalr	ra
      d8: 93 0d 00 00  	li	s11, 0
      dc: 93 0b 0d 00  	mv	s7, s10
      e0: 13 84 0b 00  	mv	s0, s7
      e4: 13 8a 0a 00  	mv	s4, s5
;               (int8_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] + offset));
      e8: 0b 75 94 40  	<unknown>
      ec: 33 05 35 01  	add	a0, a0, s3
      f0: b3 65 05 10  	<unknown>
;           deeploy_log(
      f4: 13 05 0c 00  	mv	a0, s8
      f8: 97 00 00 00  	auipc	ra, 0
      fc: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     100: 13 0a fa ff  	addi	s4, s4, -1
     104: e3 12 0a fe  	bnez	s4, 0xe8 <PrintMatrix_s8_NHWC+0xe8>
;           deeploy_log("\r\n  ");
     108: 13 85 0c 00  	mv	a0, s9
     10c: 97 00 00 00  	auipc	ra, 0
     110: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     114: 93 8d 1d 00  	addi	s11, s11, 1
     118: b3 8b 6b 01  	add	s7, s7, s6
     11c: e3 92 2d fd  	bne	s11, s2, 0xe0 <PrintMatrix_s8_NHWC+0xe0>
;         deeploy_log("]\r\n");
     120: 03 25 41 02  	lw	a0, 36(sp)
     124: 97 00 00 00  	auipc	ra, 0
     128: e7 80 00 00  	jalr	ra
     12c: 03 25 81 02  	lw	a0, 40(sp)
;     for (uint32_t c = 0; c < C; c++) {
     130: 13 05 15 00  	addi	a0, a0, 1
     134: 13 0d 1d 00  	addi	s10, s10, 1
     138: e3 18 95 f8  	bne	a0, s1, 0xc8 <PrintMatrix_s8_NHWC+0xc8>
;       deeploy_log("]\r\n");
     13c: 03 25 41 02  	lw	a0, 36(sp)
     140: 97 00 00 00  	auipc	ra, 0
     144: e7 80 00 00  	jalr	ra
     148: 83 25 81 01  	lw	a1, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     14c: 93 85 15 00  	addi	a1, a1, 1
     150: 03 24 c1 01  	lw	s0, 28(sp)
;   for (uint32_t n = 0; n < N; n++) {
     154: 03 25 01 01  	lw	a0, 16(sp)
     158: 33 04 a4 00  	add	s0, s0, a0
     15c: 03 2d 41 01  	lw	s10, 20(sp)
     160: e3 96 a5 f5  	bne	a1, s10, 0xac <PrintMatrix_s8_NHWC+0xac>
     164: 6f 00 c0 11  	j	0x280 <PrintMatrix_s8_NHWC+0x280>
     168: 37 05 00 00  	lui	a0, 0
     16c: 93 04 05 00  	mv	s1, a0
     170: 37 05 00 00  	lui	a0, 0
     174: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     178: 13 85 04 00  	mv	a0, s1
     17c: 97 00 00 00  	auipc	ra, 0
     180: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     184: 13 05 09 00  	mv	a0, s2
     188: 97 00 00 00  	auipc	ra, 0
     18c: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     190: 13 0d fd ff  	addi	s10, s10, -1
     194: e3 12 0d fe  	bnez	s10, 0x178 <PrintMatrix_s8_NHWC+0x178>
     198: 6f 00 80 0e  	j	0x280 <PrintMatrix_s8_NHWC+0x280>
     19c: 13 04 00 00  	li	s0, 0
     1a0: 37 05 00 00  	lui	a0, 0
     1a4: 13 09 05 00  	mv	s2, a0
     1a8: 37 05 00 00  	lui	a0, 0
     1ac: 93 09 05 00  	mv	s3, a0
     1b0: 37 05 00 00  	lui	a0, 0
     1b4: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1b8: 13 05 09 00  	mv	a0, s2
     1bc: 97 00 00 00  	auipc	ra, 0
     1c0: e7 80 00 00  	jalr	ra
     1c4: 93 8a 04 00  	mv	s5, s1
;         deeploy_log("  [\r\n  ");
     1c8: 13 85 09 00  	mv	a0, s3
     1cc: 97 00 00 00  	auipc	ra, 0
     1d0: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1d4: 13 05 0a 00  	mv	a0, s4
     1d8: 97 00 00 00  	auipc	ra, 0
     1dc: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1e0: 93 8a fa ff  	addi	s5, s5, -1
     1e4: e3 92 0a fe  	bnez	s5, 0x1c8 <PrintMatrix_s8_NHWC+0x1c8>
;       deeploy_log("]\r\n");
     1e8: 13 05 0a 00  	mv	a0, s4
     1ec: 97 00 00 00  	auipc	ra, 0
     1f0: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1f4: 13 04 14 00  	addi	s0, s0, 1
     1f8: e3 10 a4 fd  	bne	s0, s10, 0x1b8 <PrintMatrix_s8_NHWC+0x1b8>
     1fc: 6f 00 40 08  	j	0x280 <PrintMatrix_s8_NHWC+0x280>
     200: 13 04 00 00  	li	s0, 0
     204: 37 05 00 00  	lui	a0, 0
     208: 93 09 05 00  	mv	s3, a0
     20c: 37 05 00 00  	lui	a0, 0
     210: 13 0a 05 00  	mv	s4, a0
     214: 37 05 00 00  	lui	a0, 0
     218: 93 0a 05 00  	mv	s5, a0
     21c: 37 05 00 00  	lui	a0, 0
     220: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     224: 13 85 09 00  	mv	a0, s3
     228: 97 00 00 00  	auipc	ra, 0
     22c: e7 80 00 00  	jalr	ra
     230: 93 0b 00 00  	li	s7, 0
;         deeploy_log("  [\r\n  ");
     234: 13 05 0a 00  	mv	a0, s4
     238: 97 00 00 00  	auipc	ra, 0
     23c: e7 80 00 00  	jalr	ra
     240: 13 0c 09 00  	mv	s8, s2
;           deeploy_log("\r\n  ");
     244: 13 85 0a 00  	mv	a0, s5
     248: 97 00 00 00  	auipc	ra, 0
     24c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     250: 13 0c fc ff  	addi	s8, s8, -1
     254: e3 18 0c fe  	bnez	s8, 0x244 <PrintMatrix_s8_NHWC+0x244>
;         deeploy_log("]\r\n");
     258: 13 05 0b 00  	mv	a0, s6
     25c: 97 00 00 00  	auipc	ra, 0
     260: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     264: 93 8b 1b 00  	addi	s7, s7, 1
     268: e3 96 9b fc  	bne	s7, s1, 0x234 <PrintMatrix_s8_NHWC+0x234>
;       deeploy_log("]\r\n");
     26c: 13 05 0b 00  	mv	a0, s6
     270: 97 00 00 00  	auipc	ra, 0
     274: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     278: 13 04 14 00  	addi	s0, s0, 1
     27c: e3 14 a4 fb  	bne	s0, s10, 0x224 <PrintMatrix_s8_NHWC+0x224>
; }
     280: 83 20 c1 05  	lw	ra, 92(sp)
     284: 03 24 81 05  	lw	s0, 88(sp)
     288: 83 24 41 05  	lw	s1, 84(sp)
     28c: 03 29 01 05  	lw	s2, 80(sp)
     290: 83 29 c1 04  	lw	s3, 76(sp)
     294: 03 2a 81 04  	lw	s4, 72(sp)
     298: 83 2a 41 04  	lw	s5, 68(sp)
     29c: 03 2b 01 04  	lw	s6, 64(sp)
     2a0: 83 2b c1 03  	lw	s7, 60(sp)
     2a4: 03 2c 81 03  	lw	s8, 56(sp)
     2a8: 83 2c 41 03  	lw	s9, 52(sp)
     2ac: 03 2d 01 03  	lw	s10, 48(sp)
     2b0: 83 2d c1 02  	lw	s11, 44(sp)
     2b4: 13 01 01 06  	addi	sp, sp, 96
     2b8: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_s16_NCHW:

00000000 <PrintMatrix_s16_NCHW>:
;                           uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 24 c1 02  	sw	a2, 40(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 8c 05 24  	beqz	a1, 0x294 <PrintMatrix_s16_NCHW+0x294>
      40: 93 8d 05 00  	mv	s11, a1
      44: 93 0b 05 00  	mv	s7, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 81 02  	lw	a0, 40(sp)
      4c: 63 06 05 12  	beqz	a0, 0x178 <PrintMatrix_s16_NCHW+0x178>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 8c 06 14  	beqz	a3, 0x1ac <PrintMatrix_s16_NCHW+0x1ac>
      58: 13 0d 07 00  	mv	s10, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 0a 07 1a  	beqz	a4, 0x210 <PrintMatrix_s16_NCHW+0x210>
      60: 93 89 07 00  	mv	s3, a5
      64: 13 06 00 00  	li	a2, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 05 2d 03  	mul	a0, s10, s2
      6c: 83 25 81 02  	lw	a1, 40(sp)
      70: b3 05 b5 02  	mul	a1, a0, a1
      74: 93 95 15 00  	slli	a1, a1, 1
      78: 23 26 b1 00  	sw	a1, 12(sp)
      7c: 13 15 15 00  	slli	a0, a0, 1
      80: 23 20 a1 02  	sw	a0, 32(sp)
      84: 93 1a 1d 00  	slli	s5, s10, 1
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 24 a1 00  	sw	a0, 8(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 05 05 00  	mv	a0, a0
      9c: 23 2e a1 00  	sw	a0, 28(sp)
      a0: 37 05 00 00  	lui	a0, 0
      a4: 13 0c 05 00  	mv	s8, a0
      a8: 37 05 00 00  	lui	a0, 0
      ac: 93 0c 05 00  	mv	s9, a0
      b0: 37 05 00 00  	lui	a0, 0
      b4: 13 05 05 00  	mv	a0, a0
      b8: 23 22 a1 02  	sw	a0, 36(sp)
      bc: 23 28 b1 01  	sw	s11, 16(sp)
      c0: 23 2a c1 00  	sw	a2, 20(sp)
;       deeploy_log("[\r\n");
      c4: 03 25 81 00  	lw	a0, 8(sp)
      c8: 97 00 00 00  	auipc	ra, 0
      cc: e7 80 00 00  	jalr	ra
      d0: 13 04 00 00  	li	s0, 0
      d4: 23 2c 71 01  	sw	s7, 24(sp)
;         deeploy_log("  [\r\n  ");
      d8: 03 25 c1 01  	lw	a0, 28(sp)
      dc: 97 00 00 00  	auipc	ra, 0
      e0: e7 80 00 00  	jalr	ra
      e4: 93 04 00 00  	li	s1, 0
      e8: 13 8b 0b 00  	mv	s6, s7
      ec: 93 0d 0b 00  	mv	s11, s6
      f0: 13 0a 0d 00  	mv	s4, s10
;               (int16_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] + offset));
      f4: 0b d5 2d 00  	<unknown>
      f8: 33 05 35 01  	add	a0, a0, s3
      fc: b3 45 05 10  	<unknown>
;           deeploy_log(
     100: 13 05 0c 00  	mv	a0, s8
     104: 97 00 00 00  	auipc	ra, 0
     108: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     10c: 13 0a fa ff  	addi	s4, s4, -1
     110: e3 12 0a fe  	bnez	s4, 0xf4 <PrintMatrix_s16_NCHW+0xf4>
;           deeploy_log("\r\n  ");
     114: 13 85 0c 00  	mv	a0, s9
     118: 97 00 00 00  	auipc	ra, 0
     11c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     120: 93 84 14 00  	addi	s1, s1, 1
     124: 33 0b 5b 01  	add	s6, s6, s5
     128: e3 92 24 fd  	bne	s1, s2, 0xec <PrintMatrix_s16_NCHW+0xec>
;         deeploy_log("]\r\n");
     12c: 03 25 41 02  	lw	a0, 36(sp)
     130: 97 00 00 00  	auipc	ra, 0
     134: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     138: 13 04 14 00  	addi	s0, s0, 1
     13c: 03 25 01 02  	lw	a0, 32(sp)
     140: b3 8b ab 00  	add	s7, s7, a0
     144: 03 25 81 02  	lw	a0, 40(sp)
     148: e3 18 a4 f8  	bne	s0, a0, 0xd8 <PrintMatrix_s16_NCHW+0xd8>
;       deeploy_log("]\r\n");
     14c: 03 25 41 02  	lw	a0, 36(sp)
     150: 97 00 00 00  	auipc	ra, 0
     154: e7 80 00 00  	jalr	ra
     158: 03 26 41 01  	lw	a2, 20(sp)
;   for (uint32_t n = 0; n < N; n++) {
     15c: 13 06 16 00  	addi	a2, a2, 1
     160: 83 2b 81 01  	lw	s7, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     164: 03 25 c1 00  	lw	a0, 12(sp)
     168: b3 8b ab 00  	add	s7, s7, a0
     16c: 83 2d 01 01  	lw	s11, 16(sp)
     170: e3 18 b6 f5  	bne	a2, s11, 0xc0 <PrintMatrix_s16_NCHW+0xc0>
     174: 6f 00 00 12  	j	0x294 <PrintMatrix_s16_NCHW+0x294>
     178: 37 05 00 00  	lui	a0, 0
     17c: 93 04 05 00  	mv	s1, a0
     180: 37 05 00 00  	lui	a0, 0
     184: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     188: 13 85 04 00  	mv	a0, s1
     18c: 97 00 00 00  	auipc	ra, 0
     190: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     194: 13 05 09 00  	mv	a0, s2
     198: 97 00 00 00  	auipc	ra, 0
     19c: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1a0: 93 8d fd ff  	addi	s11, s11, -1
     1a4: e3 92 0d fe  	bnez	s11, 0x188 <PrintMatrix_s16_NCHW+0x188>
     1a8: 6f 00 c0 0e  	j	0x294 <PrintMatrix_s16_NCHW+0x294>
     1ac: 13 04 00 00  	li	s0, 0
     1b0: 37 05 00 00  	lui	a0, 0
     1b4: 13 09 05 00  	mv	s2, a0
     1b8: 37 05 00 00  	lui	a0, 0
     1bc: 93 09 05 00  	mv	s3, a0
     1c0: 37 05 00 00  	lui	a0, 0
     1c4: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1c8: 13 05 09 00  	mv	a0, s2
     1cc: 97 00 00 00  	auipc	ra, 0
     1d0: e7 80 00 00  	jalr	ra
     1d4: 83 24 81 02  	lw	s1, 40(sp)
;         deeploy_log("  [\r\n  ");
     1d8: 13 85 09 00  	mv	a0, s3
     1dc: 97 00 00 00  	auipc	ra, 0
     1e0: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1e4: 13 05 0a 00  	mv	a0, s4
     1e8: 97 00 00 00  	auipc	ra, 0
     1ec: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1f0: 93 84 f4 ff  	addi	s1, s1, -1
     1f4: e3 92 04 fe  	bnez	s1, 0x1d8 <PrintMatrix_s16_NCHW+0x1d8>
;       deeploy_log("]\r\n");
     1f8: 13 05 0a 00  	mv	a0, s4
     1fc: 97 00 00 00  	auipc	ra, 0
     200: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     204: 13 04 14 00  	addi	s0, s0, 1
     208: e3 10 b4 fd  	bne	s0, s11, 0x1c8 <PrintMatrix_s16_NCHW+0x1c8>
     20c: 6f 00 80 08  	j	0x294 <PrintMatrix_s16_NCHW+0x294>
     210: 13 04 00 00  	li	s0, 0
     214: 37 05 00 00  	lui	a0, 0
     218: 93 09 05 00  	mv	s3, a0
     21c: 37 05 00 00  	lui	a0, 0
     220: 13 0a 05 00  	mv	s4, a0
     224: 37 05 00 00  	lui	a0, 0
     228: 93 0a 05 00  	mv	s5, a0
     22c: 37 05 00 00  	lui	a0, 0
     230: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     234: 13 85 09 00  	mv	a0, s3
     238: 97 00 00 00  	auipc	ra, 0
     23c: e7 80 00 00  	jalr	ra
     240: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     244: 13 05 0a 00  	mv	a0, s4
     248: 97 00 00 00  	auipc	ra, 0
     24c: e7 80 00 00  	jalr	ra
     250: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     254: 13 85 0a 00  	mv	a0, s5
     258: 97 00 00 00  	auipc	ra, 0
     25c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     260: 93 8b fb ff  	addi	s7, s7, -1
     264: e3 98 0b fe  	bnez	s7, 0x254 <PrintMatrix_s16_NCHW+0x254>
;         deeploy_log("]\r\n");
     268: 13 05 0b 00  	mv	a0, s6
     26c: 97 00 00 00  	auipc	ra, 0
     270: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     274: 93 84 14 00  	addi	s1, s1, 1
     278: 03 25 81 02  	lw	a0, 40(sp)
     27c: e3 94 a4 fc  	bne	s1, a0, 0x244 <PrintMatrix_s16_NCHW+0x244>
;       deeploy_log("]\r\n");
     280: 13 05 0b 00  	mv	a0, s6
     284: 97 00 00 00  	auipc	ra, 0
     288: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     28c: 13 04 14 00  	addi	s0, s0, 1
     290: e3 12 b4 fb  	bne	s0, s11, 0x234 <PrintMatrix_s16_NCHW+0x234>
; }
     294: 83 20 c1 05  	lw	ra, 92(sp)
     298: 03 24 81 05  	lw	s0, 88(sp)
     29c: 83 24 41 05  	lw	s1, 84(sp)
     2a0: 03 29 01 05  	lw	s2, 80(sp)
     2a4: 83 29 c1 04  	lw	s3, 76(sp)
     2a8: 03 2a 81 04  	lw	s4, 72(sp)
     2ac: 83 2a 41 04  	lw	s5, 68(sp)
     2b0: 03 2b 01 04  	lw	s6, 64(sp)
     2b4: 83 2b c1 03  	lw	s7, 60(sp)
     2b8: 03 2c 81 03  	lw	s8, 56(sp)
     2bc: 83 2c 41 03  	lw	s9, 52(sp)
     2c0: 03 2d 01 03  	lw	s10, 48(sp)
     2c4: 83 2d c1 02  	lw	s11, 44(sp)
     2c8: 13 01 01 06  	addi	sp, sp, 96
     2cc: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_s16_NHWC:

00000000 <PrintMatrix_s16_NHWC>:
;                           uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 22 c1 02  	sw	a2, 36(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 8e 05 24  	beqz	a1, 0x298 <PrintMatrix_s16_NHWC+0x298>
      40: 13 8d 05 00  	mv	s10, a1
      44: 93 04 05 00  	mv	s1, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 41 02  	lw	a0, 36(sp)
      4c: 63 0a 05 12  	beqz	a0, 0x180 <PrintMatrix_s16_NHWC+0x180>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 80 06 16  	beqz	a3, 0x1b4 <PrintMatrix_s16_NHWC+0x1b4>
      58: 13 04 07 00  	mv	s0, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 0e 07 1a  	beqz	a4, 0x218 <PrintMatrix_s16_NHWC+0x218>
      60: 93 89 07 00  	mv	s3, a5
      64: 13 06 00 00  	li	a2, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 05 24 03  	mul	a0, s0, s2
      6c: 83 25 41 02  	lw	a1, 36(sp)
      70: 33 05 b5 02  	mul	a0, a0, a1
      74: 13 15 15 00  	slli	a0, a0, 1
      78: 23 26 a1 00  	sw	a0, 12(sp)
      7c: 33 05 b4 02  	mul	a0, s0, a1
      80: 13 1b 15 00  	slli	s6, a0, 1
      84: 93 9a 15 00  	slli	s5, a1, 1
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 24 a1 00  	sw	a0, 8(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 05 05 00  	mv	a0, a0
      9c: 23 2e a1 00  	sw	a0, 28(sp)
      a0: 37 05 00 00  	lui	a0, 0
      a4: 13 0c 05 00  	mv	s8, a0
      a8: 37 05 00 00  	lui	a0, 0
      ac: 93 0c 05 00  	mv	s9, a0
      b0: 37 05 00 00  	lui	a0, 0
      b4: 13 05 05 00  	mv	a0, a0
      b8: 23 20 a1 02  	sw	a0, 32(sp)
      bc: 23 28 a1 01  	sw	s10, 16(sp)
      c0: 23 2a c1 00  	sw	a2, 20(sp)
;       deeploy_log("[\r\n");
      c4: 03 25 81 00  	lw	a0, 8(sp)
      c8: 97 00 00 00  	auipc	ra, 0
      cc: e7 80 00 00  	jalr	ra
      d0: 93 05 00 00  	li	a1, 0
      d4: 23 2c 91 00  	sw	s1, 24(sp)
      d8: 13 8d 04 00  	mv	s10, s1
      dc: 23 24 b1 02  	sw	a1, 40(sp)
;         deeploy_log("  [\r\n  ");
      e0: 03 25 c1 01  	lw	a0, 28(sp)
      e4: 97 00 00 00  	auipc	ra, 0
      e8: e7 80 00 00  	jalr	ra
      ec: 93 04 00 00  	li	s1, 0
      f0: 93 0b 0d 00  	mv	s7, s10
      f4: 93 8d 0b 00  	mv	s11, s7
      f8: 13 0a 04 00  	mv	s4, s0
;               (int16_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] + offset));
      fc: 0b f5 5d 51  	<unknown>
     100: 33 05 35 01  	add	a0, a0, s3
     104: b3 45 05 10  	<unknown>
;           deeploy_log(
     108: 13 05 0c 00  	mv	a0, s8
     10c: 97 00 00 00  	auipc	ra, 0
     110: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     114: 13 0a fa ff  	addi	s4, s4, -1
     118: e3 12 0a fe  	bnez	s4, 0xfc <PrintMatrix_s16_NHWC+0xfc>
;           deeploy_log("\r\n  ");
     11c: 13 85 0c 00  	mv	a0, s9
     120: 97 00 00 00  	auipc	ra, 0
     124: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     128: 93 84 14 00  	addi	s1, s1, 1
     12c: b3 8b 6b 01  	add	s7, s7, s6
     130: e3 92 24 fd  	bne	s1, s2, 0xf4 <PrintMatrix_s16_NHWC+0xf4>
;         deeploy_log("]\r\n");
     134: 03 25 01 02  	lw	a0, 32(sp)
     138: 97 00 00 00  	auipc	ra, 0
     13c: e7 80 00 00  	jalr	ra
     140: 83 25 81 02  	lw	a1, 40(sp)
;     for (uint32_t c = 0; c < C; c++) {
     144: 93 85 15 00  	addi	a1, a1, 1
     148: 13 0d 2d 00  	addi	s10, s10, 2
     14c: 03 25 41 02  	lw	a0, 36(sp)
     150: e3 96 a5 f8  	bne	a1, a0, 0xdc <PrintMatrix_s16_NHWC+0xdc>
;       deeploy_log("]\r\n");
     154: 03 25 01 02  	lw	a0, 32(sp)
     158: 97 00 00 00  	auipc	ra, 0
     15c: e7 80 00 00  	jalr	ra
     160: 03 26 41 01  	lw	a2, 20(sp)
;   for (uint32_t n = 0; n < N; n++) {
     164: 13 06 16 00  	addi	a2, a2, 1
     168: 83 24 81 01  	lw	s1, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     16c: 03 25 c1 00  	lw	a0, 12(sp)
     170: b3 84 a4 00  	add	s1, s1, a0
     174: 03 2d 01 01  	lw	s10, 16(sp)
     178: e3 14 a6 f5  	bne	a2, s10, 0xc0 <PrintMatrix_s16_NHWC+0xc0>
     17c: 6f 00 c0 11  	j	0x298 <PrintMatrix_s16_NHWC+0x298>
     180: 37 05 00 00  	lui	a0, 0
     184: 93 04 05 00  	mv	s1, a0
     188: 37 05 00 00  	lui	a0, 0
     18c: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     190: 13 85 04 00  	mv	a0, s1
     194: 97 00 00 00  	auipc	ra, 0
     198: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     19c: 13 05 09 00  	mv	a0, s2
     1a0: 97 00 00 00  	auipc	ra, 0
     1a4: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1a8: 13 0d fd ff  	addi	s10, s10, -1
     1ac: e3 12 0d fe  	bnez	s10, 0x190 <PrintMatrix_s16_NHWC+0x190>
     1b0: 6f 00 80 0e  	j	0x298 <PrintMatrix_s16_NHWC+0x298>
     1b4: 13 04 00 00  	li	s0, 0
     1b8: 37 05 00 00  	lui	a0, 0
     1bc: 13 09 05 00  	mv	s2, a0
     1c0: 37 05 00 00  	lui	a0, 0
     1c4: 93 09 05 00  	mv	s3, a0
     1c8: 37 05 00 00  	lui	a0, 0
     1cc: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1d0: 13 05 09 00  	mv	a0, s2
     1d4: 97 00 00 00  	auipc	ra, 0
     1d8: e7 80 00 00  	jalr	ra
     1dc: 83 24 41 02  	lw	s1, 36(sp)
;         deeploy_log("  [\r\n  ");
     1e0: 13 85 09 00  	mv	a0, s3
     1e4: 97 00 00 00  	auipc	ra, 0
     1e8: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1ec: 13 05 0a 00  	mv	a0, s4
     1f0: 97 00 00 00  	auipc	ra, 0
     1f4: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1f8: 93 84 f4 ff  	addi	s1, s1, -1
     1fc: e3 92 04 fe  	bnez	s1, 0x1e0 <PrintMatrix_s16_NHWC+0x1e0>
;       deeploy_log("]\r\n");
     200: 13 05 0a 00  	mv	a0, s4
     204: 97 00 00 00  	auipc	ra, 0
     208: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     20c: 13 04 14 00  	addi	s0, s0, 1
     210: e3 10 a4 fd  	bne	s0, s10, 0x1d0 <PrintMatrix_s16_NHWC+0x1d0>
     214: 6f 00 40 08  	j	0x298 <PrintMatrix_s16_NHWC+0x298>
     218: 37 05 00 00  	lui	a0, 0
     21c: 93 09 05 00  	mv	s3, a0
     220: 37 05 00 00  	lui	a0, 0
     224: 13 0a 05 00  	mv	s4, a0
     228: 37 05 00 00  	lui	a0, 0
     22c: 93 0a 05 00  	mv	s5, a0
     230: 37 05 00 00  	lui	a0, 0
     234: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     238: 13 85 09 00  	mv	a0, s3
     23c: 97 00 00 00  	auipc	ra, 0
     240: e7 80 00 00  	jalr	ra
     244: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     248: 13 05 0a 00  	mv	a0, s4
     24c: 97 00 00 00  	auipc	ra, 0
     250: e7 80 00 00  	jalr	ra
     254: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     258: 13 85 0a 00  	mv	a0, s5
     25c: 97 00 00 00  	auipc	ra, 0
     260: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     264: 93 8b fb ff  	addi	s7, s7, -1
     268: e3 98 0b fe  	bnez	s7, 0x258 <PrintMatrix_s16_NHWC+0x258>
;         deeploy_log("]\r\n");
     26c: 13 05 0b 00  	mv	a0, s6
     270: 97 00 00 00  	auipc	ra, 0
     274: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     278: 93 84 14 00  	addi	s1, s1, 1
     27c: 03 25 41 02  	lw	a0, 36(sp)
     280: e3 94 a4 fc  	bne	s1, a0, 0x248 <PrintMatrix_s16_NHWC+0x248>
;       deeploy_log("]\r\n");
     284: 13 05 0b 00  	mv	a0, s6
     288: 97 00 00 00  	auipc	ra, 0
     28c: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     290: 13 04 14 00  	addi	s0, s0, 1
     294: e3 12 a4 fb  	bne	s0, s10, 0x238 <PrintMatrix_s16_NHWC+0x238>
; }
     298: 83 20 c1 05  	lw	ra, 92(sp)
     29c: 03 24 81 05  	lw	s0, 88(sp)
     2a0: 83 24 41 05  	lw	s1, 84(sp)
     2a4: 03 29 01 05  	lw	s2, 80(sp)
     2a8: 83 29 c1 04  	lw	s3, 76(sp)
     2ac: 03 2a 81 04  	lw	s4, 72(sp)
     2b0: 83 2a 41 04  	lw	s5, 68(sp)
     2b4: 03 2b 01 04  	lw	s6, 64(sp)
     2b8: 83 2b c1 03  	lw	s7, 60(sp)
     2bc: 03 2c 81 03  	lw	s8, 56(sp)
     2c0: 83 2c 41 03  	lw	s9, 52(sp)
     2c4: 03 2d 01 03  	lw	s10, 48(sp)
     2c8: 83 2d c1 02  	lw	s11, 44(sp)
     2cc: 13 01 01 06  	addi	sp, sp, 96
     2d0: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_s32_NCHW:

00000000 <PrintMatrix_s32_NCHW>:
;                           uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 24 c1 02  	sw	a2, 40(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 8a 05 24  	beqz	a1, 0x290 <PrintMatrix_s32_NCHW+0x290>
      40: 93 8d 05 00  	mv	s11, a1
      44: 93 0b 05 00  	mv	s7, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 81 02  	lw	a0, 40(sp)
      4c: 63 04 05 12  	beqz	a0, 0x174 <PrintMatrix_s32_NCHW+0x174>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 8a 06 14  	beqz	a3, 0x1a8 <PrintMatrix_s32_NCHW+0x1a8>
      58: 13 0d 07 00  	mv	s10, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 08 07 1a  	beqz	a4, 0x20c <PrintMatrix_s32_NCHW+0x20c>
      60: 93 89 07 00  	mv	s3, a5
      64: 13 06 00 00  	li	a2, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 05 2d 03  	mul	a0, s10, s2
      6c: 83 25 81 02  	lw	a1, 40(sp)
      70: b3 05 b5 02  	mul	a1, a0, a1
      74: 93 95 25 00  	slli	a1, a1, 2
      78: 23 26 b1 00  	sw	a1, 12(sp)
      7c: 13 15 25 00  	slli	a0, a0, 2
      80: 23 20 a1 02  	sw	a0, 32(sp)
      84: 93 1a 2d 00  	slli	s5, s10, 2
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 24 a1 00  	sw	a0, 8(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 05 05 00  	mv	a0, a0
      9c: 23 2e a1 00  	sw	a0, 28(sp)
      a0: 37 05 00 00  	lui	a0, 0
      a4: 13 0c 05 00  	mv	s8, a0
      a8: 37 05 00 00  	lui	a0, 0
      ac: 93 0c 05 00  	mv	s9, a0
      b0: 37 05 00 00  	lui	a0, 0
      b4: 13 05 05 00  	mv	a0, a0
      b8: 23 22 a1 02  	sw	a0, 36(sp)
      bc: 23 28 b1 01  	sw	s11, 16(sp)
      c0: 23 2a c1 00  	sw	a2, 20(sp)
;       deeploy_log("[\r\n");
      c4: 03 25 81 00  	lw	a0, 8(sp)
      c8: 97 00 00 00  	auipc	ra, 0
      cc: e7 80 00 00  	jalr	ra
      d0: 13 04 00 00  	li	s0, 0
      d4: 23 2c 71 01  	sw	s7, 24(sp)
;         deeploy_log("  [\r\n  ");
      d8: 03 25 c1 01  	lw	a0, 28(sp)
      dc: 97 00 00 00  	auipc	ra, 0
      e0: e7 80 00 00  	jalr	ra
      e4: 93 04 00 00  	li	s1, 0
      e8: 13 8b 0b 00  	mv	s6, s7
      ec: 93 0d 0b 00  	mv	s11, s6
      f0: 13 0a 0d 00  	mv	s4, s10
;               (int32_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] + offset));
      f4: 0b a5 4d 00  	<unknown>
      f8: b3 05 35 01  	add	a1, a0, s3
;           deeploy_log(
      fc: 13 05 0c 00  	mv	a0, s8
     100: 97 00 00 00  	auipc	ra, 0
     104: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     108: 13 0a fa ff  	addi	s4, s4, -1
     10c: e3 14 0a fe  	bnez	s4, 0xf4 <PrintMatrix_s32_NCHW+0xf4>
;           deeploy_log("\r\n  ");
     110: 13 85 0c 00  	mv	a0, s9
     114: 97 00 00 00  	auipc	ra, 0
     118: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     11c: 93 84 14 00  	addi	s1, s1, 1
     120: 33 0b 5b 01  	add	s6, s6, s5
     124: e3 94 24 fd  	bne	s1, s2, 0xec <PrintMatrix_s32_NCHW+0xec>
;         deeploy_log("]\r\n");
     128: 03 25 41 02  	lw	a0, 36(sp)
     12c: 97 00 00 00  	auipc	ra, 0
     130: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     134: 13 04 14 00  	addi	s0, s0, 1
     138: 03 25 01 02  	lw	a0, 32(sp)
     13c: b3 8b ab 00  	add	s7, s7, a0
     140: 03 25 81 02  	lw	a0, 40(sp)
     144: e3 1a a4 f8  	bne	s0, a0, 0xd8 <PrintMatrix_s32_NCHW+0xd8>
;       deeploy_log("]\r\n");
     148: 03 25 41 02  	lw	a0, 36(sp)
     14c: 97 00 00 00  	auipc	ra, 0
     150: e7 80 00 00  	jalr	ra
     154: 03 26 41 01  	lw	a2, 20(sp)
;   for (uint32_t n = 0; n < N; n++) {
     158: 13 06 16 00  	addi	a2, a2, 1
     15c: 83 2b 81 01  	lw	s7, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     160: 03 25 c1 00  	lw	a0, 12(sp)
     164: b3 8b ab 00  	add	s7, s7, a0
     168: 83 2d 01 01  	lw	s11, 16(sp)
     16c: e3 1a b6 f5  	bne	a2, s11, 0xc0 <PrintMatrix_s32_NCHW+0xc0>
     170: 6f 00 00 12  	j	0x290 <PrintMatrix_s32_NCHW+0x290>
     174: 37 05 00 00  	lui	a0, 0
     178: 93 04 05 00  	mv	s1, a0
     17c: 37 05 00 00  	lui	a0, 0
     180: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     184: 13 85 04 00  	mv	a0, s1
     188: 97 00 00 00  	auipc	ra, 0
     18c: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     190: 13 05 09 00  	mv	a0, s2
     194: 97 00 00 00  	auipc	ra, 0
     198: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     19c: 93 8d fd ff  	addi	s11, s11, -1
     1a0: e3 92 0d fe  	bnez	s11, 0x184 <PrintMatrix_s32_NCHW+0x184>
     1a4: 6f 00 c0 0e  	j	0x290 <PrintMatrix_s32_NCHW+0x290>
     1a8: 13 04 00 00  	li	s0, 0
     1ac: 37 05 00 00  	lui	a0, 0
     1b0: 13 09 05 00  	mv	s2, a0
     1b4: 37 05 00 00  	lui	a0, 0
     1b8: 93 09 05 00  	mv	s3, a0
     1bc: 37 05 00 00  	lui	a0, 0
     1c0: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1c4: 13 05 09 00  	mv	a0, s2
     1c8: 97 00 00 00  	auipc	ra, 0
     1cc: e7 80 00 00  	jalr	ra
     1d0: 83 24 81 02  	lw	s1, 40(sp)
;         deeploy_log("  [\r\n  ");
     1d4: 13 85 09 00  	mv	a0, s3
     1d8: 97 00 00 00  	auipc	ra, 0
     1dc: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1e0: 13 05 0a 00  	mv	a0, s4
     1e4: 97 00 00 00  	auipc	ra, 0
     1e8: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1ec: 93 84 f4 ff  	addi	s1, s1, -1
     1f0: e3 92 04 fe  	bnez	s1, 0x1d4 <PrintMatrix_s32_NCHW+0x1d4>
;       deeploy_log("]\r\n");
     1f4: 13 05 0a 00  	mv	a0, s4
     1f8: 97 00 00 00  	auipc	ra, 0
     1fc: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     200: 13 04 14 00  	addi	s0, s0, 1
     204: e3 10 b4 fd  	bne	s0, s11, 0x1c4 <PrintMatrix_s32_NCHW+0x1c4>
     208: 6f 00 80 08  	j	0x290 <PrintMatrix_s32_NCHW+0x290>
     20c: 13 04 00 00  	li	s0, 0
     210: 37 05 00 00  	lui	a0, 0
     214: 93 09 05 00  	mv	s3, a0
     218: 37 05 00 00  	lui	a0, 0
     21c: 13 0a 05 00  	mv	s4, a0
     220: 37 05 00 00  	lui	a0, 0
     224: 93 0a 05 00  	mv	s5, a0
     228: 37 05 00 00  	lui	a0, 0
     22c: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     230: 13 85 09 00  	mv	a0, s3
     234: 97 00 00 00  	auipc	ra, 0
     238: e7 80 00 00  	jalr	ra
     23c: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     240: 13 05 0a 00  	mv	a0, s4
     244: 97 00 00 00  	auipc	ra, 0
     248: e7 80 00 00  	jalr	ra
     24c: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     250: 13 85 0a 00  	mv	a0, s5
     254: 97 00 00 00  	auipc	ra, 0
     258: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     25c: 93 8b fb ff  	addi	s7, s7, -1
     260: e3 98 0b fe  	bnez	s7, 0x250 <PrintMatrix_s32_NCHW+0x250>
;         deeploy_log("]\r\n");
     264: 13 05 0b 00  	mv	a0, s6
     268: 97 00 00 00  	auipc	ra, 0
     26c: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     270: 93 84 14 00  	addi	s1, s1, 1
     274: 03 25 81 02  	lw	a0, 40(sp)
     278: e3 94 a4 fc  	bne	s1, a0, 0x240 <PrintMatrix_s32_NCHW+0x240>
;       deeploy_log("]\r\n");
     27c: 13 05 0b 00  	mv	a0, s6
     280: 97 00 00 00  	auipc	ra, 0
     284: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     288: 13 04 14 00  	addi	s0, s0, 1
     28c: e3 12 b4 fb  	bne	s0, s11, 0x230 <PrintMatrix_s32_NCHW+0x230>
; }
     290: 83 20 c1 05  	lw	ra, 92(sp)
     294: 03 24 81 05  	lw	s0, 88(sp)
     298: 83 24 41 05  	lw	s1, 84(sp)
     29c: 03 29 01 05  	lw	s2, 80(sp)
     2a0: 83 29 c1 04  	lw	s3, 76(sp)
     2a4: 03 2a 81 04  	lw	s4, 72(sp)
     2a8: 83 2a 41 04  	lw	s5, 68(sp)
     2ac: 03 2b 01 04  	lw	s6, 64(sp)
     2b0: 83 2b c1 03  	lw	s7, 60(sp)
     2b4: 03 2c 81 03  	lw	s8, 56(sp)
     2b8: 83 2c 41 03  	lw	s9, 52(sp)
     2bc: 03 2d 01 03  	lw	s10, 48(sp)
     2c0: 83 2d c1 02  	lw	s11, 44(sp)
     2c4: 13 01 01 06  	addi	sp, sp, 96
     2c8: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_s32_NHWC:

00000000 <PrintMatrix_s32_NHWC>:
;                           uint32_t C, uint32_t H, uint32_t W, int32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 22 c1 02  	sw	a2, 36(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 8c 05 24  	beqz	a1, 0x294 <PrintMatrix_s32_NHWC+0x294>
      40: 13 8d 05 00  	mv	s10, a1
      44: 93 04 05 00  	mv	s1, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 41 02  	lw	a0, 36(sp)
      4c: 63 08 05 12  	beqz	a0, 0x17c <PrintMatrix_s32_NHWC+0x17c>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 8e 06 14  	beqz	a3, 0x1b0 <PrintMatrix_s32_NHWC+0x1b0>
      58: 13 04 07 00  	mv	s0, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 0c 07 1a  	beqz	a4, 0x214 <PrintMatrix_s32_NHWC+0x214>
      60: 93 89 07 00  	mv	s3, a5
      64: 13 06 00 00  	li	a2, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 05 24 03  	mul	a0, s0, s2
      6c: 83 25 41 02  	lw	a1, 36(sp)
      70: 33 05 b5 02  	mul	a0, a0, a1
      74: 13 15 25 00  	slli	a0, a0, 2
      78: 23 26 a1 00  	sw	a0, 12(sp)
      7c: 33 05 b4 02  	mul	a0, s0, a1
      80: 13 1b 25 00  	slli	s6, a0, 2
      84: 93 9a 25 00  	slli	s5, a1, 2
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 24 a1 00  	sw	a0, 8(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 05 05 00  	mv	a0, a0
      9c: 23 2e a1 00  	sw	a0, 28(sp)
      a0: 37 05 00 00  	lui	a0, 0
      a4: 13 0c 05 00  	mv	s8, a0
      a8: 37 05 00 00  	lui	a0, 0
      ac: 93 0c 05 00  	mv	s9, a0
      b0: 37 05 00 00  	lui	a0, 0
      b4: 13 05 05 00  	mv	a0, a0
      b8: 23 20 a1 02  	sw	a0, 32(sp)
      bc: 23 28 a1 01  	sw	s10, 16(sp)
      c0: 23 2a c1 00  	sw	a2, 20(sp)
;       deeploy_log("[\r\n");
      c4: 03 25 81 00  	lw	a0, 8(sp)
      c8: 97 00 00 00  	auipc	ra, 0
      cc: e7 80 00 00  	jalr	ra
      d0: 93 05 00 00  	li	a1, 0
      d4: 23 2c 91 00  	sw	s1, 24(sp)
      d8: 13 8d 04 00  	mv	s10, s1
      dc: 23 24 b1 02  	sw	a1, 40(sp)
;         deeploy_log("  [\r\n  ");
      e0: 03 25 c1 01  	lw	a0, 28(sp)
      e4: 97 00 00 00  	auipc	ra, 0
      e8: e7 80 00 00  	jalr	ra
      ec: 93 04 00 00  	li	s1, 0
      f0: 93 0b 0d 00  	mv	s7, s10
      f4: 93 8d 0b 00  	mv	s11, s7
      f8: 13 0a 04 00  	mv	s4, s0
;               (int32_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] + offset));
      fc: 0b f5 5d 21  	<unknown>
     100: b3 05 35 01  	add	a1, a0, s3
;           deeploy_log(
     104: 13 05 0c 00  	mv	a0, s8
     108: 97 00 00 00  	auipc	ra, 0
     10c: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     110: 13 0a fa ff  	addi	s4, s4, -1
     114: e3 14 0a fe  	bnez	s4, 0xfc <PrintMatrix_s32_NHWC+0xfc>
;           deeploy_log("\r\n  ");
     118: 13 85 0c 00  	mv	a0, s9
     11c: 97 00 00 00  	auipc	ra, 0
     120: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     124: 93 84 14 00  	addi	s1, s1, 1
     128: b3 8b 6b 01  	add	s7, s7, s6
     12c: e3 94 24 fd  	bne	s1, s2, 0xf4 <PrintMatrix_s32_NHWC+0xf4>
;         deeploy_log("]\r\n");
     130: 03 25 01 02  	lw	a0, 32(sp)
     134: 97 00 00 00  	auipc	ra, 0
     138: e7 80 00 00  	jalr	ra
     13c: 83 25 81 02  	lw	a1, 40(sp)
;     for (uint32_t c = 0; c < C; c++) {
     140: 93 85 15 00  	addi	a1, a1, 1
     144: 13 0d 4d 00  	addi	s10, s10, 4
     148: 03 25 41 02  	lw	a0, 36(sp)
     14c: e3 98 a5 f8  	bne	a1, a0, 0xdc <PrintMatrix_s32_NHWC+0xdc>
;       deeploy_log("]\r\n");
     150: 03 25 01 02  	lw	a0, 32(sp)
     154: 97 00 00 00  	auipc	ra, 0
     158: e7 80 00 00  	jalr	ra
     15c: 03 26 41 01  	lw	a2, 20(sp)
;   for (uint32_t n = 0; n < N; n++) {
     160: 13 06 16 00  	addi	a2, a2, 1
     164: 83 24 81 01  	lw	s1, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     168: 03 25 c1 00  	lw	a0, 12(sp)
     16c: b3 84 a4 00  	add	s1, s1, a0
     170: 03 2d 01 01  	lw	s10, 16(sp)
     174: e3 16 a6 f5  	bne	a2, s10, 0xc0 <PrintMatrix_s32_NHWC+0xc0>
     178: 6f 00 c0 11  	j	0x294 <PrintMatrix_s32_NHWC+0x294>
     17c: 37 05 00 00  	lui	a0, 0
     180: 93 04 05 00  	mv	s1, a0
     184: 37 05 00 00  	lui	a0, 0
     188: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     18c: 13 85 04 00  	mv	a0, s1
     190: 97 00 00 00  	auipc	ra, 0
     194: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     198: 13 05 09 00  	mv	a0, s2
     19c: 97 00 00 00  	auipc	ra, 0
     1a0: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1a4: 13 0d fd ff  	addi	s10, s10, -1
     1a8: e3 12 0d fe  	bnez	s10, 0x18c <PrintMatrix_s32_NHWC+0x18c>
     1ac: 6f 00 80 0e  	j	0x294 <PrintMatrix_s32_NHWC+0x294>
     1b0: 13 04 00 00  	li	s0, 0
     1b4: 37 05 00 00  	lui	a0, 0
     1b8: 13 09 05 00  	mv	s2, a0
     1bc: 37 05 00 00  	lui	a0, 0
     1c0: 93 09 05 00  	mv	s3, a0
     1c4: 37 05 00 00  	lui	a0, 0
     1c8: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1cc: 13 05 09 00  	mv	a0, s2
     1d0: 97 00 00 00  	auipc	ra, 0
     1d4: e7 80 00 00  	jalr	ra
     1d8: 83 24 41 02  	lw	s1, 36(sp)
;         deeploy_log("  [\r\n  ");
     1dc: 13 85 09 00  	mv	a0, s3
     1e0: 97 00 00 00  	auipc	ra, 0
     1e4: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1e8: 13 05 0a 00  	mv	a0, s4
     1ec: 97 00 00 00  	auipc	ra, 0
     1f0: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1f4: 93 84 f4 ff  	addi	s1, s1, -1
     1f8: e3 92 04 fe  	bnez	s1, 0x1dc <PrintMatrix_s32_NHWC+0x1dc>
;       deeploy_log("]\r\n");
     1fc: 13 05 0a 00  	mv	a0, s4
     200: 97 00 00 00  	auipc	ra, 0
     204: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     208: 13 04 14 00  	addi	s0, s0, 1
     20c: e3 10 a4 fd  	bne	s0, s10, 0x1cc <PrintMatrix_s32_NHWC+0x1cc>
     210: 6f 00 40 08  	j	0x294 <PrintMatrix_s32_NHWC+0x294>
     214: 37 05 00 00  	lui	a0, 0
     218: 93 09 05 00  	mv	s3, a0
     21c: 37 05 00 00  	lui	a0, 0
     220: 13 0a 05 00  	mv	s4, a0
     224: 37 05 00 00  	lui	a0, 0
     228: 93 0a 05 00  	mv	s5, a0
     22c: 37 05 00 00  	lui	a0, 0
     230: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     234: 13 85 09 00  	mv	a0, s3
     238: 97 00 00 00  	auipc	ra, 0
     23c: e7 80 00 00  	jalr	ra
     240: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     244: 13 05 0a 00  	mv	a0, s4
     248: 97 00 00 00  	auipc	ra, 0
     24c: e7 80 00 00  	jalr	ra
     250: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     254: 13 85 0a 00  	mv	a0, s5
     258: 97 00 00 00  	auipc	ra, 0
     25c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     260: 93 8b fb ff  	addi	s7, s7, -1
     264: e3 98 0b fe  	bnez	s7, 0x254 <PrintMatrix_s32_NHWC+0x254>
;         deeploy_log("]\r\n");
     268: 13 05 0b 00  	mv	a0, s6
     26c: 97 00 00 00  	auipc	ra, 0
     270: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     274: 93 84 14 00  	addi	s1, s1, 1
     278: 03 25 41 02  	lw	a0, 36(sp)
     27c: e3 94 a4 fc  	bne	s1, a0, 0x244 <PrintMatrix_s32_NHWC+0x244>
;       deeploy_log("]\r\n");
     280: 13 05 0b 00  	mv	a0, s6
     284: 97 00 00 00  	auipc	ra, 0
     288: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     28c: 13 04 14 00  	addi	s0, s0, 1
     290: e3 12 a4 fb  	bne	s0, s10, 0x234 <PrintMatrix_s32_NHWC+0x234>
; }
     294: 83 20 c1 05  	lw	ra, 92(sp)
     298: 03 24 81 05  	lw	s0, 88(sp)
     29c: 83 24 41 05  	lw	s1, 84(sp)
     2a0: 03 29 01 05  	lw	s2, 80(sp)
     2a4: 83 29 c1 04  	lw	s3, 76(sp)
     2a8: 03 2a 81 04  	lw	s4, 72(sp)
     2ac: 83 2a 41 04  	lw	s5, 68(sp)
     2b0: 03 2b 01 04  	lw	s6, 64(sp)
     2b4: 83 2b c1 03  	lw	s7, 60(sp)
     2b8: 03 2c 81 03  	lw	s8, 56(sp)
     2bc: 83 2c 41 03  	lw	s9, 52(sp)
     2c0: 03 2d 01 03  	lw	s10, 48(sp)
     2c4: 83 2d c1 02  	lw	s11, 44(sp)
     2c8: 13 01 01 06  	addi	sp, sp, 96
     2cc: 67 80 00 00  	ret

Disassembly of section .text.PrintArray_s8:

00000000 <PrintArray_s8>:
;                    int32_t offset) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 23 26 31 01  	sw	s3, 12(sp)
;   for (uint32_t n = 0; n < N; n++) {
      18: 63 8c 05 02  	beqz	a1, 0x50 <PrintArray_s8+0x50>
      1c: 13 04 06 00  	mv	s0, a2
      20: 93 84 05 00  	mv	s1, a1
      24: 13 09 05 00  	mv	s2, a0
      28: 37 05 00 00  	lui	a0, 0
      2c: 93 09 05 00  	mv	s3, a0
;     deeploy_log("%4d ", (int8_t)(pSrcA[n] + offset));
      30: 0b 45 19 00  	<unknown>
      34: 33 05 85 00  	add	a0, a0, s0
      38: b3 65 05 10  	<unknown>
      3c: 13 85 09 00  	mv	a0, s3
      40: 97 00 00 00  	auipc	ra, 0
      44: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
      48: 93 84 f4 ff  	addi	s1, s1, -1
      4c: e3 92 04 fe  	bnez	s1, 0x30 <PrintArray_s8+0x30>
;   deeploy_log("\r\n");
      50: 37 05 00 00  	lui	a0, 0
      54: 13 05 05 00  	mv	a0, a0
      58: 83 20 c1 01  	lw	ra, 28(sp)
      5c: 03 24 81 01  	lw	s0, 24(sp)
      60: 83 24 41 01  	lw	s1, 20(sp)
      64: 03 29 01 01  	lw	s2, 16(sp)
      68: 83 29 c1 00  	lw	s3, 12(sp)
      6c: 13 01 01 02  	addi	sp, sp, 32
      70: 17 03 00 00  	auipc	t1, 0
      74: 67 00 03 00  	jr	t1

Disassembly of section .text.PrintArray_s16:

00000000 <PrintArray_s16>:
;                     int32_t offset) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 23 26 31 01  	sw	s3, 12(sp)
;   for (uint32_t n = 0; n < N; n++) {
      18: 63 8c 05 02  	beqz	a1, 0x50 <PrintArray_s16+0x50>
      1c: 13 04 06 00  	mv	s0, a2
      20: 93 84 05 00  	mv	s1, a1
      24: 13 09 05 00  	mv	s2, a0
      28: 37 05 00 00  	lui	a0, 0
      2c: 93 09 05 00  	mv	s3, a0
;     deeploy_log("%6hd ", (int16_t)(pSrcA[n] + offset));
      30: 0b 55 29 00  	<unknown>
      34: 33 05 85 00  	add	a0, a0, s0
      38: b3 45 05 10  	<unknown>
      3c: 13 85 09 00  	mv	a0, s3
      40: 97 00 00 00  	auipc	ra, 0
      44: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
      48: 93 84 f4 ff  	addi	s1, s1, -1
      4c: e3 92 04 fe  	bnez	s1, 0x30 <PrintArray_s16+0x30>
;   deeploy_log("\r\n");
      50: 37 05 00 00  	lui	a0, 0
      54: 13 05 05 00  	mv	a0, a0
      58: 83 20 c1 01  	lw	ra, 28(sp)
      5c: 03 24 81 01  	lw	s0, 24(sp)
      60: 83 24 41 01  	lw	s1, 20(sp)
      64: 03 29 01 01  	lw	s2, 16(sp)
      68: 83 29 c1 00  	lw	s3, 12(sp)
      6c: 13 01 01 02  	addi	sp, sp, 32
      70: 17 03 00 00  	auipc	t1, 0
      74: 67 00 03 00  	jr	t1

Disassembly of section .text.PrintArray_s32:

00000000 <PrintArray_s32>:
;                     int32_t offset) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 23 26 31 01  	sw	s3, 12(sp)
;   for (uint32_t n = 0; n < N; n++) {
      18: 63 8a 05 02  	beqz	a1, 0x4c <PrintArray_s32+0x4c>
      1c: 13 04 06 00  	mv	s0, a2
      20: 93 84 05 00  	mv	s1, a1
      24: 13 09 05 00  	mv	s2, a0
      28: 37 05 00 00  	lui	a0, 0
      2c: 93 09 05 00  	mv	s3, a0
;     deeploy_log("%11" PRId32 " ", (int32_t)(pSrcA[n] + offset));
      30: 0b 25 49 00  	<unknown>
      34: b3 05 85 00  	add	a1, a0, s0
      38: 13 85 09 00  	mv	a0, s3
      3c: 97 00 00 00  	auipc	ra, 0
      40: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
      44: 93 84 f4 ff  	addi	s1, s1, -1
      48: e3 94 04 fe  	bnez	s1, 0x30 <PrintArray_s32+0x30>
;   deeploy_log("\r\n");
      4c: 37 05 00 00  	lui	a0, 0
      50: 13 05 05 00  	mv	a0, a0
      54: 83 20 c1 01  	lw	ra, 28(sp)
      58: 03 24 81 01  	lw	s0, 24(sp)
      5c: 83 24 41 01  	lw	s1, 20(sp)
      60: 03 29 01 01  	lw	s2, 16(sp)
      64: 83 29 c1 00  	lw	s3, 12(sp)
      68: 13 01 01 02  	addi	sp, sp, 32
      6c: 17 03 00 00  	auipc	t1, 0
      70: 67 00 03 00  	jr	t1

Disassembly of section .text.PrintMatrix_u8_NCHW:

00000000 <PrintMatrix_u8_NCHW>:
;                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 24 c1 02  	sw	a2, 40(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 84 05 24  	beqz	a1, 0x284 <PrintMatrix_u8_NCHW+0x284>
      40: 93 8d 05 00  	mv	s11, a1
      44: 13 0b 05 00  	mv	s6, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 81 02  	lw	a0, 40(sp)
      4c: 63 0e 05 10  	beqz	a0, 0x168 <PrintMatrix_u8_NCHW+0x168>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 84 06 14  	beqz	a3, 0x19c <PrintMatrix_u8_NCHW+0x19c>
      58: 93 0b 07 00  	mv	s7, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 02 07 1a  	beqz	a4, 0x200 <PrintMatrix_u8_NCHW+0x200>
      60: 93 89 07 00  	mv	s3, a5
      64: 93 05 00 00  	li	a1, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 86 2b 03  	mul	a2, s7, s2
      6c: 03 25 81 02  	lw	a0, 40(sp)
      70: 23 22 c1 02  	sw	a2, 36(sp)
      74: 33 05 a6 02  	mul	a0, a2, a0
      78: 23 28 a1 00  	sw	a0, 16(sp)
      7c: 37 05 00 00  	lui	a0, 0
      80: 13 05 05 00  	mv	a0, a0
      84: 23 26 a1 00  	sw	a0, 12(sp)
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 20 a1 02  	sw	a0, 32(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 0c 05 00  	mv	s8, a0
      9c: 37 05 00 00  	lui	a0, 0
      a0: 93 0c 05 00  	mv	s9, a0
      a4: 37 05 00 00  	lui	a0, 0
      a8: 13 0d 05 00  	mv	s10, a0
      ac: 23 2a b1 01  	sw	s11, 20(sp)
      b0: 23 2c b1 00  	sw	a1, 24(sp)
;       deeploy_log("[\r\n");
      b4: 03 25 c1 00  	lw	a0, 12(sp)
      b8: 97 00 00 00  	auipc	ra, 0
      bc: e7 80 00 00  	jalr	ra
      c0: 93 0a 00 00  	li	s5, 0
      c4: 23 2e 61 01  	sw	s6, 28(sp)
;         deeploy_log("  [\r\n  ");
      c8: 03 25 01 02  	lw	a0, 32(sp)
      cc: 97 00 00 00  	auipc	ra, 0
      d0: e7 80 00 00  	jalr	ra
      d4: 93 0d 00 00  	li	s11, 0
      d8: 93 04 0b 00  	mv	s1, s6
      dc: 13 84 04 00  	mv	s0, s1
      e0: 13 8a 0b 00  	mv	s4, s7
;               (uint8_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] + offset));
      e4: 0b 45 14 00  	<unknown>
      e8: 33 05 35 01  	add	a0, a0, s3
      ec: b3 75 05 10  	<unknown>
;           deeploy_log(
      f0: 13 05 0c 00  	mv	a0, s8
      f4: 97 00 00 00  	auipc	ra, 0
      f8: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
      fc: 13 0a fa ff  	addi	s4, s4, -1
     100: e3 12 0a fe  	bnez	s4, 0xe4 <PrintMatrix_u8_NCHW+0xe4>
;           deeploy_log("\r\n  ");
     104: 13 85 0c 00  	mv	a0, s9
     108: 97 00 00 00  	auipc	ra, 0
     10c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     110: 93 8d 1d 00  	addi	s11, s11, 1
     114: b3 84 74 01  	add	s1, s1, s7
     118: e3 92 2d fd  	bne	s11, s2, 0xdc <PrintMatrix_u8_NCHW+0xdc>
;         deeploy_log("]\r\n");
     11c: 13 05 0d 00  	mv	a0, s10
     120: 97 00 00 00  	auipc	ra, 0
     124: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     128: 93 8a 1a 00  	addi	s5, s5, 1
     12c: 03 25 41 02  	lw	a0, 36(sp)
     130: 33 0b ab 00  	add	s6, s6, a0
     134: 03 25 81 02  	lw	a0, 40(sp)
     138: e3 98 aa f8  	bne	s5, a0, 0xc8 <PrintMatrix_u8_NCHW+0xc8>
;       deeploy_log("]\r\n");
     13c: 13 05 0d 00  	mv	a0, s10
     140: 97 00 00 00  	auipc	ra, 0
     144: e7 80 00 00  	jalr	ra
     148: 83 25 81 01  	lw	a1, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     14c: 93 85 15 00  	addi	a1, a1, 1
     150: 03 2b c1 01  	lw	s6, 28(sp)
;   for (uint32_t n = 0; n < N; n++) {
     154: 03 25 01 01  	lw	a0, 16(sp)
     158: 33 0b ab 00  	add	s6, s6, a0
     15c: 83 2d 41 01  	lw	s11, 20(sp)
     160: e3 98 b5 f5  	bne	a1, s11, 0xb0 <PrintMatrix_u8_NCHW+0xb0>
     164: 6f 00 00 12  	j	0x284 <PrintMatrix_u8_NCHW+0x284>
     168: 37 05 00 00  	lui	a0, 0
     16c: 93 04 05 00  	mv	s1, a0
     170: 37 05 00 00  	lui	a0, 0
     174: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     178: 13 85 04 00  	mv	a0, s1
     17c: 97 00 00 00  	auipc	ra, 0
     180: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     184: 13 05 09 00  	mv	a0, s2
     188: 97 00 00 00  	auipc	ra, 0
     18c: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     190: 93 8d fd ff  	addi	s11, s11, -1
     194: e3 92 0d fe  	bnez	s11, 0x178 <PrintMatrix_u8_NCHW+0x178>
     198: 6f 00 c0 0e  	j	0x284 <PrintMatrix_u8_NCHW+0x284>
     19c: 13 04 00 00  	li	s0, 0
     1a0: 37 05 00 00  	lui	a0, 0
     1a4: 13 09 05 00  	mv	s2, a0
     1a8: 37 05 00 00  	lui	a0, 0
     1ac: 93 09 05 00  	mv	s3, a0
     1b0: 37 05 00 00  	lui	a0, 0
     1b4: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1b8: 13 05 09 00  	mv	a0, s2
     1bc: 97 00 00 00  	auipc	ra, 0
     1c0: e7 80 00 00  	jalr	ra
     1c4: 83 24 81 02  	lw	s1, 40(sp)
;         deeploy_log("  [\r\n  ");
     1c8: 13 85 09 00  	mv	a0, s3
     1cc: 97 00 00 00  	auipc	ra, 0
     1d0: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1d4: 13 05 0a 00  	mv	a0, s4
     1d8: 97 00 00 00  	auipc	ra, 0
     1dc: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1e0: 93 84 f4 ff  	addi	s1, s1, -1
     1e4: e3 92 04 fe  	bnez	s1, 0x1c8 <PrintMatrix_u8_NCHW+0x1c8>
;       deeploy_log("]\r\n");
     1e8: 13 05 0a 00  	mv	a0, s4
     1ec: 97 00 00 00  	auipc	ra, 0
     1f0: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1f4: 13 04 14 00  	addi	s0, s0, 1
     1f8: e3 10 b4 fd  	bne	s0, s11, 0x1b8 <PrintMatrix_u8_NCHW+0x1b8>
     1fc: 6f 00 80 08  	j	0x284 <PrintMatrix_u8_NCHW+0x284>
     200: 13 04 00 00  	li	s0, 0
     204: 37 05 00 00  	lui	a0, 0
     208: 93 09 05 00  	mv	s3, a0
     20c: 37 05 00 00  	lui	a0, 0
     210: 13 0a 05 00  	mv	s4, a0
     214: 37 05 00 00  	lui	a0, 0
     218: 93 0a 05 00  	mv	s5, a0
     21c: 37 05 00 00  	lui	a0, 0
     220: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     224: 13 85 09 00  	mv	a0, s3
     228: 97 00 00 00  	auipc	ra, 0
     22c: e7 80 00 00  	jalr	ra
     230: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     234: 13 05 0a 00  	mv	a0, s4
     238: 97 00 00 00  	auipc	ra, 0
     23c: e7 80 00 00  	jalr	ra
     240: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     244: 13 85 0a 00  	mv	a0, s5
     248: 97 00 00 00  	auipc	ra, 0
     24c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     250: 93 8b fb ff  	addi	s7, s7, -1
     254: e3 98 0b fe  	bnez	s7, 0x244 <PrintMatrix_u8_NCHW+0x244>
;         deeploy_log("]\r\n");
     258: 13 05 0b 00  	mv	a0, s6
     25c: 97 00 00 00  	auipc	ra, 0
     260: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     264: 93 84 14 00  	addi	s1, s1, 1
     268: 03 25 81 02  	lw	a0, 40(sp)
     26c: e3 94 a4 fc  	bne	s1, a0, 0x234 <PrintMatrix_u8_NCHW+0x234>
;       deeploy_log("]\r\n");
     270: 13 05 0b 00  	mv	a0, s6
     274: 97 00 00 00  	auipc	ra, 0
     278: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     27c: 13 04 14 00  	addi	s0, s0, 1
     280: e3 12 b4 fb  	bne	s0, s11, 0x224 <PrintMatrix_u8_NCHW+0x224>
; }
     284: 83 20 c1 05  	lw	ra, 92(sp)
     288: 03 24 81 05  	lw	s0, 88(sp)
     28c: 83 24 41 05  	lw	s1, 84(sp)
     290: 03 29 01 05  	lw	s2, 80(sp)
     294: 83 29 c1 04  	lw	s3, 76(sp)
     298: 03 2a 81 04  	lw	s4, 72(sp)
     29c: 83 2a 41 04  	lw	s5, 68(sp)
     2a0: 03 2b 01 04  	lw	s6, 64(sp)
     2a4: 83 2b c1 03  	lw	s7, 60(sp)
     2a8: 03 2c 81 03  	lw	s8, 56(sp)
     2ac: 83 2c 41 03  	lw	s9, 52(sp)
     2b0: 03 2d 01 03  	lw	s10, 48(sp)
     2b4: 83 2d c1 02  	lw	s11, 44(sp)
     2b8: 13 01 01 06  	addi	sp, sp, 96
     2bc: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_u8_NHWC:

00000000 <PrintMatrix_u8_NHWC>:
;                          uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
;   for (uint32_t n = 0; n < N; n++) {
      38: 63 84 05 24  	beqz	a1, 0x280 <PrintMatrix_u8_NHWC+0x280>
      3c: 93 04 06 00  	mv	s1, a2
      40: 13 8d 05 00  	mv	s10, a1
;     for (uint32_t c = 0; c < C; c++) {
      44: 63 02 06 12  	beqz	a2, 0x168 <PrintMatrix_u8_NHWC+0x168>
      48: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      4c: 63 88 06 14  	beqz	a3, 0x19c <PrintMatrix_u8_NHWC+0x19c>
      50: 93 0a 07 00  	mv	s5, a4
;         for (uint32_t w = 0; w < W; w++) {
      54: 63 06 07 1a  	beqz	a4, 0x200 <PrintMatrix_u8_NHWC+0x200>
      58: 93 89 07 00  	mv	s3, a5
      5c: 13 04 05 00  	mv	s0, a0
      60: 93 05 00 00  	li	a1, 0
;   for (uint32_t n = 0; n < N; n++) {
      64: 33 85 2a 03  	mul	a0, s5, s2
      68: 33 05 95 02  	mul	a0, a0, s1
      6c: 23 28 a1 00  	sw	a0, 16(sp)
      70: 33 8b 9a 02  	mul	s6, s5, s1
      74: 37 05 00 00  	lui	a0, 0
      78: 13 05 05 00  	mv	a0, a0
      7c: 23 26 a1 00  	sw	a0, 12(sp)
      80: 37 05 00 00  	lui	a0, 0
      84: 13 05 05 00  	mv	a0, a0
      88: 23 20 a1 02  	sw	a0, 32(sp)
      8c: 37 05 00 00  	lui	a0, 0
      90: 13 0c 05 00  	mv	s8, a0
      94: 37 05 00 00  	lui	a0, 0
      98: 93 0c 05 00  	mv	s9, a0
      9c: 37 05 00 00  	lui	a0, 0
      a0: 13 05 05 00  	mv	a0, a0
      a4: 23 22 a1 02  	sw	a0, 36(sp)
      a8: 23 2a a1 01  	sw	s10, 20(sp)
      ac: 23 2c b1 00  	sw	a1, 24(sp)
;       deeploy_log("[\r\n");
      b0: 03 25 c1 00  	lw	a0, 12(sp)
      b4: 97 00 00 00  	auipc	ra, 0
      b8: e7 80 00 00  	jalr	ra
      bc: 13 05 00 00  	li	a0, 0
      c0: 23 2e 81 00  	sw	s0, 28(sp)
      c4: 13 0d 04 00  	mv	s10, s0
      c8: 23 24 a1 02  	sw	a0, 40(sp)
;         deeploy_log("  [\r\n  ");
      cc: 03 25 01 02  	lw	a0, 32(sp)
      d0: 97 00 00 00  	auipc	ra, 0
      d4: e7 80 00 00  	jalr	ra
      d8: 93 0d 00 00  	li	s11, 0
      dc: 93 0b 0d 00  	mv	s7, s10
      e0: 13 84 0b 00  	mv	s0, s7
      e4: 13 8a 0a 00  	mv	s4, s5
;               (uint8_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] + offset));
      e8: 0b 75 94 40  	<unknown>
      ec: 33 05 35 01  	add	a0, a0, s3
      f0: b3 75 05 10  	<unknown>
;           deeploy_log(
      f4: 13 05 0c 00  	mv	a0, s8
      f8: 97 00 00 00  	auipc	ra, 0
      fc: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     100: 13 0a fa ff  	addi	s4, s4, -1
     104: e3 12 0a fe  	bnez	s4, 0xe8 <PrintMatrix_u8_NHWC+0xe8>
;           deeploy_log("\r\n  ");
     108: 13 85 0c 00  	mv	a0, s9
     10c: 97 00 00 00  	auipc	ra, 0
     110: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     114: 93 8d 1d 00  	addi	s11, s11, 1
     118: b3 8b 6b 01  	add	s7, s7, s6
     11c: e3 92 2d fd  	bne	s11, s2, 0xe0 <PrintMatrix_u8_NHWC+0xe0>
;         deeploy_log("]\r\n");
     120: 03 25 41 02  	lw	a0, 36(sp)
     124: 97 00 00 00  	auipc	ra, 0
     128: e7 80 00 00  	jalr	ra
     12c: 03 25 81 02  	lw	a0, 40(sp)
;     for (uint32_t c = 0; c < C; c++) {
     130: 13 05 15 00  	addi	a0, a0, 1
     134: 13 0d 1d 00  	addi	s10, s10, 1
     138: e3 18 95 f8  	bne	a0, s1, 0xc8 <PrintMatrix_u8_NHWC+0xc8>
;       deeploy_log("]\r\n");
     13c: 03 25 41 02  	lw	a0, 36(sp)
     140: 97 00 00 00  	auipc	ra, 0
     144: e7 80 00 00  	jalr	ra
     148: 83 25 81 01  	lw	a1, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     14c: 93 85 15 00  	addi	a1, a1, 1
     150: 03 24 c1 01  	lw	s0, 28(sp)
;   for (uint32_t n = 0; n < N; n++) {
     154: 03 25 01 01  	lw	a0, 16(sp)
     158: 33 04 a4 00  	add	s0, s0, a0
     15c: 03 2d 41 01  	lw	s10, 20(sp)
     160: e3 96 a5 f5  	bne	a1, s10, 0xac <PrintMatrix_u8_NHWC+0xac>
     164: 6f 00 c0 11  	j	0x280 <PrintMatrix_u8_NHWC+0x280>
     168: 37 05 00 00  	lui	a0, 0
     16c: 93 04 05 00  	mv	s1, a0
     170: 37 05 00 00  	lui	a0, 0
     174: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     178: 13 85 04 00  	mv	a0, s1
     17c: 97 00 00 00  	auipc	ra, 0
     180: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     184: 13 05 09 00  	mv	a0, s2
     188: 97 00 00 00  	auipc	ra, 0
     18c: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     190: 13 0d fd ff  	addi	s10, s10, -1
     194: e3 12 0d fe  	bnez	s10, 0x178 <PrintMatrix_u8_NHWC+0x178>
     198: 6f 00 80 0e  	j	0x280 <PrintMatrix_u8_NHWC+0x280>
     19c: 13 04 00 00  	li	s0, 0
     1a0: 37 05 00 00  	lui	a0, 0
     1a4: 13 09 05 00  	mv	s2, a0
     1a8: 37 05 00 00  	lui	a0, 0
     1ac: 93 09 05 00  	mv	s3, a0
     1b0: 37 05 00 00  	lui	a0, 0
     1b4: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1b8: 13 05 09 00  	mv	a0, s2
     1bc: 97 00 00 00  	auipc	ra, 0
     1c0: e7 80 00 00  	jalr	ra
     1c4: 93 8a 04 00  	mv	s5, s1
;         deeploy_log("  [\r\n  ");
     1c8: 13 85 09 00  	mv	a0, s3
     1cc: 97 00 00 00  	auipc	ra, 0
     1d0: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1d4: 13 05 0a 00  	mv	a0, s4
     1d8: 97 00 00 00  	auipc	ra, 0
     1dc: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1e0: 93 8a fa ff  	addi	s5, s5, -1
     1e4: e3 92 0a fe  	bnez	s5, 0x1c8 <PrintMatrix_u8_NHWC+0x1c8>
;       deeploy_log("]\r\n");
     1e8: 13 05 0a 00  	mv	a0, s4
     1ec: 97 00 00 00  	auipc	ra, 0
     1f0: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1f4: 13 04 14 00  	addi	s0, s0, 1
     1f8: e3 10 a4 fd  	bne	s0, s10, 0x1b8 <PrintMatrix_u8_NHWC+0x1b8>
     1fc: 6f 00 40 08  	j	0x280 <PrintMatrix_u8_NHWC+0x280>
     200: 13 04 00 00  	li	s0, 0
     204: 37 05 00 00  	lui	a0, 0
     208: 93 09 05 00  	mv	s3, a0
     20c: 37 05 00 00  	lui	a0, 0
     210: 13 0a 05 00  	mv	s4, a0
     214: 37 05 00 00  	lui	a0, 0
     218: 93 0a 05 00  	mv	s5, a0
     21c: 37 05 00 00  	lui	a0, 0
     220: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     224: 13 85 09 00  	mv	a0, s3
     228: 97 00 00 00  	auipc	ra, 0
     22c: e7 80 00 00  	jalr	ra
     230: 93 0b 00 00  	li	s7, 0
;         deeploy_log("  [\r\n  ");
     234: 13 05 0a 00  	mv	a0, s4
     238: 97 00 00 00  	auipc	ra, 0
     23c: e7 80 00 00  	jalr	ra
     240: 13 0c 09 00  	mv	s8, s2
;           deeploy_log("\r\n  ");
     244: 13 85 0a 00  	mv	a0, s5
     248: 97 00 00 00  	auipc	ra, 0
     24c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     250: 13 0c fc ff  	addi	s8, s8, -1
     254: e3 18 0c fe  	bnez	s8, 0x244 <PrintMatrix_u8_NHWC+0x244>
;         deeploy_log("]\r\n");
     258: 13 05 0b 00  	mv	a0, s6
     25c: 97 00 00 00  	auipc	ra, 0
     260: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     264: 93 8b 1b 00  	addi	s7, s7, 1
     268: e3 96 9b fc  	bne	s7, s1, 0x234 <PrintMatrix_u8_NHWC+0x234>
;       deeploy_log("]\r\n");
     26c: 13 05 0b 00  	mv	a0, s6
     270: 97 00 00 00  	auipc	ra, 0
     274: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     278: 13 04 14 00  	addi	s0, s0, 1
     27c: e3 14 a4 fb  	bne	s0, s10, 0x224 <PrintMatrix_u8_NHWC+0x224>
; }
     280: 83 20 c1 05  	lw	ra, 92(sp)
     284: 03 24 81 05  	lw	s0, 88(sp)
     288: 83 24 41 05  	lw	s1, 84(sp)
     28c: 03 29 01 05  	lw	s2, 80(sp)
     290: 83 29 c1 04  	lw	s3, 76(sp)
     294: 03 2a 81 04  	lw	s4, 72(sp)
     298: 83 2a 41 04  	lw	s5, 68(sp)
     29c: 03 2b 01 04  	lw	s6, 64(sp)
     2a0: 83 2b c1 03  	lw	s7, 60(sp)
     2a4: 03 2c 81 03  	lw	s8, 56(sp)
     2a8: 83 2c 41 03  	lw	s9, 52(sp)
     2ac: 03 2d 01 03  	lw	s10, 48(sp)
     2b0: 83 2d c1 02  	lw	s11, 44(sp)
     2b4: 13 01 01 06  	addi	sp, sp, 96
     2b8: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_u16_NCHW:

00000000 <PrintMatrix_u16_NCHW>:
;                           uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 24 c1 02  	sw	a2, 40(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 8c 05 24  	beqz	a1, 0x294 <PrintMatrix_u16_NCHW+0x294>
      40: 93 8d 05 00  	mv	s11, a1
      44: 93 0b 05 00  	mv	s7, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 81 02  	lw	a0, 40(sp)
      4c: 63 06 05 12  	beqz	a0, 0x178 <PrintMatrix_u16_NCHW+0x178>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 8c 06 14  	beqz	a3, 0x1ac <PrintMatrix_u16_NCHW+0x1ac>
      58: 13 0d 07 00  	mv	s10, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 0a 07 1a  	beqz	a4, 0x210 <PrintMatrix_u16_NCHW+0x210>
      60: 93 89 07 00  	mv	s3, a5
      64: 13 06 00 00  	li	a2, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 05 2d 03  	mul	a0, s10, s2
      6c: 83 25 81 02  	lw	a1, 40(sp)
      70: b3 05 b5 02  	mul	a1, a0, a1
      74: 93 95 15 00  	slli	a1, a1, 1
      78: 23 26 b1 00  	sw	a1, 12(sp)
      7c: 13 15 15 00  	slli	a0, a0, 1
      80: 23 20 a1 02  	sw	a0, 32(sp)
      84: 93 1a 1d 00  	slli	s5, s10, 1
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 24 a1 00  	sw	a0, 8(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 05 05 00  	mv	a0, a0
      9c: 23 2e a1 00  	sw	a0, 28(sp)
      a0: 37 05 00 00  	lui	a0, 0
      a4: 13 0c 05 00  	mv	s8, a0
      a8: 37 05 00 00  	lui	a0, 0
      ac: 93 0c 05 00  	mv	s9, a0
      b0: 37 05 00 00  	lui	a0, 0
      b4: 13 05 05 00  	mv	a0, a0
      b8: 23 22 a1 02  	sw	a0, 36(sp)
      bc: 23 28 b1 01  	sw	s11, 16(sp)
      c0: 23 2a c1 00  	sw	a2, 20(sp)
;       deeploy_log("[\r\n");
      c4: 03 25 81 00  	lw	a0, 8(sp)
      c8: 97 00 00 00  	auipc	ra, 0
      cc: e7 80 00 00  	jalr	ra
      d0: 13 04 00 00  	li	s0, 0
      d4: 23 2c 71 01  	sw	s7, 24(sp)
;         deeploy_log("  [\r\n  ");
      d8: 03 25 c1 01  	lw	a0, 28(sp)
      dc: 97 00 00 00  	auipc	ra, 0
      e0: e7 80 00 00  	jalr	ra
      e4: 93 04 00 00  	li	s1, 0
      e8: 13 8b 0b 00  	mv	s6, s7
      ec: 93 0d 0b 00  	mv	s11, s6
      f0: 13 0a 0d 00  	mv	s4, s10
;                       (uint16_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] +
      f4: 0b d5 2d 00  	<unknown>
      f8: 33 05 35 01  	add	a0, a0, s3
      fc: b3 55 05 10  	<unknown>
;           deeploy_log("%6hu ",
     100: 13 05 0c 00  	mv	a0, s8
     104: 97 00 00 00  	auipc	ra, 0
     108: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     10c: 13 0a fa ff  	addi	s4, s4, -1
     110: e3 12 0a fe  	bnez	s4, 0xf4 <PrintMatrix_u16_NCHW+0xf4>
;           deeploy_log("\r\n  ");
     114: 13 85 0c 00  	mv	a0, s9
     118: 97 00 00 00  	auipc	ra, 0
     11c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     120: 93 84 14 00  	addi	s1, s1, 1
     124: 33 0b 5b 01  	add	s6, s6, s5
     128: e3 92 24 fd  	bne	s1, s2, 0xec <PrintMatrix_u16_NCHW+0xec>
;         deeploy_log("]\r\n");
     12c: 03 25 41 02  	lw	a0, 36(sp)
     130: 97 00 00 00  	auipc	ra, 0
     134: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     138: 13 04 14 00  	addi	s0, s0, 1
     13c: 03 25 01 02  	lw	a0, 32(sp)
     140: b3 8b ab 00  	add	s7, s7, a0
     144: 03 25 81 02  	lw	a0, 40(sp)
     148: e3 18 a4 f8  	bne	s0, a0, 0xd8 <PrintMatrix_u16_NCHW+0xd8>
;       deeploy_log("]\r\n");
     14c: 03 25 41 02  	lw	a0, 36(sp)
     150: 97 00 00 00  	auipc	ra, 0
     154: e7 80 00 00  	jalr	ra
     158: 03 26 41 01  	lw	a2, 20(sp)
;   for (uint32_t n = 0; n < N; n++) {
     15c: 13 06 16 00  	addi	a2, a2, 1
     160: 83 2b 81 01  	lw	s7, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     164: 03 25 c1 00  	lw	a0, 12(sp)
     168: b3 8b ab 00  	add	s7, s7, a0
     16c: 83 2d 01 01  	lw	s11, 16(sp)
     170: e3 18 b6 f5  	bne	a2, s11, 0xc0 <PrintMatrix_u16_NCHW+0xc0>
     174: 6f 00 00 12  	j	0x294 <PrintMatrix_u16_NCHW+0x294>
     178: 37 05 00 00  	lui	a0, 0
     17c: 93 04 05 00  	mv	s1, a0
     180: 37 05 00 00  	lui	a0, 0
     184: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     188: 13 85 04 00  	mv	a0, s1
     18c: 97 00 00 00  	auipc	ra, 0
     190: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     194: 13 05 09 00  	mv	a0, s2
     198: 97 00 00 00  	auipc	ra, 0
     19c: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1a0: 93 8d fd ff  	addi	s11, s11, -1
     1a4: e3 92 0d fe  	bnez	s11, 0x188 <PrintMatrix_u16_NCHW+0x188>
     1a8: 6f 00 c0 0e  	j	0x294 <PrintMatrix_u16_NCHW+0x294>
     1ac: 13 04 00 00  	li	s0, 0
     1b0: 37 05 00 00  	lui	a0, 0
     1b4: 13 09 05 00  	mv	s2, a0
     1b8: 37 05 00 00  	lui	a0, 0
     1bc: 93 09 05 00  	mv	s3, a0
     1c0: 37 05 00 00  	lui	a0, 0
     1c4: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1c8: 13 05 09 00  	mv	a0, s2
     1cc: 97 00 00 00  	auipc	ra, 0
     1d0: e7 80 00 00  	jalr	ra
     1d4: 83 24 81 02  	lw	s1, 40(sp)
;         deeploy_log("  [\r\n  ");
     1d8: 13 85 09 00  	mv	a0, s3
     1dc: 97 00 00 00  	auipc	ra, 0
     1e0: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1e4: 13 05 0a 00  	mv	a0, s4
     1e8: 97 00 00 00  	auipc	ra, 0
     1ec: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1f0: 93 84 f4 ff  	addi	s1, s1, -1
     1f4: e3 92 04 fe  	bnez	s1, 0x1d8 <PrintMatrix_u16_NCHW+0x1d8>
;       deeploy_log("]\r\n");
     1f8: 13 05 0a 00  	mv	a0, s4
     1fc: 97 00 00 00  	auipc	ra, 0
     200: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     204: 13 04 14 00  	addi	s0, s0, 1
     208: e3 10 b4 fd  	bne	s0, s11, 0x1c8 <PrintMatrix_u16_NCHW+0x1c8>
     20c: 6f 00 80 08  	j	0x294 <PrintMatrix_u16_NCHW+0x294>
     210: 13 04 00 00  	li	s0, 0
     214: 37 05 00 00  	lui	a0, 0
     218: 93 09 05 00  	mv	s3, a0
     21c: 37 05 00 00  	lui	a0, 0
     220: 13 0a 05 00  	mv	s4, a0
     224: 37 05 00 00  	lui	a0, 0
     228: 93 0a 05 00  	mv	s5, a0
     22c: 37 05 00 00  	lui	a0, 0
     230: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     234: 13 85 09 00  	mv	a0, s3
     238: 97 00 00 00  	auipc	ra, 0
     23c: e7 80 00 00  	jalr	ra
     240: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     244: 13 05 0a 00  	mv	a0, s4
     248: 97 00 00 00  	auipc	ra, 0
     24c: e7 80 00 00  	jalr	ra
     250: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     254: 13 85 0a 00  	mv	a0, s5
     258: 97 00 00 00  	auipc	ra, 0
     25c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     260: 93 8b fb ff  	addi	s7, s7, -1
     264: e3 98 0b fe  	bnez	s7, 0x254 <PrintMatrix_u16_NCHW+0x254>
;         deeploy_log("]\r\n");
     268: 13 05 0b 00  	mv	a0, s6
     26c: 97 00 00 00  	auipc	ra, 0
     270: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     274: 93 84 14 00  	addi	s1, s1, 1
     278: 03 25 81 02  	lw	a0, 40(sp)
     27c: e3 94 a4 fc  	bne	s1, a0, 0x244 <PrintMatrix_u16_NCHW+0x244>
;       deeploy_log("]\r\n");
     280: 13 05 0b 00  	mv	a0, s6
     284: 97 00 00 00  	auipc	ra, 0
     288: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     28c: 13 04 14 00  	addi	s0, s0, 1
     290: e3 12 b4 fb  	bne	s0, s11, 0x234 <PrintMatrix_u16_NCHW+0x234>
; }
     294: 83 20 c1 05  	lw	ra, 92(sp)
     298: 03 24 81 05  	lw	s0, 88(sp)
     29c: 83 24 41 05  	lw	s1, 84(sp)
     2a0: 03 29 01 05  	lw	s2, 80(sp)
     2a4: 83 29 c1 04  	lw	s3, 76(sp)
     2a8: 03 2a 81 04  	lw	s4, 72(sp)
     2ac: 83 2a 41 04  	lw	s5, 68(sp)
     2b0: 03 2b 01 04  	lw	s6, 64(sp)
     2b4: 83 2b c1 03  	lw	s7, 60(sp)
     2b8: 03 2c 81 03  	lw	s8, 56(sp)
     2bc: 83 2c 41 03  	lw	s9, 52(sp)
     2c0: 03 2d 01 03  	lw	s10, 48(sp)
     2c4: 83 2d c1 02  	lw	s11, 44(sp)
     2c8: 13 01 01 06  	addi	sp, sp, 96
     2cc: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_u16_NHWC:

00000000 <PrintMatrix_u16_NHWC>:
;                           uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 22 c1 02  	sw	a2, 36(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 8e 05 24  	beqz	a1, 0x298 <PrintMatrix_u16_NHWC+0x298>
      40: 13 8d 05 00  	mv	s10, a1
      44: 93 04 05 00  	mv	s1, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 41 02  	lw	a0, 36(sp)
      4c: 63 0a 05 12  	beqz	a0, 0x180 <PrintMatrix_u16_NHWC+0x180>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 80 06 16  	beqz	a3, 0x1b4 <PrintMatrix_u16_NHWC+0x1b4>
      58: 13 04 07 00  	mv	s0, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 0e 07 1a  	beqz	a4, 0x218 <PrintMatrix_u16_NHWC+0x218>
      60: 93 89 07 00  	mv	s3, a5
      64: 13 06 00 00  	li	a2, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 05 24 03  	mul	a0, s0, s2
      6c: 83 25 41 02  	lw	a1, 36(sp)
      70: 33 05 b5 02  	mul	a0, a0, a1
      74: 13 15 15 00  	slli	a0, a0, 1
      78: 23 26 a1 00  	sw	a0, 12(sp)
      7c: 33 05 b4 02  	mul	a0, s0, a1
      80: 13 1b 15 00  	slli	s6, a0, 1
      84: 93 9a 15 00  	slli	s5, a1, 1
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 24 a1 00  	sw	a0, 8(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 05 05 00  	mv	a0, a0
      9c: 23 2e a1 00  	sw	a0, 28(sp)
      a0: 37 05 00 00  	lui	a0, 0
      a4: 13 0c 05 00  	mv	s8, a0
      a8: 37 05 00 00  	lui	a0, 0
      ac: 93 0c 05 00  	mv	s9, a0
      b0: 37 05 00 00  	lui	a0, 0
      b4: 13 05 05 00  	mv	a0, a0
      b8: 23 20 a1 02  	sw	a0, 32(sp)
      bc: 23 28 a1 01  	sw	s10, 16(sp)
      c0: 23 2a c1 00  	sw	a2, 20(sp)
;       deeploy_log("[\r\n");
      c4: 03 25 81 00  	lw	a0, 8(sp)
      c8: 97 00 00 00  	auipc	ra, 0
      cc: e7 80 00 00  	jalr	ra
      d0: 93 05 00 00  	li	a1, 0
      d4: 23 2c 91 00  	sw	s1, 24(sp)
      d8: 13 8d 04 00  	mv	s10, s1
      dc: 23 24 b1 02  	sw	a1, 40(sp)
;         deeploy_log("  [\r\n  ");
      e0: 03 25 c1 01  	lw	a0, 28(sp)
      e4: 97 00 00 00  	auipc	ra, 0
      e8: e7 80 00 00  	jalr	ra
      ec: 93 04 00 00  	li	s1, 0
      f0: 93 0b 0d 00  	mv	s7, s10
      f4: 93 8d 0b 00  	mv	s11, s7
      f8: 13 0a 04 00  	mv	s4, s0
;                       (uint16_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] +
      fc: 0b f5 5d 51  	<unknown>
     100: 33 05 35 01  	add	a0, a0, s3
     104: b3 55 05 10  	<unknown>
;           deeploy_log("%6hu ",
     108: 13 05 0c 00  	mv	a0, s8
     10c: 97 00 00 00  	auipc	ra, 0
     110: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     114: 13 0a fa ff  	addi	s4, s4, -1
     118: e3 12 0a fe  	bnez	s4, 0xfc <PrintMatrix_u16_NHWC+0xfc>
;           deeploy_log("\r\n  ");
     11c: 13 85 0c 00  	mv	a0, s9
     120: 97 00 00 00  	auipc	ra, 0
     124: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     128: 93 84 14 00  	addi	s1, s1, 1
     12c: b3 8b 6b 01  	add	s7, s7, s6
     130: e3 92 24 fd  	bne	s1, s2, 0xf4 <PrintMatrix_u16_NHWC+0xf4>
;         deeploy_log("]\r\n");
     134: 03 25 01 02  	lw	a0, 32(sp)
     138: 97 00 00 00  	auipc	ra, 0
     13c: e7 80 00 00  	jalr	ra
     140: 83 25 81 02  	lw	a1, 40(sp)
;     for (uint32_t c = 0; c < C; c++) {
     144: 93 85 15 00  	addi	a1, a1, 1
     148: 13 0d 2d 00  	addi	s10, s10, 2
     14c: 03 25 41 02  	lw	a0, 36(sp)
     150: e3 96 a5 f8  	bne	a1, a0, 0xdc <PrintMatrix_u16_NHWC+0xdc>
;       deeploy_log("]\r\n");
     154: 03 25 01 02  	lw	a0, 32(sp)
     158: 97 00 00 00  	auipc	ra, 0
     15c: e7 80 00 00  	jalr	ra
     160: 03 26 41 01  	lw	a2, 20(sp)
;   for (uint32_t n = 0; n < N; n++) {
     164: 13 06 16 00  	addi	a2, a2, 1
     168: 83 24 81 01  	lw	s1, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     16c: 03 25 c1 00  	lw	a0, 12(sp)
     170: b3 84 a4 00  	add	s1, s1, a0
     174: 03 2d 01 01  	lw	s10, 16(sp)
     178: e3 14 a6 f5  	bne	a2, s10, 0xc0 <PrintMatrix_u16_NHWC+0xc0>
     17c: 6f 00 c0 11  	j	0x298 <PrintMatrix_u16_NHWC+0x298>
     180: 37 05 00 00  	lui	a0, 0
     184: 93 04 05 00  	mv	s1, a0
     188: 37 05 00 00  	lui	a0, 0
     18c: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     190: 13 85 04 00  	mv	a0, s1
     194: 97 00 00 00  	auipc	ra, 0
     198: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     19c: 13 05 09 00  	mv	a0, s2
     1a0: 97 00 00 00  	auipc	ra, 0
     1a4: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1a8: 13 0d fd ff  	addi	s10, s10, -1
     1ac: e3 12 0d fe  	bnez	s10, 0x190 <PrintMatrix_u16_NHWC+0x190>
     1b0: 6f 00 80 0e  	j	0x298 <PrintMatrix_u16_NHWC+0x298>
     1b4: 13 04 00 00  	li	s0, 0
     1b8: 37 05 00 00  	lui	a0, 0
     1bc: 13 09 05 00  	mv	s2, a0
     1c0: 37 05 00 00  	lui	a0, 0
     1c4: 93 09 05 00  	mv	s3, a0
     1c8: 37 05 00 00  	lui	a0, 0
     1cc: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1d0: 13 05 09 00  	mv	a0, s2
     1d4: 97 00 00 00  	auipc	ra, 0
     1d8: e7 80 00 00  	jalr	ra
     1dc: 83 24 41 02  	lw	s1, 36(sp)
;         deeploy_log("  [\r\n  ");
     1e0: 13 85 09 00  	mv	a0, s3
     1e4: 97 00 00 00  	auipc	ra, 0
     1e8: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1ec: 13 05 0a 00  	mv	a0, s4
     1f0: 97 00 00 00  	auipc	ra, 0
     1f4: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1f8: 93 84 f4 ff  	addi	s1, s1, -1
     1fc: e3 92 04 fe  	bnez	s1, 0x1e0 <PrintMatrix_u16_NHWC+0x1e0>
;       deeploy_log("]\r\n");
     200: 13 05 0a 00  	mv	a0, s4
     204: 97 00 00 00  	auipc	ra, 0
     208: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     20c: 13 04 14 00  	addi	s0, s0, 1
     210: e3 10 a4 fd  	bne	s0, s10, 0x1d0 <PrintMatrix_u16_NHWC+0x1d0>
     214: 6f 00 40 08  	j	0x298 <PrintMatrix_u16_NHWC+0x298>
     218: 37 05 00 00  	lui	a0, 0
     21c: 93 09 05 00  	mv	s3, a0
     220: 37 05 00 00  	lui	a0, 0
     224: 13 0a 05 00  	mv	s4, a0
     228: 37 05 00 00  	lui	a0, 0
     22c: 93 0a 05 00  	mv	s5, a0
     230: 37 05 00 00  	lui	a0, 0
     234: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     238: 13 85 09 00  	mv	a0, s3
     23c: 97 00 00 00  	auipc	ra, 0
     240: e7 80 00 00  	jalr	ra
     244: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     248: 13 05 0a 00  	mv	a0, s4
     24c: 97 00 00 00  	auipc	ra, 0
     250: e7 80 00 00  	jalr	ra
     254: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     258: 13 85 0a 00  	mv	a0, s5
     25c: 97 00 00 00  	auipc	ra, 0
     260: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     264: 93 8b fb ff  	addi	s7, s7, -1
     268: e3 98 0b fe  	bnez	s7, 0x258 <PrintMatrix_u16_NHWC+0x258>
;         deeploy_log("]\r\n");
     26c: 13 05 0b 00  	mv	a0, s6
     270: 97 00 00 00  	auipc	ra, 0
     274: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     278: 93 84 14 00  	addi	s1, s1, 1
     27c: 03 25 41 02  	lw	a0, 36(sp)
     280: e3 94 a4 fc  	bne	s1, a0, 0x248 <PrintMatrix_u16_NHWC+0x248>
;       deeploy_log("]\r\n");
     284: 13 05 0b 00  	mv	a0, s6
     288: 97 00 00 00  	auipc	ra, 0
     28c: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     290: 13 04 14 00  	addi	s0, s0, 1
     294: e3 12 a4 fb  	bne	s0, s10, 0x238 <PrintMatrix_u16_NHWC+0x238>
; }
     298: 83 20 c1 05  	lw	ra, 92(sp)
     29c: 03 24 81 05  	lw	s0, 88(sp)
     2a0: 83 24 41 05  	lw	s1, 84(sp)
     2a4: 03 29 01 05  	lw	s2, 80(sp)
     2a8: 83 29 c1 04  	lw	s3, 76(sp)
     2ac: 03 2a 81 04  	lw	s4, 72(sp)
     2b0: 83 2a 41 04  	lw	s5, 68(sp)
     2b4: 03 2b 01 04  	lw	s6, 64(sp)
     2b8: 83 2b c1 03  	lw	s7, 60(sp)
     2bc: 03 2c 81 03  	lw	s8, 56(sp)
     2c0: 83 2c 41 03  	lw	s9, 52(sp)
     2c4: 03 2d 01 03  	lw	s10, 48(sp)
     2c8: 83 2d c1 02  	lw	s11, 44(sp)
     2cc: 13 01 01 06  	addi	sp, sp, 96
     2d0: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_u32_NCHW:

00000000 <PrintMatrix_u32_NCHW>:
;                           uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 24 c1 02  	sw	a2, 40(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 8a 05 24  	beqz	a1, 0x290 <PrintMatrix_u32_NCHW+0x290>
      40: 93 8d 05 00  	mv	s11, a1
      44: 93 0b 05 00  	mv	s7, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 81 02  	lw	a0, 40(sp)
      4c: 63 04 05 12  	beqz	a0, 0x174 <PrintMatrix_u32_NCHW+0x174>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 8a 06 14  	beqz	a3, 0x1a8 <PrintMatrix_u32_NCHW+0x1a8>
      58: 13 0d 07 00  	mv	s10, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 08 07 1a  	beqz	a4, 0x20c <PrintMatrix_u32_NCHW+0x20c>
      60: 93 89 07 00  	mv	s3, a5
      64: 13 06 00 00  	li	a2, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 05 2d 03  	mul	a0, s10, s2
      6c: 83 25 81 02  	lw	a1, 40(sp)
      70: b3 05 b5 02  	mul	a1, a0, a1
      74: 93 95 25 00  	slli	a1, a1, 2
      78: 23 26 b1 00  	sw	a1, 12(sp)
      7c: 13 15 25 00  	slli	a0, a0, 2
      80: 23 20 a1 02  	sw	a0, 32(sp)
      84: 93 1a 2d 00  	slli	s5, s10, 2
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 24 a1 00  	sw	a0, 8(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 05 05 00  	mv	a0, a0
      9c: 23 2e a1 00  	sw	a0, 28(sp)
      a0: 37 05 00 00  	lui	a0, 0
      a4: 13 0c 05 00  	mv	s8, a0
      a8: 37 05 00 00  	lui	a0, 0
      ac: 93 0c 05 00  	mv	s9, a0
      b0: 37 05 00 00  	lui	a0, 0
      b4: 13 05 05 00  	mv	a0, a0
      b8: 23 22 a1 02  	sw	a0, 36(sp)
      bc: 23 28 b1 01  	sw	s11, 16(sp)
      c0: 23 2a c1 00  	sw	a2, 20(sp)
;       deeploy_log("[\r\n");
      c4: 03 25 81 00  	lw	a0, 8(sp)
      c8: 97 00 00 00  	auipc	ra, 0
      cc: e7 80 00 00  	jalr	ra
      d0: 13 04 00 00  	li	s0, 0
      d4: 23 2c 71 01  	sw	s7, 24(sp)
;         deeploy_log("  [\r\n  ");
      d8: 03 25 c1 01  	lw	a0, 28(sp)
      dc: 97 00 00 00  	auipc	ra, 0
      e0: e7 80 00 00  	jalr	ra
      e4: 93 04 00 00  	li	s1, 0
      e8: 13 8b 0b 00  	mv	s6, s7
      ec: 93 0d 0b 00  	mv	s11, s6
      f0: 13 0a 0d 00  	mv	s4, s10
;                       (uint32_t)(pSrcA[n * C * H * W + c * H * W + h * W + w] +
      f4: 0b a5 4d 00  	<unknown>
      f8: b3 05 35 01  	add	a1, a0, s3
;           deeploy_log("%11" PRIu32 " ",
      fc: 13 05 0c 00  	mv	a0, s8
     100: 97 00 00 00  	auipc	ra, 0
     104: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     108: 13 0a fa ff  	addi	s4, s4, -1
     10c: e3 14 0a fe  	bnez	s4, 0xf4 <PrintMatrix_u32_NCHW+0xf4>
;           deeploy_log("\r\n  ");
     110: 13 85 0c 00  	mv	a0, s9
     114: 97 00 00 00  	auipc	ra, 0
     118: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     11c: 93 84 14 00  	addi	s1, s1, 1
     120: 33 0b 5b 01  	add	s6, s6, s5
     124: e3 94 24 fd  	bne	s1, s2, 0xec <PrintMatrix_u32_NCHW+0xec>
;         deeploy_log("]\r\n");
     128: 03 25 41 02  	lw	a0, 36(sp)
     12c: 97 00 00 00  	auipc	ra, 0
     130: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     134: 13 04 14 00  	addi	s0, s0, 1
     138: 03 25 01 02  	lw	a0, 32(sp)
     13c: b3 8b ab 00  	add	s7, s7, a0
     140: 03 25 81 02  	lw	a0, 40(sp)
     144: e3 1a a4 f8  	bne	s0, a0, 0xd8 <PrintMatrix_u32_NCHW+0xd8>
;       deeploy_log("]\r\n");
     148: 03 25 41 02  	lw	a0, 36(sp)
     14c: 97 00 00 00  	auipc	ra, 0
     150: e7 80 00 00  	jalr	ra
     154: 03 26 41 01  	lw	a2, 20(sp)
;   for (uint32_t n = 0; n < N; n++) {
     158: 13 06 16 00  	addi	a2, a2, 1
     15c: 83 2b 81 01  	lw	s7, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     160: 03 25 c1 00  	lw	a0, 12(sp)
     164: b3 8b ab 00  	add	s7, s7, a0
     168: 83 2d 01 01  	lw	s11, 16(sp)
     16c: e3 1a b6 f5  	bne	a2, s11, 0xc0 <PrintMatrix_u32_NCHW+0xc0>
     170: 6f 00 00 12  	j	0x290 <PrintMatrix_u32_NCHW+0x290>
     174: 37 05 00 00  	lui	a0, 0
     178: 93 04 05 00  	mv	s1, a0
     17c: 37 05 00 00  	lui	a0, 0
     180: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     184: 13 85 04 00  	mv	a0, s1
     188: 97 00 00 00  	auipc	ra, 0
     18c: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     190: 13 05 09 00  	mv	a0, s2
     194: 97 00 00 00  	auipc	ra, 0
     198: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     19c: 93 8d fd ff  	addi	s11, s11, -1
     1a0: e3 92 0d fe  	bnez	s11, 0x184 <PrintMatrix_u32_NCHW+0x184>
     1a4: 6f 00 c0 0e  	j	0x290 <PrintMatrix_u32_NCHW+0x290>
     1a8: 13 04 00 00  	li	s0, 0
     1ac: 37 05 00 00  	lui	a0, 0
     1b0: 13 09 05 00  	mv	s2, a0
     1b4: 37 05 00 00  	lui	a0, 0
     1b8: 93 09 05 00  	mv	s3, a0
     1bc: 37 05 00 00  	lui	a0, 0
     1c0: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1c4: 13 05 09 00  	mv	a0, s2
     1c8: 97 00 00 00  	auipc	ra, 0
     1cc: e7 80 00 00  	jalr	ra
     1d0: 83 24 81 02  	lw	s1, 40(sp)
;         deeploy_log("  [\r\n  ");
     1d4: 13 85 09 00  	mv	a0, s3
     1d8: 97 00 00 00  	auipc	ra, 0
     1dc: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1e0: 13 05 0a 00  	mv	a0, s4
     1e4: 97 00 00 00  	auipc	ra, 0
     1e8: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1ec: 93 84 f4 ff  	addi	s1, s1, -1
     1f0: e3 92 04 fe  	bnez	s1, 0x1d4 <PrintMatrix_u32_NCHW+0x1d4>
;       deeploy_log("]\r\n");
     1f4: 13 05 0a 00  	mv	a0, s4
     1f8: 97 00 00 00  	auipc	ra, 0
     1fc: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     200: 13 04 14 00  	addi	s0, s0, 1
     204: e3 10 b4 fd  	bne	s0, s11, 0x1c4 <PrintMatrix_u32_NCHW+0x1c4>
     208: 6f 00 80 08  	j	0x290 <PrintMatrix_u32_NCHW+0x290>
     20c: 13 04 00 00  	li	s0, 0
     210: 37 05 00 00  	lui	a0, 0
     214: 93 09 05 00  	mv	s3, a0
     218: 37 05 00 00  	lui	a0, 0
     21c: 13 0a 05 00  	mv	s4, a0
     220: 37 05 00 00  	lui	a0, 0
     224: 93 0a 05 00  	mv	s5, a0
     228: 37 05 00 00  	lui	a0, 0
     22c: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     230: 13 85 09 00  	mv	a0, s3
     234: 97 00 00 00  	auipc	ra, 0
     238: e7 80 00 00  	jalr	ra
     23c: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     240: 13 05 0a 00  	mv	a0, s4
     244: 97 00 00 00  	auipc	ra, 0
     248: e7 80 00 00  	jalr	ra
     24c: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     250: 13 85 0a 00  	mv	a0, s5
     254: 97 00 00 00  	auipc	ra, 0
     258: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     25c: 93 8b fb ff  	addi	s7, s7, -1
     260: e3 98 0b fe  	bnez	s7, 0x250 <PrintMatrix_u32_NCHW+0x250>
;         deeploy_log("]\r\n");
     264: 13 05 0b 00  	mv	a0, s6
     268: 97 00 00 00  	auipc	ra, 0
     26c: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     270: 93 84 14 00  	addi	s1, s1, 1
     274: 03 25 81 02  	lw	a0, 40(sp)
     278: e3 94 a4 fc  	bne	s1, a0, 0x240 <PrintMatrix_u32_NCHW+0x240>
;       deeploy_log("]\r\n");
     27c: 13 05 0b 00  	mv	a0, s6
     280: 97 00 00 00  	auipc	ra, 0
     284: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     288: 13 04 14 00  	addi	s0, s0, 1
     28c: e3 12 b4 fb  	bne	s0, s11, 0x230 <PrintMatrix_u32_NCHW+0x230>
; }
     290: 83 20 c1 05  	lw	ra, 92(sp)
     294: 03 24 81 05  	lw	s0, 88(sp)
     298: 83 24 41 05  	lw	s1, 84(sp)
     29c: 03 29 01 05  	lw	s2, 80(sp)
     2a0: 83 29 c1 04  	lw	s3, 76(sp)
     2a4: 03 2a 81 04  	lw	s4, 72(sp)
     2a8: 83 2a 41 04  	lw	s5, 68(sp)
     2ac: 03 2b 01 04  	lw	s6, 64(sp)
     2b0: 83 2b c1 03  	lw	s7, 60(sp)
     2b4: 03 2c 81 03  	lw	s8, 56(sp)
     2b8: 83 2c 41 03  	lw	s9, 52(sp)
     2bc: 03 2d 01 03  	lw	s10, 48(sp)
     2c0: 83 2d c1 02  	lw	s11, 44(sp)
     2c4: 13 01 01 06  	addi	sp, sp, 96
     2c8: 67 80 00 00  	ret

Disassembly of section .text.PrintMatrix_u32_NHWC:

00000000 <PrintMatrix_u32_NHWC>:
;                           uint32_t C, uint32_t H, uint32_t W, uint32_t offset) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 23 2c 81 04  	sw	s0, 88(sp)
       c: 23 2a 91 04  	sw	s1, 84(sp)
      10: 23 28 21 05  	sw	s2, 80(sp)
      14: 23 26 31 05  	sw	s3, 76(sp)
      18: 23 24 41 05  	sw	s4, 72(sp)
      1c: 23 22 51 05  	sw	s5, 68(sp)
      20: 23 20 61 05  	sw	s6, 64(sp)
      24: 23 2e 71 03  	sw	s7, 60(sp)
      28: 23 2c 81 03  	sw	s8, 56(sp)
      2c: 23 2a 91 03  	sw	s9, 52(sp)
      30: 23 28 a1 03  	sw	s10, 48(sp)
      34: 23 26 b1 03  	sw	s11, 44(sp)
      38: 23 22 c1 02  	sw	a2, 36(sp)
;   for (uint32_t n = 0; n < N; n++) {
      3c: 63 8c 05 24  	beqz	a1, 0x294 <PrintMatrix_u32_NHWC+0x294>
      40: 13 8d 05 00  	mv	s10, a1
      44: 93 04 05 00  	mv	s1, a0
;     for (uint32_t c = 0; c < C; c++) {
      48: 03 25 41 02  	lw	a0, 36(sp)
      4c: 63 08 05 12  	beqz	a0, 0x17c <PrintMatrix_u32_NHWC+0x17c>
      50: 13 89 06 00  	mv	s2, a3
;       for (uint32_t h = 0; h < H; h++) {
      54: 63 8e 06 14  	beqz	a3, 0x1b0 <PrintMatrix_u32_NHWC+0x1b0>
      58: 13 04 07 00  	mv	s0, a4
;         for (uint32_t w = 0; w < W; w++) {
      5c: 63 0c 07 1a  	beqz	a4, 0x214 <PrintMatrix_u32_NHWC+0x214>
      60: 93 89 07 00  	mv	s3, a5
      64: 13 06 00 00  	li	a2, 0
;   for (uint32_t n = 0; n < N; n++) {
      68: 33 05 24 03  	mul	a0, s0, s2
      6c: 83 25 41 02  	lw	a1, 36(sp)
      70: 33 05 b5 02  	mul	a0, a0, a1
      74: 13 15 25 00  	slli	a0, a0, 2
      78: 23 26 a1 00  	sw	a0, 12(sp)
      7c: 33 05 b4 02  	mul	a0, s0, a1
      80: 13 1b 25 00  	slli	s6, a0, 2
      84: 93 9a 25 00  	slli	s5, a1, 2
      88: 37 05 00 00  	lui	a0, 0
      8c: 13 05 05 00  	mv	a0, a0
      90: 23 24 a1 00  	sw	a0, 8(sp)
      94: 37 05 00 00  	lui	a0, 0
      98: 13 05 05 00  	mv	a0, a0
      9c: 23 2e a1 00  	sw	a0, 28(sp)
      a0: 37 05 00 00  	lui	a0, 0
      a4: 13 0c 05 00  	mv	s8, a0
      a8: 37 05 00 00  	lui	a0, 0
      ac: 93 0c 05 00  	mv	s9, a0
      b0: 37 05 00 00  	lui	a0, 0
      b4: 13 05 05 00  	mv	a0, a0
      b8: 23 20 a1 02  	sw	a0, 32(sp)
      bc: 23 28 a1 01  	sw	s10, 16(sp)
      c0: 23 2a c1 00  	sw	a2, 20(sp)
;       deeploy_log("[\r\n");
      c4: 03 25 81 00  	lw	a0, 8(sp)
      c8: 97 00 00 00  	auipc	ra, 0
      cc: e7 80 00 00  	jalr	ra
      d0: 93 05 00 00  	li	a1, 0
      d4: 23 2c 91 00  	sw	s1, 24(sp)
      d8: 13 8d 04 00  	mv	s10, s1
      dc: 23 24 b1 02  	sw	a1, 40(sp)
;         deeploy_log("  [\r\n  ");
      e0: 03 25 c1 01  	lw	a0, 28(sp)
      e4: 97 00 00 00  	auipc	ra, 0
      e8: e7 80 00 00  	jalr	ra
      ec: 93 04 00 00  	li	s1, 0
      f0: 93 0b 0d 00  	mv	s7, s10
      f4: 93 8d 0b 00  	mv	s11, s7
      f8: 13 0a 04 00  	mv	s4, s0
;                       (uint32_t)(pSrcA[n * C * H * W + h * C * W + w * C + c] +
      fc: 0b f5 5d 21  	<unknown>
     100: b3 05 35 01  	add	a1, a0, s3
;           deeploy_log("%11" PRIu32 " ",
     104: 13 05 0c 00  	mv	a0, s8
     108: 97 00 00 00  	auipc	ra, 0
     10c: e7 80 00 00  	jalr	ra
;         for (uint32_t w = 0; w < W; w++) {
     110: 13 0a fa ff  	addi	s4, s4, -1
     114: e3 14 0a fe  	bnez	s4, 0xfc <PrintMatrix_u32_NHWC+0xfc>
;           deeploy_log("\r\n  ");
     118: 13 85 0c 00  	mv	a0, s9
     11c: 97 00 00 00  	auipc	ra, 0
     120: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     124: 93 84 14 00  	addi	s1, s1, 1
     128: b3 8b 6b 01  	add	s7, s7, s6
     12c: e3 94 24 fd  	bne	s1, s2, 0xf4 <PrintMatrix_u32_NHWC+0xf4>
;         deeploy_log("]\r\n");
     130: 03 25 01 02  	lw	a0, 32(sp)
     134: 97 00 00 00  	auipc	ra, 0
     138: e7 80 00 00  	jalr	ra
     13c: 83 25 81 02  	lw	a1, 40(sp)
;     for (uint32_t c = 0; c < C; c++) {
     140: 93 85 15 00  	addi	a1, a1, 1
     144: 13 0d 4d 00  	addi	s10, s10, 4
     148: 03 25 41 02  	lw	a0, 36(sp)
     14c: e3 98 a5 f8  	bne	a1, a0, 0xdc <PrintMatrix_u32_NHWC+0xdc>
;       deeploy_log("]\r\n");
     150: 03 25 01 02  	lw	a0, 32(sp)
     154: 97 00 00 00  	auipc	ra, 0
     158: e7 80 00 00  	jalr	ra
     15c: 03 26 41 01  	lw	a2, 20(sp)
;   for (uint32_t n = 0; n < N; n++) {
     160: 13 06 16 00  	addi	a2, a2, 1
     164: 83 24 81 01  	lw	s1, 24(sp)
;   for (uint32_t n = 0; n < N; n++) {
     168: 03 25 c1 00  	lw	a0, 12(sp)
     16c: b3 84 a4 00  	add	s1, s1, a0
     170: 03 2d 01 01  	lw	s10, 16(sp)
     174: e3 16 a6 f5  	bne	a2, s10, 0xc0 <PrintMatrix_u32_NHWC+0xc0>
     178: 6f 00 c0 11  	j	0x294 <PrintMatrix_u32_NHWC+0x294>
     17c: 37 05 00 00  	lui	a0, 0
     180: 93 04 05 00  	mv	s1, a0
     184: 37 05 00 00  	lui	a0, 0
     188: 13 09 05 00  	mv	s2, a0
;       deeploy_log("[\r\n");
     18c: 13 85 04 00  	mv	a0, s1
     190: 97 00 00 00  	auipc	ra, 0
     194: e7 80 00 00  	jalr	ra
;       deeploy_log("]\r\n");
     198: 13 05 09 00  	mv	a0, s2
     19c: 97 00 00 00  	auipc	ra, 0
     1a0: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     1a4: 13 0d fd ff  	addi	s10, s10, -1
     1a8: e3 12 0d fe  	bnez	s10, 0x18c <PrintMatrix_u32_NHWC+0x18c>
     1ac: 6f 00 80 0e  	j	0x294 <PrintMatrix_u32_NHWC+0x294>
     1b0: 13 04 00 00  	li	s0, 0
     1b4: 37 05 00 00  	lui	a0, 0
     1b8: 13 09 05 00  	mv	s2, a0
     1bc: 37 05 00 00  	lui	a0, 0
     1c0: 93 09 05 00  	mv	s3, a0
     1c4: 37 05 00 00  	lui	a0, 0
     1c8: 13 0a 05 00  	mv	s4, a0
;       deeploy_log("[\r\n");
     1cc: 13 05 09 00  	mv	a0, s2
     1d0: 97 00 00 00  	auipc	ra, 0
     1d4: e7 80 00 00  	jalr	ra
     1d8: 83 24 41 02  	lw	s1, 36(sp)
;         deeploy_log("  [\r\n  ");
     1dc: 13 85 09 00  	mv	a0, s3
     1e0: 97 00 00 00  	auipc	ra, 0
     1e4: e7 80 00 00  	jalr	ra
;         deeploy_log("]\r\n");
     1e8: 13 05 0a 00  	mv	a0, s4
     1ec: 97 00 00 00  	auipc	ra, 0
     1f0: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     1f4: 93 84 f4 ff  	addi	s1, s1, -1
     1f8: e3 92 04 fe  	bnez	s1, 0x1dc <PrintMatrix_u32_NHWC+0x1dc>
;       deeploy_log("]\r\n");
     1fc: 13 05 0a 00  	mv	a0, s4
     200: 97 00 00 00  	auipc	ra, 0
     204: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     208: 13 04 14 00  	addi	s0, s0, 1
     20c: e3 10 a4 fd  	bne	s0, s10, 0x1cc <PrintMatrix_u32_NHWC+0x1cc>
     210: 6f 00 40 08  	j	0x294 <PrintMatrix_u32_NHWC+0x294>
     214: 37 05 00 00  	lui	a0, 0
     218: 93 09 05 00  	mv	s3, a0
     21c: 37 05 00 00  	lui	a0, 0
     220: 13 0a 05 00  	mv	s4, a0
     224: 37 05 00 00  	lui	a0, 0
     228: 93 0a 05 00  	mv	s5, a0
     22c: 37 05 00 00  	lui	a0, 0
     230: 13 0b 05 00  	mv	s6, a0
;       deeploy_log("[\r\n");
     234: 13 85 09 00  	mv	a0, s3
     238: 97 00 00 00  	auipc	ra, 0
     23c: e7 80 00 00  	jalr	ra
     240: 93 04 00 00  	li	s1, 0
;         deeploy_log("  [\r\n  ");
     244: 13 05 0a 00  	mv	a0, s4
     248: 97 00 00 00  	auipc	ra, 0
     24c: e7 80 00 00  	jalr	ra
     250: 93 0b 09 00  	mv	s7, s2
;           deeploy_log("\r\n  ");
     254: 13 85 0a 00  	mv	a0, s5
     258: 97 00 00 00  	auipc	ra, 0
     25c: e7 80 00 00  	jalr	ra
;       for (uint32_t h = 0; h < H; h++) {
     260: 93 8b fb ff  	addi	s7, s7, -1
     264: e3 98 0b fe  	bnez	s7, 0x254 <PrintMatrix_u32_NHWC+0x254>
;         deeploy_log("]\r\n");
     268: 13 05 0b 00  	mv	a0, s6
     26c: 97 00 00 00  	auipc	ra, 0
     270: e7 80 00 00  	jalr	ra
;     for (uint32_t c = 0; c < C; c++) {
     274: 93 84 14 00  	addi	s1, s1, 1
     278: 03 25 41 02  	lw	a0, 36(sp)
     27c: e3 94 a4 fc  	bne	s1, a0, 0x244 <PrintMatrix_u32_NHWC+0x244>
;       deeploy_log("]\r\n");
     280: 13 05 0b 00  	mv	a0, s6
     284: 97 00 00 00  	auipc	ra, 0
     288: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
     28c: 13 04 14 00  	addi	s0, s0, 1
     290: e3 12 a4 fb  	bne	s0, s10, 0x234 <PrintMatrix_u32_NHWC+0x234>
; }
     294: 83 20 c1 05  	lw	ra, 92(sp)
     298: 03 24 81 05  	lw	s0, 88(sp)
     29c: 83 24 41 05  	lw	s1, 84(sp)
     2a0: 03 29 01 05  	lw	s2, 80(sp)
     2a4: 83 29 c1 04  	lw	s3, 76(sp)
     2a8: 03 2a 81 04  	lw	s4, 72(sp)
     2ac: 83 2a 41 04  	lw	s5, 68(sp)
     2b0: 03 2b 01 04  	lw	s6, 64(sp)
     2b4: 83 2b c1 03  	lw	s7, 60(sp)
     2b8: 03 2c 81 03  	lw	s8, 56(sp)
     2bc: 83 2c 41 03  	lw	s9, 52(sp)
     2c0: 03 2d 01 03  	lw	s10, 48(sp)
     2c4: 83 2d c1 02  	lw	s11, 44(sp)
     2c8: 13 01 01 06  	addi	sp, sp, 96
     2cc: 67 80 00 00  	ret

Disassembly of section .text.PrintArray_u8:

00000000 <PrintArray_u8>:
;                    uint32_t offset) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 23 26 31 01  	sw	s3, 12(sp)
;   for (uint32_t n = 0; n < N; n++) {
      18: 63 8c 05 02  	beqz	a1, 0x50 <PrintArray_u8+0x50>
      1c: 13 04 06 00  	mv	s0, a2
      20: 93 84 05 00  	mv	s1, a1
      24: 13 09 05 00  	mv	s2, a0
      28: 37 05 00 00  	lui	a0, 0
      2c: 93 09 05 00  	mv	s3, a0
;     deeploy_log("%4u ", (uint8_t)(pSrcA[n] + offset));
      30: 0b 45 19 00  	<unknown>
      34: 33 05 85 00  	add	a0, a0, s0
      38: b3 75 05 10  	<unknown>
      3c: 13 85 09 00  	mv	a0, s3
      40: 97 00 00 00  	auipc	ra, 0
      44: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
      48: 93 84 f4 ff  	addi	s1, s1, -1
      4c: e3 92 04 fe  	bnez	s1, 0x30 <PrintArray_u8+0x30>
;   deeploy_log("\r\n");
      50: 37 05 00 00  	lui	a0, 0
      54: 13 05 05 00  	mv	a0, a0
      58: 83 20 c1 01  	lw	ra, 28(sp)
      5c: 03 24 81 01  	lw	s0, 24(sp)
      60: 83 24 41 01  	lw	s1, 20(sp)
      64: 03 29 01 01  	lw	s2, 16(sp)
      68: 83 29 c1 00  	lw	s3, 12(sp)
      6c: 13 01 01 02  	addi	sp, sp, 32
      70: 17 03 00 00  	auipc	t1, 0
      74: 67 00 03 00  	jr	t1

Disassembly of section .text.PrintArray_u16:

00000000 <PrintArray_u16>:
;                     uint32_t offset) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 23 26 31 01  	sw	s3, 12(sp)
;   for (uint32_t n = 0; n < N; n++) {
      18: 63 8c 05 02  	beqz	a1, 0x50 <PrintArray_u16+0x50>
      1c: 13 04 06 00  	mv	s0, a2
      20: 93 84 05 00  	mv	s1, a1
      24: 13 09 05 00  	mv	s2, a0
      28: 37 05 00 00  	lui	a0, 0
      2c: 93 09 05 00  	mv	s3, a0
;     deeploy_log("%6hu ", (uint16_t)(pSrcA[n] + offset));
      30: 0b 55 29 00  	<unknown>
      34: 33 05 85 00  	add	a0, a0, s0
      38: b3 55 05 10  	<unknown>
      3c: 13 85 09 00  	mv	a0, s3
      40: 97 00 00 00  	auipc	ra, 0
      44: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
      48: 93 84 f4 ff  	addi	s1, s1, -1
      4c: e3 92 04 fe  	bnez	s1, 0x30 <PrintArray_u16+0x30>
;   deeploy_log("\r\n");
      50: 37 05 00 00  	lui	a0, 0
      54: 13 05 05 00  	mv	a0, a0
      58: 83 20 c1 01  	lw	ra, 28(sp)
      5c: 03 24 81 01  	lw	s0, 24(sp)
      60: 83 24 41 01  	lw	s1, 20(sp)
      64: 03 29 01 01  	lw	s2, 16(sp)
      68: 83 29 c1 00  	lw	s3, 12(sp)
      6c: 13 01 01 02  	addi	sp, sp, 32
      70: 17 03 00 00  	auipc	t1, 0
      74: 67 00 03 00  	jr	t1

Disassembly of section .text.PrintArray_u32:

00000000 <PrintArray_u32>:
;                     uint32_t offset) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 23 26 31 01  	sw	s3, 12(sp)
;   for (uint32_t n = 0; n < N; n++) {
      18: 63 8a 05 02  	beqz	a1, 0x4c <PrintArray_u32+0x4c>
      1c: 13 04 06 00  	mv	s0, a2
      20: 93 84 05 00  	mv	s1, a1
      24: 13 09 05 00  	mv	s2, a0
      28: 37 05 00 00  	lui	a0, 0
      2c: 93 09 05 00  	mv	s3, a0
;     deeploy_log("%11" PRIu32 " ", (uint32_t)(pSrcA[n] + offset));
      30: 0b 25 49 00  	<unknown>
      34: b3 05 85 00  	add	a1, a0, s0
      38: 13 85 09 00  	mv	a0, s3
      3c: 97 00 00 00  	auipc	ra, 0
      40: e7 80 00 00  	jalr	ra
;   for (uint32_t n = 0; n < N; n++) {
      44: 93 84 f4 ff  	addi	s1, s1, -1
      48: e3 94 04 fe  	bnez	s1, 0x30 <PrintArray_u32+0x30>
;   deeploy_log("\r\n");
      4c: 37 05 00 00  	lui	a0, 0
      50: 13 05 05 00  	mv	a0, a0
      54: 83 20 c1 01  	lw	ra, 28(sp)
      58: 03 24 81 01  	lw	s0, 24(sp)
      5c: 83 24 41 01  	lw	s1, 20(sp)
      60: 03 29 01 01  	lw	s2, 16(sp)
      64: 83 29 c1 00  	lw	s3, 12(sp)
      68: 13 01 01 02  	addi	sp, sp, 32
      6c: 17 03 00 00  	auipc	t1, 0
      70: 67 00 03 00  	jr	t1

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploybasic.a(iRMSNorm_s8.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                    Size     VMA      Type
  0                         00000000 00000000 
  1 .strtab                 00000141 00000000 
  2 .text                   00000000 00000000 TEXT
  3 .text._plp_sqrt_q32     00000058 00000000 TEXT
  4 .text.iRMSnorm_s8_s8    00000138 00000000 TEXT
  5 .debug_loclists         000002e0 00000000 DEBUG
  6 .debug_abbrev           000000f4 00000000 DEBUG
  7 .debug_info             000001f0 00000000 DEBUG
  8 .rela.debug_info        00000090 00000000 
  9 .debug_rnglists         00000047 00000000 DEBUG
 10 .debug_str_offsets      000000a8 00000000 DEBUG
 11 .rela.debug_str_offsets 000001e0 00000000 
 12 .debug_str              00000231 00000000 DEBUG
 13 .debug_addr             00000014 00000000 DEBUG
 14 .rela.debug_addr        00000024 00000000 
 15 .comment                00000073 00000000 
 16 .note.GNU-stack         00000000 00000000 
 17 .riscv.attributes       00000030 00000000 
 18 .debug_frame            00000040 00000000 DEBUG
 19 .rela.debug_frame       00000090 00000000 
 20 .debug_line             00000264 00000000 DEBUG
 21 .rela.debug_line        00000594 00000000 
 22 .debug_line_str         00000143 00000000 DEBUG
 23 .llvm_addrsig           00000000 00000000 
 24 .symtab                 00000770 00000000 

Disassembly of section .text._plp_sqrt_q32:

00000000 <_plp_sqrt_q32>:
;   int32_t number = *pSrc;
       0: 03 25 05 00  	lw	a0, 0(a0)
;   if (number > 0) {
       4: 63 56 a0 04  	blez	a0, 0x50 <_plp_sqrt_q32+0x50>
       8: 93 06 00 00  	li	a3, 0
       c: 93 07 00 00  	li	a5, 0
      10: 37 b7 00 00  	lui	a4, 11
      14: 13 08 67 50  	addi	a6, a4, 1286
      18: 6f 00 00 01  	j	0x28 <_plp_sqrt_q32+0x28>
      1c: 13 08 f7 ff  	addi	a6, a4, -1
      20: 13 87 06 00  	mv	a4, a3
;     while (start <= end) {
      24: 63 42 f8 02  	blt	a6, a5, 0x48 <_plp_sqrt_q32+0x48>
;       mid = (start + end) >> 1;
      28: 5b 27 f8 02  	<unknown>
;       if (((mid * mid) >> fracBits) == number) {
      2c: b3 08 e7 02  	mul	a7, a4, a4
      30: b3 d8 b8 00  	srl	a7, a7, a1
      34: 63 8a a8 00  	beq	a7, a0, 0x48 <_plp_sqrt_q32+0x48>
;       if (((mid * mid) >> fracBits) < number) {
      38: e3 d2 a8 fe  	bge	a7, a0, 0x1c <_plp_sqrt_q32+0x1c>
      3c: 93 07 17 00  	addi	a5, a4, 1
      40: 93 06 07 00  	mv	a3, a4
;     while (start <= end) {
      44: e3 52 f8 fe  	bge	a6, a5, 0x28 <_plp_sqrt_q32+0x28>
      48: 23 20 e6 00  	sw	a4, 0(a2)
; }
      4c: 67 80 00 00  	ret
      50: 23 20 06 00  	sw	zero, 0(a2)
; }
      54: 67 80 00 00  	ret

Disassembly of section .text.iRMSnorm_s8_s8:

00000000 <iRMSnorm_s8_s8>:
;                     int32_t log2D) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
       8: 23 24 91 00  	sw	s1, 8(sp)
       c: 23 22 21 01  	sw	s2, 4(sp)
      10: 33 47 f7 02  	div	a4, a4, a5
;   for (int i = 0; i < (size / lastDimLength); i++) {
      14: 63 5c e0 0c  	blez	a4, 0xec <iRMSnorm_s8_s8+0xec>
;     for (int j = 0; j < lastDimLength; j++) {
      18: 63 54 f0 0e  	blez	a5, 0x100 <iRMSnorm_s8_s8+0x100>
      1c: 93 08 00 00  	li	a7, 0
      20: b7 b2 00 00  	lui	t0, 11
      24: 93 82 62 50  	addi	t0, t0, 1286
      28: 13 03 00 f8  	li	t1, -128
      2c: 93 03 f0 07  	li	t2, 127
      30: fb 40 c7 05  	<unknown>
      34: 93 0e 00 00  	li	t4, 0
      38: 13 0e 00 00  	li	t3, 0
;     for (int j = 0; j < lastDimLength; j++) {
      3c: 7b c0 a7 00  	<unknown>
;       temp = (int16_t)(data_in[j + i * lastDimLength] + input_offset);
      40: 33 0f d5 01  	add	t5, a0, t4
      44: 03 0f 0f 00  	lb	t5, 0(t5)
      48: 33 0f df 00  	add	t5, t5, a3
;     for (int j = 0; j < lastDimLength; j++) {
      4c: 93 8e 1e 00  	addi	t4, t4, 1
;       sum += temp * temp;
      50: 5b 1e ef 81  	<unknown>
;     sum = sum / lastDimLength;
      54: b3 4e fe 02  	div	t4, t3, a5
;   if (number > 0) {
      58: 63 c0 0e 04  	bltz	t4, 0x98 <iRMSnorm_s8_s8+0x98>
      5c: 13 0e 00 00  	li	t3, 0
      60: 93 0f 00 00  	li	t6, 0
      64: 13 8f 1e 00  	addi	t5, t4, 1
      68: 13 84 02 00  	mv	s0, t0
      6c: 6f 00 00 01  	j	0x7c <iRMSnorm_s8_s8+0x7c>
      70: 93 8f 14 00  	addi	t6, s1, 1
      74: 13 8e 04 00  	mv	t3, s1
;     while (start <= end) {
      78: 63 46 f4 03  	blt	s0, t6, 0xa4 <iRMSnorm_s8_s8+0xa4>
;       mid = (start + end) >> 1;
      7c: db a4 8f 02  	<unknown>
;       if (((mid * mid) >> fracBits) == number) {
      80: 33 89 94 02  	mul	s2, s1, s1
      84: 63 0e e9 01  	beq	s2, t5, 0xa0 <iRMSnorm_s8_s8+0xa0>
;       if (((mid * mid) >> fracBits) < number) {
      88: e3 d4 2e ff  	bge	t4, s2, 0x70 <iRMSnorm_s8_s8+0x70>
      8c: 13 84 f4 ff  	addi	s0, s1, -1
;     while (start <= end) {
      90: e3 56 f4 ff  	bge	s0, t6, 0x7c <iRMSnorm_s8_s8+0x7c>
      94: 6f 00 00 01  	j	0xa4 <iRMSnorm_s8_s8+0xa4>
      98: 13 0e 00 00  	li	t3, 0
      9c: 6f 00 80 00  	j	0xa4 <iRMSnorm_s8_s8+0xa4>
      a0: 13 8e 04 00  	mv	t3, s1
      a4: 93 0e 00 00  	li	t4, 0
      a8: 13 0f 06 00  	mv	t5, a2
      ac: 7b c0 87 01  	<unknown>
;           ((((((int32_t)data_in[j + i * lastDimLength]) + input_offset) *
      b0: b3 0f d5 01  	add	t6, a0, t4
      b4: 83 8f 0f 00  	lb	t6, 0(t6)
;              weight[j]) /
      b8: 0b 24 4f 00  	<unknown>
;           ((((((int32_t)data_in[j + i * lastDimLength]) + input_offset) *
      bc: b3 8f df 00  	add	t6, t6, a3
      c0: b3 8f 8f 02  	mul	t6, t6, s0
;              weight[j]) /
      c4: b3 cf cf 03  	div	t6, t6, t3
;             (std)) >>
      c8: b3 df 0f 41  	sra	t6, t6, a6
;       data_out[j + i * lastDimLength] = (int8_t)CLAMP(intermediate, -128, 127);
      cc: b3 ef 6f 04  	<unknown>
      d0: b3 cf 7f 04  	<unknown>
      d4: 33 84 d5 01  	add	s0, a1, t4
;     for (int j = 0; j < lastDimLength; j++) {
      d8: 93 8e 1e 00  	addi	t4, t4, 1
;       data_out[j + i * lastDimLength] = (int8_t)CLAMP(intermediate, -128, 127);
      dc: 23 00 f4 01  	sb	t6, 0(s0)
;   for (int i = 0; i < (size / lastDimLength); i++) {
      e0: 93 88 18 00  	addi	a7, a7, 1
      e4: 33 05 f5 00  	add	a0, a0, a5
      e8: b3 85 f5 00  	add	a1, a1, a5
; }
      ec: 03 24 c1 00  	lw	s0, 12(sp)
      f0: 83 24 81 00  	lw	s1, 8(sp)
      f4: 03 29 41 00  	lw	s2, 4(sp)
      f8: 13 01 01 01  	addi	sp, sp, 16
      fc: 67 80 00 00  	ret
     100: 13 05 00 00  	li	a0, 0
     104: b7 b5 00 00  	lui	a1, 11
     108: 13 86 65 50  	addi	a2, a1, 1286
     10c: 93 05 10 00  	li	a1, 1
     110: 6f 00 c0 00  	j	0x11c <iRMSnorm_s8_s8+0x11c>
     114: 13 85 16 00  	addi	a0, a3, 1
;     while (start <= end) {
     118: e3 4a a6 fc  	blt	a2, a0, 0xec <iRMSnorm_s8_s8+0xec>
;       mid = (start + end) >> 1;
     11c: db 26 c5 02  	<unknown>
;       if (((mid * mid) >> fracBits) == number) {
     120: 33 87 d6 02  	mul	a4, a3, a3
     124: e3 04 b7 fc  	beq	a4, a1, 0xec <iRMSnorm_s8_s8+0xec>
     128: e3 06 07 fe  	beqz	a4, 0x114 <iRMSnorm_s8_s8+0x114>
     12c: 13 86 f6 ff  	addi	a2, a3, -1
;     while (start <= end) {
     130: e3 56 a6 fe  	bge	a2, a0, 0x11c <iRMSnorm_s8_s8+0x11c>
     134: 6f f0 9f fb  	j	0xec <iRMSnorm_s8_s8+0xec>
