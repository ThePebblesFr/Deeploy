
/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(Convolution_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                        Size     VMA      Type
  0                                             00000000 00000000 
  1 .strtab                                     0000016e 00000000 
  2 .text                                       00000000 00000000 TEXT
  3 .text.PULP_Conv2d_fp32_fp32_fp32_HWC        0000073c 00000000 TEXT
  4 .text.PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC 00000aa4 00000000 TEXT
  5 .debug_loclists                             00003c6e 00000000 DEBUG
  6 .debug_abbrev                               000000d6 00000000 DEBUG
  7 .debug_info                                 00000617 00000000 DEBUG
  8 .rela.debug_info                            000001c8 00000000 
  9 .debug_rnglists                             0000040d 00000000 DEBUG
 10 .debug_str_offsets                          00000144 00000000 DEBUG
 11 .rela.debug_str_offsets                     000003b4 00000000 
 12 .debug_str                                  000003ba 00000000 DEBUG
 13 .debug_addr                                 00000038 00000000 DEBUG
 14 .rela.debug_addr                            00000090 00000000 
 15 .comment                                    00000073 00000000 
 16 .note.GNU-stack                             00000000 00000000 
 17 .riscv.attributes                           00000030 00000000 
 18 .debug_frame                                00000074 00000000 DEBUG
 19 .rela.debug_frame                           000000c0 00000000 
 20 .debug_line                                 0000104f 00000000 DEBUG
 21 .rela.debug_line                            00002bf8 00000000 
 22 .debug_line_str                             000001f7 00000000 DEBUG
 23 .llvm_addrsig                               00000000 00000000 
 24 .symtab                                     00002480 00000000 

Disassembly of section .text.PULP_Conv2d_fp32_fp32_fp32_HWC:

00000000 <PULP_Conv2d_fp32_fp32_fp32_HWC>:
;     uint32_t pad_left, uint32_t pad_right, int nb_dedicated_cores) {
       0: 13 01 01 f8  	addi	sp, sp, -128
       4: 23 2e 11 06  	sw	ra, 124(sp)
       8: 23 2c 81 06  	sw	s0, 120(sp)
       c: 23 2a 91 06  	sw	s1, 116(sp)
      10: 23 28 21 07  	sw	s2, 112(sp)
      14: 23 26 31 07  	sw	s3, 108(sp)
      18: 23 24 41 07  	sw	s4, 104(sp)
      1c: 23 22 51 07  	sw	s5, 100(sp)
      20: 23 20 61 07  	sw	s6, 96(sp)
      24: 23 2e 71 05  	sw	s7, 92(sp)
      28: 23 2c 81 05  	sw	s8, 88(sp)
      2c: 23 2a 91 05  	sw	s9, 84(sp)
      30: 23 28 a1 05  	sw	s10, 80(sp)
      34: 23 26 b1 05  	sw	s11, 76(sp)
      38: 83 22 41 0a  	lw	t0, 164(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      3c: 73 23 40 f1  	csrr	t1, mhartid
;   return hart_id & 0x01f;
      40: 13 73 f3 01  	andi	t1, t1, 31
;   core_id = core_id % nb_dedicated_cores;
      44: b3 63 53 02  	rem	t2, t1, t0
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      48: 33 93 02 10  	<unknown>
;       (F_total >> log2Core) + ((F_total & (nb_dedicated_cores - 1)) != 0);
      4c: 33 63 03 10  	<unknown>
      50: 33 d3 67 00  	srl	t1, a5, t1
      54: 93 82 f2 ff  	addi	t0, t0, -1
      58: b3 f2 f2 00  	and	t0, t0, a5
      5c: b3 32 50 00  	snez	t0, t0
      60: b3 02 53 00  	add	t0, t1, t0
;   uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
      64: 33 d3 02 10  	<unknown>
      68: b3 02 73 02  	mul	t0, t1, t2
      6c: b3 d3 f2 04  	<unknown>
;   uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
      70: b3 d2 03 10  	<unknown>
      74: 33 83 62 00  	add	t1, t0, t1
      78: 33 53 f3 04  	<unknown>
;   uint16_t ch_out_count = ch_out_stop - ch_out_start;
      7c: 33 03 73 40  	sub	t1, t1, t2
;   if (ch_out_count == 0) {
      80: 33 53 03 10  	<unknown>
      84: 23 24 61 04  	sw	t1, 72(sp)
;   if (ch_out_count == 0) {
      88: 63 0c 03 66  	beqz	t1, 0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
      8c: 93 8f 06 00  	mv	t6, a3
      90: 23 22 f1 04  	sw	a5, 68(sp)
      94: 83 27 01 0a  	lw	a5, 160(sp)
      98: 03 23 c1 09  	lw	t1, 156(sp)
      9c: 03 2e 81 09  	lw	t3, 152(sp)
      a0: 03 24 41 09  	lw	s0, 148(sp)
      a4: 83 23 01 09  	lw	t2, 144(sp)
      a8: 83 2e c1 08  	lw	t4, 140(sp)
      ac: 83 24 41 08  	lw	s1, 132(sp)
      b0: 03 29 01 08  	lw	s2, 128(sp)
;   const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;
      b4: b3 86 d8 02  	mul	a3, a7, a3
      b8: 33 8f 06 03  	mul	t5, a3, a6
      bc: 33 0f 5f 02  	mul	t5, t5, t0
      c0: 13 1f 2f 00  	slli	t5, t5, 2
      c4: 33 07 e7 01  	add	a4, a4, t5
;   uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
      c8: 23 2a e1 02  	sw	a4, 52(sp)
      cc: 33 87 05 41  	sub	a4, a1, a6
      d0: 23 22 81 02  	sw	s0, 36(sp)
      d4: 33 07 87 00  	add	a4, a4, s0
      d8: 33 07 c7 01  	add	a4, a4, t3
      dc: 23 20 21 03  	sw	s2, 32(sp)
      e0: 33 5e 27 03  	divu	t3, a4, s2
;   uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
      e4: 33 07 16 41  	sub	a4, a2, a7
      e8: 23 20 61 04  	sw	t1, 64(sp)
      ec: 33 07 67 00  	add	a4, a4, t1
      f0: 33 07 f7 00  	add	a4, a4, a5
      f4: 23 2e 91 02  	sw	s1, 60(sp)
      f8: 33 5f 97 02  	divu	t5, a4, s1
;   if (has_bias) {
      fc: 13 f7 1e 00  	andi	a4, t4, 1
;   uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
     100: 13 04 1f 00  	addi	s0, t5, 1
     104: 23 2e c1 01  	sw	t3, 28(sp)
     108: 23 2c e1 03  	sw	t5, 56(sp)
     10c: 23 2c 81 00  	sw	s0, 24(sp)
;   if (has_bias) {
     110: 63 00 07 1e  	beqz	a4, 0x2f0 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x2f0>
     114: 13 07 f0 ff  	li	a4, -1
     118: 83 2e 41 04  	lw	t4, 68(sp)
;     for (uint32_t h = 0; h < H_out; ++h) {
     11c: 63 02 ee 5e  	beq	t3, a4, 0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
;       for (uint32_t w = 0; w < W_out; ++w) {
     120: 63 00 04 5e  	beqz	s0, 0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
     124: 83 29 81 08  	lw	s3, 136(sp)
;           for (uint32_t p = 0; p < P; ++p) {
     128: 63 08 08 38  	beqz	a6, 0x4b8 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x4b8>
;             for (uint32_t q = 0; q < Q; ++q) {
     12c: 63 88 08 44  	beqz	a7, 0x57c <PULP_Conv2d_fp32_fp32_fp32_HWC+0x57c>
;               for (uint32_t c = 0; c < C; ++c) {
     130: 63 88 0f 50  	beqz	t6, 0x640 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x640>
     134: 13 07 00 00  	li	a4, 0
;     for (uint32_t h = 0; h < H_out; ++h) {
     138: b3 87 08 03  	mul	a5, a7, a6
     13c: b3 87 f7 03  	mul	a5, a5, t6
     140: 13 9a 27 00  	slli	s4, a5, 2
     144: 93 9a 26 00  	slli	s5, a3, 2
     148: 13 9b 2f 00  	slli	s6, t6, 2
     14c: 83 26 01 04  	lw	a3, 64(sp)
     150: 83 27 41 02  	lw	a5, 36(sp)
     154: b3 86 c7 42  	<unknown>
     158: b3 06 d0 40  	neg	a3, a3
     15c: b3 86 df 02  	mul	a3, t6, a3
     160: 93 96 26 00  	slli	a3, a3, 2
     164: 33 0e d5 00  	add	t3, a0, a3
     168: 03 25 01 02  	lw	a0, 32(sp)
     16c: 33 05 f5 03  	mul	a0, a0, t6
     170: 33 05 c5 02  	mul	a0, a0, a2
     174: 13 15 25 00  	slli	a0, a0, 2
     178: 23 26 a1 00  	sw	a0, 12(sp)
     17c: 03 25 c1 03  	lw	a0, 60(sp)
     180: 33 05 f5 03  	mul	a0, a0, t6
     184: 13 15 25 00  	slli	a0, a0, 2
     188: 23 24 a1 02  	sw	a0, 40(sp)
     18c: 33 85 cf 02  	mul	a0, t6, a2
     190: 93 1c 25 00  	slli	s9, a0, 2
     194: 53 00 00 f0  	fmv.w.x	ft0, zero
     198: 13 05 00 00  	li	a0, 0
     19c: b3 06 87 02  	mul	a3, a4, s0
     1a0: 23 26 d1 02  	sw	a3, 44(sp)
     1a4: 83 26 01 02  	lw	a3, 32(sp)
     1a8: 23 28 e1 00  	sw	a4, 16(sp)
     1ac: b3 06 d7 02  	mul	a3, a4, a3
     1b0: 03 27 41 02  	lw	a4, 36(sp)
     1b4: b3 80 e6 40  	sub	ra, a3, a4
     1b8: 23 2a c1 01  	sw	t3, 20(sp)
     1bc: 93 04 00 00  	li	s1, 0
     1c0: 93 06 05 00  	mv	a3, a0
     1c4: 03 25 c1 02  	lw	a0, 44(sp)
     1c8: 33 85 a6 00  	add	a0, a3, a0
     1cc: b3 0e d5 03  	mul	t4, a0, t4
     1d0: 03 25 c1 03  	lw	a0, 60(sp)
     1d4: 23 28 d1 02  	sw	a3, 48(sp)
     1d8: 33 85 a6 02  	mul	a0, a3, a0
     1dc: 83 26 01 04  	lw	a3, 64(sp)
     1e0: 33 07 d5 40  	sub	a4, a0, a3
     1e4: 83 27 41 03  	lw	a5, 52(sp)
     1e8: 13 0f 00 00  	li	t5, 0
     1ec: 13 0c 0e 00  	mv	s8, t3
     1f0: 93 8d 07 00  	mv	s11, a5
     1f4: d3 00 00 20  	fmv.s	ft1, ft0
     1f8: fb 40 08 04  	<unknown>
     1fc: 33 85 e0 01  	add	a0, ra, t5
     200: 93 26 05 00  	slti	a3, a0, 0
     204: 33 25 b5 00  	slt	a0, a0, a1
     208: 13 45 15 00  	xori	a0, a0, 1
;                 if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
     20c: 33 e5 a6 00  	or	a0, a3, a0
     210: 63 10 05 06  	bnez	a0, 0x270 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x270>
     214: 13 0d 00 00  	li	s10, 0
     218: 93 0b 0c 00  	mv	s7, s8
     21c: 13 84 0d 00  	mv	s0, s11
     220: 7b c0 68 02  	<unknown>
     224: 33 05 a7 01  	add	a0, a4, s10
     228: 93 26 05 00  	slti	a3, a0, 0
     22c: 33 25 c5 00  	slt	a0, a0, a2
     230: 13 45 15 00  	xori	a0, a0, 1
;                 if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
     234: 33 e3 a6 00  	or	t1, a3, a0
     238: 13 85 0b 00  	mv	a0, s7
     23c: 13 09 04 00  	mv	s2, s0
     240: 93 86 0f 00  	mv	a3, t6
     244: 63 10 03 02  	bnez	t1, 0x264 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x264>
;                 sum += pSrcA[input_idx] * weight_ptr[weight_idx];
     248: 07 21 05 00  	flw	ft2, 0(a0)
     24c: 87 21 09 00  	flw	ft3, 0(s2)
     250: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;               for (uint32_t c = 0; c < C; ++c) {
     254: 93 86 f6 ff  	addi	a3, a3, -1
     258: 13 09 49 00  	addi	s2, s2, 4
     25c: 13 05 45 00  	addi	a0, a0, 4
     260: e3 94 06 fe  	bnez	a3, 0x248 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x248>
;             for (uint32_t q = 0; q < Q; ++q) {
     264: 13 0d 1d 00  	addi	s10, s10, 1
     268: 33 04 64 01  	add	s0, s0, s6
     26c: b3 8b 6b 01  	add	s7, s7, s6
;           for (uint32_t p = 0; p < P; ++p) {
     270: 13 0f 1f 00  	addi	t5, t5, 1
     274: b3 8d 5d 01  	add	s11, s11, s5
     278: 33 0c 9c 01  	add	s8, s8, s9
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     27c: 33 85 54 00  	add	a0, s1, t0
;           pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
     280: 93 16 25 00  	slli	a3, a0, 2
     284: b3 86 d9 00  	add	a3, s3, a3
     288: 07 a1 06 00  	flw	ft2, 0(a3)
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     28c: 33 05 d5 01  	add	a0, a0, t4
;           pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
     290: d3 70 11 00  	fadd.s	ft1, ft2, ft1
     294: 13 15 25 00  	slli	a0, a0, 2
     298: 33 85 a3 00  	add	a0, t2, a0
     29c: 27 20 15 00  	fsw	ft1, 0(a0)
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     2a0: 93 84 14 00  	addi	s1, s1, 1
     2a4: b3 87 47 01  	add	a5, a5, s4
     2a8: 03 25 81 04  	lw	a0, 72(sp)
     2ac: e3 9e a4 f2  	bne	s1, a0, 0x1e8 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x1e8>
     2b0: 03 27 01 03  	lw	a4, 48(sp)
;       for (uint32_t w = 0; w < W_out; ++w) {
     2b4: 13 05 17 00  	addi	a0, a4, 1
     2b8: 83 26 81 02  	lw	a3, 40(sp)
     2bc: 33 0e de 00  	add	t3, t3, a3
     2c0: 83 2e 41 04  	lw	t4, 68(sp)
     2c4: 83 26 81 03  	lw	a3, 56(sp)
     2c8: e3 1a d7 ee  	bne	a4, a3, 0x1bc <PULP_Conv2d_fp32_fp32_fp32_HWC+0x1bc>
     2cc: 83 26 01 01  	lw	a3, 16(sp)
;     for (uint32_t h = 0; h < H_out; ++h) {
     2d0: 13 87 16 00  	addi	a4, a3, 1
     2d4: 03 2e 41 01  	lw	t3, 20(sp)
;     for (uint32_t h = 0; h < H_out; ++h) {
     2d8: 03 25 c1 00  	lw	a0, 12(sp)
     2dc: 33 0e ae 00  	add	t3, t3, a0
     2e0: 03 25 c1 01  	lw	a0, 28(sp)
     2e4: 03 24 81 01  	lw	s0, 24(sp)
     2e8: e3 98 a6 ea  	bne	a3, a0, 0x198 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x198>
     2ec: 6f 00 40 41  	j	0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
     2f0: 13 07 f0 ff  	li	a4, -1
     2f4: 03 23 41 04  	lw	t1, 68(sp)
;     for (uint32_t h = 0; h < H_out; ++h) {
     2f8: 63 04 ee 40  	beq	t3, a4, 0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
;       for (uint32_t w = 0; w < W_out; ++w) {
     2fc: 63 02 04 40  	beqz	s0, 0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
;           for (uint32_t p = 0; p < P; ++p) {
     300: 63 02 08 22  	beqz	a6, 0x524 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x524>
;             for (uint32_t q = 0; q < Q; ++q) {
     304: 63 82 08 2e  	beqz	a7, 0x5e8 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x5e8>
;               for (uint32_t c = 0; c < C; ++c) {
     308: 63 82 0f 3a  	beqz	t6, 0x6ac <PULP_Conv2d_fp32_fp32_fp32_HWC+0x6ac>
     30c: 13 07 00 00  	li	a4, 0
;     for (uint32_t h = 0; h < H_out; ++h) {
     310: b3 87 08 03  	mul	a5, a7, a6
     314: b3 87 f7 03  	mul	a5, a5, t6
     318: 93 99 27 00  	slli	s3, a5, 2
     31c: 13 9a 26 00  	slli	s4, a3, 2
     320: 93 9a 2f 00  	slli	s5, t6, 2
     324: 83 26 01 04  	lw	a3, 64(sp)
     328: 83 27 41 02  	lw	a5, 36(sp)
     32c: b3 86 c7 42  	<unknown>
     330: b3 06 d0 40  	neg	a3, a3
     334: b3 86 df 02  	mul	a3, t6, a3
     338: 93 96 26 00  	slli	a3, a3, 2
     33c: 33 0d d5 00  	add	s10, a0, a3
     340: 03 25 01 02  	lw	a0, 32(sp)
     344: 33 05 f5 03  	mul	a0, a0, t6
     348: 33 05 c5 02  	mul	a0, a0, a2
     34c: 13 15 25 00  	slli	a0, a0, 2
     350: 23 28 a1 00  	sw	a0, 16(sp)
     354: 03 25 c1 03  	lw	a0, 60(sp)
     358: 33 05 f5 03  	mul	a0, a0, t6
     35c: 13 15 25 00  	slli	a0, a0, 2
     360: 23 26 a1 02  	sw	a0, 44(sp)
     364: 33 85 cf 02  	mul	a0, t6, a2
     368: 13 1c 25 00  	slli	s8, a0, 2
     36c: 53 00 00 f0  	fmv.w.x	ft0, zero
     370: 13 05 00 00  	li	a0, 0
     374: b3 06 87 02  	mul	a3, a4, s0
     378: 23 28 d1 02  	sw	a3, 48(sp)
     37c: 83 26 01 02  	lw	a3, 32(sp)
     380: 23 2a e1 00  	sw	a4, 20(sp)
     384: b3 06 d7 02  	mul	a3, a4, a3
     388: 03 27 41 02  	lw	a4, 36(sp)
     38c: b3 8d e6 40  	sub	s11, a3, a4
     390: 23 24 a1 03  	sw	s10, 40(sp)
     394: 13 07 00 00  	li	a4, 0
     398: 93 06 05 00  	mv	a3, a0
     39c: 03 25 01 03  	lw	a0, 48(sp)
     3a0: 33 85 a6 00  	add	a0, a3, a0
     3a4: 93 8e 02 00  	mv	t4, t0
     3a8: b3 0e 65 42  	<unknown>
     3ac: 03 25 c1 03  	lw	a0, 60(sp)
     3b0: 13 83 06 00  	mv	t1, a3
     3b4: 33 85 a6 02  	mul	a0, a3, a0
     3b8: 83 26 01 04  	lw	a3, 64(sp)
     3bc: b3 04 d5 40  	sub	s1, a0, a3
     3c0: 83 2b 41 03  	lw	s7, 52(sp)
     3c4: 13 0b 00 00  	li	s6, 0
     3c8: 93 0c 0d 00  	mv	s9, s10
     3cc: 13 8f 0b 00  	mv	t5, s7
     3d0: d3 00 00 20  	fmv.s	ft1, ft0
     3d4: fb 40 08 04  	<unknown>
     3d8: 33 85 6d 01  	add	a0, s11, s6
     3dc: 93 27 05 00  	slti	a5, a0, 0
     3e0: 33 25 b5 00  	slt	a0, a0, a1
     3e4: 13 45 15 00  	xori	a0, a0, 1
;                 if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
     3e8: 33 e5 a7 00  	or	a0, a5, a0
     3ec: 63 10 05 06  	bnez	a0, 0x44c <PULP_Conv2d_fp32_fp32_fp32_HWC+0x44c>
     3f0: 13 04 00 00  	li	s0, 0
     3f4: 13 85 0c 00  	mv	a0, s9
     3f8: 13 09 0f 00  	mv	s2, t5
     3fc: 7b c0 68 02  	<unknown>
     400: b3 87 84 00  	add	a5, s1, s0
     404: 13 ae 07 00  	slti	t3, a5, 0
     408: b3 a7 c7 00  	slt	a5, a5, a2
     40c: 93 c7 17 00  	xori	a5, a5, 1
;                 if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
     410: b3 66 fe 00  	or	a3, t3, a5
     414: 93 00 05 00  	mv	ra, a0
     418: 13 0e 09 00  	mv	t3, s2
     41c: 93 87 0f 00  	mv	a5, t6
     420: 63 90 06 02  	bnez	a3, 0x440 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x440>
;                 sum += pSrcA[input_idx] * weight_ptr[weight_idx];
     424: 07 a1 00 00  	flw	ft2, 0(ra)
     428: 87 21 0e 00  	flw	ft3, 0(t3)
     42c: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;               for (uint32_t c = 0; c < C; ++c) {
     430: 93 87 f7 ff  	addi	a5, a5, -1
     434: 13 0e 4e 00  	addi	t3, t3, 4
     438: 93 80 40 00  	addi	ra, ra, 4
     43c: e3 94 07 fe  	bnez	a5, 0x424 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x424>
;             for (uint32_t q = 0; q < Q; ++q) {
     440: 13 04 14 00  	addi	s0, s0, 1
     444: 33 09 59 01  	add	s2, s2, s5
     448: 33 05 55 01  	add	a0, a0, s5
;           for (uint32_t p = 0; p < P; ++p) {
     44c: 13 0b 1b 00  	addi	s6, s6, 1
     450: 33 0f 4f 01  	add	t5, t5, s4
     454: b3 8c 8c 01  	add	s9, s9, s8
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     458: 33 85 ee 00  	add	a0, t4, a4
;           pDstC[output_idx] = sum;
     45c: 13 15 25 00  	slli	a0, a0, 2
     460: 33 85 a3 00  	add	a0, t2, a0
     464: 27 20 15 00  	fsw	ft1, 0(a0)
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     468: 13 07 17 00  	addi	a4, a4, 1
     46c: b3 8b 3b 01  	add	s7, s7, s3
     470: 03 25 81 04  	lw	a0, 72(sp)
     474: e3 18 a7 f4  	bne	a4, a0, 0x3c4 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x3c4>
     478: 13 07 03 00  	mv	a4, t1
;       for (uint32_t w = 0; w < W_out; ++w) {
     47c: 13 05 13 00  	addi	a0, t1, 1
     480: 83 26 c1 02  	lw	a3, 44(sp)
     484: 33 0d dd 00  	add	s10, s10, a3
     488: 03 23 41 04  	lw	t1, 68(sp)
     48c: 83 26 81 03  	lw	a3, 56(sp)
     490: e3 12 d7 f0  	bne	a4, a3, 0x394 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x394>
     494: 83 26 41 01  	lw	a3, 20(sp)
;     for (uint32_t h = 0; h < H_out; ++h) {
     498: 13 87 16 00  	addi	a4, a3, 1
     49c: 03 2d 81 02  	lw	s10, 40(sp)
;     for (uint32_t h = 0; h < H_out; ++h) {
     4a0: 03 25 01 01  	lw	a0, 16(sp)
     4a4: 33 0d ad 00  	add	s10, s10, a0
     4a8: 03 25 c1 01  	lw	a0, 28(sp)
     4ac: 03 24 81 01  	lw	s0, 24(sp)
     4b0: e3 90 a6 ec  	bne	a3, a0, 0x370 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x370>
     4b4: 6f 00 c0 24  	j	0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
     4b8: 13 06 00 00  	li	a2, 0
;     for (uint32_t h = 0; h < H_out; ++h) {
     4bc: 13 95 22 00  	slli	a0, t0, 2
     4c0: 33 85 a9 00  	add	a0, s3, a0
     4c4: 93 07 00 00  	li	a5, 0
     4c8: 93 05 06 00  	mv	a1, a2
     4cc: 33 06 86 02  	mul	a2, a2, s0
     4d0: 13 07 00 00  	li	a4, 0
     4d4: 93 86 07 00  	mv	a3, a5
     4d8: b3 87 c7 00  	add	a5, a5, a2
     4dc: b3 87 d7 03  	mul	a5, a5, t4
     4e0: 13 08 05 00  	mv	a6, a0
     4e4: 03 23 81 04  	lw	t1, 72(sp)
     4e8: 93 08 03 00  	mv	a7, t1
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     4ec: 7b 40 03 01  	<unknown>
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     4f0: 33 03 57 00  	add	t1, a4, t0
;           pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
     4f4: 07 20 08 00  	flw	ft0, 0(a6)
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     4f8: 33 03 f3 00  	add	t1, t1, a5
;           pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
     4fc: 13 13 23 00  	slli	t1, t1, 2
     500: 33 83 63 00  	add	t1, t2, t1
     504: 27 20 03 00  	fsw	ft0, 0(t1)
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     508: 13 07 17 00  	addi	a4, a4, 1
     50c: 13 08 48 00  	addi	a6, a6, 4
;       for (uint32_t w = 0; w < W_out; ++w) {
     510: 93 87 16 00  	addi	a5, a3, 1
     514: e3 9e e6 fb  	bne	a3, t5, 0x4d0 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x4d0>
;     for (uint32_t h = 0; h < H_out; ++h) {
     518: 13 86 15 00  	addi	a2, a1, 1
     51c: e3 94 c5 fb  	bne	a1, t3, 0x4c4 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x4c4>
     520: 6f 00 00 1e  	j	0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
     524: 93 05 00 00  	li	a1, 0
     528: 13 07 00 00  	li	a4, 0
     52c: 13 85 05 00  	mv	a0, a1
     530: b3 85 85 02  	mul	a1, a1, s0
     534: 93 06 00 00  	li	a3, 0
     538: 13 06 07 00  	mv	a2, a4
     53c: b3 07 b7 00  	add	a5, a4, a1
     540: 13 87 02 00  	mv	a4, t0
     544: 33 87 67 42  	<unknown>
     548: 03 28 81 04  	lw	a6, 72(sp)
     54c: 93 07 08 00  	mv	a5, a6
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     550: 7b 40 a8 00  	<unknown>
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     554: 33 08 d7 00  	add	a6, a4, a3
;           pDstC[output_idx] = sum;
     558: 13 18 28 00  	slli	a6, a6, 2
     55c: 33 88 03 01  	add	a6, t2, a6
     560: 23 20 08 00  	sw	zero, 0(a6)
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     564: 93 86 16 00  	addi	a3, a3, 1
;       for (uint32_t w = 0; w < W_out; ++w) {
     568: 13 07 16 00  	addi	a4, a2, 1
     56c: e3 14 e6 fd  	bne	a2, t5, 0x534 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x534>
;     for (uint32_t h = 0; h < H_out; ++h) {
     570: 93 05 15 00  	addi	a1, a0, 1
     574: e3 1a c5 fb  	bne	a0, t3, 0x528 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x528>
     578: 6f 00 80 18  	j	0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
     57c: 13 06 00 00  	li	a2, 0
;     for (uint32_t h = 0; h < H_out; ++h) {
     580: 13 95 22 00  	slli	a0, t0, 2
     584: 33 85 a9 00  	add	a0, s3, a0
     588: 93 07 00 00  	li	a5, 0
     58c: 93 05 06 00  	mv	a1, a2
     590: 33 06 86 02  	mul	a2, a2, s0
     594: 13 07 00 00  	li	a4, 0
     598: 93 86 07 00  	mv	a3, a5
     59c: b3 87 c7 00  	add	a5, a5, a2
     5a0: b3 87 d7 03  	mul	a5, a5, t4
     5a4: 13 08 05 00  	mv	a6, a0
     5a8: 03 23 81 04  	lw	t1, 72(sp)
     5ac: 93 08 03 00  	mv	a7, t1
     5b0: 7b 40 03 01  	<unknown>
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     5b4: 33 03 57 00  	add	t1, a4, t0
;           pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
     5b8: 07 20 08 00  	flw	ft0, 0(a6)
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     5bc: 33 03 f3 00  	add	t1, t1, a5
;           pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
     5c0: 13 13 23 00  	slli	t1, t1, 2
     5c4: 33 83 63 00  	add	t1, t2, t1
     5c8: 27 20 03 00  	fsw	ft0, 0(t1)
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     5cc: 13 07 17 00  	addi	a4, a4, 1
     5d0: 13 08 48 00  	addi	a6, a6, 4
;       for (uint32_t w = 0; w < W_out; ++w) {
     5d4: 93 87 16 00  	addi	a5, a3, 1
     5d8: e3 9e e6 fb  	bne	a3, t5, 0x594 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x594>
;     for (uint32_t h = 0; h < H_out; ++h) {
     5dc: 13 86 15 00  	addi	a2, a1, 1
     5e0: e3 94 c5 fb  	bne	a1, t3, 0x588 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x588>
     5e4: 6f 00 c0 11  	j	0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
     5e8: 93 05 00 00  	li	a1, 0
     5ec: 13 07 00 00  	li	a4, 0
     5f0: 13 85 05 00  	mv	a0, a1
     5f4: b3 85 85 02  	mul	a1, a1, s0
     5f8: 93 06 00 00  	li	a3, 0
     5fc: 13 06 07 00  	mv	a2, a4
     600: b3 07 b7 00  	add	a5, a4, a1
     604: 13 87 02 00  	mv	a4, t0
     608: 33 87 67 42  	<unknown>
     60c: 03 28 81 04  	lw	a6, 72(sp)
     610: 93 07 08 00  	mv	a5, a6
     614: 7b 40 a8 00  	<unknown>
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     618: 33 08 d7 00  	add	a6, a4, a3
;           pDstC[output_idx] = sum;
     61c: 13 18 28 00  	slli	a6, a6, 2
     620: 33 88 03 01  	add	a6, t2, a6
     624: 23 20 08 00  	sw	zero, 0(a6)
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     628: 93 86 16 00  	addi	a3, a3, 1
;       for (uint32_t w = 0; w < W_out; ++w) {
     62c: 13 07 16 00  	addi	a4, a2, 1
     630: e3 14 e6 fd  	bne	a2, t5, 0x5f8 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x5f8>
;     for (uint32_t h = 0; h < H_out; ++h) {
     634: 93 05 15 00  	addi	a1, a0, 1
     638: e3 1a c5 fb  	bne	a0, t3, 0x5ec <PULP_Conv2d_fp32_fp32_fp32_HWC+0x5ec>
     63c: 6f 00 40 0c  	j	0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
     640: 13 06 00 00  	li	a2, 0
;     for (uint32_t h = 0; h < H_out; ++h) {
     644: 13 95 22 00  	slli	a0, t0, 2
     648: 33 85 a9 00  	add	a0, s3, a0
     64c: 93 07 00 00  	li	a5, 0
     650: 93 05 06 00  	mv	a1, a2
     654: 33 06 86 02  	mul	a2, a2, s0
     658: 13 07 00 00  	li	a4, 0
     65c: 93 86 07 00  	mv	a3, a5
     660: b3 87 c7 00  	add	a5, a5, a2
     664: b3 87 d7 03  	mul	a5, a5, t4
     668: 13 08 05 00  	mv	a6, a0
     66c: 03 23 81 04  	lw	t1, 72(sp)
     670: 93 08 03 00  	mv	a7, t1
     674: 7b 40 03 01  	<unknown>
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     678: 33 03 57 00  	add	t1, a4, t0
;           pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
     67c: 07 20 08 00  	flw	ft0, 0(a6)
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     680: 33 03 f3 00  	add	t1, t1, a5
;           pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
     684: 13 13 23 00  	slli	t1, t1, 2
     688: 33 83 63 00  	add	t1, t2, t1
     68c: 27 20 03 00  	fsw	ft0, 0(t1)
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     690: 13 07 17 00  	addi	a4, a4, 1
     694: 13 08 48 00  	addi	a6, a6, 4
;       for (uint32_t w = 0; w < W_out; ++w) {
     698: 93 87 16 00  	addi	a5, a3, 1
     69c: e3 9e e6 fb  	bne	a3, t5, 0x658 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x658>
;     for (uint32_t h = 0; h < H_out; ++h) {
     6a0: 13 86 15 00  	addi	a2, a1, 1
     6a4: e3 94 c5 fb  	bne	a1, t3, 0x64c <PULP_Conv2d_fp32_fp32_fp32_HWC+0x64c>
     6a8: 6f 00 80 05  	j	0x700 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x700>
     6ac: 93 05 00 00  	li	a1, 0
     6b0: 13 07 00 00  	li	a4, 0
     6b4: 13 85 05 00  	mv	a0, a1
     6b8: b3 85 85 02  	mul	a1, a1, s0
     6bc: 93 06 00 00  	li	a3, 0
     6c0: 13 06 07 00  	mv	a2, a4
     6c4: b3 07 b7 00  	add	a5, a4, a1
     6c8: 13 87 02 00  	mv	a4, t0
     6cc: 33 87 67 42  	<unknown>
     6d0: 03 28 81 04  	lw	a6, 72(sp)
     6d4: 93 07 08 00  	mv	a5, a6
     6d8: 7b 40 a8 00  	<unknown>
;           uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
     6dc: 33 08 d7 00  	add	a6, a4, a3
;           pDstC[output_idx] = sum;
     6e0: 13 18 28 00  	slli	a6, a6, 2
     6e4: 33 88 03 01  	add	a6, t2, a6
     6e8: 23 20 08 00  	sw	zero, 0(a6)
;         for (uint32_t f = 0; f < ch_out_count; ++f) {
     6ec: 93 86 16 00  	addi	a3, a3, 1
;       for (uint32_t w = 0; w < W_out; ++w) {
     6f0: 13 07 16 00  	addi	a4, a2, 1
     6f4: e3 14 e6 fd  	bne	a2, t5, 0x6bc <PULP_Conv2d_fp32_fp32_fp32_HWC+0x6bc>
;     for (uint32_t h = 0; h < H_out; ++h) {
     6f8: 93 05 15 00  	addi	a1, a0, 1
     6fc: e3 1a c5 fb  	bne	a0, t3, 0x6b0 <PULP_Conv2d_fp32_fp32_fp32_HWC+0x6b0>
; }
     700: 83 20 c1 07  	lw	ra, 124(sp)
     704: 03 24 81 07  	lw	s0, 120(sp)
     708: 83 24 41 07  	lw	s1, 116(sp)
     70c: 03 29 01 07  	lw	s2, 112(sp)
     710: 83 29 c1 06  	lw	s3, 108(sp)
     714: 03 2a 81 06  	lw	s4, 104(sp)
     718: 83 2a 41 06  	lw	s5, 100(sp)
     71c: 03 2b 01 06  	lw	s6, 96(sp)
     720: 83 2b c1 05  	lw	s7, 92(sp)
     724: 03 2c 81 05  	lw	s8, 88(sp)
     728: 83 2c 41 05  	lw	s9, 84(sp)
     72c: 03 2d 01 05  	lw	s10, 80(sp)
     730: 83 2d c1 04  	lw	s11, 76(sp)
     734: 13 01 01 08  	addi	sp, sp, 128
     738: 67 80 00 00  	ret

Disassembly of section .text.PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC:

00000000 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC>:
;     float32_t *__restrict__ pContextBuffer, int nb_dedicated_cores) {
       0: 13 01 01 f8  	addi	sp, sp, -128
       4: 23 2e 11 06  	sw	ra, 124(sp)
       8: 23 2c 81 06  	sw	s0, 120(sp)
       c: 23 2a 91 06  	sw	s1, 116(sp)
      10: 23 28 21 07  	sw	s2, 112(sp)
      14: 23 26 31 07  	sw	s3, 108(sp)
      18: 23 24 41 07  	sw	s4, 104(sp)
      1c: 23 22 51 07  	sw	s5, 100(sp)
      20: 23 20 61 07  	sw	s6, 96(sp)
      24: 23 2e 71 05  	sw	s7, 92(sp)
      28: 23 2c 81 05  	sw	s8, 88(sp)
      2c: 23 2a 91 05  	sw	s9, 84(sp)
      30: 23 28 a1 05  	sw	s10, 80(sp)
      34: 23 26 b1 05  	sw	s11, 76(sp)
      38: 83 22 81 0a  	lw	t0, 168(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      3c: 73 23 40 f1  	csrr	t1, mhartid
;   return hart_id & 0x01f;
      40: 93 74 f3 01  	andi	s1, t1, 31
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      44: 33 93 02 10  	<unknown>
;       (F_total >> log2Core) + ((F_total & (nb_dedicated_cores - 1)) != 0);
      48: 33 63 03 10  	<unknown>
      4c: 33 d3 67 00  	srl	t1, a5, t1
      50: 93 82 f2 ff  	addi	t0, t0, -1
      54: b3 f2 f2 00  	and	t0, t0, a5
      58: b3 32 50 00  	snez	t0, t0
      5c: b3 02 53 00  	add	t0, t1, t0
;   uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
      60: b3 d2 02 10  	<unknown>
      64: 33 83 92 02  	mul	t1, t0, s1
      68: 33 53 f3 04  	<unknown>
;   uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
      6c: b3 53 03 10  	<unknown>
      70: 23 24 71 04  	sw	t2, 72(sp)
      74: b3 82 53 00  	add	t0, t2, t0
      78: 23 2a f1 02  	sw	a5, 52(sp)
;   uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
      7c: b3 d3 f2 04  	<unknown>
;   uint16_t ch_out_count = ch_out_stop - ch_out_start;
      80: b3 82 63 40  	sub	t0, t2, t1
;   if (ch_out_count == 0) {
      84: 33 d3 02 10  	<unknown>
      88: e3 00 03 1e  	beqz	t1, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
      8c: 93 82 06 00  	mv	t0, a3
      90: 83 26 41 0a  	lw	a3, 164(sp)
      94: 03 2e 01 0a  	lw	t3, 160(sp)
      98: 03 29 c1 09  	lw	s2, 156(sp)
      9c: 83 2f 81 09  	lw	t6, 152(sp)
      a0: 83 29 41 09  	lw	s3, 148(sp)
      a4: 03 23 01 09  	lw	t1, 144(sp)
      a8: 03 24 c1 08  	lw	s0, 140(sp)
      ac: 03 2a 41 08  	lw	s4, 132(sp)
      b0: 83 2a 01 08  	lw	s5, 128(sp)
      b4: b3 de 03 10  	<unknown>
;   const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;
      b8: b3 03 58 02  	mul	t2, a6, t0
      bc: 33 8f 13 03  	mul	t5, t2, a7
      c0: 83 27 81 04  	lw	a5, 72(sp)
      c4: b3 03 ff 02  	mul	t2, t5, a5
      c8: 93 93 23 00  	slli	t2, t2, 2
      cc: 33 07 77 00  	add	a4, a4, t2
;   float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;
      d0: 23 20 e1 04  	sw	a4, 64(sp)
      d4: 33 07 9f 02  	mul	a4, t5, s1
      d8: 13 17 27 00  	slli	a4, a4, 2
      dc: b3 87 e6 00  	add	a5, a3, a4
;   uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
      e0: b3 86 05 41  	sub	a3, a1, a6
      e4: 23 28 31 01  	sw	s3, 16(sp)
      e8: b3 86 36 01  	add	a3, a3, s3
      ec: b3 86 f6 01  	add	a3, a3, t6
      f0: 23 26 51 01  	sw	s5, 12(sp)
      f4: b3 d4 56 03  	divu	s1, a3, s5
;   uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
      f8: b3 06 16 41  	sub	a3, a2, a7
      fc: 23 2e 21 03  	sw	s2, 60(sp)
     100: b3 86 26 01  	add	a3, a3, s2
     104: b3 86 c6 01  	add	a3, a3, t3
     108: 23 2c 41 03  	sw	s4, 56(sp)
     10c: b3 d9 46 03  	divu	s3, a3, s4
;   if (has_bias) {
     110: 93 76 14 00  	andi	a3, s0, 1
;   uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
     114: 13 87 19 00  	addi	a4, s3, 1
     118: 23 20 e1 02  	sw	a4, 32(sp)
     11c: 23 2a 91 00  	sw	s1, 20(sp)
     120: 23 22 31 05  	sw	s3, 68(sp)
;   if (has_bias) {
     124: 63 84 06 32  	beqz	a3, 0x44c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x44c>
     128: 93 06 f0 ff  	li	a3, -1
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     12c: e3 8e d4 12  	beq	s1, a3, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     130: 83 26 01 02  	lw	a3, 32(sp)
     134: e3 8a 06 12  	beqz	a3, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
     138: 83 2a 81 08  	lw	s5, 136(sp)
;         for (uint32_t p = 0; p < P; p++) {
     13c: 63 00 08 5a  	beqz	a6, 0x6dc <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x6dc>
     140: 13 07 00 00  	li	a4, 0
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     144: b3 86 58 02  	mul	a3, a7, t0
     148: 13 9b 26 00  	slli	s6, a3, 2
     14c: 93 9b 22 00  	slli	s7, t0, 2
     150: 83 26 c1 03  	lw	a3, 60(sp)
     154: 83 23 01 01  	lw	t2, 16(sp)
     158: b3 86 c3 42  	<unknown>
     15c: b3 06 d0 40  	neg	a3, a3
     160: b3 86 d2 02  	mul	a3, t0, a3
     164: 93 96 26 00  	slli	a3, a3, 2
     168: b3 03 d5 00  	add	t2, a0, a3
     16c: 03 25 c1 00  	lw	a0, 12(sp)
     170: 33 05 55 02  	mul	a0, a0, t0
     174: 33 05 c5 02  	mul	a0, a0, a2
     178: 13 15 25 00  	slli	a0, a0, 2
     17c: 23 24 a1 00  	sw	a0, 8(sp)
     180: 03 25 81 03  	lw	a0, 56(sp)
     184: 33 05 55 02  	mul	a0, a0, t0
     188: 13 15 25 00  	slli	a0, a0, 2
     18c: 23 28 a1 02  	sw	a0, 48(sp)
     190: 33 85 c2 02  	mul	a0, t0, a2
     194: 13 1d 25 00  	slli	s10, a0, 2
     198: 33 85 08 03  	mul	a0, a7, a6
     19c: 33 05 55 02  	mul	a0, a0, t0
     1a0: 93 1d 25 00  	slli	s11, a0, 2
     1a4: 03 25 81 04  	lw	a0, 72(sp)
     1a8: b3 86 ae 40  	sub	a3, t4, a0
     1ac: 23 24 d1 02  	sw	a3, 40(sp)
     1b0: 13 15 25 00  	slli	a0, a0, 2
     1b4: 33 85 aa 00  	add	a0, s5, a0
     1b8: 23 22 a1 02  	sw	a0, 36(sp)
     1bc: 53 00 00 f0  	fmv.w.x	ft0, zero
     1c0: 13 0a f0 ff  	li	s4, -1
     1c4: 6f 00 00 02  	j	0x1e4 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x1e4>
     1c8: 83 26 81 01  	lw	a3, 24(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     1cc: 13 87 16 00  	addi	a4, a3, 1
     1d0: 83 23 c1 01  	lw	t2, 28(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     1d4: 03 25 81 00  	lw	a0, 8(sp)
     1d8: b3 83 a3 00  	add	t2, t2, a0
     1dc: 03 25 41 01  	lw	a0, 20(sp)
     1e0: e3 84 a6 08  	beq	a3, a0, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
     1e4: 03 25 01 02  	lw	a0, 32(sp)
     1e8: 33 05 a7 02  	mul	a0, a4, a0
;           for (uint32_t q = 0; q < Q; q++) {
     1ec: 23 26 a1 02  	sw	a0, 44(sp)
     1f0: 23 2e 71 00  	sw	t2, 28(sp)
     1f4: 23 2c e1 00  	sw	a4, 24(sp)
     1f8: 63 8c 08 1e  	beqz	a7, 0x3f0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x3f0>
     1fc: 13 05 00 00  	li	a0, 0
     200: 83 26 c1 00  	lw	a3, 12(sp)
     204: b3 06 d7 02  	mul	a3, a4, a3
     208: 03 27 01 01  	lw	a4, 16(sp)
     20c: 33 8c e6 40  	sub	s8, a3, a4
     210: 13 87 03 00  	mv	a4, t2
     214: 6f 00 40 01  	j	0x228 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x228>
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     218: 13 05 1e 00  	addi	a0, t3, 1
     21c: 83 26 01 03  	lw	a3, 48(sp)
     220: 33 07 d7 00  	add	a4, a4, a3
     224: e3 02 3e fb  	beq	t3, s3, 0x1c8 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x1c8>
     228: 13 0e 05 00  	mv	t3, a0
;             for (uint32_t c = 0; c < C; c++) {
     22c: 63 86 02 10  	beqz	t0, 0x338 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x338>
     230: 93 0f 00 00  	li	t6, 0
     234: 03 25 81 03  	lw	a0, 56(sp)
     238: 33 05 ae 02  	mul	a0, t3, a0
     23c: 83 26 c1 03  	lw	a3, 60(sp)
     240: b3 09 d5 40  	sub	s3, a0, a3
     244: 93 04 07 00  	mv	s1, a4
     248: 13 84 07 00  	mv	s0, a5
     24c: 6f 00 40 01  	j	0x260 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x260>
;         for (uint32_t p = 0; p < P; p++) {
     250: 93 8f 1f 00  	addi	t6, t6, 1
     254: 33 04 64 01  	add	s0, s0, s6
     258: b3 84 a4 01  	add	s1, s1, s10
     25c: 63 8e 0f 0d  	beq	t6, a6, 0x338 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x338>
;           int32_t h_in = h_in_start + p;
     260: 33 85 8f 01  	add	a0, t6, s8
     264: 63 4e 05 06  	bltz	a0, 0x2e0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x2e0>
     268: 63 52 b5 0a  	bge	a0, a1, 0x30c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x30c>
     26c: 13 09 00 00  	li	s2, 0
     270: 13 85 04 00  	mv	a0, s1
     274: 93 03 04 00  	mv	t2, s0
     278: 6f 00 40 01  	j	0x28c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x28c>
;           for (uint32_t q = 0; q < Q; q++) {
     27c: 13 09 19 00  	addi	s2, s2, 1
     280: b3 83 73 01  	add	t2, t2, s7
     284: 33 05 75 01  	add	a0, a0, s7
     288: e3 04 19 fd  	beq	s2, a7, 0x250 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x250>
;             int32_t w_in = w_in_start + q;
     28c: b3 06 39 01  	add	a3, s2, s3
     290: b3 2c da 00  	slt	s9, s4, a3
     294: b3 a6 c6 00  	slt	a3, a3, a2
     298: b3 f0 dc 00  	and	ra, s9, a3
     29c: 93 86 03 00  	mv	a3, t2
     2a0: 93 8c 02 00  	mv	s9, t0
;               if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
     2a4: 63 86 00 02  	beqz	ra, 0x2d0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x2d0>
     2a8: 93 0c 00 00  	li	s9, 0
     2ac: 93 86 02 00  	mv	a3, t0
;             for (uint32_t c = 0; c < C; c++) {
     2b0: 7b c0 c2 00  	<unknown>
;                 im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
     2b4: b3 00 95 01  	add	ra, a0, s9
     2b8: 87 a0 00 00  	flw	ft1, 0(ra)
     2bc: b3 80 93 01  	add	ra, t2, s9
     2c0: 27 a0 10 00  	fsw	ft1, 0(ra)
;             for (uint32_t c = 0; c < C; c++) {
     2c4: 93 86 f6 ff  	addi	a3, a3, -1
     2c8: 93 8c 4c 00  	addi	s9, s9, 4
     2cc: 6f f0 1f fb  	j	0x27c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x27c>
     2d0: 93 8c fc ff  	addi	s9, s9, -1
;                 im2col_buffer[p * Q * C + q * C + c] = 0.0f;
     2d4: 2b a2 06 00  	<unknown>
;             for (uint32_t c = 0; c < C; c++) {
     2d8: e3 9c 0c fe  	bnez	s9, 0x2d0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x2d0>
     2dc: 6f f0 1f fa  	j	0x27c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x27c>
     2e0: 13 05 00 00  	li	a0, 0
     2e4: 93 03 04 00  	mv	t2, s0
     2e8: fb c0 e8 00  	<unknown>
     2ec: 93 86 03 00  	mv	a3, t2
     2f0: 13 89 02 00  	mv	s2, t0
     2f4: 7b c0 42 00  	<unknown>
;             for (uint32_t c = 0; c < C; c++) {
     2f8: 13 09 f9 ff  	addi	s2, s2, -1
;                 im2col_buffer[p * Q * C + q * C + c] = 0.0f;
     2fc: 2b a2 06 00  	<unknown>
;           for (uint32_t q = 0; q < Q; q++) {
     300: 13 05 15 00  	addi	a0, a0, 1
     304: b3 83 73 01  	add	t2, t2, s7
     308: 6f f0 9f f4  	j	0x250 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x250>
     30c: 13 05 00 00  	li	a0, 0
     310: 93 03 04 00  	mv	t2, s0
     314: 93 86 03 00  	mv	a3, t2
     318: 13 89 02 00  	mv	s2, t0
     31c: 7b c0 42 00  	<unknown>
;             for (uint32_t c = 0; c < C; c++) {
     320: 13 09 f9 ff  	addi	s2, s2, -1
;                 im2col_buffer[p * Q * C + q * C + c] = 0.0f;
     324: 2b a2 06 00  	<unknown>
;           for (uint32_t q = 0; q < Q; q++) {
     328: 13 05 15 00  	addi	a0, a0, 1
     32c: b3 83 73 01  	add	t2, t2, s7
     330: e3 12 15 ff  	bne	a0, a7, 0x314 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x314>
     334: 6f f0 df f1  	j	0x250 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x250>
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     338: 03 25 81 04  	lw	a0, 72(sp)
     33c: 83 29 41 04  	lw	s3, 68(sp)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     340: e3 7c d5 ed  	bgeu	a0, t4, 0x218 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x218>
     344: 03 25 c1 02  	lw	a0, 44(sp)
     348: 33 05 ae 00  	add	a0, t3, a0
     34c: 83 26 41 03  	lw	a3, 52(sp)
     350: 33 05 d5 02  	mul	a0, a0, a3
     354: 03 24 41 02  	lw	s0, 36(sp)
     358: 83 24 81 02  	lw	s1, 40(sp)
     35c: 83 2f 81 04  	lw	t6, 72(sp)
     360: 13 89 0f 00  	mv	s2, t6
     364: 83 23 01 04  	lw	t2, 64(sp)
;           for (uint32_t k = 0; k < kernel_size; k++) {
     368: 63 00 0f 06  	beqz	t5, 0x3c8 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x3c8>
     36c: 13 84 07 00  	mv	s0, a5
     370: 93 84 03 00  	mv	s1, t2
     374: 13 09 0f 00  	mv	s2, t5
     378: d3 00 00 20  	fmv.s	ft1, ft0
     37c: 7b 40 cf 00  	<unknown>
;             sum += im2col_buffer[k] * local_weight_ptr[k];
     380: 07 21 04 00  	flw	ft2, 0(s0)
     384: 87 a1 04 00  	flw	ft3, 0(s1)
     388: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;           for (uint32_t k = 0; k < kernel_size; k++) {
     38c: 13 09 f9 ff  	addi	s2, s2, -1
     390: 93 84 44 00  	addi	s1, s1, 4
     394: 13 04 44 00  	addi	s0, s0, 4
;           pDstC[out_idx] = sum + pSrcBias[f];
     398: 93 96 2f 00  	slli	a3, t6, 2
     39c: b3 86 da 00  	add	a3, s5, a3
     3a0: 07 a1 06 00  	flw	ft2, 0(a3)
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     3a4: b3 86 af 00  	add	a3, t6, a0
;           pDstC[out_idx] = sum + pSrcBias[f];
     3a8: d3 70 11 00  	fadd.s	ft1, ft2, ft1
     3ac: 93 96 26 00  	slli	a3, a3, 2
     3b0: b3 06 d3 00  	add	a3, t1, a3
     3b4: 27 a0 16 00  	fsw	ft1, 0(a3)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     3b8: 93 8f 1f 00  	addi	t6, t6, 1
     3bc: b3 83 b3 01  	add	t2, t2, s11
     3c0: e3 96 df fb  	bne	t6, t4, 0x36c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x36c>
     3c4: 6f f0 5f e5  	j	0x218 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x218>
;           pDstC[out_idx] = sum + pSrcBias[f];
     3c8: 87 20 04 00  	flw	ft1, 0(s0)
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     3cc: b3 06 a9 00  	add	a3, s2, a0
;           pDstC[out_idx] = sum + pSrcBias[f];
     3d0: 93 96 26 00  	slli	a3, a3, 2
     3d4: b3 06 d3 00  	add	a3, t1, a3
     3d8: 27 a0 16 00  	fsw	ft1, 0(a3)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     3dc: 13 09 19 00  	addi	s2, s2, 1
     3e0: 93 84 f4 ff  	addi	s1, s1, -1
     3e4: 13 04 44 00  	addi	s0, s0, 4
     3e8: e3 90 04 fe  	bnez	s1, 0x3c8 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x3c8>
     3ec: 6f f0 df e2  	j	0x218 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x218>
     3f0: 03 25 81 04  	lw	a0, 72(sp)
     3f4: e3 7a d5 dd  	bgeu	a0, t4, 0x1c8 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x1c8>
     3f8: 93 06 00 00  	li	a3, 0
     3fc: 13 85 06 00  	mv	a0, a3
     400: 83 26 c1 02  	lw	a3, 44(sp)
     404: b3 06 d5 00  	add	a3, a0, a3
     408: 03 27 41 03  	lw	a4, 52(sp)
     40c: 33 87 e6 02  	mul	a4, a3, a4
     410: 83 23 41 02  	lw	t2, 36(sp)
     414: 83 26 81 02  	lw	a3, 40(sp)
     418: 13 8e 06 00  	mv	t3, a3
     41c: 83 2f 81 04  	lw	t6, 72(sp)
     420: 7b c0 e6 00  	<unknown>
;           pDstC[out_idx] = sum + pSrcBias[f];
     424: 87 a0 03 00  	flw	ft1, 0(t2)
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     428: b3 86 ef 00  	add	a3, t6, a4
;           pDstC[out_idx] = sum + pSrcBias[f];
     42c: 93 96 26 00  	slli	a3, a3, 2
     430: b3 06 d3 00  	add	a3, t1, a3
     434: 27 a0 16 00  	fsw	ft1, 0(a3)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     438: 93 8f 1f 00  	addi	t6, t6, 1
     43c: 93 83 43 00  	addi	t2, t2, 4
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     440: 93 06 15 00  	addi	a3, a0, 1
     444: e3 1c 35 fb  	bne	a0, s3, 0x3fc <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x3fc>
     448: 6f f0 1f d8  	j	0x1c8 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x1c8>
     44c: 93 06 f0 ff  	li	a3, -1
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     450: 63 8c d4 60  	beq	s1, a3, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     454: 83 26 01 02  	lw	a3, 32(sp)
     458: 63 88 06 60  	beqz	a3, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
;         for (uint32_t p = 0; p < P; p++) {
     45c: 63 0a 08 32  	beqz	a6, 0x790 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x790>
;           for (uint32_t q = 0; q < Q; q++) {
     460: 63 8a 08 3c  	beqz	a7, 0x834 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x834>
;             for (uint32_t c = 0; c < C; c++) {
     464: 63 8c 02 42  	beqz	t0, 0x89c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x89c>
     468: 93 03 00 00  	li	t2, 0
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     46c: b3 86 58 02  	mul	a3, a7, t0
     470: 13 97 26 00  	slli	a4, a3, 2
     474: 13 9e 22 00  	slli	t3, t0, 2
     478: 83 26 c1 03  	lw	a3, 60(sp)
     47c: 83 2f 01 01  	lw	t6, 16(sp)
     480: b3 86 cf 42  	<unknown>
     484: b3 06 d0 40  	neg	a3, a3
     488: b3 86 d2 02  	mul	a3, t0, a3
     48c: 93 96 26 00  	slli	a3, a3, 2
     490: b3 0f d5 00  	add	t6, a0, a3
     494: 03 25 c1 00  	lw	a0, 12(sp)
     498: 33 05 55 02  	mul	a0, a0, t0
     49c: 33 05 c5 02  	mul	a0, a0, a2
     4a0: 13 15 25 00  	slli	a0, a0, 2
     4a4: 23 2e a1 00  	sw	a0, 28(sp)
     4a8: 03 25 81 03  	lw	a0, 56(sp)
     4ac: 33 05 55 02  	mul	a0, a0, t0
     4b0: 13 1a 25 00  	slli	s4, a0, 2
     4b4: 33 85 c2 02  	mul	a0, t0, a2
     4b8: 93 1a 25 00  	slli	s5, a0, 2
     4bc: 33 85 08 03  	mul	a0, a7, a6
     4c0: 33 05 55 02  	mul	a0, a0, t0
     4c4: 13 1b 25 00  	slli	s6, a0, 2
     4c8: 03 25 81 04  	lw	a0, 72(sp)
     4cc: 33 85 ae 40  	sub	a0, t4, a0
     4d0: 23 26 a1 02  	sw	a0, 44(sp)
     4d4: 13 0c f0 ff  	li	s8, -1
     4d8: 53 00 00 f0  	fmv.w.x	ft0, zero
     4dc: 6f 00 00 02  	j	0x4fc <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x4fc>
     4e0: 83 26 41 02  	lw	a3, 36(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     4e4: 93 83 16 00  	addi	t2, a3, 1
     4e8: 83 2f 81 02  	lw	t6, 40(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     4ec: 03 25 c1 01  	lw	a0, 28(sp)
     4f0: b3 8f af 00  	add	t6, t6, a0
     4f4: 03 25 41 01  	lw	a0, 20(sp)
     4f8: 63 88 a6 56  	beq	a3, a0, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
     4fc: 93 06 00 00  	li	a3, 0
     500: 13 84 03 00  	mv	s0, t2
     504: 03 25 c1 00  	lw	a0, 12(sp)
     508: 33 85 a3 02  	mul	a0, t2, a0
     50c: 83 23 01 01  	lw	t2, 16(sp)
     510: 33 0d 75 40  	sub	s10, a0, t2
     514: 03 25 01 02  	lw	a0, 32(sp)
     518: 23 22 81 02  	sw	s0, 36(sp)
     51c: 33 05 a4 02  	mul	a0, s0, a0
     520: 23 28 a1 02  	sw	a0, 48(sp)
     524: 23 24 f1 03  	sw	t6, 40(sp)
     528: 13 85 0f 00  	mv	a0, t6
     52c: 6f 00 40 01  	j	0x540 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x540>
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     530: 93 06 19 00  	addi	a3, s2, 1
     534: 33 05 45 01  	add	a0, a0, s4
     538: 83 23 41 04  	lw	t2, 68(sp)
     53c: e3 02 79 fa  	beq	s2, t2, 0x4e0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x4e0>
     540: 93 0c 00 00  	li	s9, 0
     544: 13 89 06 00  	mv	s2, a3
;         int32_t w_in_start = w_out * SQ - pad_left;
     548: 83 26 81 03  	lw	a3, 56(sp)
     54c: b3 06 d9 02  	mul	a3, s2, a3
     550: 83 23 c1 03  	lw	t2, 60(sp)
     554: b3 8f 76 40  	sub	t6, a3, t2
     558: 93 00 05 00  	mv	ra, a0
     55c: 93 83 07 00  	mv	t2, a5
     560: 6f 00 40 01  	j	0x574 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x574>
;         for (uint32_t p = 0; p < P; p++) {
     564: 93 8c 1c 00  	addi	s9, s9, 1
     568: b3 83 e3 00  	add	t2, t2, a4
     56c: b3 80 50 01  	add	ra, ra, s5
     570: 63 8a 0c 0d  	beq	s9, a6, 0x644 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x644>
     574: 13 04 00 00  	li	s0, 0
;           int32_t h_in = h_in_start + p;
     578: b3 86 ac 01  	add	a3, s9, s10
;               if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
     57c: 63 cc 06 06  	bltz	a3, 0x5f4 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x5f4>
     580: 63 de b6 08  	bge	a3, a1, 0x61c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x61c>
     584: 93 89 00 00  	mv	s3, ra
     588: 93 84 03 00  	mv	s1, t2
     58c: 6f 00 40 01  	j	0x5a0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x5a0>
;           for (uint32_t q = 0; q < Q; q++) {
     590: 13 04 14 00  	addi	s0, s0, 1
     594: b3 84 c4 01  	add	s1, s1, t3
     598: b3 89 c9 01  	add	s3, s3, t3
     59c: e3 04 14 fd  	beq	s0, a7, 0x564 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x564>
;             int32_t w_in = w_in_start + q;
     5a0: b3 06 f4 01  	add	a3, s0, t6
     5a4: b3 2b dc 00  	slt	s7, s8, a3
     5a8: b3 a6 c6 00  	slt	a3, a3, a2
     5ac: b3 fd db 00  	and	s11, s7, a3
     5b0: 93 86 04 00  	mv	a3, s1
     5b4: 93 8b 02 00  	mv	s7, t0
;               if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
     5b8: 63 86 0d 02  	beqz	s11, 0x5e4 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x5e4>
     5bc: 93 0b 00 00  	li	s7, 0
     5c0: 93 86 02 00  	mv	a3, t0
;             for (uint32_t c = 0; c < C; c++) {
     5c4: 7b c0 c2 00  	<unknown>
;                 im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
     5c8: b3 8d 79 01  	add	s11, s3, s7
     5cc: 87 a0 0d 00  	flw	ft1, 0(s11)
     5d0: b3 8d 74 01  	add	s11, s1, s7
     5d4: 27 a0 1d 00  	fsw	ft1, 0(s11)
;             for (uint32_t c = 0; c < C; c++) {
     5d8: 93 86 f6 ff  	addi	a3, a3, -1
     5dc: 93 8b 4b 00  	addi	s7, s7, 4
     5e0: 6f f0 1f fb  	j	0x590 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x590>
     5e4: 93 8b fb ff  	addi	s7, s7, -1
;                 im2col_buffer[p * Q * C + q * C + c] = 0.0f;
     5e8: 2b a2 06 00  	<unknown>
;             for (uint32_t c = 0; c < C; c++) {
     5ec: e3 9c 0b fe  	bnez	s7, 0x5e4 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x5e4>
     5f0: 6f f0 1f fa  	j	0x590 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x590>
     5f4: 93 84 03 00  	mv	s1, t2
     5f8: fb c0 e8 00  	<unknown>
     5fc: 93 86 04 00  	mv	a3, s1
     600: 93 89 02 00  	mv	s3, t0
     604: 7b c0 42 00  	<unknown>
;             for (uint32_t c = 0; c < C; c++) {
     608: 93 89 f9 ff  	addi	s3, s3, -1
;                 im2col_buffer[p * Q * C + q * C + c] = 0.0f;
     60c: 2b a2 06 00  	<unknown>
;           for (uint32_t q = 0; q < Q; q++) {
     610: 13 04 14 00  	addi	s0, s0, 1
     614: b3 84 c4 01  	add	s1, s1, t3
     618: 6f f0 df f4  	j	0x564 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x564>
     61c: 93 84 03 00  	mv	s1, t2
     620: 93 86 04 00  	mv	a3, s1
     624: 93 89 02 00  	mv	s3, t0
     628: 7b c0 42 00  	<unknown>
;             for (uint32_t c = 0; c < C; c++) {
     62c: 93 89 f9 ff  	addi	s3, s3, -1
;                 im2col_buffer[p * Q * C + q * C + c] = 0.0f;
     630: 2b a2 06 00  	<unknown>
;           for (uint32_t q = 0; q < Q; q++) {
     634: 13 04 14 00  	addi	s0, s0, 1
     638: b3 84 c4 01  	add	s1, s1, t3
     63c: e3 12 14 ff  	bne	s0, a7, 0x620 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x620>
     640: 6f f0 5f f2  	j	0x564 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x564>
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     644: 83 26 81 04  	lw	a3, 72(sp)
     648: e3 f4 d6 ef  	bgeu	a3, t4, 0x530 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x530>
     64c: 83 26 01 03  	lw	a3, 48(sp)
     650: b3 06 d9 00  	add	a3, s2, a3
     654: 83 23 41 03  	lw	t2, 52(sp)
     658: b3 8f 76 02  	mul	t6, a3, t2
     65c: 83 24 c1 02  	lw	s1, 44(sp)
     660: 03 24 81 04  	lw	s0, 72(sp)
     664: 93 09 04 00  	mv	s3, s0
     668: 83 23 01 04  	lw	t2, 64(sp)
;           for (uint32_t k = 0; k < kernel_size; k++) {
     66c: 63 08 0f 04  	beqz	t5, 0x6bc <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x6bc>
     670: 93 84 07 00  	mv	s1, a5
     674: 93 89 03 00  	mv	s3, t2
     678: 93 0c 0f 00  	mv	s9, t5
     67c: d3 00 00 20  	fmv.s	ft1, ft0
     680: 7b 40 cf 00  	<unknown>
;             sum += im2col_buffer[k] * local_weight_ptr[k];
     684: 07 a1 04 00  	flw	ft2, 0(s1)
     688: 87 a1 09 00  	flw	ft3, 0(s3)
     68c: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;           for (uint32_t k = 0; k < kernel_size; k++) {
     690: 93 8c fc ff  	addi	s9, s9, -1
     694: 93 89 49 00  	addi	s3, s3, 4
     698: 93 84 44 00  	addi	s1, s1, 4
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     69c: b3 06 f4 01  	add	a3, s0, t6
;           pDstC[out_idx] = sum;
     6a0: 93 96 26 00  	slli	a3, a3, 2
     6a4: b3 06 d3 00  	add	a3, t1, a3
     6a8: 27 a0 16 00  	fsw	ft1, 0(a3)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     6ac: 13 04 14 00  	addi	s0, s0, 1
     6b0: b3 83 63 01  	add	t2, t2, s6
     6b4: e3 1e d4 fb  	bne	s0, t4, 0x670 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x670>
     6b8: 6f f0 9f e7  	j	0x530 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x530>
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     6bc: b3 86 f9 01  	add	a3, s3, t6
;           pDstC[out_idx] = sum;
     6c0: 93 96 26 00  	slli	a3, a3, 2
     6c4: b3 06 d3 00  	add	a3, t1, a3
     6c8: 23 a0 06 00  	sw	zero, 0(a3)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     6cc: 93 84 f4 ff  	addi	s1, s1, -1
     6d0: 93 89 19 00  	addi	s3, s3, 1
     6d4: e3 94 04 fe  	bnez	s1, 0x6bc <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x6bc>
     6d8: 6f f0 9f e5  	j	0x530 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x530>
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     6dc: 03 25 81 04  	lw	a0, 72(sp)
     6e0: 63 74 d5 39  	bgeu	a0, t4, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
;           for (uint32_t k = 0; k < kernel_size; k++) {
     6e4: 63 0e 0f 24  	beqz	t5, 0x940 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x940>
     6e8: 13 06 00 00  	li	a2, 0
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     6ec: 33 85 08 03  	mul	a0, a7, a6
     6f0: 33 05 55 02  	mul	a0, a0, t0
     6f4: 13 15 25 00  	slli	a0, a0, 2
     6f8: 53 00 00 f0  	fmv.w.x	ft0, zero
     6fc: 03 24 41 03  	lw	s0, 52(sp)
     700: 13 07 00 00  	li	a4, 0
     704: 93 05 06 00  	mv	a1, a2
     708: 03 26 01 02  	lw	a2, 32(sp)
     70c: 33 86 c5 02  	mul	a2, a1, a2
     710: 93 06 07 00  	mv	a3, a4
     714: 33 07 c7 00  	add	a4, a4, a2
     718: 83 28 81 04  	lw	a7, 72(sp)
     71c: b3 82 1e 41  	sub	t0, t4, a7
     720: 33 07 87 02  	mul	a4, a4, s0
     724: 03 28 01 04  	lw	a6, 64(sp)
     728: fb c0 82 02  	<unknown>
     72c: 93 83 07 00  	mv	t2, a5
     730: 13 0e 08 00  	mv	t3, a6
     734: 93 0f 0f 00  	mv	t6, t5
     738: d3 00 00 20  	fmv.s	ft1, ft0
     73c: 7b 40 af 00  	<unknown>
;             sum += im2col_buffer[k] * local_weight_ptr[k];
     740: 07 a1 03 00  	flw	ft2, 0(t2)
     744: 87 21 0e 00  	flw	ft3, 0(t3)
     748: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;           for (uint32_t k = 0; k < kernel_size; k++) {
     74c: 13 0e 4e 00  	addi	t3, t3, 4
     750: 93 83 43 00  	addi	t2, t2, 4
;           pDstC[out_idx] = sum + pSrcBias[f];
     754: 93 92 28 00  	slli	t0, a7, 2
     758: b3 82 5a 00  	add	t0, s5, t0
     75c: 07 a1 02 00  	flw	ft2, 0(t0)
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     760: b3 82 e8 00  	add	t0, a7, a4
;           pDstC[out_idx] = sum + pSrcBias[f];
     764: d3 70 11 00  	fadd.s	ft1, ft2, ft1
     768: 93 92 22 00  	slli	t0, t0, 2
     76c: b3 02 53 00  	add	t0, t1, t0
     770: 27 a0 12 00  	fsw	ft1, 0(t0)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     774: 93 88 18 00  	addi	a7, a7, 1
     778: 33 08 a8 00  	add	a6, a6, a0
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     77c: 13 87 16 00  	addi	a4, a3, 1
     780: e3 98 36 f9  	bne	a3, s3, 0x710 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x710>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     784: 13 86 15 00  	addi	a2, a1, 1
     788: e3 9c 95 f6  	bne	a1, s1, 0x700 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x700>
     78c: 6f 00 c0 2d  	j	0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     790: 03 25 81 04  	lw	a0, 72(sp)
     794: 63 7a d5 2d  	bgeu	a0, t4, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
     798: 13 06 00 00  	li	a2, 0
;           for (uint32_t k = 0; k < kernel_size; k++) {
     79c: 63 0c 0f 20  	beqz	t5, 0x9b4 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x9b4>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     7a0: 33 85 08 03  	mul	a0, a7, a6
     7a4: 33 05 55 02  	mul	a0, a0, t0
     7a8: 13 15 25 00  	slli	a0, a0, 2
     7ac: 53 00 00 f0  	fmv.w.x	ft0, zero
     7b0: 03 24 41 03  	lw	s0, 52(sp)
     7b4: 13 07 00 00  	li	a4, 0
     7b8: 93 05 06 00  	mv	a1, a2
     7bc: 03 26 01 02  	lw	a2, 32(sp)
     7c0: 33 86 c5 02  	mul	a2, a1, a2
     7c4: 93 06 07 00  	mv	a3, a4
     7c8: 33 07 c7 00  	add	a4, a4, a2
     7cc: 83 28 81 04  	lw	a7, 72(sp)
     7d0: b3 82 1e 41  	sub	t0, t4, a7
     7d4: 33 07 87 02  	mul	a4, a4, s0
     7d8: 03 28 01 04  	lw	a6, 64(sp)
     7dc: fb c0 02 02  	<unknown>
     7e0: 93 83 07 00  	mv	t2, a5
     7e4: 13 0e 08 00  	mv	t3, a6
     7e8: 93 0f 0f 00  	mv	t6, t5
     7ec: d3 00 00 20  	fmv.s	ft1, ft0
     7f0: 7b 40 af 00  	<unknown>
;             sum += im2col_buffer[k] * local_weight_ptr[k];
     7f4: 07 a1 03 00  	flw	ft2, 0(t2)
     7f8: 87 21 0e 00  	flw	ft3, 0(t3)
     7fc: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;           for (uint32_t k = 0; k < kernel_size; k++) {
     800: 13 0e 4e 00  	addi	t3, t3, 4
     804: 93 83 43 00  	addi	t2, t2, 4
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     808: b3 82 e8 00  	add	t0, a7, a4
;           pDstC[out_idx] = sum;
     80c: 93 92 22 00  	slli	t0, t0, 2
     810: b3 02 53 00  	add	t0, t1, t0
     814: 27 a0 12 00  	fsw	ft1, 0(t0)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     818: 93 88 18 00  	addi	a7, a7, 1
     81c: 33 08 a8 00  	add	a6, a6, a0
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     820: 13 87 16 00  	addi	a4, a3, 1
     824: e3 90 36 fb  	bne	a3, s3, 0x7c4 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x7c4>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     828: 13 86 15 00  	addi	a2, a1, 1
     82c: e3 94 95 f8  	bne	a1, s1, 0x7b4 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x7b4>
     830: 6f 00 80 23  	j	0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     834: 03 25 81 04  	lw	a0, 72(sp)
     838: 63 78 d5 23  	bgeu	a0, t4, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
     83c: 13 06 00 00  	li	a2, 0
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     840: 03 25 81 04  	lw	a0, 72(sp)
     844: 33 85 ae 40  	sub	a0, t4, a0
     848: 83 22 41 03  	lw	t0, 52(sp)
     84c: 13 07 00 00  	li	a4, 0
     850: 93 05 06 00  	mv	a1, a2
     854: 03 26 01 02  	lw	a2, 32(sp)
     858: 33 86 c5 02  	mul	a2, a1, a2
     85c: 93 06 07 00  	mv	a3, a4
     860: 33 07 c7 00  	add	a4, a4, a2
     864: 33 07 57 02  	mul	a4, a4, t0
     868: 93 07 05 00  	mv	a5, a0
     86c: 03 28 81 04  	lw	a6, 72(sp)
     870: 7b 40 a5 00  	<unknown>
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     874: b3 08 e8 00  	add	a7, a6, a4
;           pDstC[out_idx] = sum;
     878: 93 98 28 00  	slli	a7, a7, 2
     87c: b3 08 13 01  	add	a7, t1, a7
     880: 23 a0 08 00  	sw	zero, 0(a7)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     884: 13 08 18 00  	addi	a6, a6, 1
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     888: 13 87 16 00  	addi	a4, a3, 1
     88c: e3 98 36 fd  	bne	a3, s3, 0x85c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x85c>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     890: 13 86 15 00  	addi	a2, a1, 1
     894: e3 9c 95 fa  	bne	a1, s1, 0x84c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x84c>
     898: 6f 00 00 1d  	j	0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     89c: 03 25 81 04  	lw	a0, 72(sp)
     8a0: 63 74 d5 1d  	bgeu	a0, t4, 0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
     8a4: 13 06 00 00  	li	a2, 0
;           for (uint32_t k = 0; k < kernel_size; k++) {
     8a8: 63 04 0f 16  	beqz	t5, 0xa10 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa10>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     8ac: 33 85 08 03  	mul	a0, a7, a6
     8b0: 33 05 55 02  	mul	a0, a0, t0
     8b4: 13 15 25 00  	slli	a0, a0, 2
     8b8: 53 00 00 f0  	fmv.w.x	ft0, zero
     8bc: 03 24 41 03  	lw	s0, 52(sp)
     8c0: 13 07 00 00  	li	a4, 0
     8c4: 93 05 06 00  	mv	a1, a2
     8c8: 03 26 01 02  	lw	a2, 32(sp)
     8cc: 33 86 c5 02  	mul	a2, a1, a2
     8d0: 93 06 07 00  	mv	a3, a4
     8d4: 33 07 c7 00  	add	a4, a4, a2
     8d8: 83 28 81 04  	lw	a7, 72(sp)
     8dc: b3 82 1e 41  	sub	t0, t4, a7
     8e0: 33 07 87 02  	mul	a4, a4, s0
     8e4: 03 28 01 04  	lw	a6, 64(sp)
     8e8: fb c0 02 02  	<unknown>
     8ec: 93 83 07 00  	mv	t2, a5
     8f0: 13 0e 08 00  	mv	t3, a6
     8f4: 93 0f 0f 00  	mv	t6, t5
     8f8: d3 00 00 20  	fmv.s	ft1, ft0
     8fc: 7b 40 af 00  	<unknown>
;             sum += im2col_buffer[k] * local_weight_ptr[k];
     900: 07 a1 03 00  	flw	ft2, 0(t2)
     904: 87 21 0e 00  	flw	ft3, 0(t3)
     908: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;           for (uint32_t k = 0; k < kernel_size; k++) {
     90c: 13 0e 4e 00  	addi	t3, t3, 4
     910: 93 83 43 00  	addi	t2, t2, 4
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     914: b3 82 e8 00  	add	t0, a7, a4
;           pDstC[out_idx] = sum;
     918: 93 92 22 00  	slli	t0, t0, 2
     91c: b3 02 53 00  	add	t0, t1, t0
     920: 27 a0 12 00  	fsw	ft1, 0(t0)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     924: 93 88 18 00  	addi	a7, a7, 1
     928: 33 08 a8 00  	add	a6, a6, a0
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     92c: 13 87 16 00  	addi	a4, a3, 1
     930: e3 90 36 fb  	bne	a3, s3, 0x8d0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8d0>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     934: 13 86 15 00  	addi	a2, a1, 1
     938: e3 94 95 f8  	bne	a1, s1, 0x8c0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8c0>
     93c: 6f 00 c0 12  	j	0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
     940: 93 06 00 00  	li	a3, 0
     944: 83 25 81 04  	lw	a1, 72(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     948: 33 85 be 40  	sub	a0, t4, a1
     94c: 93 95 25 00  	slli	a1, a1, 2
     950: b3 85 ba 00  	add	a1, s5, a1
     954: 03 2e 41 03  	lw	t3, 52(sp)
     958: 93 07 00 00  	li	a5, 0
     95c: 13 86 06 00  	mv	a2, a3
     960: 83 26 01 02  	lw	a3, 32(sp)
     964: b3 06 d6 02  	mul	a3, a2, a3
     968: 13 87 07 00  	mv	a4, a5
     96c: b3 87 d7 00  	add	a5, a5, a3
     970: b3 87 c7 03  	mul	a5, a5, t3
     974: 13 88 05 00  	mv	a6, a1
     978: 93 08 05 00  	mv	a7, a0
     97c: 83 23 81 04  	lw	t2, 72(sp)
     980: 7b 40 e5 00  	<unknown>
;           pDstC[out_idx] = sum + pSrcBias[f];
     984: 07 20 08 00  	flw	ft0, 0(a6)
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     988: b3 82 f3 00  	add	t0, t2, a5
;           pDstC[out_idx] = sum + pSrcBias[f];
     98c: 93 92 22 00  	slli	t0, t0, 2
     990: b3 02 53 00  	add	t0, t1, t0
     994: 27 a0 02 00  	fsw	ft0, 0(t0)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     998: 93 83 13 00  	addi	t2, t2, 1
     99c: 13 08 48 00  	addi	a6, a6, 4
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     9a0: 93 07 17 00  	addi	a5, a4, 1
     9a4: e3 12 37 fd  	bne	a4, s3, 0x968 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x968>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     9a8: 93 06 16 00  	addi	a3, a2, 1
     9ac: e3 16 96 fa  	bne	a2, s1, 0x958 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x958>
     9b0: 6f 00 80 0b  	j	0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     9b4: 03 25 81 04  	lw	a0, 72(sp)
     9b8: 33 85 ae 40  	sub	a0, t4, a0
     9bc: 83 22 41 03  	lw	t0, 52(sp)
     9c0: 13 07 00 00  	li	a4, 0
     9c4: 93 05 06 00  	mv	a1, a2
     9c8: 03 26 01 02  	lw	a2, 32(sp)
     9cc: 33 86 c5 02  	mul	a2, a1, a2
     9d0: 93 06 07 00  	mv	a3, a4
     9d4: 33 07 c7 00  	add	a4, a4, a2
     9d8: 33 07 57 02  	mul	a4, a4, t0
     9dc: 93 07 05 00  	mv	a5, a0
     9e0: 03 28 81 04  	lw	a6, 72(sp)
     9e4: 7b 40 a5 00  	<unknown>
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     9e8: b3 08 e8 00  	add	a7, a6, a4
;           pDstC[out_idx] = sum;
     9ec: 93 98 28 00  	slli	a7, a7, 2
     9f0: b3 08 13 01  	add	a7, t1, a7
     9f4: 23 a0 08 00  	sw	zero, 0(a7)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     9f8: 13 08 18 00  	addi	a6, a6, 1
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     9fc: 13 87 16 00  	addi	a4, a3, 1
     a00: e3 98 36 fd  	bne	a3, s3, 0x9d0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x9d0>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     a04: 13 86 15 00  	addi	a2, a1, 1
     a08: e3 9c 95 fa  	bne	a1, s1, 0x9c0 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x9c0>
     a0c: 6f 00 c0 05  	j	0xa68 <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa68>
     a10: 03 25 81 04  	lw	a0, 72(sp)
     a14: 33 85 ae 40  	sub	a0, t4, a0
     a18: 83 22 41 03  	lw	t0, 52(sp)
     a1c: 13 07 00 00  	li	a4, 0
     a20: 93 05 06 00  	mv	a1, a2
     a24: 03 26 01 02  	lw	a2, 32(sp)
     a28: 33 86 c5 02  	mul	a2, a1, a2
     a2c: 93 06 07 00  	mv	a3, a4
     a30: 33 07 c7 00  	add	a4, a4, a2
     a34: 33 07 57 02  	mul	a4, a4, t0
     a38: 93 07 05 00  	mv	a5, a0
     a3c: 03 28 81 04  	lw	a6, 72(sp)
     a40: 7b 40 a5 00  	<unknown>
;           uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     a44: b3 08 e8 00  	add	a7, a6, a4
;           pDstC[out_idx] = sum;
     a48: 93 98 28 00  	slli	a7, a7, 2
     a4c: b3 08 13 01  	add	a7, t1, a7
     a50: 23 a0 08 00  	sw	zero, 0(a7)
;         for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
     a54: 13 08 18 00  	addi	a6, a6, 1
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     a58: 13 87 16 00  	addi	a4, a3, 1
     a5c: e3 98 36 fd  	bne	a3, s3, 0xa2c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa2c>
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     a60: 13 86 15 00  	addi	a2, a1, 1
     a64: e3 9c 95 fa  	bne	a1, s1, 0xa1c <PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0xa1c>
; }
     a68: 83 20 c1 07  	lw	ra, 124(sp)
     a6c: 03 24 81 07  	lw	s0, 120(sp)
     a70: 83 24 41 07  	lw	s1, 116(sp)
     a74: 03 29 01 07  	lw	s2, 112(sp)
     a78: 83 29 c1 06  	lw	s3, 108(sp)
     a7c: 03 2a 81 06  	lw	s4, 104(sp)
     a80: 83 2a 41 06  	lw	s5, 100(sp)
     a84: 03 2b 01 06  	lw	s6, 96(sp)
     a88: 83 2b c1 05  	lw	s7, 92(sp)
     a8c: 03 2c 81 05  	lw	s8, 88(sp)
     a90: 83 2c 41 05  	lw	s9, 84(sp)
     a94: 03 2d 01 05  	lw	s10, 80(sp)
     a98: 83 2d c1 04  	lw	s11, 76(sp)
     a9c: 13 01 01 08  	addi	sp, sp, 128
     aa0: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(DWConvolution_fp32.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                           Size     VMA      Type
  0                                                00000000 00000000 
  1 .strtab                                        0000014e 00000000 
  2 .text                                          00000000 00000000 TEXT
  3 .text.PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC 00000930 00000000 TEXT
  4 .debug_loclists                                00001b20 00000000 DEBUG
  5 .debug_abbrev                                  000000d6 00000000 DEBUG
  6 .debug_info                                    000004b1 00000000 DEBUG
  7 .rela.debug_info                               00000330 00000000 
  8 .debug_rnglists                                00000138 00000000 DEBUG
  9 .debug_str_offsets                             00000128 00000000 DEBUG
 10 .rela.debug_str_offsets                        00000360 00000000 
 11 .debug_str                                     00000380 00000000 DEBUG
 12 .debug_addr                                    00000068 00000000 DEBUG
 13 .rela.debug_addr                               00000120 00000000 
 14 .comment                                       00000073 00000000 
 15 .note.GNU-stack                                00000000 00000000 
 16 .riscv.attributes                              00000030 00000000 
 17 .debug_frame                                   00000044 00000000 DEBUG
 18 .rela.debug_frame                              00000060 00000000 
 19 .debug_line                                    00000785 00000000 DEBUG
 20 .rela.debug_line                               000013ec 00000000 
 21 .debug_line_str                                000001fb 00000000 DEBUG
 22 .llvm_addrsig                                  00000000 00000000 
 23 .symtab                                        00001570 00000000 

Disassembly of section .text.PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC:

00000000 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC>:
;     float32_t *__restrict__ pContextBuffer, int nb_dedicated_cores) {
       0: 13 01 01 f5  	addi	sp, sp, -176
       4: 23 26 11 0a  	sw	ra, 172(sp)
       8: 23 24 81 0a  	sw	s0, 168(sp)
       c: 23 22 91 0a  	sw	s1, 164(sp)
      10: 23 20 21 0b  	sw	s2, 160(sp)
      14: 23 2e 31 09  	sw	s3, 156(sp)
      18: 23 2c 41 09  	sw	s4, 152(sp)
      1c: 23 2a 51 09  	sw	s5, 148(sp)
      20: 23 28 61 09  	sw	s6, 144(sp)
      24: 23 26 71 09  	sw	s7, 140(sp)
      28: 23 24 81 09  	sw	s8, 136(sp)
      2c: 23 22 91 09  	sw	s9, 132(sp)
      30: 23 20 a1 09  	sw	s10, 128(sp)
      34: 23 2e b1 07  	sw	s11, 124(sp)
      38: 83 22 81 0d  	lw	t0, 216(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      3c: 73 23 40 f1  	csrr	t1, mhartid
;   return hart_id & 0x01f;
      40: 13 73 f3 01  	andi	t1, t1, 31
;   core_id = core_id % nb_dedicated_cores;
      44: 33 6f 53 02  	rem	t5, t1, t0
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      48: 33 93 02 10  	<unknown>
;       (F_total >> log2Core) + ((F_total & (nb_dedicated_cores - 1)) != 0);
      4c: 33 63 03 10  	<unknown>
      50: 33 d3 67 00  	srl	t1, a5, t1
      54: 93 82 f2 ff  	addi	t0, t0, -1
      58: b3 f2 f2 00  	and	t0, t0, a5
      5c: b3 32 50 00  	snez	t0, t0
      60: b3 02 53 00  	add	t0, t1, t0
;   uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
      64: 33 d3 02 10  	<unknown>
      68: b3 02 e3 03  	mul	t0, t1, t5
      6c: b3 d3 f2 04  	<unknown>
;   uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
      70: b3 d2 03 10  	<unknown>
      74: 33 83 62 00  	add	t1, t0, t1
      78: 23 22 f1 04  	sw	a5, 68(sp)
;   uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
      7c: 33 53 f3 04  	<unknown>
;   uint16_t ch_out_count = ch_out_stop - ch_out_start;
      80: b3 03 73 40  	sub	t2, t1, t2
;   if (ch_out_count == 0) {
      84: b3 d3 03 10  	<unknown>
      88: 23 26 01 03  	sw	a6, 44(sp)
      8c: 23 24 b1 02  	sw	a1, 40(sp)
;   if (ch_out_count == 0) {
      90: e3 82 03 06  	beqz	t2, 0x8f4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8f4>
      94: 83 2f 41 0d  	lw	t6, 212(sp)
      98: 03 28 01 0d  	lw	a6, 208(sp)
      9c: 03 24 c1 0c  	lw	s0, 204(sp)
      a0: 83 23 81 0c  	lw	t2, 200(sp)
      a4: 83 24 41 0c  	lw	s1, 196(sp)
      a8: 83 2e 01 0c  	lw	t4, 192(sp)
      ac: 03 2e c1 0b  	lw	t3, 188(sp)
      b0: 03 29 41 0b  	lw	s2, 180(sp)
      b4: 83 29 01 0b  	lw	s3, 176(sp)
      b8: b3 55 03 10  	<unknown>
      bc: 23 20 b1 06  	sw	a1, 96(sp)
      c0: 83 27 c1 02  	lw	a5, 44(sp)
;   const float32_t *weight_ptr = pSrcB + ch_out_start * P * Q;
      c4: b3 80 f8 02  	mul	ra, a7, a5
      c8: b3 85 50 02  	mul	a1, ra, t0
      cc: 93 95 25 00  	slli	a1, a1, 2
      d0: 33 07 b7 00  	add	a4, a4, a1
;   float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;
      d4: b3 05 1f 02  	mul	a1, t5, ra
      d8: 93 95 25 00  	slli	a1, a1, 2
      dc: 23 20 f1 03  	sw	t6, 32(sp)
      e0: 33 83 bf 00  	add	t1, t6, a1
      e4: 23 2c 61 06  	sw	t1, 120(sp)
      e8: 93 8f 09 00  	mv	t6, s3
;   uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
      ec: 03 23 81 02  	lw	t1, 40(sp)
      f0: 33 03 f3 40  	sub	t1, t1, a5
      f4: 23 2e 91 00  	sw	s1, 28(sp)
      f8: 33 03 93 00  	add	t1, t1, s1
      fc: 33 03 73 00  	add	t1, t1, t2
     100: b3 57 33 03  	divu	a5, t1, s3
;   uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
     104: 33 03 16 41  	sub	t1, a2, a7
     108: 23 22 81 06  	sw	s0, 100(sp)
     10c: 33 03 83 00  	add	t1, t1, s0
     110: 33 08 03 01  	add	a6, t1, a6
     114: 33 53 28 03  	divu	t1, a6, s2
;   if (has_bias) {
     118: 13 78 1e 00  	andi	a6, t3, 1
     11c: 23 2a 61 04  	sw	t1, 84(sp)
;   uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
     120: 13 03 13 00  	addi	t1, t1, 1
     124: 23 22 61 02  	sw	t1, 36(sp)
     128: 23 2e 11 05  	sw	a7, 92(sp)
     12c: 23 2c 21 05  	sw	s2, 88(sp)
     130: 23 2c 31 01  	sw	s3, 24(sp)
     134: 23 2a f1 00  	sw	a5, 20(sp)
;   if (has_bias) {
     138: 63 0a 08 3e  	beqz	a6, 0x52c <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x52c>
     13c: 13 08 f0 ff  	li	a6, -1
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     140: 63 8a 07 7b  	beq	a5, a6, 0x8f4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8f4>
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     144: 83 27 41 02  	lw	a5, 36(sp)
     148: 63 86 07 7a  	beqz	a5, 0x8f4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8f4>
     14c: 13 08 00 00  	li	a6, 0
     150: 03 2b 81 0b  	lw	s6, 184(sp)
     154: 83 27 01 06  	lw	a5, 96(sp)
     158: 13 83 17 00  	addi	t1, a5, 1
     15c: 83 27 41 04  	lw	a5, 68(sp)
     160: b3 db d7 02  	divu	s7, a5, a3
     164: b3 d7 72 03  	divu	a5, t0, s7
     168: 23 26 f1 04  	sw	a5, 76(sp)
     16c: b3 57 73 03  	divu	a5, t1, s7
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     170: 23 2a f1 06  	sw	a5, 116(sp)
     174: 13 9d 28 00  	slli	s10, a7, 2
     178: 83 27 c1 02  	lw	a5, 44(sp)
     17c: b3 07 ff 02  	mul	a5, t5, a5
     180: 93 97 27 00  	slli	a5, a5, 2
     184: 23 22 f1 00  	sw	a5, 4(sp)
     188: 03 23 c1 01  	lw	t1, 28(sp)
     18c: 33 0e 60 40  	neg	t3, t1
     190: 83 27 01 02  	lw	a5, 32(sp)
     194: b3 85 b7 00  	add	a1, a5, a1
     198: 23 20 b1 04  	sw	a1, 64(sp)
     19c: 83 25 41 06  	lw	a1, 100(sp)
     1a0: b3 05 b0 40  	neg	a1, a1
     1a4: 23 20 b1 00  	sw	a1, 0(sp)
     1a8: 53 00 00 f0  	fmv.w.x	ft0, zero
     1ac: 13 0f 03 00  	mv	t5, t1
     1b0: 6f 00 80 02  	j	0x1d8 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x1d8>
     1b4: 83 27 81 00  	lw	a5, 8(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     1b8: 13 88 17 00  	addi	a6, a5, 1
     1bc: 83 2f 81 01  	lw	t6, 24(sp)
     1c0: 03 2e 01 01  	lw	t3, 16(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     1c4: 33 0e fe 01  	add	t3, t3, t6
     1c8: 03 2f c1 00  	lw	t5, 12(sp)
     1cc: 33 0f ff 41  	sub	t5, t5, t6
     1d0: 83 25 41 01  	lw	a1, 20(sp)
     1d4: 63 80 b7 72  	beq	a5, a1, 0x8f4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8f4>
     1d8: 93 03 00 00  	li	t2, 0
     1dc: 13 04 08 00  	mv	s0, a6
     1e0: b3 65 0e 04  	<unknown>
     1e4: b3 85 e5 01  	add	a1, a1, t5
     1e8: 83 27 41 06  	lw	a5, 100(sp)
     1ec: b3 87 b8 42  	<unknown>
     1f0: 23 28 f1 06  	sw	a5, 112(sp)
     1f4: 93 95 25 00  	slli	a1, a1, 2
     1f8: 03 23 41 00  	lw	t1, 4(sp)
     1fc: b3 05 b3 00  	add	a1, t1, a1
     200: 03 28 01 02  	lw	a6, 32(sp)
     204: 93 07 08 00  	mv	a5, a6
     208: b3 87 b8 42  	<unknown>
     20c: 23 28 f1 02  	sw	a5, 48(sp)
     210: 83 27 81 02  	lw	a5, 40(sp)
     214: 23 28 c1 01  	sw	t3, 16(sp)
     218: b3 f5 c7 05  	<unknown>
     21c: 23 26 e1 01  	sw	t5, 12(sp)
     220: b3 85 e5 01  	add	a1, a1, t5
     224: 93 95 25 00  	slli	a1, a1, 2
     228: b3 05 b3 00  	add	a1, t1, a1
     22c: 33 88 b8 42  	<unknown>
     230: 23 2e 01 03  	sw	a6, 60(sp)
     234: b3 05 f4 03  	mul	a1, s0, t6
     238: 03 28 c1 01  	lw	a6, 28(sp)
     23c: 33 88 05 41  	sub	a6, a1, a6
     240: 83 25 c1 02  	lw	a1, 44(sp)
     244: b3 04 b8 00  	add	s1, a6, a1
     248: b3 c5 04 04  	<unknown>
     24c: 33 73 f8 04  	<unknown>
     250: 23 28 61 04  	sw	t1, 80(sp)
     254: 23 24 01 05  	sw	a6, 72(sp)
     258: 33 6e 08 04  	<unknown>
     25c: 33 d3 f4 04  	<unknown>
     260: b3 37 6e 00  	sltu	a5, t3, t1
     264: 93 c7 17 00  	xori	a5, a5, 1
     268: 23 2c f1 02  	sw	a5, 56(sp)
     26c: 83 27 41 02  	lw	a5, 36(sp)
     270: 23 24 81 00  	sw	s0, 8(sp)
     274: b3 07 f4 02  	mul	a5, s0, a5
     278: 23 2a f1 02  	sw	a5, 52(sp)
     27c: 83 27 01 00  	lw	a5, 0(sp)
     280: 23 26 f1 06  	sw	a5, 108(sp)
     284: 6f 00 40 03  	j	0x2b8 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x2b8>
     288: 03 28 81 06  	lw	a6, 104(sp)
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     28c: 93 03 18 00  	addi	t2, a6, 1
     290: 03 29 81 05  	lw	s2, 88(sp)
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     294: 83 27 c1 06  	lw	a5, 108(sp)
     298: b3 87 27 01  	add	a5, a5, s2
     29c: 23 26 f1 06  	sw	a5, 108(sp)
     2a0: 83 27 01 07  	lw	a5, 112(sp)
     2a4: b3 87 27 41  	sub	a5, a5, s2
     2a8: 23 28 f1 06  	sw	a5, 112(sp)
     2ac: 83 28 c1 05  	lw	a7, 92(sp)
     2b0: 83 27 41 05  	lw	a5, 84(sp)
     2b4: e3 00 f8 f0  	beq	a6, a5, 0x1b4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x1b4>
     2b8: 83 2f 81 04  	lw	t6, 72(sp)
     2bc: b3 a7 bf 00  	slt	a5, t6, a1
     2c0: 23 24 71 06  	sw	t2, 104(sp)
;         int32_t w_in_start = w_out * SQ - pad_left;
     2c4: 33 88 23 03  	mul	a6, t2, s2
     2c8: 83 23 41 06  	lw	t2, 100(sp)
     2cc: 33 0a 78 40  	sub	s4, a6, t2
     2d0: b3 0a 1a 01  	add	s5, s4, a7
     2d4: 33 28 5a 01  	slt	a6, s4, s5
;         for (int32_t h_in = (int32_t)h_in_start;
     2d8: b3 f7 07 01  	and	a5, a5, a6
     2dc: 03 2f 81 07  	lw	t5, 120(sp)
     2e0: 63 82 07 02  	beqz	a5, 0x304 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x304>
     2e4: 93 07 0f 00  	mv	a5, t5
     2e8: 93 83 08 00  	mv	t2, a7
     2ec: 7b c0 48 00  	<unknown>
     2f0: 13 00 00 00  	nop
;             im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
     2f4: 2b a2 07 00  	<unknown>
;              h_in < MIN(0, (int32_t)(h_in_start + P)); h_in++) {
     2f8: 93 8f 1f 00  	addi	t6, t6, 1
;         for (int32_t h_in = (int32_t)h_in_start;
     2fc: 33 0f af 01  	add	t5, t5, s10
     300: e3 c2 bf fe  	blt	t6, a1, 0x2e4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x2e4>
     304: 83 27 01 05  	lw	a5, 80(sp)
     308: b3 b7 97 00  	sltu	a5, a5, s1
;         for (uint32_t h_in = MAX(H, h_in_start); h_in < h_in_start + P;
     30c: b3 f7 07 01  	and	a5, a5, a6
     310: 63 88 07 02  	beqz	a5, 0x340 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x340>
     314: 03 2f 01 05  	lw	t5, 80(sp)
;         for (uint32_t h_in = MAX(H, h_in_start); h_in < h_in_start + P;
     318: b3 87 e4 41  	sub	a5, s1, t5
     31c: 03 28 c1 03  	lw	a6, 60(sp)
     320: fb c0 e7 00  	<unknown>
     324: 93 07 08 00  	mv	a5, a6
     328: 93 83 08 00  	mv	t2, a7
     32c: 7b c0 48 00  	<unknown>
     330: 13 00 00 00  	nop
;             im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
     334: 2b a2 07 00  	<unknown>
;              h_in++) {
     338: 13 0f 1f 00  	addi	t5, t5, 1
;         for (uint32_t h_in = MAX(H, h_in_start); h_in < h_in_start + P;
     33c: 33 08 a8 01  	add	a6, a6, s10
     340: 83 28 01 06  	lw	a7, 96(sp)
;         for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
     344: 63 74 6e 08  	bgeu	t3, t1, 0x3cc <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x3cc>
     348: 33 c8 0a 04  	<unknown>
;           for (int32_t w_in = (int32_t)w_in_start;
     34c: 63 5a 0a 03  	bge	s4, a6, 0x380 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x380>
     350: 03 2f 01 03  	lw	t5, 48(sp)
     354: 93 0f 0e 00  	mv	t6, t3
     358: 33 04 48 41  	sub	s0, a6, s4
     35c: 93 07 0f 00  	mv	a5, t5
     360: 93 03 0a 00  	mv	t2, s4
     364: 7b 40 44 00  	<unknown>
;                w_in < MIN(0, (int32_t)(w_in_start + Q)); w_in++) {
     368: 93 83 13 00  	addi	t2, t2, 1
;             im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
     36c: 2b a2 07 00  	<unknown>
;              h_in++) {
     370: 93 8f 1f 00  	addi	t6, t6, 1
;         for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
     374: 33 0f af 01  	add	t5, t5, s10
     378: e3 e0 6f fe  	bltu	t6, t1, 0x358 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x358>
;         for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
     37c: 63 78 6e 04  	bgeu	t3, t1, 0x3cc <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x3cc>
     380: 33 78 ca 04  	<unknown>
;           for (uint32_t w_in = MAX(W, w_in_start); w_in < w_in_start + Q;
     384: 63 74 58 05  	bgeu	a6, s5, 0x3cc <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x3cc>
     388: 83 27 c1 06  	lw	a5, 108(sp)
     38c: b3 77 f6 04  	<unknown>
     390: 83 23 01 07  	lw	t2, 112(sp)
     394: b3 87 77 00  	add	a5, a5, t2
     398: 93 97 27 00  	slli	a5, a5, 2
     39c: 83 23 01 04  	lw	t2, 64(sp)
     3a0: 33 8f f3 00  	add	t5, t2, a5
     3a4: 93 0f 0e 00  	mv	t6, t3
     3a8: 33 84 0a 41  	sub	s0, s5, a6
     3ac: 93 07 0f 00  	mv	a5, t5
     3b0: 93 03 08 00  	mv	t2, a6
     3b4: 7b 40 44 00  	<unknown>
;                w_in++) {
     3b8: 93 83 13 00  	addi	t2, t2, 1
;             im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
     3bc: 2b a2 07 00  	<unknown>
;              h_in++) {
     3c0: 93 8f 1f 00  	addi	t6, t6, 1
;         for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
     3c4: 33 0f af 01  	add	t5, t5, s10
     3c8: e3 e0 6f fe  	bltu	t6, t1, 0x3a8 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x3a8>
;         for (uint32_t c = ch_out_start / (F_total / C);
     3cc: 83 27 c1 04  	lw	a5, 76(sp)
     3d0: 03 28 41 07  	lw	a6, 116(sp)
     3d4: e3 fa 07 eb  	bgeu	a5, a6, 0x288 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x288>
     3d8: 83 27 c1 06  	lw	a5, 108(sp)
     3dc: b3 e7 07 04  	<unknown>
     3e0: 03 28 01 07  	lw	a6, 112(sp)
     3e4: b3 87 07 01  	add	a5, a5, a6
     3e8: 93 97 27 00  	slli	a5, a5, 2
     3ec: 03 28 01 04  	lw	a6, 64(sp)
     3f0: b3 0d f8 00  	add	s11, a6, a5
     3f4: 33 6a 0a 04  	<unknown>
     3f8: b3 da ca 04  	<unknown>
     3fc: b3 37 5a 01  	sltu	a5, s4, s5
     400: 93 c7 17 00  	xori	a5, a5, 1
     404: 03 28 41 03  	lw	a6, 52(sp)
     408: 83 23 81 06  	lw	t2, 104(sp)
     40c: 33 88 03 01  	add	a6, t2, a6
     410: 83 23 41 04  	lw	t2, 68(sp)
     414: 33 0c 78 02  	mul	s8, a6, t2
     418: 03 28 81 03  	lw	a6, 56(sp)
     41c: b3 67 f8 00  	or	a5, a6, a5
     420: 83 2c c1 04  	lw	s9, 76(sp)
     424: 6f 00 c0 00  	j	0x430 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x430>
;         for (uint32_t c = ch_out_start / (F_total / C);
     428: 03 28 41 07  	lw	a6, 116(sp)
     42c: e3 8e 0c e5  	beq	s9, a6, 0x288 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x288>
     430: 13 88 0d 00  	mv	a6, s11
     434: 13 0f 0e 00  	mv	t5, t3
;           for (uint32_t h_in = MAX(0, h_in_start);
     438: 63 94 07 04  	bnez	a5, 0x480 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x480>
     43c: b3 89 4a 41  	sub	s3, s5, s4
     440: b3 03 cf 02  	mul	t2, t5, a2
     444: 13 04 08 00  	mv	s0, a6
     448: 13 09 0a 00  	mv	s2, s4
     44c: 7b c0 29 01  	<unknown>
;               uint32_t in_idx = (h_in * W + w_in) * C + c;
     450: b3 09 79 00  	add	s3, s2, t2
     454: 93 8f 0c 00  	mv	t6, s9
     458: b3 8f d9 42  	<unknown>
;                   pSrcA[in_idx];
     45c: 93 9f 2f 00  	slli	t6, t6, 2
     460: b3 0f f5 01  	add	t6, a0, t6
     464: 87 a0 0f 00  	flw	ft1, 0(t6)
;               im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] =
     468: 27 20 14 00  	fsw	ft1, 0(s0)
;                  w_in < MIN(W, w_in_start + Q); w_in++) {
     46c: 13 09 19 00  	addi	s2, s2, 1
     470: 13 04 44 00  	addi	s0, s0, 4
;                h_in < MIN(H, h_in_start + P); h_in++) {
     474: 13 0f 1f 00  	addi	t5, t5, 1
;           for (uint32_t h_in = MAX(0, h_in_start);
     478: 33 08 a8 01  	add	a6, a6, s10
     47c: e3 60 6f fc  	bltu	t5, t1, 0x43c <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x43c>
;           if (c * (F_total / C) < ch_out_start) {
     480: 33 88 7c 03  	mul	a6, s9, s7
     484: 33 7f 58 04  	<unknown>
;           if ((c + 1) * (F_total / C) < ch_out_stop) {
     488: 93 8c 1c 00  	addi	s9, s9, 1
     48c: 33 88 7c 03  	mul	a6, s9, s7
     490: 33 58 18 05  	<unknown>
;           for (uint32_t f = lower_f; f < upper_f; f++) {
     494: e3 7a 0f f9  	bgeu	t5, a6, 0x428 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x428>
     498: 63 86 00 06  	beqz	ra, 0x504 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x504>
     49c: 13 09 00 00  	li	s2, 0
     4a0: b3 03 5f 40  	sub	t2, t5, t0
     4a4: b3 83 70 02  	mul	t2, ra, t2
     4a8: 03 24 81 07  	lw	s0, 120(sp)
     4ac: 93 89 00 00  	mv	s3, ra
     4b0: d3 00 00 20  	fmv.s	ft1, ft0
     4b4: 7b c0 00 01  	<unknown>
;                   im2col_buffer[im2col_idx] *
     4b8: 07 21 04 00  	flw	ft2, 0(s0)
;                   weight_ptr[(f - ch_out_start) * P * Q + im2col_idx % (P * Q)];
     4bc: b3 0f 79 00  	add	t6, s2, t2
     4c0: 93 9f 2f 00  	slli	t6, t6, 2
     4c4: b3 0f f7 01  	add	t6, a4, t6
     4c8: 87 a1 0f 00  	flw	ft3, 0(t6)
;               sum +=
     4cc: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;             for (uint32_t im2col_idx = 0; im2col_idx < P * Q; im2col_idx++) {
     4d0: 13 09 19 00  	addi	s2, s2, 1
     4d4: 13 04 44 00  	addi	s0, s0, 4
;             pDstC[out_idx] = sum + pSrcBias[f];
     4d8: 93 13 2f 00  	slli	t2, t5, 2
     4dc: b3 03 7b 00  	add	t2, s6, t2
     4e0: 07 a1 03 00  	flw	ft2, 0(t2)
;             uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     4e4: b3 03 8f 01  	add	t2, t5, s8
;             pDstC[out_idx] = sum + pSrcBias[f];
     4e8: d3 70 11 00  	fadd.s	ft1, ft2, ft1
     4ec: 93 93 23 00  	slli	t2, t2, 2
     4f0: b3 83 7e 00  	add	t2, t4, t2
;           for (uint32_t f = lower_f; f < upper_f; f++) {
     4f4: 13 0f 1f 00  	addi	t5, t5, 1
;             pDstC[out_idx] = sum + pSrcBias[f];
     4f8: 27 a0 13 00  	fsw	ft1, 0(t2)
     4fc: e3 60 0f fb  	bltu	t5, a6, 0x49c <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x49c>
     500: 6f f0 9f f2  	j	0x428 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x428>
;             uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     504: b3 03 8f 01  	add	t2, t5, s8
;             pDstC[out_idx] = sum + pSrcBias[f];
     508: 13 14 2f 00  	slli	s0, t5, 2
     50c: 33 04 8b 00  	add	s0, s6, s0
     510: 87 20 04 00  	flw	ft1, 0(s0)
     514: 93 93 23 00  	slli	t2, t2, 2
     518: b3 83 7e 00  	add	t2, t4, t2
;           for (uint32_t f = lower_f; f < upper_f; f++) {
     51c: 13 0f 1f 00  	addi	t5, t5, 1
;             pDstC[out_idx] = sum + pSrcBias[f];
     520: 27 a0 13 00  	fsw	ft1, 0(t2)
;           for (uint32_t f = lower_f; f < upper_f; f++) {
     524: e3 60 0f ff  	bltu	t5, a6, 0x504 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x504>
     528: 6f f0 1f f0  	j	0x428 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x428>
     52c: 13 08 f0 ff  	li	a6, -1
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     530: 63 82 07 3d  	beq	a5, a6, 0x8f4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8f4>
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     534: 83 27 41 02  	lw	a5, 36(sp)
     538: 63 8e 07 3a  	beqz	a5, 0x8f4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8f4>
     53c: 13 08 00 00  	li	a6, 0
     540: 83 27 01 06  	lw	a5, 96(sp)
     544: 13 83 17 00  	addi	t1, a5, 1
     548: 83 27 41 04  	lw	a5, 68(sp)
     54c: b3 da d7 02  	divu	s5, a5, a3
     550: b3 d7 52 03  	divu	a5, t0, s5
     554: 23 28 f1 04  	sw	a5, 80(sp)
     558: 33 5b 53 03  	divu	s6, t1, s5
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     55c: 93 9b 28 00  	slli	s7, a7, 2
     560: 83 27 c1 02  	lw	a5, 44(sp)
     564: b3 07 ff 02  	mul	a5, t5, a5
     568: 93 97 27 00  	slli	a5, a5, 2
     56c: 23 24 f1 00  	sw	a5, 8(sp)
     570: 03 23 c1 01  	lw	t1, 28(sp)
     574: 33 0e 60 40  	neg	t3, t1
     578: 83 27 01 02  	lw	a5, 32(sp)
     57c: b3 85 b7 00  	add	a1, a5, a1
     580: 23 24 b1 04  	sw	a1, 72(sp)
     584: 83 25 41 06  	lw	a1, 100(sp)
     588: b3 05 b0 40  	neg	a1, a1
     58c: 23 22 b1 00  	sw	a1, 4(sp)
     590: 53 00 00 f0  	fmv.w.x	ft0, zero
     594: 13 0f 03 00  	mv	t5, t1
     598: 6f 00 80 02  	j	0x5c0 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x5c0>
     59c: 83 27 c1 00  	lw	a5, 12(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     5a0: 13 88 17 00  	addi	a6, a5, 1
     5a4: 83 2f 81 01  	lw	t6, 24(sp)
     5a8: 03 2e 01 03  	lw	t3, 48(sp)
;     for (uint32_t h_out = 0; h_out < H_out; h_out++) {
     5ac: 33 0e fe 01  	add	t3, t3, t6
     5b0: 03 2f 01 01  	lw	t5, 16(sp)
     5b4: 33 0f ff 41  	sub	t5, t5, t6
     5b8: 83 25 41 01  	lw	a1, 20(sp)
     5bc: 63 8c b7 32  	beq	a5, a1, 0x8f4 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8f4>
     5c0: 93 05 00 00  	li	a1, 0
     5c4: 93 04 08 00  	mv	s1, a6
     5c8: b3 67 0e 04  	<unknown>
     5cc: b3 87 e7 01  	add	a5, a5, t5
     5d0: 03 28 41 06  	lw	a6, 100(sp)
     5d4: 33 88 f8 42  	<unknown>
     5d8: 23 2a 01 07  	sw	a6, 116(sp)
     5dc: 93 97 27 00  	slli	a5, a5, 2
     5e0: 83 23 81 00  	lw	t2, 8(sp)
     5e4: b3 87 f3 00  	add	a5, t2, a5
     5e8: 03 23 01 02  	lw	t1, 32(sp)
     5ec: 13 08 03 00  	mv	a6, t1
     5f0: 33 88 f8 42  	<unknown>
     5f4: 23 2a 01 03  	sw	a6, 52(sp)
     5f8: 03 28 81 02  	lw	a6, 40(sp)
     5fc: 23 28 c1 03  	sw	t3, 48(sp)
     600: b3 77 c8 05  	<unknown>
     604: 23 28 e1 01  	sw	t5, 16(sp)
     608: b3 87 e7 01  	add	a5, a5, t5
     60c: 93 97 27 00  	slli	a5, a5, 2
     610: b3 87 f3 00  	add	a5, t2, a5
     614: 33 83 f8 42  	<unknown>
     618: 23 20 61 04  	sw	t1, 64(sp)
     61c: b3 87 f4 03  	mul	a5, s1, t6
     620: 03 23 c1 01  	lw	t1, 28(sp)
     624: b3 83 67 40  	sub	t2, a5, t1
     628: 83 27 c1 02  	lw	a5, 44(sp)
     62c: b3 8f f3 00  	add	t6, t2, a5
     630: 33 c3 0f 04  	<unknown>
     634: b3 f7 03 05  	<unknown>
     638: 23 24 f1 06  	sw	a5, 104(sp)
     63c: 23 26 71 04  	sw	t2, 76(sp)
     640: 33 e4 03 04  	<unknown>
     644: b3 dc 0f 05  	<unknown>
     648: b3 37 94 01  	sltu	a5, s0, s9
     64c: 93 c7 17 00  	xori	a5, a5, 1
     650: 23 2e f1 02  	sw	a5, 60(sp)
     654: 83 27 41 02  	lw	a5, 36(sp)
     658: 23 26 91 00  	sw	s1, 12(sp)
     65c: b3 87 f4 02  	mul	a5, s1, a5
     660: 23 2c f1 02  	sw	a5, 56(sp)
     664: 83 27 41 00  	lw	a5, 4(sp)
     668: 23 28 f1 06  	sw	a5, 112(sp)
     66c: 6f 00 40 03  	j	0x6a0 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x6a0>
     670: 03 28 c1 06  	lw	a6, 108(sp)
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     674: 93 05 18 00  	addi	a1, a6, 1
     678: 03 29 81 05  	lw	s2, 88(sp)
;       for (uint32_t w_out = 0; w_out < W_out; w_out++) {
     67c: 83 27 01 07  	lw	a5, 112(sp)
     680: b3 87 27 01  	add	a5, a5, s2
     684: 23 28 f1 06  	sw	a5, 112(sp)
     688: 83 27 41 07  	lw	a5, 116(sp)
     68c: b3 87 27 41  	sub	a5, a5, s2
     690: 23 2a f1 06  	sw	a5, 116(sp)
     694: 83 28 c1 05  	lw	a7, 92(sp)
     698: 83 27 41 05  	lw	a5, 84(sp)
     69c: e3 00 f8 f0  	beq	a6, a5, 0x59c <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x59c>
     6a0: 93 87 05 00  	mv	a5, a1
     6a4: 03 2f c1 04  	lw	t5, 76(sp)
     6a8: b3 25 6f 00  	slt	a1, t5, t1
     6ac: 23 26 f1 06  	sw	a5, 108(sp)
;         int32_t w_in_start = w_out * SQ - pad_left;
     6b0: b3 87 27 03  	mul	a5, a5, s2
     6b4: 03 28 41 06  	lw	a6, 100(sp)
     6b8: 33 8d 07 41  	sub	s10, a5, a6
     6bc: b3 07 1d 01  	add	a5, s10, a7
     6c0: b3 23 fd 00  	slt	t2, s10, a5
;         for (int32_t h_in = (int32_t)h_in_start;
     6c4: b3 f5 75 00  	and	a1, a1, t2
     6c8: 03 2e 81 07  	lw	t3, 120(sp)
     6cc: 63 82 05 02  	beqz	a1, 0x6f0 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x6f0>
     6d0: 93 05 0e 00  	mv	a1, t3
     6d4: 13 88 08 00  	mv	a6, a7
     6d8: 7b c0 48 00  	<unknown>
     6dc: 13 00 00 00  	nop
;             im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
     6e0: 2b a2 05 00  	<unknown>
;              h_in < MIN(0, (int32_t)(h_in_start + P)); h_in++) {
     6e4: 13 0f 1f 00  	addi	t5, t5, 1
;         for (int32_t h_in = (int32_t)h_in_start;
     6e8: 33 0e 7e 01  	add	t3, t3, s7
     6ec: e3 42 6f fe  	blt	t5, t1, 0x6d0 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x6d0>
     6f0: 83 25 81 06  	lw	a1, 104(sp)
     6f4: b3 b5 f5 01  	sltu	a1, a1, t6
;         for (uint32_t h_in = MAX(H, h_in_start); h_in < h_in_start + P;
     6f8: b3 f5 75 00  	and	a1, a1, t2
     6fc: 63 88 05 02  	beqz	a1, 0x72c <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x72c>
     700: 03 2e 81 06  	lw	t3, 104(sp)
;         for (uint32_t h_in = MAX(H, h_in_start); h_in < h_in_start + P;
     704: b3 85 cf 41  	sub	a1, t6, t3
     708: 83 23 01 04  	lw	t2, 64(sp)
     70c: fb c0 e5 00  	<unknown>
     710: 93 85 03 00  	mv	a1, t2
     714: 13 88 08 00  	mv	a6, a7
     718: 7b c0 48 00  	<unknown>
     71c: 13 00 00 00  	nop
;             im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
     720: 2b a2 05 00  	<unknown>
;              h_in++) {
     724: 13 0e 1e 00  	addi	t3, t3, 1
;         for (uint32_t h_in = MAX(H, h_in_start); h_in < h_in_start + P;
     728: b3 83 73 01  	add	t2, t2, s7
     72c: 83 28 01 06  	lw	a7, 96(sp)
;         for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
     730: 63 74 94 09  	bgeu	s0, s9, 0x7b8 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x7b8>
     734: b3 c3 07 04  	<unknown>
;           for (int32_t w_in = (int32_t)w_in_start;
     738: 63 5a 7d 02  	bge	s10, t2, 0x76c <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x76c>
     73c: 03 2e 41 03  	lw	t3, 52(sp)
     740: 13 0f 04 00  	mv	t5, s0
     744: b3 84 a3 41  	sub	s1, t2, s10
     748: 93 05 0e 00  	mv	a1, t3
     74c: 13 08 0d 00  	mv	a6, s10
     750: 7b c0 44 00  	<unknown>
;                w_in < MIN(0, (int32_t)(w_in_start + Q)); w_in++) {
     754: 13 08 18 00  	addi	a6, a6, 1
;             im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
     758: 2b a2 05 00  	<unknown>
;              h_in++) {
     75c: 13 0f 1f 00  	addi	t5, t5, 1
;         for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
     760: 33 0e 7e 01  	add	t3, t3, s7
     764: e3 60 9f ff  	bltu	t5, s9, 0x744 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x744>
;         for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
     768: 63 78 94 05  	bgeu	s0, s9, 0x7b8 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x7b8>
     76c: b3 73 cd 04  	<unknown>
;           for (uint32_t w_in = MAX(W, w_in_start); w_in < w_in_start + Q;
     770: 63 f4 f3 04  	bgeu	t2, a5, 0x7b8 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x7b8>
     774: 83 25 01 07  	lw	a1, 112(sp)
     778: b3 75 b6 04  	<unknown>
     77c: 03 28 41 07  	lw	a6, 116(sp)
     780: b3 85 05 01  	add	a1, a1, a6
     784: 93 95 25 00  	slli	a1, a1, 2
     788: 03 28 81 04  	lw	a6, 72(sp)
     78c: 33 0e b8 00  	add	t3, a6, a1
     790: 13 0f 04 00  	mv	t5, s0
     794: b3 84 77 40  	sub	s1, a5, t2
     798: 93 05 0e 00  	mv	a1, t3
     79c: 13 88 03 00  	mv	a6, t2
     7a0: 7b c0 44 00  	<unknown>
;                w_in++) {
     7a4: 13 08 18 00  	addi	a6, a6, 1
;             im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] = 0.0f;
     7a8: 2b a2 05 00  	<unknown>
;              h_in++) {
     7ac: 13 0f 1f 00  	addi	t5, t5, 1
;         for (uint32_t h_in = MAX(0, h_in_start); h_in < MIN(H, h_in_start + P);
     7b0: 33 0e 7e 01  	add	t3, t3, s7
     7b4: e3 60 9f ff  	bltu	t5, s9, 0x794 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x794>
;         for (uint32_t c = ch_out_start / (F_total / C);
     7b8: 83 25 01 05  	lw	a1, 80(sp)
     7bc: e3 fa 65 eb  	bgeu	a1, s6, 0x670 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x670>
     7c0: 83 25 01 07  	lw	a1, 112(sp)
     7c4: b3 e5 05 04  	<unknown>
     7c8: 03 28 41 07  	lw	a6, 116(sp)
     7cc: b3 85 05 01  	add	a1, a1, a6
     7d0: 93 95 25 00  	slli	a1, a1, 2
     7d4: 03 28 81 04  	lw	a6, 72(sp)
     7d8: 33 0a b8 00  	add	s4, a6, a1
     7dc: 33 6d 0d 04  	<unknown>
     7e0: b3 d7 c7 04  	<unknown>
     7e4: b3 35 fd 00  	sltu	a1, s10, a5
     7e8: 93 c5 15 00  	xori	a1, a1, 1
     7ec: 03 28 81 03  	lw	a6, 56(sp)
     7f0: 83 23 c1 06  	lw	t2, 108(sp)
     7f4: 33 88 03 01  	add	a6, t2, a6
     7f8: 83 23 41 04  	lw	t2, 68(sp)
     7fc: b3 0d 78 02  	mul	s11, a6, t2
     800: 03 28 c1 03  	lw	a6, 60(sp)
     804: 33 6e b8 00  	or	t3, a6, a1
     808: 03 2c 01 05  	lw	s8, 80(sp)
     80c: 6f 00 80 00  	j	0x814 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x814>
;         for (uint32_t c = ch_out_start / (F_total / C);
     810: e3 00 6c e7  	beq	s8, s6, 0x670 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x670>
     814: 93 03 0a 00  	mv	t2, s4
     818: 13 0f 04 00  	mv	t5, s0
;           for (uint32_t h_in = MAX(0, h_in_start);
     81c: 63 14 0e 04  	bnez	t3, 0x864 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x864>
     820: 33 89 a7 41  	sub	s2, a5, s10
     824: b3 05 cf 02  	mul	a1, t5, a2
     828: 13 88 03 00  	mv	a6, t2
     82c: 93 04 0d 00  	mv	s1, s10
     830: 7b 40 29 01  	<unknown>
;               uint32_t in_idx = (h_in * W + w_in) * C + c;
     834: 33 89 b4 00  	add	s2, s1, a1
     838: 93 09 0c 00  	mv	s3, s8
     83c: b3 09 d9 42  	<unknown>
;                   pSrcA[in_idx];
     840: 13 99 29 00  	slli	s2, s3, 2
     844: 33 09 25 01  	add	s2, a0, s2
     848: 87 20 09 00  	flw	ft1, 0(s2)
;               im2col_buffer[(h_in - h_in_start) * Q + (w_in - w_in_start)] =
     84c: 27 20 18 00  	fsw	ft1, 0(a6)
;                  w_in < MIN(W, w_in_start + Q); w_in++) {
     850: 93 84 14 00  	addi	s1, s1, 1
     854: 13 08 48 00  	addi	a6, a6, 4
;                h_in < MIN(H, h_in_start + P); h_in++) {
     858: 13 0f 1f 00  	addi	t5, t5, 1
;           for (uint32_t h_in = MAX(0, h_in_start);
     85c: b3 83 73 01  	add	t2, t2, s7
     860: e3 60 9f fd  	bltu	t5, s9, 0x820 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x820>
;           if (c * (F_total / C) < ch_out_start) {
     864: b3 05 5c 03  	mul	a1, s8, s5
     868: 33 ff 55 04  	<unknown>
;           if ((c + 1) * (F_total / C) < ch_out_stop) {
     86c: 13 0c 1c 00  	addi	s8, s8, 1
     870: b3 05 5c 03  	mul	a1, s8, s5
     874: b3 d3 15 05  	<unknown>
;           for (uint32_t f = lower_f; f < upper_f; f++) {
     878: e3 7c 7f f8  	bgeu	t5, t2, 0x810 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x810>
     87c: 63 8e 00 04  	beqz	ra, 0x8d8 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8d8>
     880: 93 05 00 00  	li	a1, 0
     884: 33 08 5f 40  	sub	a6, t5, t0
     888: 33 88 00 03  	mul	a6, ra, a6
     88c: 03 29 81 07  	lw	s2, 120(sp)
     890: 93 84 00 00  	mv	s1, ra
     894: d3 00 00 20  	fmv.s	ft1, ft0
     898: 7b c0 00 01  	<unknown>
;                   im2col_buffer[im2col_idx] *
     89c: 07 21 09 00  	flw	ft2, 0(s2)
;                   weight_ptr[(f - ch_out_start) * P * Q + im2col_idx % (P * Q)];
     8a0: b3 89 05 01  	add	s3, a1, a6
     8a4: 93 99 29 00  	slli	s3, s3, 2
     8a8: b3 09 37 01  	add	s3, a4, s3
     8ac: 87 a1 09 00  	flw	ft3, 0(s3)
;               sum +=
     8b0: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;             for (uint32_t im2col_idx = 0; im2col_idx < P * Q; im2col_idx++) {
     8b4: 93 85 15 00  	addi	a1, a1, 1
     8b8: 13 09 49 00  	addi	s2, s2, 4
;             uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     8bc: b3 05 bf 01  	add	a1, t5, s11
;             pDstC[out_idx] = sum;
     8c0: 93 95 25 00  	slli	a1, a1, 2
     8c4: b3 85 be 00  	add	a1, t4, a1
;           for (uint32_t f = lower_f; f < upper_f; f++) {
     8c8: 13 0f 1f 00  	addi	t5, t5, 1
;             pDstC[out_idx] = sum;
     8cc: 27 a0 15 00  	fsw	ft1, 0(a1)
     8d0: e3 68 7f fa  	bltu	t5, t2, 0x880 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x880>
     8d4: 6f f0 df f3  	j	0x810 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x810>
;             uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;
     8d8: b3 05 bf 01  	add	a1, t5, s11
;             pDstC[out_idx] = sum;
     8dc: 93 95 25 00  	slli	a1, a1, 2
     8e0: b3 85 be 00  	add	a1, t4, a1
;           for (uint32_t f = lower_f; f < upper_f; f++) {
     8e4: 13 0f 1f 00  	addi	t5, t5, 1
;             pDstC[out_idx] = sum;
     8e8: 23 a0 05 00  	sw	zero, 0(a1)
;           for (uint32_t f = lower_f; f < upper_f; f++) {
     8ec: e3 66 7f fe  	bltu	t5, t2, 0x8d8 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x8d8>
     8f0: 6f f0 1f f2  	j	0x810 <PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC+0x810>
; }
     8f4: 83 20 c1 0a  	lw	ra, 172(sp)
     8f8: 03 24 81 0a  	lw	s0, 168(sp)
     8fc: 83 24 41 0a  	lw	s1, 164(sp)
     900: 03 29 01 0a  	lw	s2, 160(sp)
     904: 83 29 c1 09  	lw	s3, 156(sp)
     908: 03 2a 81 09  	lw	s4, 152(sp)
     90c: 83 2a 41 09  	lw	s5, 148(sp)
     910: 03 2b 01 09  	lw	s6, 144(sp)
     914: 83 2b c1 08  	lw	s7, 140(sp)
     918: 03 2c 81 08  	lw	s8, 136(sp)
     91c: 83 2c 41 08  	lw	s9, 132(sp)
     920: 03 2d 01 08  	lw	s10, 128(sp)
     924: 83 2d c1 07  	lw	s11, 124(sp)
     928: 13 01 01 0b  	addi	sp, sp, 176
     92c: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(GELU.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                   Size     VMA      Type
  0                                        00000000 00000000 
  1 .strtab                                00000196 00000000 
  2 .text                                  00000000 00000000 TEXT
  3 .sdata                                 00000014 00000000 DATA
  4 .text.PULP_GELU_fp32_fp32              000000ec 00000000 TEXT
  5 .rela.text.PULP_GELU_fp32_fp32         00000054 00000000 
  6 .text.PULP_GELU_fp32_fp32_sigmoid      000000d8 00000000 TEXT
  7 .rela.text.PULP_GELU_fp32_fp32_sigmoid 0000003c 00000000 
  8 .debug_loclists                        00000209 00000000 DEBUG
  9 .debug_abbrev                          000000c8 00000000 DEBUG
 10 .debug_info                            0000029f 00000000 DEBUG
 11 .rela.debug_info                       000001c8 00000000 
 12 .debug_rnglists                        00000019 00000000 DEBUG
 13 .debug_str_offsets                     000000b0 00000000 DEBUG
 14 .rela.debug_str_offsets                000001f8 00000000 
 15 .debug_str                             00000279 00000000 DEBUG
 16 .debug_addr                            00000038 00000000 DEBUG
 17 .rela.debug_addr                       00000090 00000000 
 18 .comment                               00000073 00000000 
 19 .note.GNU-stack                        00000000 00000000 
 20 .riscv.attributes                      00000030 00000000 
 21 .debug_frame                           0000005c 00000000 DEBUG
 22 .rela.debug_frame                      000000c0 00000000 
 23 .debug_line                            00000293 00000000 DEBUG
 24 .rela.debug_line                       00000570 00000000 
 25 .debug_line_str                        000001df 00000000 DEBUG
 26 .llvm_addrsig                          00000000 00000000 
 27 .symtab                                000008d0 00000000 

Disassembly of section .text.PULP_GELU_fp32_fp32:

00000000 <PULP_GELU_fp32_fp32>:
;                          int32_t dataSize, int nb_dedicated_cores) {
       0: 13 01 01 fe  	addi	sp, sp, -32
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 27 26 81 00  	fsw	fs0, 12(sp)
      18: 27 24 91 00  	fsw	fs1, 8(sp)
      1c: 27 22 21 01  	fsw	fs2, 4(sp)
      20: 27 20 31 01  	fsw	fs3, 0(sp)
      24: 73 27 40 f1  	csrr	a4, mhartid
;   return hart_id & 0x01f;
      28: 13 77 f7 01  	andi	a4, a4, 31
;   core_id = core_id % nb_dedicated_cores;
      2c: 33 67 d7 02  	rem	a4, a4, a3
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      30: b3 97 06 10  	<unknown>
;   int32_t chunk = (dataSize >> log2Core) + ((dataSize & (nb_dedicated_cores - 1)) != 0);
      34: b3 e7 07 10  	<unknown>
      38: b3 57 f6 40  	sra	a5, a2, a5
      3c: 93 86 f6 ff  	addi	a3, a3, -1
      40: b3 f6 c6 00  	and	a3, a3, a2
      44: b3 36 d0 00  	snez	a3, a3
      48: b3 87 d7 00  	add	a5, a5, a3
;   int32_t chunk_start = MIN(chunk * core_id, dataSize);
      4c: b3 86 e7 02  	mul	a3, a5, a4
      50: b3 c6 c6 04  	<unknown>
;   int32_t chunk_stop = MIN(chunk_start + chunk, dataSize);
      54: 33 87 f6 00  	add	a4, a3, a5
      58: 33 46 c7 04  	<unknown>
;   for (int32_t i = chunk_start; i < chunk_stop; i++) {
      5c: 63 d4 c6 06  	bge	a3, a2, 0xc4 <PULP_GELU_fp32_fp32+0xc4>
      60: 33 86 c6 40  	sub	a2, a3, a2
      64: 37 07 00 00  	lui	a4, 0
      68: 07 24 07 00  	flw	fs0, 0(a4)
      6c: 37 07 00 00  	lui	a4, 0
      70: 87 24 07 00  	flw	fs1, 0(a4)
      74: 37 07 00 00  	lui	a4, 0
      78: 07 29 07 00  	flw	fs2, 0(a4)
;   for (int32_t i = chunk_start; i < chunk_stop; i++) {
      7c: 93 96 26 00  	slli	a3, a3, 2
      80: 33 84 d5 00  	add	s0, a1, a3
      84: b3 04 d5 00  	add	s1, a0, a3
;     float32_t x = data_in[i];
      88: 87 a9 04 00  	flw	fs3, 0(s1)
      8c: 13 09 06 00  	mv	s2, a2
;                                           (x + 0.044715f * powf(x, 3.0f)))));
      90: 53 f0 39 11  	fmul.s	ft0, fs3, fs3
      94: 53 70 80 10  	fmul.s	ft0, ft0, fs0
      98: 43 70 30 99  	fmadd.s	ft0, ft0, fs3, fs3
;     float32_t cdf = 0.5f * (1.0f + tanhf((sqrtf(2.0f / (float)M_PI) *
      9c: 53 75 90 10  	fmul.s	fa0, ft0, fs1
      a0: 97 00 00 00  	auipc	ra, 0
      a4: e7 80 00 00  	jalr	ra
      a8: 43 70 25 91  	fmadd.s	ft0, fa0, fs2, fs2
;     data_out[i] = x * cdf;
      ac: 53 70 30 11  	fmul.s	ft0, ft0, fs3
      b0: 27 20 04 00  	fsw	ft0, 0(s0)
;   for (int32_t i = chunk_start; i < chunk_stop; i++) {
      b4: 13 06 19 00  	addi	a2, s2, 1
      b8: 13 04 44 00  	addi	s0, s0, 4
      bc: 93 84 44 00  	addi	s1, s1, 4
      c0: e3 74 26 fd  	bgeu	a2, s2, 0x88 <PULP_GELU_fp32_fp32+0x88>
; }
      c4: 83 20 c1 01  	lw	ra, 28(sp)
      c8: 03 24 81 01  	lw	s0, 24(sp)
      cc: 83 24 41 01  	lw	s1, 20(sp)
      d0: 03 29 01 01  	lw	s2, 16(sp)
      d4: 07 24 c1 00  	flw	fs0, 12(sp)
      d8: 87 24 81 00  	flw	fs1, 8(sp)
      dc: 07 29 41 00  	flw	fs2, 4(sp)
      e0: 87 29 01 00  	flw	fs3, 0(sp)
      e4: 13 01 01 02  	addi	sp, sp, 32
      e8: 67 80 00 00  	ret

Disassembly of section .text.PULP_GELU_fp32_fp32_sigmoid:

00000000 <PULP_GELU_fp32_fp32_sigmoid>:
;                                  int32_t dataSize, int nb_dedicated_cores) {
       0: 13 01 01 fe  	addi	sp, sp, -32
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
       4: 23 2e 11 00  	sw	ra, 28(sp)
       8: 23 2c 81 00  	sw	s0, 24(sp)
       c: 23 2a 91 00  	sw	s1, 20(sp)
      10: 23 28 21 01  	sw	s2, 16(sp)
      14: 27 26 81 00  	fsw	fs0, 12(sp)
      18: 27 24 91 00  	fsw	fs1, 8(sp)
      1c: 27 22 21 01  	fsw	fs2, 4(sp)
      20: 73 27 40 f1  	csrr	a4, mhartid
;   return hart_id & 0x01f;
      24: 13 77 f7 01  	andi	a4, a4, 31
;   core_id = core_id % nb_dedicated_cores;
      28: 33 67 d7 02  	rem	a4, a4, a3
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      2c: b3 97 06 10  	<unknown>
;   int16_t chunk = (dataSize >> log2Core) + ((dataSize & (nb_dedicated_cores - 1)) != 0);
      30: b3 e7 07 10  	<unknown>
      34: b3 57 f6 40  	sra	a5, a2, a5
      38: 93 86 f6 ff  	addi	a3, a3, -1
      3c: b3 f6 c6 00  	and	a3, a3, a2
      40: b3 36 d0 00  	snez	a3, a3
      44: b3 86 d7 00  	add	a3, a5, a3
;   int16_t chunk_start = MIN(chunk * core_id, dataSize);
      48: b3 c7 06 10  	<unknown>
      4c: b3 86 e7 02  	mul	a3, a5, a4
      50: b3 c6 c6 04  	<unknown>
;   int16_t chunk_stop = MIN(chunk_start + chunk, dataSize);
      54: b3 c6 06 10  	<unknown>
      58: 33 87 f6 00  	add	a4, a3, a5
      5c: 33 46 c7 04  	<unknown>
      60: 33 46 06 10  	<unknown>
;   for (uint32_t i = chunk_start; i < chunk_stop; i++) {
      64: 63 f8 c6 04  	bgeu	a3, a2, 0xb4 <PULP_GELU_fp32_fp32_sigmoid+0xb4>
      68: 33 04 d6 40  	sub	s0, a2, a3
      6c: 37 06 00 00  	lui	a2, 0
      70: 07 24 06 00  	flw	fs0, 0(a2)
      74: 37 06 00 00  	lui	a2, 0
      78: 87 24 06 00  	flw	fs1, 0(a2)
      7c: 13 96 26 00  	slli	a2, a3, 2
      80: b3 84 c5 00  	add	s1, a1, a2
      84: 33 09 c5 00  	add	s2, a0, a2
;     float32_t x = data_in[i];
      88: 07 29 09 00  	flw	fs2, 0(s2)
;     float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
      8c: 53 75 89 10  	fmul.s	fa0, fs2, fs0
      90: 97 00 00 00  	auipc	ra, 0
      94: e7 80 00 00  	jalr	ra
      98: 53 70 95 00  	fadd.s	ft0, fa0, fs1
;     data_out[i] = x * sigmoid;
      9c: 53 70 09 18  	fdiv.s	ft0, fs2, ft0
      a0: 27 a0 04 00  	fsw	ft0, 0(s1)
;   for (uint32_t i = chunk_start; i < chunk_stop; i++) {
      a4: 13 04 f4 ff  	addi	s0, s0, -1
      a8: 93 84 44 00  	addi	s1, s1, 4
      ac: 13 09 49 00  	addi	s2, s2, 4
      b0: e3 1c 04 fc  	bnez	s0, 0x88 <PULP_GELU_fp32_fp32_sigmoid+0x88>
; }
      b4: 83 20 c1 01  	lw	ra, 28(sp)
      b8: 03 24 81 01  	lw	s0, 24(sp)
      bc: 83 24 41 01  	lw	s1, 20(sp)
      c0: 03 29 01 01  	lw	s2, 16(sp)
      c4: 07 24 c1 00  	flw	fs0, 12(sp)
      c8: 87 24 81 00  	flw	fs1, 8(sp)
      cc: 07 29 41 00  	flw	fs2, 4(sp)
      d0: 13 01 01 02  	addi	sp, sp, 32
      d4: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(Gemm.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                Size     VMA      Type
  0                                     00000000 00000000 
  1 .strtab                             00000135 00000000 
  2 .text                               00000000 00000000 TEXT
  3 .text.PULP_Gemm_fp32_fp32_fp32_fp32 0000018c 00000000 TEXT
  4 .debug_loclists                     000003c7 00000000 DEBUG
  5 .debug_abbrev                       000000d6 00000000 DEBUG
  6 .debug_info                         000001ef 00000000 DEBUG
  7 .rela.debug_info                    00000120 00000000 
  8 .debug_rnglists                     0000003d 00000000 DEBUG
  9 .debug_str_offsets                  000000b0 00000000 DEBUG
 10 .rela.debug_str_offsets             000001f8 00000000 
 11 .debug_str                          00000231 00000000 DEBUG
 12 .debug_addr                         00000020 00000000 DEBUG
 13 .rela.debug_addr                    00000048 00000000 
 14 .comment                            00000073 00000000 
 15 .note.GNU-stack                     00000000 00000000 
 16 .riscv.attributes                   00000030 00000000 
 17 .debug_frame                        00000038 00000000 DEBUG
 18 .rela.debug_frame                   00000060 00000000 
 19 .debug_line                         00000240 00000000 DEBUG
 20 .rela.debug_line                    000004a4 00000000 
 21 .debug_line_str                     000001df 00000000 DEBUG
 22 .llvm_addrsig                       00000000 00000000 
 23 .symtab                             00000700 00000000 

Disassembly of section .text.PULP_Gemm_fp32_fp32_fp32_fp32:

00000000 <PULP_Gemm_fp32_fp32_fp32_fp32>:
;                                    uint32_t transB, int nb_dedicated_cores) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 81 00  	sw	s0, 28(sp)
       8: 23 2c 91 00  	sw	s1, 24(sp)
       c: 23 2a 21 01  	sw	s2, 20(sp)
      10: 23 28 31 01  	sw	s3, 16(sp)
      14: 23 26 41 01  	sw	s4, 12(sp)
      18: 23 24 51 01  	sw	s5, 8(sp)
      1c: 23 22 61 01  	sw	s6, 4(sp)
      20: 83 22 41 02  	lw	t0, 36(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      24: 73 23 40 f1  	csrr	t1, mhartid
;   return hart_id & 0x01f;
      28: 13 73 f3 01  	andi	t1, t1, 31
;   core_id = core_id % nb_dedicated_cores;
      2c: 33 63 53 02  	rem	t1, t1, t0
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      30: b3 93 02 10  	<unknown>
;   uint32_t M_chunk = (M >> log2Core) + ((M & (nb_dedicated_cores - 1)) != 0);
      34: b3 e3 03 10  	<unknown>
      38: b3 53 77 00  	srl	t2, a4, t2
      3c: 93 82 f2 ff  	addi	t0, t0, -1
      40: b3 f2 e2 00  	and	t0, t0, a4
      44: b3 32 50 00  	snez	t0, t0
      48: b3 83 53 00  	add	t2, t2, t0
;   uint32_t M_start = MIN(core_id * M_chunk, M);
      4c: b3 82 63 02  	mul	t0, t2, t1
      50: b3 d2 e2 04  	<unknown>
;   uint32_t M_end = MIN(M_start + M_chunk, M);
      54: 33 83 72 00  	add	t1, t0, t2
      58: 33 53 e3 04  	<unknown>
;   if (M_size == 0) {
      5c: 63 f6 62 10  	bgeu	t0, t1, 0x168 <PULP_Gemm_fp32_fp32_fp32_fp32+0x168>
;     for (uint32_t j = 0; j < O; ++j) {
      60: 63 04 08 10  	beqz	a6, 0x168 <PULP_Gemm_fp32_fp32_fp32_fp32+0x168>
;       for (uint32_t k = 0; k < N; ++k) {
      64: 63 8c 07 0a  	beqz	a5, 0x11c <PULP_Gemm_fp32_fp32_fp32_fp32+0x11c>
      68: 83 23 01 02  	lw	t2, 32(sp)
;   for (uint32_t i = M_start; i < M_end; ++i) {
      6c: 33 8e 57 02  	mul	t3, a5, t0
      70: 53 00 00 f0  	fmv.w.x	ft0, zero
      74: 93 0e 00 00  	li	t4, 0
      78: 13 0f 00 00  	li	t5, 0
      7c: b3 8f 02 03  	mul	t6, t0, a6
      80: fb 40 48 04  	<unknown>
      84: 13 84 02 00  	mv	s0, t0
      88: 93 04 0e 00  	mv	s1, t3
      8c: 13 89 0e 00  	mv	s2, t4
      90: 93 09 0f 00  	mv	s3, t5
      94: 13 8a 07 00  	mv	s4, a5
      98: d3 00 00 20  	fmv.s	ft1, ft0
      9c: 7b c0 47 02  	<unknown>
      a0: 93 8a 04 00  	mv	s5, s1
;         uint32_t a_idx = transA ? (k * M + i) : (i * N + k);
      a4: 63 84 08 00  	beqz	a7, 0xac <PULP_Gemm_fp32_fp32_fp32_fp32+0xac>
      a8: 93 0a 04 00  	mv	s5, s0
      ac: 13 8b 09 00  	mv	s6, s3
;         uint32_t b_idx = transB ? (j * N + k) : (k * O + j);
      b0: 63 84 03 00  	beqz	t2, 0xb8 <PULP_Gemm_fp32_fp32_fp32_fp32+0xb8>
      b4: 13 0b 09 00  	mv	s6, s2
;         sum += pSrcA[a_idx] * pSrcB[b_idx];
      b8: 93 9a 2a 00  	slli	s5, s5, 2
      bc: b3 0a 55 01  	add	s5, a0, s5
      c0: 07 a1 0a 00  	flw	ft2, 0(s5)
      c4: 93 1a 2b 00  	slli	s5, s6, 2
      c8: b3 8a 55 01  	add	s5, a1, s5
      cc: 87 a1 0a 00  	flw	ft3, 0(s5)
      d0: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;       for (uint32_t k = 0; k < N; ++k) {
      d4: 13 0a fa ff  	addi	s4, s4, -1
      d8: b3 89 09 01  	add	s3, s3, a6
      dc: 13 09 19 00  	addi	s2, s2, 1
      e0: 93 84 14 00  	addi	s1, s1, 1
      e4: 33 04 e4 00  	add	s0, s0, a4
;       pDstY[i * O + j] = sum + pDstC[i * O + j];
      e8: 33 04 ff 01  	add	s0, t5, t6
      ec: 13 14 24 00  	slli	s0, s0, 2
      f0: b3 04 86 00  	add	s1, a2, s0
      f4: 07 a1 04 00  	flw	ft2, 0(s1)
      f8: d3 70 11 00  	fadd.s	ft1, ft2, ft1
      fc: 33 84 86 00  	add	s0, a3, s0
     100: 27 20 14 00  	fsw	ft1, 0(s0)
;     for (uint32_t j = 0; j < O; ++j) {
     104: 13 0f 1f 00  	addi	t5, t5, 1
     108: b3 8e fe 00  	add	t4, t4, a5
;   for (uint32_t i = M_start; i < M_end; ++i) {
     10c: 93 82 12 00  	addi	t0, t0, 1
     110: 33 0e fe 00  	add	t3, t3, a5
     114: e3 90 62 f6  	bne	t0, t1, 0x74 <PULP_Gemm_fp32_fp32_fp32_fp32+0x74>
     118: 6f 00 00 05  	j	0x168 <PULP_Gemm_fp32_fp32_fp32_fp32+0x168>
     11c: 33 05 58 02  	mul	a0, a6, t0
     120: 13 17 25 00  	slli	a4, a0, 2
     124: 33 85 e6 00  	add	a0, a3, a4
     128: 93 15 28 00  	slli	a1, a6, 2
     12c: b3 06 53 40  	sub	a3, t1, t0
     130: 33 06 e6 00  	add	a2, a2, a4
     134: fb c0 86 01  	<unknown>
     138: 93 06 06 00  	mv	a3, a2
     13c: 13 07 05 00  	mv	a4, a0
     140: 93 07 08 00  	mv	a5, a6
     144: 7b 40 a8 00  	<unknown>
;       pDstY[i * O + j] = sum + pDstC[i * O + j];
     148: 07 a0 06 00  	flw	ft0, 0(a3)
     14c: 27 20 07 00  	fsw	ft0, 0(a4)
;     for (uint32_t j = 0; j < O; ++j) {
     150: 93 87 f7 ff  	addi	a5, a5, -1
     154: 13 07 47 00  	addi	a4, a4, 4
     158: 93 86 46 00  	addi	a3, a3, 4
;   for (uint32_t i = M_start; i < M_end; ++i) {
     15c: 93 82 12 00  	addi	t0, t0, 1
     160: 33 05 b5 00  	add	a0, a0, a1
     164: 33 06 b6 00  	add	a2, a2, a1
; }
     168: 03 24 c1 01  	lw	s0, 28(sp)
     16c: 83 24 81 01  	lw	s1, 24(sp)
     170: 03 29 41 01  	lw	s2, 20(sp)
     174: 83 29 01 01  	lw	s3, 16(sp)
     178: 03 2a c1 00  	lw	s4, 12(sp)
     17c: 83 2a 81 00  	lw	s5, 8(sp)
     180: 03 2b 41 00  	lw	s6, 4(sp)
     184: 13 01 01 02  	addi	sp, sp, 32
     188: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(Layernorm.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                Size     VMA      Type
  0                                     00000000 00000000 
  1 .strtab                             0000013a 00000000 
  2 .text                               00000000 00000000 TEXT
  3 .sdata                              00000004 00000000 DATA
  4 .text.PULP_Layernorm_fp32_fp32      0000013c 00000000 TEXT
  5 .rela.text.PULP_Layernorm_fp32_fp32 00000018 00000000 
  6 .debug_loclists                     00000339 00000000 DEBUG
  7 .debug_abbrev                       000000bf 00000000 DEBUG
  8 .debug_info                         00000224 00000000 DEBUG
  9 .rela.debug_info                    00000144 00000000 
 10 .debug_str_offsets                  000000c8 00000000 DEBUG
 11 .rela.debug_str_offsets             00000240 00000000 
 12 .debug_str                          000002a0 00000000 DEBUG
 13 .debug_addr                         00000024 00000000 DEBUG
 14 .rela.debug_addr                    00000054 00000000 
 15 .comment                            00000073 00000000 
 16 .note.GNU-stack                     00000000 00000000 
 17 .riscv.attributes                   00000030 00000000 
 18 .debug_frame                        00000024 00000000 DEBUG
 19 .rela.debug_frame                   00000030 00000000 
 20 .debug_line                         0000027b 00000000 DEBUG
 21 .rela.debug_line                    0000054c 00000000 
 22 .debug_line_str                     000001e9 00000000 DEBUG
 23 .llvm_addrsig                       00000000 00000000 
 24 .symtab                             000007e0 00000000 

Disassembly of section .text.PULP_Layernorm_fp32_fp32:

00000000 <PULP_Layernorm_fp32_fp32>:
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
       0: f3 28 40 f1  	csrr	a7, mhartid
;   return hart_id & 0x01f;
       4: 93 f8 f8 01  	andi	a7, a7, 31
;   core_id = core_id % nb_dedicated_cores;
       8: b3 e8 08 03  	rem	a7, a7, a6
;   int8_t log2Core = LOG2(nb_dedicated_cores);
       c: b3 12 08 10  	<unknown>
;   int32_t seq_length = size / lastDimLength;
      10: 33 57 f7 02  	divu	a4, a4, a5
;       (seq_length >> log2Core) + ((seq_length & (nb_dedicated_cores - 1)) != 0);
      14: b3 e2 02 10  	<unknown>
      18: b3 52 57 40  	sra	t0, a4, t0
      1c: 13 08 f8 ff  	addi	a6, a6, -1
      20: 33 78 07 01  	and	a6, a4, a6
      24: 33 38 00 01  	snez	a6, a6
      28: 33 88 02 01  	add	a6, t0, a6
;   int32_t start_seq = MIN(chunk * core_id, seq_length);
      2c: b3 08 18 03  	mul	a7, a6, a7
      30: b3 c8 e8 04  	<unknown>
;   int32_t end_seq = MIN(start_seq + chunk, seq_length);
      34: 33 88 08 01  	add	a6, a7, a6
      38: 33 47 e8 04  	<unknown>
;   int32_t elem_end = end_seq * lastDimLength;
      3c: 33 07 f7 02  	mul	a4, a4, a5
;   int32_t local_size = elem_end - elem_start;
      40: 33 97 f8 42  	<unknown>
;   int32_t local_seq_count = local_size / lastDimLength;
      44: 33 57 f7 02  	divu	a4, a4, a5
;   for (int32_t i = 0; i < local_seq_count; i++) {
      48: 13 28 17 00  	slti	a6, a4, 1
      4c: 93 b2 17 00  	seqz	t0, a5
;   for (int32_t i = 0; i < local_seq_count; i++) {
      50: 33 68 58 00  	or	a6, a6, t0
      54: 63 12 08 0e  	bnez	a6, 0x138 <PULP_Layernorm_fp32_fp32+0x138>
      58: b3 88 f8 02  	mul	a7, a7, a5
      5c: 93 98 28 00  	slli	a7, a7, 2
      60: b7 02 00 00  	lui	t0, 0
      64: 07 a0 02 00  	flw	ft0, 0(t0)
      68: 33 05 15 01  	add	a0, a0, a7
      6c: b3 85 15 01  	add	a1, a1, a7
      70: d3 f0 17 d0  	fcvt.s.wu	ft1, a5
      74: d3 70 10 18  	fdiv.s	ft1, ft0, ft1
      78: 53 01 00 f0  	fmv.w.x	ft2, zero
      7c: 93 02 00 00  	li	t0, 0
      80: b3 08 f8 02  	mul	a7, a6, a5
      84: 13 83 07 00  	mv	t1, a5
      88: d3 01 21 20  	fmv.s	ft3, ft2
;     for (int32_t j = 0; j < lastDimLength; j++) {
      8c: 7b c0 c7 00  	<unknown>
;       mean += local_data_in[j + i * lastDimLength];
      90: b3 83 12 01  	add	t2, t0, a7
      94: 93 93 23 00  	slli	t2, t2, 2
      98: b3 03 75 00  	add	t2, a0, t2
      9c: 07 a2 03 00  	flw	ft4, 0(t2)
      a0: d3 71 32 00  	fadd.s	ft3, ft4, ft3
;     for (int32_t j = 0; j < lastDimLength; j++) {
      a4: 93 82 12 00  	addi	t0, t0, 1
      a8: 93 02 00 00  	li	t0, 0
      ac: d3 f1 11 10  	fmul.s	ft3, ft3, ft1
      b0: 13 83 07 00  	mv	t1, a5
      b4: 53 02 21 20  	fmv.s	ft4, ft2
;     for (int32_t j = 0; j < lastDimLength; j++) {
      b8: 7b c0 e7 00  	<unknown>
;       temp = local_data_in[j + i * lastDimLength] - mean;
      bc: b3 83 12 01  	add	t2, t0, a7
      c0: 93 93 23 00  	slli	t2, t2, 2
      c4: b3 03 75 00  	add	t2, a0, t2
      c8: 87 a2 03 00  	flw	ft5, 0(t2)
      cc: d3 f2 32 08  	fsub.s	ft5, ft5, ft3
;       sum += temp * temp;
      d0: 43 f2 52 20  	fmadd.s	ft4, ft5, ft5, ft4
;     for (int32_t j = 0; j < lastDimLength; j++) {
      d4: 93 82 12 00  	addi	t0, t0, 1
      d8: 93 02 00 00  	li	t0, 0
;     sum += epsilon;
      dc: 43 72 12 50  	fmadd.s	ft4, ft4, ft1, fa0
;     std = sqrtf(sum);
      e0: 53 72 02 58  	fsqrt.s	ft4, ft4
      e4: 53 72 40 18  	fdiv.s	ft4, ft0, ft4
      e8: 13 03 06 00  	mv	t1, a2
      ec: 93 83 06 00  	mv	t2, a3
      f0: 13 8e 07 00  	mv	t3, a5
      f4: 7b c0 c7 01  	<unknown>
;           ((local_data_in[j + i * lastDimLength] - mean) / std) * scale[j] +
      f8: b3 8e 12 01  	add	t4, t0, a7
      fc: 93 9e 2e 00  	slli	t4, t4, 2
     100: 33 0f d5 01  	add	t5, a0, t4
     104: 87 22 0f 00  	flw	ft5, 0(t5)
     108: 07 23 03 00  	flw	ft6, 0(t1)
;           bias[j];
     10c: 87 a3 03 00  	flw	ft7, 0(t2)
;           ((local_data_in[j + i * lastDimLength] - mean) / std) * scale[j] +
     110: d3 f2 32 08  	fsub.s	ft5, ft5, ft3
     114: d3 f2 62 10  	fmul.s	ft5, ft5, ft6
     118: c3 f2 42 38  	fmadd.s	ft5, ft5, ft4, ft7
;       local_data_out[j + i * lastDimLength] =
     11c: b3 8e d5 01  	add	t4, a1, t4
     120: 27 a0 5e 00  	fsw	ft5, 0(t4)
;     for (int32_t j = 0; j < lastDimLength; j++) {
     124: 93 82 12 00  	addi	t0, t0, 1
     128: 93 83 43 00  	addi	t2, t2, 4
     12c: 13 03 43 00  	addi	t1, t1, 4
;   for (int32_t i = 0; i < local_seq_count; i++) {
     130: 13 08 18 00  	addi	a6, a6, 1
     134: e3 14 e8 f4  	bne	a6, a4, 0x7c <PULP_Layernorm_fp32_fp32+0x7c>
; }
     138: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(Matmul.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                       Size     VMA      Type
  0                                            00000000 00000000 
  1 .strtab                                    0000013e 00000000 
  2 .text                                      00000000 00000000 TEXT
  3 .text.PULP_MatMul_fp32_fp32_fp32_unroll1x7 000002ec 00000000 TEXT
  4 .debug_loclists                            000005a9 00000000 DEBUG
  5 .debug_abbrev                              000000d6 00000000 DEBUG
  6 .debug_info                                00000296 00000000 DEBUG
  7 .rela.debug_info                           00000120 00000000 
  8 .debug_rnglists                            00000093 00000000 DEBUG
  9 .debug_str_offsets                         000000ec 00000000 DEBUG
 10 .rela.debug_str_offsets                    000002ac 00000000 
 11 .debug_str                                 00000281 00000000 DEBUG
 12 .debug_addr                                00000020 00000000 DEBUG
 13 .rela.debug_addr                           00000048 00000000 
 14 .comment                                   00000073 00000000 
 15 .note.GNU-stack                            00000000 00000000 
 16 .riscv.attributes                          00000030 00000000 
 17 .debug_frame                               00000030 00000000 DEBUG
 18 .rela.debug_frame                          00000060 00000000 
 19 .debug_line                                000003d2 00000000 DEBUG
 20 .rela.debug_line                           000009b4 00000000 
 21 .debug_line_str                            000001e3 00000000 DEBUG
 22 .llvm_addrsig                              00000000 00000000 
 23 .symtab                                    00000b50 00000000 

Disassembly of section .text.PULP_MatMul_fp32_fp32_fp32_unroll1x7:

00000000 <PULP_MatMul_fp32_fp32_fp32_unroll1x7>:
;                                           uint32_t M, uint32_t N, uint32_t O, int nb_dedicated_cores) {
       0: 13 01 01 ff  	addi	sp, sp, -16
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
       4: 23 26 81 00  	sw	s0, 12(sp)
       8: 23 24 91 00  	sw	s1, 8(sp)
       c: 23 22 21 01  	sw	s2, 4(sp)
      10: 23 20 31 01  	sw	s3, 0(sp)
      14: f3 28 40 f1  	csrr	a7, mhartid
;   return hart_id & 0x01f;
      18: 93 f8 f8 01  	andi	a7, a7, 31
;   core_id = core_id % nb_dedicated_cores;
      1c: b3 e8 08 03  	rem	a7, a7, a6
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      20: b3 12 08 10  	<unknown>
;   uint32_t M_chunk = (M >> log2Core) + ((M & (nb_dedicated_cores - 1)) != 0);
      24: b3 e2 02 10  	<unknown>
      28: b3 d2 56 00  	srl	t0, a3, t0
      2c: 13 08 f8 ff  	addi	a6, a6, -1
      30: 33 78 d8 00  	and	a6, a6, a3
      34: 33 38 00 01  	snez	a6, a6
      38: b3 82 02 01  	add	t0, t0, a6
;   uint32_t M_start = MIN(core_id * M_chunk, M);
      3c: 33 88 12 03  	mul	a6, t0, a7
      40: 33 58 d8 04  	<unknown>
;   uint32_t M_end = MIN(M_start + M_chunk, M);
      44: b3 08 58 00  	add	a7, a6, t0
      48: b3 d6 d8 04  	<unknown>
;   if (M_size == 0) {
      4c: 63 88 06 25  	beq	a3, a6, 0x29c <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x29c>
      50: b3 86 06 41  	sub	a3, a3, a6
;   const float32_t *local_pSrcA = pSrcA + M_start * N;
      54: b3 08 e8 02  	mul	a7, a6, a4
      58: 93 98 28 00  	slli	a7, a7, 2
      5c: 33 05 15 01  	add	a0, a0, a7
;   float32_t *local_pDstY = pDstY + M_start * O;
      60: 33 0f f8 02  	mul	t5, a6, a5
      64: 13 18 2f 00  	slli	a6, t5, 2
      68: b3 08 06 01  	add	a7, a2, a6
      6c: 37 58 92 24  	lui	a6, 149797
      70: 13 08 58 92  	addi	a6, a6, -1755
;   uint32_t O_block = O - (O % 7);
      74: 33 b8 07 03  	mulhu	a6, a5, a6
      78: db b2 07 83  	<unknown>
      7c: 5b a8 02 85  	<unknown>
      80: 93 12 38 00  	slli	t0, a6, 3
      84: 33 88 02 41  	sub	a6, t0, a6
      88: b3 82 07 41  	sub	t0, a5, a6
;     for (uint32_t j = 0; j < O_block; j += 7) {
      8c: 63 16 08 08  	bnez	a6, 0x118 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x118>
;     for (uint32_t j = O_block; j < O; j++) {
      90: 63 76 f8 20  	bgeu	a6, a5, 0x29c <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x29c>
;       for (uint32_t k = 0; k < N; k++) {
      94: 63 00 07 22  	beqz	a4, 0x2b4 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x2b4>
      98: 13 06 00 00  	li	a2, 0
;   for (uint32_t i = 0; i < M_size; i++) {
      9c: 93 12 28 00  	slli	t0, a6, 2
      a0: b3 85 55 00  	add	a1, a1, t0
      a4: 93 92 27 00  	slli	t0, a5, 2
      a8: 13 13 27 00  	slli	t1, a4, 2
      ac: 53 00 00 f0  	fmv.w.x	ft0, zero
      b0: 33 8f 07 41  	sub	t5, a5, a6
      b4: b3 03 f6 02  	mul	t2, a2, a5
      b8: 13 8e 05 00  	mv	t3, a1
      bc: 93 0e 08 00  	mv	t4, a6
      c0: fb 40 2f 02  	<unknown>
      c4: 13 0f 05 00  	mv	t5, a0
      c8: 93 0f 0e 00  	mv	t6, t3
      cc: 13 04 07 00  	mv	s0, a4
      d0: d3 00 00 20  	fmv.s	ft1, ft0
      d4: 7b 40 c7 00  	<unknown>
;         float32_t a_val = local_pSrcA[i * N + k];
      d8: 07 21 0f 00  	flw	ft2, 0(t5)
;         float32_t b_val = pSrcB[k * O + j];
      dc: 87 a1 0f 00  	flw	ft3, 0(t6)
;         sum += a_val * b_val;
      e0: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;       for (uint32_t k = 0; k < N; k++) {
      e4: 13 04 f4 ff  	addi	s0, s0, -1
      e8: b3 8f 5f 00  	add	t6, t6, t0
      ec: 13 0f 4f 00  	addi	t5, t5, 4
;       local_pDstY[i * O + j] = sum;
      f0: 33 8f 7e 00  	add	t5, t4, t2
      f4: 13 1f 2f 00  	slli	t5, t5, 2
      f8: 33 8f e8 01  	add	t5, a7, t5
      fc: 27 20 1f 00  	fsw	ft1, 0(t5)
;     for (uint32_t j = O_block; j < O; j++) {
     100: 93 8e 1e 00  	addi	t4, t4, 1
     104: 13 0e 4e 00  	addi	t3, t3, 4
;   for (uint32_t i = 0; i < M_size; i++) {
     108: 13 06 16 00  	addi	a2, a2, 1
     10c: 33 05 65 00  	add	a0, a0, t1
     110: e3 10 d6 fa  	bne	a2, a3, 0xb0 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0xb0>
     114: 6f 00 80 18  	j	0x29c <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x29c>
     118: 13 03 00 00  	li	t1, 0
;   for (uint32_t i = 0; i < M_size; i++) {
     11c: 93 13 27 00  	slli	t2, a4, 2
     120: 13 8e c5 00  	addi	t3, a1, 12
     124: 93 9e 27 00  	slli	t4, a5, 2
     128: 93 1f 28 00  	slli	t6, a6, 2
     12c: b3 85 f5 01  	add	a1, a1, t6
     130: 33 0f 0f 01  	add	t5, t5, a6
     134: 13 1f 2f 00  	slli	t5, t5, 2
     138: 33 06 e6 01  	add	a2, a2, t5
     13c: 53 00 00 f0  	fmv.w.x	ft0, zero
     140: fb c0 c6 0a  	<unknown>
     144: 93 0f 00 00  	li	t6, 0
     148: 33 0f f3 02  	mul	t5, t1, a5
     14c: 13 04 0e 00  	mv	s0, t3
     150: 6f 00 80 03  	j	0x188 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x188>
;       local_pDstY[i * O + (j + 0)] = sum0;
     154: b3 84 ef 01  	add	s1, t6, t5
     158: 93 94 24 00  	slli	s1, s1, 2
     15c: b3 84 98 00  	add	s1, a7, s1
     160: 27 a0 74 00  	fsw	ft7, 0(s1)
;       local_pDstY[i * O + (j + 1)] = sum1;
     164: 27 a2 64 00  	fsw	ft6, 4(s1)
;       local_pDstY[i * O + (j + 2)] = sum2;
     168: 27 a4 54 00  	fsw	ft5, 8(s1)
;       local_pDstY[i * O + (j + 3)] = sum3;
     16c: 27 a6 44 00  	fsw	ft4, 12(s1)
;       local_pDstY[i * O + (j + 4)] = sum4;
     170: 27 a8 34 00  	fsw	ft3, 16(s1)
;       local_pDstY[i * O + (j + 5)] = sum5;
     174: 27 aa 24 00  	fsw	ft2, 20(s1)
;       local_pDstY[i * O + (j + 6)] = sum6;
     178: 27 ac 14 00  	fsw	ft1, 24(s1)
;     for (uint32_t j = 0; j < O_block; j += 7) {
     17c: 93 8f 7f 00  	addi	t6, t6, 7
     180: 13 04 c4 01  	addi	s0, s0, 28
     184: 63 fe 0f 09  	bgeu	t6, a6, 0x220 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x220>
     188: d3 00 00 20  	fmv.s	ft1, ft0
     18c: 53 01 00 20  	fmv.s	ft2, ft0
     190: d3 01 00 20  	fmv.s	ft3, ft0
     194: 53 02 00 20  	fmv.s	ft4, ft0
     198: d3 02 00 20  	fmv.s	ft5, ft0
     19c: 53 03 00 20  	fmv.s	ft6, ft0
     1a0: d3 03 00 20  	fmv.s	ft7, ft0
;       for (uint32_t k = 0; k < N; k++) {
     1a4: e3 08 07 fa  	beqz	a4, 0x154 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x154>
     1a8: 93 04 04 00  	mv	s1, s0
     1ac: 13 09 05 00  	mv	s2, a0
     1b0: 93 09 07 00  	mv	s3, a4
     1b4: d3 03 00 20  	fmv.s	ft7, ft0
     1b8: 53 03 00 20  	fmv.s	ft6, ft0
     1bc: d3 02 00 20  	fmv.s	ft5, ft0
     1c0: 53 02 00 20  	fmv.s	ft4, ft0
     1c4: d3 01 00 20  	fmv.s	ft3, ft0
     1c8: 53 01 00 20  	fmv.s	ft2, ft0
     1cc: d3 00 00 20  	fmv.s	ft1, ft0
;       for (uint32_t k = 0; k < N; k++) {
     1d0: 7b 40 47 02  	<unknown>
;         float32_t a0 = local_pSrcA[i * N + k];
     1d4: 07 25 09 00  	flw	fa0, 0(s2)
;         float32_t b0 = pSrcB[k * O + (j + 0)];
     1d8: 87 a5 44 ff  	flw	fa1, -12(s1)
;         float32_t b1 = pSrcB[k * O + (j + 1)];
     1dc: 07 a6 84 ff  	flw	fa2, -8(s1)
;         float32_t b2 = pSrcB[k * O + (j + 2)];
     1e0: 87 a6 c4 ff  	flw	fa3, -4(s1)
;         float32_t b3 = pSrcB[k * O + (j + 3)];
     1e4: 07 a7 04 00  	flw	fa4, 0(s1)
;         float32_t b4 = pSrcB[k * O + (j + 4)];
     1e8: 87 a7 44 00  	flw	fa5, 4(s1)
;         float32_t b5 = pSrcB[k * O + (j + 5)];
     1ec: 07 a8 84 00  	flw	fa6, 8(s1)
;         float32_t b6 = pSrcB[k * O + (j + 6)];
     1f0: 87 a8 c4 00  	flw	fa7, 12(s1)
;         sum0 += a0 * b0;
     1f4: c3 f3 a5 38  	fmadd.s	ft7, fa1, fa0, ft7
;         sum1 += a0 * b1;
     1f8: 43 73 a6 30  	fmadd.s	ft6, fa2, fa0, ft6
;         sum2 += a0 * b2;
     1fc: c3 f2 a6 28  	fmadd.s	ft5, fa3, fa0, ft5
;         sum3 += a0 * b3;
     200: 43 72 a7 20  	fmadd.s	ft4, fa4, fa0, ft4
;         sum4 += a0 * b4;
     204: c3 f1 a7 18  	fmadd.s	ft3, fa5, fa0, ft3
;         sum5 += a0 * b5;
     208: 43 71 a8 10  	fmadd.s	ft2, fa6, fa0, ft2
;         sum6 += a0 * b6;
     20c: c3 f0 a8 08  	fmadd.s	ft1, fa7, fa0, ft1
;       for (uint32_t k = 0; k < N; k++) {
     210: 93 89 f9 ff  	addi	s3, s3, -1
     214: 13 09 49 00  	addi	s2, s2, 4
     218: b3 84 d4 01  	add	s1, s1, t4
     21c: 6f f0 9f f3  	j	0x154 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x154>
;     for (uint32_t j = O_block; j < O; j++) {
     220: 63 78 f8 06  	bgeu	a6, a5, 0x290 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x290>
     224: 93 04 06 00  	mv	s1, a2
     228: 13 89 02 00  	mv	s2, t0
     22c: 93 8f 05 00  	mv	t6, a1
     230: 13 04 08 00  	mv	s0, a6
;       for (uint32_t k = 0; k < N; k++) {
     234: 63 08 07 04  	beqz	a4, 0x284 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x284>
     238: 93 04 05 00  	mv	s1, a0
     23c: 13 89 0f 00  	mv	s2, t6
     240: 93 09 07 00  	mv	s3, a4
     244: d3 00 00 20  	fmv.s	ft1, ft0
     248: 7b 40 c7 00  	<unknown>
;         float32_t a_val = local_pSrcA[i * N + k];
     24c: 07 a1 04 00  	flw	ft2, 0(s1)
;         float32_t b_val = pSrcB[k * O + j];
     250: 87 21 09 00  	flw	ft3, 0(s2)
;         sum += a_val * b_val;
     254: c3 f0 21 08  	fmadd.s	ft1, ft3, ft2, ft1
;       for (uint32_t k = 0; k < N; k++) {
     258: 93 89 f9 ff  	addi	s3, s3, -1
     25c: 33 09 d9 01  	add	s2, s2, t4
     260: 93 84 44 00  	addi	s1, s1, 4
;       local_pDstY[i * O + j] = sum;
     264: b3 04 e4 01  	add	s1, s0, t5
     268: 93 94 24 00  	slli	s1, s1, 2
     26c: b3 84 98 00  	add	s1, a7, s1
     270: 27 a0 14 00  	fsw	ft1, 0(s1)
;     for (uint32_t j = O_block; j < O; j++) {
     274: 13 04 14 00  	addi	s0, s0, 1
     278: 93 8f 4f 00  	addi	t6, t6, 4
     27c: e3 1e f4 fa  	bne	s0, a5, 0x238 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x238>
     280: 6f 00 00 01  	j	0x290 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x290>
     284: 13 09 f9 ff  	addi	s2, s2, -1
;       local_pDstY[i * O + j] = sum;
     288: 2b a2 04 00  	<unknown>
;     for (uint32_t j = O_block; j < O; j++) {
     28c: e3 1c 09 fe  	bnez	s2, 0x284 <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x284>
;   for (uint32_t i = 0; i < M_size; i++) {
     290: 13 03 13 00  	addi	t1, t1, 1
     294: 33 05 75 00  	add	a0, a0, t2
     298: 33 06 d6 01  	add	a2, a2, t4
; }
     29c: 03 24 c1 00  	lw	s0, 12(sp)
     2a0: 83 24 81 00  	lw	s1, 8(sp)
     2a4: 03 29 41 00  	lw	s2, 4(sp)
     2a8: 83 29 01 00  	lw	s3, 0(sp)
     2ac: 13 01 01 01  	addi	sp, sp, 16
     2b0: 67 80 00 00  	ret
     2b4: 13 05 00 00  	li	a0, 0
;   for (uint32_t i = 0; i < M_size; i++) {
     2b8: b3 05 0f 01  	add	a1, t5, a6
     2bc: 93 95 25 00  	slli	a1, a1, 2
     2c0: b3 05 b6 00  	add	a1, a2, a1
     2c4: 13 96 27 00  	slli	a2, a5, 2
     2c8: fb c0 e6 00  	<unknown>
     2cc: 13 87 05 00  	mv	a4, a1
     2d0: 93 87 02 00  	mv	a5, t0
     2d4: 7b c0 42 00  	<unknown>
     2d8: 13 00 00 00  	nop
;       local_pDstY[i * O + j] = sum;
     2dc: 2b 22 07 00  	<unknown>
;   for (uint32_t i = 0; i < M_size; i++) {
     2e0: 13 05 15 00  	addi	a0, a0, 1
     2e4: b3 85 c5 00  	add	a1, a1, a2
     2e8: 6f f0 5f fb  	j	0x29c <PULP_MatMul_fp32_fp32_fp32_unroll1x7+0x29c>

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(MaxPool.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                                    Size     VMA      Type
  0                                         00000000 00000000 
  1 .strtab                                 0000014c 00000000 
  2 .text                                   00000000 00000000 TEXT
  3 .sdata                                  00000004 00000000 DATA
  4 .text.PULP_MaxPool2d_fp32_fp32_HWC      0000033c 00000000 TEXT
  5 .rela.text.PULP_MaxPool2d_fp32_fp32_HWC 00000018 00000000 
  6 .debug_loclists                         00000703 00000000 DEBUG
  7 .debug_abbrev                           0000009c 00000000 DEBUG
  8 .debug_info                             0000020e 00000000 DEBUG
  9 .rela.debug_info                        000000a8 00000000 
 10 .debug_rnglists                         00000089 00000000 DEBUG
 11 .debug_str_offsets                      000000e8 00000000 DEBUG
 12 .rela.debug_str_offsets                 000002a0 00000000 
 13 .debug_str                              000002bb 00000000 DEBUG
 14 .debug_addr                             00000014 00000000 DEBUG
 15 .rela.debug_addr                        00000024 00000000 
 16 .comment                                00000073 00000000 
 17 .note.GNU-stack                         00000000 00000000 
 18 .riscv.attributes                       00000030 00000000 
 19 .debug_frame                            00000044 00000000 DEBUG
 20 .rela.debug_frame                       00000060 00000000 
 21 .debug_line                             00000286 00000000 DEBUG
 22 .rela.debug_line                        000005c4 00000000 
 23 .debug_line_str                         00000162 00000000 DEBUG
 24 .llvm_addrsig                           00000000 00000000 
 25 .symtab                                 00000870 00000000 

Disassembly of section .text.PULP_MaxPool2d_fp32_fp32_HWC:

00000000 <PULP_MaxPool2d_fp32_fp32_HWC>:
;                                   uint32_t pad_left, uint32_t pad_right, int nb_dedicated_cores) {
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
      38: 03 23 41 06  	lw	t1, 100(sp)
      3c: 83 22 81 06  	lw	t0, 104(sp)
;   uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
      40: b3 03 f6 40  	sub	t2, a2, a5
      44: 23 2e 61 00  	sw	t1, 28(sp)
      48: 33 83 63 00  	add	t1, t2, t1
      4c: b3 02 53 00  	add	t0, t1, t0
      50: 23 2c 11 01  	sw	a7, 24(sp)
      54: 33 de 12 03  	divu	t3, t0, a7
      58: 93 08 f0 ff  	li	a7, -1
      5c: 23 24 01 03  	sw	a6, 40(sp)
;   for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      60: 63 00 1e 2b  	beq	t3, a7, 0x300 <PULP_MaxPool2d_fp32_fp32_HWC+0x300>
      64: 13 0b 07 00  	mv	s6, a4
      68: 03 23 c1 06  	lw	t1, 108(sp)
      6c: 03 27 41 07  	lw	a4, 116(sp)
      70: 83 28 01 07  	lw	a7, 112(sp)
      74: f3 22 40 f1  	csrr	t0, mhartid
      78: 93 f2 f2 01  	andi	t0, t0, 31
      7c: b3 e2 e2 02  	rem	t0, t0, a4
      80: b3 13 07 10  	<unknown>
      84: b3 e3 03 10  	<unknown>
      88: b3 d3 76 00  	srl	t2, a3, t2
      8c: 13 07 f7 ff  	addi	a4, a4, -1
      90: 33 77 d7 00  	and	a4, a4, a3
      94: 33 37 e0 00  	snez	a4, a4
      98: 33 87 e3 00  	add	a4, t2, a4
      9c: 33 57 07 10  	<unknown>
      a0: b3 02 57 02  	mul	t0, a4, t0
      a4: b3 d2 d2 04  	<unknown>
      a8: 33 d4 02 10  	<unknown>
      ac: 33 07 e4 00  	add	a4, s0, a4
      b0: 33 57 d7 04  	<unknown>
      b4: b3 5f 07 10  	<unknown>
      b8: 33 87 65 41  	sub	a4, a1, s6
      bc: 33 07 67 00  	add	a4, a4, t1
      c0: 33 07 17 01  	add	a4, a4, a7
      c4: 03 28 81 02  	lw	a6, 40(sp)
      c8: b3 53 07 03  	divu	t2, a4, a6
      cc: 13 8f 13 00  	addi	t5, t2, 1
      d0: 33 37 e0 01  	snez	a4, t5
      d4: b3 38 f4 01  	sltu	a7, s0, t6
;     for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      d8: 33 77 17 01  	and	a4, a4, a7
      dc: 63 02 07 22  	beqz	a4, 0x300 <PULP_MaxPool2d_fp32_fp32_HWC+0x300>
      e0: 83 2e 01 06  	lw	t4, 96(sp)
      e4: 13 07 00 00  	li	a4, 0
;         for (uint32_t p = 0; p < P; ++p) {
      e8: 63 8a 07 16  	beqz	a5, 0x25c <PULP_MaxPool2d_fp32_fp32_HWC+0x25c>
;           for (uint32_t q = 0; q < Q; ++q) {
      ec: 63 02 0b 1c  	beqz	s6, 0x2b0 <PULP_MaxPool2d_fp32_fp32_HWC+0x2b0>
;   for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      f0: 93 08 03 00  	mv	a7, t1
      f4: 03 28 c1 01  	lw	a6, 28(sp)
      f8: b3 08 b8 42  	<unknown>
      fc: b3 08 10 41  	neg	a7, a7
     100: 93 02 04 00  	mv	t0, s0
     104: b3 82 16 43  	<unknown>
     108: 93 98 22 00  	slli	a7, t0, 2
     10c: 33 0a 15 01  	add	s4, a0, a7
     110: 03 25 81 01  	lw	a0, 24(sp)
     114: 33 05 d5 02  	mul	a0, a0, a3
     118: 33 05 b5 02  	mul	a0, a0, a1
     11c: 13 15 25 00  	slli	a0, a0, 2
     120: 23 26 a1 00  	sw	a0, 12(sp)
     124: 03 25 81 02  	lw	a0, 40(sp)
     128: 33 05 d5 02  	mul	a0, a0, a3
     12c: 13 18 25 00  	slli	a6, a0, 2
     130: 33 85 b6 02  	mul	a0, a3, a1
     134: 13 19 25 00  	slli	s2, a0, 2
     138: 37 05 00 00  	lui	a0, 0
     13c: 07 20 05 00  	flw	ft0, 0(a0)
     140: 93 99 26 00  	slli	s3, a3, 2
     144: 33 05 60 40  	neg	a0, t1
     148: 23 24 a1 00  	sw	a0, 8(sp)
     14c: 93 0a f0 ff  	li	s5, -1
     150: 23 2a c1 01  	sw	t3, 20(sp)
     154: 23 28 e1 01  	sw	t5, 16(sp)
     158: 13 05 00 00  	li	a0, 0
     15c: 93 02 07 00  	mv	t0, a4
     160: 03 27 81 01  	lw	a4, 24(sp)
     164: 33 87 e2 02  	mul	a4, t0, a4
     168: 83 28 c1 01  	lw	a7, 28(sp)
     16c: b3 0b 17 41  	sub	s7, a4, a7
     170: 23 20 51 02  	sw	t0, 32(sp)
     174: 33 8c e2 03  	mul	s8, t0, t5
     178: 83 22 81 00  	lw	t0, 8(sp)
     17c: 23 22 41 03  	sw	s4, 36(sp)
     180: 93 0d 05 00  	mv	s11, a0
     184: 33 05 85 01  	add	a0, a0, s8
     188: 33 87 8f 40  	sub	a4, t6, s0
     18c: b3 00 d5 02  	mul	ra, a0, a3
     190: 13 0f 0a 00  	mv	t5, s4
     194: 13 03 04 00  	mv	t1, s0
     198: fb 40 47 04  	<unknown>
     19c: 93 08 00 00  	li	a7, 0
     1a0: 13 05 0f 00  	mv	a0, t5
     1a4: d3 00 00 20  	fmv.s	ft1, ft0
     1a8: 7b c0 e7 02  	<unknown>
;           int32_t h_in = h_in_start + p;
     1ac: 33 87 78 01  	add	a4, a7, s7
;           if (h_in < 0 || h_in >= (int32_t)H) {
     1b0: b3 ac ea 00  	slt	s9, s5, a4
     1b4: 33 27 c7 00  	slt	a4, a4, a2
     1b8: 33 fe ec 00  	and	t3, s9, a4
     1bc: 93 8c 02 00  	mv	s9, t0
     1c0: 13 0d 05 00  	mv	s10, a0
     1c4: 13 07 0b 00  	mv	a4, s6
     1c8: 63 12 0e 02  	bnez	t3, 0x1ec <PULP_MaxPool2d_fp32_fp32_HWC+0x1ec>
;         for (uint32_t p = 0; p < P; ++p) {
     1cc: 93 88 18 00  	addi	a7, a7, 1
     1d0: 33 05 25 01  	add	a0, a0, s2
     1d4: e3 9c f8 fc  	bne	a7, a5, 0x1ac <PULP_MaxPool2d_fp32_fp32_HWC+0x1ac>
     1d8: 6f 00 40 03  	j	0x20c <PULP_MaxPool2d_fp32_fp32_HWC+0x20c>
;           for (uint32_t q = 0; q < Q; ++q) {
     1dc: 13 07 f7 ff  	addi	a4, a4, -1
     1e0: 33 0d 3d 01  	add	s10, s10, s3
     1e4: 93 8c 1c 00  	addi	s9, s9, 1
     1e8: e3 02 07 fe  	beqz	a4, 0x1cc <PULP_MaxPool2d_fp32_fp32_HWC+0x1cc>
;             if (w_in < 0 || w_in >= (int32_t)W) {
     1ec: 13 ae 0c 00  	slti	t3, s9, 0
     1f0: b3 a4 bc 00  	slt	s1, s9, a1
     1f4: 93 c4 14 00  	xori	s1, s1, 1
     1f8: 33 6e 9e 00  	or	t3, t3, s1
     1fc: e3 10 0e fe  	bnez	t3, 0x1dc <PULP_MaxPool2d_fp32_fp32_HWC+0x1dc>
;             float32_t val = pSrcA[input_idx];
     200: 07 21 0d 00  	flw	ft2, 0(s10)
;             if (val > max_val) {
     204: d3 10 11 28  	fmax.s	ft1, ft2, ft1
     208: 6f f0 5f fd  	j	0x1dc <PULP_MaxPool2d_fp32_fp32_HWC+0x1dc>
;         uint32_t output_idx = (h_out * W_out + w_out) * C + c;
     20c: 33 05 13 00  	add	a0, t1, ra
;         pDstC[output_idx] = max_val;
     210: 13 15 25 00  	slli	a0, a0, 2
     214: 33 85 ae 00  	add	a0, t4, a0
     218: 27 20 15 00  	fsw	ft1, 0(a0)
;       for (uint32_t c = ch_start; c < ch_stop; ++c) {
     21c: 13 03 13 00  	addi	t1, t1, 1
     220: 13 0f 4f 00  	addi	t5, t5, 4
;     for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
     224: 13 85 1d 00  	addi	a0, s11, 1
     228: 33 0a 0a 01  	add	s4, s4, a6
     22c: 03 27 81 02  	lw	a4, 40(sp)
     230: b3 82 e2 00  	add	t0, t0, a4
     234: e3 96 7d f4  	bne	s11, t2, 0x180 <PULP_MaxPool2d_fp32_fp32_HWC+0x180>
     238: 83 28 01 02  	lw	a7, 32(sp)
;   for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
     23c: 13 87 18 00  	addi	a4, a7, 1
     240: 03 2a 41 02  	lw	s4, 36(sp)
;   for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
     244: 03 25 c1 00  	lw	a0, 12(sp)
     248: 33 0a aa 00  	add	s4, s4, a0
     24c: 03 2e 41 01  	lw	t3, 20(sp)
     250: 03 2f 01 01  	lw	t5, 16(sp)
     254: e3 92 c8 f1  	bne	a7, t3, 0x158 <PULP_MaxPool2d_fp32_fp32_HWC+0x158>
     258: 6f 00 80 0a  	j	0x300 <PULP_MaxPool2d_fp32_fp32_HWC+0x300>
     25c: 33 85 8f 40  	sub	a0, t6, s0
     260: b7 05 80 ff  	lui	a1, 1046528
     264: 13 08 00 00  	li	a6, 0
     268: 13 06 07 00  	mv	a2, a4
     26c: 33 07 e7 03  	mul	a4, a4, t5
     270: 93 07 08 00  	mv	a5, a6
     274: 33 08 e8 00  	add	a6, a6, a4
     278: 33 08 d8 02  	mul	a6, a6, a3
     27c: 93 08 05 00  	mv	a7, a0
     280: 93 02 04 00  	mv	t0, s0
     284: 7b 40 a5 00  	<unknown>
;         uint32_t output_idx = (h_out * W_out + w_out) * C + c;
     288: 33 83 02 01  	add	t1, t0, a6
;         pDstC[output_idx] = max_val;
     28c: 13 13 23 00  	slli	t1, t1, 2
     290: 33 83 6e 00  	add	t1, t4, t1
     294: 23 20 b3 00  	sw	a1, 0(t1)
;       for (uint32_t c = ch_start; c < ch_stop; ++c) {
     298: 93 82 12 00  	addi	t0, t0, 1
;     for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
     29c: 13 88 17 00  	addi	a6, a5, 1
     2a0: e3 98 77 fc  	bne	a5, t2, 0x270 <PULP_MaxPool2d_fp32_fp32_HWC+0x270>
;   for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
     2a4: 13 07 16 00  	addi	a4, a2, 1
     2a8: e3 1e c6 fb  	bne	a2, t3, 0x264 <PULP_MaxPool2d_fp32_fp32_HWC+0x264>
     2ac: 6f 00 40 05  	j	0x300 <PULP_MaxPool2d_fp32_fp32_HWC+0x300>
     2b0: 33 85 8f 40  	sub	a0, t6, s0
     2b4: b7 05 80 ff  	lui	a1, 1046528
     2b8: 13 08 00 00  	li	a6, 0
     2bc: 13 06 07 00  	mv	a2, a4
     2c0: 33 07 e7 03  	mul	a4, a4, t5
     2c4: 93 07 08 00  	mv	a5, a6
     2c8: 33 08 e8 00  	add	a6, a6, a4
     2cc: 33 08 d8 02  	mul	a6, a6, a3
     2d0: 93 08 05 00  	mv	a7, a0
     2d4: 93 02 04 00  	mv	t0, s0
     2d8: 7b 40 a5 00  	<unknown>
;         uint32_t output_idx = (h_out * W_out + w_out) * C + c;
     2dc: 33 83 02 01  	add	t1, t0, a6
;         pDstC[output_idx] = max_val;
     2e0: 13 13 23 00  	slli	t1, t1, 2
     2e4: 33 83 6e 00  	add	t1, t4, t1
     2e8: 23 20 b3 00  	sw	a1, 0(t1)
;       for (uint32_t c = ch_start; c < ch_stop; ++c) {
     2ec: 93 82 12 00  	addi	t0, t0, 1
;     for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
     2f0: 13 88 17 00  	addi	a6, a5, 1
     2f4: e3 98 77 fc  	bne	a5, t2, 0x2c4 <PULP_MaxPool2d_fp32_fp32_HWC+0x2c4>
;   for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
     2f8: 13 07 16 00  	addi	a4, a2, 1
     2fc: e3 1e c6 fb  	bne	a2, t3, 0x2b8 <PULP_MaxPool2d_fp32_fp32_HWC+0x2b8>
; }
     300: 83 20 c1 05  	lw	ra, 92(sp)
     304: 03 24 81 05  	lw	s0, 88(sp)
     308: 83 24 41 05  	lw	s1, 84(sp)
     30c: 03 29 01 05  	lw	s2, 80(sp)
     310: 83 29 c1 04  	lw	s3, 76(sp)
     314: 03 2a 81 04  	lw	s4, 72(sp)
     318: 83 2a 41 04  	lw	s5, 68(sp)
     31c: 03 2b 01 04  	lw	s6, 64(sp)
     320: 83 2b c1 03  	lw	s7, 60(sp)
     324: 03 2c 81 03  	lw	s8, 56(sp)
     328: 83 2c 41 03  	lw	s9, 52(sp)
     32c: 03 2d 01 03  	lw	s10, 48(sp)
     330: 83 2d c1 02  	lw	s11, 44(sp)
     334: 13 01 01 06  	addi	sp, sp, 96
     338: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(RQiHardswish.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                         Size     VMA      Type
  0                              00000000 00000000 
  1 .strtab                      00000136 00000000 
  2 .text                        00000000 00000000 TEXT
  3 .text.RQiHardswish_s8_s8_plp 0000015c 00000000 TEXT
  4 .debug_loclists              0000018c 00000000 DEBUG
  5 .debug_abbrev                000000c8 00000000 DEBUG
  6 .debug_info                  000001bf 00000000 DEBUG
  7 .rela.debug_info             000000f0 00000000 
  8 .debug_rnglists              0000001d 00000000 DEBUG
  9 .debug_str_offsets           000000b0 00000000 DEBUG
 10 .rela.debug_str_offsets      000001f8 00000000 
 11 .debug_str                   0000025f 00000000 DEBUG
 12 .debug_addr                  00000018 00000000 DEBUG
 13 .rela.debug_addr             00000030 00000000 
 14 .comment                     00000073 00000000 
 15 .note.GNU-stack              00000000 00000000 
 16 .riscv.attributes            00000030 00000000 
 17 .debug_frame                 00000024 00000000 DEBUG
 18 .rela.debug_frame            00000030 00000000 
 19 .debug_line                  000002a8 00000000 DEBUG
 20 .rela.debug_line             00000630 00000000 
 21 .debug_line_str              000001cb 00000000 DEBUG
 22 .llvm_addrsig                00000000 00000000 
 23 .symtab                      000007b0 00000000 

Disassembly of section .text.RQiHardswish_s8_s8_plp:

00000000 <RQiHardswish_s8_s8_plp>:
;                             int32_t mul, int32_t add, int32_t shift, int nb_dedicated_cores) {
       0: 83 22 41 00  	lw	t0, 4(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
       4: 73 23 40 f1  	csrr	t1, mhartid
;   return hart_id & 0x01f;
       8: 13 73 f3 01  	andi	t1, t1, 31
;   core_id = core_id % nb_dedicated_cores;
       c: 33 63 53 02  	rem	t1, t1, t0
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      10: b3 93 02 10  	<unknown>
;   int16_t chunk = (size >> log2Core) + ((size & (nb_dedicated_cores - 1)) != 0);
      14: b3 e3 03 10  	<unknown>
      18: b3 53 76 40  	sra	t2, a2, t2
      1c: 93 82 f2 ff  	addi	t0, t0, -1
      20: b3 f2 c2 00  	and	t0, t0, a2
      24: b3 32 50 00  	snez	t0, t0
      28: b3 82 53 00  	add	t0, t2, t0
;   int16_t chunk_start = MIN(chunk * core_id, size);
      2c: 33 ce 02 10  	<unknown>
      30: b3 02 6e 02  	mul	t0, t3, t1
      34: b3 c3 c2 04  	<unknown>
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      38: b3 c2 03 10  	<unknown>
      3c: 33 83 c2 01  	add	t1, t0, t3
      40: 13 06 16 00  	addi	a2, a2, 1
      44: 33 4e c3 04  	<unknown>
      48: 33 43 0e 10  	<unknown>
;   for (int i = chunk_start; i < chunk_stop; i++) {
      4c: 63 da 62 06  	bge	t0, t1, 0xc0 <RQiHardswish_s8_s8_plp+0xc0>
      50: 03 26 01 00  	lw	a2, 0(sp)
;   rnd = (1 << (shift - 1));
      54: 93 0e f6 ff  	addi	t4, a2, -1
      58: 13 0f 10 00  	li	t5, 1
;   rnd = (1 << (shift - 1));
      5c: b3 1e df 01  	sll	t4, t5, t4
      60: b3 06 d8 02  	mul	a3, a6, a3
      64: 33 88 1e 01  	add	a6, t4, a7
;   for (int i = chunk_start; i < chunk_stop; i++) {
      68: b3 08 7e 40  	sub	a7, t3, t2
      6c: 93 f3 18 00  	andi	t2, a7, 1
      70: 93 88 12 00  	addi	a7, t0, 1
      74: 63 84 03 04  	beqz	t2, 0xbc <RQiHardswish_s8_s8_plp+0xbc>
;     temp = input[i] + three;
      78: b3 03 55 00  	add	t2, a0, t0
      7c: 83 83 03 00  	lb	t2, 0(t2)
      80: b3 8e e3 00  	add	t4, t2, a4
      84: 13 8e 07 00  	mv	t3, a5
;     temp = CLAMP(temp, 0, six);
      88: 63 c4 d7 01  	blt	a5, t4, 0x90 <RQiHardswish_s8_s8_plp+0x90>
      8c: 33 ee 0e 04  	<unknown>
;     temp = input[i] * temp;
      90: b3 83 76 02  	mul	t2, a3, t2
;     temp = temp * (mul) + (add + rnd);
      94: b3 83 c3 03  	mul	t2, t2, t3
;     temp = temp >> shift;
      98: 13 0e 08 00  	mv	t3, a6
      9c: 5b ae c3 40  	<unknown>
      a0: 93 03 00 f8  	li	t2, -128
;     output[i] = (int8_t)CLAMP(temp, -128, 127);
      a4: b3 63 7e 04  	<unknown>
      a8: 13 0e f0 07  	li	t3, 127
      ac: b3 c3 c3 05  	<unknown>
      b0: b3 82 55 00  	add	t0, a1, t0
      b4: 23 80 72 00  	sb	t2, 0(t0)
      b8: 93 82 08 00  	mv	t0, a7
;   for (int i = chunk_start; i < chunk_stop; i++) {
      bc: 63 14 13 01  	bne	t1, a7, 0xc4 <RQiHardswish_s8_s8_plp+0xc4>
; }
      c0: 67 80 00 00  	ret
;   for (int i = chunk_start; i < chunk_stop; i++) {
      c4: b3 08 53 40  	sub	a7, t1, t0
      c8: 93 82 12 00  	addi	t0, t0, 1
      cc: b3 85 55 00  	add	a1, a1, t0
      d0: 33 05 55 00  	add	a0, a0, t0
      d4: 93 02 00 f8  	li	t0, -128
;   for (int i = chunk_start; i < chunk_stop; i++) {
      d8: 93 d3 18 00  	srli	t2, a7, 1
      dc: 13 03 f0 07  	li	t1, 127
      e0: 7b c0 a3 03  	<unknown>
      e4: 6f 00 00 03  	j	0x114 <RQiHardswish_s8_s8_plp+0x114>
;     temp = input[i] * temp;
      e8: b3 83 76 02  	mul	t2, a3, t2
;     temp = temp * (mul) + (add + rnd);
      ec: b3 83 c3 03  	mul	t2, t2, t3
;     temp = temp >> shift;
      f0: 13 0e 08 00  	mv	t3, a6
      f4: 5b ae c3 40  	<unknown>
;     output[i] = (int8_t)CLAMP(temp, -128, 127);
      f8: b3 63 5e 04  	<unknown>
      fc: b3 c3 63 04  	<unknown>
     100: 23 80 75 00  	sb	t2, 0(a1)
;   for (int i = chunk_start; i < chunk_stop; i++) {
     104: 93 88 e8 ff  	addi	a7, a7, -2
     108: 93 85 25 00  	addi	a1, a1, 2
     10c: 13 05 25 00  	addi	a0, a0, 2
     110: e3 88 08 fa  	beqz	a7, 0xc0 <RQiHardswish_s8_s8_plp+0xc0>
;     temp = input[i] + three;
     114: 83 03 f5 ff  	lb	t2, -1(a0)
     118: b3 8e e3 00  	add	t4, t2, a4
     11c: 13 8e 07 00  	mv	t3, a5
;     temp = CLAMP(temp, 0, six);
     120: 63 c4 d7 01  	blt	a5, t4, 0x128 <RQiHardswish_s8_s8_plp+0x128>
     124: 33 ee 0e 04  	<unknown>
;     temp = input[i] * temp;
     128: b3 83 76 02  	mul	t2, a3, t2
;     temp = temp * (mul) + (add + rnd);
     12c: b3 83 c3 03  	mul	t2, t2, t3
;     temp = temp >> shift;
     130: 13 0e 08 00  	mv	t3, a6
     134: 5b ae c3 40  	<unknown>
;     output[i] = (int8_t)CLAMP(temp, -128, 127);
     138: b3 63 5e 04  	<unknown>
     13c: b3 c3 63 04  	<unknown>
     140: a3 8f 75 fe  	sb	t2, -1(a1)
;     temp = input[i] + three;
     144: 83 03 05 00  	lb	t2, 0(a0)
     148: b3 8e e3 00  	add	t4, t2, a4
     14c: 13 8e 07 00  	mv	t3, a5
;     temp = CLAMP(temp, 0, six);
     150: e3 cc d7 f9  	blt	a5, t4, 0xe8 <RQiHardswish_s8_s8_plp+0xe8>
     154: 33 ee 0e 04  	<unknown>
     158: 6f f0 1f f9  	j	0xe8 <RQiHardswish_s8_s8_plp+0xe8>

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(Relu.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                      Size     VMA      Type
  0                           00000000 00000000 
  1 .strtab                   0000012b 00000000 
  2 .text                     00000000 00000000 TEXT
  3 .text.PULP_Relu_fp32_fp32 00000078 00000000 TEXT
  4 .debug_loclists           000000f8 00000000 DEBUG
  5 .debug_abbrev             000000c8 00000000 DEBUG
  6 .debug_info               000001a1 00000000 DEBUG
  7 .rela.debug_info          00000108 00000000 
  8 .debug_rnglists           00000017 00000000 DEBUG
  9 .debug_str_offsets        00000094 00000000 DEBUG
 10 .rela.debug_str_offsets   000001a4 00000000 
 11 .debug_str                00000222 00000000 DEBUG
 12 .debug_addr               00000018 00000000 DEBUG
 13 .rela.debug_addr          00000030 00000000 
 14 .comment                  00000073 00000000 
 15 .note.GNU-stack           00000000 00000000 
 16 .riscv.attributes         00000030 00000000 
 17 .debug_frame              00000024 00000000 DEBUG
 18 .rela.debug_frame         00000030 00000000 
 19 .debug_line               000001a0 00000000 DEBUG
 20 .rela.debug_line          000002dc 00000000 
 21 .debug_line_str           000001df 00000000 DEBUG
 22 .llvm_addrsig             00000000 00000000 
 23 .symtab                   00000510 00000000 

Disassembly of section .text.PULP_Relu_fp32_fp32:

00000000 <PULP_Relu_fp32_fp32>:
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
       0: 73 27 40 f1  	csrr	a4, mhartid
;   return hart_id & 0x01f;
       4: 13 77 f7 01  	andi	a4, a4, 31
;   core_id = core_id % nb_dedicated_cores;
       8: 33 67 d7 02  	rem	a4, a4, a3
;   int8_t log2Core = LOG2(nb_dedicated_cores);
       c: b3 97 06 10  	<unknown>
;   int32_t chunk = (size >> log2Core) + ((size & (nb_dedicated_cores - 1)) != 0);
      10: b3 e7 07 10  	<unknown>
      14: b3 57 f6 00  	srl	a5, a2, a5
      18: 93 86 f6 ff  	addi	a3, a3, -1
      1c: b3 f6 c6 00  	and	a3, a3, a2
      20: b3 36 d0 00  	snez	a3, a3
      24: b3 87 d7 00  	add	a5, a5, a3
;   int32_t start = MIN(chunk * core_id, size);
      28: b3 86 e7 02  	mul	a3, a5, a4
      2c: b3 d6 c6 04  	<unknown>
;   int32_t end = MIN(start + chunk, size);
      30: 33 87 f6 00  	add	a4, a3, a5
      34: 33 56 c7 04  	<unknown>
;   int32_t local_size = end - start;
      38: 33 07 d6 40  	sub	a4, a2, a3
;   for (int32_t i = 0; i < local_size; i++) {
      3c: 63 5c e0 02  	blez	a4, 0x74 <PULP_Relu_fp32_fp32+0x74>
      40: 13 97 26 00  	slli	a4, a3, 2
      44: 33 05 e5 00  	add	a0, a0, a4
      48: b3 85 e5 00  	add	a1, a1, a4
;   for (int32_t i = 0; i < local_size; i++) {
      4c: 33 86 c6 40  	sub	a2, a3, a2
      50: 53 00 00 f0  	fmv.w.x	ft0, zero
;     local_output[i] = MAX(local_input[i], 0.0f);
      54: 87 20 05 00  	flw	ft1, 0(a0)
      58: 93 06 06 00  	mv	a3, a2
;     local_output[i] = MAX(local_input[i], 0.0f);
      5c: d3 90 00 28  	fmax.s	ft1, ft1, ft0
      60: 27 a0 15 00  	fsw	ft1, 0(a1)
;   for (int32_t i = 0; i < local_size; i++) {
      64: 13 06 16 00  	addi	a2, a2, 1
      68: 93 85 45 00  	addi	a1, a1, 4
      6c: 13 05 45 00  	addi	a0, a0, 4
      70: e3 72 d6 fe  	bgeu	a2, a3, 0x54 <PULP_Relu_fp32_fp32+0x54>
; }
      74: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(RequantShift.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                           Size     VMA      Type
  0                                00000000 00000000 
  1 .strtab                        00000341 00000000 
  2 .text                          00000000 00000000 TEXT
  3 .text.RequantShift_u8_s8_NHWC  0000009c 00000000 TEXT
  4 .text.RequantShift_u16_s8_NHWC 00000098 00000000 TEXT
  5 .text.RequantShift_u32_s8_NHWC 00000098 00000000 TEXT
  6 .text.RequantShift_u8_s8_NCHW  0000009c 00000000 TEXT
  7 .text.RequantShift_u16_s8_NCHW 00000098 00000000 TEXT
  8 .text.RequantShift_u32_s8_NCHW 00000098 00000000 TEXT
  9 .text.RequantShift_u8_u8_NHWC  0000009c 00000000 TEXT
 10 .text.RequantShift_u16_u8_NHWC 00000098 00000000 TEXT
 11 .text.RequantShift_u32_u8_NHWC 00000098 00000000 TEXT
 12 .text.RequantShift_u8_u8_NCHW  0000009c 00000000 TEXT
 13 .text.RequantShift_u16_u8_NCHW 00000098 00000000 TEXT
 14 .text.RequantShift_u32_u8_NCHW 00000098 00000000 TEXT
 15 .text.RequantShift_s8_u8_NHWC  0000009c 00000000 TEXT
 16 .text.RequantShift_s16_u8_NHWC 00000098 00000000 TEXT
 17 .text.RequantShift_s32_u8_NHWC 00000098 00000000 TEXT
 18 .text.RequantShift_s8_u8_NCHW  0000009c 00000000 TEXT
 19 .text.RequantShift_s16_u8_NCHW 00000098 00000000 TEXT
 20 .text.RequantShift_s32_u8_NCHW 00000098 00000000 TEXT
 21 .debug_loclists                00001572 00000000 DEBUG
 22 .debug_abbrev                  000000c9 00000000 DEBUG
 23 .debug_info                    00000c3c 00000000 DEBUG
 24 .rela.debug_info               000003a8 00000000 
 25 .debug_rnglists                00000059 00000000 DEBUG
 26 .debug_str_offsets             000000f0 00000000 DEBUG
 27 .rela.debug_str_offsets        000002b8 00000000 
 28 .debug_str                     0000040e 00000000 DEBUG
 29 .debug_addr                    00000098 00000000 DEBUG
 30 .rela.debug_addr               000001b0 00000000 
 31 .comment                       00000073 00000000 
 32 .note.GNU-stack                00000000 00000000 
 33 .riscv.attributes              00000030 00000000 
 34 .debug_frame                   000001c4 00000000 DEBUG
 35 .rela.debug_frame              000006c0 00000000 
 36 .debug_line                    00000e49 00000000 DEBUG
 37 .rela.debug_line               0000264c 00000000 
 38 .debug_line_str                00000148 00000000 DEBUG
 39 .llvm_addrsig                  00000000 00000000 
 40 .symtab                        00002720 00000000 

Disassembly of section .text.RequantShift_u8_s8_NHWC:

00000000 <RequantShift_u8_s8_NHWC>:
;                              int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 54 b0 08  	blez	a1, 0x90 <RequantShift_u8_s8_NHWC+0x90>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 6e 0e 10  	<unknown>
      30: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 a5 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u8_s8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 84 55 04  	beq	a1, t0, 0x90 <RequantShift_u8_s8_NHWC+0x90>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 33 0f 55 00  	add	t5, a0, t0
      50: 03 4f 0f 00  	lbu	t5, 0(t5)
      54: b3 ef 02 03  	rem	t6, t0, a6
      58: 93 9f 2f 00  	slli	t6, t6, 2
      5c: 33 04 f6 01  	add	s0, a2, t6
      60: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      64: b3 8f f6 01  	add	t6, a3, t6
      68: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      6c: 33 0f 1f 01  	add	t5, t5, a7
      70: 33 0f 8f 02  	mul	t5, t5, s0
      74: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      78: db 2f ff 40  	<unknown>
      7c: b3 8f 6f 00  	add	t6, t6, t1
      80: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      84: e3 4c fe fb  	blt	t3, t6, 0x3c <RequantShift_u8_s8_NHWC+0x3c>
      88: 33 ef df 05  	<unknown>
      8c: 6f f0 1f fb  	j	0x3c <RequantShift_u8_s8_NHWC+0x3c>
; }
      90: 03 24 c1 00  	lw	s0, 12(sp)
      94: 13 01 01 01  	addi	sp, sp, 16
      98: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u16_s8_NHWC:

00000000 <RequantShift_u16_s8_NHWC>:
;                               int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_u16_s8_NHWC+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 6e 0e 10  	<unknown>
      30: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u16_s8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_u16_s8_NHWC+0x8c>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 0b 5f 25 00  	<unknown>
      50: b3 ef 02 03  	rem	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f 8f 02  	mul	t5, t5, s0
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      80: e3 4e fe fb  	blt	t3, t6, 0x3c <RequantShift_u16_s8_NHWC+0x3c>
      84: 33 ef df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_u16_s8_NHWC+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u32_s8_NHWC:

00000000 <RequantShift_u32_s8_NHWC>:
;                               int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_u32_s8_NHWC+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 6e 0e 10  	<unknown>
      30: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u32_s8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_u32_s8_NHWC+0x8c>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 0b 2f 45 00  	<unknown>
      50: b3 ef 02 03  	rem	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f e4 03  	mul	t5, s0, t5
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      80: e3 4e fe fb  	blt	t3, t6, 0x3c <RequantShift_u32_s8_NHWC+0x3c>
      84: 33 ef df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_u32_s8_NHWC+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u8_s8_NCHW:

00000000 <RequantShift_u8_s8_NCHW>:
;                              int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 54 b0 08  	blez	a1, 0x90 <RequantShift_u8_s8_NCHW+0x90>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 6e 0e 10  	<unknown>
      30: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 a5 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u8_s8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 84 55 04  	beq	a1, t0, 0x90 <RequantShift_u8_s8_NCHW+0x90>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 33 0f 55 00  	add	t5, a0, t0
      50: 03 4f 0f 00  	lbu	t5, 0(t5)
      54: b3 cf 02 03  	div	t6, t0, a6
      58: 93 9f 2f 00  	slli	t6, t6, 2
      5c: 33 04 f6 01  	add	s0, a2, t6
      60: 03 24 04 00  	lw	s0, 0(s0)
      64: b3 8f f6 01  	add	t6, a3, t6
      68: 83 af 0f 00  	lw	t6, 0(t6)
      6c: 33 0f 1f 01  	add	t5, t5, a7
      70: 33 0f 8f 02  	mul	t5, t5, s0
      74: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      78: db 2f ff 40  	<unknown>
      7c: b3 8f 6f 00  	add	t6, t6, t1
      80: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      84: e3 4c fe fb  	blt	t3, t6, 0x3c <RequantShift_u8_s8_NCHW+0x3c>
      88: 33 ef df 05  	<unknown>
      8c: 6f f0 1f fb  	j	0x3c <RequantShift_u8_s8_NCHW+0x3c>
; }
      90: 03 24 c1 00  	lw	s0, 12(sp)
      94: 13 01 01 01  	addi	sp, sp, 16
      98: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u16_s8_NCHW:

00000000 <RequantShift_u16_s8_NCHW>:
;                               int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_u16_s8_NCHW+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 6e 0e 10  	<unknown>
      30: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u16_s8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_u16_s8_NCHW+0x8c>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 0b 5f 25 00  	<unknown>
      50: b3 cf 02 03  	div	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f 8f 02  	mul	t5, t5, s0
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      80: e3 4e fe fb  	blt	t3, t6, 0x3c <RequantShift_u16_s8_NCHW+0x3c>
      84: 33 ef df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_u16_s8_NCHW+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u32_s8_NCHW:

00000000 <RequantShift_u32_s8_NCHW>:
;                               int8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_u32_s8_NCHW+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 6e 0e 10  	<unknown>
      30: b3 ee 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u32_s8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_u32_s8_NCHW+0x8c>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 0b 2f 45 00  	<unknown>
      50: b3 cf 02 03  	div	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f e4 03  	mul	t5, s0, t5
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      80: e3 4e fe fb  	blt	t3, t6, 0x3c <RequantShift_u32_s8_NCHW+0x3c>
      84: 33 ef df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_u32_s8_NCHW+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u8_u8_NHWC:

00000000 <RequantShift_u8_u8_NHWC>:
;                              uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 54 b0 08  	blez	a1, 0x90 <RequantShift_u8_u8_NHWC+0x90>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 a5 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u8_u8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 84 55 04  	beq	a1, t0, 0x90 <RequantShift_u8_u8_NHWC+0x90>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 33 0f 55 00  	add	t5, a0, t0
      50: 03 4f 0f 00  	lbu	t5, 0(t5)
      54: b3 ef 02 03  	rem	t6, t0, a6
      58: 93 9f 2f 00  	slli	t6, t6, 2
      5c: 33 04 f6 01  	add	s0, a2, t6
      60: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      64: b3 8f f6 01  	add	t6, a3, t6
      68: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      6c: 33 0f 1f 01  	add	t5, t5, a7
      70: 33 0f 8f 02  	mul	t5, t5, s0
      74: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      78: db 2f ff 40  	<unknown>
      7c: b3 8f 6f 00  	add	t6, t6, t1
      80: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP(intermediate, output_min, output_max);
      84: e3 4c fe fb  	blt	t3, t6, 0x3c <RequantShift_u8_u8_NHWC+0x3c>
      88: 33 ef df 05  	<unknown>
      8c: 6f f0 1f fb  	j	0x3c <RequantShift_u8_u8_NHWC+0x3c>
; }
      90: 03 24 c1 00  	lw	s0, 12(sp)
      94: 13 01 01 01  	addi	sp, sp, 16
      98: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u16_u8_NHWC:

00000000 <RequantShift_u16_u8_NHWC>:
;                               uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_u16_u8_NHWC+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u16_u8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_u16_u8_NHWC+0x8c>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 0b 5f 25 00  	<unknown>
      50: b3 ef 02 03  	rem	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f 8f 02  	mul	t5, t5, s0
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
      80: e3 6e fe fb  	bltu	t3, t6, 0x3c <RequantShift_u16_u8_NHWC+0x3c>
      84: 33 ff df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_u16_u8_NHWC+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u32_u8_NHWC:

00000000 <RequantShift_u32_u8_NHWC>:
;                               uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_u32_u8_NHWC+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u32_u8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_u32_u8_NHWC+0x8c>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 0b 2f 45 00  	<unknown>
      50: b3 ef 02 03  	rem	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f e4 03  	mul	t5, s0, t5
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
      80: e3 6e fe fb  	bltu	t3, t6, 0x3c <RequantShift_u32_u8_NHWC+0x3c>
      84: 33 ff df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_u32_u8_NHWC+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u8_u8_NCHW:

00000000 <RequantShift_u8_u8_NCHW>:
;                              uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 54 b0 08  	blez	a1, 0x90 <RequantShift_u8_u8_NCHW+0x90>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 a5 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u8_u8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 84 55 04  	beq	a1, t0, 0x90 <RequantShift_u8_u8_NCHW+0x90>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 33 0f 55 00  	add	t5, a0, t0
      50: 03 4f 0f 00  	lbu	t5, 0(t5)
      54: b3 cf 02 03  	div	t6, t0, a6
      58: 93 9f 2f 00  	slli	t6, t6, 2
      5c: 33 04 f6 01  	add	s0, a2, t6
      60: 03 24 04 00  	lw	s0, 0(s0)
      64: b3 8f f6 01  	add	t6, a3, t6
      68: 83 af 0f 00  	lw	t6, 0(t6)
      6c: 33 0f 1f 01  	add	t5, t5, a7
      70: 33 0f 8f 02  	mul	t5, t5, s0
      74: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      78: db 2f ff 40  	<unknown>
      7c: b3 8f 6f 00  	add	t6, t6, t1
      80: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
      84: e3 6c fe fb  	bltu	t3, t6, 0x3c <RequantShift_u8_u8_NCHW+0x3c>
      88: 33 ff df 05  	<unknown>
      8c: 6f f0 1f fb  	j	0x3c <RequantShift_u8_u8_NCHW+0x3c>
; }
      90: 03 24 c1 00  	lw	s0, 12(sp)
      94: 13 01 01 01  	addi	sp, sp, 16
      98: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u16_u8_NCHW:

00000000 <RequantShift_u16_u8_NCHW>:
;                               uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_u16_u8_NCHW+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u16_u8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_u16_u8_NCHW+0x8c>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 0b 5f 25 00  	<unknown>
      50: b3 cf 02 03  	div	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f 8f 02  	mul	t5, t5, s0
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
      80: e3 6e fe fb  	bltu	t3, t6, 0x3c <RequantShift_u16_u8_NCHW+0x3c>
      84: 33 ff df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_u16_u8_NCHW+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_u32_u8_NCHW:

00000000 <RequantShift_u32_u8_NCHW>:
;                               uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_u32_u8_NCHW+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_u32_u8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_u32_u8_NCHW+0x8c>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 0b 2f 45 00  	<unknown>
      50: b3 cf 02 03  	div	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f e4 03  	mul	t5, s0, t5
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
      80: e3 6e fe fb  	bltu	t3, t6, 0x3c <RequantShift_u32_u8_NCHW+0x3c>
      84: 33 ff df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_u32_u8_NCHW+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s8_u8_NHWC:

00000000 <RequantShift_s8_u8_NHWC>:
;                              uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 54 b0 08  	blez	a1, 0x90 <RequantShift_s8_u8_NHWC+0x90>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 a5 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_s8_u8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 84 55 04  	beq	a1, t0, 0x90 <RequantShift_s8_u8_NHWC+0x90>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 33 0f 55 00  	add	t5, a0, t0
      50: 03 0f 0f 00  	lb	t5, 0(t5)
      54: b3 ef 02 03  	rem	t6, t0, a6
      58: 93 9f 2f 00  	slli	t6, t6, 2
      5c: 33 04 f6 01  	add	s0, a2, t6
      60: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      64: b3 8f f6 01  	add	t6, a3, t6
      68: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      6c: 33 0f 1f 01  	add	t5, t5, a7
      70: 33 0f 8f 02  	mul	t5, t5, s0
      74: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      78: db 2f ff 40  	<unknown>
      7c: b3 8f 6f 00  	add	t6, t6, t1
      80: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP(intermediate, output_min, output_max);
      84: e3 4c fe fb  	blt	t3, t6, 0x3c <RequantShift_s8_u8_NHWC+0x3c>
      88: 33 ef df 05  	<unknown>
      8c: 6f f0 1f fb  	j	0x3c <RequantShift_s8_u8_NHWC+0x3c>
; }
      90: 03 24 c1 00  	lw	s0, 12(sp)
      94: 13 01 01 01  	addi	sp, sp, 16
      98: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s16_u8_NHWC:

00000000 <RequantShift_s16_u8_NHWC>:
;                               uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_s16_u8_NHWC+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_s16_u8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_s16_u8_NHWC+0x8c>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 0b 1f 25 00  	<unknown>
      50: b3 ef 02 03  	rem	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f 8f 02  	mul	t5, t5, s0
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP(intermediate, output_min, output_max);
      80: e3 4e fe fb  	blt	t3, t6, 0x3c <RequantShift_s16_u8_NHWC+0x3c>
      84: 33 ef df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_s16_u8_NHWC+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s32_u8_NHWC:

00000000 <RequantShift_s32_u8_NHWC>:
;                               uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_s32_u8_NHWC+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_s32_u8_NHWC+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_s32_u8_NHWC+0x8c>
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      4c: 0b 2f 45 00  	<unknown>
      50: b3 ef 02 03  	rem	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
;                    add[i % channels];
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
;     intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f e4 03  	mul	t5, s0, t5
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP(intermediate, output_min, output_max);
      80: e3 4e fe fb  	blt	t3, t6, 0x3c <RequantShift_s32_u8_NHWC+0x3c>
      84: 33 ef df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_s32_u8_NHWC+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s8_u8_NCHW:

00000000 <RequantShift_s8_u8_NCHW>:
;                              uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 54 b0 08  	blez	a1, 0x90 <RequantShift_s8_u8_NCHW+0x90>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 a5 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_s8_u8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 84 55 04  	beq	a1, t0, 0x90 <RequantShift_s8_u8_NCHW+0x90>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 33 0f 55 00  	add	t5, a0, t0
      50: 03 0f 0f 00  	lb	t5, 0(t5)
      54: b3 cf 02 03  	div	t6, t0, a6
      58: 93 9f 2f 00  	slli	t6, t6, 2
      5c: 33 04 f6 01  	add	s0, a2, t6
      60: 03 24 04 00  	lw	s0, 0(s0)
      64: b3 8f f6 01  	add	t6, a3, t6
      68: 83 af 0f 00  	lw	t6, 0(t6)
      6c: 33 0f 1f 01  	add	t5, t5, a7
      70: 33 0f 8f 02  	mul	t5, t5, s0
      74: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      78: db 2f ff 40  	<unknown>
      7c: b3 8f 6f 00  	add	t6, t6, t1
      80: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP(intermediate, output_min, output_max);
      84: e3 4c fe fb  	blt	t3, t6, 0x3c <RequantShift_s8_u8_NCHW+0x3c>
      88: 33 ef df 05  	<unknown>
      8c: 6f f0 1f fb  	j	0x3c <RequantShift_s8_u8_NCHW+0x3c>
; }
      90: 03 24 c1 00  	lw	s0, 12(sp)
      94: 13 01 01 01  	addi	sp, sp, 16
      98: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s16_u8_NCHW:

00000000 <RequantShift_s16_u8_NCHW>:
;                               uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_s16_u8_NCHW+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_s16_u8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_s16_u8_NCHW+0x8c>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 0b 1f 25 00  	<unknown>
      50: b3 cf 02 03  	div	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f 8f 02  	mul	t5, t5, s0
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP(intermediate, output_min, output_max);
      80: e3 4e fe fb  	blt	t3, t6, 0x3c <RequantShift_s16_u8_NCHW+0x3c>
      84: 33 ef df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_s16_u8_NCHW+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

Disassembly of section .text.RequantShift_s32_u8_NCHW:

00000000 <RequantShift_s32_u8_NCHW>:
;                               uint8_t output_max, bool rounding) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
;   for (int i = 0; i < size; i++) {
       8: 63 52 b0 08  	blez	a1, 0x8c <RequantShift_s32_u8_NCHW+0x8c>
       c: 93 02 00 00  	li	t0, 0
      10: 83 23 c1 01  	lw	t2, 28(sp)
      14: 03 2e 81 01  	lw	t3, 24(sp)
      18: 83 2e 41 01  	lw	t4, 20(sp)
      1c: 03 23 01 01  	lw	t1, 16(sp)
      20: 13 8f f7 ff  	addi	t5, a5, -1
      24: 93 f3 13 00  	andi	t2, t2, 1
      28: b3 93 e3 01  	sll	t2, t2, t5
      2c: 33 7e 0e 10  	<unknown>
      30: b3 fe 0e 10  	<unknown>
;   for (int i = 0; i < size; i++) {
      34: 7b c0 85 02  	<unknown>
      38: 6f 00 40 01  	j	0x4c <RequantShift_s32_u8_NCHW+0x4c>
;     data_out[i] = out;
      3c: b3 0f 57 00  	add	t6, a4, t0
;   for (int i = 0; i < size; i++) {
      40: 93 82 12 00  	addi	t0, t0, 1
;     data_out[i] = out;
      44: 23 80 ef 01  	sb	t5, 0(t6)
;   for (int i = 0; i < size; i++) {
      48: 63 82 55 04  	beq	a1, t0, 0x8c <RequantShift_s32_u8_NCHW+0x8c>
;         ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
      4c: 0b 2f 45 00  	<unknown>
      50: b3 cf 02 03  	div	t6, t0, a6
      54: 93 9f 2f 00  	slli	t6, t6, 2
      58: 33 04 f6 01  	add	s0, a2, t6
      5c: 03 24 04 00  	lw	s0, 0(s0)
      60: b3 8f f6 01  	add	t6, a3, t6
      64: 83 af 0f 00  	lw	t6, 0(t6)
      68: 33 0f 1f 01  	add	t5, t5, a7
      6c: 33 0f e4 03  	mul	t5, s0, t5
      70: b3 8f 7f 00  	add	t6, t6, t2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      74: db 2f ff 40  	<unknown>
      78: b3 8f 6f 00  	add	t6, t6, t1
      7c: 13 0f 0e 00  	mv	t5, t3
;     out = (uint8_t)CLAMP(intermediate, output_min, output_max);
      80: e3 4e fe fb  	blt	t3, t6, 0x3c <RequantShift_s32_u8_NCHW+0x3c>
      84: 33 ef df 05  	<unknown>
      88: 6f f0 5f fb  	j	0x3c <RequantShift_s32_u8_NCHW+0x3c>
; }
      8c: 03 24 c1 00  	lw	s0, 12(sp)
      90: 13 01 01 01  	addi	sp, sp, 16
      94: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(Softmax.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                              Size     VMA      Type
  0                                   00000000 00000000 
  1 .strtab                           00000184 00000000 
  2 .text                             00000000 00000000 TEXT
  3 .text.PULPSoftmax_u8_u8           00000124 00000000 TEXT
  4 .text.PULPSoftmax_i8_u8           00000124 00000000 TEXT
  5 .sdata                            00000008 00000000 DATA
  6 .text.PULP_Softmax_fp32_fp32      00000198 00000000 TEXT
  7 .rela.text.PULP_Softmax_fp32_fp32 0000003c 00000000 
  8 .debug_loclists                   00000735 00000000 DEBUG
  9 .debug_abbrev                     000000c8 00000000 DEBUG
 10 .debug_info                       00000487 00000000 DEBUG
 11 .rela.debug_info                  000002a0 00000000 
 12 .debug_rnglists                   00000072 00000000 DEBUG
 13 .debug_str_offsets                00000118 00000000 DEBUG
 14 .rela.debug_str_offsets           00000330 00000000 
 15 .debug_str                        0000036b 00000000 DEBUG
 16 .debug_addr                       00000050 00000000 DEBUG
 17 .rela.debug_addr                  000000d8 00000000 
 18 .comment                          00000073 00000000 
 19 .note.GNU-stack                   00000000 00000000 
 20 .riscv.attributes                 00000030 00000000 
 21 .debug_frame                      00000078 00000000 DEBUG
 22 .rela.debug_frame                 00000120 00000000 
 23 .debug_line                       000005c3 00000000 DEBUG
 24 .rela.debug_line                  00000e64 00000000 
 25 .debug_line_str                   000001e5 00000000 DEBUG
 26 .llvm_addrsig                     00000000 00000000 
 27 .symtab                           00001160 00000000 

Disassembly of section .text.PULPSoftmax_u8_u8:

00000000 <PULPSoftmax_u8_u8>:
;                        int32_t log2, int nb_dedicated_cores) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
       8: 83 22 01 01  	lw	t0, 16(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
       c: 73 23 40 f1  	csrr	t1, mhartid
;   return hart_id & 0x01f;
      10: 13 73 f3 01  	andi	t1, t1, 31
;   core_id = core_id % nb_dedicated_cores;
      14: 33 63 53 02  	rem	t1, t1, t0
;   if (core_id < (nb_dedicated_cores - 1)) {
      18: 13 8e f2 ff  	addi	t3, t0, -1
      1c: b3 d3 e6 02  	divu	t2, a3, a4
      20: b3 d2 53 02  	divu	t0, t2, t0
;   if (core_id < (nb_dedicated_cores - 1)) {
      24: 63 5e c3 01  	bge	t1, t3, 0x40 <PULPSoftmax_u8_u8+0x40>
;     offset = chunk * lastDimLength * core_id;
      28: 33 03 e3 02  	mul	t1, t1, a4
      2c: b3 06 53 02  	mul	a3, t1, t0
      30: b3 82 e2 02  	mul	t0, t0, a4
      34: b3 82 56 00  	add	t0, a3, t0
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
      38: 63 e0 56 02  	bltu	a3, t0, 0x58 <PULPSoftmax_u8_u8+0x58>
      3c: 6f 00 80 0d  	j	0x114 <PULPSoftmax_u8_u8+0x114>
;     chunk = (size / lastDimLength) - prevChunk * (nb_dedicated_cores - 1);
      40: b3 93 c2 43  	<unknown>
;     offset = size - (chunk * lastDimLength);
      44: b3 82 e3 02  	mul	t0, t2, a4
      48: b3 96 e3 42  	<unknown>
      4c: 33 03 e3 02  	mul	t1, t1, a4
      50: b3 82 56 00  	add	t0, a3, t0
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
      54: 63 f0 56 0c  	bgeu	a3, t0, 0x114 <PULPSoftmax_u8_u8+0x114>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      58: 63 04 07 0c  	beqz	a4, 0x120 <PULPSoftmax_u8_u8+0x120>
      5c: 13 13 23 00  	slli	t1, t1, 2
      60: 33 06 66 00  	add	a2, a2, t1
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
      64: 33 05 d5 00  	add	a0, a0, a3
      68: b3 85 d5 00  	add	a1, a1, a3
      6c: 13 03 f0 01  	li	t1, 31
      70: 93 03 00 00  	li	t2, 0
      74: 13 0e 00 00  	li	t3, 0
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      78: 7b 40 a7 00  	<unknown>
;       if (data_in[j + i] > x_max) {
      7c: b3 0e 75 00  	add	t4, a0, t2
      80: 83 ce 0e 00  	lbu	t4, 0(t4)
      84: 33 7e 0e 10  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      88: 93 83 13 00  	addi	t2, t2, 1
;       if (data_in[j + i] > x_max) {
      8c: 33 fe ce 05  	<unknown>
      90: 93 0e 00 00  	li	t4, 0
      94: 93 03 00 00  	li	t2, 0
      98: 13 0f 06 00  	mv	t5, a2
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      9c: 7b 40 e7 01  	<unknown>
;       xTilde = ((data_in[j + i]) - x_max);
      a0: b3 0f d5 01  	add	t6, a0, t4
      a4: 83 cf 0f 00  	lbu	t6, 0(t6)
      a8: b3 8f cf 41  	sub	t6, t6, t3
;       z = (uint8_t)(-(xTilde / log2));
      ac: 33 c4 1f 03  	div	s0, t6, a7
      b0: 33 04 80 40  	neg	s0, s0
;       z = CLAMP(z, 0, 31);
      b4: 33 74 04 10  	<unknown>
      b8: 33 54 64 04  	<unknown>
;       p = (xTilde + z * log2);
      bc: b3 0f 14 43  	<unknown>
;       intermediateResult = (uint32_t)(((p + coeffB) * (p + coeffB)) + coeffC);
      c0: b3 cf 0f 10  	<unknown>
      c4: b3 8f ff 00  	add	t6, t6, a5
      c8: b3 8f ff 03  	mul	t6, t6, t6
;       lastDimBuffer[j] = (uint32_t)(intermediateResult >> (z));
      cc: db 2f 88 c0  	<unknown>
      d0: 2b 22 ff 01  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      d4: 93 8e 1e 00  	addi	t4, t4, 1
;       y_sum += lastDimBuffer[j];
      d8: b3 83 7f 00  	add	t2, t6, t2
      dc: 13 0e 06 00  	mv	t3, a2
      e0: 93 8e 05 00  	mv	t4, a1
      e4: 13 0f 07 00  	mv	t5, a4
;       data_out[j + i] = (uint8_t)((lastDimBuffer[j] * 255) / (y_sum));
      e8: 8b 2f 4e 00  	<unknown>
      ec: 13 94 8f 00  	slli	s0, t6, 8
      f0: b3 0f f4 41  	sub	t6, s0, t6
      f4: b3 df 7f 02  	divu	t6, t6, t2
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      f8: 13 0f ff ff  	addi	t5, t5, -1
;       data_out[j + i] = (uint8_t)((lastDimBuffer[j] * 255) / (y_sum));
      fc: ab 80 fe 01  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
     100: e3 14 0f fe  	bnez	t5, 0xe8 <PULPSoftmax_u8_u8+0xe8>
;        i += lastDimLength) {
     104: b3 86 e6 00  	add	a3, a3, a4
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
     108: 33 05 e5 00  	add	a0, a0, a4
     10c: b3 85 e5 00  	add	a1, a1, a4
     110: e3 e0 56 f6  	bltu	a3, t0, 0x70 <PULPSoftmax_u8_u8+0x70>
; }
     114: 03 24 c1 00  	lw	s0, 12(sp)
     118: 13 01 01 01  	addi	sp, sp, 16
     11c: 67 80 00 00  	ret
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
     120: 6f 00 00 00  	j	0x120 <PULPSoftmax_u8_u8+0x120>

Disassembly of section .text.PULPSoftmax_i8_u8:

00000000 <PULPSoftmax_i8_u8>:
;                        int32_t log2, int nb_dedicated_cores) {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 81 00  	sw	s0, 12(sp)
       8: 83 22 01 01  	lw	t0, 16(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
       c: 73 23 40 f1  	csrr	t1, mhartid
;   return hart_id & 0x01f;
      10: 13 73 f3 01  	andi	t1, t1, 31
;   core_id = core_id % nb_dedicated_cores;
      14: 33 63 53 02  	rem	t1, t1, t0
;   if (core_id < (nb_dedicated_cores - 1)) {
      18: 13 8e f2 ff  	addi	t3, t0, -1
      1c: b3 d3 e6 02  	divu	t2, a3, a4
      20: b3 d2 53 02  	divu	t0, t2, t0
;   if (core_id < (nb_dedicated_cores - 1)) {
      24: 63 5e c3 01  	bge	t1, t3, 0x40 <PULPSoftmax_i8_u8+0x40>
;     offset = chunk * lastDimLength * core_id;
      28: 33 03 e3 02  	mul	t1, t1, a4
      2c: b3 06 53 02  	mul	a3, t1, t0
      30: b3 82 e2 02  	mul	t0, t0, a4
      34: b3 82 56 00  	add	t0, a3, t0
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
      38: 63 e0 56 02  	bltu	a3, t0, 0x58 <PULPSoftmax_i8_u8+0x58>
      3c: 6f 00 80 0d  	j	0x114 <PULPSoftmax_i8_u8+0x114>
;     chunk = (size / lastDimLength) - prevChunk * (nb_dedicated_cores - 1);
      40: b3 93 c2 43  	<unknown>
;     offset = size - (chunk * lastDimLength);
      44: b3 82 e3 02  	mul	t0, t2, a4
      48: b3 96 e3 42  	<unknown>
      4c: 33 03 e3 02  	mul	t1, t1, a4
      50: b3 82 56 00  	add	t0, a3, t0
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
      54: 63 f0 56 0c  	bgeu	a3, t0, 0x114 <PULPSoftmax_i8_u8+0x114>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      58: 63 04 07 0c  	beqz	a4, 0x120 <PULPSoftmax_i8_u8+0x120>
      5c: 13 13 23 00  	slli	t1, t1, 2
      60: 33 06 66 00  	add	a2, a2, t1
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
      64: 33 05 d5 00  	add	a0, a0, a3
      68: b3 85 d5 00  	add	a1, a1, a3
      6c: 13 03 f0 01  	li	t1, 31
      70: 93 03 00 00  	li	t2, 0
      74: 13 0e 00 08  	li	t3, 128
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      78: 7b 40 a7 00  	<unknown>
;       if (data_in[j + i] > x_max) {
      7c: b3 0e 75 00  	add	t4, a0, t2
      80: 83 8e 0e 00  	lb	t4, 0(t4)
      84: 33 6e 0e 10  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      88: 93 83 13 00  	addi	t2, t2, 1
;       if (data_in[j + i] > x_max) {
      8c: 33 ee ce 05  	<unknown>
      90: 93 0e 00 00  	li	t4, 0
      94: 93 03 00 00  	li	t2, 0
      98: 13 0f 06 00  	mv	t5, a2
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      9c: 7b 40 e7 01  	<unknown>
;       xTilde = ((data_in[j + i]) - x_max);
      a0: b3 0f d5 01  	add	t6, a0, t4
      a4: 83 8f 0f 00  	lb	t6, 0(t6)
      a8: b3 8f cf 41  	sub	t6, t6, t3
;       z = (uint8_t)(-(xTilde / log2));
      ac: 33 c4 1f 03  	div	s0, t6, a7
      b0: 33 04 80 40  	neg	s0, s0
;       z = CLAMP(z, 0, 31);
      b4: 33 74 04 10  	<unknown>
      b8: 33 54 64 04  	<unknown>
;       p = (xTilde + z * log2);
      bc: b3 0f 14 43  	<unknown>
;       intermediateResult = (((p + coeffB) * (p + coeffB)) + coeffC);
      c0: b3 cf 0f 10  	<unknown>
      c4: b3 8f ff 00  	add	t6, t6, a5
      c8: b3 8f ff 03  	mul	t6, t6, t6
;       lastDimBuffer[j] = (intermediateResult >> (z));
      cc: db 2f 88 c0  	<unknown>
      d0: 2b 22 ff 01  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      d4: 93 8e 1e 00  	addi	t4, t4, 1
;       y_sum += lastDimBuffer[j];
      d8: b3 83 7f 00  	add	t2, t6, t2
      dc: 13 0e 06 00  	mv	t3, a2
      e0: 93 8e 05 00  	mv	t4, a1
      e4: 13 0f 07 00  	mv	t5, a4
;       data_out[j + i] = (uint8_t)((lastDimBuffer[j] * 255) / (y_sum));
      e8: 8b 2f 4e 00  	<unknown>
      ec: 13 94 8f 00  	slli	s0, t6, 8
      f0: b3 0f f4 41  	sub	t6, s0, t6
      f4: b3 df 7f 02  	divu	t6, t6, t2
;     for (uint32_t j = 0; j < lastDimLength; j++) {
      f8: 13 0f ff ff  	addi	t5, t5, -1
;       data_out[j + i] = (uint8_t)((lastDimBuffer[j] * 255) / (y_sum));
      fc: ab 80 fe 01  	<unknown>
;     for (uint32_t j = 0; j < lastDimLength; j++) {
     100: e3 14 0f fe  	bnez	t5, 0xe8 <PULPSoftmax_i8_u8+0xe8>
;        i += lastDimLength) {
     104: b3 86 e6 00  	add	a3, a3, a4
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
     108: 33 05 e5 00  	add	a0, a0, a4
     10c: b3 85 e5 00  	add	a1, a1, a4
     110: e3 e0 56 f6  	bltu	a3, t0, 0x70 <PULPSoftmax_i8_u8+0x70>
; }
     114: 03 24 c1 00  	lw	s0, 12(sp)
     118: 13 01 01 01  	addi	sp, sp, 16
     11c: 67 80 00 00  	ret
;   for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
     120: 6f 00 00 00  	j	0x120 <PULPSoftmax_i8_u8+0x120>

Disassembly of section .text.PULP_Softmax_fp32_fp32:

00000000 <PULP_Softmax_fp32_fp32>:
;                             uint32_t last_dim_length, int nb_dedicated_cores) {
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
      2c: 27 2a 81 00  	fsw	fs0, 20(sp)
      30: 27 28 91 00  	fsw	fs1, 16(sp)
      34: 27 26 21 01  	fsw	fs2, 12(sp)
      38: 27 24 31 01  	fsw	fs3, 8(sp)
      3c: 27 22 41 01  	fsw	fs4, 4(sp)
      40: 13 84 06 00  	mv	s0, a3
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      44: f3 26 40 f1  	csrr	a3, mhartid
;   return hart_id & 0x01f;
      48: 93 f6 f6 01  	andi	a3, a3, 31
;   core_id = core_id % nb_dedicated_cores;
      4c: b3 e6 e6 02  	rem	a3, a3, a4
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      50: b3 17 07 10  	<unknown>
;   int32_t num_vectors = size / last_dim_length;
      54: 33 58 86 02  	divu	a6, a2, s0
;       (num_vectors >> log2Core) + ((num_vectors & (nb_dedicated_cores - 1)) != 0);
      58: 33 e6 07 10  	<unknown>
      5c: 33 56 c8 40  	sra	a2, a6, a2
      60: 13 07 f7 ff  	addi	a4, a4, -1
      64: 33 77 e8 00  	and	a4, a6, a4
      68: 33 37 e0 00  	snez	a4, a4
      6c: 33 07 e6 00  	add	a4, a2, a4
;   int32_t vector_start = MIN(chunk * core_id, num_vectors);
      70: 33 06 d7 02  	mul	a2, a4, a3
      74: 33 46 06 05  	<unknown>
;   int32_t vector_end = MIN(vector_start + chunk, num_vectors);
      78: b3 06 e6 00  	add	a3, a2, a4
      7c: b3 c6 06 05  	<unknown>
;   int32_t local_vectors = vector_end - vector_start;
      80: b3 84 c6 40  	sub	s1, a3, a2
;   if (local_vectors <= 0) {
      84: 63 58 90 0c  	blez	s1, 0x154 <PULP_Softmax_fp32_fp32+0x154>
;     for (int32_t i = 0; i < last_dim_length; i++) {
      88: 63 06 04 0c  	beqz	s0, 0x154 <PULP_Softmax_fp32_fp32+0x154>
      8c: 13 09 00 00  	li	s2, 0
      90: 33 06 86 02  	mul	a2, a2, s0
      94: 13 16 26 00  	slli	a2, a2, 2
      98: b7 06 00 00  	lui	a3, 0
      9c: 07 a4 06 00  	flw	fs0, 0(a3)
      a0: b7 06 00 00  	lui	a3, 0
      a4: 87 a4 06 00  	flw	fs1, 0(a3)
      a8: b3 09 c5 00  	add	s3, a0, a2
      ac: 33 8a c5 00  	add	s4, a1, a2
      b0: 53 09 00 f0  	fmv.w.x	fs2, zero
      b4: 13 05 00 00  	li	a0, 0
      b8: b3 0a 89 02  	mul	s5, s2, s0
      bc: 93 05 04 00  	mv	a1, s0
      c0: d3 09 84 20  	fmv.s	fs3, fs0
;     for (int32_t i = 0; i < last_dim_length; i++) {
      c4: 7b 40 c4 00  	<unknown>
;       if (local_input[b * last_dim_length + i] > max_val) {
      c8: 33 06 55 01  	add	a2, a0, s5
      cc: 13 16 26 00  	slli	a2, a2, 2
      d0: 33 86 c9 00  	add	a2, s3, a2
      d4: 07 20 06 00  	flw	ft0, 0(a2)
      d8: d3 19 30 29  	fmax.s	fs3, ft0, fs3
;     for (int32_t i = 0; i < last_dim_length; i++) {
      dc: 13 05 15 00  	addi	a0, a0, 1
      e0: 13 0b 00 00  	li	s6, 0
      e4: 93 0b 04 00  	mv	s7, s0
      e8: 53 0a 29 21  	fmv.s	fs4, fs2
;       float32_t exp_val = local_input[b * last_dim_length + i] - max_val;
      ec: 33 05 5b 01  	add	a0, s6, s5
      f0: 13 1c 25 00  	slli	s8, a0, 2
      f4: 33 85 89 01  	add	a0, s3, s8
      f8: 07 20 05 00  	flw	ft0, 0(a0)
      fc: 53 75 30 09  	fsub.s	fa0, ft0, fs3
;       local_output[b * last_dim_length + i] = expf(exp_val);
     100: 97 00 00 00  	auipc	ra, 0
     104: e7 80 00 00  	jalr	ra
     108: 33 05 8a 01  	add	a0, s4, s8
     10c: 27 20 a5 00  	fsw	fa0, 0(a0)
;       sum += local_output[b * last_dim_length + i];
     110: 53 7a 45 01  	fadd.s	fs4, fa0, fs4
;     for (int32_t i = 0; i < last_dim_length; i++) {
     114: 93 8b fb ff  	addi	s7, s7, -1
     118: 13 0b 1b 00  	addi	s6, s6, 1
     11c: e3 98 0b fc  	bnez	s7, 0xec <PULP_Softmax_fp32_fp32+0xec>
     120: 13 05 00 00  	li	a0, 0
     124: 53 f0 44 19  	fdiv.s	ft0, fs1, fs4
     128: 93 05 04 00  	mv	a1, s0
;     for (int32_t i = 0; i < last_dim_length; i++) {
     12c: 7b 40 e4 00  	<unknown>
;       local_output[b * last_dim_length + i] *= inv_sum;
     130: 33 06 55 01  	add	a2, a0, s5
     134: 13 16 26 00  	slli	a2, a2, 2
     138: 33 06 ca 00  	add	a2, s4, a2
     13c: 87 20 06 00  	flw	ft1, 0(a2)
     140: d3 f0 00 10  	fmul.s	ft1, ft1, ft0
     144: 27 20 16 00  	fsw	ft1, 0(a2)
;     for (int32_t i = 0; i < last_dim_length; i++) {
     148: 13 05 15 00  	addi	a0, a0, 1
;   for (int32_t b = 0; b < local_vectors; b++) {
     14c: 13 09 19 00  	addi	s2, s2, 1
     150: e3 12 99 f6  	bne	s2, s1, 0xb4 <PULP_Softmax_fp32_fp32+0xb4>
; }
     154: 83 20 c1 03  	lw	ra, 60(sp)
     158: 03 24 81 03  	lw	s0, 56(sp)
     15c: 83 24 41 03  	lw	s1, 52(sp)
     160: 03 29 01 03  	lw	s2, 48(sp)
     164: 83 29 c1 02  	lw	s3, 44(sp)
     168: 03 2a 81 02  	lw	s4, 40(sp)
     16c: 83 2a 41 02  	lw	s5, 36(sp)
     170: 03 2b 01 02  	lw	s6, 32(sp)
     174: 83 2b c1 01  	lw	s7, 28(sp)
     178: 03 2c 81 01  	lw	s8, 24(sp)
     17c: 07 24 41 01  	flw	fs0, 20(sp)
     180: 87 24 01 01  	flw	fs1, 16(sp)
     184: 07 29 c1 00  	flw	fs2, 12(sp)
     188: 87 29 81 00  	flw	fs3, 8(sp)
     18c: 07 2a 41 00  	flw	fs4, 4(sp)
     190: 13 01 01 04  	addi	sp, sp, 64
     194: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(UniformRequantShift.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                             Size     VMA      Type
  0                                  00000000 00000000 
  1 .strtab                          000001a2 00000000 
  2 .text                            00000000 00000000 TEXT
  3 .text.UniformRequantShift_s8_s8  000001c4 00000000 TEXT
  4 .text.UniformRequantShift_u8_s8  000001c4 00000000 TEXT
  5 .text.UniformRequantShift_s16_s8 000001e4 00000000 TEXT
  6 .text.UniformRequantShift_s32_s8 000001e0 00000000 TEXT
  7 .debug_loclists                  000005d6 00000000 DEBUG
  8 .debug_abbrev                    000000ed 00000000 DEBUG
  9 .debug_info                      00000617 00000000 DEBUG
 10 .rela.debug_info                 00000228 00000000 
 11 .debug_rnglists                  000000c1 00000000 DEBUG
 12 .debug_str_offsets               000000e8 00000000 DEBUG
 13 .rela.debug_str_offsets          000002a0 00000000 
 14 .debug_str                       0000034d 00000000 DEBUG
 15 .debug_addr                      00000038 00000000 DEBUG
 16 .rela.debug_addr                 00000090 00000000 
 17 .comment                         00000073 00000000 
 18 .note.GNU-stack                  00000000 00000000 
 19 .riscv.attributes                00000030 00000000 
 20 .debug_frame                     0000008c 00000000 DEBUG
 21 .rela.debug_frame                00000180 00000000 
 22 .debug_line                      00000aef 00000000 DEBUG
 23 .rela.debug_line                 00001cd4 00000000 
 24 .debug_line_str                  000001d9 00000000 DEBUG
 25 .llvm_addrsig                    00000000 00000000 
 26 .symtab                          000019b0 00000000 

Disassembly of section .text.UniformRequantShift_s8_s8:

00000000 <UniformRequantShift_s8_s8>:
;                                int8_t output_max, bool rounding, int nb_dedicated_cores) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 81 00  	sw	s0, 28(sp)
       8: 23 2c 91 00  	sw	s1, 24(sp)
       c: 23 2a 21 01  	sw	s2, 20(sp)
      10: 23 28 31 01  	sw	s3, 16(sp)
      14: 03 23 c1 02  	lw	t1, 44(sp)
      18: 83 22 81 02  	lw	t0, 40(sp)
      1c: 03 2f 41 02  	lw	t5, 36(sp)
      20: 83 23 01 03  	lw	t2, 48(sp)
      24: 03 28 01 02  	lw	a6, 32(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      28: 73 2e 40 f1  	csrr	t3, mhartid
;   return hart_id & 0x01f;
      2c: 13 7e fe 01  	andi	t3, t3, 31
;   core_id = core_id % nb_dedicated_cores;
      30: 33 6e 7e 02  	rem	t3, t3, t2
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      34: b3 9e 03 10  	<unknown>
;   int16_t chunk = (size >> log2Core) + ((size & (nb_dedicated_cores - 1)) != 0);
      38: b3 ee 0e 10  	<unknown>
      3c: b3 de d5 41  	sra	t4, a1, t4
      40: 93 83 f3 ff  	addi	t2, t2, -1
      44: b3 f3 b3 00  	and	t2, t2, a1
      48: b3 33 70 00  	snez	t2, t2
      4c: b3 8e 7e 00  	add	t4, t4, t2
;   int16_t chunk_start = MIN(chunk * core_id, size);
      50: b3 cf 0e 10  	<unknown>
      54: b3 83 cf 03  	mul	t2, t6, t3
      58: b3 c3 b3 04  	<unknown>
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      5c: 33 ce 03 10  	<unknown>
      60: b3 0f fe 01  	add	t6, t3, t6
;   int32_t volatile halfChunkSize = chunk >> 1;
      64: b3 8e 1e dc  	<unknown>
      68: 23 26 d1 01  	sw	t4, 12(sp)
;   reg_data_in_A = data_in[chunk_start];
      6c: b3 0e c5 01  	add	t4, a0, t3
      70: 83 c4 0e 00  	lbu	s1, 0(t4)
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      74: 03 24 c1 00  	lw	s0, 12(sp)
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      78: 93 85 15 00  	addi	a1, a1, 1
      7c: b3 ce bf 04  	<unknown>
      80: 13 73 13 00  	andi	t1, t1, 1
      84: b3 65 0f 10  	<unknown>
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      88: 63 54 80 0a  	blez	s0, 0x130 <UniformRequantShift_s8_s8+0x130>
      8c: 13 8f f7 ff  	addi	t5, a5, -1
      90: 33 1f e3 01  	sll	t5, t1, t5
      94: 33 0f df 00  	add	t5, t5, a3
      98: b3 ef 02 10  	<unknown>
      9c: 13 04 0e 00  	mv	s0, t3
      a0: 6f 00 80 02  	j	0xc8 <UniformRequantShift_s8_s8+0xc8>
;     data_out[halfChunkSize + i] = out;
      a4: 83 29 c1 00  	lw	s3, 12(sp)
      a8: b3 09 34 01  	add	s3, s0, s3
      ac: b3 09 37 01  	add	s3, a4, s3
      b0: 23 80 29 01  	sb	s2, 0(s3)
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      b4: 03 29 c1 00  	lw	s2, 12(sp)
      b8: 93 09 14 00  	addi	s3, s0, 1
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      bc: 33 09 c9 01  	add	s2, s2, t3
      c0: 13 84 09 00  	mv	s0, s3
      c4: 63 d6 29 07  	bge	s3, s2, 0x130 <UniformRequantShift_s8_s8+0x130>
;     reg_data_in_B = data_in[halfChunkSize + i];
      c8: 03 29 c1 00  	lw	s2, 12(sp)
;     intermediate = (reg_data_in_A + input_offset) * mul + add;
      cc: b3 e4 04 10  	<unknown>
      d0: b3 84 14 01  	add	s1, s1, a7
      d4: b3 84 c4 02  	mul	s1, s1, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      d8: 93 09 0f 00  	mv	s3, t5
      dc: db a9 f4 40  	<unknown>
      e0: b3 89 09 01  	add	s3, s3, a6
      e4: 93 84 0f 00  	mv	s1, t6
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      e8: 63 c4 3f 01  	blt	t6, s3, 0xf0 <UniformRequantShift_s8_s8+0xf0>
      ec: b3 e4 b9 04  	<unknown>
      f0: 33 09 24 01  	add	s2, s0, s2
      f4: 33 09 25 01  	add	s2, a0, s2
      f8: 03 09 09 00  	lb	s2, 0(s2)
;     data_out[i] = out;
      fc: b3 09 87 00  	add	s3, a4, s0
     100: 23 80 99 00  	sb	s1, 0(s3)
;     reg_data_in_A = data_in[i + 1];
     104: b3 04 85 00  	add	s1, a0, s0
     108: 83 c4 14 00  	lbu	s1, 1(s1)
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     10c: 33 09 19 01  	add	s2, s2, a7
     110: 33 09 c9 02  	mul	s2, s2, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     114: 93 09 0f 00  	mv	s3, t5
     118: db 29 f9 40  	<unknown>
     11c: b3 89 09 01  	add	s3, s3, a6
     120: 13 89 0f 00  	mv	s2, t6
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     124: e3 c0 3f f9  	blt	t6, s3, 0xa4 <UniformRequantShift_s8_s8+0xa4>
     128: 33 e9 b9 04  	<unknown>
     12c: 6f f0 9f f7  	j	0xa4 <UniformRequantShift_s8_s8+0xa4>
;   if ((chunk_stop - chunk_start) % 2) {
     130: b3 83 7e 40  	sub	t2, t4, t2
     134: 93 f3 13 00  	andi	t2, t2, 1
     138: 63 8a 03 06  	beqz	t2, 0x1ac <UniformRequantShift_s8_s8+0x1ac>
     13c: b3 c3 0e 10  	<unknown>
;     reg_data_in_B = data_in[chunk_stop - 1];
     140: 13 8e f3 ff  	addi	t3, t2, -1
     144: b3 0e c5 01  	add	t4, a0, t3
     148: 83 8e 0e 00  	lb	t4, 0(t4)
     14c: b3 e2 02 10  	<unknown>
;     reg_data_in_A = data_in[chunk_stop];
     150: 33 05 75 00  	add	a0, a0, t2
     154: 03 05 05 00  	lb	a0, 0(a0)
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     158: b3 8e 1e 01  	add	t4, t4, a7
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     15c: 13 8f f7 ff  	addi	t5, a5, -1
     160: 33 13 e3 01  	sll	t1, t1, t5
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     164: b3 06 d3 00  	add	a3, t1, a3
     168: 33 83 ce 02  	mul	t1, t4, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     16c: 93 8e 06 00  	mv	t4, a3
     170: db 2e f3 40  	<unknown>
     174: b3 8e 0e 01  	add	t4, t4, a6
     178: 13 83 02 00  	mv	t1, t0
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     17c: 63 c4 d2 01  	blt	t0, t4, 0x184 <UniformRequantShift_s8_s8+0x184>
     180: 33 e3 be 04  	<unknown>
;     data_out[chunk_stop - 1] = out;
     184: 33 0e c7 01  	add	t3, a4, t3
;     intermediate = (reg_data_in_A + input_offset) * mul + add;
     188: 33 05 15 01  	add	a0, a0, a7
     18c: 33 05 c5 02  	mul	a0, a0, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     190: db 26 f5 40  	<unknown>
     194: 33 85 06 01  	add	a0, a3, a6
;     data_out[chunk_stop - 1] = out;
     198: 23 00 6e 00  	sb	t1, 0(t3)
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     19c: 63 c4 a2 00  	blt	t0, a0, 0x1a4 <UniformRequantShift_s8_s8+0x1a4>
     1a0: b3 62 b5 04  	<unknown>
;     data_out[chunk_stop] = out;
     1a4: 33 05 77 00  	add	a0, a4, t2
     1a8: 23 00 55 00  	sb	t0, 0(a0)
; }
     1ac: 03 24 c1 01  	lw	s0, 28(sp)
     1b0: 83 24 81 01  	lw	s1, 24(sp)
     1b4: 03 29 41 01  	lw	s2, 20(sp)
     1b8: 83 29 01 01  	lw	s3, 16(sp)
     1bc: 13 01 01 02  	addi	sp, sp, 32
     1c0: 67 80 00 00  	ret

Disassembly of section .text.UniformRequantShift_u8_s8:

00000000 <UniformRequantShift_u8_s8>:
;                                int8_t output_max, bool rounding, int nb_dedicated_cores) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 81 00  	sw	s0, 28(sp)
       8: 23 2c 91 00  	sw	s1, 24(sp)
       c: 23 2a 21 01  	sw	s2, 20(sp)
      10: 23 28 31 01  	sw	s3, 16(sp)
      14: 03 23 c1 02  	lw	t1, 44(sp)
      18: 83 22 81 02  	lw	t0, 40(sp)
      1c: 03 2f 41 02  	lw	t5, 36(sp)
      20: 83 23 01 03  	lw	t2, 48(sp)
      24: 03 28 01 02  	lw	a6, 32(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      28: 73 2e 40 f1  	csrr	t3, mhartid
;   return hart_id & 0x01f;
      2c: 13 7e fe 01  	andi	t3, t3, 31
;   core_id = core_id % nb_dedicated_cores;
      30: 33 6e 7e 02  	rem	t3, t3, t2
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      34: b3 9e 03 10  	<unknown>
;   int16_t chunk = (size >> log2Core) + ((size & (nb_dedicated_cores - 1)) != 0);
      38: b3 ee 0e 10  	<unknown>
      3c: b3 de d5 41  	sra	t4, a1, t4
      40: 93 83 f3 ff  	addi	t2, t2, -1
      44: b3 f3 b3 00  	and	t2, t2, a1
      48: b3 33 70 00  	snez	t2, t2
      4c: b3 8e 7e 00  	add	t4, t4, t2
;   int16_t chunk_start = MIN(chunk * core_id, size);
      50: b3 cf 0e 10  	<unknown>
      54: b3 83 cf 03  	mul	t2, t6, t3
      58: b3 c3 b3 04  	<unknown>
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      5c: 33 ce 03 10  	<unknown>
      60: b3 0f fe 01  	add	t6, t3, t6
;   int32_t volatile halfChunkSize = chunk >> 1;
      64: b3 8e 1e dc  	<unknown>
      68: 23 26 d1 01  	sw	t4, 12(sp)
;   reg_data_in_A = data_in[chunk_start];
      6c: b3 0e c5 01  	add	t4, a0, t3
      70: 83 c4 0e 00  	lbu	s1, 0(t4)
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      74: 03 24 c1 00  	lw	s0, 12(sp)
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      78: 93 85 15 00  	addi	a1, a1, 1
      7c: b3 ce bf 04  	<unknown>
      80: 13 73 13 00  	andi	t1, t1, 1
      84: b3 65 0f 10  	<unknown>
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      88: 63 54 80 0a  	blez	s0, 0x130 <UniformRequantShift_u8_s8+0x130>
      8c: 13 8f f7 ff  	addi	t5, a5, -1
      90: 33 1f e3 01  	sll	t5, t1, t5
      94: 33 0f df 00  	add	t5, t5, a3
      98: b3 ef 02 10  	<unknown>
      9c: 13 04 0e 00  	mv	s0, t3
      a0: 6f 00 80 02  	j	0xc8 <UniformRequantShift_u8_s8+0xc8>
;     data_out[halfChunkSize + i] = out;
      a4: 83 29 c1 00  	lw	s3, 12(sp)
      a8: b3 09 34 01  	add	s3, s0, s3
      ac: b3 09 37 01  	add	s3, a4, s3
      b0: 23 80 29 01  	sb	s2, 0(s3)
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      b4: 03 29 c1 00  	lw	s2, 12(sp)
      b8: 93 09 14 00  	addi	s3, s0, 1
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      bc: 33 09 c9 01  	add	s2, s2, t3
      c0: 13 84 09 00  	mv	s0, s3
      c4: 63 d6 29 07  	bge	s3, s2, 0x130 <UniformRequantShift_u8_s8+0x130>
;     reg_data_in_B = data_in[halfChunkSize + i];
      c8: 03 29 c1 00  	lw	s2, 12(sp)
;     intermediate = (reg_data_in_A + input_offset) * mul + add;
      cc: b3 f4 04 10  	<unknown>
      d0: b3 84 14 01  	add	s1, s1, a7
      d4: b3 84 c4 02  	mul	s1, s1, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      d8: 93 09 0f 00  	mv	s3, t5
      dc: db a9 f4 40  	<unknown>
      e0: b3 89 09 01  	add	s3, s3, a6
      e4: 93 84 0f 00  	mv	s1, t6
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      e8: 63 c4 3f 01  	blt	t6, s3, 0xf0 <UniformRequantShift_u8_s8+0xf0>
      ec: b3 e4 b9 04  	<unknown>
      f0: 33 09 24 01  	add	s2, s0, s2
      f4: 33 09 25 01  	add	s2, a0, s2
      f8: 03 49 09 00  	lbu	s2, 0(s2)
;     data_out[i] = out;
      fc: b3 09 87 00  	add	s3, a4, s0
     100: 23 80 99 00  	sb	s1, 0(s3)
;     reg_data_in_A = data_in[i + 1];
     104: b3 04 85 00  	add	s1, a0, s0
     108: 83 c4 14 00  	lbu	s1, 1(s1)
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     10c: 33 09 19 01  	add	s2, s2, a7
     110: 33 09 c9 02  	mul	s2, s2, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     114: 93 09 0f 00  	mv	s3, t5
     118: db 29 f9 40  	<unknown>
     11c: b3 89 09 01  	add	s3, s3, a6
     120: 13 89 0f 00  	mv	s2, t6
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     124: e3 c0 3f f9  	blt	t6, s3, 0xa4 <UniformRequantShift_u8_s8+0xa4>
     128: 33 e9 b9 04  	<unknown>
     12c: 6f f0 9f f7  	j	0xa4 <UniformRequantShift_u8_s8+0xa4>
;   if ((chunk_stop - chunk_start) % 2) {
     130: b3 83 7e 40  	sub	t2, t4, t2
     134: 93 f3 13 00  	andi	t2, t2, 1
     138: 63 8a 03 06  	beqz	t2, 0x1ac <UniformRequantShift_u8_s8+0x1ac>
     13c: b3 c3 0e 10  	<unknown>
;     reg_data_in_B = data_in[chunk_stop - 1];
     140: 13 8e f3 ff  	addi	t3, t2, -1
     144: b3 0e c5 01  	add	t4, a0, t3
     148: 83 ce 0e 00  	lbu	t4, 0(t4)
     14c: b3 e2 02 10  	<unknown>
;     reg_data_in_A = data_in[chunk_stop];
     150: 33 05 75 00  	add	a0, a0, t2
     154: 03 45 05 00  	lbu	a0, 0(a0)
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     158: b3 8e 1e 01  	add	t4, t4, a7
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     15c: 13 8f f7 ff  	addi	t5, a5, -1
     160: 33 13 e3 01  	sll	t1, t1, t5
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     164: b3 06 d3 00  	add	a3, t1, a3
     168: 33 83 ce 02  	mul	t1, t4, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     16c: 93 8e 06 00  	mv	t4, a3
     170: db 2e f3 40  	<unknown>
     174: b3 8e 0e 01  	add	t4, t4, a6
     178: 13 83 02 00  	mv	t1, t0
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     17c: 63 c4 d2 01  	blt	t0, t4, 0x184 <UniformRequantShift_u8_s8+0x184>
     180: 33 e3 be 04  	<unknown>
;     data_out[chunk_stop - 1] = out;
     184: 33 0e c7 01  	add	t3, a4, t3
;     intermediate = (reg_data_in_A + input_offset) * mul + add;
     188: 33 05 15 01  	add	a0, a0, a7
     18c: 33 05 c5 02  	mul	a0, a0, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     190: db 26 f5 40  	<unknown>
     194: 33 85 06 01  	add	a0, a3, a6
;     data_out[chunk_stop - 1] = out;
     198: 23 00 6e 00  	sb	t1, 0(t3)
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     19c: 63 c4 a2 00  	blt	t0, a0, 0x1a4 <UniformRequantShift_u8_s8+0x1a4>
     1a0: b3 62 b5 04  	<unknown>
;     data_out[chunk_stop] = out;
     1a4: 33 05 77 00  	add	a0, a4, t2
     1a8: 23 00 55 00  	sb	t0, 0(a0)
; }
     1ac: 03 24 c1 01  	lw	s0, 28(sp)
     1b0: 83 24 81 01  	lw	s1, 24(sp)
     1b4: 03 29 41 01  	lw	s2, 20(sp)
     1b8: 83 29 01 01  	lw	s3, 16(sp)
     1bc: 13 01 01 02  	addi	sp, sp, 32
     1c0: 67 80 00 00  	ret

Disassembly of section .text.UniformRequantShift_s16_s8:

00000000 <UniformRequantShift_s16_s8>:
;                                 int8_t output_max, bool rounding, int nb_dedicated_cores) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 81 00  	sw	s0, 28(sp)
       8: 23 2c 91 00  	sw	s1, 24(sp)
       c: 23 2a 21 01  	sw	s2, 20(sp)
      10: 23 28 31 01  	sw	s3, 16(sp)
      14: 23 26 41 01  	sw	s4, 12(sp)
      18: 03 23 c1 02  	lw	t1, 44(sp)
      1c: 83 22 81 02  	lw	t0, 40(sp)
      20: 03 2f 41 02  	lw	t5, 36(sp)
      24: 83 23 01 03  	lw	t2, 48(sp)
      28: 03 28 01 02  	lw	a6, 32(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      2c: 73 2e 40 f1  	csrr	t3, mhartid
;   return hart_id & 0x01f;
      30: 13 7e fe 01  	andi	t3, t3, 31
;   core_id = core_id % nb_dedicated_cores;
      34: 33 6e 7e 02  	rem	t3, t3, t2
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      38: b3 9e 03 10  	<unknown>
;   int16_t chunk = (size >> log2Core) + ((size & (nb_dedicated_cores - 1)) != 0);
      3c: b3 ee 0e 10  	<unknown>
      40: b3 de d5 41  	sra	t4, a1, t4
      44: 93 83 f3 ff  	addi	t2, t2, -1
      48: b3 f3 b3 00  	and	t2, t2, a1
      4c: b3 33 70 00  	snez	t2, t2
      50: b3 8e 7e 00  	add	t4, t4, t2
;   int16_t chunk_start = MIN(chunk * core_id, size);
      54: b3 cf 0e 10  	<unknown>
      58: b3 83 cf 03  	mul	t2, t6, t3
      5c: b3 c3 b3 04  	<unknown>
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      60: 33 ce 03 10  	<unknown>
      64: b3 0f fe 01  	add	t6, t3, t6
;   int32_t volatile halfChunkSize = chunk >> 1;
      68: b3 8e 1e dc  	<unknown>
      6c: 23 24 d1 01  	sw	t4, 8(sp)
;   reg_data_in_A = data_in[chunk_start];
      70: 13 14 1e 00  	slli	s0, t3, 1
      74: b3 0e 85 00  	add	t4, a0, s0
      78: 03 d9 0e 00  	lhu	s2, 0(t4)
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      7c: 83 24 81 00  	lw	s1, 8(sp)
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      80: 93 85 15 00  	addi	a1, a1, 1
      84: b3 ce bf 04  	<unknown>
      88: 13 73 13 00  	andi	t1, t1, 1
      8c: b3 65 0f 10  	<unknown>
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      90: 63 58 90 0a  	blez	s1, 0x140 <UniformRequantShift_s16_s8+0x140>
      94: 13 8f f7 ff  	addi	t5, a5, -1
      98: 33 1f e3 01  	sll	t5, t1, t5
      9c: 33 0f df 00  	add	t5, t5, a3
      a0: b3 ef 02 10  	<unknown>
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      a4: 33 04 a4 00  	add	s0, s0, a0
      a8: 13 04 24 00  	addi	s0, s0, 2
      ac: 93 04 0e 00  	mv	s1, t3
      b0: 6f 00 80 02  	j	0xd8 <UniformRequantShift_s16_s8+0xd8>
;     data_out[halfChunkSize + i] = out;
      b4: 03 2a 81 00  	lw	s4, 8(sp)
      b8: 33 8a 44 01  	add	s4, s1, s4
      bc: 33 0a 47 01  	add	s4, a4, s4
      c0: 23 00 3a 01  	sb	s3, 0(s4)
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      c4: 83 29 81 00  	lw	s3, 8(sp)
      c8: 13 8a 14 00  	addi	s4, s1, 1
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      cc: b3 89 c9 01  	add	s3, s3, t3
      d0: 93 04 0a 00  	mv	s1, s4
      d4: 63 56 3a 07  	bge	s4, s3, 0x140 <UniformRequantShift_s16_s8+0x140>
;     reg_data_in_B = data_in[halfChunkSize + i];
      d8: 83 29 81 00  	lw	s3, 8(sp)
;     intermediate = (reg_data_in_A + input_offset) * mul + add;
      dc: 33 49 09 10  	<unknown>
      e0: 33 09 19 01  	add	s2, s2, a7
      e4: 33 09 c9 02  	mul	s2, s2, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      e8: 13 0a 0f 00  	mv	s4, t5
      ec: 5b 2a f9 40  	<unknown>
      f0: 33 0a 0a 01  	add	s4, s4, a6
      f4: 13 89 0f 00  	mv	s2, t6
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      f8: 63 c4 4f 01  	blt	t6, s4, 0x100 <UniformRequantShift_s16_s8+0x100>
      fc: 33 69 ba 04  	<unknown>
     100: b3 89 34 01  	add	s3, s1, s3
     104: 93 99 19 00  	slli	s3, s3, 1
     108: b3 09 35 01  	add	s3, a0, s3
     10c: 83 99 09 00  	lh	s3, 0(s3)
;     data_out[i] = out;
     110: 33 0a 97 00  	add	s4, a4, s1
     114: 23 00 2a 01  	sb	s2, 0(s4)
;     reg_data_in_A = data_in[i + 1];
     118: 0b 59 24 00  	<unknown>
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     11c: b3 89 19 01  	add	s3, s3, a7
     120: b3 89 c9 02  	mul	s3, s3, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     124: 13 0a 0f 00  	mv	s4, t5
     128: 5b aa f9 40  	<unknown>
     12c: 33 0a 0a 01  	add	s4, s4, a6
     130: 93 89 0f 00  	mv	s3, t6
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     134: e3 c0 4f f9  	blt	t6, s4, 0xb4 <UniformRequantShift_s16_s8+0xb4>
     138: b3 69 ba 04  	<unknown>
     13c: 6f f0 9f f7  	j	0xb4 <UniformRequantShift_s16_s8+0xb4>
;   if ((chunk_stop - chunk_start) % 2) {
     140: b3 83 7e 40  	sub	t2, t4, t2
     144: 93 f3 13 00  	andi	t2, t2, 1
     148: 63 80 03 08  	beqz	t2, 0x1c8 <UniformRequantShift_s16_s8+0x1c8>
     14c: b3 e2 02 10  	<unknown>
;   if ((chunk_stop - chunk_start) % 2) {
     150: 13 9f 0e 01  	slli	t5, t4, 16
     154: b3 c3 0e 10  	<unknown>
;     reg_data_in_B = data_in[chunk_stop - 1];
     158: 13 8e f3 ff  	addi	t3, t2, -1
     15c: 93 1e 1e 00  	slli	t4, t3, 1
     160: b3 0e d5 01  	add	t4, a0, t4
     164: 83 9e 0e 00  	lh	t4, 0(t4)
;     reg_data_in_A = data_in[chunk_stop];
     168: 13 5f ff 40  	srai	t5, t5, 15
     16c: 33 05 e5 01  	add	a0, a0, t5
     170: 03 15 05 00  	lh	a0, 0(a0)
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     174: b3 8e 1e 01  	add	t4, t4, a7
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     178: 13 8f f7 ff  	addi	t5, a5, -1
     17c: 33 13 e3 01  	sll	t1, t1, t5
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     180: b3 06 d3 00  	add	a3, t1, a3
     184: 33 83 ce 02  	mul	t1, t4, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     188: 93 8e 06 00  	mv	t4, a3
     18c: db 2e f3 40  	<unknown>
     190: b3 8e 0e 01  	add	t4, t4, a6
     194: 13 83 02 00  	mv	t1, t0
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     198: 63 c4 d2 01  	blt	t0, t4, 0x1a0 <UniformRequantShift_s16_s8+0x1a0>
     19c: 33 e3 be 04  	<unknown>
;     data_out[chunk_stop - 1] = out;
     1a0: 33 0e c7 01  	add	t3, a4, t3
;     intermediate = (reg_data_in_A + input_offset) * mul + add;
     1a4: 33 05 15 01  	add	a0, a0, a7
     1a8: 33 05 c5 02  	mul	a0, a0, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     1ac: db 26 f5 40  	<unknown>
     1b0: 33 85 06 01  	add	a0, a3, a6
;     data_out[chunk_stop - 1] = out;
     1b4: 23 00 6e 00  	sb	t1, 0(t3)
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     1b8: 63 c4 a2 00  	blt	t0, a0, 0x1c0 <UniformRequantShift_s16_s8+0x1c0>
     1bc: b3 62 b5 04  	<unknown>
;     data_out[chunk_stop] = out;
     1c0: 33 05 77 00  	add	a0, a4, t2
     1c4: 23 00 55 00  	sb	t0, 0(a0)
; }
     1c8: 03 24 c1 01  	lw	s0, 28(sp)
     1cc: 83 24 81 01  	lw	s1, 24(sp)
     1d0: 03 29 41 01  	lw	s2, 20(sp)
     1d4: 83 29 01 01  	lw	s3, 16(sp)
     1d8: 03 2a c1 00  	lw	s4, 12(sp)
     1dc: 13 01 01 02  	addi	sp, sp, 32
     1e0: 67 80 00 00  	ret

Disassembly of section .text.UniformRequantShift_s32_s8:

00000000 <UniformRequantShift_s32_s8>:
;                                 int8_t output_max, bool rounding, int nb_dedicated_cores) {
       0: 13 01 01 fe  	addi	sp, sp, -32
       4: 23 2e 81 00  	sw	s0, 28(sp)
       8: 23 2c 91 00  	sw	s1, 24(sp)
       c: 23 2a 21 01  	sw	s2, 20(sp)
      10: 23 28 31 01  	sw	s3, 16(sp)
      14: 23 26 41 01  	sw	s4, 12(sp)
      18: 03 23 c1 02  	lw	t1, 44(sp)
      1c: 83 22 81 02  	lw	t0, 40(sp)
      20: 03 2f 41 02  	lw	t5, 36(sp)
      24: 83 23 01 03  	lw	t2, 48(sp)
      28: 03 28 01 02  	lw	a6, 32(sp)
;   asm("csrr %0, 0xF14" : "=r" (hart_id) : );
      2c: 73 2e 40 f1  	csrr	t3, mhartid
;   return hart_id & 0x01f;
      30: 13 7e fe 01  	andi	t3, t3, 31
;   core_id = core_id % nb_dedicated_cores;
      34: 33 6e 7e 02  	rem	t3, t3, t2
;   int8_t log2Core = LOG2(nb_dedicated_cores);
      38: b3 9e 03 10  	<unknown>
;   int16_t chunk = (size >> log2Core) + ((size & (nb_dedicated_cores - 1)) != 0);
      3c: b3 ee 0e 10  	<unknown>
      40: b3 de d5 41  	sra	t4, a1, t4
      44: 93 83 f3 ff  	addi	t2, t2, -1
      48: b3 f3 b3 00  	and	t2, t2, a1
      4c: b3 33 70 00  	snez	t2, t2
      50: b3 8e 7e 00  	add	t4, t4, t2
;   int16_t chunk_start = MIN(chunk * core_id, size);
      54: b3 cf 0e 10  	<unknown>
      58: b3 83 cf 03  	mul	t2, t6, t3
      5c: b3 c3 b3 04  	<unknown>
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      60: 33 ce 03 10  	<unknown>
      64: b3 0f fe 01  	add	t6, t3, t6
;   int32_t volatile halfChunkSize = chunk >> 1;
      68: b3 8e 1e dc  	<unknown>
      6c: 23 24 d1 01  	sw	t4, 8(sp)
;   reg_data_in_A = data_in[chunk_start];
      70: 13 14 2e 00  	slli	s0, t3, 2
      74: b3 0e 85 00  	add	t4, a0, s0
      78: 03 a9 0e 00  	lw	s2, 0(t4)
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      7c: 83 24 81 00  	lw	s1, 8(sp)
;   int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);
      80: 93 85 15 00  	addi	a1, a1, 1
      84: b3 ce bf 04  	<unknown>
      88: 13 73 13 00  	andi	t1, t1, 1
      8c: b3 65 0f 10  	<unknown>
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      90: 63 56 90 0a  	blez	s1, 0x13c <UniformRequantShift_s32_s8+0x13c>
      94: 13 8f f7 ff  	addi	t5, a5, -1
      98: 33 1f e3 01  	sll	t5, t1, t5
      9c: 33 0f df 00  	add	t5, t5, a3
      a0: b3 ef 02 10  	<unknown>
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      a4: 33 04 a4 00  	add	s0, s0, a0
      a8: 13 04 44 00  	addi	s0, s0, 4
      ac: 93 04 0e 00  	mv	s1, t3
      b0: 6f 00 80 02  	j	0xd8 <UniformRequantShift_s32_s8+0xd8>
;     data_out[halfChunkSize + i] = out;
      b4: 03 2a 81 00  	lw	s4, 8(sp)
      b8: 33 8a 44 01  	add	s4, s1, s4
      bc: 33 0a 47 01  	add	s4, a4, s4
      c0: 23 00 3a 01  	sb	s3, 0(s4)
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      c4: 83 29 81 00  	lw	s3, 8(sp)
      c8: 13 8a 14 00  	addi	s4, s1, 1
;   for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {
      cc: b3 89 c9 01  	add	s3, s3, t3
      d0: 93 04 0a 00  	mv	s1, s4
      d4: 63 54 3a 07  	bge	s4, s3, 0x13c <UniformRequantShift_s32_s8+0x13c>
;     reg_data_in_B = data_in[halfChunkSize + i];
      d8: 83 29 81 00  	lw	s3, 8(sp)
;     intermediate = (reg_data_in_A + input_offset) * mul + add;
      dc: 33 09 19 01  	add	s2, s2, a7
      e0: 33 09 c9 02  	mul	s2, s2, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
      e4: 13 0a 0f 00  	mv	s4, t5
      e8: 5b 2a f9 40  	<unknown>
      ec: 33 0a 0a 01  	add	s4, s4, a6
      f0: 13 89 0f 00  	mv	s2, t6
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
      f4: 63 c4 4f 01  	blt	t6, s4, 0xfc <UniformRequantShift_s32_s8+0xfc>
      f8: 33 69 ba 04  	<unknown>
      fc: b3 89 34 01  	add	s3, s1, s3
     100: 93 99 29 00  	slli	s3, s3, 2
     104: b3 09 35 01  	add	s3, a0, s3
     108: 83 a9 09 00  	lw	s3, 0(s3)
;     data_out[i] = out;
     10c: 33 0a 97 00  	add	s4, a4, s1
     110: 23 00 2a 01  	sb	s2, 0(s4)
;     reg_data_in_A = data_in[i + 1];
     114: 0b 29 44 00  	<unknown>
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     118: b3 89 19 01  	add	s3, s3, a7
     11c: b3 89 c9 02  	mul	s3, s3, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     120: 13 0a 0f 00  	mv	s4, t5
     124: 5b aa f9 40  	<unknown>
     128: 33 0a 0a 01  	add	s4, s4, a6
     12c: 93 89 0f 00  	mv	s3, t6
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     130: e3 c2 4f f9  	blt	t6, s4, 0xb4 <UniformRequantShift_s32_s8+0xb4>
     134: b3 69 ba 04  	<unknown>
     138: 6f f0 df f7  	j	0xb4 <UniformRequantShift_s32_s8+0xb4>
;   if ((chunk_stop - chunk_start) % 2) {
     13c: b3 83 7e 40  	sub	t2, t4, t2
     140: 93 f3 13 00  	andi	t2, t2, 1
     144: 63 80 03 08  	beqz	t2, 0x1c4 <UniformRequantShift_s32_s8+0x1c4>
     148: b3 e2 02 10  	<unknown>
;   if ((chunk_stop - chunk_start) % 2) {
     14c: 13 9f 0e 01  	slli	t5, t4, 16
     150: b3 c3 0e 10  	<unknown>
;     reg_data_in_B = data_in[chunk_stop - 1];
     154: 13 8e f3 ff  	addi	t3, t2, -1
     158: 93 1e 2e 00  	slli	t4, t3, 2
     15c: b3 0e d5 01  	add	t4, a0, t4
     160: 83 ae 0e 00  	lw	t4, 0(t4)
;     reg_data_in_A = data_in[chunk_stop];
     164: 13 5f ef 40  	srai	t5, t5, 14
     168: 33 05 e5 01  	add	a0, a0, t5
     16c: 03 25 05 00  	lw	a0, 0(a0)
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     170: b3 8e 1e 01  	add	t4, t4, a7
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     174: 13 8f f7 ff  	addi	t5, a5, -1
     178: 33 13 e3 01  	sll	t1, t1, t5
;     intermediate = (reg_data_in_B + input_offset) * mul + add;
     17c: b3 06 d3 00  	add	a3, t1, a3
     180: 33 83 ce 02  	mul	t1, t4, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     184: 93 8e 06 00  	mv	t4, a3
     188: db 2e f3 40  	<unknown>
     18c: b3 8e 0e 01  	add	t4, t4, a6
     190: 13 83 02 00  	mv	t1, t0
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     194: 63 c4 d2 01  	blt	t0, t4, 0x19c <UniformRequantShift_s32_s8+0x19c>
     198: 33 e3 be 04  	<unknown>
;     data_out[chunk_stop - 1] = out;
     19c: 33 0e c7 01  	add	t3, a4, t3
;     intermediate = (reg_data_in_A + input_offset) * mul + add;
     1a0: 33 05 15 01  	add	a0, a0, a7
     1a4: 33 05 c5 02  	mul	a0, a0, a2
;     intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
     1a8: db 26 f5 40  	<unknown>
     1ac: 33 85 06 01  	add	a0, a3, a6
;     data_out[chunk_stop - 1] = out;
     1b0: 23 00 6e 00  	sb	t1, 0(t3)
;     out = (int8_t)CLAMP(intermediate, output_min, output_max);
     1b4: 63 c4 a2 00  	blt	t0, a0, 0x1bc <UniformRequantShift_s32_s8+0x1bc>
     1b8: b3 62 b5 04  	<unknown>
;     data_out[chunk_stop] = out;
     1bc: 33 05 77 00  	add	a0, a4, t2
     1c0: 23 00 55 00  	sb	t0, 0(a0)
; }
     1c4: 03 24 c1 01  	lw	s0, 28(sp)
     1c8: 83 24 81 01  	lw	s1, 24(sp)
     1cc: 03 29 41 01  	lw	s2, 20(sp)
     1d0: 83 29 01 01  	lw	s3, 16(sp)
     1d4: 03 2a c1 00  	lw	s4, 12(sp)
     1d8: 13 01 01 02  	addi	sp, sp, 32
     1dc: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(Util.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                      Size     VMA      Type
  0                           00000000 00000000 
  1 .strtab                   00000162 00000000 
  2 .text                     00000000 00000000 TEXT
  3 .text.deeploy_log         00000044 00000000 TEXT
  4 .rela.text.deeploy_log    0000000c 00000000 
  5 .text.deeploy_malloc      00000008 00000000 TEXT
  6 .rela.text.deeploy_malloc 0000000c 00000000 
  7 .text.deeploy_free        00000008 00000000 TEXT
  8 .rela.text.deeploy_free   0000000c 00000000 
  9 .debug_loclists           00000022 00000000 DEBUG
 10 .debug_abbrev             000000c9 00000000 DEBUG
 11 .debug_info               000000c6 00000000 DEBUG
 12 .rela.debug_info          00000090 00000000 
 13 .debug_rnglists           0000001a 00000000 DEBUG
 14 .debug_str_offsets        0000004c 00000000 DEBUG
 15 .rela.debug_str_offsets   000000cc 00000000 
 16 .debug_str                0000016a 00000000 DEBUG
 17 .debug_addr               00000014 00000000 DEBUG
 18 .rela.debug_addr          00000024 00000000 
 19 .comment                  00000073 00000000 
 20 .note.GNU-stack           00000000 00000000 
 21 .riscv.attributes         00000030 00000000 
 22 .debug_frame              0000004c 00000000 DEBUG
 23 .rela.debug_frame         000000c0 00000000 
 24 .debug_line               000000e5 00000000 DEBUG
 25 .rela.debug_line          00000138 00000000 
 26 .debug_line_str           00000117 00000000 DEBUG
 27 .llvm_addrsig             00000000 00000000 
 28 .symtab                   00000370 00000000 

Disassembly of section .text.deeploy_log:

00000000 <deeploy_log>:
; int deeploy_log(const char *__restrict fmt, ...) {
       0: 13 01 01 fd  	addi	sp, sp, -48
       4: 23 26 11 00  	sw	ra, 12(sp)
       8: 23 26 11 03  	sw	a7, 44(sp)
       c: 23 24 01 03  	sw	a6, 40(sp)
      10: 23 22 f1 02  	sw	a5, 36(sp)
      14: 23 20 e1 02  	sw	a4, 32(sp)
      18: 23 2e d1 00  	sw	a3, 28(sp)
      1c: 23 2c c1 00  	sw	a2, 24(sp)
      20: 23 2a b1 00  	sw	a1, 20(sp)
      24: 93 05 41 01  	addi	a1, sp, 20
;   va_start(args, fmt);
      28: 23 24 b1 00  	sw	a1, 8(sp)
;   ret = vprintf(fmt, args);
      2c: 93 05 41 01  	addi	a1, sp, 20
      30: 97 00 00 00  	auipc	ra, 0
      34: e7 80 00 00  	jalr	ra
;   return ret;
      38: 83 20 c1 00  	lw	ra, 12(sp)
      3c: 13 01 01 03  	addi	sp, sp, 48
      40: 67 80 00 00  	ret

Disassembly of section .text.deeploy_malloc:

00000000 <deeploy_malloc>:
; void *deeploy_malloc(const size_t size) { return malloc(size); }
       0: 17 03 00 00  	auipc	t1, 0
       4: 67 00 03 00  	jr	t1

Disassembly of section .text.deeploy_free:

00000000 <deeploy_free>:
; void deeploy_free(void *const ptr) { free(ptr); }
       0: 17 03 00 00  	auipc	t1, 0
       4: 67 00 03 00  	jr	t1

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(dory_mem.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                          Size     VMA      Type
  0                               00000000 00000000 
  1 .strtab                       000003de 00000000 
  2 .text                         00000000 00000000 TEXT
  3 .text.open_fs                 00000094 00000000 TEXT
  4 .rela.text.open_fs            000000b4 00000000 
  5 .text.mem_init                000000ec 00000000 TEXT
  6 .rela.text.mem_init           00000138 00000000 
  7 .text.get_ram_ptr             0000000c 00000000 TEXT
  8 .rela.text.get_ram_ptr        00000018 00000000 
  9 .text.ram_malloc              00000034 00000000 TEXT
 10 .rela.text.ram_malloc         00000024 00000000 
 11 .text.ram_free                00000024 00000000 TEXT
 12 .rela.text.ram_free           00000024 00000000 
 13 .text.ram_read                00000078 00000000 TEXT
 14 .rela.text.ram_read           00000048 00000000 
 15 .text.ram_write               0000007c 00000000 TEXT
 16 .rela.text.ram_write          00000048 00000000 
 17 .text.cl_ram_malloc           0000005c 00000000 TEXT
 18 .rela.text.cl_ram_malloc      00000024 00000000 
 19 .text.cl_ram_free             0000005c 00000000 TEXT
 20 .rela.text.cl_ram_free        00000024 00000000 
 21 .text.cl_ram_read             00000060 00000000 TEXT
 22 .rela.text.cl_ram_read        00000024 00000000 
 23 .text.cl_ram_write            00000064 00000000 TEXT
 24 .rela.text.cl_ram_write       00000024 00000000 
 25 .text.load_file_to_ram        00000180 00000000 TEXT
 26 .rela.text.load_file_to_ram   000000b4 00000000 
 27 .text.load_file_to_local      00000150 00000000 TEXT
 28 .rela.text.load_file_to_local 0000009c 00000000 
 29 .bss.fs_conf                  00000014 00000000 BSS
 30 .bss.flash                    0000000c 00000000 BSS
 31 .bss.fs                       0000000c 00000000 BSS
 32 .bss.flash_conf               00000014 00000000 BSS
 33 .bss.ram_conf                 00000024 00000000 BSS
 34 .bss.ram                      0000000c 00000000 BSS
 35 .rodata.str1.1                0000009a 00000000 DATA
 36 .bss.buffer                   00000800 00000000 BSS
 37 .debug_loclists               00000458 00000000 DEBUG
 38 .debug_abbrev                 00000350 00000000 DEBUG
 39 .debug_info                   00001a53 00000000 DEBUG
 40 .rela.debug_info              00000840 00000000 
 41 .debug_rnglists               00000081 00000000 DEBUG
 42 .debug_str_offsets            0000035c 00000000 DEBUG
 43 .rela.debug_str_offsets       000009fc 00000000 
 44 .debug_str                    00000a12 00000000 DEBUG
 45 .debug_addr                   0000013c 00000000 DEBUG
 46 .rela.debug_addr              0000039c 00000000 
 47 .comment                      00000073 00000000 
 48 .note.GNU-stack               00000000 00000000 
 49 .riscv.attributes             00000030 00000000 
 50 .debug_frame                  00000170 00000000 DEBUG
 51 .rela.debug_frame             00000480 00000000 
 52 .debug_line                   00000942 00000000 DEBUG
 53 .rela.debug_line              00001428 00000000 
 54 .debug_line_str               00000506 00000000 DEBUG
 55 .llvm_addrsig                 0000000b 00000000 
 56 .symtab                       000026b0 00000000 

Disassembly of section .text.open_fs:

00000000 <open_fs>:
; void open_fs() {
       0: 13 01 01 ff  	addi	sp, sp, -16
       4: 23 26 11 00  	sw	ra, 12(sp)
       8: 23 24 81 00  	sw	s0, 8(sp)
       c: 23 22 91 00  	sw	s1, 4(sp)
      10: 37 05 00 00  	lui	a0, 0
      14: 13 04 05 00  	mv	s0, a0
;   pi_readfs_conf_init(&fs_conf);
      18: 13 05 04 00  	mv	a0, s0
      1c: 97 00 00 00  	auipc	ra, 0
      20: e7 80 00 00  	jalr	ra
;   fs_conf.fs.flash = &flash;
      24: 37 05 00 00  	lui	a0, 0
      28: 13 05 05 00  	mv	a0, a0
      2c: 23 22 a4 00  	sw	a0, 4(s0)
;   pi_open_from_conf(&fs, &fs_conf);
      30: 37 05 00 00  	lui	a0, 0
      34: 93 04 05 00  	mv	s1, a0
      38: 13 85 04 00  	mv	a0, s1
      3c: 93 05 04 00  	mv	a1, s0
      40: 97 00 00 00  	auipc	ra, 0
      44: e7 80 00 00  	jalr	ra
;   if (pi_fs_mount(&fs)) {
      48: 13 85 04 00  	mv	a0, s1
      4c: 97 00 00 00  	auipc	ra, 0
      50: e7 80 00 00  	jalr	ra
      54: 63 1c 05 00  	bnez	a0, 0x6c <open_fs+0x6c>
; }
      58: 83 20 c1 00  	lw	ra, 12(sp)
      5c: 03 24 81 00  	lw	s0, 8(sp)
      60: 83 24 41 00  	lw	s1, 4(sp)
      64: 13 01 01 01  	addi	sp, sp, 16
      68: 67 80 00 00  	ret
;     printf("ERROR: Cannot mount filesystem! Exiting...\n");
      6c: 37 05 00 00  	lui	a0, 0
      70: 13 05 05 00  	mv	a0, a0
      74: 97 00 00 00  	auipc	ra, 0
      78: e7 80 00 00  	jalr	ra
;     pos_kernel_pmsis_exit_value = err;
      7c: 37 05 00 00  	lui	a0, 0
      80: 93 05 e0 ff  	li	a1, -2
      84: 23 20 b5 00  	sw	a1, 0(a0)
;     exit(err);
      88: 13 05 e0 ff  	li	a0, -2
      8c: 97 00 00 00  	auipc	ra, 0
      90: e7 80 00 00  	jalr	ra

Disassembly of section .text.mem_init:

00000000 <mem_init>:
; void mem_init() {
       0: 13 01 01 ff  	addi	sp, sp, -16
;   flash_conf_init(&flash_conf);
       4: 23 26 11 00  	sw	ra, 12(sp)
       8: 23 24 81 00  	sw	s0, 8(sp)
       c: 23 22 91 00  	sw	s1, 4(sp)
      10: 37 05 00 00  	lui	a0, 0
      14: 13 04 05 00  	mv	s0, a0
      18: 13 05 04 00  	mv	a0, s0
      1c: 97 00 00 00  	auipc	ra, 0
      20: e7 80 00 00  	jalr	ra
;   pi_open_from_conf(&flash, &flash_conf);
      24: 37 05 00 00  	lui	a0, 0
      28: 93 04 05 00  	mv	s1, a0
      2c: 13 85 04 00  	mv	a0, s1
      30: 93 05 04 00  	mv	a1, s0
      34: 97 00 00 00  	auipc	ra, 0
      38: e7 80 00 00  	jalr	ra
;   if (pi_flash_open(&flash)) {
      3c: 13 85 04 00  	mv	a0, s1
      40: 97 00 00 00  	auipc	ra, 0
      44: e7 80 00 00  	jalr	ra
      48: 63 1a 05 04  	bnez	a0, 0x9c <mem_init+0x9c>
;   ram_conf_init(&ram_conf);
      4c: 37 05 00 00  	lui	a0, 0
      50: 13 04 05 00  	mv	s0, a0
      54: 13 05 04 00  	mv	a0, s0
      58: 97 00 00 00  	auipc	ra, 0
      5c: e7 80 00 00  	jalr	ra
;   pi_open_from_conf(&ram, &ram_conf);
      60: 37 05 00 00  	lui	a0, 0
      64: 93 04 05 00  	mv	s1, a0
      68: 13 85 04 00  	mv	a0, s1
      6c: 93 05 04 00  	mv	a1, s0
      70: 97 00 00 00  	auipc	ra, 0
      74: e7 80 00 00  	jalr	ra
;   if (pi_ram_open(&ram)) {
      78: 13 85 04 00  	mv	a0, s1
      7c: 97 00 00 00  	auipc	ra, 0
      80: e7 80 00 00  	jalr	ra
      84: 63 10 05 04  	bnez	a0, 0xc4 <mem_init+0xc4>
; }
      88: 83 20 c1 00  	lw	ra, 12(sp)
      8c: 03 24 81 00  	lw	s0, 8(sp)
      90: 83 24 41 00  	lw	s1, 4(sp)
      94: 13 01 01 01  	addi	sp, sp, 16
      98: 67 80 00 00  	ret
;     printf("ERROR: Cannot open flash! Exiting...\n");
      9c: 37 05 00 00  	lui	a0, 0
      a0: 13 05 05 00  	mv	a0, a0
      a4: 97 00 00 00  	auipc	ra, 0
      a8: e7 80 00 00  	jalr	ra
;     pos_kernel_pmsis_exit_value = err;
      ac: 37 05 00 00  	lui	a0, 0
      b0: 93 05 f0 ff  	li	a1, -1
      b4: 23 20 b5 00  	sw	a1, 0(a0)
;     exit(err);
      b8: 13 05 f0 ff  	li	a0, -1
      bc: 97 00 00 00  	auipc	ra, 0
      c0: e7 80 00 00  	jalr	ra
;     printf("ERROR: Cannot open ram! Exiting...\n");
      c4: 37 05 00 00  	lui	a0, 0
      c8: 13 05 05 00  	mv	a0, a0
      cc: 97 00 00 00  	auipc	ra, 0
      d0: e7 80 00 00  	jalr	ra
;     pos_kernel_pmsis_exit_value = err;
      d4: 37 05 00 00  	lui	a0, 0
      d8: 93 05 d0 ff  	li	a1, -3
      dc: 23 20 b5 00  	sw	a1, 0(a0)
;     exit(err);
      e0: 13 05 d0 ff  	li	a0, -3
      e4: 97 00 00 00  	auipc	ra, 0
      e8: e7 80 00 00  	jalr	ra

Disassembly of section .text.get_ram_ptr:

00000000 <get_ram_ptr>:
; struct pi_device *get_ram_ptr() { return &ram; }
       0: 37 05 00 00  	lui	a0, 0
       4: 13 05 05 00  	mv	a0, a0
       8: 67 80 00 00  	ret

Disassembly of section .text.ram_malloc:

00000000 <ram_malloc>:
; void *ram_malloc(size_t size) {
       0: 13 01 01 ff  	addi	sp, sp, -16
;     pi_ram_api_t *api = (pi_ram_api_t *)device->api;
       4: 23 26 11 00  	sw	ra, 12(sp)
       8: b7 05 00 00  	lui	a1, 0
       c: 03 a6 05 00  	lw	a2, 0(a1)
;     return api->alloc(device, addr, size);
      10: 83 26 46 01  	lw	a3, 20(a2)
      14: 13 06 05 00  	mv	a2, a0
;     pi_ram_api_t *api = (pi_ram_api_t *)device->api;
      18: 13 85 05 00  	mv	a0, a1
;     return api->alloc(device, addr, size);
      1c: 93 05 81 00  	addi	a1, sp, 8
      20: e7 80 06 00  	jalr	a3
;   return (void *)ptr;
      24: 03 25 81 00  	lw	a0, 8(sp)
      28: 83 20 c1 00  	lw	ra, 12(sp)
      2c: 13 01 01 01  	addi	sp, sp, 16
      30: 67 80 00 00  	ret

Disassembly of section .text.ram_free:

00000000 <ram_free>:
;     pi_ram_api_t *api = (pi_ram_api_t *)device->api;
       0: 37 06 00 00  	lui	a2, 0
       4: 83 26 06 00  	lw	a3, 0(a2)
;     return api->free(device, addr, size);
       8: 03 a3 86 01  	lw	t1, 24(a3)
;     pi_ram_api_t *api = (pi_ram_api_t *)device->api;
       c: 13 06 06 00  	mv	a2, a2
      10: 93 86 05 00  	mv	a3, a1
      14: 93 05 05 00  	mv	a1, a0
;     return api->free(device, addr, size);
      18: 13 05 06 00  	mv	a0, a2
      1c: 13 86 06 00  	mv	a2, a3
      20: 67 00 03 00  	jr	t1

Disassembly of section .text.ram_read:

00000000 <ram_read>:
; void ram_read(void *dest, void *src, const size_t size) {
       0: 13 01 01 fb  	addi	sp, sp, -80
; 	task->arg[0] = (uint32_t)pos_task_handle_blocking;
       4: 23 26 11 04  	sw	ra, 76(sp)
       8: 23 24 81 04  	sw	s0, 72(sp)
       c: b7 06 00 00  	lui	a3, 0
      10: 93 86 06 00  	mv	a3, a3
      14: 23 26 d1 00  	sw	a3, 12(sp)
;     pi_ram_api_t *api = (pi_ram_api_t *)device->api;
      18: 37 07 00 00  	lui	a4, 0
      1c: 83 26 07 00  	lw	a3, 0(a4)
      20: 93 07 81 00  	addi	a5, sp, 8
; 	task->arg[1] = (uint32_t)task;
      24: 23 28 f1 00  	sw	a5, 16(sp)
;   	task->done = 0;
      28: 23 0e 01 00  	sb	zero, 28(sp)
;     api->copy_async(device, pi_ram_addr, data, size, 1, task);
      2c: 03 a8 c6 00  	lw	a6, 12(a3)
      30: 93 06 06 00  	mv	a3, a2
      34: 13 06 05 00  	mv	a2, a0
;     pi_ram_api_t *api = (pi_ram_api_t *)device->api;
      38: 13 05 07 00  	mv	a0, a4
;     api->copy_async(device, pi_ram_addr, data, size, 1, task);
      3c: 13 07 10 00  	li	a4, 1
      40: 93 07 81 00  	addi	a5, sp, 8
      44: e7 00 08 00  	jalr	a6
;   int irq = hal_spr_read_then_clr(0x300, 0x1<<3);
      48: 73 74 04 30  	csrrci	s0, mstatus, 8
;     while(likely(task->done == 0))
      4c: 03 45 c1 01  	lbu	a0, 28(sp)
      50: 63 1a 05 00  	bnez	a0, 0x64 <ram_read+0x64>
;         pos_task_handle();
      54: 97 00 00 00  	auipc	ra, 0
      58: e7 80 00 00  	jalr	ra
;     while(likely(task->done == 0))
      5c: 03 45 c1 01  	lbu	a0, 28(sp)
      60: e3 0a 05 fe  	beqz	a0, 0x54 <ram_read+0x54>
;   hal_spr_write(0x300, state);
      64: 73 10 04 30  	csrw	mstatus, s0
; }
      68: 83 20 c1 04  	lw	ra, 76(sp)
      6c: 03 24 81 04  	lw	s0, 72(sp)
      70: 13 01 01 05  	addi	sp, sp, 80
      74: 67 80 00 00  	ret

Disassembly of section .text.ram_write:

00000000 <ram_write>:
; void ram_write(void *dest, void *src, const size_t size) {
       0: 13 01 01 fb  	addi	sp, sp, -80
; 	task->arg[0] = (uint32_t)pos_task_handle_blocking;
       4: 23 26 11 04  	sw	ra, 76(sp)
       8: 23 24 81 04  	sw	s0, 72(sp)
       c: b7 06 00 00  	lui	a3, 0
      10: 93 86 06 00  	mv	a3, a3
      14: 23 26 d1 00  	sw	a3, 12(sp)
;     pi_ram_api_t *api = (pi_ram_api_t *)device->api;
      18: 37 07 00 00  	lui	a4, 0
      1c: 83 26 07 00  	lw	a3, 0(a4)
      20: 93 07 81 00  	addi	a5, sp, 8
; 	task->arg[1] = (uint32_t)task;
      24: 23 28 f1 00  	sw	a5, 16(sp)
;   	task->done = 0;
      28: 23 0e 01 00  	sb	zero, 28(sp)
;     api->copy_async(device, pi_ram_addr, data, size, 0, task);
      2c: 03 a8 c6 00  	lw	a6, 12(a3)
      30: 93 06 06 00  	mv	a3, a2
      34: 13 86 05 00  	mv	a2, a1
      38: 93 05 05 00  	mv	a1, a0
;     pi_ram_api_t *api = (pi_ram_api_t *)device->api;
      3c: 13 05 07 00  	mv	a0, a4
;     api->copy_async(device, pi_ram_addr, data, size, 0, task);
      40: 93 07 81 00  	addi	a5, sp, 8
      44: 13 07 00 00  	li	a4, 0
      48: e7 00 08 00  	jalr	a6
;   int irq = hal_spr_read_then_clr(0x300, 0x1<<3);
      4c: 73 74 04 30  	csrrci	s0, mstatus, 8
;     while(likely(task->done == 0))
      50: 03 45 c1 01  	lbu	a0, 28(sp)
      54: 63 1a 05 00  	bnez	a0, 0x68 <ram_write+0x68>
;         pos_task_handle();
      58: 97 00 00 00  	auipc	ra, 0
      5c: e7 80 00 00  	jalr	ra
;     while(likely(task->done == 0))
      60: 03 45 c1 01  	lbu	a0, 28(sp)
      64: e3 0a 05 fe  	beqz	a0, 0x58 <ram_write+0x58>
;   hal_spr_write(0x300, state);
      68: 73 10 04 30  	csrw	mstatus, s0
; }
      6c: 83 20 c1 04  	lw	ra, 76(sp)
      70: 03 24 81 04  	lw	s0, 72(sp)
      74: 13 01 01 05  	addi	sp, sp, 80
      78: 67 80 00 00  	ret

Disassembly of section .text.cl_ram_malloc:

00000000 <cl_ram_malloc>:
; void *cl_ram_malloc(size_t size) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 93 05 05 00  	mv	a1, a0
;   pi_cl_ram_alloc(&ram, size, &req);
       c: 37 05 00 00  	lui	a0, 0
      10: 13 05 05 00  	mv	a0, a0
      14: 13 06 81 00  	addi	a2, sp, 8
      18: 97 00 00 00  	auipc	ra, 0
      1c: e7 80 00 00  	jalr	ra
;     while ((*(volatile char *)done) == 0)
      20: 03 45 41 05  	lbu	a0, 84(sp)
      24: 63 14 05 02  	bnez	a0, 0x4c <cl_ram_malloc+0x4c>
      28: 13 05 80 00  	li	a0, 8
      2c: b7 45 20 00  	lui	a1, 516
      30: 13 06 20 00  	li	a2, 2
      34: 93 06 40 00  	li	a3, 4
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_OR, evtMask);
      38: 23 e5 c5 00  	<unknown>
;   value = pulp_read32(base + offset);
      3c: 03 a7 c5 03  	lw	a4, 60(a1)
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_AND, evtMask);
      40: a3 e6 c5 00  	<unknown>
;     while ((*(volatile char *)done) == 0)
      44: 03 47 41 05  	lbu	a4, 84(sp)
      48: e3 08 07 fe  	beqz	a4, 0x38 <cl_ram_malloc+0x38>
;     *chunk = req->result;
      4c: 03 25 c1 00  	lw	a0, 12(sp)
;   return (void *)addr;
      50: 83 20 c1 05  	lw	ra, 92(sp)
      54: 13 01 01 06  	addi	sp, sp, 96
      58: 67 80 00 00  	ret

Disassembly of section .text.cl_ram_free:

00000000 <cl_ram_free>:
; void cl_ram_free(void *ptr, size_t size) {
       0: 13 01 01 fa  	addi	sp, sp, -96
       4: 23 2e 11 04  	sw	ra, 92(sp)
       8: 13 86 05 00  	mv	a2, a1
       c: 93 05 05 00  	mv	a1, a0
;   pi_cl_ram_free(&ram, (uint32_t)ptr, size, &req);
      10: 37 05 00 00  	lui	a0, 0
      14: 13 05 05 00  	mv	a0, a0
      18: 93 06 81 00  	addi	a3, sp, 8
      1c: 97 00 00 00  	auipc	ra, 0
      20: e7 80 00 00  	jalr	ra
;     while ((*(volatile char *)done) == 0)
      24: 03 45 81 05  	lbu	a0, 88(sp)
      28: 63 14 05 02  	bnez	a0, 0x50 <cl_ram_free+0x50>
      2c: 13 05 80 00  	li	a0, 8
      30: b7 45 20 00  	lui	a1, 516
      34: 13 06 20 00  	li	a2, 2
      38: 93 06 40 00  	li	a3, 4
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_OR, evtMask);
      3c: 23 e5 c5 00  	<unknown>
;   value = pulp_read32(base + offset);
      40: 03 a7 c5 03  	lw	a4, 60(a1)
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_AND, evtMask);
      44: a3 e6 c5 00  	<unknown>
;     while ((*(volatile char *)done) == 0)
      48: 03 47 81 05  	lbu	a4, 88(sp)
      4c: e3 08 07 fe  	beqz	a4, 0x3c <cl_ram_free+0x3c>
; }
      50: 83 20 c1 05  	lw	ra, 92(sp)
      54: 13 01 01 06  	addi	sp, sp, 96
      58: 67 80 00 00  	ret

Disassembly of section .text.cl_ram_read:

00000000 <cl_ram_read>:
; void cl_ram_read(void *dest, void *src, const size_t size) {
       0: 13 01 01 f9  	addi	sp, sp, -112
       4: 23 26 11 06  	sw	ra, 108(sp)
       8: 93 06 06 00  	mv	a3, a2
       c: 13 06 05 00  	mv	a2, a0
;     pi_cl_ram_copy(device, pi_ram_addr, addr, size, 1, req);
      10: 37 05 00 00  	lui	a0, 0
      14: 13 05 05 00  	mv	a0, a0
      18: 13 07 10 00  	li	a4, 1
      1c: 93 07 81 00  	addi	a5, sp, 8
      20: 97 00 00 00  	auipc	ra, 0
      24: e7 80 00 00  	jalr	ra
;     while ((*(volatile char *)done) == 0)
      28: 03 45 41 06  	lbu	a0, 100(sp)
      2c: 63 14 05 02  	bnez	a0, 0x54 <cl_ram_read+0x54>
      30: 13 05 80 00  	li	a0, 8
      34: b7 45 20 00  	lui	a1, 516
      38: 13 06 20 00  	li	a2, 2
      3c: 93 06 40 00  	li	a3, 4
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_OR, evtMask);
      40: 23 e5 c5 00  	<unknown>
;   value = pulp_read32(base + offset);
      44: 03 a7 c5 03  	lw	a4, 60(a1)
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_AND, evtMask);
      48: a3 e6 c5 00  	<unknown>
;     while ((*(volatile char *)done) == 0)
      4c: 03 47 41 06  	lbu	a4, 100(sp)
      50: e3 08 07 fe  	beqz	a4, 0x40 <cl_ram_read+0x40>
; }
      54: 83 20 c1 06  	lw	ra, 108(sp)
      58: 13 01 01 07  	addi	sp, sp, 112
      5c: 67 80 00 00  	ret

Disassembly of section .text.cl_ram_write:

00000000 <cl_ram_write>:
; void cl_ram_write(void *dest, void *src, const size_t size) {
       0: 13 01 01 f9  	addi	sp, sp, -112
       4: 23 26 11 06  	sw	ra, 108(sp)
       8: 93 06 06 00  	mv	a3, a2
       c: 13 86 05 00  	mv	a2, a1
      10: 93 05 05 00  	mv	a1, a0
;     pi_cl_ram_copy(device, pi_ram_addr, addr, size, 0, req);
      14: 37 05 00 00  	lui	a0, 0
      18: 13 05 05 00  	mv	a0, a0
      1c: 93 07 81 00  	addi	a5, sp, 8
      20: 13 07 00 00  	li	a4, 0
      24: 97 00 00 00  	auipc	ra, 0
      28: e7 80 00 00  	jalr	ra
;     while ((*(volatile char *)done) == 0)
      2c: 03 45 41 06  	lbu	a0, 100(sp)
      30: 63 14 05 02  	bnez	a0, 0x58 <cl_ram_write+0x58>
      34: 13 05 80 00  	li	a0, 8
      38: b7 45 20 00  	lui	a1, 516
      3c: 13 06 20 00  	li	a2, 2
      40: 93 06 40 00  	li	a3, 4
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_OR, evtMask);
      44: 23 e5 c5 00  	<unknown>
;   value = pulp_read32(base + offset);
      48: 03 a7 c5 03  	lw	a4, 60(a1)
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_AND, evtMask);
      4c: a3 e6 c5 00  	<unknown>
;     while ((*(volatile char *)done) == 0)
      50: 03 47 41 06  	lbu	a4, 100(sp)
      54: e3 08 07 fe  	beqz	a4, 0x44 <cl_ram_write+0x44>
; }
      58: 83 20 c1 06  	lw	ra, 108(sp)
      5c: 13 01 01 07  	addi	sp, sp, 112
      60: 67 80 00 00  	ret

Disassembly of section .text.load_file_to_ram:

00000000 <load_file_to_ram>:
; size_t load_file_to_ram(const void *dest, const char *filename) {
       0: 13 01 01 f4  	addi	sp, sp, -192
       4: 23 2e 11 0a  	sw	ra, 188(sp)
       8: 23 2c 81 0a  	sw	s0, 184(sp)
       c: 23 2a 91 0a  	sw	s1, 180(sp)
      10: 23 28 21 0b  	sw	s2, 176(sp)
      14: 23 26 31 0b  	sw	s3, 172(sp)
      18: 23 24 41 0b  	sw	s4, 168(sp)
      1c: 23 22 51 0b  	sw	s5, 164(sp)
      20: 23 20 61 0b  	sw	s6, 160(sp)
      24: 23 2e 71 09  	sw	s7, 156(sp)
      28: 23 2c 81 09  	sw	s8, 152(sp)
      2c: 23 2a 91 09  	sw	s9, 148(sp)
      30: 23 28 a1 09  	sw	s10, 144(sp)
      34: 23 26 b1 09  	sw	s11, 140(sp)
      38: 13 89 05 00  	mv	s2, a1
      3c: 13 04 05 00  	mv	s0, a0
;   pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
      40: 37 05 00 00  	lui	a0, 0
      44: 13 05 05 00  	mv	a0, a0
      48: 13 06 00 00  	li	a2, 0
      4c: 97 00 00 00  	auipc	ra, 0
      50: e7 80 00 00  	jalr	ra
;   if (fd == NULL) {
      54: 63 00 05 10  	beqz	a0, 0x154 <load_file_to_ram+0x154>
      58: 93 04 05 00  	mv	s1, a0
      5c: 13 09 00 00  	li	s2, 0
;   size_t size = fd->size;
      60: 03 2b c5 00  	lw	s6, 12(a0)
      64: 37 15 00 00  	lui	a0, 1
      68: 93 0b 05 80  	addi	s7, a0, -2048
      6c: 37 05 00 00  	lui	a0, 0
      70: 93 09 05 00  	mv	s3, a0
      74: 13 0c 80 00  	li	s8, 8
      78: b7 4c 20 00  	lui	s9, 516
      7c: 13 0d 20 00  	li	s10, 2
      80: 93 0d 40 00  	li	s11, 4
      84: 37 05 00 00  	lui	a0, 0
      88: 13 0a 05 00  	mv	s4, a0
      8c: 6f 00 c0 00  	j	0x98 <load_file_to_ram+0x98>
;     offset += load_size;
      90: 33 89 2a 01  	add	s2, s5, s2
;   } while (offset < size);
      94: 63 70 69 09  	bgeu	s2, s6, 0x114 <load_file_to_ram+0x114>
;     remaining_size = size - offset;
      98: 33 05 2b 41  	sub	a0, s6, s2
;     load_size = BUFFER_SIZE < remaining_size ? BUFFER_SIZE : remaining_size;
      9c: b3 5a 75 05  	<unknown>
;     pi_cl_fs_read(fd, buffer, load_size, &req);
      a0: 93 06 01 00  	mv	a3, sp
      a4: 13 85 04 00  	mv	a0, s1
      a8: 93 85 09 00  	mv	a1, s3
      ac: 13 86 0a 00  	mv	a2, s5
      b0: 97 00 00 00  	auipc	ra, 0
      b4: e7 80 00 00  	jalr	ra
;     while ((*(volatile char *)done) == 0)
      b8: 03 45 01 01  	lbu	a0, 16(sp)
      bc: 63 1c 05 00  	bnez	a0, 0xd4 <load_file_to_ram+0xd4>
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_OR, evtMask);
      c0: 23 ec ac 01  	<unknown>
;   value = pulp_read32(base + offset);
      c4: 03 a5 cc 03  	lw	a0, 60(s9)
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_AND, evtMask);
      c8: a3 ed ac 01  	<unknown>
;     while ((*(volatile char *)done) == 0)
      cc: 03 45 01 01  	lbu	a0, 16(sp)
      d0: e3 08 05 fe  	beqz	a0, 0xc0 <load_file_to_ram+0xc0>
;     cl_ram_write((void *)dest + offset, buffer, load_size);
      d4: b3 05 24 01  	add	a1, s0, s2
;     pi_cl_ram_copy(device, pi_ram_addr, addr, size, 0, req);
      d8: 93 07 81 02  	addi	a5, sp, 40
      dc: 13 05 0a 00  	mv	a0, s4
      e0: 13 86 09 00  	mv	a2, s3
      e4: 93 86 0a 00  	mv	a3, s5
      e8: 13 07 00 00  	li	a4, 0
      ec: 97 00 00 00  	auipc	ra, 0
      f0: e7 80 00 00  	jalr	ra
;     while ((*(volatile char *)done) == 0)
      f4: 03 45 41 08  	lbu	a0, 132(sp)
      f8: e3 1c 05 f8  	bnez	a0, 0x90 <load_file_to_ram+0x90>
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_OR, evtMask);
      fc: 23 ec ac 01  	<unknown>
;   value = pulp_read32(base + offset);
     100: 03 a5 cc 03  	lw	a0, 60(s9)
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_AND, evtMask);
     104: a3 ed ac 01  	<unknown>
;     while ((*(volatile char *)done) == 0)
     108: 03 45 41 08  	lbu	a0, 132(sp)
     10c: e3 08 05 fe  	beqz	a0, 0xfc <load_file_to_ram+0xfc>
     110: 6f f0 1f f8  	j	0x90 <load_file_to_ram+0x90>
;   return offset;
     114: 13 05 09 00  	mv	a0, s2
     118: 83 20 c1 0b  	lw	ra, 188(sp)
     11c: 03 24 81 0b  	lw	s0, 184(sp)
     120: 83 24 41 0b  	lw	s1, 180(sp)
     124: 03 29 01 0b  	lw	s2, 176(sp)
     128: 83 29 c1 0a  	lw	s3, 172(sp)
     12c: 03 2a 81 0a  	lw	s4, 168(sp)
     130: 83 2a 41 0a  	lw	s5, 164(sp)
     134: 03 2b 01 0a  	lw	s6, 160(sp)
     138: 83 2b c1 09  	lw	s7, 156(sp)
     13c: 03 2c 81 09  	lw	s8, 152(sp)
     140: 83 2c 41 09  	lw	s9, 148(sp)
     144: 03 2d 01 09  	lw	s10, 144(sp)
     148: 83 2d c1 08  	lw	s11, 140(sp)
     14c: 13 01 01 0c  	addi	sp, sp, 192
     150: 67 80 00 00  	ret
;     printf("ERROR: Cannot open file %s! Exiting...", filename);
     154: 37 05 00 00  	lui	a0, 0
     158: 13 05 05 00  	mv	a0, a0
     15c: 93 05 09 00  	mv	a1, s2
     160: 97 00 00 00  	auipc	ra, 0
     164: e7 80 00 00  	jalr	ra
;     pos_kernel_pmsis_exit_value = err;
     168: 37 05 00 00  	lui	a0, 0
     16c: 93 05 c0 ff  	li	a1, -4
     170: 23 20 b5 00  	sw	a1, 0(a0)
;     exit(err);
     174: 13 05 c0 ff  	li	a0, -4
     178: 97 00 00 00  	auipc	ra, 0
     17c: e7 80 00 00  	jalr	ra

Disassembly of section .text.load_file_to_local:

00000000 <load_file_to_local>:
; size_t load_file_to_local(const void *dest, const char *filename) {
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
      34: 13 89 05 00  	mv	s2, a1
      38: 13 04 05 00  	mv	s0, a0
;   pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
      3c: 37 05 00 00  	lui	a0, 0
      40: 13 05 05 00  	mv	a0, a0
      44: 13 06 00 00  	li	a2, 0
      48: 97 00 00 00  	auipc	ra, 0
      4c: e7 80 00 00  	jalr	ra
;   if (fd == NULL) {
      50: 63 0a 05 0c  	beqz	a0, 0x124 <load_file_to_local+0x124>
      54: 93 04 05 00  	mv	s1, a0
;   const size_t size = fd->size;
      58: 83 2a c5 00  	lw	s5, 12(a0)
;   while (offset < size) {
      5c: 63 84 0a 08  	beqz	s5, 0xe4 <load_file_to_local+0xe4>
      60: 13 09 00 00  	li	s2, 0
      64: 37 15 00 00  	lui	a0, 1
      68: 13 0b 05 80  	addi	s6, a0, -2048
      6c: 37 05 00 00  	lui	a0, 0
      70: 93 09 05 00  	mv	s3, a0
      74: 93 0b 80 00  	li	s7, 8
      78: 37 4c 20 00  	lui	s8, 516
      7c: 93 0c 20 00  	li	s9, 2
      80: 13 0d 40 00  	li	s10, 4
      84: 6f 00 00 02  	j	0xa4 <load_file_to_local+0xa4>
;     memcpy(dest + offset, buffer, load_size);
      88: 33 05 24 01  	add	a0, s0, s2
      8c: 93 85 09 00  	mv	a1, s3
      90: 13 06 0a 00  	mv	a2, s4
      94: 97 00 00 00  	auipc	ra, 0
      98: e7 80 00 00  	jalr	ra
;     offset += load_size;
      9c: 33 09 2a 01  	add	s2, s4, s2
;   while (offset < size) {
      a0: 63 74 59 05  	bgeu	s2, s5, 0xe8 <load_file_to_local+0xe8>
;     remaining_size = size - offset;
      a4: 33 85 2a 41  	sub	a0, s5, s2
;         BUFFER_SIZE < remaining_size ? BUFFER_SIZE : remaining_size;
      a8: 33 5a 65 05  	<unknown>
;     pi_cl_fs_read(fd, buffer, load_size, &req);
      ac: 93 06 81 00  	addi	a3, sp, 8
      b0: 13 85 04 00  	mv	a0, s1
      b4: 93 85 09 00  	mv	a1, s3
      b8: 13 06 0a 00  	mv	a2, s4
      bc: 97 00 00 00  	auipc	ra, 0
      c0: e7 80 00 00  	jalr	ra
;     while ((*(volatile char *)done) == 0)
      c4: 03 45 81 01  	lbu	a0, 24(sp)
      c8: e3 10 05 fc  	bnez	a0, 0x88 <load_file_to_local+0x88>
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_OR, evtMask);
      cc: a3 6b 9c 01  	<unknown>
;   value = pulp_read32(base + offset);
      d0: 03 25 cc 03  	lw	a0, 60(s8)
;   ARCHI_WRITE(ARCHI_EU_DEMUX_ADDR, EU_CORE_MASK_AND, evtMask);
      d4: 23 6d 9c 01  	<unknown>
;     while ((*(volatile char *)done) == 0)
      d8: 03 45 81 01  	lbu	a0, 24(sp)
      dc: e3 08 05 fe  	beqz	a0, 0xcc <load_file_to_local+0xcc>
      e0: 6f f0 9f fa  	j	0x88 <load_file_to_local+0x88>
      e4: 13 09 00 00  	li	s2, 0
;   return offset;
      e8: 13 05 09 00  	mv	a0, s2
      ec: 83 20 c1 05  	lw	ra, 92(sp)
      f0: 03 24 81 05  	lw	s0, 88(sp)
      f4: 83 24 41 05  	lw	s1, 84(sp)
      f8: 03 29 01 05  	lw	s2, 80(sp)
      fc: 83 29 c1 04  	lw	s3, 76(sp)
     100: 03 2a 81 04  	lw	s4, 72(sp)
     104: 83 2a 41 04  	lw	s5, 68(sp)
     108: 03 2b 01 04  	lw	s6, 64(sp)
     10c: 83 2b c1 03  	lw	s7, 60(sp)
     110: 03 2c 81 03  	lw	s8, 56(sp)
     114: 83 2c 41 03  	lw	s9, 52(sp)
     118: 03 2d 01 03  	lw	s10, 48(sp)
     11c: 13 01 01 06  	addi	sp, sp, 96
     120: 67 80 00 00  	ret
;     printf("ERROR: Cannot open file %s! Exiting...", filename);
     124: 37 05 00 00  	lui	a0, 0
     128: 13 05 05 00  	mv	a0, a0
     12c: 93 05 09 00  	mv	a1, s2
     130: 97 00 00 00  	auipc	ra, 0
     134: e7 80 00 00  	jalr	ra
;     pos_kernel_pmsis_exit_value = err;
     138: 37 05 00 00  	lui	a0, 0
     13c: 93 05 c0 ff  	li	a1, -4
     140: 23 20 b5 00  	sw	a1, 0(a0)
;     exit(err);
     144: 13 05 c0 ff  	li	a0, -4
     148: 97 00 00 00  	auipc	ra, 0
     14c: e7 80 00 00  	jalr	ra

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(gemv.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                      Size     VMA      Type
  0                           00000000 00000000 
  1 .strtab                   0000015c 00000000 
  2 .text                     00000000 00000000 TEXT
  3 .text.gemv_s8_s8_plp      00000460 00000000 TEXT
  4 .rela.text.gemv_s8_s8_plp 00000048 00000000 
  5 .text.pulp_nn_bn_quant_i8 0000001c 00000000 TEXT
  6 .text.pulp_nn_quant_i8    00000018 00000000 TEXT
  7 .debug_loclists           000005af 00000000 DEBUG
  8 .debug_abbrev             000001a3 00000000 DEBUG
  9 .debug_info               000003c1 00000000 DEBUG
 10 .rela.debug_info          00000138 00000000 
 11 .debug_rnglists           0000007d 00000000 DEBUG
 12 .debug_str_offsets        00000138 00000000 DEBUG
 13 .rela.debug_str_offsets   00000390 00000000 
 14 .debug_str                0000035d 00000000 DEBUG
 15 .debug_addr               00000040 00000000 DEBUG
 16 .rela.debug_addr          000000a8 00000000 
 17 .comment                  00000073 00000000 
 18 .note.GNU-stack           00000000 00000000 
 19 .riscv.attributes         00000030 00000000 
 20 .debug_frame              00000064 00000000 DEBUG
 21 .rela.debug_frame         000000c0 00000000 
 22 .debug_line               00000611 00000000 DEBUG
 23 .rela.debug_line          00000f54 00000000 
 24 .debug_line_str           00000284 00000000 DEBUG
 25 .llvm_addrsig             00000000 00000000 
 26 .symtab                   00001100 00000000 

Disassembly of section .text.gemv_s8_s8_plp:

00000000 <gemv_s8_s8_plp>:
;                     uint8_t flag_batch_norm) {
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
      38: 23 2e 11 01  	sw	a7, 28(sp)
      3c: 23 20 01 01  	sw	a6, 0(sp)
      40: 93 8a 06 00  	mv	s5, a3
      44: 13 04 06 00  	mv	s0, a2
      48: 93 8b 05 00  	mv	s7, a1
      4c: 13 0b 05 00  	mv	s6, a0
      50: 83 26 41 06  	lw	a3, 100(sp)
      54: 03 25 c1 06  	lw	a0, 108(sp)
      58: 83 25 81 06  	lw	a1, 104(sp)
      5c: 03 28 01 06  	lw	a6, 96(sp)
;   int lft_neurons = num_o_neurons & 0x01;
      60: 13 f6 16 00  	andi	a2, a3, 1
      64: 23 2a c1 00  	sw	a2, 20(sp)
      68: 37 06 01 00  	lui	a2, 16
      6c: 13 06 e6 ff  	addi	a2, a2, -2
      70: 23 2c d1 00  	sw	a3, 24(sp)
;   int stop_even = stop - lft_neurons;
      74: b3 f9 c6 00  	and	s3, a3, a2
      78: 33 5a 08 10  	<unknown>
      7c: b3 76 05 10  	<unknown>
      80: 33 f6 05 10  	<unknown>
;   for (i = start; i < stop_even; i += 2) {
      84: 23 28 f1 00  	sw	a5, 16(sp)
      88: 23 26 e1 00  	sw	a4, 12(sp)
      8c: 23 24 c1 02  	sw	a2, 40(sp)
      90: 23 24 d1 00  	sw	a3, 8(sp)
      94: 23 22 01 01  	sw	a6, 4(sp)
      98: 63 8e 09 12  	beqz	s3, 0x1d4 <gemv_s8_s8_plp+0x1d4>
      9c: 93 74 38 00  	andi	s1, a6, 3
      a0: 13 55 2a 00  	srli	a0, s4, 2
      a4: b3 35 d0 00  	snez	a1, a3
      a8: 33 36 c0 00  	snez	a2, a2
      ac: b3 75 b6 00  	and	a1, a2, a1
      b0: 93 02 10 00  	li	t0, 1
;   for (i = start; i < stop_even; i += 2) {
      b4: 33 75 55 04  	<unknown>
      b8: 23 22 a1 02  	sw	a0, 36(sp)
      bc: 13 15 25 00  	slli	a0, a0, 2
      c0: 33 05 ab 00  	add	a0, s6, a0
;     if (flag_batch_norm && flag_relu) {
      c4: 23 20 a1 02  	sw	a0, 32(sp)
      c8: 13 09 00 00  	li	s2, 0
      cc: 63 88 05 10  	beqz	a1, 0x1dc <gemv_s8_s8_plp+0x1dc>
      d0: 03 25 c1 01  	lw	a0, 28(sp)
      d4: 33 6c 05 10  	<unknown>
      d8: 13 8d 07 00  	mv	s10, a5
      dc: 93 0d 07 00  	mv	s11, a4
      e0: 6f 00 c0 04  	j	0x12c <gemv_s8_s8_plp+0x12c>
;       *pOutBuffer = pulp_nn_bn_quant_i8(sum, *k1, *lambda1, out_shift);
      e4: 83 a5 0d 00  	lw	a1, 0(s11)
      e8: 03 26 0d 00  	lw	a2, 0(s10)
      ec: 93 06 0c 00  	mv	a3, s8
      f0: 97 00 00 00  	auipc	ra, 0
      f4: e7 80 00 00  	jalr	ra
      f8: 23 00 a4 00  	sb	a0, 0(s0)
;           pulp_nn_bn_quant_i8(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
      fc: 83 a5 4d 00  	lw	a1, 4(s11)
     100: 03 26 4d 00  	lw	a2, 4(s10)
     104: 13 85 0c 00  	mv	a0, s9
     108: 93 06 0c 00  	mv	a3, s8
     10c: 97 00 00 00  	auipc	ra, 0
     110: e7 80 00 00  	jalr	ra
;       *pOutBuffer =
     114: a3 00 a4 00  	sb	a0, 1(s0)
;       k1 += 2;
     118: 93 8d 8d 00  	addi	s11, s11, 8
;       lambda1 += 2;
     11c: 13 0d 8d 00  	addi	s10, s10, 8
;   for (i = start; i < stop_even; i += 2) {
     120: 13 09 29 00  	addi	s2, s2, 2
     124: 13 04 24 00  	addi	s0, s0, 2
;   for (i = start; i < stop_even; i += 2) {
     128: 63 70 39 1f  	bgeu	s2, s3, 0x308 <gemv_s8_s8_plp+0x308>
;     if (pBias != NULL) {
     12c: 63 8a 0b 02  	beqz	s7, 0x160 <gemv_s8_s8_plp+0x160>
;       sum = *(int32_t *)(pBias + 4 * i);
     130: 13 15 29 00  	slli	a0, s2, 2
     134: b3 85 ab 00  	add	a1, s7, a0
     138: 03 a5 05 00  	lw	a0, 0(a1)
;       sum2 = *(int32_t *)(pBias + 4 * i + 4);
     13c: 83 ac 45 00  	lw	s9, 4(a1)
;     int8_t *pB = pWeight + (i * dim_vec_wt);
     140: 93 85 0a 00  	mv	a1, s5
     144: b3 05 49 43  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     148: 13 06 40 00  	li	a2, 4
     14c: 63 76 ca 02  	bgeu	s4, a2, 0x178 <gemv_s8_s8_plp+0x178>
     150: 33 86 45 01  	add	a2, a1, s4
     154: 93 06 0b 00  	mv	a3, s6
     158: 63 9a 04 04  	bnez	s1, 0x1ac <gemv_s8_s8_plp+0x1ac>
     15c: 6f f0 9f f8  	j	0xe4 <gemv_s8_s8_plp+0xe4>
     160: 13 05 00 00  	li	a0, 0
     164: 93 0c 00 00  	li	s9, 0
;     int8_t *pB = pWeight + (i * dim_vec_wt);
     168: 93 85 0a 00  	mv	a1, s5
     16c: b3 05 49 43  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     170: 13 06 40 00  	li	a2, 4
     174: e3 6e ca fc  	bltu	s4, a2, 0x150 <gemv_s8_s8_plp+0x150>
     178: 03 26 41 02  	lw	a2, 36(sp)
     17c: 93 06 0b 00  	mv	a3, s6
;       vecA = *((v4s *)pA);
     180: 0b a7 46 00  	<unknown>
;       vecB2 = *((v4s *)pB2);
     184: b3 87 45 01  	add	a5, a1, s4
;       vecB = *((v4s *)pB);
     188: 0b a8 45 00  	<unknown>
;       vecB2 = *((v4s *)pB2);
     18c: 83 a7 07 00  	lw	a5, 0(a5)
;       sum = SumDotps4(vecA, vecB, sum);
     190: 57 15 07 b9  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     194: 13 06 f6 ff  	addi	a2, a2, -1
;       sum2 = SumDotps4(vecA, vecB2, sum2);
     198: d7 1c f7 b8  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     19c: e3 12 06 fe  	bnez	a2, 0x180 <gemv_s8_s8_plp+0x180>
;     while (col_cnt) {
     1a0: 33 86 45 01  	add	a2, a1, s4
     1a4: 83 26 01 02  	lw	a3, 32(sp)
     1a8: e3 8e 04 f2  	beqz	s1, 0xe4 <gemv_s8_s8_plp+0xe4>
     1ac: 13 87 04 00  	mv	a4, s1
;       int8_t inA = *pA;
     1b0: 8b 87 16 00  	<unknown>
;       int8_t inB = *pB;
     1b4: 0b 88 15 00  	<unknown>
;       int8_t inB2 = *pB2;
     1b8: 8b 08 16 00  	<unknown>
;       sum += inA * inB;
     1bc: 33 05 f8 42  	<unknown>
;       col_cnt--;
     1c0: 13 07 f7 ff  	addi	a4, a4, -1
     1c4: 33 58 07 10  	<unknown>
;       sum2 += inA * inB2;
     1c8: b3 8c f8 42  	<unknown>
;     while (col_cnt) {
     1cc: e3 12 08 fe  	bnez	a6, 0x1b0 <gemv_s8_s8_plp+0x1b0>
     1d0: 6f f0 5f f1  	j	0xe4 <gemv_s8_s8_plp+0xe4>
     1d4: 13 09 00 00  	li	s2, 0
     1d8: 6f 00 00 13  	j	0x308 <gemv_s8_s8_plp+0x308>
     1dc: 13 03 40 00  	li	t1, 4
     1e0: 03 25 01 00  	lw	a0, 0(sp)
     1e4: 33 4c 05 10  	<unknown>
     1e8: 03 25 c1 01  	lw	a0, 28(sp)
     1ec: b3 6c 05 10  	<unknown>
     1f0: 93 0d f0 07  	li	s11, 127
     1f4: 6f 00 c0 02  	j	0x220 <gemv_s8_s8_plp+0x220>
     1f8: 83 25 c1 01  	lw	a1, 28(sp)
;         *pOutBuffer = (int8_t)clips8(sum >> out_shift);
     1fc: 33 55 b5 40  	sra	a0, a0, a1
     200: 33 55 b5 15  	<unknown>
     204: 23 00 a4 00  	sb	a0, 0(s0)
;         *pOutBuffer = (int8_t)clips8(sum2 >> out_shift);
     208: 33 55 bd 40  	sra	a0, s10, a1
     20c: 33 55 b5 15  	<unknown>
;         *pOutBuffer = pulp_nn_quant_i8(sum2, out_mult, out_shift);
     210: a3 00 a4 00  	sb	a0, 1(s0)
;   for (i = start; i < stop_even; i += 2) {
     214: 13 09 29 00  	addi	s2, s2, 2
     218: 13 04 24 00  	addi	s0, s0, 2
;   for (i = start; i < stop_even; i += 2) {
     21c: 63 76 39 0f  	bgeu	s2, s3, 0x308 <gemv_s8_s8_plp+0x308>
     220: 63 8a 0b 02  	beqz	s7, 0x254 <gemv_s8_s8_plp+0x254>
;       sum = *(int32_t *)(pBias + 4 * i);
     224: 13 15 29 00  	slli	a0, s2, 2
     228: b3 85 ab 00  	add	a1, s7, a0
     22c: 03 a5 05 00  	lw	a0, 0(a1)
;       sum2 = *(int32_t *)(pBias + 4 * i + 4);
     230: 03 ad 45 00  	lw	s10, 4(a1)
;     int8_t *pB = pWeight + (i * dim_vec_wt);
     234: 93 85 0a 00  	mv	a1, s5
     238: b3 05 49 43  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     23c: 63 76 6a 02  	bgeu	s4, t1, 0x268 <gemv_s8_s8_plp+0x268>
     240: 33 86 45 01  	add	a2, a1, s4
     244: 93 06 0b 00  	mv	a3, s6
     248: 13 87 04 00  	mv	a4, s1
;     while (col_cnt) {
     24c: 63 9a 04 04  	bnez	s1, 0x2a0 <gemv_s8_s8_plp+0x2a0>
     250: 6f 00 00 07  	j	0x2c0 <gemv_s8_s8_plp+0x2c0>
     254: 13 05 00 00  	li	a0, 0
     258: 13 0d 00 00  	li	s10, 0
;     int8_t *pB = pWeight + (i * dim_vec_wt);
     25c: 93 85 0a 00  	mv	a1, s5
     260: b3 05 49 43  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     264: e3 6e 6a fc  	bltu	s4, t1, 0x240 <gemv_s8_s8_plp+0x240>
     268: 03 26 41 02  	lw	a2, 36(sp)
     26c: 93 06 0b 00  	mv	a3, s6
;       vecA = *((v4s *)pA);
     270: 0b a7 46 00  	<unknown>
;       vecB2 = *((v4s *)pB2);
     274: b3 87 45 01  	add	a5, a1, s4
;       vecB = *((v4s *)pB);
     278: 0b a8 45 00  	<unknown>
;       vecB2 = *((v4s *)pB2);
     27c: 83 a7 07 00  	lw	a5, 0(a5)
;       sum = SumDotps4(vecA, vecB, sum);
     280: 57 15 07 b9  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     284: 13 06 f6 ff  	addi	a2, a2, -1
;       sum2 = SumDotps4(vecA, vecB2, sum2);
     288: 57 1d f7 b8  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     28c: e3 12 06 fe  	bnez	a2, 0x270 <gemv_s8_s8_plp+0x270>
;     while (col_cnt) {
     290: 33 86 45 01  	add	a2, a1, s4
     294: 83 26 01 02  	lw	a3, 32(sp)
     298: 13 87 04 00  	mv	a4, s1
;     while (col_cnt) {
     29c: 63 82 04 02  	beqz	s1, 0x2c0 <gemv_s8_s8_plp+0x2c0>
;       int8_t inA = *pA;
     2a0: 8b 87 16 00  	<unknown>
;       int8_t inB = *pB;
     2a4: 0b 88 15 00  	<unknown>
;       int8_t inB2 = *pB2;
     2a8: 8b 08 16 00  	<unknown>
;       sum += inA * inB;
     2ac: 33 05 f8 42  	<unknown>
;       col_cnt--;
     2b0: 13 07 f7 ff  	addi	a4, a4, -1
     2b4: 33 58 07 10  	<unknown>
;       sum2 += inA * inB2;
     2b8: 33 8d f8 42  	<unknown>
;     while (col_cnt) {
     2bc: e3 12 08 fe  	bnez	a6, 0x2a0 <gemv_s8_s8_plp+0x2a0>
;       if (flag_relu == 1) {
     2c0: 83 25 81 02  	lw	a1, 40(sp)
     2c4: e3 9a 55 f2  	bne	a1, t0, 0x1f8 <gemv_s8_s8_plp+0x1f8>
;         *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);
     2c8: 93 05 0c 00  	mv	a1, s8
     2cc: 13 86 0c 00  	mv	a2, s9
     2d0: 97 00 00 00  	auipc	ra, 0
     2d4: e7 80 00 00  	jalr	ra
     2d8: 23 00 a4 00  	sb	a0, 0(s0)
;         *pOutBuffer = pulp_nn_quant_i8(sum2, out_mult, out_shift);
     2dc: 13 05 0d 00  	mv	a0, s10
     2e0: 93 05 0c 00  	mv	a1, s8
     2e4: 13 86 0c 00  	mv	a2, s9
     2e8: 97 00 00 00  	auipc	ra, 0
     2ec: e7 80 00 00  	jalr	ra
     2f0: 13 03 40 00  	li	t1, 4
     2f4: 93 02 10 00  	li	t0, 1
;         *pOutBuffer = pulp_nn_quant_i8(sum2, out_mult, out_shift);
     2f8: a3 00 a4 00  	sb	a0, 1(s0)
;   for (i = start; i < stop_even; i += 2) {
     2fc: 13 09 29 00  	addi	s2, s2, 2
     300: 13 04 24 00  	addi	s0, s0, 2
;   for (i = start; i < stop_even; i += 2) {
     304: e3 6e 39 f1  	bltu	s2, s3, 0x220 <gemv_s8_s8_plp+0x220>
;   if (lft_neurons && (stop - start) > 0) {
     308: 03 25 41 01  	lw	a0, 20(sp)
     30c: 13 35 15 00  	seqz	a0, a0
     310: 83 25 81 01  	lw	a1, 24(sp)
     314: b3 d5 05 10  	<unknown>
     318: 93 b5 15 00  	seqz	a1, a1
     31c: 33 e5 a5 00  	or	a0, a1, a0
     320: 63 1e 05 0e  	bnez	a0, 0x41c <gemv_s8_s8_plp+0x41c>
;     if (pBias != NULL) {
     324: 63 8a 0b 00  	beqz	s7, 0x338 <gemv_s8_s8_plp+0x338>
;       sum = *(int32_t *)(pBias + 4 * i);
     328: 13 15 29 00  	slli	a0, s2, 2
     32c: 33 85 ab 00  	add	a0, s7, a0
     330: 03 25 05 00  	lw	a0, 0(a0)
     334: 6f 00 80 00  	j	0x33c <gemv_s8_s8_plp+0x33c>
     338: 13 05 00 00  	li	a0, 0
     33c: 03 28 01 01  	lw	a6, 16(sp)
     340: 83 28 c1 00  	lw	a7, 12(sp)
     344: 83 23 41 00  	lw	t2, 4(sp)
     348: 93 05 40 00  	li	a1, 4
;     int8_t *pB = pWeight + (i * dim_vec_wt);
     34c: b3 0a 49 43  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     350: 63 7c ba 00  	bgeu	s4, a1, 0x368 <gemv_s8_s8_plp+0x368>
     354: 83 22 81 02  	lw	t0, 40(sp)
     358: 03 23 81 00  	lw	t1, 8(sp)
;     uint16_t col_cnt = dim_vec & 0x3;
     35c: 93 f5 33 00  	andi	a1, t2, 3
;     while (col_cnt) {
     360: 63 90 05 04  	bnez	a1, 0x3a0 <gemv_s8_s8_plp+0x3a0>
     364: 6f 00 40 05  	j	0x3b8 <gemv_s8_s8_plp+0x3b8>
     368: 93 55 2a 00  	srli	a1, s4, 2
     36c: 13 06 10 00  	li	a2, 1
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     370: 33 f6 c5 04  	<unknown>
     374: 93 15 26 00  	slli	a1, a2, 2
     378: 93 06 0b 00  	mv	a3, s6
     37c: 83 22 81 02  	lw	t0, 40(sp)
     380: 03 23 81 00  	lw	t1, 8(sp)
     384: 7b 40 66 00  	<unknown>
;       vecA = *((v4s *)pA);
     388: 0b a7 46 00  	<unknown>
;       vecB = *((v4s *)pB);
     38c: 8b a7 4a 00  	<unknown>
;       sum = SumDotps4(vecA, vecB, sum);
     390: 57 15 f7 b8  	<unknown>
;     for (int j = 0; j < (dim_vec >> 2); j++) {
     394: 33 0b bb 00  	add	s6, s6, a1
;     uint16_t col_cnt = dim_vec & 0x3;
     398: 93 f5 33 00  	andi	a1, t2, 3
;     while (col_cnt) {
     39c: 63 8e 05 00  	beqz	a1, 0x3b8 <gemv_s8_s8_plp+0x3b8>
;       int8_t inA = *pA;
     3a0: 0b 06 1b 00  	<unknown>
;       int8_t inB = *pB;
     3a4: 8b 86 1a 00  	<unknown>
;       col_cnt--;
     3a8: 93 85 f5 ff  	addi	a1, a1, -1
     3ac: 33 d7 05 10  	<unknown>
;       sum += inA * inB;
     3b0: 33 85 c6 42  	<unknown>
;     while (col_cnt) {
     3b4: e3 16 07 fe  	bnez	a4, 0x3a0 <gemv_s8_s8_plp+0x3a0>
;     if (flag_batch_norm && flag_relu) {
     3b8: 93 35 13 00  	seqz	a1, t1
     3bc: 13 b6 12 00  	seqz	a2, t0
;     if (flag_batch_norm && flag_relu) {
     3c0: b3 65 b6 00  	or	a1, a2, a1
     3c4: 63 90 05 02  	bnez	a1, 0x3e4 <gemv_s8_s8_plp+0x3e4>
;       *pOutBuffer = pulp_nn_bn_quant_i8(sum, *pKappa, *pLambda, out_shift);
     3c8: 83 a5 08 00  	lw	a1, 0(a7)
     3cc: 03 26 08 00  	lw	a2, 0(a6)
     3d0: 83 26 c1 01  	lw	a3, 28(sp)
     3d4: b3 e6 06 10  	<unknown>
     3d8: 97 00 00 00  	auipc	ra, 0
     3dc: e7 80 00 00  	jalr	ra
     3e0: 6f 00 80 03  	j	0x418 <gemv_s8_s8_plp+0x418>
     3e4: 93 05 10 00  	li	a1, 1
;       if (flag_relu == 1) {
     3e8: 63 90 b2 02  	bne	t0, a1, 0x408 <gemv_s8_s8_plp+0x408>
;         *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);
     3ec: 83 25 01 00  	lw	a1, 0(sp)
     3f0: b3 c5 05 10  	<unknown>
     3f4: 03 26 c1 01  	lw	a2, 28(sp)
     3f8: 33 66 06 10  	<unknown>
     3fc: 97 00 00 00  	auipc	ra, 0
     400: e7 80 00 00  	jalr	ra
     404: 6f 00 40 01  	j	0x418 <gemv_s8_s8_plp+0x418>
;         *pOutBuffer = (int8_t)clips8(sum >> out_shift);
     408: 83 25 c1 01  	lw	a1, 28(sp)
     40c: 33 55 b5 40  	sra	a0, a0, a1
     410: 93 05 f0 07  	li	a1, 127
;         *pOutBuffer = (int8_t)clips8(sum >> out_shift);
     414: 33 55 b5 14  	<unknown>
     418: 23 00 a4 00  	sb	a0, 0(s0)
;   __asm__ __volatile__ ("" : : : "memory");
     41c: 37 45 20 00  	lui	a0, 516
;   value = pulp_read32(base + offset);
     420: 03 25 c5 21  	lw	a0, 540(a0)
; }
     424: 83 20 c1 05  	lw	ra, 92(sp)
     428: 03 24 81 05  	lw	s0, 88(sp)
     42c: 83 24 41 05  	lw	s1, 84(sp)
     430: 03 29 01 05  	lw	s2, 80(sp)
     434: 83 29 c1 04  	lw	s3, 76(sp)
     438: 03 2a 81 04  	lw	s4, 72(sp)
     43c: 83 2a 41 04  	lw	s5, 68(sp)
     440: 03 2b 01 04  	lw	s6, 64(sp)
     444: 83 2b c1 03  	lw	s7, 60(sp)
     448: 03 2c 81 03  	lw	s8, 56(sp)
     44c: 83 2c 41 03  	lw	s9, 52(sp)
     450: 03 2d 01 03  	lw	s10, 48(sp)
     454: 83 2d c1 02  	lw	s11, 44(sp)
     458: 13 01 01 06  	addi	sp, sp, 96
     45c: 67 80 00 00  	ret

Disassembly of section .text.pulp_nn_bn_quant_i8:

00000000 <pulp_nn_bn_quant_i8>:
;   int32_t integer_image_phi = (k * phi) + lambda;
       0: 33 85 a5 02  	mul	a0, a1, a0
;   int32_t x = (integer_image_phi) >> d;
       4: b3 f5 06 10  	<unknown>
       8: 5b 25 b6 40  	<unknown>
       c: 93 05 f0 07  	li	a1, 127
;   int8_t res = clips8(x);
      10: 33 55 b5 14  	<unknown>
;   return res;
      14: 33 65 05 10  	<unknown>
      18: 67 80 00 00  	ret

Disassembly of section .text.pulp_nn_quant_i8:

00000000 <pulp_nn_quant_i8>:
;   int32_t x = (m * phi) >> d;
       0: 33 85 a5 02  	mul	a0, a1, a0
       4: 33 55 c5 40  	sra	a0, a0, a2
       8: 93 05 f0 07  	li	a1, 127
;   int8_t res = clips8(x);
       c: 33 55 b5 14  	<unknown>
;   return res;
      10: 33 65 05 10  	<unknown>
      14: 67 80 00 00  	ret

/app/Deeploy/DynMultiDeeployTest/TEST_SIRACUSA/build_dyn/lib/libdeeploypulp.a(iRMSnorm.c.obj):	file format elf32-littleriscv

Sections:
Idx Name                     Size     VMA      Type
  0                          00000000 00000000 
  1 .strtab                  0000012e 00000000 
  2 .text                    00000000 00000000 TEXT
  3 .text.iRMSnorm_s8_s8_plp 00000360 00000000 TEXT
  4 .debug_loclists          00000471 00000000 DEBUG
  5 .debug_abbrev            000000c7 00000000 DEBUG
  6 .debug_info              000001ba 00000000 DEBUG
  7 .rela.debug_info         00000090 00000000 
  8 .debug_rnglists          00000057 00000000 DEBUG
  9 .debug_str_offsets       000000b8 00000000 DEBUG
 10 .rela.debug_str_offsets  00000210 00000000 
 11 .debug_str               00000257 00000000 DEBUG
 12 .debug_addr              00000010 00000000 DEBUG
 13 .rela.debug_addr         00000018 00000000 
 14 .comment                 00000073 00000000 
 15 .note.GNU-stack          00000000 00000000 
 16 .riscv.attributes        00000030 00000000 
 17 .debug_frame             0000003c 00000000 DEBUG
 18 .rela.debug_frame        00000060 00000000 
 19 .debug_line              000003d1 00000000 DEBUG
 20 .rela.debug_line         000009f0 00000000 
 21 .debug_line_str          00000140 00000000 DEBUG
 22 .llvm_addrsig            00000000 00000000 
 23 .symtab                  00000a50 00000000 

Disassembly of section .text.iRMSnorm_s8_s8_plp:

00000000 <iRMSnorm_s8_s8_plp>:
;                         int32_t size, int32_t lastDimLength, int32_t log2D, int nb_dedicated_cores) {
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
      28: b3 c6 e6 02  	div	a3, a3, a4
;   for (int i = 0; i < (size / lastDimLength); i++) {
      2c: 63 54 d0 1c  	blez	a3, 0x1f4 <iRMSnorm_s8_s8_plp+0x1f4>
      30: f3 28 40 f1  	csrr	a7, mhartid
      34: 93 f8 f8 01  	andi	a7, a7, 31
      38: b3 e8 08 03  	rem	a7, a7, a6
      3c: b3 12 08 10  	<unknown>
      40: b3 e2 02 10  	<unknown>
      44: b3 52 57 40  	sra	t0, a4, t0
      48: 13 08 f8 ff  	addi	a6, a6, -1
      4c: 33 78 e8 00  	and	a6, a6, a4
      50: 33 38 00 01  	snez	a6, a6
      54: 33 88 02 01  	add	a6, t0, a6
      58: b3 42 08 10  	<unknown>
      5c: 33 88 12 03  	mul	a6, t0, a7
      60: 33 48 e8 04  	<unknown>
;   int16_t chunk_stop = MIN(chunk_start + chunk, lastDimLength + 1);
      64: 33 48 08 10  	<unknown>
      68: b3 08 58 00  	add	a7, a6, t0
      6c: 93 02 17 00  	addi	t0, a4, 1
      70: b3 c8 58 04  	<unknown>
      74: b3 c8 08 10  	<unknown>
;     for (int j = chunk_start; j < chunk_stop; j++) {
      78: 63 54 18 1b  	bge	a6, a7, 0x220 <iRMSnorm_s8_s8_plp+0x220>
      7c: 93 02 00 00  	li	t0, 0
;   for (int i = 0; i < (size / lastDimLength); i++) {
      80: 13 03 f7 ff  	addi	t1, a4, -1
      84: 93 73 77 00  	andi	t2, a4, 7
      88: 13 7e 87 ff  	andi	t3, a4, -8
      8c: 93 0e 35 00  	addi	t4, a0, 3
      90: 13 1f 28 00  	slli	t5, a6, 2
      94: 33 06 e6 01  	add	a2, a2, t5
      98: 13 0f 70 00  	li	t5, 7
      9c: b7 0f 01 00  	lui	t6, 16
      a0: 13 04 00 f8  	li	s0, -128
      a4: 93 04 f0 07  	li	s1, 127
      a8: fb c0 46 0a  	<unknown>
;     for (int j = 0; j < lastDimLength; j++) {
      ac: 63 5a e0 00  	blez	a4, 0xc0 <iRMSnorm_s8_s8_plp+0xc0>
      b0: 63 7c e3 01  	bgeu	t1, t5, 0xc8 <iRMSnorm_s8_s8_plp+0xc8>
      b4: 93 09 00 00  	li	s3, 0
      b8: 13 09 00 00  	li	s2, 0
      bc: 6f 00 40 06  	j	0x120 <iRMSnorm_s8_s8_plp+0x120>
      c0: 13 09 00 00  	li	s2, 0
      c4: 6f 00 40 07  	j	0x138 <iRMSnorm_s8_s8_plp+0x138>
;     for (int j = 0; j < lastDimLength; j++) {
      c8: 13 5a 3e 00  	srli	s4, t3, 3
      cc: 93 09 00 00  	li	s3, 0
      d0: 13 09 00 00  	li	s2, 0
      d4: 7b 40 4a 02  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
      d8: 33 8a 3e 01  	add	s4, t4, s3
      dc: 83 0a da ff  	lb	s5, -3(s4)
      e0: 03 0b ea ff  	lb	s6, -2(s4)
      e4: 83 0b fa ff  	lb	s7, -1(s4)
;       sum += temp * temp;
      e8: 33 89 5a 43  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
      ec: 83 0a 0a 00  	lb	s5, 0(s4)
;       sum += temp * temp;
      f0: 33 09 6b 43  	<unknown>
      f4: 33 89 7b 43  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
      f8: 03 0b 1a 00  	lb	s6, 1(s4)
;       sum += temp * temp;
      fc: 33 89 5a 43  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     100: 83 0a 2a 00  	lb	s5, 2(s4)
     104: 83 0b 3a 00  	lb	s7, 3(s4)
;       sum += temp * temp;
     108: 33 09 6b 43  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     10c: 03 0a 4a 00  	lb	s4, 4(s4)
;       sum += temp * temp;
     110: 33 89 5a 43  	<unknown>
     114: 33 89 7b 43  	<unknown>
;     for (int j = 0; j < lastDimLength; j++) {
     118: 93 89 89 00  	addi	s3, s3, 8
;       sum += temp * temp;
     11c: 33 09 4a 43  	<unknown>
;     for (int j = 0; j < lastDimLength; j++) {
     120: 63 8c 03 00  	beqz	t2, 0x138 <iRMSnorm_s8_s8_plp+0x138>
     124: b3 09 35 01  	add	s3, a0, s3
     128: 13 8a 03 00  	mv	s4, t2
     12c: 7b c0 43 00  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     130: 8b 8a 19 00  	<unknown>
;       sum += temp * temp;
     134: 33 89 5a 43  	<unknown>
     138: 93 0a 00 00  	li	s5, 0
     13c: 93 09 00 00  	li	s3, 0
;     sum = sum / lastDimLength;
     140: 33 49 e9 02  	div	s2, s2, a4
     144: 13 19 09 01  	slli	s2, s2, 16
     148: 5b 29 f9 21  	<unknown>
     14c: 13 0a f0 0f  	li	s4, 255
     150: 6f 00 40 01  	j	0x164 <iRMSnorm_s8_s8_plp+0x164>
     154: b3 c9 09 10  	<unknown>
     158: 33 4a 0a 10  	<unknown>
     15c: 93 0a 0b 00  	mv	s5, s6
;   while (start <= end) {
     160: 63 42 3a 05  	blt	s4, s3, 0x1a4 <iRMSnorm_s8_s8_plp+0x1a4>
     164: 33 4b 0a 10  	<unknown>
     168: 33 cc 09 10  	<unknown>
;     if (((mid * mid)) == number) {
     16c: db 2b 6c 03  	<unknown>
     170: b3 8b 7b 03  	mul	s7, s7, s7
;     mid = (start + end) >> 1;
     174: 5b 2b 6c 83  	<unknown>
;     if (((mid * mid)) == number) {
     178: 63 86 2b 03  	beq	s7, s2, 0x1a4 <iRMSnorm_s8_s8_plp+0x1a4>
;     if (((mid * mid)) < number) {
     17c: 63 c8 2b 01  	blt	s7, s2, 0x18c <iRMSnorm_s8_s8_plp+0x18c>
     180: 63 da 2b 01  	bge	s7, s2, 0x194 <iRMSnorm_s8_s8_plp+0x194>
     184: e3 c8 2b fd  	blt	s7, s2, 0x154 <iRMSnorm_s8_s8_plp+0x154>
     188: 6f 00 40 01  	j	0x19c <iRMSnorm_s8_s8_plp+0x19c>
     18c: 93 09 1b 00  	addi	s3, s6, 1
;     if (((mid * mid)) < number) {
     190: e3 ca 2b ff  	blt	s7, s2, 0x184 <iRMSnorm_s8_s8_plp+0x184>
     194: 13 0a fb ff  	addi	s4, s6, -1
;     if (((mid * mid)) < number) {
     198: e3 ce 2b fb  	blt	s7, s2, 0x154 <iRMSnorm_s8_s8_plp+0x154>
     19c: 13 8b 0a 00  	mv	s6, s5
     1a0: 6f f0 5f fb  	j	0x154 <iRMSnorm_s8_s8_plp+0x154>
     1a4: b3 8a 08 41  	sub	s5, a7, a6
;     std = _plp_sqrt_q16((int16_t)sum);
     1a8: 33 49 0b 10  	<unknown>
     1ac: 93 09 06 00  	mv	s3, a2
     1b0: 13 0a 08 00  	mv	s4, a6
     1b4: 7b c0 6a 01  	<unknown>
;           (((data_in[j + i * lastDimLength] * weight[j]) / (std)) >> log2D);
     1b8: b3 0a 45 01  	add	s5, a0, s4
     1bc: 83 8a 0a 00  	lb	s5, 0(s5)
     1c0: 0b ab 49 00  	<unknown>
     1c4: b3 0a 5b 03  	mul	s5, s6, s5
     1c8: b3 ca 2a 03  	div	s5, s5, s2
     1cc: b3 da fa 40  	sra	s5, s5, a5
;       data_out[j + i * lastDimLength] = CLAMP(intermediate, -128, 127);
     1d0: b3 ea 8a 04  	<unknown>
     1d4: b3 ca 9a 04  	<unknown>
     1d8: 33 8b 45 01  	add	s6, a1, s4
;     for (int j = chunk_start; j < chunk_stop; j++) {
     1dc: 13 0a 1a 00  	addi	s4, s4, 1
;       data_out[j + i * lastDimLength] = CLAMP(intermediate, -128, 127);
     1e0: 23 00 5b 01  	sb	s5, 0(s6)
;   for (int i = 0; i < (size / lastDimLength); i++) {
     1e4: 93 82 12 00  	addi	t0, t0, 1
     1e8: b3 8e ee 00  	add	t4, t4, a4
     1ec: 33 05 e5 00  	add	a0, a0, a4
     1f0: b3 85 e5 00  	add	a1, a1, a4
; }
     1f4: 03 24 c1 02  	lw	s0, 44(sp)
     1f8: 83 24 81 02  	lw	s1, 40(sp)
     1fc: 03 29 41 02  	lw	s2, 36(sp)
     200: 83 29 01 02  	lw	s3, 32(sp)
     204: 03 2a c1 01  	lw	s4, 28(sp)
     208: 83 2a 81 01  	lw	s5, 24(sp)
     20c: 03 2b 41 01  	lw	s6, 20(sp)
     210: 83 2b 01 01  	lw	s7, 16(sp)
     214: 03 2c c1 00  	lw	s8, 12(sp)
     218: 13 01 01 03  	addi	sp, sp, 48
     21c: 67 80 00 00  	ret
;     for (int j = 0; j < lastDimLength; j++) {
     220: 63 50 e0 02  	blez	a4, 0x240 <iRMSnorm_s8_s8_plp+0x240>
;   for (int i = 0; i < (size / lastDimLength); i++) {
     224: 13 06 f7 ff  	addi	a2, a4, -1
     228: 93 06 70 00  	li	a3, 7
;     for (int j = 0; j < lastDimLength; j++) {
     22c: 93 75 77 00  	andi	a1, a4, 7
     230: 63 70 d6 06  	bgeu	a2, a3, 0x290 <iRMSnorm_s8_s8_plp+0x290>
     234: 93 06 00 00  	li	a3, 0
     238: 13 06 00 00  	li	a2, 0
     23c: 6f 00 40 0b  	j	0x2f0 <iRMSnorm_s8_s8_plp+0x2f0>
     240: 93 05 00 00  	li	a1, 0
     244: 13 06 f0 0f  	li	a2, 255
     248: 13 05 10 00  	li	a0, 1
     24c: 6f 00 00 01  	j	0x25c <iRMSnorm_s8_s8_plp+0x25c>
     250: b3 c5 05 10  	<unknown>
;     if (((mid * mid)) < number) {
     254: 33 46 06 10  	<unknown>
;   while (start <= end) {
     258: e3 4e b6 f8  	blt	a2, a1, 0x1f4 <iRMSnorm_s8_s8_plp+0x1f4>
     25c: 33 47 06 10  	<unknown>
     260: b3 c7 05 10  	<unknown>
;     if (((mid * mid)) == number) {
     264: db a6 e7 02  	<unknown>
     268: b3 86 d6 02  	mul	a3, a3, a3
     26c: e3 84 a6 f8  	beq	a3, a0, 0x1f4 <iRMSnorm_s8_s8_plp+0x1f4>
     270: 5b a7 e7 82  	<unknown>
     274: 63 86 06 00  	beqz	a3, 0x280 <iRMSnorm_s8_s8_plp+0x280>
     278: e3 8c 06 fc  	beqz	a3, 0x250 <iRMSnorm_s8_s8_plp+0x250>
     27c: 6f 00 c0 00  	j	0x288 <iRMSnorm_s8_s8_plp+0x288>
     280: 93 05 17 00  	addi	a1, a4, 1
     284: e3 86 06 fc  	beqz	a3, 0x250 <iRMSnorm_s8_s8_plp+0x250>
     288: 13 06 f7 ff  	addi	a2, a4, -1
     28c: 6f f0 5f fc  	j	0x250 <iRMSnorm_s8_s8_plp+0x250>
     290: 93 06 00 00  	li	a3, 0
     294: 13 06 00 00  	li	a2, 0
;     for (int j = 0; j < lastDimLength; j++) {
     298: 93 77 87 ff  	andi	a5, a4, -8
     29c: 93 d8 37 00  	srli	a7, a5, 3
     2a0: 13 08 35 00  	addi	a6, a0, 3
     2a4: 7b c0 48 02  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     2a8: b3 08 d8 00  	add	a7, a6, a3
     2ac: 83 82 d8 ff  	lb	t0, -3(a7)
     2b0: 03 83 e8 ff  	lb	t1, -2(a7)
     2b4: 83 83 f8 ff  	lb	t2, -1(a7)
;       sum += temp * temp;
     2b8: 33 86 52 42  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     2bc: 83 82 08 00  	lb	t0, 0(a7)
;       sum += temp * temp;
     2c0: 33 06 63 42  	<unknown>
     2c4: 33 86 73 42  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     2c8: 03 83 18 00  	lb	t1, 1(a7)
;       sum += temp * temp;
     2cc: 33 86 52 42  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     2d0: 83 82 28 00  	lb	t0, 2(a7)
     2d4: 83 83 38 00  	lb	t2, 3(a7)
;       sum += temp * temp;
     2d8: 33 06 63 42  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     2dc: 83 88 48 00  	lb	a7, 4(a7)
;       sum += temp * temp;
     2e0: 33 86 52 42  	<unknown>
     2e4: 33 86 73 42  	<unknown>
;     for (int j = 0; j < lastDimLength; j++) {
     2e8: 93 86 86 00  	addi	a3, a3, 8
;       sum += temp * temp;
     2ec: 33 86 18 43  	<unknown>
;     for (int j = 0; j < lastDimLength; j++) {
     2f0: 63 8a 05 00  	beqz	a1, 0x304 <iRMSnorm_s8_s8_plp+0x304>
     2f4: 33 05 d5 00  	add	a0, a0, a3
     2f8: 7b c0 45 00  	<unknown>
;       temp = (data_in[j + i * lastDimLength]);
     2fc: 8b 06 15 00  	<unknown>
;       sum += temp * temp;
     300: 33 86 d6 42  	<unknown>
     304: 13 05 00 00  	li	a0, 0
;     sum = sum / lastDimLength;
     308: b3 45 e6 02  	div	a1, a2, a4
     30c: 93 95 05 01  	slli	a1, a1, 16
     310: 37 06 01 00  	lui	a2, 16
     314: db a5 c5 20  	<unknown>
     318: 13 06 f0 0f  	li	a2, 255
     31c: 6f 00 00 01  	j	0x32c <iRMSnorm_s8_s8_plp+0x32c>
     320: 33 45 05 10  	<unknown>
;     if (((mid * mid)) < number) {
     324: 33 46 06 10  	<unknown>
;   while (start <= end) {
     328: e3 46 a6 ec  	blt	a2, a0, 0x1f4 <iRMSnorm_s8_s8_plp+0x1f4>
     32c: 33 47 06 10  	<unknown>
     330: b3 47 05 10  	<unknown>
;     if (((mid * mid)) == number) {
     334: db a6 e7 02  	<unknown>
     338: b3 86 d6 02  	mul	a3, a3, a3
     33c: e3 8c b6 ea  	beq	a3, a1, 0x1f4 <iRMSnorm_s8_s8_plp+0x1f4>
     340: 5b a7 e7 82  	<unknown>
;     if (((mid * mid)) < number) {
     344: 63 c6 b6 00  	blt	a3, a1, 0x350 <iRMSnorm_s8_s8_plp+0x350>
     348: e3 cc b6 fc  	blt	a3, a1, 0x320 <iRMSnorm_s8_s8_plp+0x320>
     34c: 6f 00 c0 00  	j	0x358 <iRMSnorm_s8_s8_plp+0x358>
     350: 13 05 17 00  	addi	a0, a4, 1
;     if (((mid * mid)) < number) {
     354: e3 c6 b6 fc  	blt	a3, a1, 0x320 <iRMSnorm_s8_s8_plp+0x320>
     358: 13 06 f7 ff  	addi	a2, a4, -1
     35c: 6f f0 5f fc  	j	0x320 <iRMSnorm_s8_s8_plp+0x320>
