#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Network.h"

int8_t *EEGFormer_MEMORYARENA_L1; // L2_init_template
int8_t *EEGFormer_MEMORYARENA_L2; // L2_init_template
uint8_t *EEGFormer_input_0;       // L2_init_template
int8_t *EEGFormer_output_0;       // L2_init_template

static PI_L2 int8_t EEGFormer_onnxMatMul_86_tensor[256] = {
    -21, 9,   -4,  -5,  -23, -8,  -8,  -20, 6,   -15, -18, 12,  6,   -28, -28, 10,  -17, -15, 10,  -5,  32,  25,  -15, 9,   -15, 12,  1,   29,  -6,
    15,  1,   29,  10,  30,  -11, 27,  -20, 18,  6,   -22, 32,  -9,  16,  -25, 16,  27,  -21, 16,  -12, -29, -7,  -18, -28, 6,   -14, -5,  -3,  -4,
    -9,  -11, 25,  23,  -8,  -20, 27,  21,  -26, 32,  13,  -5,  9,   11,  -5,  -15, -28, 14,  30,  24,  5,   -6,  -11, -4,  -1,  17,  6,   -19, -19,
    -5,  -19, 19,  -5,  -13, 28,  29,  8,   29,  24,  -10, -28, 16,  -6,  -11, 22,  17,  3,   -17, 27,  2,   -5,  -14, -21, 28,  13,  -23, -27, 17,
    11,  -14, -28, 2,   -8,  12,  -14, -25, -15, -31, -24, -26, -1,  -1,  24,  -31, -2,  -14, -15, 29,  22,  -10, 8,   -20, -18, -18, -6,  -15, 2,
    -25, -26, -14, 17,  23,  31,  5,   -26, -1,  12,  8,   10,  -26, 9,   -17, -30, -21, 28,  -16, -3,  16,  -17, 23,  7,   25,  -17, 6,   16,  2,
    -30, 25,  16,  26,  19,  -4,  26,  26,  6,   -26, -4,  -2,  20,  20,  7,   24,  -13, -5,  -4,  20,  8,   -24, 7,   10,  25,  -16, 15,  -5,  17,
    -14, -10, 31,  -17, -30, 23,  -25, 9,   4,   -1,  -21, -5,  -22, -9,  11,  -2,  -2,  -21, -22, -29, -15, 4,   -7,  27,  -22, -30, 0,   27,  -11,
    -2,  -5,  -10, -22, 9,   -5,  31,  20,  -10, 14,  -2,  27,  20,  31,  25,  24,  28,  28,  22,  -14, 23,  -8,  -13, -19};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_21[16] = {
    -1770601.0, -1250433.0, -1248377.0, -1246835.0, -1430333.0, 1653924.0, 1191067.0, -331915.0,
    -1551380.0, 1969777.0,  -1567571.0, -850027.0,  1280246.0,  532633.0,  1552666.0, -522095.0};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_21[16] = {
    257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0};

static PI_L2 int8_t EEGFormer_onnxMatMul_87_tensor[256] = {
    23,  -28, 4,   -25, -7,  -13, 17,  -25, -31, -8,  7,   -25, -2,  -19, 4,   -14, 5,   26,  31,  20,  12,  -18, -12, 12,  10, -25, -30, 8,   -23,
    9,   -25, 1,   -2,  13,  -27, -6,  22,  10,  30,  28,  -21, -5,  -5,  22,  8,   1,   -3,  -9,  -28, 22,  24,  12,  20,  25, -25, 31,  -31, -20,
    -18, 16,  -23, -26, -27, 20,  -8,  -4,  12,  -29, 20,  -6,  -23, 19,  5,   13,  -13, 31,  -21, 13,  1,   -4,  -2,  -24, 0,  11,  7,   10,  24,
    -20, 21,  5,   12,  8,   6,   -9,  -28, -11, 6,   -23, 8,   9,   -8,  -28, -5,  -18, -12, 26,  -7,  18,  -8,  -9,  -9,  -7, 25,  -18, 7,   15,
    12,  2,   -27, -21, -3,  -6,  25,  23,  -18, 4,   14,  21,  -24, -24, 22,  -12, -26, -12, 14,  22,  -6,  -12, 30,  13,  30, 17,  12,  19,  11,
    4,   16,  6,   0,   -2,  -1,  16,  -4,  -30, 8,   32,  -8,  23,  -29, 31,  -15, -21, -26, -25, -20, -18, 14,  -6,  -21, 24, 24,  -20, -23, 14,
    20,  -19, -29, 0,   28,  -31, -21, 4,   -2,  -23, 8,   -5,  2,   16,  26,  2,   28,  16,  -10, -29, 18,  5,   18,  5,   -1, -8,  15,  -4,  -11,
    -11, 18,  -8,  -1,  26,  4,   8,   8,   32,  10,  22,  11,  2,   30,  16,  -16, -7,  -30, 2,   1,   3,   -24, 13,  -7,  15, -12, -18, 8,   10,
    19,  31,  23,  10,  5,   -7,  -1,  18,  -14, -27, -6,  -11, -8,  -8,  -20, 17,  19,  31,  -1,  -21, 15,  26,  2,   3};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_24[16] = {
    -1700697.0, 1076702.0, 1072076.0, 613588.0,  1739762.0, -966191.0, 743630.0,   1970805.0,
    -353246.0,  -454247.0, -74401.0,  1703011.0, -849513.0, 797857.0,  -1035067.0, 1936624.0};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_24[16] = {
    257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0};

static PI_L2 int8_t EEGFormer_onnxMatMul_85_tensor[256] = {
    -7,  28,  -12, 17,  -13, -11, 20,  15,  -10, 28,  -3,  24,  32,  -12, -20, -28, -17, 27,  14,  -6,  31,  -19, 17,  -29, -11, 9,   15,  0,   2,
    5,   6,   -18, 20,  7,   7,   -28, -30, -20, -13, -1,  12,  -31, 23,  13,  29,  -11, 0,   -14, -31, 10,  20,  15,  -23, 9,   -29, -26, 19,  24,
    9,   16,  14,  -26, -22, 29,  24,  -23, 13,  -5,  20,  -28, -30, -30, 9,   8,   9,   15,  -9,  7,   6,   -20, 15,  2,   -14, 22,  -21, 25,  24,
    31,  20,  1,   -21, -30, -27, 32,  -30, -1,  -21, -11, -18, -31, 20,  28,  2,   -1,  -5,  -27, 23,  12,  11,  28,  -24, -8,  5,   -12, 7,   -9,
    -25, -4,  3,   14,  -30, -18, -26, 10,  13,  -20, 8,   -4,  -19, 9,   31,  -18, 3,   -29, 28,  -30, -28, 32,  -27, 32,  32,  -20, -30, -1,  30,
    31,  10,  -20, -6,  9,   15,  -21, 17,  -24, -9,  -19, -16, 9,   13,  5,   -9,  14,  6,   -25, 20,  31,  13,  31,  18,  -15, 20,  -12, -10, -24,
    16,  -16, -31, -24, 15,  0,   1,   -13, 6,   5,   -29, 32,  -1,  -1,  14,  -22, 3,   26,  -19, 13,  8,   6,   -8,  -11, -16, -1,  -29, -14, 26,
    7,   5,   1,   -1,  1,   -1,  -25, -4,  -7,  -24, -3,  29,  -14, -16, 32,  -6,  -11, -3,  8,   30,  -21, -21, -17, -23, -28, -2,  22,  -10, -25,
    -25, -30, 28,  16,  -26, 31,  -19, -15, -18, 14,  -13, 21,  26,  -12, 29,  28,  20,  -2,  -23, 30,  23,  29,  -23, 1};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_18[16] = {
    -676038.0, 1661634.0,  314697.0,   1670886.0,  -777039.0,  -997288.0, -253787.0, -223204.0,
    1727940.0, -1872630.0, -1980570.0, -1033525.0, -1367625.0, 1153802.0, 1300035.0, -1069248.0};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_18[16] = {
    257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0, 257.0};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_41[66] = {
    32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0,
    32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0,
    32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0,
    32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0, 32768.0};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_41[66] = {
    514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0,
    514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0,
    514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_43[2] = {32768.0, 32768.0};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_43[2] = {514.0, 514.0};

static PI_L2 int8_t EEGFormer_onnxMatMul_99_tensor[256] = {
    3,   22,  24,  13,  14,  8,   -15, -21, -24, -25, -15, -25, 30,  -26, -31, -23, -27, 14,  -18, -8,  19,  31,  21,  28,  17,  21,  0,   -13, -16,
    21,  32,  -25, 0,   -9,  15,  14,  23,  8,   32,  14,  -16, 8,   16,  -31, -4,  -4,  -21, -19, -15, 22,  25,  -26, 4,   1,   -4,  8,   -18, 24,
    -22, -17, 15,  -5,  3,   -30, 17,  3,   -29, 32,  -10, -27, -10, 19,  -2,  -10, -20, -4,  -21, -31, 2,   -12, -22, -3,  1,   -31, -13, -21, 31,
    -4,  9,   12,  32,  2,   24,  20,  -26, -21, 11,  -23, 9,   30,  -6,  7,   31,  -17, -24, -13, 30,  -7,  1,   14,  26,  -29, 15,  -26, 24,  -22,
    -5,  8,   28,  -17, 5,   14,  24,  5,   -30, 2,   12,  16,  -21, -21, -20, 18,  7,   27,  11,  -12, 7,   25,  -17, -16, -24, -13, 4,   -20, 22,
    -10, 30,  20,  -15, -21, 3,   -23, -8,  1,   18,  -9,  -25, -11, -15, -4,  6,   -20, -8,  -20, 24,  16,  21,  -11, 17,  -28, 4,   25,  -18, 22,
    -9,  -20, -19, 10,  -10, -1,  7,   -7,  17,  28,  -2,  -29, -31, -19, -20, -8,  -15, 5,   9,   -21, -5,  -21, -30, 0,   -24, -3,  21,  30,  24,
    23,  26,  -13, -4,  21,  -24, -17, 13,  26,  -16, 20,  30,  -4,  6,   6,   -17, 7,   -10, -17, -1,  -20, -28, 4,   13,  -25, 21,  -17, -17, 10,
    3,   -15, -13, 20,  -2,  20,  -22, -17, 30,  -24, -4,  3,   -24, 15,  6,   -8,  7,   -29, 1,   11,  5,   31,  -7,  15};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_53[16] = {
    1080814.0, 996004.0,  -1915292.0, -908880.0, 627466.0,  1303376.0, -928412.0, -1868518.0,
    1395382.0, 1368654.0, -953084.0,  1259686.0, 1883168.0, -875470.0, 282572.0,  -746970.0};

static PI_L2 int32_t EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_53[16] = {
    514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0, 514.0};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_numTiles[2] = {0, 1};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_numTiles[2] = {0, 1};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_numTiles[2] = {0, 1};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1_Transpose_31_numTiles[2] = {0, 1};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_numTiles[2] = {0, 1};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_numTiles[2] = {0, 1};

static PI_L1 uint16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O[4] = {64, 64, 2, 2};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch[4] = {6, 2, 6, 2};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_numTiles[2] = {0, 4};

static PI_L1 uint32_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_cmd[4] = {1442584, 1442056, 1442584, 1442056};

static PI_L1 int16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_relativeOffset[4] = {792, -792, 792, 0};

static PI_L1 uint32_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_cmd[4] = {1966848, 1966336, 1966104, 1966088};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_size_1d[4] = {128, 128, 4, 4};

static PI_L1 int16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_relativeOffset[4] = {792, -664, 792, 0};

static PI_L1 uint32_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_cmd[4] = {1442048, 1442048, 1441800, 1441800};

static PI_L1 uint16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_relativeOffset[4] = {0, 256, 0, 0};

static PI_L1 uint32_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_cmd[4] = {1442048, 1442048, 1441800, 1441800};

static PI_L1 uint16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_relativeOffset[4] = {0, 256, 0, 0};

static PI_L1 uint32_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_cmd[4] = {1860352, 1843456, 1835800, 1835272};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_size_1d[4] = {64, 64, 2, 2};

static PI_L1 int16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_relativeOffset[4] = {26136, -26072, 26136, 0};

static PI_L1 uint16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M[32] = {60, 60, 60, 60, 60, 60, 60, 60, 6, 6, 6, 6, 6, 6, 6, 6,
                                                                                        60, 60, 60, 60, 60, 60, 60, 60, 6, 6, 6, 6, 6, 6, 6, 6};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_numTiles[2] = {0, 32};

static PI_L1 uint32_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_cmd[32] = {
    1970040, 1970040, 1970040, 1970040, 1970040, 1970040, 1970040, 1970040, 1966476, 1966476, 1966476, 1966476, 1966476, 1966476, 1966476, 1966476,
    1970040, 1970040, 1970040, 1970040, 1970040, 1970040, 1970040, 1970040, 1966476, 1966476, 1966476, 1966476, 1966476, 1966476, 1966476, 1966476};

static PI_L1 uint16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_size_1d[32] = {
    3960, 3960, 3960, 3960, 3960, 3960, 3960, 3960, 396, 396, 396, 396, 396, 396, 396, 396,
    3960, 3960, 3960, 3960, 3960, 3960, 3960, 3960, 396, 396, 396, 396, 396, 396, 396, 396};

static PI_L1 int32_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_relativeOffset[32] = {
    4356, 4356, 4356, 4356, 4356, 4356, 4356, -26532, 4356, 4356, 4356, 4356, 4356, 4356, 4356, -34452,
    4356, 4356, 4356, 4356, 4356, 4356, 4356, -26532, 4356, 4356, 4356, 4356, 4356, 4356, 4356, 0};

static PI_L1 int16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_relativeOffset[32] = {132, 132,  132, 132, 132,  132, 132, -924, 132, 132, 132,
                                                                                                      132, 132,  132, 132, -858, 132, 132, 132,  132, 132, 132,
                                                                                                      132, -924, 132, 132, 132,  132, 132, 132,  132, 0};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_relativeOffset[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
                                                                                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_relativeOffset[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
                                                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_stride_0[32] = {60, 60, 60, 60, 60, 60, 60, 60, 6, 6, 6, 6, 6, 6, 6, 6,
                                                                                                       60, 60, 60, 60, 60, 60, 60, 60, 6, 6, 6, 6, 6, 6, 6, 6};

static PI_L1 uint32_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_cmd[32] = {
    1835068, 1835068, 1835068, 1835068, 1835068, 1835068, 1835068, 1835068, 1835014, 1835014, 1835014, 1835014, 1835014, 1835014, 1835014, 1835014,
    1835068, 1835068, 1835068, 1835068, 1835068, 1835068, 1835068, 1835068, 1835014, 1835014, 1835014, 1835014, 1835014, 1835014, 1835014, 1835014};

static PI_L1 int16_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_relativeOffset[32] = {
    132, 132, 132, 132, 132, 132, 132, -804, 132, 132, 132, 132, 132, 132, 132, -1043,
    132, 132, 132, 132, 132, 132, 132, -804, 132, 132, 132, 132, 132, 132, 132, 0};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1_Transpose_44_numTiles[2] = {0, 1};

static PI_L1 uint8_t EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_numTiles[2] = {0, 1};

void *EEGFormer_inputs[1];
void *EEGFormer_outputs[1];
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 66; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_u8_i8_i8(
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref, NULL,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref,
          EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref, 1, 16, 16, 16, 1, 1,
          NUM_CORES);
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref += 16;
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref += 16;
    }
    ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref += 16 * 16;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args_t){
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxReshape_28_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args;
  int8_t *EEGFormer_onnxReshape_28_tensor = args->EEGFormer_onnxReshape_28_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref = (uint8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2112);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2432);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2368);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 1056);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_input_0_ref = (void *)((char *)EEGFormer_input_0 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_onnxMatMul_86_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_86_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_21_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_21 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_21_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_21 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_onnxReshape_28_tensor_ref = (void *)((char *)EEGFormer_onnxReshape_28_tensor + 0);

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_prefix[] = "[_MERGE_GEMM_MATMUL_RQ_PASS_0_L2][SB][40128 ops][Tile ";

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_kernel_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442848, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_input_0_ref);
    mchan_transfer_1d(1442048, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_onnxMatMul_86_tensor_ref);
    mchan_transfer_1d(
        1441856, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_21_ref);
    mchan_transfer_1d(
        1441856, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_21_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args =
        (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args_t){
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_A_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_B_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_mul_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_C_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork,
                    &EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_cluster_fork_args);

    _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311776, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_data_out_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_onnxReshape_28_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr > 0)
                ? EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxReshape_28_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args;
  int8_t *EEGFormer_onnxReshape_28_tensor = args->EEGFormer_onnxReshape_28_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args_t){.EEGFormer_onnxReshape_28_tensor = EEGFormer_onnxReshape_28_tensor,
                                                              .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr =
                                                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 66; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_u8_i8_i8(
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref, NULL,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref,
          EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref, 1, 16, 16, 16, 1, 1,
          NUM_CORES);
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref += 16;
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref += 16;
    }
    ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref += 16 * 16;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args_t){
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxReshape_32_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args;
  int8_t *EEGFormer_onnxReshape_32_tensor = args->EEGFormer_onnxReshape_32_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref = (uint8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2112);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2432);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2368);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 1056);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_input_0_ref = (void *)((char *)EEGFormer_input_0 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_onnxMatMul_87_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_87_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_24_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_24 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_24_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_24 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_onnxReshape_32_tensor_ref = (void *)((char *)EEGFormer_onnxReshape_32_tensor + 0);

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_suffix[] = " cycles \n";

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_prefix[] = "[_MERGE_GEMM_MATMUL_RQ_PASS_1_L2][SB][40128 ops][Tile ";

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_kernel_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442848, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_input_0_ref);
    mchan_transfer_1d(1442048, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_onnxMatMul_87_tensor_ref);
    mchan_transfer_1d(
        1441856, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_24_ref);
    mchan_transfer_1d(
        1441856, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_24_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_kernel_start_measurements[TILING_I] = getCycles();

    EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args =
        (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args_t){
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_A_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_B_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_mul_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_C_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork,
                    &EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_cluster_fork_args);

    _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311776, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_data_out_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_onnxReshape_32_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr > 0)
                ? EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_kernel_end_measurements[PROFILING_I] - _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_1_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxReshape_32_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args;
  int8_t *EEGFormer_onnxReshape_32_tensor = args->EEGFormer_onnxReshape_32_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args_t){.EEGFormer_onnxReshape_32_tensor = EEGFormer_onnxReshape_32_tensor,
                                                              .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr =
                                                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 66; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_u8_i8_i8(
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref, NULL,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref,
          EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref, 1, 16, 16, 16, 1, 1,
          NUM_CORES);
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref += 16;
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref += 16;
    }
    ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref += 16 * 16;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args_t){
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxReshape_24_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args;
  int8_t *EEGFormer_onnxReshape_24_tensor = args->EEGFormer_onnxReshape_24_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref = (uint8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2112);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2432);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2368);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 1056);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_input_0_ref = (void *)((char *)EEGFormer_input_0 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_onnxMatMul_85_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_85_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_18_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_18 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_18_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_18 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_onnxReshape_24_tensor_ref = (void *)((char *)EEGFormer_onnxReshape_24_tensor + 0);

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_suffix[] = " cycles \n";

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_prefix[] = "[_MERGE_GEMM_MATMUL_RQ_PASS_2_L2][SB][40128 ops][Tile ";

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_kernel_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442848, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_input_0_ref);
    mchan_transfer_1d(1442048, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_onnxMatMul_85_tensor_ref);
    mchan_transfer_1d(
        1441856, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_18_ref);
    mchan_transfer_1d(
        1441856, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_18_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_kernel_start_measurements[TILING_I] = getCycles();

    EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args =
        (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args_t){
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_A_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_B_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_mul_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_C_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork,
                    &EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_cluster_fork_args);

    _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311776, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_data_out_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_onnxReshape_24_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr > 0)
                ? EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_kernel_end_measurements[PROFILING_I] - _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_2_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxReshape_24_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args;
  int8_t *EEGFormer_onnxReshape_24_tensor = args->EEGFormer_onnxReshape_24_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args_t){.EEGFormer_onnxReshape_24_tensor = EEGFormer_onnxReshape_24_tensor,
                                                              .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr =
                                                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref;
} EEGFormer_Transpose_31_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer_Transpose_31_tiling_closure(void *EEGFormer_Transpose_31_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer_Transpose_31_tiling_closure_args_t *args = (EEGFormer_Transpose_31_tiling_closure_args_t *)EEGFormer_Transpose_31_tiling_closure_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref = args->EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 66, 8, 2] -> [1, 8, 66, 2] (Name: Transpose_31, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 66;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 2;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

          ((int8_t (*)[dimLen_2][dimLen_1][dimLen_3])EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref)[i_0][i_2][i_1][i_3] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref;
} EEGFormer_Transpose_31_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer_Transpose_31_cluster_fork(void *EEGFormer_Transpose_31_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer_Transpose_31_cluster_fork_args_t *args = (EEGFormer_Transpose_31_cluster_fork_args_t *)EEGFormer_Transpose_31_cluster_fork_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref = args->EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer_Transpose_31_tiling_closure_args_t EEGFormer_EEGFormer_Transpose_31_tiling_closure_args = (EEGFormer_Transpose_31_tiling_closure_args_t){
      .EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref = EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref,
      .EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref = EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref};

  // EEGFormer_Transpose_31_tiling_closure CLOSURE CALL
  EEGFormer_Transpose_31_tiling_closure(&EEGFormer_EEGFormer_Transpose_31_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxTranspose_44_tensor;
  int8_t *EEGFormer_onnxMatMul_45_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr;
} EEGFormer_Transpose_31_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer_Transpose_31_closure(void *EEGFormer_Transpose_31_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer_Transpose_31_closure_args_t *args = (EEGFormer_Transpose_31_closure_args_t *)EEGFormer_Transpose_31_closure_args;
  int8_t *EEGFormer_onnxTranspose_44_tensor = args->EEGFormer_onnxTranspose_44_tensor;
  int8_t *EEGFormer_onnxMatMul_45_tensor = args->EEGFormer_onnxMatMul_45_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 1056);
  void *EEGFormer_TILING_CODEGEN_L1_Transpose_31_onnxTranspose_44_tensor_ref = (void *)((char *)EEGFormer_onnxTranspose_44_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1_Transpose_31_onnxMatMul_45_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_45_tensor + 0);

  const static char Transpose_31_L2_suffix[] = " cycles \n";

  const static char Transpose_31_L2_prefix[] = "[Transpose_31_L2][SB][0 ops][Tile ";

  uint32_t Transpose_31_L2_egress_dma_wait_end_measurements[1];

  uint32_t Transpose_31_L2_egress_dma_wait_start_measurements[1];

  uint32_t Transpose_31_L2_ingress_dma_wait_end_measurements[1];

  uint32_t Transpose_31_L2_ingress_dma_wait_start_measurements[1];

  uint32_t Transpose_31_L2_kernel_end_measurements[1];

  uint32_t Transpose_31_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1_Transpose_31_numTiles[*EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1_Transpose_31_numTiles[(*EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr) + 1]; TILING_I++) {

    Transpose_31_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442848, EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref, EEGFormer_TILING_CODEGEN_L1_Transpose_31_onnxTranspose_44_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    Transpose_31_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    Transpose_31_L2_kernel_start_measurements[TILING_I] = getCycles();

    EEGFormer_Transpose_31_cluster_fork_args_t EEGFormer_EEGFormer_Transpose_31_cluster_fork_args = (EEGFormer_Transpose_31_cluster_fork_args_t){
        .EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref = EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_in_ref,
        .EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref = EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer_Transpose_31_cluster_fork, &EEGFormer_EEGFormer_Transpose_31_cluster_fork_args);

    Transpose_31_L2_kernel_end_measurements[TILING_I] = getCycles();

    Transpose_31_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311776, EEGFormer_TILING_CODEGEN_L1_Transpose_31_data_out_ref, EEGFormer_TILING_CODEGEN_L1_Transpose_31_onnxMatMul_45_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    Transpose_31_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr > 0)
                              ? EEGFormer_TILING_CODEGEN_L1_Transpose_31_numTiles[(*EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1_Transpose_31_numTiles[*EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr]; PROFILING_I++) {

    printf("%s%u] %s%u%s", Transpose_31_L2_prefix, PROFILING_I, "Input DMA took ",
           Transpose_31_L2_ingress_dma_wait_end_measurements[PROFILING_I] - Transpose_31_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           Transpose_31_L2_suffix);

    printf("%s%u] %s%u%s", Transpose_31_L2_prefix, PROFILING_I, "Kernel took ",
           Transpose_31_L2_kernel_end_measurements[PROFILING_I] - Transpose_31_L2_kernel_start_measurements[PROFILING_I], Transpose_31_L2_suffix);

    printf("%s%u] %s%u%s", Transpose_31_L2_prefix, PROFILING_I, "Output DMA took ",
           Transpose_31_L2_egress_dma_wait_end_measurements[PROFILING_I] - Transpose_31_L2_egress_dma_wait_start_measurements[PROFILING_I],
           Transpose_31_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxTranspose_44_tensor;
  int8_t *EEGFormer_onnxMatMul_45_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr;
} EEGFormer_Transpose_31_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer_Transpose_31_closure_L3(void *EEGFormer_Transpose_31_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer_Transpose_31_closure_L3_args_t *args = (EEGFormer_Transpose_31_closure_L3_args_t *)EEGFormer_Transpose_31_closure_L3_args;
  int8_t *EEGFormer_onnxTranspose_44_tensor = args->EEGFormer_onnxTranspose_44_tensor;
  int8_t *EEGFormer_onnxMatMul_45_tensor = args->EEGFormer_onnxMatMul_45_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer_Transpose_31_closure_args_t EEGFormer_EEGFormer_Transpose_31_closure_args =
      (EEGFormer_Transpose_31_closure_args_t){.EEGFormer_onnxTranspose_44_tensor = EEGFormer_onnxTranspose_44_tensor,
                                              .EEGFormer_onnxMatMul_45_tensor = EEGFormer_onnxMatMul_45_tensor,
                                              .EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr};

  // EEGFormer_Transpose_31_closure CLOSURE CALL
  EEGFormer_Transpose_31_closure(&EEGFormer_EEGFormer_Transpose_31_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref;
} EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure(void *EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args_t *args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args_t *)EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 66, 8, 2] -> [1, 8, 66, 2] (Name: _MERGE_TRANSPOSES_PASS_0, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 66;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 2;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

          ((int8_t (*)[dimLen_2][dimLen_1][dimLen_3])EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref)[i_0][i_2][i_1][i_3] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref;
} EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork(void *EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args_t *args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args_t *)EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args_t EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args_t){
          .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref};

  // EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure CLOSURE CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure(&EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxTranspose_54_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr;
} EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_TRANSPOSES_PASS_0_closure(void *EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args_t *args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args_t *)EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args;
  int8_t *EEGFormer_onnxTranspose_54_tensor = args->EEGFormer_onnxTranspose_54_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = args->EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 1056);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_onnxTranspose_54_tensor_ref = (void *)((char *)EEGFormer_onnxTranspose_54_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed_ref =
      (void *)((char *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed + 0);

  const static char _MERGE_TRANSPOSES_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_TRANSPOSES_PASS_0_L2_prefix[] = "[_MERGE_TRANSPOSES_PASS_0_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_TRANSPOSES_PASS_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_0_L2_kernel_end_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_TRANSPOSES_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442848, EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_onnxTranspose_54_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_TRANSPOSES_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_TRANSPOSES_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args_t EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args =
        (EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args_t){
            .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_in_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork, &EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_0_cluster_fork_args);

    _MERGE_TRANSPOSES_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_TRANSPOSES_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311776, EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_data_out_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_TRANSPOSES_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr > 0)
                ? EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_TRANSPOSES_PASS_0_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_TRANSPOSES_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_TRANSPOSES_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_TRANSPOSES_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_TRANSPOSES_PASS_0_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_TRANSPOSES_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_TRANSPOSES_PASS_0_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_TRANSPOSES_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_TRANSPOSES_PASS_0_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_TRANSPOSES_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_TRANSPOSES_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_TRANSPOSES_PASS_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxTranspose_54_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr;
} EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3(void *EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args_t *args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args_t *)EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args;
  int8_t *EEGFormer_onnxTranspose_54_tensor = args->EEGFormer_onnxTranspose_54_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = args->EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args_t EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args_t){
          .EEGFormer_onnxTranspose_54_tensor = EEGFormer_onnxTranspose_54_tensor,
          .EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr};

  // EEGFormer__MERGE_TRANSPOSES_PASS_0_closure CLOSURE CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_0_closure(&EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref;
} EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure(void *EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args_t *args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args_t *)EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 66, 8, 2] -> [1, 8, 2, 66] (Name: _MERGE_TRANSPOSES_PASS_1, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 66;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 2;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          ((int8_t (*)[dimLen_2][dimLen_3][dimLen_1])EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref)[i_0][i_2][i_3][i_1] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref;
} EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork(void *EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args_t *args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args_t *)EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args_t EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args_t){
          .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref};

  // EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure CLOSURE CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure(&EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_1_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxTranspose_63_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr;
} EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_TRANSPOSES_PASS_1_closure(void *EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args_t *args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args_t *)EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args;
  int8_t *EEGFormer_onnxTranspose_63_tensor = args->EEGFormer_onnxTranspose_63_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = args->EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 1056);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_onnxTranspose_63_tensor_ref = (void *)((char *)EEGFormer_onnxTranspose_63_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed_ref =
      (void *)((char *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed + 0);

  const static char _MERGE_TRANSPOSES_PASS_1_L2_suffix[] = " cycles \n";

  const static char _MERGE_TRANSPOSES_PASS_1_L2_prefix[] = "[_MERGE_TRANSPOSES_PASS_1_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_TRANSPOSES_PASS_1_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_1_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_1_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_1_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_1_L2_kernel_end_measurements[1];

  uint32_t _MERGE_TRANSPOSES_PASS_1_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_TRANSPOSES_PASS_1_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442848, EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_onnxTranspose_63_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_TRANSPOSES_PASS_1_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_TRANSPOSES_PASS_1_L2_kernel_start_measurements[TILING_I] = getCycles();

    EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args_t EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args =
        (EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args_t){
            .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_in_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork, &EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_1_cluster_fork_args);

    _MERGE_TRANSPOSES_PASS_1_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_TRANSPOSES_PASS_1_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311776, EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_data_out_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_TRANSPOSES_PASS_1_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr > 0)
                ? EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_TRANSPOSES_PASS_1_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_TRANSPOSES_PASS_1_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_TRANSPOSES_PASS_1_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_TRANSPOSES_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_TRANSPOSES_PASS_1_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_TRANSPOSES_PASS_1_L2_kernel_end_measurements[PROFILING_I] - _MERGE_TRANSPOSES_PASS_1_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_TRANSPOSES_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_TRANSPOSES_PASS_1_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_TRANSPOSES_PASS_1_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_TRANSPOSES_PASS_1_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_TRANSPOSES_PASS_1_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxTranspose_63_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr;
} EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3(void *EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args_t *args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args_t *)EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args;
  int8_t *EEGFormer_onnxTranspose_63_tensor = args->EEGFormer_onnxTranspose_63_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = args->EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args_t EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args_t){
          .EEGFormer_onnxTranspose_63_tensor = EEGFormer_onnxTranspose_63_tensor,
          .EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr};

  // EEGFormer__MERGE_TRANSPOSES_PASS_1_closure CLOSURE CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_1_closure(&EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args;
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref;
  for (int i = 0; i < *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref; i++) {
    for (int j = 0; j < 66; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_i8_i8_i8(
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref, NULL,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref,
          EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref, 1, 16, 2,
          *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref, 1, 1, NUM_CORES);
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref += 2;
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref +=
          *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref;
    }
    ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref +=
        2 * *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args;
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args_t){
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxMatMul_45_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed;
  int8_t *EEGFormer_onnxMatMul_67_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args;
  int8_t *EEGFormer_onnxMatMul_45_tensor = args->EEGFormer_onnxMatMul_45_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = args->EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed;
  int8_t *EEGFormer_onnxMatMul_67_tensor = args->EEGFormer_onnxMatMul_67_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref =
      (uint16_t *)((char *)EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O + 0);
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref =
      (uint8_t *)((char *)EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 25344);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 26400);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 27680);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 27424);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_45_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_45_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed_ref =
      (void *)((char *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_41_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_41 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_41_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_41 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_67_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_67_tensor + 0);

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_suffix[] = " cycles \n";

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_prefix[] = "[_MERGE_GEMM_MATMUL_RQ_PASS_3_L2][SB][247104 ops][Tile ";

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_egress_dma_wait_end_measurements[4];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_egress_dma_wait_start_measurements[4];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_ingress_dma_wait_end_measurements[4];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_ingress_dma_wait_start_measurements[4];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_kernel_end_measurements[4];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_kernel_start_measurements[4];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_cmd[TILING_I], EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_45_tensor_ref);

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_45_tensor_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_45_tensor_ref =
        (void *)((char *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_45_tensor_ref) +
                 EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_relativeOffset[TILING_I]);

    mchan_transfer_2d_ext_strided(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_cmd[TILING_I],
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref,
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed_ref,
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_size_1d[TILING_I], 132);

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed_ref =
        (void *)((char *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed_ref) +
                 EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_cmd[TILING_I], EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_41_ref);

    // UPDATE VARIABLE
    // EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_41_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_41_ref =
        (void
             *)((char
                     *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_41_ref) +
                EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_cmd[TILING_I], EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_41_ref);

    // UPDATE VARIABLE
    // EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_41_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_41_ref =
        (void
             *)((char
                     *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_41_ref) +
                EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_relativeOffset[TILING_I]);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_kernel_start_measurements[TILING_I] = getCycles();

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref
    *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch[TILING_I];

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref
    *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O[TILING_I];

    EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args =
        (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args_t){
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_O_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_batch_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_A_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_B_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_mul_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_C_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork,
                    &EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_cluster_fork_args);

    _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_cmd[TILING_I],
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_ref,
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_67_tensor_ref,
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_size_1d[TILING_I], 66);

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_67_tensor_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_67_tensor_ref =
        (void *)((char *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_67_tensor_ref) +
                 EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr > 0)
                ? EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_kernel_end_measurements[PROFILING_I] - _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_3_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxMatMul_45_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed;
  int8_t *EEGFormer_onnxMatMul_67_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args;
  int8_t *EEGFormer_onnxMatMul_45_tensor = args->EEGFormer_onnxMatMul_45_tensor;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = args->EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed;
  int8_t *EEGFormer_onnxMatMul_67_tensor = args->EEGFormer_onnxMatMul_67_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args_t){
          .EEGFormer_onnxMatMul_45_tensor = EEGFormer_onnxMatMul_45_tensor,
          .EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed,
          .EEGFormer_onnxMatMul_67_tensor = EEGFormer_onnxMatMul_67_tensor,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args;
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_i8_i8_i8(
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref, NULL,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref,
          EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref, 1, 16, 66, 1, 1, 1,
          NUM_CORES);
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref += 66;
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref += 1;
    }
    ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref += 66 * 1;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args;
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args_t){
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed;
  int8_t *EEGFormer_onnxMatMul_67_tensor;
  int8_t *EEGFormer_onnxTranspose_69_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = args->EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed;
  int8_t *EEGFormer_onnxMatMul_67_tensor = args->EEGFormer_onnxMatMul_67_tensor;
  int8_t *EEGFormer_onnxTranspose_69_tensor = args->EEGFormer_onnxTranspose_69_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref =
      (uint16_t *)((char *)EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 31680);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 31942);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 31938);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 31878);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_67_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_67_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed_ref =
      (void *)((char *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_43_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_43 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_43_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_43 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxTranspose_69_tensor_ref = (void *)((char *)EEGFormer_onnxTranspose_69_tensor + 0);

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_suffix[] = " cycles \n";

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_prefix[] = "[_MERGE_GEMM_MATMUL_RQ_PASS_4_L2][SB][247104 ops][Tile ";

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_egress_dma_wait_end_measurements[32];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_egress_dma_wait_start_measurements[32];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_ingress_dma_wait_end_measurements[32];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_ingress_dma_wait_start_measurements[32];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_kernel_end_measurements[32];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_kernel_start_measurements[32];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_cmd[TILING_I],
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref,
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_67_tensor_ref,
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_size_1d[TILING_I], 4356);

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_67_tensor_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_67_tensor_ref =
        (void *)((char *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_67_tensor_ref) +
                 EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_relativeOffset[TILING_I]);

    mchan_transfer_2d_ext_strided(1966146, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref,
                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed_ref, 66,
                                  132);

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed_ref =
        (void *)((char *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed_ref) +
                 EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        1441796, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_43_ref);

    // UPDATE VARIABLE
    // EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_43_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_43_ref =
        (void
             *)((char
                     *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_43_ref) +
                EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        1441796, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_43_ref);

    // UPDATE VARIABLE
    // EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_43_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_43_ref =
        (void
             *)((char
                     *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_43_ref) +
                EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_relativeOffset[TILING_I]);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_kernel_start_measurements[TILING_I] = getCycles();

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref
    *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M[TILING_I];

    EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args =
        (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args_t){
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_M_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_A_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_B_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_mul_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_C_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork,
                    &EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_cluster_fork_args);

    _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    for (uint32_t i_0 = 0; i_0 < 1; i_0++) {
      const uint32_t ext_offset = i_0 * 132;
      const uint32_t loc_offset = i_0 * EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_stride_0[TILING_I];
      void *const local_buffer_offsetted = (void *)((char *)EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_ref + loc_offset);
      void *const external_buffer_offsetted =
          (void *)((char *)EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxTranspose_69_tensor_ref + ext_offset);
      mchan_transfer_2d_ext_strided(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_cmd[TILING_I], local_buffer_offsetted,
                                    external_buffer_offsetted, 1, 2);
    }

    // UPDATE VARIABLE EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxTranspose_69_tensor_ref
    EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxTranspose_69_tensor_ref =
        (void *)((char *)(EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxTranspose_69_tensor_ref) +
                 EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr > 0)
                ? EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_kernel_end_measurements[PROFILING_I] - _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_4_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed;
  int8_t *EEGFormer_onnxMatMul_67_tensor;
  int8_t *EEGFormer_onnxTranspose_69_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args;
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = args->EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed;
  int8_t *EEGFormer_onnxMatMul_67_tensor = args->EEGFormer_onnxMatMul_67_tensor;
  int8_t *EEGFormer_onnxTranspose_69_tensor = args->EEGFormer_onnxTranspose_69_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args_t){
          .EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed,
          .EEGFormer_onnxMatMul_67_tensor = EEGFormer_onnxMatMul_67_tensor,
          .EEGFormer_onnxTranspose_69_tensor = EEGFormer_onnxTranspose_69_tensor,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref;
} EEGFormer_Transpose_44_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer_Transpose_44_tiling_closure(void *EEGFormer_Transpose_44_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer_Transpose_44_tiling_closure_args_t *args = (EEGFormer_Transpose_44_tiling_closure_args_t *)EEGFormer_Transpose_44_tiling_closure_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref = args->EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 66, 2] -> [1, 66, 8, 2] (Name: Transpose_44, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 66;

  uint16_t dimLen_3 = 2;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

          ((int8_t (*)[dimLen_2][dimLen_1][dimLen_3])EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref)[i_0][i_2][i_1][i_3] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref;
} EEGFormer_Transpose_44_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer_Transpose_44_cluster_fork(void *EEGFormer_Transpose_44_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer_Transpose_44_cluster_fork_args_t *args = (EEGFormer_Transpose_44_cluster_fork_args_t *)EEGFormer_Transpose_44_cluster_fork_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref = args->EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer_Transpose_44_tiling_closure_args_t EEGFormer_EEGFormer_Transpose_44_tiling_closure_args = (EEGFormer_Transpose_44_tiling_closure_args_t){
      .EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref = EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref,
      .EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref = EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref};

  // EEGFormer_Transpose_44_tiling_closure CLOSURE CALL
  EEGFormer_Transpose_44_tiling_closure(&EEGFormer_EEGFormer_Transpose_44_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxTranspose_69_tensor;
  int8_t *EEGFormer_onnxReshape_70_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr;
} EEGFormer_Transpose_44_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer_Transpose_44_closure(void *EEGFormer_Transpose_44_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer_Transpose_44_closure_args_t *args = (EEGFormer_Transpose_44_closure_args_t *)EEGFormer_Transpose_44_closure_args;
  int8_t *EEGFormer_onnxTranspose_69_tensor = args->EEGFormer_onnxTranspose_69_tensor;
  int8_t *EEGFormer_onnxReshape_70_tensor = args->EEGFormer_onnxReshape_70_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  int8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 1056);
  void *EEGFormer_TILING_CODEGEN_L1_Transpose_44_onnxTranspose_69_tensor_ref = (void *)((char *)EEGFormer_onnxTranspose_69_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1_Transpose_44_onnxReshape_70_tensor_ref = (void *)((char *)EEGFormer_onnxReshape_70_tensor + 0);

  const static char Transpose_44_L2_suffix[] = " cycles \n";

  const static char Transpose_44_L2_prefix[] = "[Transpose_44_L2][SB][0 ops][Tile ";

  uint32_t Transpose_44_L2_egress_dma_wait_end_measurements[1];

  uint32_t Transpose_44_L2_egress_dma_wait_start_measurements[1];

  uint32_t Transpose_44_L2_ingress_dma_wait_end_measurements[1];

  uint32_t Transpose_44_L2_ingress_dma_wait_start_measurements[1];

  uint32_t Transpose_44_L2_kernel_end_measurements[1];

  uint32_t Transpose_44_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1_Transpose_44_numTiles[*EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1_Transpose_44_numTiles[(*EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr) + 1]; TILING_I++) {

    Transpose_44_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442848, EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref, EEGFormer_TILING_CODEGEN_L1_Transpose_44_onnxTranspose_69_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    Transpose_44_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    Transpose_44_L2_kernel_start_measurements[TILING_I] = getCycles();

    EEGFormer_Transpose_44_cluster_fork_args_t EEGFormer_EEGFormer_Transpose_44_cluster_fork_args = (EEGFormer_Transpose_44_cluster_fork_args_t){
        .EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref = EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_in_ref,
        .EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref = EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer_Transpose_44_cluster_fork, &EEGFormer_EEGFormer_Transpose_44_cluster_fork_args);

    Transpose_44_L2_kernel_end_measurements[TILING_I] = getCycles();

    Transpose_44_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311776, EEGFormer_TILING_CODEGEN_L1_Transpose_44_data_out_ref, EEGFormer_TILING_CODEGEN_L1_Transpose_44_onnxReshape_70_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    Transpose_44_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr > 0)
                              ? EEGFormer_TILING_CODEGEN_L1_Transpose_44_numTiles[(*EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1_Transpose_44_numTiles[*EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr]; PROFILING_I++) {

    printf("%s%u] %s%u%s", Transpose_44_L2_prefix, PROFILING_I, "Input DMA took ",
           Transpose_44_L2_ingress_dma_wait_end_measurements[PROFILING_I] - Transpose_44_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           Transpose_44_L2_suffix);

    printf("%s%u] %s%u%s", Transpose_44_L2_prefix, PROFILING_I, "Kernel took ",
           Transpose_44_L2_kernel_end_measurements[PROFILING_I] - Transpose_44_L2_kernel_start_measurements[PROFILING_I], Transpose_44_L2_suffix);

    printf("%s%u] %s%u%s", Transpose_44_L2_prefix, PROFILING_I, "Output DMA took ",
           Transpose_44_L2_egress_dma_wait_end_measurements[PROFILING_I] - Transpose_44_L2_egress_dma_wait_start_measurements[PROFILING_I],
           Transpose_44_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxTranspose_69_tensor;
  int8_t *EEGFormer_onnxReshape_70_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr;
} EEGFormer_Transpose_44_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer_Transpose_44_closure_L3(void *EEGFormer_Transpose_44_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer_Transpose_44_closure_L3_args_t *args = (EEGFormer_Transpose_44_closure_L3_args_t *)EEGFormer_Transpose_44_closure_L3_args;
  int8_t *EEGFormer_onnxTranspose_69_tensor = args->EEGFormer_onnxTranspose_69_tensor;
  int8_t *EEGFormer_onnxReshape_70_tensor = args->EEGFormer_onnxReshape_70_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer_Transpose_44_closure_args_t EEGFormer_EEGFormer_Transpose_44_closure_args =
      (EEGFormer_Transpose_44_closure_args_t){.EEGFormer_onnxTranspose_69_tensor = EEGFormer_onnxTranspose_69_tensor,
                                              .EEGFormer_onnxReshape_70_tensor = EEGFormer_onnxReshape_70_tensor,
                                              .EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr};

  // EEGFormer_Transpose_44_closure CLOSURE CALL
  EEGFormer_Transpose_44_closure(&EEGFormer_EEGFormer_Transpose_44_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref;
  int8_t *ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref =
      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 66; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_i8_i8_i8(
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref, NULL,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref,
          ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref,
          EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref, 1, 16, 16, 16, 1, 1,
          NUM_CORES);
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref += 16;
      ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref += 16;
    }
    ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref += 16 * 16;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref;
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref;
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args_t){
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxMatMul_80_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args;
  int8_t *EEGFormer_onnxMatMul_80_tensor = args->EEGFormer_onnxMatMul_80_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 1056);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2112);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2432);
  int32_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref = (int32_t *)((char *)EEGFormer_MEMORYARENA_L1 + 2368);
  int8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref = (int8_t *)((char *)EEGFormer_MEMORYARENA_L1 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_onnxMatMul_80_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_80_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_onnxMatMul_99_tensor_ref = (void *)((char *)EEGFormer_onnxMatMul_99_tensor + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_53_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_53 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_53_ref =
      (void *)((char *)EEGFormer__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_53 + 0);
  void *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_output_0_ref = (void *)((char *)EEGFormer_output_0 + 0);

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_suffix[] = " cycles \n";

  const static char _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_prefix[] = "[_MERGE_GEMM_MATMUL_RQ_PASS_5_L2][SB][40128 ops][Tile ";

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_kernel_end_measurements[1];

  uint32_t _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr];
       TILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442848, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_onnxMatMul_80_tensor_ref);
    mchan_transfer_1d(1442048, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_onnxMatMul_99_tensor_ref);
    mchan_transfer_1d(
        1441856, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_0add_tensor_DUPLICATE_FOR_RequantShift_53_ref);
    mchan_transfer_1d(
        1441856, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref,
        EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5__QL_REPLACED__INTEGERIZE_SIGNED_ACT_PASS_3mul_tensor_DUPLICATE_FOR_RequantShift_53_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_kernel_start_measurements[TILING_I] = getCycles();

    EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args =
        (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args_t){
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_A_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_B_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_mul_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_C_ref,
            .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork,
                    &EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_cluster_fork_args);

    _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311776, EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_data_out_ref,
                      EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr > 0)
                ? EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_numTiles[(*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_numTiles[*EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_kernel_end_measurements[PROFILING_I] - _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMM_MATMUL_RQ_PASS_5_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *EEGFormer_onnxMatMul_80_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr;
} EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3(void *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args) {
  // CLOSURE ARG CAST
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args_t *args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args_t *)EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args;
  int8_t *EEGFormer_onnxMatMul_80_tensor = args->EEGFormer_onnxMatMul_80_tensor;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr = args->EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args_t){.EEGFormer_onnxMatMul_80_tensor = EEGFormer_onnxMatMul_80_tensor,
                                                              .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr =
                                                                  EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {
  int8_t *EEGFormer_onnxReshape_28_tensor;                                        // L2_init_template
  int8_t *EEGFormer_onnxReshape_32_tensor;                                        // L2_init_template
  int8_t *EEGFormer_onnxReshape_24_tensor;                                        // L2_init_template
  int8_t *EEGFormer_onnxTranspose_54_tensor;                                      // L2_init_template
  int8_t *EEGFormer_onnxTranspose_44_tensor;                                      // L2_init_template
  int8_t *EEGFormer_onnxTranspose_63_tensor;                                      // L2_init_template
  int8_t *EEGFormer_onnxMatMul_45_tensor;                                         // L2_init_template
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed; // L2_init_template
  int8_t *EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed; // L2_init_template
  int8_t *EEGFormer_onnxMatMul_67_tensor;                                         // L2_init_template
  int8_t *EEGFormer_onnxTranspose_69_tensor;                                      // L2_init_template
  int8_t *EEGFormer_onnxReshape_70_tensor;                                        // L2_init_template
  int8_t *EEGFormer_onnxMatMul_80_tensor;                                         // L2_init_template

  uint8_t bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr;
  uint8_t bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr = 0;
  uint8_t *EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr = &bu_EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr;
  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0 CALL START

  EEGFormer_onnxReshape_28_tensor = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 0);
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args_t){.EEGFormer_onnxReshape_28_tensor = EEGFormer_onnxReshape_28_tensor,
                                                                 .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr =
                                                                     EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_0_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3 CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_0 CALL END

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1 CALL START

  EEGFormer_onnxReshape_32_tensor = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 2112);
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args_t){.EEGFormer_onnxReshape_32_tensor = EEGFormer_onnxReshape_32_tensor,
                                                                 .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr =
                                                                     EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_1_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3 CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_1 CALL END

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2 CALL START

  EEGFormer_onnxReshape_24_tensor = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 1056);
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args_t){.EEGFormer_onnxReshape_24_tensor = EEGFormer_onnxReshape_24_tensor,
                                                                 .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr =
                                                                     EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_2_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3 CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_2 CALL END

  // RUNNETRWORK LAYER EEGFormer_Reshape_34 CALL START

  // Reshape (Name: Reshape_34, Op: Reshape)
  EEGFormer_onnxTranspose_54_tensor = EEGFormer_onnxReshape_28_tensor;

  // RUNNETRWORK LAYER EEGFormer_Reshape_34 CALL END

  // RUNNETRWORK LAYER EEGFormer_Reshape_30 CALL START

  // Reshape (Name: Reshape_30, Op: Reshape)
  EEGFormer_onnxTranspose_44_tensor = EEGFormer_onnxReshape_32_tensor;

  // RUNNETRWORK LAYER EEGFormer_Reshape_30 CALL END

  // RUNNETRWORK LAYER EEGFormer_Reshape_37 CALL START

  // Reshape (Name: Reshape_37, Op: Reshape)
  EEGFormer_onnxTranspose_63_tensor = EEGFormer_onnxReshape_24_tensor;

  // RUNNETRWORK LAYER EEGFormer_Reshape_37 CALL END

  // RUNNETRWORK LAYER EEGFormer_Transpose_31 CALL START

  EEGFormer_onnxMatMul_45_tensor = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 35904);
  EEGFormer_Transpose_31_closure_L3_args_t EEGFormer_EEGFormer_Transpose_31_closure_L3_args =
      (EEGFormer_Transpose_31_closure_L3_args_t){.EEGFormer_onnxTranspose_44_tensor = EEGFormer_onnxTranspose_44_tensor,
                                                 .EEGFormer_onnxMatMul_45_tensor = EEGFormer_onnxMatMul_45_tensor,
                                                 .EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1_Transpose_31_tileIdxPtr};

  // EEGFormer_Transpose_31_closure_L3 CLOSURE CALL
  EEGFormer_Transpose_31_closure_L3(&EEGFormer_EEGFormer_Transpose_31_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer_Transpose_31 CALL END

  // RUNNETRWORK LAYER EEGFormer__MERGE_TRANSPOSES_PASS_0 CALL START

  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 36960);
  EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args_t EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args_t){
          .EEGFormer_onnxTranspose_54_tensor = EEGFormer_onnxTranspose_54_tensor,
          .EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_0_tileIdxPtr};

  // EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3 CLOSURE CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3(&EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_0_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer__MERGE_TRANSPOSES_PASS_0 CALL END

  // RUNNETRWORK LAYER EEGFormer__MERGE_TRANSPOSES_PASS_1 CALL START

  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 0);
  EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args_t EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args =
      (EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args_t){
          .EEGFormer_onnxTranspose_63_tensor = EEGFormer_onnxTranspose_63_tensor,
          .EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1__MERGE_TRANSPOSES_PASS_1_tileIdxPtr};

  // EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3 CLOSURE CALL
  EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3(&EEGFormer_EEGFormer__MERGE_TRANSPOSES_PASS_1_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer__MERGE_TRANSPOSES_PASS_1 CALL END

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3 CALL START

  EEGFormer_onnxMatMul_67_tensor = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 1056);
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args_t){
          .EEGFormer_onnxMatMul_45_tensor = EEGFormer_onnxMatMul_45_tensor,
          .EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed = EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_onnxMatMul_65_tensor_transposed,
          .EEGFormer_onnxMatMul_67_tensor = EEGFormer_onnxMatMul_67_tensor,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_3_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3 CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_3 CALL END

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4 CALL START

  EEGFormer_onnxTranspose_69_tensor = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 35904);
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args_t){
          .EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed = EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_onnxMatMul_64_tensor_transposed,
          .EEGFormer_onnxMatMul_67_tensor = EEGFormer_onnxMatMul_67_tensor,
          .EEGFormer_onnxTranspose_69_tensor = EEGFormer_onnxTranspose_69_tensor,
          .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_4_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3 CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_4 CALL END

  // RUNNETRWORK LAYER EEGFormer_Transpose_44 CALL START

  EEGFormer_onnxReshape_70_tensor = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 0);
  EEGFormer_Transpose_44_closure_L3_args_t EEGFormer_EEGFormer_Transpose_44_closure_L3_args =
      (EEGFormer_Transpose_44_closure_L3_args_t){.EEGFormer_onnxTranspose_69_tensor = EEGFormer_onnxTranspose_69_tensor,
                                                 .EEGFormer_onnxReshape_70_tensor = EEGFormer_onnxReshape_70_tensor,
                                                 .EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr = EEGFormer_TILING_CODEGEN_L1_Transpose_44_tileIdxPtr};

  // EEGFormer_Transpose_44_closure_L3 CLOSURE CALL
  EEGFormer_Transpose_44_closure_L3(&EEGFormer_EEGFormer_Transpose_44_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer_Transpose_44 CALL END

  // RUNNETRWORK LAYER EEGFormer_Reshape_50 CALL START

  // Reshape (Name: Reshape_50, Op: Reshape)
  EEGFormer_onnxMatMul_80_tensor = EEGFormer_onnxReshape_70_tensor;

  // RUNNETRWORK LAYER EEGFormer_Reshape_50 CALL END

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5 CALL START

  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args_t EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args =
      (EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args_t){.EEGFormer_onnxMatMul_80_tensor = EEGFormer_onnxMatMul_80_tensor,
                                                                 .EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr =
                                                                     EEGFormer_TILING_CODEGEN_L1__MERGE_GEMM_MATMUL_RQ_PASS_5_tileIdxPtr};

  // EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3 CLOSURE CALL
  EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3(&EEGFormer_EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5_closure_L3_args);

  // RUNNETRWORK LAYER EEGFormer__MERGE_GEMM_MATMUL_RQ_PASS_5 CALL END
}

void InitNetwork() {

  EEGFormer_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 31946);

  EEGFormer_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 38016);

  EEGFormer_input_0 = (uint8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 3168);
  EEGFormer_output_0 = (int8_t *)((char *)EEGFormer_MEMORYARENA_L2 + 2112);
  EEGFormer_inputs[0] = (void *)EEGFormer_input_0;
  EEGFormer_outputs[0] = (void *)EEGFormer_output_0;
}
