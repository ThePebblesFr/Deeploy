
// ===== GLOBALS from KeywordSpotting_small_4 START =====

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

int8_t *KeywordSpotting_small_4_MEMORYARENA_L1; // L2_init_template
int8_t *KeywordSpotting_small_4_MEMORYARENA_L2; // L2_init_template
int8_t *KeywordSpotting_small_4_input_0;        // L2_init_template
int8_t *KeywordSpotting_small_4_output_0;       // L2_init_template

static PI_L2 int8_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor[72] = {
    99,   107, -30,  119, -28, 26,   -63, 76,   114, -95, 112, 24,  96, 18,  62, -18, 100, 19,  -60, 33,  -60, -15, -53, 86,
    -102, -60, -37,  -78, 12,  -128, 117, -110, 100, 22,  -42, 80,  20, 104, 14, -41, 35,  -35, 54,  115, 75,  -57, 75,  23,
    66,   -79, -128, -50, -99, 106,  37,  54,   41,  -2,  101, -92, 8,  -88, 40, -45, 40,  -27, 107, -77, -77, -77, 116, 43};

static PI_L2 int32_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor[8] = {473, 473, 473, 473, 473, 473, 473, 473};

static PI_L2 int32_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor[8] = {117993, 46419,   391189,  -30944,
                                                                                                               166165, -104412, -222232, 132940};

static PI_L2 int8_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor[72] = {
    123, -106, -127, -100, -86, 52,  46,  106,  -66, -87, 68,   -52, 78,   -30, 73,  -99, -65,  39,   27,  -33, 76, 87,   -93, -68,
    117, -43,  -45,  -124, -73, 32,  -17, -93,  3,   -87, -109, -71, -112, -82, 127, 24,  39,   -119, -84, -43, 20, -113, -55, -77,
    0,   -48,  -9,   -87,  -88, -75, -44, -101, 107, -25, 110,  40,  -108, 89,  -35, -49, -106, -127, 37,  -28, 50, -105, 95,  -94};

static PI_L2 int32_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor[8] = {688, 688, 688, 688, 688, 688, 688, 688};

static PI_L2 int32_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor[8] = {327062, -113131, 551862, 189068,
                                                                                                                      471827, -36476,  449284, 255184};

static PI_L2 int8_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor[64] = {
    -106, 127, 127,  127,  127,  -128, 127, -128, -67,  -128, 127,  -128, 127,  127,  127, -128, 127, 127,  79,   -128, 127, -92,
    -128, 127, 127,  -128, -128, -29,  127, -128, -128, -128, -128, 127,  127,  -128, 127, 127,  111, -128, 127,  127,  127, -107,
    -128, 65,  -108, -128, 127,  127,  -60, -128, 9,    -36,  127,  127,  -128, -128, 127, 127,  -30, -128, -128, -128};

static PI_L2 int32_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor[8] = {151, 151, 151, 151, 151, 151, 151, 151};

static PI_L2 int32_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor[8] = {68573,  53293,  43196,   1520,
                                                                                                                      458300, -45796, -477186, 155164};

static PI_L2 int8_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor[72] = {
    -42, 43,  82, 59, -114, -77, -20,  124,  19, -33, 53,  -49, -83, 94,   -58, -26,  -128, 86, 97,  47,  -90,  -127, -104, 96,
    62,  108, 67, 33, -1,   -98, -110, -120, 53, -63, -26, -74, -23, -91,  -84, 43,   -38,  79, -41, -94, -23,  -62,  -39,  -122,
    72,  -90, 65, 58, 92,   -99, 92,   -61,  48, 121, -18, -1,  -30, -107, 62,  -128, 80,   96, 122, -30, -106, 29,   71,   -128};

static PI_L2 int32_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor[8] = {679, 679, 679, 679, 679, 679, 679, 679};

static PI_L2 int32_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor[8] = {481546, 200993, 186984, 189067,
                                                                                                                      284144, 276920, 139472, 397262};

static PI_L2 int8_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor[64] = {
    -29, -78,  -11,  -64,  -53,  -41, -123, 106, 109, -20,  -15,  -53, -117, -126, 48,  -71, -83, -10, -43, -42, 4,  -28,
    -45, -62,  -106, 109,  -52,  34,  -45,  11,  121, 60,   -112, 51,  123,  34,   87,  127, -20, 27,  -90, -27, 96, 66,
    -82, -104, -89,  -128, -100, -32, 88,   22,  -99, -104, 65,   -96, -16,  62,   -60, -14, -11, -31, -66, -116};

static PI_L2 int32_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor[8] = {715, 715, 715, 715, 715, 715, 715, 715};

static PI_L2 int32_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor[8] = {86813,  135178, -194987, 58608,
                                                                                                                      360564, -53830, 198203,  49973};

static PI_L2 int8_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor[96] = {
    -104, -69, 125, -62, -87,  31,  36,  71,  98,  72, -128, 11,   78,  -12,  -76,  123, -48, -74, -116, 6,   57,  29,  26,  -98,
    -121, 2,   118, 74,  -75,  -17, -95, -62, 23,  70, 107,  -119, 86,  -91,  48,   109, 2,   117, -110, -49, 75,  -28, -26, -54,
    89,   63,  41,  -73, -105, 14,  38,  -60, -36, 87, 10,   6,    -32, -117, -121, -62, -66, 40,  -38,  -51, 123, 45,  92,  -62,
    -53,  47,  -86, -84, -6,   -47, -97, 77,  104, 21, -22,  -120, -47, 33,   61,   -16, -51, 72,  -103, 82,  -50, 2,   -26, 16};

static PI_L2 int32_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor[12] = {-3461, 8543, -240, 9167,  11274, -5617,
                                                                                                        -8443, 9418, 9201, -8839, -838,  -11434};

static PI_L2 int32_t KeywordSpotting_small_4_RequantShift_final_mul_tensor[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint16_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x[2] = {47, 4};

static PI_L1 uint16_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x[2] = {23, 2};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top[2] = {1, 0};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom[2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 2};

static PI_L1 uint32_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[2] = {1442262, 1441832};

static PI_L1 uint32_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[2] = {1311640, 1310800};

static PI_L1 uint8_t
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_numTiles
        [2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[2] = {0, 1};

static PI_L1 uint8_t
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_numTiles
        [2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[2] = {0, 1};

static PI_L1 uint8_t
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_numTiles
        [2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_numTiles[2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[2] = {0, 1};

void *KeywordSpotting_small_4_inputs[1];
void *KeywordSpotting_small_4_outputs[1];
extern struct pi_device cluster_dev;

// ===== GLOBALS from KeywordSpotting_small_4 END =====


// ===== GLOBALS from AnomalyDetection_small_4 START =====



int8_t *AnomalyDetection_small_4_MEMORYARENA_L1; // L2_init_template
int8_t *AnomalyDetection_small_4_MEMORYARENA_L2; // L2_init_template
int8_t *AnomalyDetection_small_4_input_0;        // L2_init_template
int8_t *AnomalyDetection_small_4_output_0;       // L2_init_template

static PI_L2 int8_t AnomalyDetection_small_4_layer0_weight_tensor[1280] = {
    9,   21,  14,  16,   -13, 4,   -13, 1,   15,  -21, 24,  -10, 16,  7,   17,  -16, -1,  -9,  -14, -6,   -14, -2,  -20, 27,  -17, 37,  -11, 10,  15,  -12, 13,
    -11, 25,  3,   -31,  2,   6,   17,  -23, 47,  45,  -6,  1,   14,  -8,  5,   -9,  -3,  -30, -22, -21,  16,  -16, 28,  1,   9,   12,  -9,  -5,  -32, 7,   -13,
    -9,  -5,  15,  -11,  22,  4,   -27, -7,  17,  39,  47,  -25, -14, -12, 9,   20,  -5,  19,  -5,  24,   22,  33,  63,  9,   60,  -41, 15,  -40, 17,  -41, 2,
    5,   7,   -21, -9,   -12, -12, -6,  31,  4,   6,   -6,  -37, -31, -24, -12, -30, 18,  3,   21,  17,   -72, -9,  59,  2,   -16, -45, -42, 10,  2,   -31, 12,
    -9,  23,  -85, -115, -8,  29,  18,  6,   -16, 8,   -16, -9,  -18, -15, 5,   -10, 3,   -15, 23,  -30,  -18, -14, 10,  -3,  9,   -1,  -2,  13,  -7,  18,  8,
    -7,  4,   12,  -2,   0,   12,  2,   -24, -20, 8,   -5,  -16, 67,  45,  -17, -10, -1,  -2,  2,   -4,   -10, -24, 10,  -6,  5,   -2,  22,  -12, -15, 0,   -12,
    8,   -42, 1,   -3,   -3,  4,   4,   4,   -8,  9,   -16, -5,  10,  20,  12,  -19, -10, -22, 7,   7,    -12, -6,  5,   38,  20,  22,  24,  -1,  29,  6,   44,
    -34, 25,  -12, -4,   -12, 19,  -30, -33, 28,  12,  6,   19,  -24, -8,  18,  -10, 8,   -13, 7,   9,    9,   -14, 0,   29,  -47, -11, 69,  11,  -4,  -20, -35,
    8,   29,  -21, 20,   15,  26,  -62, -94, -7,  4,   7,   0,   22,  2,   11,  -6,  9,   15,  -16, -4,   -9,  -16, 1,   -26, 16,  15,  21,  -24, -10, 4,   7,
    -9,  -26, 35,  -22,  8,   12,  2,   -16, 13,  8,   -10, -5,  -26, -16, -14, -11, 35,  12,  1,   2,    -14, -23, -7,  -12, -13, -47, 11,  -30, 2,   -25, 12,
    -2,  -4,  -4,  -7,   13,  -40, -3,  6,   -1,  10,  -10, -5,  3,   9,   -20, -1,  -1,  7,   10,  -8,   3,   -7,  5,   1,   15,  -5,  -6,  37,  8,   34,  50,
    -7,  40,  -16, 8,    -44, 12,  -22, -2,  -11, 16,  -3,  -4,  -1,  -13, -8,  33,  7,   -3,  -11, -22,  -21, -42, 16,  -4,  17,  -11, 0,   -3,  -46, 6,   65,
    -10, -8,  -28, -33,  14,  32,  5,   32,  12,  14,  -71, -77, -3,  30,  -4,  -13, 10,  -7,  -6,  -13,  7,   -10, 6,   -1,  -10, -1,  11,  -35, -21, 17,  -3,
    -13, -4,  -29, -9,   5,   -16, 37,  -3,  8,   8,   8,   25,  -14, -3,  -8,  -28, -35, -7,  10,  -11,  68,  30,  7,   -6,  -10, -6,  11,  11,  3,   -30, 10,
    -31, 20,  -20, 5,    -20, -8,  -13, -16, -16, -33, 9,   1,   -7,  23,  13,  -24, 7,   0,   -31, 6,    18,  28,  5,   26,  -3,  -13, 3,   29,  -9,  10,  3,
    24,  -5,  20,  60,   23,  49,  -21, 41,  -20, -2,  -10, -9,  -6,  18,  -6,  -11, -15, -17, -9,  30,   -18, 22,  3,   -5,  16,  -31, 34,  -5,  8,   3,   13,
    20,  -16, -25, 49,   5,   -15, -18, -37, 31,  16,  -39, 18,  21,  31,  -78, -83, -1,  12,  -13, -12,  26,  2,   -11, 8,   -1,  9,   13,  12,  -6,  6,   1,
    -23, -14, -5,  -4,   -28, 3,   4,   26,  13,  -15, 45,  -10, 26,  -2,  11,  -13, -17, 16,  -12, -11,  -8,  17,  12,  -20, 68,  40,  -6,  -7,  -29, -12, -5,
    1,   7,   -25, 2,    -21, 6,   -24, -11, -13, -14, -3,  -29, -4,  -56, -5,  13,  -9,  13,  16,  8,    3,   -4,  -12, -9,  7,   33,  28,  -8,  -6,  -1,  21,
    16,  10,  26,  2,    35,  16,  15,  54,  1,   62,  2,   27,  -15, 2,   -28, 18,  -15, -4,  -18, -7,   27,  1,   -23, 21,  24,  9,   5,   -31, -6,  -24, 22,
    -9,  37,  -30, 2,    32,  -43, -22, 83,  -13, -19, -42, -45, 16,  30,  -46, 47,  7,   32,  -93, -100, 16,  -23, 1,   21,  -19, -11, 8,   8,   6,   -13, 25,
    0,   25,  -1,  4,    36,  19,  -3,  -3,  34,  -20, 5,   -19, 7,   6,   -49, 16,  -27, 4,   -36, 7,    0,   19,  15,  4,   34,  -7,  -1,  2,   -60, -17, 3,
    -6,  19,  16,  7,    2,   -1,  18,  -6,  3,   -2,  2,   14,  12,  17,  18,  32,  11,  37,  9,   -12,  6,   -24, 0,   -16, 11,  -11, -13, -2,  5,   -17, -4,
    -28, 12,  14,  -16,  -34, 5,   -4,  -18, -27, 4,   -22, -56, -6,  -32, -8,  -58, 34,  6,   11,  -10,  32,  -13, 29,  14,  -21, 17,  3,   -31, 8,   1,   13,
    18,  -4,  20,  -25,  -17, -15, 34,  -7,  -21, 45,  17,  -87, 2,   5,   14,  52,  -25, -20, 20,  -51,  -41, -24, 83,  65,  13,  -7,  11,  11,  -23, -1,  4,
    -5,  -25, -11, 6,    1,   11,  -23, 9,   20,  -3,  -9,  18,  31,  2,   6,   -4,  -6,  17,  -61, 35,   -38, -5,  -7,  -8,  12,  6,   14,  5,   12,  -5,  -21,
    8,   -32, -13, -13,  -18, 5,   25,  1,   3,   -6,  19,  21,  20,  -11, 19,  10,  1,   -8,  6,   28,   17,  24,  4,   -2,  11,  -12, -10, 4,   -14, -1,  -2,
    1,   2,   -28, -29,  -19, 14,  6,   -17, -34, -1,  -24, -1,  -2,  10,  -20, -77, -6,  -50, 39,  -20,  37,  9,   26,  -8,  12,  -5,  10,  -11, 19,  34,  14,
    -37, -32, -22, 35,   33,  21,  23,  -10, 11,  -30, 8,   -31, -5,  45,  10,  -67, 2,   10,  23,  37,   -41, -6,  12,  -49, -25, -39, 74,  76,  13,  -35, -1,
    2,   21,  -8,  34,   -5,  7,   21,  -16, 6,   1,   -27, -10, 23,  28,  20,  31,  8,   -17, 14,  4,    -24, -1,  -44, 3,   -20, 4,   -19, -19, 25,  4,   4,
    26,  5,   -26, -29,  14,  -62, -46, 7,   -7,  -3,  2,   -6,  -6,  -9,  -1,  19,  -1,  -14, -3,  -1,   11,  5,   1,   33,  26,  23,  1,   6,   14,  -2,  -27,
    0,   -4,  -1,  -5,   3,   -9,  -40, -27, -8,  28,  24,  -18, -35, 23,  -19, -9,  -5,  3,   -9,  -44,  -11, -40, 15,  -59, 25,  -5,  15,  -6,  12,  -10, 31,
    22,  -8,  7,   -4,   -18, -7,  -18, 12,  25,  -10, -2,  -8,  1,   -24, 7,   -32, -36, 38,  28,  -71,  -23, 5,   11,  35,  -29, -7,  38,  -43, -20, -47, 57,
    83,  18,  -9,  -6,   -10, 10,  -15, 14,  -8,  5,   -1,  4,   12,  1,   -12, -2,  17,  -12, 25,  6,    20,  -14, -12, -13, -8,  9,   -41, 22,  -23, 2,   -11,
    21,  1,   -10, 7,    4,   -4,  -15, -4,  13,  -27, -28, 11,  -16, 1,   16,  14,  17,  11,  19,  18,   -1,  6,   3,   -5,  -8,  4,   -9,  27,  -1,  30,  16,
    3,   11,  12,  -6,   -22, 1,   -11, -15, 9,   12,  -21, -29, 24,  20,  15,  -23, -6,  -1,  -3,  -5,   -21, -13, -28, -37, 16,  -34, 10,  -30, 39,  -19, 31,
    -17, 8,   -9,  23,   11,  -22, 0,   -4,  -23, -32, 9,   22,  49,  24,  10,  5,   7,   -36, 23,  -24,  -11, 68,  -3,  -82, -7,  -2,  29,  35,  -7,  -15, 2,
    -61, -11, -31, 56,   79,  17,  -28, -16, -4,  23,  -9,  11,  14,  0,   22,  7,   28,  5,   -6,  -10,  31,  -1,  3,   10,  12,  -10, 26,  20,  2,   5,   -33,
    12,  -5,  -13, -7,   -17, 1,   10,  4,   25,  22,  8,   -4,  5,   -34, -24, -2,  -21, -12, 9,   -1,   8,   14,  30,  16,  12,  -3,  -1,  -20, 1,   -1,  -1,
    21,  13,  14,  6,    13,  15,  2,   -5,  7,   -4,  -18, 5,   -9,  4,   -26, -10, -9,  14,  29,  -13,  -27, 14,  2,   -17, -25, -9,  -51, -63, -19, -36, 26,
    -57, 30,  -22, 11,   -4,  -8,  -43, 10,  17,  6,   15,  -24, -45, 0,   -8,  25,  29,  1,   21,  -18,  8,   -13, -13, -36, -5,  48,  2,   -53, -17, -1,  26,
    40,  -13, 3,   11,   -39, -9,  -31, 72,  103};

static PI_L2 int32_t AnomalyDetection_small_4_layer0_bias_tensor[2] = {-44506, -3738};

static PI_L2 int32_t AnomalyDetection_small_4_layer0_mul_tensor[2] = {1, 1};

static PI_L1 uint8_t AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[2] = {0, 1};

void *AnomalyDetection_small_4_inputs[1];
void *AnomalyDetection_small_4_outputs[1];

// ===== GLOBALS from AnomalyDetection_small_4 END =====


// ===== SUBGROUP BARRIERS START =====


#include "hal/eu/eu_v3.h"
#include "archi/eu/eu_v3.h"

static inline uint32_t subgroup_core_mask(uint32_t group_start, uint32_t group_end)
{
    uint32_t mask = 0;

    for (uint32_t c = group_start; c <= group_end; c++) {
        for (volatile int i = 0; i < 8; i++) {
            asm volatile("nop");
        }
        // printf("Adding core %u to subgroup barrier mask (bit %08x)\n", c, bit);
        mask |= (1u << c);
    }

    return mask;
}

static inline void subgroup_barrier_init(uint32_t barrier_id,
                                         uint32_t group_start,
                                         uint32_t group_end)
{
    uint32_t mask = subgroup_core_mask(group_start, group_end);
    eu_bar_setup(eu_bar_addr(barrier_id), mask);
}

static inline void subgroup_barrier_wait(uint32_t barrier_id, uint32_t core_id)
{
    // printf("Core %u waiting on barrier %u\n", core_id, barrier_id);
    // Deliberately does NOT clear PULP_HW_BAR_EVENT from the mask after use
    // (a fix attempted and reverted -- see git history/session notes): PULP-OS's
    // OWN pi_cl_team_barrier()/pi_cl_team_fork() join-wait (pos_team_offload_wait)
    // call eu_bar_trig_wait_clr() DIRECTLY, without ever arming
    // PULP_HW_BAR_EVENT themselves -- they rely on it staying permanently
    // enabled once armed here. Clearing it after use breaks the very next
    // pi_cl_team_fork's join-wait (used to dispatch every Combined_closure_*),
    // which then hangs forever since nothing re-arms the bit for it. Confirmed
    // by tracing pi_cl_team_barrier()'s implementation
    // (pulp_hal/.../pos/implem/pe.h) after a mask-clear "fix" here caused
    // exactly that hang.
    eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);
    eu_bar_trig_wait_clr(eu_bar_addr(barrier_id));
    // printf("Core %u passed barrier %u\n", core_id, barrier_id);
}
// ===== SUBGROUP BARRIERS END =====


// ===== FUSED FUNCTIONS =====

void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_poolGlobalAveragePool_output_0_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_Flatten_output_0_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref;
void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref;
void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref;
void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref;
void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref;
void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref;
void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_output_0_ref;

typedef struct {
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t *KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;

  uint8_t *KeywordSpotting_small_4_Flatten_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;

  uint8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;

} Combined_closure_4_4_args_t;

int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref;
uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref;
uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;
uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref;
uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;
uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;
uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref;
uint8_t *KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref;
uint8_t *KeywordSpotting_small_4_Flatten_output_0_tensor;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;
uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
uint8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;

uint32_t KeywordSpotting_small_4_channel_input;
uint32_t KeywordSpotting_small_4_channel_output;
uint32_t AnomalyDetection_small_4_channel_input;
uint32_t AnomalyDetection_small_4_channel_output;

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_0(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 490);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref= (void *)((char *)KeywordSpotting_small_4_input_0 + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref=
      (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);
  // TILING LOOP
  for (int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442282, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // Transpose [1, 1, 49, 10] -> [1, 49, 10, 1] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 49;

  uint16_t dimLen_3 = 10;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / 4;
    const uint32_t leftover = dimLen_2 - baseChunk * 4;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          ((int8_t (*)[dimLen_2][dimLen_3][dimLen_1])
               KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref)[i_0][i_2][i_3][i_1] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                   KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311210, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_1(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref =
      (uint16_t *)((char *)KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref =
      (uint16_t *)((char *)KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref =
      (uint8_t *)((char *)KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref =
      (uint8_t *)((char *)KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 920);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1544);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1616);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1648);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref=
      (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer= (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1400);
  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) +
                                                                               1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[TILING_I],
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
        (void *)((char *)(KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref) + 450);

    mchan_transfer_1d(1441864, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }


    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref
    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref =
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom[TILING_I];

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref
    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref =
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top[TILING_I];

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref
    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref =
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x[TILING_I];

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref
    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref =
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x[TILING_I];

            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_i8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, 1, 18, 10,
      *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref, 1, 5,
      *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref, 8, 3, 3,
      *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref,
      *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref, 1, 1, 2, 2, 1, 1, 4);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[TILING_I],
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref);

    // UPDATE VARIABLE
    // KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref =
        (void
             *)((char
                     *)(KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref) +
                920);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_2(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref=
          (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor + 0);
  // TILING LOOP
  for (
      int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      TILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_numTiles
          [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr) +
           1];
      TILING_I++) {
        if (core_id == 0) {
        }
        subgroup_barrier_wait(2, core_id);

  // Transpose [1, 25, 5, 8] -> [1, 8, 25, 5] (Name: _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose,
  // Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 25;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / 4;
    const uint32_t leftover = dimLen_3 - baseChunk * 4;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((uint8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref)
              [i_0][i_3][i_1][i_2] =
                  ((uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref)
                      [i_0][i_1][i_2][i_3];
        }
      }
    }
  }

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_3(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1152);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3152);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3224);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3256);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2152);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer= (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr) +
                                                                               1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref);
    mchan_transfer_1d(1441864, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_depthwise_u8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
      NULL, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref, 1,
      18, 5, 25, 8, 5, 25, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_4(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2128);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2192);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2224);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer= (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2000);
  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr) +
                                                                               1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441856, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref, 1, 18, 5,
      25, 8, 5, 25, 8, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_5(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref=
          (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor + 0);
  // TILING LOOP
  for (
      int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      TILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_numTiles
          [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr) +
           1];
      TILING_I++) {
        if (core_id == 0) {
        }
        subgroup_barrier_wait(2, core_id);

  // Transpose [1, 25, 5, 8] -> [1, 8, 25, 5] (Name: _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose,
  // Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 25;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / 4;
    const uint32_t leftover = dimLen_3 - baseChunk * 4;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((uint8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref)
              [i_0][i_3][i_1][i_2] =
                  ((uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref)
                      [i_0][i_1][i_2][i_3];
        }
      }
    }
  }

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_6(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1152);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3152);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3224);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3256);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2152);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer= (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr) +
                                                                               1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref);
    mchan_transfer_1d(1441864, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_depthwise_u8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
      NULL, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref, 1,
      18, 5, 25, 8, 5, 25, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_7(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2128);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2192);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2224);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer= (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2000);
  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr) +
                                                                               1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441856, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref, 1, 18, 5,
      25, 8, 5, 25, 8, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_8(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed +
                   0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref=
          (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor + 0);
  // TILING LOOP
  for (
      int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      TILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_numTiles
          [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr) +
           1];
      TILING_I++) {
        if (core_id == 0) {
        }
        subgroup_barrier_wait(2, core_id);

  // Transpose [1, 25, 5, 8] -> [1, 8, 25, 5] (Name: _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose,
  // Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 25;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / 4;
    const uint32_t leftover = dimLen_3 - baseChunk * 4;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((uint8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref)
              [i_0][i_3][i_1][i_2] =
                  ((uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref)
                      [i_0][i_1][i_2][i_3];
        }
      }
    }
  }

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_9(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor = args->KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;
  KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_poolGlobalAveragePool_output_0_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor + 0);
  // TILING LOOP
  for (int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_numTiles
           [*KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr) +
                                                                                1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

        }
        subgroup_barrier_wait(2, core_id);

  // ReduceMean (Name: poolGlobalAveragePool, Op: ReduceMean)
  BEGIN_SINGLE_CORE
  int32_t KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator = 0;

  uint8_t *dummy_KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref =
      KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref;

  for (uint32_t i_0 = 0; i_0 < 1; i_0++) {
    for (uint32_t i_1 = 0; i_1 < 8; i_1++) {
      KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator = 128 * 125;
      for (uint32_t i_2 = 0; i_2 < 25; i_2++) {
        for (uint32_t i_3 = 0; i_3 < 5; i_3++) {
        }
      }
      int8_t KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_sgn = 0;
      KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_sgn =
          -(KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator < 0) +
          (KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator >= 0);
      *dummy_KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref++ =
          (uint8_t)((KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator +
                     KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_sgn * (125 >> 1)) /
                        125 +
                    0);
    }
  }
  END_SINGLE_CORE

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310728, KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_poolGlobalAveragePool_output_0_tensor_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) KeywordSpotting_small_4_layer_10(Combined_closure_4_4_args_t* args, int core_id) {
  KeywordSpotting_small_4_Flatten_output_0_tensor = args->KeywordSpotting_small_4_Flatten_output_0_tensor;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 204);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 12);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 156);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 108);
  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_Flatten_output_0_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_Flatten_output_0_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref=
      (void *)((char *)KeywordSpotting_small_4_RequantShift_final_mul_tensor + 0);
KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref= (void *)((char *)KeywordSpotting_small_4_output_0 + 0);
  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr) +
                                                                               1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441800, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_Flatten_output_0_tensor_ref);
    mchan_transfer_1d(1441888, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref);
    mchan_transfer_1d(1441840, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref);
    mchan_transfer_1d(1441840, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN GEMM
  int8_t *ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref =
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref =
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int8_t *
      ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
          KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_u8_i8_i8(
          ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
          NULL,
          ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref,
          ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
          KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref, 1, 7, 8,
          12, 1, 1, 4);
    }
  }

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310732, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
}

static void __attribute__((noinline)) AnomalyDetection_small_4_layer_0(Combined_closure_4_4_args_t* args, int core_id) {
  AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
  AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 1282);
  AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 2);
  AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = (int32_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 1930);
  AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = (int32_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 1922);
  AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 0);
AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref= (void *)((char *)AnomalyDetection_small_4_input_0 + 0);
AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref=
      (void *)((char *)AnomalyDetection_small_4_layer0_weight_tensor + 0);
AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref= (void *)((char *)AnomalyDetection_small_4_layer0_bias_tensor + 0);
AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref= (void *)((char *)AnomalyDetection_small_4_layer0_mul_tensor + 0);
AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_output_0_ref= (void *)((char *)AnomalyDetection_small_4_output_0 + 0);
  // TILING LOOP
  for (int TILING_I =
           AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[*AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr];
       TILING_I <
       AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[(*AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 4) {
    // Transfer input tiles
    AnomalyDetection_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442432, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref);
    mchan_transfer_1d(1443072, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref);
    mchan_transfer_1d(1441800, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref);
    mchan_transfer_1d(1441800, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref);

    // Wait for input tiles

    if (AnomalyDetection_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_small_4_channel_input);
      mchan_channel_free(AnomalyDetection_small_4_channel_input);
    }


            }
        subgroup_barrier_wait(3, core_id);

  // PULP NN GEMM
  int8_t *ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref =
      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref =
      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int8_t
      *ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
          AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_i8_i8_i8(
          ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
          NULL,
          ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref,
          ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
          AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref, 1, 10, 640,
          2, 1, 1, 4);
    }
  }

        subgroup_barrier_wait(3, core_id);
        if (core_id == 4) {
    // Transfer output tiles
    AnomalyDetection_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310722, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_output_0_ref);

    // Wait for output tiles

    if (AnomalyDetection_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_small_4_channel_output);
      mchan_channel_free(AnomalyDetection_small_4_channel_output);
    }

        }
        subgroup_barrier_wait(3, core_id);
      }
}

static void Combined_closure_4_4(Combined_closure_4_4_args_t* closure_args) {
    Combined_closure_4_4_args_t* args = (Combined_closure_4_4_args_t*) closure_args;
    int core_id = pi_core_id();

    if (core_id == 0) {
        subgroup_barrier_init(2, 0, 3);
        subgroup_barrier_init(3, 4, 7);
    }
    subgroup_barrier_init(4, 0, NUM_CORES - 1);
    pi_cl_team_barrier();
    eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);

    KeywordSpotting_small_4_channel_input = (uint32_t)-1;
    KeywordSpotting_small_4_channel_output = (uint32_t)-1;
    if (0 <= core_id && core_id <= 3) {
        KeywordSpotting_small_4_layer_0(args, core_id);
        KeywordSpotting_small_4_layer_1(args, core_id);
        KeywordSpotting_small_4_layer_2(args, core_id);
        KeywordSpotting_small_4_layer_3(args, core_id);
        KeywordSpotting_small_4_layer_4(args, core_id);
        KeywordSpotting_small_4_layer_5(args, core_id);
        KeywordSpotting_small_4_layer_6(args, core_id);
        KeywordSpotting_small_4_layer_7(args, core_id);
        KeywordSpotting_small_4_layer_8(args, core_id);
        KeywordSpotting_small_4_layer_9(args, core_id);
        KeywordSpotting_small_4_layer_10(args, core_id);
    }
    AnomalyDetection_small_4_channel_input = (uint32_t)-1;
    AnomalyDetection_small_4_channel_output = (uint32_t)-1;
    if (4 <= core_id && core_id <= 7) {
        AnomalyDetection_small_4_layer_0(args, core_id);
    }
    subgroup_barrier_wait(4, core_id);
}


void RunNetwork() {
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed; // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;                                 // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;                                 // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;                                 // L2_init_template
  uint8_t *KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;                                                                 // L2_init_template
  uint8_t *KeywordSpotting_small_4_Flatten_output_0_tensor;                                                                               // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer;                                                                              // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer;                                                                              // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer;                                                                              // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer;                                                                              // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer;                                                                              // L2_init_template

  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  uint8_t
      bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          0;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
  uint8_t
      bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          0;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
  uint8_t
      bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          0;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;


  uint8_t bu_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr = 0;
  uint8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      &bu_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;


  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);

  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);

  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);

  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);

  KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);

  KeywordSpotting_small_4_Flatten_output_0_tensor = KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;


Combined_closure_4_4_args_t Combined_closure_args = (Combined_closure_4_4_args_t) {

              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr,

          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr,

              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr,

          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr,

          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr,

              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr,

          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr,

          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr,

              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr,

          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor,
          .KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor = KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr,
.KeywordSpotting_small_4_Flatten_output_0_tensor =
                                                                           KeywordSpotting_small_4_Flatten_output_0_tensor,
                                                                       .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
                                                                           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr,
.AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
                                                                         AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr,
};
    pi_cl_team_fork(NUM_CORES, (void *)Combined_closure_4_4, &Combined_closure_args);

}

void InitNetwork() {

  KeywordSpotting_small_4_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 3288);

  KeywordSpotting_small_4_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 2000);

  KeywordSpotting_small_4_input_0 = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);
  KeywordSpotting_small_4_output_0 = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 8);
  KeywordSpotting_small_4_inputs[0] = (void *)KeywordSpotting_small_4_input_0;
  KeywordSpotting_small_4_outputs[0] = (void *)KeywordSpotting_small_4_output_0;

  AnomalyDetection_small_4_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 1938);

  AnomalyDetection_small_4_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 642);

  AnomalyDetection_small_4_input_0 = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L2 + 2);
  AnomalyDetection_small_4_output_0 = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L2 + 0);
  AnomalyDetection_small_4_inputs[0] = (void *)AnomalyDetection_small_4_input_0;
  AnomalyDetection_small_4_outputs[0] = (void *)AnomalyDetection_small_4_output_0;
}
