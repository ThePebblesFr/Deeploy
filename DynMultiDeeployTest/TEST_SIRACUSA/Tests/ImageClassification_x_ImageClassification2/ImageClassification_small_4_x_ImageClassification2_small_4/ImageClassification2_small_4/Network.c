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

int8_t *ImageClassification2_small_4_MEMORYARENA_L1; // L2_init_template
int8_t *ImageClassification2_small_4_MEMORYARENA_L2; // L2_init_template
int8_t *ImageClassification2_small_4_input_0;        // L2_init_template
int8_t *ImageClassification2_small_4_output_0;       // L2_init_template

static PI_L2 int8_t ImageClassification2_small_4_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor[432] = {
    29,  25,  -7, 23,  -5,  7,   14,  -21, -11, -31, -10, 17,  11,  -8,  13,  -18, -11, 26,  0,  12,  20,  -24, 25,  20,  -11, -2,  10,  21,  5,   32,  -3,
    -6,  -18, 1,  5,   -7,  -3,  -11, -13, 13,  -23, -9,  -20, 15,  2,   -13, -13, 8,   -3,  -9, -7,  26,  -19, 18,  -12, 21,  -2,  -11, 22,  8,   -21, 13,
    1,   1,   34, 7,   0,   8,   9,   11,  6,   -9,  -1,  -2,  -33, 28,  -15, -11, -17, 20,  1,  -5,  -15, 1,   -20, 10,  10,  -8,  -25, 9,   9,   -12, 17,
    8,   21,  -6, 18,  8,   -2,  1,   -37, 12,  12,  8,   7,   -7,  12,  3,   4,   2,   11,  20, -5,  7,   0,   8,   30,  -4,  0,   16,  -21, 4,   -21, -1,
    2,   -16, -9, 12,  -2,  8,   17,  25,  11,  6,   10,  9,   -14, 15,  18,  15,  -6,  1,   25, 25,  -27, 22,  -37, 14,  5,   -6,  -5,  -3,  -17, 16,  -11,
    13,  -10, 2,  -28, 10,  6,   4,   1,   1,   -14, -5,  -23, -23, 5,   12,  -17, -10, 12,  -7, 3,   31,  -7,  -16, 1,   -22, -24, 2,   -28, 21,  -12, 2,
    20,  -3,  16, -8,  8,   7,   11,  21,  -12, 11,  8,   -15, 27,  32,  -7,  -13, -7,  -8,  15, -14, 3,   -5,  3,   7,   -17, 17,  -2,  6,   20,  -34, -5,
    26,  -30, 17, 3,   -16, -26, -15, 1,   23,  -8,  2,   5,   2,   3,   -7,  7,   7,   16,  30, -13, 26,  -1,  5,   0,   -13, -8,  7,   9,   7,   -13, 9,
    -1,  0,   8,  12,  -5,  4,   -24, 22,  -10, -1,  19,  9,   11,  -10, -22, 1,   13,  -24, 20, 8,   -22, 4,   -29, 1,   15,  1,   18,  -15, -20, 30,  -8,
    10,  11,  -6, -38, 15,  -4,  0,   7,   -1,  -1,  18,  -10, -11, 5,   0,   26,  0,   31,  4,  13,  -15, 15,  -14, -11, 11,  7,   0,   0,   19,  19,  2,
    -34, -14, 21, -19, 7,   11,  3,   -10, 7,   -31, 16,  -10, -2,  -21, 7,   4,   -7,  6,   -8, 1,   0,   2,   10,  2,   14,  25,  6,   3,   -1,  11,  12,
    17,  -25, 0,  15,  -12, 9,   -10, 7,   -10, -8,  -4,  29,  3,   -11, 26,  33,  6,   1,   26, 2,   -2,  6,   -18, -26, 10,  -4,  -10, -3,  -12, -5,  11,
    -8,  29,  -2, 30,  -3,  -17, -19, 3,   15,  7,   23,  -4,  -8,  -14, -12, 13,  -9,  13,  -7, -1,  1,   -5,  -14, -15, 11,  -37, -11, -1,  -4,  -6,  -3,
    13,  15,  -2, -2,  -24, 3,   12,  -7,  0,   11,  -11, 15,  -19, 5,   -23, -7,  3,   15,  10, 16,  18,  -7,  -9,  -2,  -9,  -7,  -4,  0,   11};

static PI_L2 int32_t ImageClassification2_small_4_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor[16] = {
    17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0};

static PI_L2 int32_t ImageClassification2_small_4_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor[16] = {
    524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0,
    524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor[64] = {
    -44, 103, -41, 127, 77,  -4, 60,  95,  62,   -118, 4,    -105, -9,  108, 69,  -67,  -124, 44,   -90, 3,  -38, -88,
    -57, 75,  127, -10, -43, 11, 7,   -84, -113, 98,   99,   -61,  6,   0,   15,  108,  -117, -126, -96, 28, 38,  -64,
    127, 127, 127, -98, 127, 77, -36, -54, -45,  -55,  -127, -94,  -40, -6,  -49, -103, -40,  -2,   35,  -4};

static PI_L2 int32_t
    ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1[4] = {
        2056.0, 2056.0, 2056.0, 2056.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1[4] = {
        524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor[576] = {
    36,  -77, -18, -11, 11,  19,  -5,  -1,  15,  -33, 23,  -4,  20,  -69, 23,  -17, -9,  22,  5,   -53, -79, 37,  50,  56,  20,  6,   2,   -27, 5,   39,  36,
    -32, 20,  25,  46,  3,   -43, -18, 60,  79,  36,  -23, -9,  10,  54,  7,   -12, -11, 3,   19,  7,   -57, -29, -52, -37, 12,  1,   -53, -7,  25,  19,  -25,
    16,  25,  -22, 13,  -21, -19, 8,   -3,  34,  12,  17,  -7,  -36, -33, -56, 50,  -21, 46,  -19, -10, -28, 41,  -5,  -41, 54,  6,   -18, -13, -28, 15,  70,
    48,  -12, 2,   -38, 16,  16,  -38, 37,  34,  -62, -7,  -5,  66,  2,   -36, -27, -2,  5,   44,  60,  -40, -13, 24,  -12, -18, -12, 10,  19,  -14, 3,   4,
    21,  13,  3,   7,   -28, 6,   6,   20,  31,  -23, -29, -13, -24, -38, 23,  14,  -13, 12,  17,  16,  -38, 8,   -6,  0,   -15, -8,  -2,  26,  -40, 4,   -38,
    39,  54,  10,  -12, -11, -1,  -40, -65, -15, 29,  4,   47,  10,  1,   25,  12,  9,   -29, -11, 18,  16,  58,  54,  3,   26,  9,   27,  -3,  14,  -12, 62,
    11,  27,  -37, 86,  11,  -24, 11,  -2,  33,  2,   -17, 5,   -12, 46,  -21, 24,  -43, -24, -69, 25,  -94, -14, 4,   -4,  -3,  -50, -84, 23,  44,  39,  -10,
    9,   -44, -42, 0,   24,  -43, 19,  -22, -26, -8,  64,  -21, 9,   -21, -3,  12,  6,   -6,  14,  -25, 56,  -43, -8,  -35, -19, -28, -45, 16,  83,  -37, 42,
    31,  11,  7,   16,  -49, 18,  -18, 24,  24,  -12, -2,  5,   -51, 13,  91,  -14, 39,  3,   -2,  -40, 10,  3,   -7,  -23, 14,  -20, -30, -35, 26,  -9,  41,
    -29, 29,  52,  30,  58,  36,  -48, -21, -28, 1,   2,   0,   -30, 1,   65,  -37, 12,  -42, -64, 39,  -37, 28,  10,  48,  10,  3,   -45, -43, -20, -30, -19,
    -17, 25,  -70, -49, 39,  -47, -11, 56,  -12, 34,  23,  11,  -18, -14, 7,   4,   -8,  -26, -8,  -19, -35, -76, -46, 34,  56,  -36, 41,  23,  -5,  -38, -22,
    -6,  2,   45,  -4,  11,  -2,  42,  -24, -15, -32, 79,  4,   43,  6,   33,  42,  22,  13,  26,  -4,  -11, -29, -16, -21, -22, 27,  -2,  -73, 21,  -37, -32,
    -9,  11,  15,  -6,  35,  15,  45,  23,  -11, 5,   -25, -2,  13,  -4,  -12, -59, -27, 48,  15,  30,  -7,  -71, 43,  -25, 29,  -7,  31,  38,  58,  28,  44,
    9,   58,  -10, 31,  22,  50,  -16, -39, 2,   -24, 16,  -26, 27,  -22, 16,  10,  14,  -49, -13, -17, -4,  -16, -17, -39, 41,  -71, 50,  -17, -28, 2,   -31,
    4,   3,   -18, 66,  -48, -68, -30, -15, -41, 21,  -39, -2,  -1,  32,  -9,  -39, 15,  11,  -23, -19, -10, -4,  13,  10,  -13, -1,  20,  40,  0,   25,  -67,
    22,  9,   44,  40,  14,  2,   -1,  65,  3,   5,   3,   -2,  -45, 21,  -24, -13, 11,  -17, -10, -27, 26,  21,  4,   -15, 20,  34,  -49, 9,   -8,  4,   27,
    13,  18,  18,  -7,  -7,  -50, 56,  -16, 33,  -12, 17,  -22, -10, 64,  11,  -11, -11, 28,  -23, -13, 28,  36,  -57, 3,   38,  36,  48,  37,  -3,  -3,  4,
    -10, -6,  21,  -5,  -21, -13, -86, -41, -4,  30,  22,  -36, 13,  36,  19,  9,   -34, -12, 8,   -40, -4,  -28, -11, 17,  -16, 7,   21,  29,  0,   -36, 24,
    41,  -53, 27,  4,   -15, -9,  -21, -21, 14,  -2,  15,  -24, -47, -3,  9,   24,  44,  -76};

static PI_L2 int32_t ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2[4] = {
    4112.0, 4112.0, 4112.0, 4112.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2[4] = {
        524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor[144] = {
    3,  43,  -10, -45, -23, 24,  26,  10,  12,  15, 7,   -43, 46,  -5,  12,  58,  -26, -31, 37,  -1, 16,  -3,  37,  -8, 11, -27, 22,  43,  8,
    8,  -1,  -27, 36,  3,   -44, -23, -12, -33, 48, 104, 34,  -23, -31, -33, 8,   -6,  -22, -13, -3, 1,   -11, 0,   24, -6, 5,   17,  -33, 37,
    -2, 4,   30,  21,  34,  -26, -35, -56, 2,   37, 29,  6,   -1,  10,  -3,  -2,  -13, -32, 75,  16, -17, -30, 18,  36, 54, -29, 5,   66,  -22,
    88, -39, 54,  13,  44,  -18, 8,   -17, -16, 20, -73, 4,   -28, -7,  -10, -9,  -22, -42, -40, 26, 5,   -7,  -19, 7,  33, -22, -26, 45,  -15,
    38, -46, 17,  -25, 31,  -23, 11,  -6,  31,  18, 26,  69,  20,  23,  62,  -23, 27,  17,  12,  -8, 25,  40,  15,  48, 6,  52,  0,   -8};

static PI_L2 int32_t ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3[4] = {
    0.0, 0.0, 0.0, 0.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3[4] = {
        524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor[16] = {
    -8, 96, 127, 80, -21, 50, -5, -47, -127, 48, -62, -7, -24, 117, -127, -28};

static PI_L2 int32_t
    ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5[4] = {
        2056.0, 2056.0, 2056.0, 2056.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5[4] = {
        524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor[144] = {
    4,   6,   -23, 2,   -44, -34, 3,   -20, -51, 19,  -22, -16, -16, -25, 29, 25,  17,  -14, 14,  69,  -40, 10,  34,  -53, 38, 22,  16,  -29, -61,
    -18, -34, 20,  43,  17,  -22, 31,  29,  -3,  34,  -5,  1,   -18, 8,   1,  -32, -38, -13, 3,   -22, -8,  -27, 21,  74,  -8, -5,  30,  -57, 33,
    -20, -22, 10,  6,   -3,  -38, -44, 16,  48,  -18, -14, -29, -46, 42,  8,  -28, -21, -11, -13, -70, -10, -57, -35, -23, 14, 2,   43,  25,  6,
    -14, 0,   37,  -15, -36, 0,   -1,  -16, -10, -29, 35,  1,   1,   -8,  37, -45, -1,  24,  -9,  16,  36,  52,  20,  -14, -7, 62,  -31, 20,  -8,
    -7,  -15, 17,  -18, -40, -34, 38,  -1,  -15, -4,  -6,  -8,  -20, -19, 42, 44,  15,  13,  -24, 1,   -43, 10,  11,  -16, 5,  -25, -8,  -14};

static PI_L2 int32_t ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6[4] = {
    4112.0, 4112.0, 4112.0, 4112.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6[4] = {
        524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor[144] = {
    87,  5,  -5,  -5,  -35, 22, -38, 32,  58,  -41, 18,  40,  -34, 36,  -10, 10,  -47, 1,   -15, 5,   3,   60,  -11, 9,   -27, -3,  52,  -35, -11,
    -34, 6,  -31, -29, 28,  54, -14, -26, -16, 60,  -47, 17,  3,   -23, 20,  -26, 26,  -36, -13, 24,  -31, -3,  -30, 20,  18,  10,  12,  -8,  31,
    -13, 2,  8,   -23, -23, 18, -40, 12,  -22, -41, -24, 13,  46,  34,  9,   3,   -36, 19,  32,  52,  12,  -23, -1,  -18, -7,  -20, 6,   7,   -15,
    -11, -6, 46,  -14, 14,  -5, -34, -16, 12,  -25, -21, -31, -57, 36,  41,  8,   8,   -10, 46,  6,   43,  21,  -17, -9,  -58, 45,  -14, 10,  -46,
    -33, 4,  -3,  -5,  -22, 12, 3,   -19, 31,  -58, -44, -37, -55, -23, 16,  -37, 9,   -4,  23,  -45, 21,  -10, -6,  -16, 22,  4,   18,  38};

static PI_L2 int32_t ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7[4] = {
    0.0, 0.0, 0.0, 0.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7[4] = {
        524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor[32] = {
    -103, -57, 90, 58, 33, -53, -66, 36, -60, 67, -72, -127, 82, 15, 21, 85, -54, -44, -13, -49, 44, 50, -70, -127, -63, -38, -45, -29, 36, 127, 68, -9};

static PI_L2 int32_t ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor[8] = {
    2056.0, 2056.0, 2056.0, 2056.0, 2056.0, 2056.0, 2056.0, 2056.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9[8] = {
        524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor[288] = {
    -18, -32, -27, 15,  5,   -17, -29, 20,  42,  25,  0,   6,   -15, 5,   -13, 14,  -18, 9,   27,  -15, -6,  1,   -17, -18, -11, -12, 25,  20,  -7,
    -3,  -25, -12, -26, -6,  0,   9,   -16, 44,  18,  23,  -3,  15,  21,  19,  9,   -23, -3,  -18, 30,  -6,  -34, -5,  -17, 10,  -1,  21,  34,  -6,
    -21, -3,  -23, -8,  25,  -2,  26,  9,   35,  16,  26,  25,  -6,  8,   -11, -13, 11,  -4,  14,  5,   -3,  -31, -63, -9,  3,   -12, 4,   -2,  27,
    18,  -10, 0,   -16, -4,  -20, 36,  20,  -26, -32, 23,  9,   6,   33,  -3,  6,   -24, -19, -26, 3,   17,  1,   19,  -12, 5,   9,   11,  27,  8,
    -35, -17, 19,  -20, -8,  8,   -11, 34,  3,   -26, 27,  40,  -2,  -12, -19, -22, 2,   -40, 17,  -4,  16,  -20, 35,  6,   -19, 8,   2,   25,  -2,
    -32, 40,  6,   -1,  -12, 12,  29,  19,  7,   2,   -19, -18, -1,  10,  -33, -12, -24, 21,  15,  23,  -58, 10,  -21, 26,  0,   -4,  -7,  17,  -19,
    -31, 5,   -27, 10,  11,  24,  -31, -19, 27,  -1,  -5,  26,  62,  3,   -10, -28, 42,  3,   -1,  -10, -11, 0,   -19, 37,  -27, 9,   14,  -12, 23,
    6,   17,  -61, -9,  -18, -42, 24,  29,  31,  -14, 2,   -51, -40, -12, -11, -14, -14, -14, -6,  -10, -17, -1,  2,   9,   -6,  21,  45,  15,  -2,
    4,   17,  2,   -17, -2,  -7,  7,   7,   -4,  27,  2,   35,  -7,  -5,  37,  -22, -22, 5,   7,   7,   7,   4,   -17, 1,   8,   -17, -48, 25,  45,
    -36, -27, 20,  -11, 24,  7,   -36, -37, 15,  13,  1,   3,   25,  -9,  2,   8,   -3,  -18, -28, 11,  12,  -31, -22, 13,  -50, 21,  23};

static PI_L2 int32_t ImageClassification2_small_4_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor[8] = {4112.0, 4112.0, 4112.0, 4112.0,
                                                                                                                              4112.0, 4112.0, 4112.0, 4112.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10[8] = {
        524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor[576] = {
    12,  25,  -3,  -11, 2,   25,  1,   -3,  33,  -37, -37, -9,  -18, 15,  -14, 25,  1,   16,  31,  -3,  -24, 19,  -40, 35,  7,   -2,  11,  -33, -17, -8,  7,
    10,  1,   -20, -8,  -32, 28,  -16, -14, -22, 20,  -30, 43,  19,  27,  37,  38,  -13, -19, -16, 1,   10,  -5,  16,  -6,  -21, 25,  13,  -5,  6,   -28, 0,
    12,  9,   -14, 5,   -1,  -1,  -11, 23,  22,  7,   -37, 10,  7,   22,  -4,  14,  38,  -3,  10,  21,  15,  46,  16,  4,   -1,  11,  2,   -14, 18,  5,   31,
    29,  20,  12,  -14, 35,  -23, 26,  5,   -7,  -29, -38, -6,  -5,  13,  -26, -17, -10, -15, -12, 15,  29,  18,  -20, -20, 11,  20,  -12, -3,  -1,  -82, -3,
    -7,  19,  -9,  -7,  19,  -7,  3,   7,   -26, 8,   -10, -15, 12,  -5,  43,  16,  -1,  23,  11,  -31, 16,  19,  -20, 13,  -20, -13, -27, -7,  -5,  2,   -12,
    24,  -19, -30, 41,  -12, 10,  22,  5,   9,   27,  -12, -3,  -1,  1,   22,  9,   15,  8,   -14, 14,  7,   -1,  -25, -18, 5,   -42, 11,  15,  35,  6,   21,
    28,  -20, 1,   16,  12,  -31, 11,  10,  3,   -16, 10,  -4,  -16, -6,  -31, -21, 13,  15,  26,  -12, 0,   -8,  -27, -5,  -27, -9,  -21, -9,  -17, -16, -32,
    22,  -10, -18, -21, -21, -21, 25,  -29, -41, -3,  -18, -30, 5,   2,   -18, 21,  -11, -32, 33,  16,  -17, -11, -10, 1,   -3,  -7,  1,   13,  21,  -29, -20,
    6,   -26, -15, 0,   -28, -25, 41,  -7,  3,   -3,  -32, -9,  -1,  2,   29,  -37, -11, 15,  4,   40,  -22, -3,  -18, 7,   20,  3,   -3,  -18, 7,   9,   -39,
    -9,  44,  -14, -11, 10,  -47, 11,  29,  21,  9,   8,   28,  -5,  45,  25,  7,   5,   36,  -8,  5,   -30, 3,   -7,  -2,  9,   -38, 22,  30,  -22, 1,   19,
    -16, 4,   -10, -8,  15,  24,  -5,  -30, -23, 29,  7,   39,  10,  5,   -14, -18, 19,  -5,  18,  -16, -1,  11,  -16, 1,   -35, 1,   6,   -27, 24,  7,   -36,
    14,  7,   -11, 12,  -27, 60,  -25, 26,  19,  3,   -30, -41, -2,  -1,  -19, 10,  25,  18,  60,  -21, 21,  16,  31,  -1,  8,   -6,  9,   22,  -20, -29, 38,
    -6,  -14, -14, -21, 19,  -1,  -18, 8,   21,  -6,  0,   17,  6,   14,  -32, 14,  3,   -25, -5,  19,  28,  -32, 3,   14,  -1,  2,   -16, 2,   -22, -9,  7,
    9,   -20, 4,   10,  -7,  0,   -34, -15, 15,  31,  13,  13,  -2,  -41, 30,  19,  -17, 3,   33,  25,  4,   14,  9,   -16, 38,  -3,  2,   -50, -17, 35,  36,
    48,  9,   8,   43,  17,  30,  -1,  9,   22,  10,  28,  3,   5,   -1,  35,  0,   -6,  0,   -32, 7,   18,  -12, -4,  -2,  -16, 28,  23,  13,  19,  -24, -4,
    11,  -4,  24,  -16, -4,  39,  -23, 3,   -27, 30,  -21, -9,  -52, 32,  -16, 15,  8,   -20, -3,  0,   -34, 7,   -12, -2,  0,   44,  -10, 6,   38,  -24, 2,
    32,  -8,  9,   8,   19,  -8,  -10, -16, -30, 40,  -2,  -16, 10,  3,   10,  -5,  -19, -20, 17,  1,   12,  -3,  21,  6,   -1,  -20, -11, -15, 5,   13,  9,
    -9,  11,  38,  -10, 20,  26,  26,  -11, 18,  47,  33,  -15, 3,   7,   17,  3,   -29, 25,  -22, -10, 5,   37,  11,  -23, -14, 14,  30,  16,  -27, 52,  10,
    28,  19,  -9,  -4,  -41, 48,  6,   -17, -1,  34,  -29, 20,  -12, 18,  29,  -14, 24,  -12};

static PI_L2 int32_t ImageClassification2_small_4_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor[8] = {0.0, 0.0, 0.0, 0.0,
                                                                                                                            0.0, 0.0, 0.0, 0.0};

static PI_L2 int32_t
    ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11[8] = {
        524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor[80] = {
    14,  120, -43, -86, -23, -25, -72, 85,  -11, -67, 9,   37,  -13, 23,  -7,  15,  -21, -94, -18, 121, -31, 23,  -127, 22, 12, -57, -34,
    11,  -65, 77,  -56, -34, 50,  42,  16,  99,  77,  -72, 26,  38,  -37, -50, -61, 82,  -63, 23,  -27, -16, -11, -11,  96, 39, 67,  2,
    -52, 13,  72,  -46, 118, 18,  -84, -11, 2,   17,  24,  -46, -39, 52,  -43, -11, -20, -88, -49, -22, 96,  -6,  -31,  30, 40, 42};

static PI_L2 int32_t ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor[10] = {5449.5,  4205.5,  8333.5,  -517.5,   9507.5,
                                                                                                             -4668.5, -5293.5, -4244.5, -10784.5, -7546.5};

static PI_L2 int32_t ImageClassification2_small_4_RequantShift_final_mul_tensor[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[2] = {0, 1};

void *ImageClassification2_small_4_inputs[1];
void *ImageClassification2_small_4_outputs[1];
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(
    void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *)
          ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 3, 32, 32] -> [1, 32, 32, 3] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 3;

  uint16_t dimLen_2 = 32;

  uint16_t dimLen_3 = 32;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          ((int8_t (*)[dimLen_2][dimLen_3][dimLen_1])
               ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref)[i_0][i_2][i_3][i_1] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                   ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork(
    void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *)
          ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t){
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure(
    void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *)
          ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 3072);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref =
      (void *)((char *)ImageClassification2_small_4_input_0 + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);

  const static char _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_input_0_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1444864, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t){
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1313792, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(
    void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *)
          ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_i8_u8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
      1, 19, 32, 32, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 16384);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 19456);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 20320);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 20384);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 19888);

  const static char _MERGE_CONVRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_L2][SB][887808 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1444864, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);
    mchan_transfer_1d(1442224, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref);
    mchan_transfer_1d(1441856, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref);
    mchan_transfer_1d(1441856, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1327104, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args_t){
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
      1, 19, 32, 32, 16, 32, 32, 4, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 20736);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 20800);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 20816);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 16384);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1 +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1 +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 20480);

  const static char _MERGE_CONVRQ_PASS_1_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_1_L2_prefix[] = "[_MERGE_CONVRQ_PASS_1_L2][SB][134144 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1458176, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1441856, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args);

    _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1314816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_1_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args_t){
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
      1, 19, 32, 32, 16, 32, 32, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 22784);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 23360);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 23376);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 16384);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2 +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2_ref =
          (void *)((char *)ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2 + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 20480);

  const static char _MERGE_CONVRQ_PASS_2_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_2_L2_prefix[] = "[_MERGE_CONVRQ_PASS_2_L2][SB][1182720 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1458176, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1442368, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args);

    _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1314816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_2_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_2_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_2_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args_t){
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
      1, 19, 32, 32, 4, 32, 32, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 8768);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 8912);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 8928);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 4096);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3 +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3_ref =
          (void *)((char *)ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3 + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 8192);

  const static char _MERGE_CONVRQ_PASS_3_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_3_L2_prefix[] = "[_MERGE_CONVRQ_PASS_3_L2][SB][297984 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1445888, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1441936, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args);

    _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1314816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_3_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_3_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_3_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args_t){
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN RQADD
  pulp_nn_add_i8_i8_u8(ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
                       ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
                       ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 1, 0, 0, 1, 0, 0, 1048576, 524288, 19, 1, 4096, 1, 1,
                       NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args_t){
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 4096);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 8192);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);

  const static char _MERGE_ADDRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_ADDRQ_PASS_0_L2_prefix[] = "[_MERGE_ADDRQ_PASS_0_L2][SB][4096 ops][Tile ";

  uint32_t _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_kernel_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1445888, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1445888, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args_t){
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_cluster_fork_args);

    _MERGE_ADDRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1314816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_0_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_0_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_ADDRQ_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_0_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_0_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args_t){
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
      1, 19, 32, 32, 4, 16, 16, 4, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 5184);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 5200);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 5216);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 4096);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5 +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5 +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 5120);

  const static char _MERGE_CONVRQ_PASS_4_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_4_L2_prefix[] = "[_MERGE_CONVRQ_PASS_4_L2][SB][8960 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1445888, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args);

    _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311744, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_4_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_4_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_4_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args_t){
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref,
      1, 19, 32, 32, 4, 16, 16, 4, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 5696);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 5840);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 5856);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 4096);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6 +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6_ref =
          (void *)((char *)ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6 + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 5120);

  const static char _MERGE_CONVRQ_PASS_5_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_5_L2_prefix[] = "[_MERGE_CONVRQ_PASS_5_L2][SB][74496 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_5_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_5_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_5_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_5_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_5_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_5_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_5_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1445888, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1441936, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_5_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_5_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_cluster_fork_args);

    _MERGE_CONVRQ_PASS_5_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_5_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311744, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_5_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_5_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_5_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_5_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_5_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_5_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_5_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_5_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_5_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_5_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_5_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_5_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_5_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args_t){
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref,
      1, 19, 16, 16, 4, 16, 16, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2624);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2768);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2784);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1024);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7 +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7_ref =
          (void *)((char *)ImageClassification2_small_4_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7 + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2048);

  const static char _MERGE_CONVRQ_PASS_6_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_6_L2_prefix[] = "[_MERGE_CONVRQ_PASS_6_L2][SB][74496 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_6_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_6_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_6_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_6_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_6_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_6_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_6_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1441936, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7_ref);
    mchan_transfer_1d(
        1441808, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_6_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_6_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_cluster_fork_args);

    _MERGE_CONVRQ_PASS_6_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_6_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311744, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_6_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_6_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_6_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_6_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_6_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_6_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_6_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_6_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_6_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_6_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_6_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_6_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_6_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args_t){
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN RQADD
  pulp_nn_add_i8_i8_u8(ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref,
                       ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref,
                       ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref, 1, 0, 0, 1, 0, 0, 1048576, 524288, 19, 1, 1024, 1, 1,
                       NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args_t){
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1024);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2048);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split + 0);

  const static char _MERGE_ADDRQ_PASS_1_L2_suffix[] = " cycles \n";

  const static char _MERGE_ADDRQ_PASS_1_L2_prefix[] = "[_MERGE_ADDRQ_PASS_1_L2][SB][1024 ops][Tile ";

  uint32_t _MERGE_ADDRQ_PASS_1_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_1_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_1_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_1_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_1_L2_kernel_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_1_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_ADDRQ_PASS_1_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1442816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_ADDRQ_PASS_1_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_1_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args_t){
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_cluster_fork_args);

    _MERGE_ADDRQ_PASS_1_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_1_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311744, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_ADDRQ_PASS_1_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_1_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_ADDRQ_PASS_1_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_1_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_1_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_ADDRQ_PASS_1_L2_kernel_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_1_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_1_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_ADDRQ_PASS_1_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_1_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_1_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args_t){
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref,
      1, 19, 16, 16, 4, 8, 8, 8, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1600);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1632);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1664);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1024);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9_ref =
          (void *)((char *)
                       ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9 +
                   0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1536);

  const static char _MERGE_CONVRQ_PASS_7_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_7_L2_prefix[] = "[_MERGE_CONVRQ_PASS_7_L2][SB][4288 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_7_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_7_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_7_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_7_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_7_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_7_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_7_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1441824, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor_ref);
    mchan_transfer_1d(
        1441824, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9_ref);
    mchan_transfer_1d(
        1441824, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_7_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_7_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_cluster_fork_args);

    _MERGE_CONVRQ_PASS_7_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_7_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_7_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_7_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_7_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_7_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_7_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_7_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_7_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_7_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_7_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_7_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_7_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_7_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_7_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args_t){
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref,
      1, 19, 16, 16, 4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2112);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2400);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2432);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1600);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10_ref =
          (void
               *)((char *)
                      ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10 +
                  0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1024);

  const static char _MERGE_CONVRQ_PASS_8_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_8_L2_prefix[] = "[_MERGE_CONVRQ_PASS_8_L2][SB][37056 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_8_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_8_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_8_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_8_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_8_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_8_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_8_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442816, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1442080, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor_ref);
    mchan_transfer_1d(
        1441824, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10_ref);
    mchan_transfer_1d(
        1441824, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_8_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_8_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_cluster_fork_args);

    _MERGE_CONVRQ_PASS_8_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_8_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_8_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_8_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_8_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_8_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_8_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_8_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_8_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_8_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_8_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_8_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_8_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_8_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_8_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args_t){
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref, ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer, NULL,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref,
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref,
      1, 19, 8, 8, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args;
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer = args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1728);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1152);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2752);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref =
      (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2784);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 2240);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11_ref =
          (void
               *)((char *)
                      ImageClassification2_small_4_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11 +
                  0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor + 0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer = (void *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);

  const static char _MERGE_CONVRQ_PASS_9_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_9_L2_prefix[] = "[_MERGE_CONVRQ_PASS_9_L2][SB][73920 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_9_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_9_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_9_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_9_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_9_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_9_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_9_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442304, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1442368, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor_ref);
    mchan_transfer_1d(
        1441824, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11_ref);
    mchan_transfer_1d(
        1441824, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_9_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_9_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args_t){
                .ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer = ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_cluster_fork_args);

    _MERGE_CONVRQ_PASS_9_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_9_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_9_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_9_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_9_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_9_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_9_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_9_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_9_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_9_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_9_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_9_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_9_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_9_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_9_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr;
} ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3(void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args_t *)ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args;
  uint8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args =
      (ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args_t){
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN RQADD
  pulp_nn_add_i8_i8_u8(ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref,
                       ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref,
                       ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref, 1, 0, 0, 1, 0, 0, 1048576, 524288, 19, 1, 512, 1, 1,
                       NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args_t){
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 512);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 1024);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed +
                   0);
  void *
      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)
                       ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed +
                   0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp + 0);

  const static char _MERGE_ADDRQ_PASS_2_L2_suffix[] = " cycles \n";

  const static char _MERGE_ADDRQ_PASS_2_L2_prefix[] = "[_MERGE_ADDRQ_PASS_2_L2][SB][512 ops][Tile ";

  uint32_t _MERGE_ADDRQ_PASS_2_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_L2_kernel_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_ADDRQ_PASS_2_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442304, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1442304, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_ADDRQ_PASS_2_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_2_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args_t){
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_cluster_fork_args);

    _MERGE_ADDRQ_PASS_2_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_2_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_ADDRQ_PASS_2_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_2_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_ADDRQ_PASS_2_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_2_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_2_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_2_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_ADDRQ_PASS_2_L2_kernel_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_2_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_2_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_2_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_ADDRQ_PASS_2_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_2_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_2_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args_t){
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
              ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
          .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
              ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp,
          .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr =
              ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure(
    void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args_t *)
          ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 8, 8] -> [1, 8, 8, 8] (Name: _MERGE_ADDRQ_PASS_2_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / NUM_CORES;
    const uint32_t leftover = dimLen_3 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((uint8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref)[i_0][i_3][i_1][i_2] =
              ((uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                   ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void
ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args_t *)
          ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args_t){
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args_t *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 512);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor + 0);

  const static char _MERGE_ADDRQ_PASS_2_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_ADDRQ_PASS_2_transpose_L2_prefix[] = "[_MERGE_ADDRQ_PASS_2_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_ADDRQ_PASS_2_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_ADDRQ_PASS_2_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_ADDRQ_PASS_2_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442304, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_ADDRQ_PASS_2_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_2_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args_t){
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_cluster_fork_args);

    _MERGE_ADDRQ_PASS_2_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_2_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_ADDRQ_PASS_2_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_2_transpose_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_ADDRQ_PASS_2_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_ADDRQ_PASS_2_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_2_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_2_transpose_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_ADDRQ_PASS_2_transpose_L2_kernel_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_2_transpose_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_2_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_2_transpose_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_ADDRQ_PASS_2_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_ADDRQ_PASS_2_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_2_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr;
} ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void
ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3(void *ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args_t *)
          ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp,
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref;
} ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void
ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure(void *ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args_t *args =
      (ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args_t *)ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref;

  // CLOSURE FUNCTION CALL

  // ReduceMean (Name: avgpoolGlobalAveragePool, Op: ReduceMean)
  BEGIN_SINGLE_CORE
  int32_t ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator = 0;

  uint8_t *dummy_ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref =
      ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref;

  for (uint32_t i_0 = 0; i_0 < 1; i_0++) {
    for (uint32_t i_1 = 0; i_1 < 8; i_1++) {
      ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator = 128 * 64;
      for (uint32_t i_2 = 0; i_2 < 8; i_2++) {
        for (uint32_t i_3 = 0; i_3 < 8; i_3++) {
          ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator +=
              ((uint8_t (*)[8][8][8])ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
      int8_t ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_sgn = 0;
      ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_sgn =
          -(ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator < 0) +
          (ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator >= 0);
      *dummy_ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref++ =
          (uint8_t)((ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator +
                     ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_sgn * (64 >> 1)) /
                        64 +
                    0);
    }
  }
  END_SINGLE_CORE

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr;
} ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4_avgpoolGlobalAveragePool_closure(void *ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args_t *args =
      (ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args_t *)ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor = args->ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 512);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_avgpoolGlobalAveragePool_output_0_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor + 0);

  const static char avgpoolGlobalAveragePool_L2_suffix[] = " cycles \n";

  const static char avgpoolGlobalAveragePool_L2_prefix[] = "[avgpoolGlobalAveragePool_L2][SB][0 ops][Tile ";

  uint32_t avgpoolGlobalAveragePool_L2_egress_dma_wait_end_measurements[1];

  uint32_t avgpoolGlobalAveragePool_L2_egress_dma_wait_start_measurements[1];

  uint32_t avgpoolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[1];

  uint32_t avgpoolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[1];

  uint32_t avgpoolGlobalAveragePool_L2_kernel_end_measurements[1];

  uint32_t avgpoolGlobalAveragePool_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr) + 1];
       TILING_I++) {

    avgpoolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442304, ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref,
        ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    avgpoolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    avgpoolGlobalAveragePool_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args_t
        ImageClassification2_small_4_ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args =
            (ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args_t){
                .ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref};

    // ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure CLOSURE CALL
    ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure(
        &ImageClassification2_small_4_ImageClassification2_small_4_avgpoolGlobalAveragePool_tiling_closure_args);

    avgpoolGlobalAveragePool_L2_kernel_end_measurements[TILING_I] = getCycles();

    avgpoolGlobalAveragePool_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310728, ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_avgpoolGlobalAveragePool_output_0_tensor_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    avgpoolGlobalAveragePool_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", avgpoolGlobalAveragePool_L2_prefix, PROFILING_I, "Input DMA took ",
           avgpoolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               avgpoolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           avgpoolGlobalAveragePool_L2_suffix);

    printf("%s%u] %s%u%s", avgpoolGlobalAveragePool_L2_prefix, PROFILING_I, "Kernel took ",
           avgpoolGlobalAveragePool_L2_kernel_end_measurements[PROFILING_I] - avgpoolGlobalAveragePool_L2_kernel_start_measurements[PROFILING_I],
           avgpoolGlobalAveragePool_L2_suffix);

    printf("%s%u] %s%u%s", avgpoolGlobalAveragePool_L2_prefix, PROFILING_I, "Output DMA took ",
           avgpoolGlobalAveragePool_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               avgpoolGlobalAveragePool_L2_egress_dma_wait_start_measurements[PROFILING_I],
           avgpoolGlobalAveragePool_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr;
} ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3(void *ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args_t *args =
      (ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args_t *)ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args;
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor = args->ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args =
          (ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor = ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor,
              .ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr};

  // ImageClassification2_small_4_avgpoolGlobalAveragePool_closure CLOSURE CALL
  ImageClassification2_small_4_avgpoolGlobalAveragePool_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
} ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure(void *ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t *)ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *
      ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref =
          ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *
      ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref =
          ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int8_t *
      ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
          ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
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
          ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
          NULL,
          ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref,
          ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
          ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
          ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref, 1, 0, 8, 10, 1, 1, NUM_CORES);
      ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref +=
          8;
      ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref +=
          10;
    }
    ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref +=
        8 * 10;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
} ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork(void *ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t *args =
      (ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t *)ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args =
          (ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t){
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref};

  // ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4_flattenFlatten_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;
} ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure(void *ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t *args =
      (ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t *)ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args;
  uint8_t *ImageClassification2_small_4_flattenFlatten_output_0_tensor = args->ImageClassification2_small_4_flattenFlatten_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref = (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 170);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref = (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 10);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref = (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 130);
  int32_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref = (int32_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 90);
  int8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L1 + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_flattenFlatten_output_0_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_flattenFlatten_output_0_tensor + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref =
      (void *)((char *)ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref =
      (void *)((char *)ImageClassification2_small_4_RequantShift_final_mul_tensor + 0);
  void *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref = (void *)((char *)ImageClassification2_small_4_output_0 + 0);

  const static char _MERGE_GEMMRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_GEMMRQ_PASS_0_L2_prefix[] = "[_MERGE_GEMMRQ_PASS_0_L2][SB][214 ops][Tile ";

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_kernel_end_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t ImageClassification2_small_4_channel_output = (uint32_t)-1;
  uint32_t ImageClassification2_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles
           [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr];
       TILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles
                      [(*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    ImageClassification2_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441800, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_flattenFlatten_output_0_tensor_ref);
    mchan_transfer_1d(1441872, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref);
    mchan_transfer_1d(1441832, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref);
    mchan_transfer_1d(1441832, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification2_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_input);
      mchan_channel_free(ImageClassification2_small_4_channel_input);
    }

    _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMMRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t
        ImageClassification2_small_4_ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args =
            (ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t){
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
                .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
                    ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork,
                    &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args);

    _MERGE_GEMMRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    ImageClassification2_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310730, ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref,
                      ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref);

    // Wait for output tiles

    if (ImageClassification2_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification2_small_4_channel_output);
      mchan_channel_free(ImageClassification2_small_4_channel_output);
    }

    _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr > 0)
                              ? ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[(
                                    *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles
                         [*ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_GEMMRQ_PASS_0_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMMRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMMRQ_PASS_0_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_GEMMRQ_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_GEMMRQ_PASS_0_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_GEMMRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMMRQ_PASS_0_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMMRQ_PASS_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *ImageClassification2_small_4_flattenFlatten_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;
} ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3(void *ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t *args =
      (ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t *)ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args;
  uint8_t *ImageClassification2_small_4_flattenFlatten_output_0_tensor = args->ImageClassification2_small_4_flattenFlatten_output_0_tensor;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      args->ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t ImageClassification2_small_4_ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args =
      (ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t){.ImageClassification2_small_4_flattenFlatten_output_0_tensor =
                                                                             ImageClassification2_small_4_flattenFlatten_output_0_tensor,
                                                                         .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
                                                                             ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure CLOSURE CALL
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {
  int8_t *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;                                      // L2_init_template
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split; // L2_init_template
  int8_t
      *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *
      ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed; // L2_init_template
  int8_t
      *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;                              // L2_init_template
  int8_t
      *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *
      ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed; // L2_init_template
  int8_t
      *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;                              // L2_init_template
  int8_t
      *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *
      ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed; // L2_init_template
  int8_t
      *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;                                 // L2_init_template
  uint8_t *ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;                                    // L2_init_template
  uint8_t *ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor;                                                            // L2_init_template
  uint8_t *ImageClassification2_small_4_flattenFlatten_output_0_tensor;                                                                      // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_buffer;                                                                            // L2_init_template
  void *ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_buffer;                                                                            // L2_init_template

  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr;
  uint8_t bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      &bu_ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;
  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 0);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_0 CALL START

  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 4096);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_0 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_1 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 0);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_1 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_2 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 20480);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_2 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_3 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 8192);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_3 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_ADDRQ_PASS_0 CALL START

  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 4096);
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_0_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_ADDRQ_PASS_0 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_4 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 0);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_4 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_5 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 1024);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_5_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_5 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_6 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 2048);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_6_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_6 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_ADDRQ_PASS_1 CALL START

  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 1024);
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_1_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_ADDRQ_PASS_1 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_7 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 0);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_7_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_7 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_8 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 512);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_8_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_8 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_9 CALL START

  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 1032);
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args =
          (ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_CONVRQ_PASS_9_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_CONVRQ_PASS_9 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_ADDRQ_PASS_2 CALL START

  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 520);
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args_t){
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
                  ImageClassification2_small_4__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_ADDRQ_PASS_2 CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose CALL START

  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 8);
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args =
          (ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp,
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3(
      &ImageClassification2_small_4_ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_ADDRQ_PASS_2_transpose CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4_avgpoolGlobalAveragePool CALL START

  ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor = (uint8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 0);
  ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args =
          (ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args_t){
              .ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  ImageClassification2_small_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor = ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor,
              .ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr};

  // ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3 CLOSURE CALL
  ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3(
      &ImageClassification2_small_4_ImageClassification2_small_4_avgpoolGlobalAveragePool_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4_avgpoolGlobalAveragePool CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4_flattenFlatten CALL START

  // Reshape (Name: flattenFlatten, Op: Flatten)
  ImageClassification2_small_4_flattenFlatten_output_0_tensor = ImageClassification2_small_4_avgpoolGlobalAveragePool_output_0_tensor;

  // RUNNETRWORK LAYER ImageClassification2_small_4_flattenFlatten CALL END

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0 CALL START

  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t
      ImageClassification2_small_4_ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args =
          (ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t){
              .ImageClassification2_small_4_flattenFlatten_output_0_tensor = ImageClassification2_small_4_flattenFlatten_output_0_tensor,
              .ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
                  ImageClassification2_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr};

  // ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3 CLOSURE CALL
  ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3(&ImageClassification2_small_4_ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args);

  // RUNNETRWORK LAYER ImageClassification2_small_4__MERGE_GEMMRQ_PASS_0 CALL END
}

void InitNetwork() {

  ImageClassification2_small_4_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 23392);

  ImageClassification2_small_4_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 24576);

  ImageClassification2_small_4_input_0 = (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 3072);
  ImageClassification2_small_4_output_0 = (int8_t *)((char *)ImageClassification2_small_4_MEMORYARENA_L2 + 8);
  ImageClassification2_small_4_inputs[0] = (void *)ImageClassification2_small_4_input_0;
  ImageClassification2_small_4_outputs[0] = (void *)ImageClassification2_small_4_output_0;
}
