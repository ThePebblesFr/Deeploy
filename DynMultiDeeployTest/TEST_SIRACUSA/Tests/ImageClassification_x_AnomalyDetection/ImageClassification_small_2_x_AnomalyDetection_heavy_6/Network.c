
// ===== GLOBALS from ImageClassification_small_2 START =====

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

int8_t *ImageClassification_small_2_MEMORYARENA_L1; // L2_init_template
int8_t *ImageClassification_small_2_MEMORYARENA_L2; // L2_init_template
int8_t *ImageClassification_small_2_input_0;        // L2_init_template
int8_t *ImageClassification_small_2_output_0;       // L2_init_template

static PI_L2 int8_t ImageClassification_small_2_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor[432] = {
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

static PI_L2 int32_t ImageClassification_small_2_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor[16] = {
    17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0, 17456.0};

static PI_L2 int32_t ImageClassification_small_2_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor[16] = {
    524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0,
    524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor[64] = {
    -44, 103, -41, 127, 77,  -4, 60,  95,  62,   -118, 4,    -105, -9,  108, 69,  -67,  -124, 44,   -90, 3,  -38, -88,
    -57, 75,  127, -10, -43, 11, 7,   -84, -113, 98,   99,   -61,  6,   0,   15,  108,  -117, -126, -96, 28, 38,  -64,
    127, 127, 127, -98, 127, 77, -36, -54, -45,  -55,  -127, -94,  -40, -6,  -49, -103, -40,  -2,   35,  -4};

static PI_L2 int32_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1[4] =
    {2056.0, 2056.0, 2056.0, 2056.0};

static PI_L2 int32_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1[4] =
    {524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor[576] = {
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

static PI_L2 int32_t ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2[4] = {
    4112.0, 4112.0, 4112.0, 4112.0};

static PI_L2 int32_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2[4] =
    {524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor[144] = {
    3,  43,  -10, -45, -23, 24,  26,  10,  12,  15, 7,   -43, 46,  -5,  12,  58,  -26, -31, 37,  -1, 16,  -3,  37,  -8, 11, -27, 22,  43,  8,
    8,  -1,  -27, 36,  3,   -44, -23, -12, -33, 48, 104, 34,  -23, -31, -33, 8,   -6,  -22, -13, -3, 1,   -11, 0,   24, -6, 5,   17,  -33, 37,
    -2, 4,   30,  21,  34,  -26, -35, -56, 2,   37, 29,  6,   -1,  10,  -3,  -2,  -13, -32, 75,  16, -17, -30, 18,  36, 54, -29, 5,   66,  -22,
    88, -39, 54,  13,  44,  -18, 8,   -17, -16, 20, -73, 4,   -28, -7,  -10, -9,  -22, -42, -40, 26, 5,   -7,  -19, 7,  33, -22, -26, 45,  -15,
    38, -46, 17,  -25, 31,  -23, 11,  -6,  31,  18, 26,  69,  20,  23,  62,  -23, 27,  17,  12,  -8, 25,  40,  15,  48, 6,  52,  0,   -8};

static PI_L2 int32_t ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3[4] = {
    0.0, 0.0, 0.0, 0.0};

static PI_L2 int32_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3[4] =
    {524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor[16] = {
    -8, 96, 127, 80, -21, 50, -5, -47, -127, 48, -62, -7, -24, 117, -127, -28};

static PI_L2 int32_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5[4] =
    {2056.0, 2056.0, 2056.0, 2056.0};

static PI_L2 int32_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5[4] =
    {524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor[144] = {
    4,   6,   -23, 2,   -44, -34, 3,   -20, -51, 19,  -22, -16, -16, -25, 29, 25,  17,  -14, 14,  69,  -40, 10,  34,  -53, 38, 22,  16,  -29, -61,
    -18, -34, 20,  43,  17,  -22, 31,  29,  -3,  34,  -5,  1,   -18, 8,   1,  -32, -38, -13, 3,   -22, -8,  -27, 21,  74,  -8, -5,  30,  -57, 33,
    -20, -22, 10,  6,   -3,  -38, -44, 16,  48,  -18, -14, -29, -46, 42,  8,  -28, -21, -11, -13, -70, -10, -57, -35, -23, 14, 2,   43,  25,  6,
    -14, 0,   37,  -15, -36, 0,   -1,  -16, -10, -29, 35,  1,   1,   -8,  37, -45, -1,  24,  -9,  16,  36,  52,  20,  -14, -7, 62,  -31, 20,  -8,
    -7,  -15, 17,  -18, -40, -34, 38,  -1,  -15, -4,  -6,  -8,  -20, -19, 42, 44,  15,  13,  -24, 1,   -43, 10,  11,  -16, 5,  -25, -8,  -14};

static PI_L2 int32_t ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6[4] = {
    4112.0, 4112.0, 4112.0, 4112.0};

static PI_L2 int32_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6[4] =
    {524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor[144] = {
    87,  5,  -5,  -5,  -35, 22, -38, 32,  58,  -41, 18,  40,  -34, 36,  -10, 10,  -47, 1,   -15, 5,   3,   60,  -11, 9,   -27, -3,  52,  -35, -11,
    -34, 6,  -31, -29, 28,  54, -14, -26, -16, 60,  -47, 17,  3,   -23, 20,  -26, 26,  -36, -13, 24,  -31, -3,  -30, 20,  18,  10,  12,  -8,  31,
    -13, 2,  8,   -23, -23, 18, -40, 12,  -22, -41, -24, 13,  46,  34,  9,   3,   -36, 19,  32,  52,  12,  -23, -1,  -18, -7,  -20, 6,   7,   -15,
    -11, -6, 46,  -14, 14,  -5, -34, -16, 12,  -25, -21, -31, -57, 36,  41,  8,   8,   -10, 46,  6,   43,  21,  -17, -9,  -58, 45,  -14, 10,  -46,
    -33, 4,  -3,  -5,  -22, 12, 3,   -19, 31,  -58, -44, -37, -55, -23, 16,  -37, 9,   -4,  23,  -45, 21,  -10, -6,  -16, 22,  4,   18,  38};

static PI_L2 int32_t ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7[4] = {
    0.0, 0.0, 0.0, 0.0};

static PI_L2 int32_t ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7[4] =
    {524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor[32] = {
    -103, -57, 90, 58, 33, -53, -66, 36, -60, 67, -72, -127, 82, 15, 21, 85, -54, -44, -13, -49, 44, 50, -70, -127, -63, -38, -45, -29, 36, 127, 68, -9};

static PI_L2 int32_t ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor[8] = {
    2056.0, 2056.0, 2056.0, 2056.0, 2056.0, 2056.0, 2056.0, 2056.0};

static PI_L2 int32_t ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9[8] =
    {524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor[288] = {
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

static PI_L2 int32_t ImageClassification_small_2_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor[8] = {4112.0, 4112.0, 4112.0, 4112.0,
                                                                                                                             4112.0, 4112.0, 4112.0, 4112.0};

static PI_L2 int32_t
    ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10[8] = {
        524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor[576] = {
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

static PI_L2 int32_t ImageClassification_small_2_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor[8] = {0.0, 0.0, 0.0, 0.0,
                                                                                                                           0.0, 0.0, 0.0, 0.0};

static PI_L2 int32_t
    ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11[8] = {
        524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0, 524288.0};

static PI_L2 int8_t ImageClassification_small_2__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor[80] = {
    14,  120, -43, -86, -23, -25, -72, 85,  -11, -67, 9,   37,  -13, 23,  -7,  15,  -21, -94, -18, 121, -31, 23,  -127, 22, 12, -57, -34,
    11,  -65, 77,  -56, -34, 50,  42,  16,  99,  77,  -72, 26,  38,  -37, -50, -61, 82,  -63, 23,  -27, -16, -11, -11,  96, 39, 67,  2,
    -52, 13,  72,  -46, 118, 18,  -84, -11, 2,   17,  24,  -46, -39, 52,  -43, -11, -20, -88, -49, -22, 96,  -6,  -31,  30, 40, 42};

static PI_L2 int32_t ImageClassification_small_2__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor[10] = {5449.5,  4205.5,  8333.5,  -517.5,   9507.5,
                                                                                                            -4668.5, -5293.5, -4244.5, -10784.5, -7546.5};

static PI_L2 int32_t ImageClassification_small_2_RequantShift_final_mul_tensor[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_y[2] = {25, 9};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_y[2] = {24, 8};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_left[2] = {1, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_right[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 2};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[2] = {1968480, 1966944};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size_1d[2] = {75, 27};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[2] = {1847296, 1839104};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size_1d[2] = {384, 128};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x[8] = {30, 2, 30, 2, 30, 2, 30, 2};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_y[8] = {29, 29, 3, 3, 29, 29, 3, 3};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x[8] = {30, 2, 30, 2, 30, 2, 30, 2};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_y[8] = {29, 29, 3, 3, 29, 29, 3, 3};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[2] = {0, 8};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cmd[8] = {1980000, 1967008, 1967520, 1966176,
                                                                                                           1980000, 1967008, 1967520, 1966176};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_size_1d[8] = {464, 464, 48, 48, 464, 464, 48, 48};

static PI_L1 int16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_relativeOffset[8] = {15360, -14896, 15360, -15824,
                                                                                                                     15360, -14896, 15360, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_relativeOffset[8] = {0, 0, 0, 32, 0, 0, 0, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_relativeOffset[8] = {0, 0, 0, 8, 0, 0, 0, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_relativeOffset[8] = {0, 0, 0, 8, 0, 0, 0, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_end_0[8] = {30, 2, 30, 2, 30, 2, 30, 2};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_stride_0[8] = {58, 58, 6, 6, 58, 58, 6, 6};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cmd[8] = {1835066, 1835066, 1835014, 1835014,
                                                                                                            1835066, 1835066, 1835014, 1835014};

static PI_L1 int16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_relativeOffset[8] = {3840, -3724, 3840, -3954,
                                                                                                                      3840, -3724, 3840, 0};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_x[8] = {30, 4, 30, 4, 30, 4, 30, 4};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_y[8] = {22, 22, 12, 12, 22, 22, 12, 12};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_x[8] = {29, 3, 29, 3, 29, 3, 29, 3};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_y[8] = {21, 21, 11, 11, 21, 21, 11, 11};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_ch_im_out[8] = {3, 3, 3, 3, 1, 1, 1, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_top[8] = {1, 0, 1, 0, 1, 0, 1, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_bottom[8] = {0, 1, 0, 1, 0, 1, 0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_left[8] = {1, 1, 0, 0, 1, 1, 0, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_right[8] = {0, 0, 1, 1, 0, 0, 1, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[2] = {0, 8};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_cmd[8] = {1976640, 1967488, 1971840, 1966848,
                                                                                                           1976640, 1967488, 1971840, 1966848};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_size_1d[8] = {352, 352, 192, 192, 352, 352, 192, 192};

static PI_L1 int16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_relativeOffset[8] = {14336, -14016, 14336, -14656,
                                                                                                                     14336, -14016, 14336, 0};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_cmd[8] = {1442224, 1442224, 1442224, 1442224,
                                                                                                          1441936, 1441936, 1441936, 1441936};

static PI_L1 uint16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_relativeOffset[8] = {0, 0, 0, 432, 0, 0, 0, 0};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_cmd[8] = {1441804, 1441804, 1441804, 1441804,
                                                                                                       1441796, 1441796, 1441796, 1441796};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_relativeOffset[8] = {0, 0, 0, 12, 0, 0, 0, 0};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_cmd[8] = {1441804, 1441804, 1441804, 1441804,
                                                                                                       1441796, 1441796, 1441796, 1441796};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_relativeOffset[8] = {0, 0, 0, 12, 0, 0, 0, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_end_0[8] = {29, 3, 29, 3, 29, 3, 29, 3};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_stride_0[8] = {63, 63, 33, 33, 21, 21, 11, 11};

static PI_L1 uint32_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_cmd[8] = {1835071, 1835071, 1835041, 1835041,
                                                                                                            1835029, 1835029, 1835019, 1835019};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_size_1d[8] = {3, 3, 3, 3, 1, 1, 1, 1};

static PI_L1 int16_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_relativeOffset[8] = {3712, -3628, 3712, -3793,
                                                                                                                      3712, -3628, 3712, 0};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_numTiles[2] = {0, 1};

static PI_L1 uint8_t ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[2] = {0, 1};

void *ImageClassification_small_2_inputs[1];
void *ImageClassification_small_2_outputs[1];
extern struct pi_device cluster_dev;

// ===== GLOBALS from ImageClassification_small_2 END =====


// ===== GLOBALS from AnomalyDetection_heavy_6 START =====



int8_t *AnomalyDetection_heavy_6_MEMORYARENA_L1; // L2_init_template
int8_t *AnomalyDetection_heavy_6_MEMORYARENA_L2; // L2_init_template
int8_t *AnomalyDetection_heavy_6_input_0;        // L2_init_template
int8_t *AnomalyDetection_heavy_6_output_0;       // L2_init_template

static PI_L2 int8_t AnomalyDetection_heavy_6_layer0_weight_tensor[81920] = {
    0,  3,  2,  0,  1,  2,  2,  0,  -1, -1, 1,  0,  -1, 2,   3,  1,  1,  4,  3,  2,  1,  -1, -2, -1, -3, 0,   -3, 2,   -1, 1,  1,  0,  3,  1,  0,  2,  2,  -5,
    2,  3,  2,  2,  1,  3,  3,  0,  2,  1,  1,  0,  1,  -1,  0,  4,  -1, -1, 0,  2,  2,  2,  1,  1,  1,  3,   2,  1,   -2, -2, -2, 1,  0,  2,  3,  1,  2,  1,
    1,  1,  -2, 1,  2,  0,  0,  0,  0,  1,  5,  1,  0,  -2,  -1, -1, -1, 2,  1,  1,  0,  -1, -1, 1,  5,  -1,  0,  1,   1,  2,  3,  -1, 4,  -1, -2, -3, 0,  1,
    -1, -4, -3, -1, -3, -2, 0,  -3, 0,  -2, -1, 0,  -1, -1,  -1, 4,  4,  2,  2,  2,  0,  0,  -4, -1, 2,  2,   -1, 2,   4,  -2, -1, 4,  3,  1,  3,  0,  -1, -1,
    -3, -1, 0,  2,  0,  0,  -1, 0,  3,  2,  2,  3,  0,  -4,  0,  4,  4,  0,  2,  -2, 2,  0,  0,  2,  0,  1,   1,  -2,  2,  1,  0,  -1, 1,  1,  3,  3,  0,  2,
    2,  3,  1,  0,  -2, -2, -1, 0,  0,  3,  -1, 0,  5,  1,   0,  2,  -1, 0,  1,  1,  3,  3,  -1, 2,  3,  3,   1,  -1,  -2, -1, -1, 4,  3,  1,  -1, 2,  -2, 0,
    3,  0,  -3, 4,  1,  4,  4,  1,  5,  0,  -3, -3, 0,  0,   -1, -4, -3, 0,  -3, -2, -2, -1, 0,  1,  1,  0,   -1, 1,   1,  2,  3,  3,  1,  -1, -1, 2,  -4, -1,
    0,  1,  -2, 3,  1,  0,  -1, 4,  2,  1,  4,  0,  -1, 0,   -1, 1,  -3, 1,  1,  3,  -1, 2,  2,  1,  3,  0,   -1, -5,  -1, 3,  3,  0,  -1, 0,  2,  -1, 0,  2,
    1,  1,  2,  -1, -1, 2,  0,  -3, 0,  2,  3,  2,  1,  3,   0,  3,  1,  -1, -3, -1, 0,  0,  0,  2,  2,  0,   3,  1,   -1, 0,  -1, -1, 1,  1,  -1, 3,  0,  -1,
    3,  4,  2,  1,  0,  0,  -1, 1,  2,  1,  0,  3,  -4, 1,   3,  -2, -3, 3,  0,  4,  4,  -1, 7,  2,  -2, -2,  1,  4,   -1, -3, -3, 1,  -3, -2, -3, -3, 2,  -1,
    2,  1,  1,  2,  0,  5,  0,  1,  2,  -2, 0,  0,  -1, -1,  -1, 0,  -4, 2,  -1, -1, 0,  4,  2,  0,  3,  0,   -2, 1,   0,  -1, -3, 4,  2,  1,  1,  3,  0,  -1,
    2,  1,  0,  -5, -1, 1,  1,  -1, 1,  0,  3,  0,  1,  3,   4,  0,  2,  -1, -1, 1,  -1, -1, 1,  3,  3,  3,   1,  2,   1,  3,  0,  -2, -2, -2, 0,  0,  1,  4,
    1,  1,  5,  1,  0,  2,  0,  1,  1,  1,  -2, 1,  1,  3,   5,  2,  3,  0,  -1, 1,  1,  -1, 2,  1,  2,  3,   -3, -1,  4,  1,  -4, 1,  1,  5,  2,  0,  3,  4,
    -1, -2, 0,  2,  -2, -3, -4, -1, -5, -3, -2, 1,  3,  -1,  -2, -1, 1,  -2, 1,  2,  2,  1,  3,  -2, 2,  0,   0,  0,   -1, 1,  -2, 2,  2,  -2, -2, 2,  3,  1,
    3,  -2, 0,  0,  -1, 0,  -1, 2,  -3, 2,  2,  2,  5,  -2,  0,  0,  0,  -4, -3, 1,  5,  0,  0,  1,  3,  2,   1,  2,   2,  0,  1,  -1, -2, 1,  0,  -3, 1,  0,
    3,  2,  -1, 1,  1,  2,  1,  1,  -2, -1, 0,  0,  1,  2,   1,  2,  4,  -1, 1,  2,  0,  3,  -2, 2,  -1, 2,   1,  1,   5,  2,  2,  0,  1,  0,  -1, 0,  0,  2,
    0,  3,  2,  1,  4,  0,  -3, 3,  0,  3,  3,  -2, 1,  4,   -1, -3, 0,  0,  -3, -2, -4, -2, -4, -3, -2, 1,   2,  2,   -1, 1,  0,  -2, -2, -1, 1,  0,  -1, -3,
    3,  -1, -1, 1,  0,  -1, -1, 1,  -1, 1,  -1, -1, 1,  -1,  0,  -4, 3,  3,  2,  -1, 1,  -1, 1,  1,  3,  3,   1,  -1,  0,  3,  2,  1,  -4, 6,  9,  2,  0,  5,
    -2, 2,  3,  -4, -3, 5,  3,  1,  2,  0,  3,  -1, 1,  -3,  -3, 1,  2,  1,  1,  6,  5,  0,  3,  0,  -1, 3,   1,  5,   -1, -2, 0,  3,  -2, -5, -3, 0,  0,  4,
    5,  2,  -1, 2,  2,  0,  -3, -1, -2, 0,  3,  2,  0,  0,   -1, 1,  3,  2,  1,  2,  2,  3,  -4, 1,  0,  0,   0,  3,   2,  -2, -2, 0,  2,  4,  2,  2,  1,  0,
    -1, -3, 3,  -3, 2,  6,  2,  -2, 0,  2,  2,  1,  1,  -2,  0,  -1, 1,  2,  0,  -2, 2,  1,  1,  0,  1,  -4,  -1, 0,   0,  -2, 3,  4,  2,  -3, 0,  -2, 3,  3,
    1,  3,  2,  3,  0,  0,  5,  -1, -2, 7,  8,  0,  -1, 1,   -2, 3,  3,  -2, -3, 4,  0,  1,  2,  -3, 2,  -1,  3,  -3,  0,  3,  1,  1,  2,  2,  2,  0,  0,  2,
    1,  3,  0,  3,  0,  -1, -1, 2,  -3, -3, -4, -1, 3,  2,   1,  2,  -1, 2,  2,  -1, -2, -1, -3, 2,  -1, 2,   4,  0,   -3, 1,  3,  5,  1,  3,  2,  2,  -3, -1,
    0,  1,  1,  0,  3,  -3, -2, 0,  1,  2,  2,  2,  2,  -1,  -2, -1, 0,  -1, 3,  2,  2,  -3, 0,  -1, 1,  1,   3,  -1,  0,  -3, 2,  3,  1,  0,  -1, -2, -2, -1,
    -3, -1, -2, 0,  1,  0,  2,  2,  1,  0,  2,  -1, 4,  1,   0,  2,  1,  -2, -2, 1,  5,  1,  2,  5,  5,  -1,  2,  2,   -4, 3,  2,  -2, -1, 5,  0,  0,  2,  -3,
    -1, 0,  0,  0,  0,  1,  0,  1,  2,  4,  3,  1,  -1, 3,   0,  4,  2,  4,  0,  0,  3,  1,  -5, -3, -3, 0,   1,  3,   6,  2,  4,  0,  0,  1,  -4, -2, -3, 1,
    2,  1,  3,  1,  0,  2,  1,  4,  2,  1,  5,  0,  -4, 0,   0,  1,  -1, -1, 1,  -3, 0,  -2, -1, 1,  2,  1,   1,  2,   -1, 0,  1,  1,  3,  3,  0,  -4, 2,  -2,
    1,  1,  1,  -3, -1, -2, 2,  1,  2,  2,  -2, -4, -1, 0,   -3, -3, 0,  -2, 1,  -1, 3,  3,  1,  -2, 2,  2,   2,  -1,  5,  1,  2,  -1, -1, 2,  2,  2,  0,  4,
    4,  2,  4,  2,  -2, 2,  1,  2,  -3, 4,  4,  0,  4,  -1,  -2, -2, -1, -1, -1, 2,  -1, 1,  0,  2,  2,  2,   1,  3,   -2, 4,  1,  2,  1,  -1, 2,  0,  -3, -2,
    -1, 4,  -2, 3,  4,  3,  3,  1,  0,  -2, -2, 0,  -2, -1,  2,  1,  0,  1,  -3, -1, 2,  2,  0,  -2, 4,  2,   0,  2,   0,  0,  0,  -3, 4,  -1, -1, -2, -1, -2,
    2,  1,  2,  0,  -3, -3, 0,  3,  2,  1,  -1, 0,  1,  -2,  2,  1,  1,  -5, 2,  0,  1,  2,  1,  2,  -3, -5,  -2, -1,  -5, -4, -1, -2, 2,  1,  2,  2,  2,  -2,
    0,  0,  4,  -1, -1, 2,  2,  -1, 3,  0,  4,  2,  0,  8,   7,  0,  0,  0,  0,  2,  -1, 1,  -2, 4,  5,  1,   1,  0,   1,  -1, -1, 2,  1,  2,  -1, 1,  0,  4,
    2,  2,  5,  4,  0,  2,  3,  5,  0,  -3, 3,  1,  -1, -2,  -1, 0,  0,  2,  6,  4,  3,  5,  3,  0,  -4, 1,   -2, 1,   4,  0,  -1, 0,  0,  -1, 0,  0,  -3, 0,
    5,  4,  1,  1,  -1, 0,  4,  0,  1,  -4, -3, -2, -1, -2,  1,  1,  1,  0,  -3, -2, 0,  2,  3,  2,  -2, -1,  1,  3,   4,  2,  5,  0,  1,  -1, 1,  4,  -1, 1,
    2,  1,  2,  -1, 3,  1,  2,  2,  3,  1,  0,  -3, 0,  -2,  -2, -1, 0,  0,  -1, -1, -2, -5, 0,  2,  -2, 0,   -1, 0,   2,  2,  0,  1,  0,  -1, 3,  2,  1,  5,
    -2, -1, 1,  1,  -4, 0,  2,  -2, 0,  2,  2,  -1, 0,  0,   0,  -2, 3,  3,  0,  2,  0,  0,  -1, 0,  4,  4,   -1, 2,   -2, -4, 2,  3,  -3, -2, -1, -3, 1,  -2,
    -2, 0,  -1, 0,  1,  -1, 0,  3,  1,  -1, 0,  1,  1,  -1,  -2, 0,  3,  2,  2,  -3, 0,  0,  0,  -1, -1, 3,   2,  0,   0,  1,  0,  1,  -1, -2, 0,  3,  -1, -1,
    2,  -1, 1,  1,  3,  0,  4,  3,  2,  0,  -1, 2,  -1, -1,  2,  1,  3,  -2, 4,  0,  1,  0,  4,  1,  3,  -3,  0,  -3,  -1, 1,  -1, 0,  -4, 0,  0,  -4, 2,  1,
    -2, 2,  1,  1,  4,  0,  0,  3,  1,  -1, 1,  1,  1,  5,   -1, 0,  0,  0,  -3, 0,  3,  -2, 0,  1,  1,  0,   0,  0,   0,  -2, 2,  5,  0,  2,  3,  1,  -2, 1,
    3,  4,  0,  0,  0,  0,  2,  0,  -3, -2, -2, -2, -1, 1,   -3, -1, -2, 1,  1,  -1, 3,  2,  1,  -1, 0,  2,   1,  1,   -2, 1,  3,  3,  0,  0,  -1, 3,  1,  -1,
    -1, 3,  2,  0,  0,  1,  2,  3,  1,  0,  1,  3,  -1, -1,  3,  0,  0,  -1, 3,  0,  3,  2,  3,  0,  -1, 1,   1,  -2,  0,  2,  2,  0,  4,  -1, 0,  3,  2,  3,
    3,  -2, -3, -5, 2,  1,  1,  -1, -2, 0,  -1, -4, -2, 2,   0,  0,  3,  1,  1,  0,  -3, 0,  2,  0,  1,  3,   2,  2,   1,  0,  -1, 1,  0,  -3, 3,  -2, -1, -1,
    0,  0,  0,  3,  0,  -2, 0,  1,  1,  3,  2,  0,  0,  0,   2,  3,  0,  2,  1,  1,  2,  1,  0,  -1, -4, -2,  1,  1,   -1, 0,  -1, -2, 3,  1,  3,  1,  -1, -1,
    -1, 1,  1,  0,  -1, 4,  3,  1,  0,  0,  1,  0,  -1, -2,  -1, 0,  1,  0,  1,  0,  1,  1,  0,  -2, 1,  2,   0,  -1,  5,  1,  -2, 2,  4,  1,  3,  4,  3,  0,
    -1, 1,  -1, -3, 0,  0,  3,  0,  3,  0,  0,  1,  2,  2,   3,  -1, -3, -5, -1, 1,  2,  0,  -4, 0,  0,  -3,  0,  1,   2,  4,  3,  1,  5,  1,  -1, 2,  0,  0,
    1,  3,  0,  2,  2,  -1, 1,  0,  0,  -4, 3,  1,  -1, 1,   1,  2,  0,  2,  1,  -4, 0,  2,  1,  3,  2,  0,   -1, 1,   0,  3,  -2, 1,  2,  1,  3,  2,  -1, -1,
    -3, -1, 2,  2,  -4, 1,  -2, 0,  5,  1,  0,  2,  -1, -1,  -2, 3,  -1, 1,  -3, 4,  5,  2,  0,  -1, 0,  1,   -1, -1,  -2, 1,  1,  0,  1,  1,  1,  2,  -1, -2,
    0,  3,  -1, -4, 4,  0,  -2, 1,  4,  1,  1,  1,  2,  0,   -1, -1, -1, -1, 0,  1,  1,  -2, 4,  0,  2,  4,   0,  2,   3,  0,  -1, -6, -1, 1,  0,  -1, -2, -2,
    2,  -6, 3,  0,  0,  3,  3,  -1, 3,  0,  -3, 4,  0,  3,   2,  0,  -1, 2,  -1, 0,  0,  0,  1,  0,  3,  1,   1,  0,   1,  1,  1,  1,  0,  -3, -1, 1,  0,  1,
    3,  1,  -1, 1,  -1, 3,  1,  1,  2,  2,  3,  1,  -1, -1,  -2, 0,  2,  2,  -2, -2, -1, -1, 2,  -1, 0,  2,   1,  -1,  2,  3,  1,  1,  -3, 2,  3,  2,  1,  -1,
    4,  0,  -1, -1, 0,  1,  0,  1,  0,  2,  0,  1,  1,  -3,  -1, 2,  0,  -3, 4,  1,  -1, 0,  0,  -1, 1,  2,   2,  2,   0,  1,  4,  1,  0,  -1, 3,  0,  2,  -5,
    7,  2,  4,  -1, 3,  7,  -1, 1,  -2, 1,  3,  -1, -4, 4,   -2, -2, 1,  0,  2,  1,  1,  -3, 0,  0,  3,  3,   2,  0,   0,  2,  1,  0,  0,  1,  -1, 0,  -2, 1,
    3,  5,  0,  2,  4,  -1, -1, -1, 1,  6,  0,  4,  -3, 0,   5,  5,  2,  0,  -2, -3, -3, 3,  0,  -1, 4,  0,   2,  3,   -3, 3,  0,  1,  -1, -3, -2, -3, 1,  -2,
    0,  -1, -1, 0,  -2, 1,  -3, 0,  0,  -1, -3, -4, -2, -1,  -1, 3,  8,  2,  -1, -2, -3, 2,  0,  0,  -4, -3,  -2, -6,  0,  -1, -7, -2, -1, -6, 0,  1,  0,  0,
    2,  0,  2,  5,  -7, 1,  2,  3,  1,  4,  2,  0,  3,  -2,  8,  1,  7,  -1, 0,  7,  -5, 2,  -1, 3,  3,  -2,  -2, 0,   2,  -2, 2,  2,  2,  -1, -4, -4, 0,  -2,
    2,  1,  4,  1,  0,  -1, 0,  1,  1,  2,  2,  -4, -4, 1,   3,  4,  1,  4,  2,  0,  1,  0,  1,  4,  -1, 5,   -3, 0,   2,  5,  -1, 3,  -1, -2, -1, 5,  0,  1,
    2,  -2, 1,  0,  -1, 3,  2,  2,  1,  -3, -2, 0,  2,  -1,  1,  0,  -4, 2,  -1, 1,  -1, 0,  -2, -1, -2, -4,  -2, 1,   -2, 6,  3,  1,  -2, -1, -1, 2,  3,  1,
    -3, -3, -1, -4, 1,  1,  -4, 0,  -1, -3, 0,  0,  -3, 0,   3,  -1, 3,  2,  -4, 1,  4,  3,  -1, 1,  1,  1,   0,  0,   3,  2,  3,  -1, 0,  6,  -4, -2, 0,  2,
    1,  1,  -3, 0,  1,  -3, 3,  2,  2,  0,  -3, -2, 0,  0,   0,  0,  4,  1,  -1, -3, -1, 4,  1,  2,  1,  0,   -2, -1,  2,  2,  0,  2,  1,  0,  -1, -1, -2, 3,
    0,  4,  -1, -2, -1, 1,  -1, 4,  -2, -5, -2, 4,  -1, 3,   1,  -2, 1,  1,  3,  3,  0,  3,  4,  -2, -3, -4,  1,  -2,  2,  3,  -1, 5,  1,  1,  1,  1,  -3, -3,
    -3, -4, -1, 2,  -1, 1,  7,  4,  0,  0,  0,  0,  1,  2,   2,  2,  1,  -7, 1,  -1, -5, -1, 3,  -3, 0,  -3,  0,  -1,  2,  -1, 3,  2,  -3, 2,  4,  1,  -1, 0,
    1,  2,  -1, -1, 4,  0,  2,  0,  1,  6,  0,  0,  0,  -2,  1,  -4, -4, 0,  -1, 0,  4,  2,  4,  2,  -4, -2,  0,  -1,  1,  3,  3,  -1, -1, -2, -2, 6,  -2, 1,
    3,  1,  -1, -1, 4,  3,  -1, 1,  2,  -1, -1, 1,  -2, 5,   -1, 3,  -2, -3, 1,  4,  0,  3,  -2, -4, 2,  3,   -1, 2,   1,  -2, 2,  2,  2,  2,  1,  3,  2,  -3,
    -4, -2, 3,  -4, 1,  1,  -1, 4,  1,  5,  2,  -1, -3, -4,  -4, -3, -1, 1,  2,  3,  3,  4,  -1, 3,  -1, 0,   2,  2,   1,  3,  3,  -3, 1,  -1, -3, 0,  5,  -1,
    2,  0,  0,  2,  3,  0,  2,  2,  -3, 0,  5,  2,  3,  1,   1,  2,  -2, -3, 3,  2,  0,  0,  0,  6,  -1, 0,   1,  2,   -1, -4, 1,  1,  -1, 1,  3,  1,  0,  1,
    -1, -4, 2,  0,  -2, -1, 2,  -2, -1, 0,  0,  4,  -1, 1,   1,  1,  -1, 2,  4,  2,  -1, -1, 0,  -1, 0,  1,   -1, 3,   0,  0,  1,  -3, 0,  3,  0,  -1, -6, -3,
    2,  4,  -2, 6,  2,  0,  2,  -1, 0,  4,  1,  4,  0,  -5,  -4, -2, 0,  -1, 0,  1,  1,  5,  1,  3,  1,  1,   -2, -3,  -6, -5, -1, -1, 2,  3,  2,  4,  -1, 1,
    0,  0,  2,  3,  -1, 0,  3,  -5, 1,  -1, -3, 2,  4,  -2,  3,  1,  3,  4,  -3, -4, 5,  2,  1,  5,  1,  2,   1,  3,   2,  2,  1,  -3, 1,  -3, -1, 1,  0,  -2,
    4,  5,  2,  -2, 6,  0,  -1, 2,  2,  0,  2,  -1, -1, 0,   1,  -1, 0,  1,  2,  -1, -2, 1,  1,  -2, 1,  2,   2,  1,   0,  0,  1,  1,  3,  5,  1,  1,  0,  0,
    -1, 2,  2,  3,  -2, -1, 1,  1,  1,  1,  -1, 2,  -4, 2,   1,  2,  2,  0,  -1, -1, 1,  3,  -1, -2, 0,  5,   1,  -2,  -3, -1, -3, -2, 1,  0,  -1, -1, -3, 0,
    -1, -2, 4,  -1, 1,  -1, -3, 0,  1,  1,  5,  3,  2,  3,   1,  2,  2,  0,  1,  1,  2,  0,  -1, -1, -1, 0,   -1, 4,   3,  1,  3,  3,  -2, -5, 4,  1,  -3, 5,
    2,  2,  0,  4,  0,  0,  -2, -2, 2,  1,  1,  0,  -2, -1,  3,  7,  3,  -3, 4,  1,  -1, 1,  0,  2,  3,  1,   2,  0,   0,  1,  1,  0,  0,  -4, -1, 1,  2,  -1,
    2,  3,  1,  1,  0,  0,  3,  4,  3,  5,  2,  3,  1,  0,   1,  4,  1,  1,  2,  0,  0,  3,  1,  0,  0,  2,   -2, 3,   0,  1,  -1, 0,  -2, 0,  0,  1,  -1, -2,
    -3, 3,  2,  -1, -1, -1, -2, -2, 5,  1,  -1, 0,  -2, 0,   -1, 0,  4,  -5, 0,  -2, 0,  -2, 2,  2,  4,  3,   2,  3,   2,  4,  1,  1,  1,  2,  2,  0,  -1, -3,
    -1, -1, -3, 2,  3,  1,  3,  5,  -2, -3, 4,  3,  -3, 4,   3,  1,  2,  2,  0,  -4, 0,  -1, 1,  1,  1,  -1,  -4, -2,  1,  8,  3,  1,  3,  -3, 0,  0,  0,  2,
    1,  2,  -1, 0,  0,  0,  -1, 0,  -2, 0,  -1, 1,  2,  -1,  1,  3,  -1, -2, 1,  0,  1,  4,  4,  3,  2,  3,   0,  0,   1,  4,  1,  0,  0,  0,  -1, 3,  -1, -2,
    0,  2,  -2, 4,  0,  -1, -1, 0,  -1, 4,  -2, 0,  -2, -1,  -2, 4,  1,  3,  -3, -1, 1,  -3, 3,  4,  -1, 1,   -1, 1,   -4, 1,  1,  -5, 3,  0,  0,  -1, -1, 1,
    4,  4,  0,  3,  0,  4,  3,  2,  1,  4,  4,  0,  -2, 0,   0,  -3, -2, 4,  3,  3,  6,  6,  2,  -2, 2,  1,   -1, 3,   3,  3,  1,  -1, 4,  -6, 1,  0,  -3, 1,
    1,  1,  -3, -4, 3,  4,  3,  1,  2,  -2, -1, -2, 0,  2,   2,  -1, 1,  3,  1,  -1, 0,  0,  -2, -2, 0,  2,   -1, -1,  1,  2,  -2, -1, 0,  -1, 1,  4,  4,  1,
    3,  1,  -2, -1, 1,  4,  2,  3,  0,  0,  -2, -1, -3, 0,   1,  5,  -1, 2,  0,  -2, 0,  0,  -2, 1,  0,  -1,  -1, -2,  -2, 3,  1,  0,  1,  0,  -2, 0,  1,  2,
    -1, 0,  1,  1,  -3, -1, 2,  -3, 1,  -2, -2, 0,  -3, 2,   3,  4,  -1, 1,  0,  3,  2,  0,  0,  2,  2,  -1,  0,  0,   1,  -4, -2, 1,  4,  3,  7,  4,  2,  -1,
    2,  0,  -2, 3,  2,  1,  3,  1,  2,  -5, 1,  0,  -1, -1,  3,  0,  -2, 0,  3,  6,  4,  2,  4,  0,  0,  -2,  2,  3,   3,  2,  1,  1,  0,  1,  1,  1,  -4, 0,
    -1, 1,  1,  0,  1,  5,  0,  1,  0,  0,  1,  4,  4,  5,   5,  3,  0,  0,  2,  1,  2,  2,  2,  0,  0,  -2,  -2, -1,  2,  4,  1,  1,  1,  -3, 1,  0,  -2, 2,
    -1, -1, 0,  1,  1,  4,  1,  2,  1,  0,  -3, 0,  1,  4,   -1, 2,  2,  1,  -1, -2, -1, -4, 2,  -4, -1, 2,   0,  1,   6,  4,  0,  0,  3,  1,  2,  0,  1,  4,
    2,  1,  0,  -2, -3, -4, -1, 2,  0,  -1, 2,  1,  -2, 0,   -2, 0,  8,  -1, 0,  1,  1,  4,  0,  1,  6,  3,   2,  3,   1,  0,  -1, 2,  2,  1,  2,  4,  2,  3,
    3,  -2, 1,  6,  0,  5,  4,  -1, 3,  -2, 0,  0,  2,  -1,  1,  -1, -1, 0,  0,  1,  -3, -2, 1,  0,  -3, 4,   -2, 1,   -2, 1,  1,  1,  1,  2,  -1, 4,  2,  1,
    -2, -2, 1,  -1, -2, 1,  -3, 2,  2,  2,  -2, 6,  -4, -1,  0,  0,  1,  3,  -1, 2,  1,  3,  2,  1,  3,  0,   0,  -5,  -3, 1,  1,  1,  3,  1,  4,  0,  -1, 0,
    0,  -3, -1, 4,  8,  2,  1,  1,  0,  -2, 1,  4,  5,  2,   1,  1,  5,  1,  2,  2,  0,  2,  1,  1,  5,  -2,  -4, -1,  -3, 0,  4,  1,  -3, 1,  -2, 0,  0,  -1,
    7,  4,  1,  2,  -1, 1,  0,  6,  1,  4,  2,  4,  2,  3,   0,  1,  0,  3,  0,  1,  3,  -1, 1,  0,  -3, 0,   4,  0,   0,  -3, -1, 0,  -3, 1,  -2, -2, 2,  0,
    -3, 3,  -2, 1,  -1, 1,  0,  0,  0,  0,  -1, 1,  2,  2,   -1, -1, 2,  -1, -1, 0,  -1, -2, 2,  3,  -2, 3,   -2, -2,  -1, 1,  2,  0,  1,  0,  1,  2,  5,  -2,
    2,  2,  -1, -4, -2, -1, 2,  0,  2,  2,  4,  1,  1,  1,   2,  -1, 0,  2,  3,  0,  -3, 1,  -2, -2, 1,  2,   5,  1,   0,  0,  5,  0,  1,  1,  -1, 3,  -2, 2,
    1,  -1, -4, 1,  -4, 1,  3,  1,  -5, 0,  0,  5,  -2, 2,   5,  3,  0,  5,  -4, -1, 2,  5,  5,  4,  4,  5,   3,  4,   1,  -2, 3,  5,  1,  -1, 3,  0,  4,  -2,
    1,  0,  3,  -1, 3,  -2, -2, 0,  -1, 5,  -2, -1, 2,  2,   -2, 3,  -2, 0,  -1, 0,  0,  2,  1,  2,  -2, 4,   0,  1,   1,  1,  3,  -2, -2, 1,  1,  -1, 2,  3,
    2,  4,  0,  -2, 1,  -1, 5,  1,  -2, 3,  1,  2,  3,  -3,  1,  0,  0,  -3, -1, 1,  1,  0,  3,  1,  1,  0,   0,  1,   2,  -1, -1, 2,  4,  1,  2,  4,  -1, -1,
    1,  2,  4,  0,  0,  2,  7,  3,  0,  -2, 4,  4,  0,  5,   3,  -1, -1, 4,  -5, 0,  5,  -1, -1, 0,  1,  5,   -4, 0,   5,  2,  3,  2,  -3, -4, -1, 5,  3,  0,
    3,  3,  0,  2,  3,  0,  3,  4,  -2, 0,  1,  2,  2,  0,   1,  -2, -2, 1,  1,  -1, -1, 0,  0,  3,  0,  -1,  1,  1,   -3, 3,  -3, 2,  -4, 3,  -1, 1,  0,  1,
    -3, 2,  0,  -1, 1,  0,  2,  -1, -1, 0,  1,  -2, 4,  2,   0,  2,  -1, 1,  2,  -1, 3,  4,  -3, 4,  -1, 3,   2,  -3,  2,  0,  0,  -3, 0,  0,  3,  0,  0,  1,
    1,  0,  0,  0,  0,  2,  0,  3,  4,  2,  1,  5,  1,  1,   -3, 1,  6,  -2, -1, 3,  6,  4,  0,  1,  3,  5,   -2, 1,   3,  -2, -1, 3,  -3, 0,  5,  0,  -3, -2,
    1,  5,  -3, 2,  5,  1,  3,  4,  -3, -3, -2, 3,  5,  1,   5,  3,  3,  2,  0,  0,  -1, 2,  -3, -1, 2,  3,   3,  -1,  1,  -1, 0,  0,  1,  -1, -1, 1,  2,  2,
    1,  -2, 1,  1,  2,  1,  -1, 1,  -3, 2,  0,  2,  0,  1,   -3, 2,  1,  0,  1,  -2, 1,  -2, -3, -1, 0,  -1,  3,  1,   1,  4,  1,  -3, 0,  -2, 1,  0,  1,  2,
    1,  1,  3,  -2, 1,  -1, -1, -3, 1,  0,  0,  1,  -1, 3,   -1, 2,  1,  2,  1,  3,  0,  3,  5,  1,  0,  4,   0,  3,   0,  1,  5,  2,  1,  3,  4,  6,  2,  2,
    3,  4,  1,  1,  -1, 0,  -2, 3,  2,  1,  2,  1,  0,  1,   0,  0,  0,  -1, 0,  0,  1,  0,  2,  1,  0,  -2,  1,  -2,  -1, 1,  2,  -1, 2,  2,  1,  2,  0,  -1,
    0,  2,  3,  1,  2,  -2, 0,  0,  1,  0,  0,  -2, 2,  2,   0,  0,  1,  1,  0,  1,  0,  -1, -1, 1,  0,  0,   0,  -1,  -3, -2, 0,  0,  2,  0,  1,  1,  0,  1,
    2,  -2, 0,  -1, 2,  1,  4,  2,  1,  0,  0,  -1, 3,  2,   2,  1,  2,  4,  2,  2,  2,  3,  0,  0,  -2, 0,   0,  2,   3,  3,  3,  5,  3,  0,  0,  1,  3,  0,
    1,  2,  1,  0,  -1, -2, 4,  3,  0,  -3, -2, 1,  0,  -1,  -1, -1, 2,  2,  1,  -1, 0,  0,  -2, 0,  -2, 0,   0,  3,   0,  1,  1,  1,  2,  1,  1,  -1, 1,  2,
    1,  1,  1,  -1, -1, 0,  1,  1,  -1, -1, 0,  0,  0,  -1,  2,  0,  2,  1,  0,  -2, 2,  1,  3,  1,  2,  0,   2,  0,   1,  0,  3,  1,  2,  3,  1,  1,  0,  0,
    -2, 0,  0,  0,  -3, -3, 2,  1,  2,  1,  1,  -1, 2,  3,   0,  -1, 2,  -1, 1,  2,  2,  1,  0,  1,  0,  1,   0,  1,   2,  3,  3,  3,  2,  4,  2,  3,  2,  1,
    1,  2,  0,  0,  0,  3,  1,  4,  2,  1,  3,  0,  1,  -1,  -1, 2,  0,  1,  2,  -1, 2,  3,  0,  -3, -2, 2,   -1, 1,   1,  1,  0,  1,  -1, 3,  3,  3,  -3, 1,
    1,  -4, 0,  2,  1,  2,  -1, 3,  1,  1,  1,  3,  3,  1,   3,  1,  2,  -1, 0,  2,  2,  -1, -2, 0,  3,  3,   0,  -1,  1,  3,  3,  2,  2,  -1, 1,  4,  2,  0,
    2,  2,  2,  0,  0,  0,  -1, 1,  1,  1,  0,  2,  2,  -2,  -1, 1,  0,  1,  0,  -1, 1,  -1, 4,  1,  1,  1,   1,  -1,  1,  0,  -1, -1, -1, 1,  3,  3,  1,  1,
    0,  2,  1,  0,  -1, 0,  3,  5,  0,  -1, -1, 1,  1,  1,   -2, 2,  0,  0,  -3, 0,  2,  4,  1,  2,  1,  1,   0,  0,   -3, 3,  0,  -1, -2, 0,  4,  3,  0,  -2,
    1,  -1, -3, -3, -1, -1, 0,  0,  -1, 2,  0,  -1, 2,  -1,  3,  0,  0,  3,  1,  1,  0,  2,  -1, -1, 3,  2,   2,  1,   3,  -1, 3,  -2, -1, -1, 0,  2,  2,  2,
    1,  4,  2,  1,  2,  -2, 3,  1,  2,  2,  0,  5,  4,  2,   2,  2,  1,  -2, 1,  0,  3,  0,  2,  0,  1,  1,   3,  1,   1,  1,  1,  2,  0,  0,  -1, 4,  2,  4,
    2,  0,  1,  0,  -1, 2,  2,  0,  -2, -1, 1,  0,  1,  -2,  0,  1,  4,  0,  -2, 1,  1,  5,  1,  1,  0,  2,   4,  2,   -3, 3,  -1, 0,  -1, 3,  1,  3,  5,  2,
    3,  3,  4,  -1, -1, 3,  0,  0,  0,  3,  7,  5,  3,  3,   2,  1,  1,  2,  4,  2,  1,  5,  0,  2,  0,  0,   1,  0,   -1, 1,  -1, 1,  0,  -2, 1,  2,  -2, 1,
    -1, -4, -1, 0,  -1, 0,  1,  -4, -1, -1, 1,  2,  0,  0,   1,  -2, -2, -3, 1,  -2, 3,  0,  -2, 0,  0,  -2,  1,  -2,  -1, 2,  -2, -4, -3, -2, -2, -1, -1, 0,
    0,  0,  -1, -4, -2, -1, -1, -3, -2, -6, -3, -2, -2, 0,   -2, -2, -2, -1, 1,  0,  -3, -3, -3, 1,  -1, -1,  2,  0,   -2, -2, 2,  -3, -3, 0,  0,  1,  0,  -2,
    -1, -2, -4, 0,  -1, 0,  -1, -2, 0,  -1, -1, 3,  1,  0,   1,  0,  -2, -2, -2, 5,  0,  -1, -1, 0,  1,  -1,  -3, -3,  -3, -1, 1,  0,  3,  3,  2,  -2, 1,  1,
    2,  1,  2,  -2, 0,  0,  1,  0,  0,  0,  -2, 2,  0,  0,   4,  1,  1,  3,  3,  2,  0,  2,  2,  3,  4,  1,   -1, -1,  3,  2,  2,  1,  3,  -3, -3, -4, 3,  -1,
    -3, 0,  1,  1,  1,  0,  2,  4,  -1, 1,  1,  -1, 0,  -1,  3,  0,  -3, 3,  2,  1,  0,  -1, -1, -1, 1,  1,   4,  2,   2,  -1, -1, 0,  1,  1,  1,  2,  0,  -1,
    -1, 6,  1,  -3, 1,  1,  0,  -1, 0,  2,  -1, -2, -1, -2,  1,  2,  3,  0,  1,  0,  -1, 3,  0,  1,  3,  -2,  8,  2,   3,  6,  3,  9,  2,  2,  0,  0,  3,  2,
    2,  4,  2,  -4, -4, 2,  -3, -1, -5, -1, 4,  -3, 2,  2,   -1, 4,  -2, -1, 4,  -2, 2,  -1, 3,  0,  -1, 3,   1,  1,   3,  0,  0,  1,  1,  3,  -1, 1,  2,  2,
    3,  2,  -2, 0,  1,  2,  2,  -1, 2,  0,  -3, -3, 3,  1,   -4, 2,  2,  0,  -2, 2,  5,  5,  0,  1,  4,  0,   -1, -3,  1,  0,  -1, 0,  0,  2,  0,  -1, 0,  3,
    1,  1,  1,  2,  0,  -1, 0,  2,  -1, 3,  3,  3,  3,  1,   0,  3,  0,  -4, -2, 2,  1,  -1, -2, 2,  -3, -3,  -2, -5,  2,  3,  0,  -2, -1, -1, 0,  3,  1,  5,
    2,  0,  5,  2,  4,  9,  3,  9,  3,  1,  -1, -1, 0,  0,   1,  2,  3,  -6, -6, 3,  -2, -2, -1, -1, 4,  -4,  2,  2,   2,  0,  -1, -1, 5,  -2, 3,  -3, 2,  1,
    -3, 5,  -2, 0,  1,  -4, 1,  0,  1,  5,  2,  -3, 1,  -1,  3,  1,  -2, -1, 0,  1,  3,  0,  -1, -2, 1,  -2,  3,  1,   -3, 0,  0,  1,  -1, 3,  3,  3,  2,  2,
    3,  1,  0,  -2, 2,  1,  -1, 2,  0,  -1, -1, 1,  0,  1,   1,  2,  0,  3,  1,  1,  1,  -1, 1,  3,  3,  4,   -1, 4,   2,  1,  -3, -2, -3, 0,  0,  1,  -1, 3,
    -3, -4, 1,  -4, 1,  3,  0,  0,  0,  -5, 0,  4,  -1, 1,   6,  -1, 7,  0,  2,  8,  2,  7,  0,  2,  -4, -1,  3,  0,   2,  3,  0,  -2, -6, 3,  -3, -1, -1, 2,
    4,  -2, 2,  1,  -2, 1,  -1, -1, 4,  -4, 1,  -1, 0,  1,   2,  2,  -2, 2,  2,  -5, 4,  0,  0,  5,  0,  -3,  0,  0,   3,  -1, 0,  -2, 0,  1,  3,  3,  1,  -2,
    -3, -1, 5,  -1, -4, 1,  1,  -1, 0,  3,  5,  9,  3,  2,   5,  -1, 2,  -3, 0,  4,  -1, 2,  1,  1,  0,  2,   0,  -1,  0,  -1, -1, 4,  0,  0,  1,  0,  0,  -1,
    1,  4,  2,  3,  2,  2,  -2, -2, -3, -2, -1, 1,  -1, 2,   -3, -2, -1, -1, -2, 2,  -2, -3, 0,  -2, 1,  3,   1,  2,   4,  -4, 7,  1,  3,  6,  3,  10, 2,  2,
    -4, -3, 0,  0,  0,  3,  -3, -3, -3, -2, -3, -3, -2, 2,   3,  -3, 3,  3,  0,  0,  -1, -3, 5,  -2, 0,  1,   0,  -2,  5,  5,  -6, 1,  2,  -4, 4,  -3, -1, 6,
    -1, -2, 4,  0,  1,  -2, 2,  1,  0,  0,  5,  3,  1,  1,   0,  -1, 3,  1,  -4, 3,  -1, 0,  0,  2,  5,  8,   2,  2,   3,  0,  3,  -4, 1,  2,  0,  0,  1,  2,
    1,  2,  1,  -1, 0,  -2, -2, 5,  2,  -1, 0,  0,  3,  -1,  3,  5,  2,  3,  2,  2,  -3, -3, 1,  -1, 0,  0,   -3, -1,  -3, -4, -4, -4, 0,  4,  2,  -3, 0,  -3,
    0,  4,  -3, 2,  2,  -3, 7,  2,  -2, 7,  6,  9,  1,  1,   1,  -3, 1,  1,  5,  4,  -3, -8, -3, 0,  -6, -1,  -2, 3,   1,  0,  -1, 2,  0,  -1, 3,  -1, -4, 0,
    2,  -4, 0,  -1, -1, 1,  0,  -2, 1,  -1, 2,  0,  3,  -1,  0,  -2, 4,  4,  0,  2,  4,  1,  -1, 1,  5,  2,   0,  -3,  -1, -2, -4, -2, -2, -1, 2,  4,  -1, -4,
    2,  5,  2,  -1, -2, 2,  4,  1,  2,  2,  1,  1,  1,  2,   0,  -2, 1,  0,  -1, -1, -1, 0,  -2, 2,  1,  2,   0,  3,   -1, 2,  3,  -1, -3, 2,  4,  0,  0,  -3,
    -1, 0,  -3, -2, -1, -2, 1,  -3, -1, 2,  3,  0,  2,  -1,  0,  -2, -3, -3, 0,  1,  0,  1,  1,  1,  1,  2,   1,  2,   1,  2,  2,  5,  3,  1,  4,  2,  5,  1,
    0,  1,  4,  6,  -3, -2, 0,  2,  -2, 0,  2,  -1, -5, 1,   1,  -2, -3, -1, 0,  0,  1,  -2, -2, -1, 1,  1,   1,  0,   -1, 0,  6,  0,  -1, 1,  4,  0,  -5, 0,
    4,  4,  1,  1,  -2, -1, -2, -2, -2, 1,  1,  5,  0,  -1,  4,  5,  3,  -1, -1, 4,  4,  2,  3,  3,  -1, 3,   -1, 0,   0,  -1, -3, 0,  1,  0,  1,  2,  0,  2,
    1,  -1, 2,  3,  -1, 2,  2,  -4, 0,  0,  5,  -1, 1,  -2,  3,  0,  -2, -3, -2, -2, -1, 0,  0,  2,  2,  -1,  1,  3,   0,  0,  -2, -3, -1, 1,  -1, 2,  2,  1,
    2,  2,  0,  1,  1,  -3, 4,  8,  3,  0,  3,  2,  5,  -2,  1,  1,  4,  6,  -2, -3, 0,  1,  1,  0,  1,  -1,  -4, 2,   1,  -2, -7, 1,  2,  1,  2,  -1, -1, -3,
    2,  4,  4,  -1, -1, 2,  5,  2,  0,  3,  6,  0,  -3, 1,   3,  3,  1,  -2, -1, -3, -3, -2, -2, -2, 0,  3,   1,  -2,  2,  3,  0,  -1, -3, 2,  6,  3,  1,  1,
    -1, 0,  0,  0,  -1, 0,  -1, 1,  2,  -2, 2,  -1, -2, -1,  -2, 3,  2,  0,  -1, 3,  1,  -3, -2, -1, 3,  0,   1,  -3,  0,  0,  -3, 0,  -1, -3, -4, -1, -2, 1,
    3,  -2, 0,  0,  0,  -3, -2, 1,  -2, 1,  -2, 2,  1,  1,   1,  0,  0,  5,  -1, -4, 2,  6,  2,  0,  1,  1,   5,  -3,  -1, 0,  4,  6,  0,  -5, 0,  0,  -1, 1,
    1,  1,  -4, 1,  2,  -3, -5, 3,  2,  3,  2,  -1, 1,  0,   2,  2,  1,  0,  -2, 2,  7,  0,  -1, 1,  6,  2,   -2, 5,   3,  4,  -1, -3, -2, -1, -3, -3, -1, 0,
    3,  4,  3,  1,  0,  2,  1,  0,  -1, 2,  4,  3,  2,  3,   0,  2,  0,  1,  -1, 2,  -1, -1, 2,  -2, 2,  1,   1,  3,   -3, 1,  2,  -1, -2, 3,  0,  -3, -2, 0,
    2,  1,  2,  -3, 4,  -1, -3, 0,  -1, -3, -2, 0,  -1, -1,  5,  0,  3,  3,  2,  -1, 0,  0,  -1, 0,  -2, 3,   3,  -1,  2,  4,  0,  4,  0,  -4, 2,  6,  4,  3,
    6,  3,  8,  -1, 1,  1,  3,  7,  2,  -5, 3,  1,  0,  -2,  1,  1,  -5, 1,  3,  -3, -4, 1,  4,  -1, 3,  -2,  -1, 1,   0,  3,  1,  2,  -1, 1,  7,  1,  0,  1,
    5,  0,  -1, 1,  1,  3,  0,  -1, -3, -3, -5, 1,  -3, -4,  1,  2,  0,  1,  1,  4,  0,  -1, -2, 4,  1,  2,   2,  5,   -1, 3,  0,  -1, -1, -1, -4, 0,  -1, -3,
    0,  0,  -2, 0,  -1, -1, 3,  1,  0,  3,  3,  -1, -3, 2,   3,  1,  0,  -2, 2,  -1, -2, 2,  0,  -2, -3, 0,   1,  -2,  3,  -3, -2, 0,  1,  1,  -1, 0,  -1, 1,
    0,  1,  3,  -2, 3,  4,  0,  2,  0,  -4, 0,  5,  3,  1,   4,  0,  6,  -2, 0,  1,  2,  6,  1,  -5, 0,  1,   -2, 1,   -3, -1, -1, 6,  2,  -1, 0,  2,  1,  0,
    1,  4,  4,  3,  2,  2,  4,  -4, 0,  -1, 2,  0,  -3, -3,  2,  -3, -1, -1, 2,  3,  -1, -2, -2, -1, 1,  1,   3,  -3,  0,  1,  4,  4,  -1, 0,  3,  1,  0,  0,
    2,  -1, -3, 1,  0,  7,  2,  1,  1,  2,  4,  3,  0,  0,   3,  5,  0,  0,  2,  -2, -3, -2, -1, 1,  0,  -2,  -2, 0,   1,  3,  2,  1,  1,  6,  3,  6,  0,  5,
    -1, 3,  4,  2,  4,  -2, -2, -1, -5, 4,  2,  -1, -1, -1,  3,  6,  -1, 1,  2,  3,  1,  1,  0,  5,  2,  3,   4,  3,   2,  2,  4,  4,  1,  0,  1,  -1, 1,  -1,
    -2, 1,  0,  -3, -3, -1, 4,  4,  -2, 0,  4,  2,  4,  2,   1,  4,  2,  3,  2,  0,  7,  -3, -3, -1, 4,  -2,  -4, 0,   0,  -1, -1, 0,  -2, 1,  1,  -3, -4, -2,
    1,  1,  0,  -3, -1, 4,  2,  2,  -2, 0,  0,  1,  -2, 0,   0,  -1, -2, 0,  0,  1,  1,  0,  0,  0,  6,  2,   2,  1,   1,  2,  1,  0,  0,  -1, -2, -3, -4, 2,
    0,  1,  -1, 1,  0,  2,  0,  4,  -2, 4,  1,  3,  0,  4,   0,  -1, 2,  0,  1,  -1, 0,  -2, -4, 5,  0,  -2,  1,  -5,  -3, 3,  0,  0,  -2, 0,  -3, -2, -2, 2,
    1,  2,  2,  -1, 0,  3,  4,  1,  1,  1,  -2, -1, 0,  -1,  -2, 0,  -1, -3, 0,  2,  4,  1,  -2, 1,  3,  5,   3,  3,   0,  1,  1,  1,  4,  2,  6,  -2, 0,  0,
    4,  -1, -1, 2,  -4, 1,  1,  2,  1,  4,  1,  -4, -5, 0,   -2, 2,  1,  -1, -1, 5,  3,  0,  -2, -2, -1, 2,   -1, 3,   1,  1,  0,  -2, -1, 5,  2,  1,  0,  0,
    5,  -1, 0,  0,  1,  5,  1,  1,  2,  1,  -3, -1, -1, 1,   1,  0,  -1, 0,  -2, -1, -2, 4,  1,  2,  0,  1,   1,  2,   0,  3,  5,  1,  3,  1,  3,  -1, -3, 5,
    -1, -1, 0,  -3, -2, 2,  -3, -1, 0,  2,  -3, 1,  -1, 0,   4,  1,  1,  1,  3,  6,  3,  -2, -2, 0,  -1, -2,  1,  2,   1,  -1, 1,  -1, -1, 2,  6,  2,  -3, 0,
    2,  2,  1,  3,  -2, 3,  -4, 5,  5,  3,  4,  -5, -2, 1,   -2, -1, 1,  2,  -2, -2, 0,  3,  2,  4,  -2, 1,   -4, -1,  -1, 1,  -1, -3, -2, 2,  2,  -2, -3, -2,
    2,  1,  0,  1,  1,  1,  2,  1,  2,  3,  2,  -1, -3, 0,   3,  1,  0,  -2, 1,  2,  1,  0,  2,  0,  -5, -2,  -4, 3,   3,  2,  -3, -1, -3, 2,  -3, 1,  2,  1,
    0,  2,  1,  3,  2,  0,  2,  -3, 1,  3,  -1, -3, -4, 4,   0,  1,  3,  -4, -1, -2, -3, -1, 0,  0,  -2, 1,   2,  2,   0,  4,  4,  1,  1,  0,  3,  -3, -1, -2,
    1,  -1, 1,  1,  2,  1,  -2, -1, 1,  -2, 4,  2,  0,  1,   1,  4,  1,  4,  0,  3,  -4, 4,  3,  1,  5,  -6,  0,  -1,  0,  0,  2,  0,  0,  -1, 1,  4,  1,  -1,
    -2, -1, -4, 0,  -2, -1, -1, 0,  1,  3,  2,  0,  2,  1,   1,  1,  -2, -1, 1,  0,  1,  2,  1,  1,  1,  1,   1,  0,   6,  -1, 0,  -3, 1,  2,  0,  1,  0,  0,
    -3, -1, -1, 0,  1,  2,  0,  1,  -4, 1,  -4, 3,  -2, 3,   -1, 3,  -3, -2, 2,  4,  3,  -2, 5,  0,  0,  0,   -1, -1,  -3, -1, 2,  -4, -2, 1,  -4, -1, -1, -1,
    2,  -1, 0,  2,  2,  5,  4,  1,  5,  1,  -1, -6, 0,  -1,  -1, -1, 3,  -2, 3,  2,  2,  0,  2,  -1, -1, 0,   -1, -1,  0,  0,  0,  -5, 2,  3,  -5, 3,  -1, -1,
    2,  1,  2,  0,  4,  3,  0,  0,  0,  0,  0,  2,  1,  0,   0,  -1, -2, -1, -3, 3,  -2, 2,  -2, 0,  3,  -3,  -4, 1,   2,  -2, 2,  -1, 3,  1,  1,  4,  5,  -1,
    1,  2,  -1, -1, 0,  -2, 2,  2,  -3, -3, 2,  1,  2,  -1,  -1, -5, 1,  5,  1,  -2, 2,  6,  0,  3,  2,  4,   1,  -1,  1,  3,  1,  -4, -5, -1, -3, 0,  0,  1,
    1,  -2, -5, -3, 1,  -3, -5, 2,  2,  3,  3,  -2, 2,  3,   0,  0,  0,  -3, -2, 0,  1,  4,  1,  3,  5,  -2,  -3, 1,   6,  2,  2,  1,  3,  2,  0,  1,  2,  1,
    0,  2,  -3, 2,  0,  -1, -1, -4, 1,  2,  -6, 1,  -3, 2,   2,  6,  1,  -2, 1,  1,  0,  1,  0,  -2, -1, 1,   -1, -1,  1,  1,  0,  1,  -1, 1,  2,  3,  -3, 4,
    3,  -1, -2, -1, 3,  -3, -2, 0,  1,  2,  4,  4,  2,  0,   -1, 1,  0,  -1, 1,  -2, 1,  1,  -2, -1, 2,  1,   4,  -2,  -1, -2, 0,  4,  0,  -1, 3,  6,  1,  3,
    3,  5,  3,  3,  2,  2,  1,  -3, -5, -2, -3, -1, -1, 1,   0,  -1, -2, -3, -3, -1, -2, 1,  -1, 1,  4,  -1,  0,  1,   0,  -3, -3, -1, -1, 0,  -2, 2,  1,  3,
    5,  -2, -1, 1,  4,  1,  2,  6,  4,  1,  3,  2,  -1, 0,   0,  1,  -3, 1,  1,  -4, -2, -2, 1,  -2, -3, 1,   -4, 3,   0,  3,  0,  -1, 0,  2,  1,  3,  -1, -1,
    -2, 0,  0,  1,  2,  3,  1,  3,  -3, 4,  3,  4,  -3, 1,   1,  -1, -2, 1,  1,  -4, -1, 0,  0,  2,  4,  4,   0,  2,   0,  1,  0,  -1, 1,  -1, -1, 2,  -2, -2,
    1,  1,  3,  -2, -1, -1, -1, 3,  0,  1,  3,  5,  1,  1,   3,  5,  3,  2,  2,  2,  0,  -3, -2, -3, -3, 0,   -1, 0,   0,  1,  -2, -3, -3, -6, 1,  -1, 1,  2,
    1,  0,  0,  2,  0,  -5, -3, -2, 1,  1,  2,  -1, -1, 2,   4,  0,  0,  2,  8,  1,  0,  6,  5,  3,  1,  2,   -1, 1,   1,  -1, 1,  0,  0,  -4, -1, -1, 1,  1,
    -2, 2,  -2, 1,  2,  2,  -1, 0,  1,  3,  3,  2,  -3, -1,  -1, -1, 1,  1,  0,  1,  3,  4,  -3, 2,  5,  2,   -2, 1,   4,  -3, -1, 3,  1,  -1, -2, -2, 2,  3,
    3,  5,  -1, 3,  0,  1,  1,  0,  -1, -1, 2,  -2, -2, -2,  0,  2,  3,  0,  -5, -2, 1,  4,  0,  3,  3,  2,   -4, 1,   0,  4,  5,  0,  -2, 3,  0,  -3, -1, -2,
    -1, 1,  2,  0,  1,  -2, -4, -6, -1, -5, -1, 0,  1,  2,   2,  -1, 1,  -1, -1, -6, -3, 2,  2,  -2, 0,  1,   -1, 1,   4,  -3, -1, 5,  6,  1,  0,  3,  4,  1,
    0,  0,  -4, 0,  3,  -3, 1,  0,  0,  -4, 3,  0,  1,  -2,  -1, 2,  -4, 3,  0,  1,  0,  1,  0,  3,  5,  5,   -2, 0,   0,  1,  1,  2,  2,  2,  5,  2,  -1, 3,
    1,  3,  -1, 0,  2,  -3, 0,  6,  0,  -1, -1, 1,  2,  2,   3,  4,  0,  0,  2,  2,  0,  -2, -1, 0,  1,  -2,  -3, -3,  2,  4,  3,  -2, -1, 0,  -2, 4,  -4, 4,
    5,  2,  -1, 3,  1,  3,  4,  -1, -1, 3,  -1, -6, -2, -1,  1,  -1, 0,  -1, -1, -3, -3, -1, 2,  -3, 2,  0,   3,  0,   1,  -1, 4,  1,  -3, -6, -3, 1,  0,  -3,
    0,  -2, 0,  1,  3,  -1, -1, 3,  5,  -4, 1,  3,  0,  -2,  -1, 0,  0,  0,  -3, 2,  2,  -1, 2,  0,  3,  0,   0,  3,   1,  -4, 1,  0,  -1, 0,  0,  1,  4,  0,
    3,  -2, -2, -1, 0,  -1, 0,  4,  4,  1,  -1, 6,  6,  -2,  -2, -1, -4, -2, -1, 1,  -3, 3,  -1, 0,  1,  0,   -3, -2,  1,  2,  -3, 0,  2,  -1, -2, 0,  2,  1,
    0,  0,  1,  -1, 1,  5,  3,  -1, 1,  2,  -1, 0,  1,  2,   2,  0,  -2, -1, 2,  0,  1,  -1, -1, -2, 0,  1,   -2, -2,  3,  1,  1,  3,  3,  0,  1,  -2, -1, -1,
    0,  -1, 0,  3,  4,  4,  -3, 1,  2,  2,  3,  0,  3,  -1,  -1, 1,  2,  2,  -1, 5,  1,  1,  -3, -2, 2,  1,   -1, 1,   -1, 1,  0,  0,  0,  -2, 1,  1,  2,  0,
    3,  -1, -2, 1,  4,  -3, 5,  -2, -2, 0,  1,  2,  1,  4,   2,  1,  0,  -1, -1, -1, 0,  3,  1,  2,  -1, 9,   5,  -1,  -1, -2, -5, -3, -2, -2, -2, 4,  -1, 0,
    1,  -1, 1,  -3, 3,  0,  -2, 0,  0,  0,  -2, 2,  2,  1,   0,  -1, -1, 0,  4,  2,  0,  -4, 2,  2,  0,  1,   0,  2,   4,  0,  0,  0,  -2, 0,  5,  -2, -1, -1,
    1,  1,  0,  -1, 3,  3,  -1, 2,  4,  -2, -2, 1,  -1, 0,   -1, 0,  0,  3,  3,  3,  -2, 1,  4,  2,  2,  -1,  2,  -1,  -1, 1,  2,  3,  -1, 2,  0,  1,  -5, -5,
    1,  0,  1,  1,  0,  0,  3,  0,  -1, 2,  -1, -1, 1,  2,   4,  -2, -1, 1,  2,  -3, 5,  -1, 0,  -1, 1,  1,   1,  2,   4,  -1, 1,  -1, -4, 0,  2,  2,  2,  0,
    2,  9,  4,  2,  2,  1,  -4, -1, 0,  -1, -1, 2,  -3, 2,   2,  -1, 1,  -4, 0,  -1, 1,  1,  0,  0,  -2, 2,   1,  1,   2,  1,  1,  0,  3,  0,  3,  -1, 1,  2,
    2,  2,  0,  1,  3,  -1, 1,  0,  0,  1,  4,  -2, 0,  -2,  -1, 1,  0,  1,  3,  0,  -1, 1,  2,  0,  -1, -2,  1,  -1,  1,  1,  2,  0,  2,  2,  -2, 0,  4,  4,
    3,  1,  2,  -1, -3, 0,  0,  5,  1,  3,  -1, 1,  -3, -4,  3,  2,  2,  -1, -2, 0,  2,  0,  0,  2,  -1, -1,  0,  3,   4,  1,  -2, 0,  -1, -2, 4,  -4, -1, 1,
    1,  2,  0,  2,  1,  0,  -2, 0,  -2, -4, 0,  4,  3,  0,   2,  8,  2,  2,  2,  0,  -4, 2,  -1, -2, 0,  -1,  -2, 1,   1,  1,  1,  -3, -1, 0,  -1, 1,  2,  -1,
    -1, 1,  3,  1,  2,  3,  2,  1,  1,  2,  1,  -3, 1,  1,   1,  4,  1,  2,  2,  -1, 0,  0,  0,  -1, 1,  0,   -1, -2,  0,  1,  1,  1,  1,  0,  1,  2,  1,  0,
    -3, -1, -3, 0,  0,  -1, 2,  1,  5,  5,  0,  1,  3,  2,   3,  2,  3,  0,  -3, -1, -2, 2,  0,  2,  0,  0,   -2, -3,  1,  -1, 2,  0,  -2, 1,  0,  -1, 2,  4,
    -1, -3, 2,  5,  3,  1,  -3, 2,  -1, -2, 4,  -2, 1,  0,   0,  2,  -1, 3,  3,  0,  -1, 0,  -3, -1, 1,  3,   2,  0,   1,  7,  3,  0,  3,  2,  -3, -1, -2, 1,
    -2, 1,  -3, 0,  2,  0,  1,  -2, 0,  2,  -1, 0,  -1, -1,  0,  2,  2,  2,  2,  2,  1,  -1, 2,  1,  4,  -1,  -4, -1,  2,  4,  1,  0,  3,  0,  -2, 0,  4,  1,
    0,  -1, -2, -3, 1,  0,  0,  0,  2,  2,  -1, 1,  3,  1,   0,  1,  -1, 2,  0,  0,  0,  1,  4,  4,  3,  1,   6,  3,   5,  1,  2,  0,  -2, -1, -2, 1,  3,  2,
    -1, 2,  -1, -2, -1, 1,  1,  0,  -3, -4, -1, 3,  1,  4,   1,  1,  1,  -1, 0,  4,  1,  -1, -3, 1,  3,  1,   -2, 1,   1,  2,  1,  -1, 2,  -2, 1,  3,  3,  1,
    1,  -1, 3,  1,  0,  2,  -1, 2,  -1, -3, 2,  4,  -1, -5,  -2, 4,  5,  -1, -2, 0,  0,  -1, 1,  -1, 1,  0,   -1, 2,   0,  0,  -3, 4,  -1, -1, 4,  -2, 2,  1,
    -4, 1,  -2, 2,  2,  0,  1,  2,  1,  2,  2,  2,  5,  2,   4,  -2, 1,  3,  0,  1,  3,  0,  4,  1,  0,  1,   4,  -2,  -1, -2, -1, 2,  1,  0,  4,  4,  3,  -1,
    0,  -2, 0,  0,  -1, 1,  0,  -1, -2, -2, -1, 2,  0,  4,   3,  1,  0,  -1, 1,  2,  0,  -2, -2, -4, -1, 2,   2,  1,   3,  -1, 4,  -3, -2, 1,  2,  -2, -1, -1,
    1,  1,  -1, 2,  1,  1,  0,  -3, 2,  -2, 1,  1,  0,  3,   0,  -1, 0,  2,  -1, 2,  -1, 1,  0,  -2, 1,  3,   -1, -3,  -6, 4,  7,  -3, -2, 0,  -3, -1, 1,  -1,
    1,  1,  0,  2,  1,  -3, -4, 4,  -1, 1,  2,  0,  0,  1,   -3, -1, -2, 0,  3,  2,  1,  -1, 2,  -3, 2,  0,   2,  3,   4,  -1, 4,  2,  2,  0,  2,  0,  4,  1,
    1,  1,  1,  0,  -3, 1,  1,  -1, 1,  0,  0,  1,  4,  -2,  -1, 0,  0,  1,  0,  2,  1,  -2, -2, -2, 2,  2,   1,  2,   3,  3,  2,  0,  2,  0,  0,  -1, -3, -1,
    -1, 4,  2,  1,  1,  3,  5,  -1, 0,  3,  0,  0,  -4, 0,   1,  -2, 0,  2,  2,  4,  0,  1,  4,  -1, 0,  0,   2,  6,   2,  -5, 4,  4,  -1, 2,  -1, 4,  4,  -2,
    1,  2,  -4, -3, -4, 1,  6,  0,  -1, 0,  -2, -1, 1,  0,   0,  1,  1,  -1, -1, 0,  1,  4,  -3, 0,  1,  1,   0,  2,   -1, 0,  -4, -1, 2,  1,  4,  3,  2,  -1,
    3,  0,  2,  -1, 1,  1,  2,  1,  0,  0,  0,  0,  2,  -2,  1,  0,  1,  0,  1,  -2, 1,  1,  -1, 3,  2,  2,   4,  -2,  -1, 0,  -1, -1, -1, 1,  3,  -1, -3, -3,
    1,  3,  2,  3,  4,  4,  0,  0,  -1, 1,  1,  2,  -1, -1,  -2, 3,  2,  -2, 3,  1,  2,  -1, -1, 4,  -1, -3,  -3, -2,  -1, -1, -3, 2,  -1, 3,  -1, 2,  3,  -4,
    3,  -1, 3,  2,  2,  -3, 1,  3,  0,  -2, 3,  4,  4,  -2,  1,  1,  -2, -4, -6, 0,  3,  0,  -1, -2, -3, -3,  -1, 1,   1,  1,  -3, 0,  -1, -1, 0,  4,  0,  2,
    0,  2,  0,  2,  -1, 2,  -3, 1,  2,  0,  2,  2,  1,  0,   3,  -3, 3,  0,  0,  -1, 2,  4,  2,  1,  -1, -1,  3,  0,   0,  3,  2,  0,  -2, 0,  -1, 1,  1,  2,
    2,  4,  4,  -2, -2, 1,  0,  0,  0,  1,  3,  -2, -2, -2,  0,  1,  4,  3,  3,  2,  1,  0,  -2, -2, 2,  0,   2,  -1,  -1, 4,  2,  0,  3,  -1, 4,  3,  -1, -2,
    1,  -2, 0,  1,  -3, -1, 0,  1,  -3, 3,  2,  1,  3,  -1,  1,  0,  2,  4,  0,  -3, 0,  2,  0,  3,  2,  6,   1,  -3,  0,  1,  -1, -4, -4, 2,  3,  -1, 0,  -2,
    -3, -1, 1,  1,  1,  2,  0,  -1, 0,  -1, -1, 2,  -1, 0,   2,  1,  1,  3,  -1, 1,  -1, 1,  2,  -1, 3,  2,   -1, -1,  2,  -2, 1,  1,  -1, 0,  2,  -2, 0,  0,
    -1, 0,  1,  2,  1,  3,  0,  -1, -1, 2,  2,  -1, 1,  2,   1,  7,  1,  -2, -1, 0,  2,  0,  0,  3,  2,  2,   -1, -4,  -1, 1,  3,  4,  1,  4,  1,  -2, 0,  3,
    2,  1,  2,  3,  4,  -2, -2, -4, -2, 3,  1,  2,  0,  5,   -1, 2,  4,  0,  -2, 0,  2,  -2, 4,  4,  -2, 5,   -1, -2,  -1, 1,  -1, -1, 1,  1,  -2, 2,  2,  -1,
    -1, -1, 2,  -5, -1, 0,  3,  3,  0,  1,  3,  -1, -2, 0,   1,  2,  1,  4,  2,  5,  0,  1,  1,  4,  4,  2,   4,  4,   3,  5,  1,  4,  0,  -2, 0,  -1, -2, -2,
    -2, 0,  -3, 1,  -1, -1, 0,  1,  6,  -3, -1, -3, 1,  -1,  0,  1,  -2, 1,  2,  4,  1,  3,  -1, 3,  0,  3,   4,  2,   2,  2,  0,  1,  3,  6,  1,  3,  1,  1,
    0,  -5, 2,  1,  0,  -4, -2, -5, -5, -2, 0,  1,  0,  4,   2,  0,  -1, 5,  3,  0,  0,  -3, -1, 2,  1,  3,   1,  1,   -3, 2,  3,  1,  -3, 0,  -1, -1, 6,  4,
    -1, 2,  0,  -3, 1,  1,  1,  -1, -1, 2,  -4, 0,  2,  0,   0,  -1, -1, -3, 0,  -2, 3,  1,  2,  2,  0,  0,   -2, -3,  1,  5,  0,  2,  -1, 3,  0,  1,  2,  3,
    5,  3,  2,  3,  2,  0,  0,  2,  1,  -5, -2, -2, -2, -4,  -3, -3, -2, 2,  0,  0,  0,  2,  5,  0,  2,  -3,  0,  -1,  0,  2,  -1, 1,  2,  3,  1,  4,  0,  1,
    0,  4,  3,  1,  3,  0,  0,  0,  2,  5,  4,  2,  1,  -1,  0,  0,  1,  0,  0,  -5, -4, -5, -5, 1,  1,  3,   0,  2,   2,  -3, 2,  1,  0,  0,  -1, -1, 2,  5,
    2,  -1, 1,  5,  -4, 2,  4,  2,  -2, 1,  -3, 0,  4,  2,   -1, 0,  -1, -2, 0,  0,  -3, -2, 2,  4,  -1, 2,   1,  -1,  -1, 0,  1,  -2, 0,  0,  1,  -2, 2,  0,
    -1, 1,  -2, 0,  0,  0,  3,  5,  3,  3,  1,  0,  0,  3,   3,  4,  4,  2,  2,  2,  2,  6,  -2, -3, 2,  0,   -2, -1,  -7, -3, -1, 2,  2,  0,  3,  3,  3,  2,
    2,  -2, -1, -2, -1, 2,  -1, 1,  5,  3,  0,  2,  -1, 1,   -1, 4,  -3, 2,  1,  0,  3,  2,  1,  3,  3,  3,   0,  0,   1,  1,  4,  -1, -2, -3, -2, -4, -3, -2,
    3,  5,  1,  3,  -1, -1, 3,  1,  2,  0,  -1, 1,  0,  1,   0,  -1, 1,  3,  2,  0,  2,  -1, 0,  1,  -4, -1,  3,  2,   0,  1,  -1, -4, -2, -2, 0,  3,  -2, -1,
    -2, 0,  3,  -1, -2, 1,  3,  -1, -1, -3, 0,  -2, 2,  1,   0,  0,  -4, -3, 3,  -1, 0,  1,  -2, 1,  0,  0,   2,  1,   2,  5,  5,  0,  3,  1,  2,  3,  -2, -3,
    4,  -1, -4, -5, -5, -2, -1, 2,  1,  -1, 5,  5,  3,  3,   3,  -3, -1, 0,  -2, 1,  -3, 3,  2,  4,  4,  2,   2,  1,   -2, 2,  0,  3,  0,  -2, 3,  1,  4,  5,
    3,  2,  1,  1,  3,  -1, 2,  0,  -1, -6, -2, 0,  -3, -1,  5,  3,  2,  0,  -2, -2, 2,  0,  -1, -1, -1, -1,  2,  5,   1,  -1, 2,  2,  0,  0,  2,  -2, -2, 2,
    -2, 0,  3,  3,  -1, -1, 1,  -5, -2, -1, 1,  1,  -1, -2,  -1, 0,  1,  -1, 0,  3,  3,  -1, -5, -2, 1,  2,   6,  2,   1,  -1, -1, -3, 1,  -2, -1, 3,  2,  4,
    0,  -1, 2,  4,  2,  0,  -1, 1,  1,  1,  2,  7,  1,  -1,  2,  -1, 1,  -2, -4, -1, -1, 0,  2,  2,  2,  2,   2,  1,   2,  -4, -3, -2, -1, 1,  -2, 4,  2,  4,
    3,  3,  1,  1,  2,  6,  -2, 1,  3,  -2, 1,  3,  5,  4,   5,  3,  6,  2,  0,  -2, 1,  1,  -2, -2, -1, -3,  -4, -2,  4,  4,  -1, 3,  -2, 1,  1,  -1, 1,  -3,
    0,  -4, 0,  3,  3,  4,  -6, 4,  -3, 1,  -3, 0,  0,  -4,  -2, -2, -2, 1,  2,  5,  -1, -2, -2, -1, 0,  1,   2,  1,   -1, -1, -2, -4, 4,  2,  3,  -4, 1,  -2,
    0,  1,  -1, -1, -1, -1, 3,  0,  0,  -2, 6,  3,  0,  3,   3,  2,  2,  0,  0,  5,  1,  2,  -1, -2, 4,  -3,  -2, -5,  -4, 3,  -1, 2,  2,  -2, 3,  2,  3,  2,
    4,  7,  3,  2,  7,  0,  0,  1,  3,  4,  3,  -2, 0,  4,   -1, 1,  2,  1,  0,  4,  3,  -1, 3,  1,  -5, -1,  -2, -1,  -1, 1,  -2, 2,  3,  3,  -3, -2, -2, 1,
    0,  -3, -1, 0,  -1, 4,  0,  0,  0,  2,  3,  -2, 4,  -4,  -1, -4, -1, 3,  2,  5,  -5, 1,  -3, 0,  -3, 4,   -2, -5,  0,  1,  -2, -1, 0,  4,  2,  -3, -2, 1,
    0,  4,  2,  0,  -1, -1, -3, -4, 1,  2,  1,  -2, -2, -2,  0,  0,  0,  0,  3,  0,  1,  1,  0,  -2, 5,  1,   3,  5,   1,  0,  2,  -1, -2, 0,  2,  1,  1,  1,
    0,  -3, -2, -4, -3, 6,  -2, 2,  1,  0,  4,  2,  2,  3,   5,  3,  4,  1,  5,  -2, 3,  2,  1,  3,  6,  -4,  -3, 5,   2,  1,  -1, 1,  2,  3,  2,  -2, 2,  2,
    -3, -2, -1, -3, 2,  2,  -1, 3,  1,  2,  -4, -2, -2, 3,   1,  2,  0,  2,  0,  4,  -1, 0,  5,  3,  1,  -2,  3,  -1,  -1, -2, -2, 1,  2,  3,  -3, -1, 0,  -1,
    1,  4,  -1, -1, -2, 1,  -4, -2, 1,  3,  1,  -2, 0,  -1,  -1, 1,  3,  0,  -2, -1, -1, -4, 2,  4,  1,  -3,  -3, -4,  -1, 1,  1,  -2, 4,  -1, 2,  1,  -1, -1,
    5,  3,  2,  4,  3,  -1, 3,  1,  -2, 0,  0,  3,  0,  1,   -1, 0,  -3, -2, -4, 0,  -1, 0,  1,  -1, 3,  0,   2,  2,   5,  2,  2,  4,  4,  1,  2,  1,  2,  3,
    2,  -2, 0,  3,  3,  3,  -1, 1,  4,  4,  1,  0,  1,  1,   0,  -3, -3, -1, 0,  2,  0,  -2, -1, 2,  -3, 0,   -1, 2,   -2, 1,  1,  1,  2,  6,  -2, 2,  -1, 2,
    2,  -3, -1, -2, 1,  0,  -2, -1, 2,  2,  0,  -2, 3,  -1,  -2, -1, -2, -4, -2, 1,  -4, -2, -2, 3,  -1, 0,   3,  1,   0,  0,  6,  -2, -2, -3, -1, -3, 3,  4,
    2,  -4, -3, -4, 0,  0,  -2, -1, 3,  -2, 3,  1,  -5, -2,  4,  1,  2,  5,  5,  1,  1,  0,  -3, -1, -1, 1,   0,  6,   1,  0,  -4, -3, -4, 2,  0,  1,  0,  1,
    1,  2,  1,  1,  3,  0,  4,  2,  5,  4,  4,  3,  2,  1,   0,  -2, -3, 2,  4,  0,  0,  2,  2,  2,  0,  -1,  -3, 4,   0,  1,  -2, -2, -4, 0,  -2, 2,  0,  3,
    0,  -3, -1, 1,  2,  0,  0,  3,  0,  5,  -2, 0,  0,  3,   3,  -3, -1, -1, 1,  -3, -1, 2,  0,  3,  2,  1,   -1, -4,  -1, -2, 0,  -2, -1, 0,  -5, -2, -2, 1,
    0,  0,  -1, -2, 3,  0,  3,  2,  1,  -3, 1,  -3, 1,  4,   5,  -3, -3, 0,  0,  -1, -3, 2,  2,  0,  2,  1,   -1, -1,  5,  2,  3,  5,  2,  1,  2,  0,  1,  -1,
    -2, 2,  -2, 2,  1,  1,  -5, -4, 0,  0,  -1, 1,  2,  -1,  -1, -1, 0,  1,  6,  1,  4,  2,  4,  0,  3,  2,   1,  4,   -1, -3, -3, 3,  3,  0,  -2, 2,  3,  3,
    3,  0,  1,  2,  -1, -1, -3, 1,  -2, 1,  -2, -1, -2, 3,   -1, -2, -2, -2, 0,  -1, -1, 1,  -1, 2,  1,  1,   -2, 2,   1,  1,  5,  0,  -1, 3,  4,  4,  2,  1,
    -1, 2,  1,  -1, -2, 0,  0,  3,  0,  2,  2,  3,  -2, 4,   -1, 1,  -1, 1,  -1, -2, -1, -2, -3, 1,  0,  3,   -2, 1,   2,  6,  3,  -2, -3, -3, 0,  4,  -1, -3,
    0,  -3, -1, 0,  -1, 3,  1,  3,  2,  1,  0,  0,  -1, 3,   2,  1,  -1, -1, 0,  0,  0,  1,  2,  2,  0,  0,   4,  2,   1,  2,  4,  1,  -1, -1, 1,  0,  1,  -1,
    5,  -3, -1, -3, 1,  2,  2,  2,  -2, -2, -3, -2, 1,  -2,  2,  -4, 1,  0,  -3, 7,  3,  4,  1,  0,  2,  1,   -1, -1,  -1, 0,  -1, -5, 3,  0,  5,  1,  0,  -2,
    0,  2,  -1, 1,  1,  -1, 2,  -4, 1,  5,  3,  3,  2,  1,   -4, -3, 1,  -2, 0,  -1, 3,  3,  -1, 2,  4,  1,   0,  1,   -1, 1,  -3, 0,  -3, -5, 0,  -1, -2, 3,
    1,  3,  -1, 0,  -2, 6,  2,  -3, -4, -1, 3,  4,  -1, -2,  0,  -1, -3, -1, -2, 2,  3,  1,  0,  -1, 2,  0,   0,  2,   1,  3,  -1, -4, -1, 0,  1,  2,  -1, 4,
    -1, 2,  4,  4,  2,  3,  0,  1,  -1, 2,  5,  5,  0,  1,   5,  0,  0,  2,  1,  -1, 0,  -4, -5, 0,  -4, -2,  5,  -1,  1,  1,  0,  2,  -3, 5,  5,  5,  3,  2,
    4,  -1, -1, -3, -1, 2,  0,  -2, 4,  1,  2,  2,  -1, -4,  0,  3,  -1, 5,  0,  -3, 4,  -3, 2,  6,  6,  2,   0,  0,   -5, 0,  1,  -4, -3, -2, 3,  2,  1,  0,
    5,  2,  -1, 3,  -4, 4,  0,  1,  -2, -1, 1,  1,  0,  1,   1,  3,  0,  0,  2,  4,  2,  1,  -2, -1, 5,  4,   2,  -1,  -1, 0,  -3, -1, 1,  2,  1,  2,  0,  1,
    1,  1,  1,  2,  -1, 3,  0,  2,  -3, -1, 2,  2,  1,  1,   1,  0,  0,  1,  -3, 4,  1,  -3, 1,  3,  3,  2,   -2, -2,  5,  -2, 1,  0,  -2, -2, -1, 0,  0,  1,
    -2, -2, 0,  0,  2,  -2, -1, 1,  -2, 1,  7,  3,  4,  5,   3,  1,  -1, -2, -1, -1, -3, -2, 3,  0,  3,  0,   -2, -3,  -1, 3,  0,  2,  1,  -2, 4,  -3, 2,  6,
    3,  -1, 2,  3,  -1, -2, 4,  -1, -2, -2, 3,  2,  2,  1,   4,  2,  -1, 1,  0,  1,  -1, -3, -4, -2, 3,  -3,  -2, 2,   1,  3,  0,  -1, -2, 4,  2,  -4, -2, -2,
    0,  0,  -1, -4, 1,  0,  -1, -3, 3,  1,  4,  2,  0,  1,   1,  1,  1,  2,  -2, 1,  0,  1,  1,  -1, 3,  1,   1,  -1,  1,  -1, -1, 1,  2,  3,  -1, 1,  4,  0,
    -1, 2,  -2, -1, 4,  1,  1,  -1, 0,  1,  0,  5,  0,  -1,  -4, -2, -1, 0,  0,  -2, -3, 0,  -2, 4,  2,  2,   4,  1,   4,  1,  0,  -3, -3, -1, -3, -2, 2,  0,
    3,  -1, -3, -1, -2, 0,  1,  2,  0,  0,  3,  -2, -1, 4,   1,  2,  4,  4,  3,  -1, 0,  -4, -5, 1,  5,  2,   1,  1,   3,  0,  -2, 1,  -1, 1,  -2, -2, -5, 1,
    4,  -6, -1, 1,  2,  2,  -1, -1, 0,  2,  4,  1,  -2, 0,   0,  1,  0,  -1, -1, 1,  0,  -1, 4,  2,  2,  4,   0,  1,   2,  -1, 2,  1,  1,  -2, -2, 2,  -2, 0,
    2,  3,  -1, 0,  1,  -2, -2, 3,  -2, 3,  -3, -2, 3,  0,   -3, 1,  -2, 0,  4,  1,  -3, 0,  -1, 3,  3,  2,   0,  -1,  -2, -3, -1, -1, -3, -4, -4, 0,  -1, 4,
    1,  2,  2,  -2, 3,  2,  0,  -4, -1, -1, 0,  0,  4,  1,   3,  -2, -5, 2,  1,  0,  3,  1,  1,  -2, 2,  0,   4,  -1,  1,  -3, 0,  2,  1,  -2, 6,  -1, -2, 3,
    3,  3,  4,  3,  1,  -1, 0,  0,  2,  -5, -1, 0,  1,  -1,  0,  2,  3,  1,  2,  -1, -2, 2,  0,  2,  5,  -1,  3,  4,   3,  4,  1,  0,  1,  2,  -1, -1, 2,  -4,
    1,  2,  -1, -1, 0,  -2, 2,  3,  1,  0,  -2, 0,  5,  2,   2,  -3, 6,  0,  -2, 4,  0,  2,  -1, -3, 0,  -1,  2,  -1,  5,  1,  5,  1,  -1, 1,  4,  -1, -1, -2,
    -3, 2,  -3, 4,  2,  0,  0,  2,  1,  -4, -2, 0,  -3, -1,  1,  -1, 1,  1,  1,  4,  4,  6,  -1, -4, -2, -4,  -4, -3,  -1, -1, -2, 4,  1,  3,  1,  1,  1,  -1,
    2,  0,  2,  0,  -1, -3, 0,  1,  0,  -1, 3,  2,  0,  0,   3,  4,  3,  5,  -1, 0,  -2, 1,  3,  -3, 1,  -1,  3,  0,   1,  2,  2,  0,  1,  0,  -2, 0,  1,  3,
    1,  0,  3,  2,  1,  4,  1,  0,  2,  3,  -1, -2, -1, -4,  1,  1,  1,  -1, 4,  -2, 1,  4,  1,  4,  0,  2,   3,  2,   4,  -3, 4,  0,  -3, 2,  -3, 1,  3,  -3,
    0,  -1, -1, -1, 3,  -3, 3,  -2, -1, 2,  5,  0,  1,  -4,  -2, 4,  -4, 5,  -1, 1,  -1, 1,  -1, -2, 0,  1,   -2, -3,  1,  -3, 1,  3,  4,  7,  1,  4,  2,  -4,
    -4, -4, -7, -2, 0,  -2, 1,  3,  1,  2,  4,  3,  4,  -2,  3,  -1, 0,  -3, 1,  -2, 1,  2,  1,  0,  2,  2,   -4, 2,   5,  2,  4,  2,  0,  1,  1,  -1, 3,  -2,
    3,  0,  4,  0,  2,  1,  1,  2,  -2, 0,  -3, 2,  1,  1,   3,  2,  4,  1,  3,  1,  -1, 0,  2,  4,  -1, -2,  0,  -4,  3,  2,  0,  -2, 2,  0,  2,  2,  1,  5,
    -1, 2,  2,  0,  3,  -1, 3,  1,  -1, 2,  0,  1,  3,  -1,  -2, -3, -3, -2, 2,  1,  0,  -4, -2, -1, 6,  0,   0,  -5,  -4, 3,  -6, 4,  -1, 2,  0,  0,  -1, -4,
    -2, -1, 2,  -1, 1,  -2, -1, 3,  4,  7,  -1, 1,  1,  -5,  -2, 0,  -2, -4, 0,  -2, 1,  3,  2,  0,  4,  1,   0,  -3,  3,  -3, 0,  2,  0,  0,  2,  2,  -1, -1,
    5,  2,  -2, 2,  3,  2,  2,  5,  0,  1,  4,  -1, 0,  -2,  1,  0,  3,  -1, 2,  1,  1,  -2, -1, -1, -2, 2,   -1, 3,   5,  2,  3,  2,  5,  1,  -1, 4,  1,  4,
    0,  -1, -1, -1, 0,  1,  -1, 1,  1,  -1, 1,  1,  1,  4,   0,  3,  5,  1,  2,  3,  4,  3,  2,  6,  1,  1,   -1, -2,  -1, -2, 0,  -3, 0,  0,  0,  -7, -3, 0,
    4,  1,  1,  -3, -4, 0,  -3, 4,  -2, 0,  2,  1,  3,  -4,  -1, -2, 4,  0,  2,  -1, 0,  3,  4,  3,  -1, 0,   0,  -4,  -3, 0,  -7, -1, 2,  -3, 1,  6,  4,  2,
    6,  3,  0,  -4, 0,  -3, 2,  2,  -1, 0,  2,  2,  1,  -1,  4,  0,  -4, 1,  4,  2,  2,  4,  0,  4,  2,  -2,  2,  0,   4,  -1, 4,  2,  1,  0,  1,  3,  -1, -1,
    0,  3,  -2, 2,  6,  0,  4,  6,  4,  2,  -2, 2,  1,  5,   -1, 0,  -1, 0,  0,  1,  0,  -1, 1,  1,  1,  2,   4,  4,   -1, 5,  6,  0,  0,  2,  6,  3,  -1, 3,
    1,  3,  1,  1,  4,  -3, -2, -3, 1,  2,  1,  -6, -1, -1,  5,  2,  1,  -3, -5, 3,  0,  3,  -1, 3,  2,  0,   1,  -2,  0,  0,  1,  0,  0,  0,  0,  5,  2,  7,
    -1, 2,  -1, -4, -3, -2, -3, -4, -1, -3, -1, 6,  -1, 0,   0,  -5, -4, 2,  -7, 0,  0,  0,  2,  -2, -1, -1,  3,  3,   1,  0,  -2, -1, 1,  -1, -3, -3, -3, -2,
    -3, 1,  -1, 2,  1,  3,  3,  2,  -2, -1, 0,  0,  -1, 1,   4,  -1, -2, 3,  5,  5,  0,  0,  1,  5,  -1, 1,   6,  -2,  1,  2,  3,  -3, 3,  4,  1,  1,  -1, -1,
    0,  -2, 2,  0,  -1, 3,  1,  1,  -1, 4,  0,  -1, 1,  0,   2,  -1, -4, 6,  1,  -1, 4,  1,  2,  5,  -1, 4,   0,  -3,  -2, 0,  1,  6,  -2, -4, 0,  0,  0,  2,
    6,  3,  1,  4,  0,  0,  0,  -1, 0,  6,  0,  0,  -1, 0,   6,  0,  0,  2,  6,  0,  -5, -2, 0,  1,  -1, 0,   -1, 5,   -1, -3, -4, 2,  -5, 1,  -2, 1,  -1, -2,
    -3, -1, 0,  1,  1,  -1, -2, -1, 1,  -3, -1, -4, -2, -2,  -3, 2,  0,  0,  -1, 4,  5,  4,  -1, 0,  -1, -1,  1,  -1,  3,  -2, 2,  0,  4,  4,  3,  2,  -2, 4,
    0,  3,  2,  0,  -1, 2,  2,  2,  3,  2,  2,  2,  -3, -1,  0,  -4, 0,  1,  -2, 1,  4,  4,  3,  3,  1,  1,   -2, -2,  -3, -3, -5, 5,  4,  1,  2,  0,  2,  -1,
    2,  5,  -3, -1, -2, 1,  4,  4,  -2, -4, -1, -2, 0,  4,   5,  2,  3,  2,  -2, 1,  -4, 0,  5,  5,  -2, -1,  -1, -1,  1,  1,  2,  0,  3,  2,  -4, -1, -2, -2,
    0,  -5, 0,  4,  1,  -3, -3, 3,  -4, 3,  1,  0,  -1, -3,  -2, 1,  0,  3,  1,  -1, 0,  0,  3,  -2, -1, -3,  -3, -4,  -2, 2,  -1, 1,  2,  3,  1,  2,  -2, -1,
    -1, 1,  0,  0,  3,  1,  0,  1,  4,  5,  3,  -1, 0,  3,   0,  1,  1,  -2, 2,  1,  4,  -1, 4,  1,  2,  2,   -2, 0,   0,  0,  2,  3,  -1, 1,  -2, 0,  3,  3,
    1,  -4, -3, -3, -1, -3, 1,  9,  5,  1,  0,  -1, 2,  1,   1,  5,  0,  1,  -1, -3, 6,  5,  1,  -2, -4, -4,  -2, 2,   4,  3,  2,  2,  1,  0,  -2, 1,  5,  2,
    -2, -2, 3,  0,  2,  3,  1,  1,  3,  3,  -1, 1,  0,  2,   2,  -4, 0,  2,  1,  -1, -5, -1, -4, 4,  3,  -1,  -1, -1,  -2, -1, -3, 1,  2,  0,  2,  -1, 3,  -2,
    -2, -3, -2, -1, -3, 1,  -1, 1,  3,  4,  4,  3,  0,  2,   -2, -1, 1,  1,  0,  0,  -2, -1, 3,  3,  2,  -1,  -1, 3,   2,  3,  1,  -3, 1,  1,  2,  3,  3,  0,
    2,  0,  -3, -2, -1, -3, 2,  5,  -2, -2, 1,  -2, 2,  1,   2,  2,  0,  -4, -3, -4, -1, 8,  4,  2,  0,  -2,  -1, 0,   0,  4,  0,  2,  1,  -2, 3,  4,  1,  -3,
    -3, -2, -5, -1, 2,  2,  3,  2,  2,  0,  -2, 3,  -1, 1,   0,  1,  2,  3,  5,  2,  0,  1,  6,  5,  -1, -2,  -1, 3,   3,  -2, 0,  -1, 3,  -1, -6, 3,  -3, 4,
    2,  -1, 0,  -3, -1, 1,  0,  -2, 3,  -1, 3,  -1, 1,  2,   -3, -2, -2, -1, -3, -5, 0,  -1, 2,  2,  4,  3,   -1, -2,  -2, -1, 2,  0,  1,  2,  2,  -2, 3,  1,
    3,  -4, 0,  5,  4,  3,  5,  -4, 1,  -2, 1,  3,  1,  2,   3,  1,  0,  0,  -2, -6, 5,  5,  0,  0,  0,  1,   2,  1,   -3, 0,  -3, -3, -1, -5, 1,  11, 4,  -1,
    4,  -1, 1,  1,  0,  5,  2,  2,  1,  -4, 2,  3,  4,  -2,  -5, -1, -2, -2, 4,  3,  2,  4,  4,  0,  -4, 1,   -1, 3,   0,  -1, 4,  2,  4,  5,  -1, 0,  5,  4,
    -1, 2,  0,  1,  2,  -6, -2, 2,  1,  0,  0,  -5, 1,  0,   -2, 2,  0,  0,  2,  1,  1,  -1, 1,  6,  2,  3,   0,  0,   0,  -1, 0,  2,  -1, 1,  1,  4,  2,  -3,
    0,  -1, 1,  2,  0,  2,  -2, 6,  6,  -2, 0,  0,  -2, 0,   3,  3,  4,  -1, -2, 2,  -4, 2,  1,  -1, 0,  0,   -1, 1,   2,  2,  -1, 2,  1,  2,  -1, 0,  3,  -1,
    2,  1,  0,  -1, -3, -1, 0,  4,  2,  -1, 2,  -2, -2, -1,  0,  4,  2,  -1, 6,  1,  -2, 3,  0,  -1, -3, 1,   2,  6,   2,  -1, 4,  3,  2,  2,  2,  2,  -2, 0,
    0,  0,  0,  -1, 1,  -1, 1,  1,  3,  0,  -1, -2, 0,  0,   -1, 0,  3,  3,  1,  0,  0,  3,  2,  -1, -1, 0,   1,  0,   0,  1,  2,  0,  1,  4,  1,  0,  1,  4,
    4,  1,  2,  0,  -1, -2, 0,  1,  -2, -1, 3,  2,  1,  -4,  0,  0,  2,  0,  0,  -2, -1, 8,  5,  -2, 1,  -1,  -3, 2,   1,  4,  7,  -4, -3, 3,  -2, 3,  1,  0,
    -2, 0,  -1, 1,  1,  1,  -1, 3,  1,  -1, -2, 0,  2,  3,   -1, 1,  0,  1,  -1, 0,  2,  3,  1,  -3, 2,  0,   -4, 0,   -1, 5,  1,  1,  4,  -1, 2,  2,  -1, -1,
    1,  3,  3,  7,  0,  -1, 2,  2,  1,  -3, 0,  2,  -2, -1,  1,  -2, 1,  0,  0,  0,  0,  -1, 1,  1,  0,  -1,  2,  1,   -1, -1, 3,  4,  2,  0,  0,  0,  2,  -1,
    0,  -2, 1,  0,  -2, 3,  2,  -1, 0,  3,  2,  -2, 1,  1,   2,  3,  1,  2,  -2, -1, 2,  1,  -1, 0,  1,  3,   0,  -2,  -1, -1, 0,  2,  -2, -2, -1, 8,  9,  -3,
    -2, 1,  -2, 2,  -1, 4,  7,  -2, -3, 2,  0,  2,  1,  -1,  -3, 1,  -2, 2,  1,  1,  2,  1,  2,  1,  1,  -2,  0,  0,   1,  3,  0,  0,  -1, 0,  0,  0,  0,  -1,
    0,  1,  -3, 2,  -1, 4,  0,  -3, 2,  0,  0,  2,  0,  1,   1,  3,  3,  5,  -2, 1,  3,  0,  -1, -1, -3, 4,   -2, 2,   -4, 1,  -2, -1, 0,  1,  2,  0,  0,  0,
    0,  -1, -1, 1,  -3, 0,  3,  3,  4,  -1, 0,  3,  2,  0,   -2, 0,  -3, -1, -1, -3, 1,  2,  0,  2,  4,  -2,  -1, -1,  3,  4,  -1, -1, 0,  -2, 4,  0,  -2, 0,
    2,  3,  -1, -2, 1,  -2, 1,  0,  -1, 0,  0,  5,  7,  0,   -1, 0,  -1, 1,  1,  5,  7,  -2, 2,  0,  1,  0,   1,  0,   -3, 2,  -1, 1,  1,  -1, 1,  1,  3,  1,
    0,  1,  1,  0,  1,  5,  1,  0,  0,  0,  0,  1,  -3, -1,  0,  1,  1,  1,  3,  6,  0,  1,  1,  1,  -1, 4,   -1, 1,   -1, 2,  2,  4,  1,  1,  3,  0,  1,  0,
    -1, 5,  0,  1,  -2, -2, -1, -1, 1,  0,  3,  -1, -1, 1,   0,  -1, 2,  -1, -1, 1,  2,  3,  4,  1,  0,  0,   1,  0,   -4, 0,  0,  0,  0,  -5, 1,  4,  -1, -2,
    3,  -2, -2, 1,  2,  4,  0,  -1, 0,  -2, -1, 0,  -1, 1,   0,  3,  2,  -1, 0,  -2, 0,  0,  3,  0,  -2, 2,   6,  -1,  3,  0,  -2, 2,  1,  5,  6,  -3, 2,  1,
    2,  1,  0,  4,  -2, 0,  -2, 2,  0,  1,  0,  1,  2,  -1,  0,  0,  -1, 2,  1,  2,  0,  -1, 0,  -1, 0,  2,   -1, -2,  4,  -2, 0,  1,  0,  3,  0,  2,  4,  0,
    -3, 3,  2,  3,  2,  2,  5,  4,  1,  -1, 2,  -3, 0,  1,   -4, 2,  -2, 0,  -2, -2, -1, -1, -1, 1,  3,  1,   -2, 2,   -2, -1, 2,  2,  0,  2,  2,  4,  4,  -1,
    0,  4,  0,  -2, -3, 0,  3,  -5, 0,  -7, -3, 3,  0,  -4,  -3, -2, -2, 1,  -5, 3,  1,  -1, 8,  6,  -1, -1,  0,  0,   1,  0,  -1, 3,  -1, 1,  0,  3,  1,  -4,
    2,  3,  -4, 3,  -3, -2, 2,  1,  0,  -4, 1,  -1, 4,  4,   3,  4,  5,  5,  3,  5,  1,  2,  0,  -1, -1, 0,   0,  2,   0,  0,  1,  1,  -2, 1,  2,  -2, -6, -2,
    3,  6,  4,  -1, 5,  2,  -2, 1,  3,  -1, -3, 1,  -6, -2,  -1, -1, 3,  1,  2,  8,  0,  1,  2,  5,  3,  3,   -2, -2,  -2, 4,  3,  -2, 1,  3,  0,  3,  0,  -1,
    -2, 0,  0,  0,  0,  -1, 2,  -1, 1,  2,  0,  1,  -1, -2,  -2, 4,  1,  1,  -4, 1,  3,  0,  0,  -5, -5, 0,   -1, -3,  -3, -6, -1, 1,  -5, 3,  2,  -3, 2,  5,
    -4, 1,  -1, 0,  2,  -1, -2, 3,  0,  0,  3,  5,  2,  -2,  1,  5,  0,  2,  1,  0,  2,  1,  -1, 1,  2,  3,   1,  3,   3,  6,  2,  2,  4,  6,  2,  2,  -2, -1,
    0,  1,  0,  3,  3,  -1, 3,  -1, -2, 0,  3,  -2, -7, -4,  5,  2,  2,  2,  6,  2,  -3, 1,  5,  -2, -4, 2,   -4, -3,  -1, -1, 1,  -1, -1, 6,  1,  -1, 0,  5,
    2,  4,  0,  0,  -2, 1,  0,  2,  -1, 4,  5,  3,  -2, 1,   -4, 0,  2,  2,  2,  1,  1,  0,  1,  2,  -2, 1,   1,  -2,  -1, 4,  2,  5,  -4, 0,  -1, 1,  0,  -4,
    -4, -3, 1,  -1, -2, -1, 0,  1,  -5, 1,  1,  -1, 4,  8,   -6, 1,  0,  0,  -1, -4, 1,  1,  0,  -1, 0,  4,   -2, 1,   -1, 2,  -2, 1,  1,  0,  2,  1,  -1, -2,
    3,  -2, -1, 0,  5,  5,  1,  3,  6,  4,  2,  -2, -2, -2,  1,  0,  -1, -2, 2,  0,  3,  -2, -2, -1, 2,  -3,  -5, -6,  4,  3,  1,  2,  -1, 0,  -2, -1, 2,  -2,
    -5, 3,  -5, -1, -1, 0,  -1, -2, -3, 3,  2,  0,  -1, 1,   2,  1,  -2, 1,  0,  -1, 1,  -2, -1, 5,  0,  3,   1,  -2,  -2, 2,  1,  -1, 3,  1,  -1, -2, 1,  2,
    -1, 2,  1,  -2, -2, 0,  3,  0,  1,  0,  2,  1,  0,  -2,  -7, -1, 2,  -3, 2,  -6, 3,  5,  -3, -2, 3,  2,   2,  4,   -4, 4,  0,  1,  0,  -2, -1, -2, -1, 2,
    4,  5,  0,  -5, -1, 4,  -3, 2,  2,  1,  1,  0,  -1, 2,   2,  -1, 3,  0,  6,  4,  2,  2,  5,  4,  -1, 0,   1,  -1,  1,  1,  0,  1,  3,  0,  4,  2,  -2, 1,
    4,  1,  -6, -1, 2,  2,  0,  1,  3,  3,  0,  -2, 5,  1,   -5, 4,  -2, 0,  -3, 3,  3,  -1, 0,  8,  2,  -1,  -2, -2,  1,  -1, -1, 4,  2,  2,  5,  -1, -2, 3,
    1,  1,  -4, -3, -1, 2,  3,  2,  2,  4,  2,  -4, 2,  2,   -2, -1, 2,  -2, -1, -1, 4,  1,  -2, 0,  2,  -2,  3,  -3,  -6, -1, 2,  -4, -3, -5, 2,  3,  -4, -1,
    -1, -3, 8,  4,  -1, 2,  -3, 0,  1,  0,  -2, -3, -1, 1,   4,  5,  0,  -4, 1,  2,  2,  -1, 2,  -2, -1, 2,   1,  -1,  0,  0,  1,  0,  6,  1,  1,  1,  2,  6,
    1,  -2, 0,  -1, 1,  0,  1,  -2, 1,  -1, 2,  -3, -4, -1,  3,  1,  -6, -3, -1, 3,  0,  0,  -1, 4,  -1, 1,   5,  -4,  -3, 3,  -1, 2,  -1, 0,  0,  -3, -1, 5,
    3,  0,  2,  -2, 0,  -1, -3, 0,  3,  -1, 5,  -3, 3,  3,   -1, 1,  -3, -1, -3, 1,  2,  3,  1,  4,  2,  -1,  2,  2,   0,  0,  -1, -1, 2,  2,  0,  -4, -3, 3,
    -3, 1,  -2, -5, 0,  1,  2,  -1, 1,  4,  2,  -3, 0,  -2,  3,  0,  0,  2,  1,  -2, 1,  0,  0,  0,  3,  1,   -1, 5,   1,  -3, -2, 3,  -1, 1,  0,  0,  0,  -1,
    3,  -2, 1,  1,  2,  0,  2,  1,  1,  4,  1,  -1, -1, -1,  -4, -1, -3, -2, 1,  -3, 0,  0,  2,  -3, -3, 2,   0,  0,   0,  -1, 1,  0,  -1, 0,  1,  -1, 4,  1,
    -1, 0,  2,  1,  -1, 1,  0,  2,  -5, -1, 0,  -4, -2, 1,   -2, -3, -2, -2, -3, 2,  1,  1,  -1, -2, 2,  4,   -2, 2,   1,  5,  5,  1,  1,  5,  2,  -2, -3, 0,
    1,  0,  1,  3,  4,  3,  2,  0,  3,  2,  0,  -4, -2, 4,   -2, 4,  -3, -3, 1,  0,  1,  1,  1,  3,  4,  -3,  0,  0,   1,  -1, 1,  4,  -1, -1, 1,  -2, -1, 0,
    3,  -1, 0,  3,  1,  -2, -1, 3,  -3, 1,  0,  1,  -2, 1,   0,  1,  -1, 1,  3,  -1, 2,  0,  2,  5,  -1, -1,  0,  0,   -1, -1, -5, 0,  -1, -2, -2, 2,  3,  -2,
    1,  2,  1,  0,  -2, -2, 1,  -1, 0,  -2, 0,  0,  3,  1,   -1, 1,  5,  1,  -2, 1,  0,  1,  -1, 2,  0,  -3,  0,  0,   -1, -1, 0,  2,  -2, 2,  2,  2,  -2, -2,
    2,  3,  -4, 4,  1,  4,  5,  3,  0,  5,  4,  -1, -5, 0,   -1, 1,  3,  4,  3,  2,  4,  0,  2,  3,  -1, -2,  0,  0,   2,  -1, -3, -3, 0,  4,  -1, 2,  0,  1,
    5,  -1, 0,  1,  3,  1,  1,  2,  1,  0,  -1, -1, 4,  0,   3,  -1, 2,  3,  -2, 0,  0,  0,  0,  1,  1,  0,   -1, 2,   0,  -2, 0,  1,  3,  -2, -1, 1,  3,  0,
    -1, -1, 0,  -2, -3, 0,  -4, 0,  1,  -2, 1,  2,  3,  -2,  1,  2,  -1, 0,  0,  1,  0,  0,  2,  -1, -1, 1,   3,  2,   2,  1,  1,  1,  -3, 1,  -1, -1, -1, 0,
    0,  -5, 1,  -1, 1,  1,  -1, -2, 2,  2,  2,  4,  0,  -3,  2,  5,  -5, 2,  -4, 3,  4,  1,  0,  7,  4,  -3,  -3, 0,   1,  2,  2,  4,  2,  -2, 0,  0,  2,  0,
    -2, 0,  1,  1,  -1, 0,  -2, -1, -1, 5,  1,  1,  4,  1,   2,  3,  2,  1,  4,  3,  0,  0,  1,  0,  1,  1,   2,  1,   1,  -1, 3,  1,  1,  -1, 0,  2,  0,  -1,
    2,  -1, 0,  2,  0,  2,  0,  0,  3,  -2, 0,  2,  1,  2,   -3, 0,  1,  -1, -5, -1, -5, -1, 0,  -2, -2, 0,   2,  -2,  1,  1,  -2, 1,  2,  0,  -2, 0,  3,  -1,
    -1, 0,  2,  0,  2,  2,  2,  0,  -1, 2,  -1, -2, -1, 1,   -1, -3, -1, 1,  1,  0,  0,  -2, 3,  2,  -1, 3,   0,  -2,  1,  5,  0,  3,  -2, 3,  3,  1,  2,  4,
    2,  -1, -5, -3, 2,  0,  2,  4,  0,  -1, 0,  -3, 3,  -2,  0,  -1, 1,  2,  -1, 0,  -1, -3, 1,  6,  2,  1,   1,  3,   4,  2,  2,  0,  3,  3,  -1, 4,  0,  -1,
    1,  3,  4,  3,  4,  -1, 2,  1,  -1, 2,  -1, -2, 2,  0,   0,  1,  1,  3,  2,  0,  0,  1,  5,  0,  2,  2,   2,  2,   -3, -2, 3,  0,  -3, 1,  -4, -3, 0,  -2,
    2,  3,  1,  -3, 1,  2,  1,  4,  2,  2,  1,  2,  4,  1,   0,  5,  1,  0,  0,  -1, -1, 2,  -1, 0,  -4, 0,   1,  -1,  -1, -4, -2, -3, -3, -1, -1, -5, 4,  3,
    -2, 1,  -1, -3, 1,  6,  3,  3,  0,  2,  1,  1,  1,  6,   2,  -2, -7, -2, 2,  1,  0,  3,  1,  0,  -2, -1,  -2, 1,   2,  1,  2,  3,  -1, 1,  0,  -1, -2, 3,
    -2, -3, 0,  2,  -3, -3, 3,  -5, 0,  4,  -1, 0,  -1, 1,   -1, 3,  3,  3,  4,  0,  1,  0,  0,  0,  4,  -3,  1,  -2,  1,  0,  2,  0,  4,  0,  -2, -1, -3, 3,
    0,  1,  0,  -3, -2, 0,  -1, 1,  4,  2,  1,  0,  1,  0,   0,  5,  1,  2,  1,  5,  1,  0,  1,  4,  -1, 0,   0,  1,   -1, -2, 0,  1,  2,  -1, 4,  1,  2,  4,
    -1, 1,  1,  -1, 0,  -3, -2, 1,  -2, 3,  -1, -3, -1, -1,  -1, 3,  2,  3,  -1, 2,  2,  4,  1,  0,  3,  2,   -4, 0,   0,  -2, 0,  3,  1,  2,  7,  0,  0,  0,
    3,  3,  -1, -1, 1,  -2, 1,  3,  -3, 2,  0,  1,  -3, 3,   -1, 0,  -2, 0,  -3, -2, 3,  -2, 2,  0,  -3, -2,  -1, 3,   -1, 3,  1,  0,  5,  0,  2,  0,  -3, 1,
    2,  -2, 1,  0,  0,  1,  1,  0,  3,  -3, -2, -2, -3, 5,   0,  -2, -1, -2, -3, 1,  1,  -1, 4,  2,  -1, 0,   0,  -1,  4,  4,  1,  2,  -1, 4,  1,  -3, 2,  1,
    3,  1,  1,  -1, -2, 0,  2,  2,  5,  -1, 3,  5,  3,  4,   -2, -1, 0,  -2, 0,  -4, -2, 0,  0,  1,  -2, -1,  -3, -1,  -3, 4,  2,  0,  1,  -1, 2,  4,  -2, 0,
    1,  4,  -3, -3, -1, -2, 0,  3,  1,  3,  7,  2,  -1, -3,  3,  4,  0,  0,  -1, 0,  0,  1,  1,  3,  0,  1,   -3, -1,  -1, 1,  1,  0,  0,  -3, 0,  -3, 1,  2,
    0,  0,  -3, 4,  1,  6,  0,  0,  2,  2,  2,  -1, 2,  1,   3,  -4, 3,  1,  0,  1,  -1, 1,  4,  -2, -1, 0,   -2, 2,   -1, 0,  -1, -2, -4, 1,  1,  -1, 2,  1,
    1,  1,  0,  0,  6,  4,  1,  1,  1,  4,  0,  0,  3,  0,   1,  1,  0,  0,  -2, 0,  1,  1,  2,  -2, 3,  2,   4,  1,   -3, -1, -1, 2,  1,  -2, 1,  -1, -2, 1,
    -2, -1, -5, 1,  0,  3,  3,  -1, -1, 0,  3,  3,  -2, 3,   3,  1,  -1, -4, -1, -1, -1, -1, 3,  3,  7,  0,   0,  -3,  2,  4,  -1, 1,  -2, 1,  2,  2,  1,  1,
    -2, -1, -3, -1, -1, -1, 1,  1,  -3, -2, 0,  -4, 3,  2,   2,  -1, -4, 2,  2,  4,  2,  3,  2,  0,  0,  -2,  1,  2,   0,  -5, 1,  2,  0,  2,  1,  1,  5,  -2,
    0,  -3, -5, 1,  -1, 0,  0,  -1, -3, -3, 2,  3,  0,  0,   1,  0,  -1, 0,  6,  4,  1,  3,  1,  5,  0,  2,   6,  2,   0,  -1, -1, 1,  -3, 0,  1,  0,  3,  -1,
    1,  1,  4,  0,  -1, 2,  -1, 2,  1,  -2, 0,  -2, 1,  1,   -1, 0,  -6, 2,  2,  6,  4,  1,  2,  -1, 4,  2,   -3, 1,   2,  2,  1,  0,  -1, -1, -2, 2,  0,  0,
    6,  0,  2,  -3, 1,  5,  -3, 1,  1,  -3, 1,  2,  -2, -2,  0,  -1, 0,  0,  1,  0,  1,  0,  1,  -2, 2,  -3,  1,  -1,  2,  -1, 0,  7,  1,  1,  3,  -2, 4,  -1,
    -1, 0,  -2, 2,  2,  -5, 1,  1,  1,  0,  0,  2,  4,  -3,  -3, -2, -2, 0,  -1, -1, 0,  -1, -2, -2, 0,  1,   1,  0,   2,  2,  0,  1,  6,  3,  -1, 2,  1,  4,
    1,  0,  5,  3,  -1, 1,  -3, 1,  -3, 0,  1,  -3, 2,  2,   4,  4,  6,  1,  -2, -1, -1, 1,  3,  0,  0,  -1,  1,  -2,  0,  3,  -5, 1,  3,  3,  -1, -1, 1,  -1,
    1,  1,  -2, 2,  3,  2,  1,  -4, -5, -4, 0,  2,  0,  -1,  9,  1,  2,  -2, 2,  4,  4,  5,  3,  -1, 1,  2,   2,  5,   -2, -9, -1, 1,  -1, -1, -4, 3,  -1, -3,
    4,  4,  -1, -1, 5,  -2, -1, 3,  -2, 0,  0,  1,  1,  4,   0,  -1, -2, -5, -1, -1, 1,  0,  -1, 3,  0,  -2,  2,  0,   -1, 1,  -2, -2, -2, -1, -4, 0,  -1, -1,
    -2, 0,  3,  2,  -1, -3, 0,  3,  1,  3,  -2, 0,  4,  -8,  3,  -2, 2,  -1, -2, -2, 1,  -1, -3, 1,  5,  -2,  -3, 0,   2,  -3, -2, -2, 7,  0,  2,  3,  5,  3,
    6,  -1, -3, -1, 0,  2,  1,  -1, 6,  -2, 0,  -2, 2,  1,   4,  0,  5,  6,  4,  -1, 2,  4,  1,  0,  1,  5,   4,  -7,  -1, -1, -2, 0,  -4, -8, 7,  1,  3,  1,
    1,  1,  5,  2,  -1, -2, 0,  -2, 1,  -2, -2, 4,  2,  -5,  5,  1,  -1, 1,  6,  -1, -4, 4,  1,  -1, -1, 0,   1,  0,   0,  -1, -3, -1, 1,  0,  0,  2,  -1, 1,
    0,  -2, 0,  -1, -3, -1, -1, -2, 0,  -2, -2, 0,  -1, 3,   2,  0,  3,  3,  0,  -2, -1, 2,  3,  2,  -2, 1,   1,  -4,  0,  -2, 2,  0,  0,  1,  2,  0,  -2, 0,
    2,  -2, -2, 1,  3,  -2, -2, -3, 3,  0,  0,  4,  2,  3,   3,  -2, -1, 2,  1,  -1, 2,  2,  6,  -1, 2,  0,   2,  2,   7,  2,  5,  7,  4,  -1, -1, 2,  0,  2,
    -1, 1,  2,  -6, 0,  1,  2,  3,  -4, -2, 8,  0,  1,  -1,  2,  0,  2,  0,  0,  -2, 0,  -1, -1, 0,  -1, 7,   -1, -1,  3,  -3, 1,  -1, 5,  -2, -3, 3,  2,  -3,
    2,  2,  -1, 1,  0,  0,  0,  1,  2,  0,  0,  0,  -3, 0,   -1, -1, 0,  -2, -2, 0,  -4, -2, -6, -1, -2, 3,   3,  2,   3,  -1, 2,  0,  -2, -1, -2, -1, 1,  1,
    1,  0,  2,  -3, 3,  0,  1,  -2, -1, -3, 3,  -1, -5, 1,   -1, -2, -3, 4,  1,  -3, -2, -3, 0,  -2, 2,  3,   2,  0,   1,  -1, -4, 1,  -3, -2, -2, 0,  7,  -5,
    -3, -1, 3,  2,  2,  3,  5,  5,  0,  -2, 0,  3,  2,  2,   0,  1,  2,  -5, 1,  4,  2,  1,  -4, -1, 8,  2,   2,  -1,  2,  0,  -1, -2, 2,  -3, 0,  1,  -1, 2,
    0,  3,  -3, 0,  2,  -2, 0,  1,  5,  -1, 0,  3,  0,  0,   0,  1,  2,  0,  -2, -2, 2,  1,  -2, -2, -1, -3,  -2, 2,   -1, 4,  0,  -3, -3, 1,  0,  -4, -2, 1,
    -4, 3,  2,  2,  5,  0,  3,  2,  0,  -2, -1, -3, -1, 2,   2,  0,  5,  0,  4,  0,  0,  1,  0,  -2, -2, -1,  -5, 1,   0,  1,  -3, 1,  1,  -1, -2, -2, 1,  -1,
    1,  4,  2,  0,  1,  0,  -7, 1,  -3, -1, 1,  -1, 5,  -2,  2,  -2, 0,  1,  -1, 1,  6,  7,  6,  -3, -1, 2,   2,  -1,  2,  3,  0,  -6, 2,  4,  3,  1,  -5, -2,
    7,  1,  2,  0,  3,  -1, -1, 2,  -1, 1,  1,  2,  1,  2,   0,  3,  -2, 1,  1,  -3, 2,  4,  4,  -1, -2, 2,   3,  -1,  3,  -1, 3,  3,  -1, -2, 2,  -1, 2,  -2,
    -2, -1, -2, 0,  -1, 3,  3,  -1, -1, 1,  0,  -2, -1, 2,   -4, -1, 1,  2,  1,  -2, 5,  0,  0,  -1, 0,  1,   -1, 1,   5,  0,  2,  3,  1,  1,  0,  1,  -3, -3,
    2,  3,  -1, -1, 1,  -1, -3, 2,  1,  1,  0,  -4, 1,  -2,  2,  4,  2,  4,  3,  3,  -4, 5,  -2, -2, 4,  -1,  4,  2,   7,  3,  1,  2,  5,  4,  6,  7,  3,  -1,
    -2, 2,  0,  2,  0,  1,  -2, -6, 2,  5,  1,  1,  -4, 0,   -2, 4,  -1, -2, 0,  2,  0,  0,  -1, 1,  -1, -1,  2,  -4,  -1, 5,  -2, 0,  -1, 1,  1,  -2, 1,  2,
    -3, -7, -1, -2, -1, 1,  0,  0,  2,  3,  1,  0,  1,  0,   3,  -4, 6,  3,  1,  -1, 4,  2,  0,  0,  2,  -1,  1,  -1,  1,  1,  2,  2,  2,  0,  1,  2,  -3, -1,
    0,  0,  0,  -1, 1,  5,  -1, -4, 4,  2,  0,  -2, 4,  6,   4,  1,  -2, 2,  3,  -2, 4,  0,  1,  1,  6,  0,   1,  1,   0,  1,  0,  -2, 0,  -1, 0,  3,  0,  -2,
    -1, 2,  -1, 2,  5,  0,  3,  4,  1,  -1, -5, 0,  -2, -2,  2,  3,  3,  -2, -1, -2, -4, 2,  -5, -4, 2,  4,   3,  1,   -1, 6,  0,  -1, 0,  2,  1,  1,  -1, -1,
    1,  0,  1,  -3, -1, 4,  -1, -2, -1, 2,  -1, -3, 3,  0,   -1, -3, 1,  1,  -4, 2,  2,  1,  1,  0,  2,  1,   1,  0,   2,  -1, 6,  4,  3,  1,  8,  2,  1,  3,
    5,  1,  4,  -1, 2,  3,  2,  2,  2,  -2, 1,  3,  -4, -2,  1,  0,  0,  0,  -2, 4,  0,  -5, 6,  3,  -1, 0,   3,  4,   4,  2,  0,  1,  4,  2,  5,  0,  -1, 2,
    6,  -1, 0,  0,  1,  1,  2,  -3, 1,  2,  1,  3,  -1, -2,  2,  0,  -1, 2,  6,  1,  3,  3,  4,  0,  -5, -1,  1,  -1,  2,  1,  3,  -2, -1, 0,  -3, 1,  -2, -2,
    -1, 4,  4,  3,  0,  6,  -2, -1, -2, 2,  3,  1,  1,  1,   1,  0,  2,  -5, -2, 3,  0,  -3, -3, 1,  -1, 0,   -3, 4,   0,  -5, 1,  1,  -4, 4,  1,  1,  3,  1,
    1,  2,  0,  -1, 3,  -1, 6,  4,  0,  2,  4,  1,  1,  2,   4,  1,  1,  -2, 0,  1,  2,  2,  1,  -2, 1,  2,   0,  -3,  0,  1,  1,  2,  1,  4,  0,  -4, 4,  3,
    -3, -1, 3,  2,  4,  -1, -4, 1,  3,  1,  2,  -1, -1, 1,   6,  -3, -3, 0,  -2, 1,  -1, -2, 1,  2,  2,  4,   0,  1,   4,  -1, -3, 5,  6,  1,  3,  4,  4,  2,
    -4, -1, 0,  -1, 5,  2,  2,  -3, 0,  -2, -3, 2,  -2, -3,  0,  6,  4,  4,  -1, 6,  -1, 0,  3,  4,  -1, 2,   0,  -1,  -2, 1,  1,  -6, -1, 2,  1,  -5, -3, 0,
    -3, 1,  0,  2,  -1, -3, -2, 3,  -5, 0,  -1, -2, 2,  1,   0,  2,  -2, -2, 1,  -1, 4,  3,  3,  2,  4,  2,   0,  2,   1,  2,  -1, -1, 0,  2,  1,  2,  1,  -4,
    0,  0,  -1, -3, 0,  -2, 0,  0,  -1, 2,  -2, -1, 1,  3,   -3, -1, 3,  1,  3,  -1, -3, 0,  2,  2,  3,  0,   -2, 0,   7,  -3, -1, -2, -1, 0,  0,  -2, 2,  3,
    1,  3,  -3, -2, 5,  -2, -3, 2,  5,  -3, 3,  2,  2,  -2,  -4, -1, -3, -3, 4,  4,  -1, -4, 0,  -2, -2, 0,   -3, -5,  3,  2,  2,  3,  -3, 3,  -1, -3, 5,  2,
    -1, 0,  -1, 2,  -3, -1, 1,  -3, -1, 2,  3,  -4, -1, -2,  -2, 2,  0,  1,  -1, -4, 0,  2,  -6, 4,  2,  -1,  1,  3,   2,  1,  -3, 0,  0,  -2, 5,  3,  -3, 3,
    4,  2,  0,  0,  0,  2,  -2, 0,  1,  3,  2,  0,  0,  1,   2,  1,  -1, -1, 0,  -2, 2,  1,  1,  3,  0,  -4,  4,  4,   -2, 0,  4,  3,  4,  0,  -1, 2,  1,  3,
    1,  -3, -3, -2, 7,  -1, -2, -1, 0,  1,  2,  1,  3,  3,   3,  5,  0,  -1, 4,  1,  2,  7,  6,  -3, 3,  3,   2,  2,   -2, 0,  -2, -1, 1,  1,  0,  -1, 1,  -2,
    0,  1,  -1, -5, 4,  2,  4,  5,  2,  -1, -1, -3, 0,  3,   0,  -4, 5,  -1, -3, -3, 3,  0,  -2, 4,  6,  -1,  2,  2,   2,  0,  0,  -1, 1,  0,  0,  0,  6,  0,
    5,  2,  0,  1,  1,  2,  1,  2,  0,  2,  1,  -2, -1, 0,   2,  2,  2,  0,  -3, 3,  2,  2,  5,  2,  -1, 2,   0,  -2,  -1, 0,  -4, 0,  1,  3,  2,  0,  2,  7,
    2,  -3, 0,  -3, 4,  4,  -1, 1,  2,  4,  -1, 2,  3,  1,   1,  3,  2,  2,  0,  0,  0,  0,  -4, -4, -3, 5,   -1, 4,   -2, -3, 3,  1,  -3, 3,  -3, -2, 0,  1,
    2,  2,  -3, 3,  1,  1,  1,  -2, 0,  -3, 2,  1,  2,  1,   -3, -1, 0,  -2, -3, -3, -3, 1,  1,  2,  -1, -1,  0,  2,   0,  -3, 3,  0,  -2, -2, 3,  5,  0,  2,
    6,  1,  0,  1,  3,  -1, -3, -2, 0,  1,  0,  4,  0,  1,   2,  4,  1,  0,  -1, 0,  0,  1,  0,  2,  2,  -2,  0,  -2,  -2, 0,  -1, 0,  -3, 2,  -3, 2,  5,  1,
    2,  2,  0,  4,  -1, 1,  -1, 1,  1,  3,  2,  0,  1,  6,   1,  1,  4,  0,  3,  1,  -1, 0,  2,  4,  -2, -1,  3,  1,   1,  3,  4,  2,  0,  0,  1,  0,  -2, -3,
    -3, 4,  -1, 3,  0,  -3, 4,  0,  0,  1,  -3, 1,  1,  0,   1,  0,  0,  5,  3,  3,  1,  -3, -1, -1, 3,  1,   4,  -1,  -1, -1, -1, -1, -3, -2, -1, 3,  1,  2,
    3,  1,  1,  0,  -2, -3, 1,  -3, -2, 1,  6,  5,  -1, 3,   4,  0,  -1, 4,  5,  -2, -2, -3, 0,  3,  -5, -1,  3,  2,   0,  1,  -1, -2, 0,  3,  1,  2,  -2, 1,
    5,  -1, 1,  -2, -3, 0,  -1, -1, 0,  1,  -1, 3,  4,  3,   5,  1,  0,  3,  -3, 0,  0,  -1, 0,  1,  2,  3,   1,  4,   0,  1,  1,  -2, -1, 0,  -1, 0,  3,  1,
    -3, 0,  3,  5,  -1, 4,  3,  -2, 1,  -2, 2,  -2, -1, 0,   0,  2,  3,  2,  -3, -4, 1,  2,  4,  0,  -1, 1,   -2, 0,   1,  1,  1,  5,  3,  2,  1,  -1, 1,  0,
    5,  2,  5,  -2, -3, -3, -2, 2,  -2, -3, 0,  1,  2,  4,   -3, 1,  0,  -3, -2, -1, 0,  1,  0,  0,  5,  3,   0,  4,   3,  0,  -4, 2,  4,  -3, 0,  -1, 2,  0,
    -3, 0,  0,  -1, 2,  1,  -3, -3, -1, 1,  0,  0,  -1, 0,   0,  0,  0,  0,  -1, 0,  -1, -2, -2, -1, 0,  0,   4,  1,   2,  0,  -1, 1,  1,  0,  0,  -1, 0,  0,
    3,  3,  2,  2,  -1, -1, 1,  -1, 0,  -1, -1, 2,  2,  1,   1,  0,  5,  7,  0,  4,  5,  4,  1,  0,  2,  1,   -3, -3,  -3, 2,  -1, 4,  -1, -1, 2,  3,  3,  1,
    -1, 0,  -2, 2,  4,  2,  4,  3,  0,  3,  1,  -2, 1,  0,   4,  2,  6,  3,  -1, -4, -1, 1,  -3, 0,  -1, 4,   1,  4,   1,  -2, 1,  -3, -1, -4, -2, 2,  1,  1,
    3,  1,  -2, 0,  5,  0,  -1, 0,  2,  -1, 1,  -6, -2, -1,  -2, -1, -3, -2, -1, 1,  -3, 0,  2,  3,  2,  2,   1,  0,   3,  0,  0,  0,  0,  -1, -2, -1, -2, -4,
    0,  4,  3,  1,  3,  1,  0,  0,  1,  2,  1,  -1, 0,  1,   5,  -1, 1,  3,  0,  0,  0,  2,  3,  -1, 1,  2,   -1, 0,   0,  1,  5,  6,  2,  3,  3,  1,  -2, -2,
    -1, 0,  -2, 0,  -1, 2,  0,  5,  -1, 0,  1,  5,  4,  3,   -1, 0,  -2, 3,  3,  2,  2,  5,  2,  3,  1,  0,   4,  0,   4,  0,  7,  2,  -1, -2, -1, 1,  0,  0,
    1,  1,  1,  2,  0,  2,  0,  -5, 3,  1,  -3, -3, -5, 0,   2,  1,  -3, 4,  0,  1,  0,  -5, -1, 1,  -2, 0,   0,  -1,  -2, -1, 0,  -2, -3, -1, -2, 2,  -2, 2,
    2,  1,  3,  -3, -2, -3, -4, -2, -1, 0,  0,  1,  1,  3,   2,  1,  -1, -2, 1,  0,  -2, -1, -2, 0,  3,  0,   0,  0,   -1, 0,  -1, -2, -3, 1,  2,  0,  2,  -1,
    1,  0,  -2, 1,  -2, -2, 1,  5,  4,  0,  1,  -1, -1, 2,   0,  2,  -1, 2,  1,  4,  7,  0,  6,  -1, 2,  3,   3,  0,   3,  0,  2,  3,  0,  0,  1,  1,  1,  2,
    0,  4,  2,  2,  -1, 4,  1,  2,  1,  1,  1,  0,  1,  3,   6,  -1, 3,  1,  -3, 1,  -1, -5, 3,  0,  -3, -3,  -2, 2,   1,  0,  -3, 1,  -2, 4,  -1, -1, 1,  2,
    -2, -1, 0,  1,  0,  -3, -1, -3, -1, 1,  -2, 0,  -3, 2,   2,  1,  3,  -2, -1, -5, -5, -1, 0,  0,  0,  0,   1,  4,   2,  1,  -1, -2, -2, 0,  -1, 0,  -3, 1,
    3,  1,  1,  -3, -1, -1, 0,  -2, -1, 2,  3,  3,  0,  1,   3,  1,  -1, 2,  1,  -2, 2,  3,  2,  0,  -1, -3,  -1, 2,   -4, 2,  -3, 2,  2,  3,  6,  1,  4,  1,
    1,  3,  2,  0,  3,  -2, 0,  1,  -3, 0,  1,  1,  0,  2,   -1, 1,  3,  0,  1,  2,  3,  4,  0,  1,  0,  -1,  0,  2,   4,  0,  5,  4,  -3, -1, -1, -4, 2,  1,
    -2, -1, 0,  1,  1,  0,  -3, 2,  -1, 3,  -1, -4, -1, -1,  0,  -4, 2,  1,  -1, -5, -1, -2, 1,  -3, -3, -1,  -2, 5,   2,  -1, 1,  0,  -2, -4, -4, -1, 1,  3,
    0,  0,  6,  1,  -1, 1,  -1, -1, -1, 0,  -2, 2,  -2, 0,   4,  -1, 0,  -1, -1, -1, -2, -1, -1, 4,  2,  1,   -1, -1,  2,  2,  -1, -1, 2,  -1, -3, 4,  1,  0,
    -3, 1,  -5, 2,  -3, 1,  -2, 2,  2,  5,  4,  -1, 7,  1,   3,  1,  0,  1,  1,  -2, 0,  -1, 0,  1,  2,  3,   2,  2,   1,  0,  2,  -2, 2,  1,  3,  4,  1,  1,
    0,  -2, 1,  1,  4,  -1, 5,  4,  0,  0,  -2, -1, 0,  0,   0,  -1, -4, 1,  0,  2,  -3, 2,  3,  2,  0,  -3,  -1, -1,  -2, -2, 1,  -2, -1, -4, -2, -2, 1,  1,
    -3, -1, -2, 2,  3,  0,  0,  -1, -4, -2, -2, 0,  1,  2,   2,  2,  4,  3,  0,  1,  1,  -3, -3, -1, -4, -1,  0,  0,   1,  1,  0,  0,  -2, -1, 0,  -1, -2, 2,
    3,  4,  -1, 0,  2,  -1, 1,  -1, 1,  -1, 1,  3,  -2, -1,  -3, 1,  -4, 0,  1,  4,  -3, -3, 0,  3,  3,  0,   6,  3,   2,  -2, -1, 1,  0,  0,  0,  -1, -2, -3,
    3,  5,  -1, 1,  3,  0,  3,  -1, 1,  4,  2,  3,  1,  1,   0,  -2, 0,  2,  3,  -1, 3,  1,  0,  0,  -5, -1,  2,  2,   -2, -1, -2, 2,  1,  3,  0,  1,  1,  2,
    -1, 0,  -1, 0,  2,  -1, 2,  0,  -1, -5, 1,  -2, -1, 0,   0,  -1, -3, 3,  4,  -1, -2, -1, -3, -3, -2, 2,   2,  2,   3,  2,  4,  0,  1,  -1, 2,  -2, -3, -1,
    -1, 1,  0,  2,  2,  1,  1,  1,  -2, -1, 0,  0,  -3, 2,   2,  2,  -2, -1, 1,  0,  2,  -1, 2,  0,  4,  2,   -1, 1,   -3, 1,  -3, -1, 2,  4,  -4, -3, -2, 3,
    2,  0,  5,  2,  1,  -3, 0,  -3, 1,  0,  -1, -2, -4, -3,  5,  3,  -1, 2,  3,  2,  2,  0,  0,  1,  1,  6,   2,  1,   0,  -2, -1, -1, 4,  0,  0,  4,  0,  -2,
    0,  0,  2,  -3, -2, 1,  3,  2,  1,  -1, 7,  1,  0,  2,   0,  2,  5,  -3, -1, -2, 3,  4,  3,  1,  -2, 3,   2,  1,   -1, 0,  2,  4,  4,  1,  3,  4,  3,  2,
    0,  5,  0,  -1, 1,  -4, 0,  -3, -1, -3, 2,  1,  -3, -1,  2,  1,  1,  2,  1,  0,  1,  -1, 1,  1,  -1, 0,   3,  1,   -2, -1, 1,  0,  -2, -1, 0,  3,  1,  2,
    2,  1,  3,  4,  1,  2,  -1, 6,  1,  -1, 1,  -2, -4, -1,  -3, 2,  1,  3,  5,  -1, -4, -1, 6,  0,  -2, 1,   0,  -1,  -2, -2, 0,  -2, 0,  -4, -1, 1,  1,  1,
    -1, 1,  -2, 0,  -1, 3,  6,  3,  -2, -1, 0,  0,  0,  -2,  0,  0,  3,  -2, -5, 4,  -1, 0,  3,  1,  7,  -1,  1,  3,   1,  1,  1,  -1, 0,  -2, 2,  3,  0,  1,
    1,  1,  2,  2,  0,  -1, 2,  3,  1,  1,  0,  5,  1,  -2,  1,  7,  1,  0,  2,  -1, -2, -5, 2,  -1, -1, 0,   -1, -6,  -2, 0,  0,  0,  1,  1,  1,  -1, 1,  1,
    1,  4,  2,  2,  0,  -1, 0,  -2, -4, -4, 2,  2,  -1, 1,   4,  -1, -1, 2,  4,  1,  1,  5,  1,  -2, -1, -1,  1,  -1,  0,  0,  0,  0,  4,  -2, -3, -2, 2,  1,
    -1, 4,  0,  -2, -1, -1, 2,  0,  -2, -3, 1,  2,  0,  2,   1,  4,  1,  -1, -1, 2,  4,  3,  0,  -2, 0,  -1,  -1, -1,  0,  4,  2,  -2, -1, 4,  -3, 1,  3,  -3,
    5,  0,  -1, 2,  2,  2,  1,  -2, 3,  -2, 3,  2,  0,  -1,  0,  1,  3,  0,  -1, -1, -2, 2,  4,  2,  2,  7,   0,  -1,  1,  5,  3,  0,  0,  1,  -2, -5, -1, -2,
    2,  -2, 2,  -3, 0,  -1, 0,  0,  1,  0,  2,  2,  1,  0,   0,  2,  1,  1,  -1, -1, 0,  -2, -4, -3, 3,  3,   -1, 1,   4,  0,  4,  2,  5,  1,  0,  4,  1,  -4,
    -2, -1, 0,  -1, 0,  2,  2,  -1, 3,  2,  -3, -3, 2,  0,   1,  4,  1,  -3, 2,  0,  0,  -1, 0,  1,  2,  1,   0,  2,   3,  1,  0,  0,  -1, 2,  5,  4,  2,  0,
    0,  -3, 2,  2,  0,  -1, 3,  0,  -3, 3,  0,  0,  2,  -2,  7,  -1, -2, 1,  2,  5,  2,  -4, 4,  -4, 2,  5,   1,  -1,  -1, 0,  0,  0,  -2, -2, -1, 2,  3,  0,
    0,  7,  0,  0,  -1, 4,  3,  -1, 1,  0,  0,  -5, 1,  -1,  0,  0,  -1, -4, 2,  -2, 2,  -1, 2,  0,  2,  2,   1,  2,   -1, 2,  2,  -1, 1,  -1, -1, -1, -5, -4,
    2,  1,  -1, -2, 3,  -2, 4,  4,  7,  3,  0,  4,  2,  -4,  -1, -1, -2, 1,  -2, 1,  0,  -1, 3,  -1, -3, -2,  5,  3,   2,  1,  -1, -2, 0,  -1, 1,  0,  0,  0,
    0,  3,  2,  2,  2,  -2, -1, 2,  -3, -1, 3,  6,  -1, -1,  0,  -4, 0,  3,  1,  -1, -1, 0,  -4, 1,  3,  -2,  3,  -1,  7,  2,  -2, 2,  0,  1,  1,  -2, 3,  -1,
    1,  6,  1,  -2, 2,  0,  -2, 2,  -4, 0,  0,  0,  3,  1,   1,  6,  1,  4,  -1, 6,  2,  -2, 0,  -1, -2, -4,  -2, 0,   0,  0,  0,  -6, -2, 0,  3,  -2, 3,  0,
    1,  6,  2,  2,  2,  3,  2,  -1, -3, -1, -1, -4, -6, 0,   1,  1,  1,  -5, 3,  -1, 0,  4,  9,  5,  -1, 3,   1,  2,   -2, 0,  -1, -2, 1,  -1, -3, -3, 0,  -1,
    -5, -2, 2,  1,  2,  -1, -3, -2, 2,  1,  0,  1,  1,  0,   1,  2,  1,  3,  1,  -2, 0,  1,  -2, 0,  3,  5,   0,  2,   0,  0,  -1, 2,  6,  0,  3,  2,  -3, -2,
    3,  0,  -2, 4,  3,  0,  0,  3,  0,  -3, -2, 3,  1,  2,   -1, 3,  1,  0,  -1, 0,  1,  -2, -2, -2, 2,  3,   3,  -2,  -1, 0,  1,  1,  0,  -4, 0,  -3, 3,  -1,
    -2, -2, -1, -2, -1, 2,  -2, -3, -2, -3, 0,  0,  0,  -1,  1,  0,  -2, 1,  0,  -2, 0,  0,  0,  0,  -1, -2,  2,  4,   -1, -1, -4, -4, 1,  2,  4,  -1, -3, 0,
    1,  -2, -1, -4, 3,  0,  2,  2,  0,  2,  -4, 0,  4,  3,   -1, -1, 1,  2,  -1, 1,  2,  0,  4,  4,  0,  -3,  -1, 0,   -3, 4,  1,  0,  0,  0,  2,  -2, 2,  1,
    0,  1,  6,  6,  -2, 0,  0,  1,  5,  -2, 1,  2,  -4, 1,   1,  0,  1,  3,  3,  0,  -2, 1,  -1, -5, -1, 4,   2,  2,   1,  2,  6,  1,  -1, -4, 1,  -2, -1, -2,
    2,  3,  3,  2,  -1, -1, 0,  1,  0,  -4, 0,  -3, 1,  1,   0,  -3, -1, 0,  -1, -1, -4, -3, -2, -1, 1,  0,   -2, -1,  1,  0,  -4, -2, -1, 1,  0,  1,  0,  -1,
    -1, -1, 0,  0,  0,  2,  -2, -5, 0,  0,  2,  0,  -2, 0,   4,  -4, 1,  -2, 1,  2,  0,  2,  2,  2,  -1, 0,   1,  1,   -2, 1,  2,  -1, -1, 2,  0,  -1, 2,  5,
    2,  -3, 0,  1,  -1, 3,  0,  1,  1,  0,  0,  -1, 1,  0,   0,  2,  5,  1,  -2, 0,  1,  -1, 5,  1,  3,  1,   -4, 2,   1,  -1, 2,  3,  4,  1,  1,  0,  -3, -5,
    1,  4,  1,  2,  2,  1,  4,  2,  1,  -1, 0,  -1, 1,  0,   2,  3,  4,  2,  1,  1,  0,  -1, 0,  0,  -1, -2,  2,  1,   -2, -1, -2, 0,  -1, 1,  -2, -1, -3, 0,
    3,  2,  -2, 2,  2,  0,  -3, -3, 0,  -1, -1, 4,  -2, -3,  0,  0,  -1, 0,  -1, -1, -2, -3, 0,  1,  2,  0,   1,  0,   0,  0,  1,  -5, 1,  3,  2,  3,  3,  3,
    1,  -2, 3,  0,  1,  3,  2,  3,  1,  3,  0,  0,  3,  7,   2,  -1, 2,  1,  -4, 1,  -1, 3,  0,  1,  -1, 1,   -1, -2,  -1, -1, 7,  4,  -2, -1, 1,  -2, 5,  2,
    2,  -1, 0,  -1, -1, 0,  3,  1,  4,  0,  -2, 0,  2,  -2,  -2, 2,  0,  1,  3,  1,  3,  2,  -2, 0,  0,  -4,  0,  -3,  3,  1,  3,  1,  0,  -2, 0,  0,  -2, -3,
    1,  -1, 4,  2,  -5, -2, 0,  -1, -3, -1, -3, -2, 2,  1,   0,  2,  1,  1,  2,  0,  -3, -2, 1,  0,  -1, 4,   -3, 0,   -2, 2,  -1, -3, 0,  0,  -2, -3, 0,  -1,
    0,  -2, 1,  -2, 0,  -2, 0,  -2, 0,  3,  1,  4,  2,  2,   -2, -1, 2,  2,  1,  2,  2,  3,  0,  2,  0,  -1,  2,  3,   1,  -2, 1,  1,  -3, 1,  1,  0,  2,  -1,
    -2, 0,  0,  0,  0,  -2, 8,  2,  -3, -2, 0,  0,  3,  -1,  1,  -1, -1, -2, 0,  -1, 1,  3,  6,  0,  0,  1,   0,  -1,  -1, 1,  1,  0,  0,  1,  2,  1,  1,  0,
    2,  -3, 1,  -2, 2,  2,  3,  1,  1,  0,  -1, -2, 1,  -1,  1,  -1, 3,  2,  -1, -1, -1, 0,  -2, -1, -3, -2,  -1, -1,  0,  2,  2,  1,  1,  1,  -1, -3, 0,  -1,
    1,  2,  -4, 0,  0,  0,  -1, -2, -4, -1, -2, -2, 0,  -2,  -1, -2, 1,  0,  0,  -1, 1,  -4, 4,  3,  1,  4,   1,  2,   -2, -2, 1,  3,  1,  0,  -1, 3,  -1, 1,
    1,  0,  2,  4,  -1, -1, 0,  -1, -3, -4, 0,  -1, -2, -1,  -3, 1,  1,  0,  -2, -1, 7,  5,  -1, 1,  4,  -1,  -4, 1,   4,  -4, 5,  7,  -3, 3,  1,  -2, 2,  2,
    -1, 2,  -1, 0,  1,  3,  2,  5,  -1, -1, -3, -3, 0,  -2,  1,  1,  2,  2,  -2, 1,  1,  3,  3,  2,  2,  -1,  -3, 4,   -1, -3, 1,  6,  1,  -3, 1,  1,  -1, 3,
    1,  6,  3,  2,  3,  0,  1,  -1, 0,  -5, 2,  1,  -4, 0,   -2, 5,  2,  1,  0,  -1, 2,  0,  1,  4,  0,  0,   -2, 1,   1,  6,  3,  1,  6,  -2, -1, -1, 5,  0,
    7,  4,  1,  -7, -2, -6, -1, 4,  1,  -1, -2, -2, 4,  2,   0,  0,  1,  3,  -1, 4,  2,  0,  1,  -1, -2, -2,  -2, -1,  3,  4,  -1, 0,  0,  -3, -2, 0,  0,  0,
    2,  0,  -5, 2,  5,  -3, 4,  4,  0,  4,  -1, -2, 3,  3,   -1, 3,  1,  0,  2,  5,  0,  4,  1,  -1, -3, -3,  0,  -1,  1,  1,  0,  0,  -2, -1, 2,  3,  1,  -2,
    3,  0,  -1, 4,  0,  -4, 1,  4,  2,  -3, 0,  1,  -2, 3,   1,  4,  4,  1,  1,  -2, 1,  -1, -1, -3, 0,  2,   -4, 0,   -1, 3,  1,  5,  -1, 0,  2,  0,  0,  3,
    0,  -1, -2, 1,  2,  6,  4,  1,  2,  -3, 0,  -2, 3,  -2,  7,  3,  0,  -6, 0,  -4, 0,  2,  0,  -1, 0,  -1,  4,  3,   0,  0,  -2, 0,  0,  2,  1,  2,  2,  0,
    -2, -1, 0,  1,  0,  5,  2,  -2, -2, -2, 2,  1,  0,  1,   1,  0,  -3, 2,  5,  -4, 4,  6,  3,  3,  -2, -1,  4,  3,   -4, 1,  2,  -3, -2, 4,  1,  4,  1,  -3,
    -5, -2, 1,  0,  -1, 0,  -1, 0,  -3, 0,  -2, 2,  3,  1,   2,  -1, 1,  3,  3,  -1, 1,  5,  3,  -2, 0,  2,   -2, 0,   3,  2,  5,  -1, 0,  0,  -2, -1, -1, -3,
    1,  1,  -2, 0,  3,  2,  0,  5,  0,  0,  -1, 2,  1,  2,   -1, 1,  -2, 2,  1,  5,  5,  1,  2,  -2, -1, 0,   2,  -5,  5,  2,  1,  -4, 0,  -4, -3, 2,  2,  -2,
    -1, -4, 5,  4,  -1, 0,  -3, -2, 1,  0,  3,  1,  3,  1,   -4, -3, -2, 0,  1,  6,  0,  1,  3,  0,  -2, 0,   0,  0,   2,  0,  -3, 2,  3,  -2, 2,  2,  3,  4,
    -1, -5, 4,  3,  -3, 3,  0,  -2, 1,  4,  2,  2,  2,  -2,  -3, -4, 6,  0,  -2, 1,  0,  -3, -3, 0,  1,  -1,  4,  3,   0,  0,  2,  3,  2,  1,  0,  4,  2,  -6,
    2,  4,  -2, 1,  0,  3,  5,  2,  1,  0,  -1, -1, -2, -2,  1,  0,  -1, 0,  -1, 4,  -1, 4,  1,  -1, 1,  4,   0,  0,   -1, 1,  -1, 3,  2,  3,  4,  0,  3,  -1,
    1,  -3, 2,  -2, 4,  3,  1,  -4, 1,  -1, -2, 2,  2,  -1,  -1, -4, 3,  3,  -1, -3, -3, 2,  1,  -1, 2,  0,   1,  1,   -2, -2, -4, 0,  0,  7,  0,  3,  2,  0,
    -2, 2,  1,  -1, 0,  -1, -4, 1,  4,  -2, 4,  3,  3,  4,   0,  -2, 1,  -1, 0,  4,  -1, 0,  0,  5,  3,  4,   3,  0,   -1, -2, 4,  -1, -1, 2,  0,  -4, -2, 2,
    2,  0,  4,  1,  2,  0,  0,  4,  -2, -3, 1,  4,  2,  -3,  4,  2,  0,  0,  -4, 2,  4,  1,  1,  2,  -1, 0,   0,  -1,  -2, 0,  -1, 1,  1,  5,  0,  2,  4,  1,
    2,  1,  -1, -1, 1,  1,  -2, 2,  3,  3,  6,  1,  3,  -2,  3,  -2, 4,  -5, 4,  4,  1,  -2, 1,  -1, -1, 1,   2,  -2,  0,  -2, 4,  2,  -3, 0,  -1, 3,  0,  -1,
    4,  0,  1,  2,  -4, -4, -4, 0,  -2, 5,  1,  0,  2,  -3,  -4, 1,  3,  0,  -1, 1,  2,  -2, -1, 1,  -1, 0,   0,  -2,  -1, 2,  -1, -1, 1,  -2, 0,  2,  0,  -2,
    1,  4,  1,  0,  2,  2,  1,  -3, 5,  2,  -2, -2, 0,  1,   1,  2,  -1, -2, -2, 2,  1,  -2, 2,  -1, -2, -1,  1,  -4,  1,  -1, 1,  2,  1,  1,  1,  2,  2,  2,
    1,  1,  1,  -1, 1,  3,  0,  -1, 0,  1,  3,  -1, -1, -1,  4,  5,  -1, -2, 2,  -2, 0,  -4, -1, 2,  1,  1,   -2, -1,  2,  -1, -3, -3, 2,  -1, -2, -3, -1, -1,
    -2, 0,  -1, 0,  1,  -3, 0,  -4, 2,  0,  -5, -2, 3,  4,   3,  0,  0,  -2, 1,  1,  1,  0,  2,  7,  7,  -1,  -1, 0,   -3, 5,  5,  1,  -3, 2,  1,  -2, 1,  1,
    0,  -1, 2,  -1, 1,  3,  -1, 1,  -3, -1, -1, 1,  -2, 1,   0,  2,  -2, 1,  2,  2,  1,  -4, 1,  1,  0,  -2,  4,  3,   0,  2,  0,  -1, 0,  1,  1,  0,  1,  -1,
    -1, 1,  0,  0,  1,  2,  3,  1,  1,  4,  1,  0,  1,  2,   1,  1,  0,  0,  1,  3,  1,  -1, 0,  0,  -1, -2,  0,  1,   1,  4,  0,  0,  2,  -2, 2,  -2, 1,  3,
    4,  -2, -1, 0,  0,  1,  0,  -2, 5,  1,  0,  -4, 1,  0,   0,  1,  -1, 5,  -1, 0,  -2, -4, 3,  1,  -4, -2,  6,  2,   1,  -1, -1, -2, 1,  0,  0,  1,  1,  3,
    2,  1,  -2, -2, -1, 2,  6,  3,  -3, 2,  -1, -1, 2,  0,   -1, 2,  3,  0,  -3, 3,  -1, -2, -3, 0,  2,  3,   -2, 1,   2,  -1, -2, 2,  1,  5,  0,  -3, 2,  1,
    -1, 0,  1,  4,  1,  1,  1,  -1, -1, 0,  0,  3,  3,  2,   0,  1,  -2, 0,  2,  3,  2,  3,  -1, 2,  2,  2,   1,  1,   2,  1,  -1, -2, 2,  6,  4,  0,  2,  3,
    0,  -3, 0,  2,  -1, 5,  1,  1,  -2, -2, 2,  -2, 1,  4,   2,  0,  -1, 1,  3,  1,  1,  1,  4,  0,  1,  -4,  0,  4,   0,  -1, 0,  3,  2,  1,  -2, -4, 0,  0,
    -4, -1, 6,  4,  0,  1,  -1, -3, -1, -1, 3,  2,  3,  5,   0,  0,  -3, 0,  -3, 2,  3,  2,  -2, -1, 0,  2,   1,  -1,  0,  1,  3,  -3, 1,  1,  -2, 0,  1,  -1,
    3,  2,  2,  0,  -1, 3,  -1, 1,  2,  3,  1,  -2, 2,  -2,  -1, 1,  1,  4,  2,  5,  0,  -2, 2,  2,  3,  3,   1,  1,   -2, 0,  0,  1,  -1, 2,  3,  3,  -1, 1,
    -1, 5,  2,  1,  1,  1,  0,  0,  3,  4,  3,  2,  4,  0,   3,  -2, 1,  4,  0,  5,  2,  3,  0,  1,  0,  -3,  1,  3,   3,  1,  0,  2,  0,  2,  3,  1,  3,  0,
    -2, -1, 0,  3,  0,  0,  1,  1,  2,  1,  -2, -3, 4,  1,   -5, 1,  4,  4,  1,  1,  0,  0,  2,  1,  5,  1,   2,  4,   1,  1,  -2, 1,  0,  0,  3,  2,  -3, -1,
    -1, 0,  0,  0,  -1, -1, 2,  0,  -2, 1,  -3, -1, -2, -2,  3,  5,  -2, -2, -1, 2,  0,  2,  1,  2,  -3, -1,  2,  -1,  0,  1,  -1, 3,  1,  3,  -1, 0,  1,  2,
    2,  -2, 1,  -1, -4, 1,  -3, 1,  1,  -1, 1,  3,  2,  2,   0,  2,  0,  0,  1,  0,  -1, 2,  3,  3,  2,  0,   1,  -2,  -1, -2, -1, 1,  2,  2,  2,  -2, -3, 2,
    1,  -1, 1,  5,  1,  -1, -4, 1,  -1, 2,  1,  -1, 3,  -1,  -3, -2, 2,  3,  1,  0,  0,  1,  3,  2,  -3, -6,  1,  0,   -2, 0,  3,  3,  1,  -2, -2, 0,  1,  -1,
    2,  -5, 0,  1,  1,  -1, -1, 2,  -3, 4,  3,  -2, 2,  -1,  0,  1,  -3, 3,  -1, -2, 1,  0,  0,  0,  -1, -5,  5,  2,   0,  0,  -1, 1,  -1, -1, 2,  -2, 5,  -1,
    1,  -2, -1, -3, -3, -2, -1, 1,  -1, 1,  -1, -5, 3,  1,   0,  -1, 4,  0,  0,  -1, 1,  1,  0,  1,  3,  1,   -2, 1,   1,  4,  1,  3,  2,  2,  -1, -4, 1,  0,
    1,  2,  -1, 5,  3,  2,  4,  0,  4,  -4, -5, 0,  1,  1,   4,  5,  -1, 7,  2,  0,  2,  3,  2,  6,  5,  -4,  -9, 1,   5,  3,  1,  -2, 5,  3,  5,  0,  -3, -4,
    -5, -1, 2,  3,  -2, 1,  2,  3,  4,  1,  -2, 3,  -2, -6,  2,  -1, -1, -3, -2, 0,  -4, -2, -1, -4, 2,  -1,  2,  2,   -1, 2,  -2, 0,  0,  -3, 2,  -4, 0,  1,
    -2, -2, 3,  1,  0,  -2, 0,  3,  -2, -2, 2,  0,  4,  1,   3,  0,  -1, -3, -6, 0,  -1, 2,  0,  -1, -2, -2,  2,  1,   -1, 0,  2,  0,  1,  0,  -3, 1,  0,  2,
    1,  1,  -1, -1, -1, 2,  0,  3,  2,  2,  -3, -3, 2,  0,   1,  1,  0,  4,  3,  5,  2,  -1, 2,  -1, -3, 1,   0,  0,   3,  3,  -4, 3,  1,  4,  1,  3,  3,  4,
    5,  -3, -7, 1,  4,  1,  -1, 1,  3,  2,  3,  0,  -1, -2,  -5, 2,  2,  7,  -2, 2,  5,  1,  2,  2,  0,  1,   -1, -4,  4,  1,  -1, -1, -2, 0,  -2, -3, -2, -2,
    2,  1,  1,  1,  1,  0,  -1, 0,  0,  -4, 0,  -2, 0,  2,   -2, -1, 4,  4,  -3, 0,  1,  1,  -2, 0,  3,  4,   4,  0,   3,  1,  -2, -1, -5, 0,  1,  0,  0,  1,
    -2, -2, 2,  -1, -2, 0,  1,  0,  0,  -1, 1,  -1, 2,  2,   4,  2,  0,  1,  1,  3,  0,  4,  -1, 0,  -2, -3,  3,  2,   0,  1,  0,  4,  3,  2,  -1, -2, 2,  -3,
    -3, 2,  2,  -1, 2,  1,  -4, 4,  2,  4,  1,  4,  3,  6,   5,  -5, -5, 2,  5,  1,  1,  0,  1,  2,  0,  0,   -1, -2,  -4, 3,  1,  3,  -4, 5,  4,  -1, 1,  3,
    2,  2,  2,  -4, 2,  1,  1,  -1, -2, 2,  0,  -2, -1, -2,  0,  -1, 2,  2,  -2, 2,  0,  0,  0,  -1, 0,  -2,  1,  5,   -4, -3, 4,  4,  -4, -1, -1, 0,  -1, -1,
    3,  3,  6,  -1, 1,  0,  -3, -2, -2, 1,  0,  1,  -3, 0,   -1, -1, 0,  -2, 2,  4,  0,  1,  1,  0,  -1, 1,   2,  1,   2,  4,  -2, 1,  1,  3,  -3, 2,  1,  -1,
    -2, -2, 3,  1,  -1, 0,  4,  4,  1,  1,  4,  -3, 0,  -1,  0,  2,  3,  -2, 4,  2,  2,  5,  0,  1,  3,  3,   2,  5,   3,  -2, -2, 2,  3,  1,  1,  -1, -1, 3,
    3,  4,  -2, -2, -2, 2,  0,  4,  -4, 3,  3,  -2, -2, 4,   3,  0,  2,  -2, 1,  0,  1,  0,  -2, 1,  -2, -3,  -3, 0,   -1, 0,  0,  2,  -2, 1,  0,  2,  0,  0,
    2,  -3, 3,  4,  1,  -3, 1,  2,  -3, -2, -1, -1, -1, -1,  4,  3,  5,  2,  -3, 0,  0,  1,  -3, 1,  3,  0,   -1, 0,   0,  -2, -2, 0,  1,  1,  1,  3,  0,  -1,
    -2, 3,  1,  1,  3,  1,  0,  1,  0,  -1, 1,  2,  0,  1,   0,  0,  1,  2,  1,  3,  5,  4,  4,  2,  4,  -1,  -1, 0,   2,  3,  6,  1,  4,  3,  0,  3,  4,  1,
    1,  2,  5,  5,  6,  -2, -5, 1,  3,  2,  2,  -4, -2, 3,   3,  3,  1,  -3, -5, 5,  2,  2,  0,  2,  0,  1,   1,  2,   2,  3,  2,  -3, 2,  2,  1,  3,  0,  2,
    -4, -3, -5, 2,  -1, 2,  1,  -1, -2, 0,  -5, -2, 1,  1,   0,  1,  2,  3,  -4, -2, -2, -1, -2, 3,  4,  4,   1,  3,   1,  1,  1,  -2, -1, 2,  1,  0,  -2, 0,
    0,  0,  2,  1,  0,  -4, 5,  1,  -1, -2, 0,  0,  1,  0,   1,  2,  2,  2,  2,  0,  3,  6,  0,  -3, 0,  2,   1,  -1,  -1, 1,  0,  3,  0,  1,  3,  -2, 0,  2,
    3,  2,  -2, -1, 3,  2,  2,  1,  0,  -1, 1,  1,  6,  2,   0,  2,  3,  4,  4,  1,  -5, 0,  -3, -1, 2,  -1,  1,  -3,  0,  -1, 0,  0,  -2, -1, 0,  2,  1,  2,
    -1, 1,  3,  1,  -1, 1,  3,  1,  -2, 4,  3,  0,  4,  1,   3,  2,  -3, -3, -1, 5,  2,  0,  -4, 1,  -4, -1,  3,  0,   -1, 1,  1,  3,  -3, -2, 1,  -1, -3, 2,
    5,  4,  0,  1,  1,  -1, 0,  -3, -2, 1,  2,  2,  0,  -1,  0,  2,  1,  2,  0,  -1, 1,  -1, -2, -1, 0,  -1,  0,  1,   3,  2,  -1, -2, 2,  -1, 3,  4,  -2, -3,
    -1, 1,  1,  0,  1,  0,  0,  1,  0,  0,  1,  -2, -2, -1,  1,  3,  -3, 1,  2,  0,  -1, 1,  1,  -1, 1,  3,   5,  1,   -1, 5,  2,  1,  0,  -2, -4, 0,  -2, 2,
    2,  -3, -2, -2, -1, -1, -3, 1,  -3, -1, 0,  2,  2,  4,   0,  3,  1,  -1, -2, 4,  2,  0,  -1, 2,  2,  -2,  4,  3,   3,  0,  -6, -2, -2, 3,  2,  0,  -1, 1,
    -5, -2, 2,  -1, -3, 2,  1,  0,  -5, -2, 2,  -2, -1, 0,   5,  4,  -3, -2, 0,  1,  0,  -1, -3, 3,  3,  2,   -1, -1,  1,  2,  2,  -1, 3,  -2, 2,  -2, 0,  1,
    0,  1,  0,  -1, 3,  2,  -1, 0,  -1, -1, 1,  4,  0,  -4,  -1, 2,  1,  -1, -1, 2,  1,  2,  0,  0,  1,  -1,  -1, -1,  -1, 1,  0,  0,  0,  0,  -1, 2,  3,  0,
    0,  4,  6,  2,  1,  3,  3,  0,  1,  -1, -5, 0,  -3, 2,   1,  -3, -1, 0,  1,  -1, -4, -1, -4, -2, 0,  3,   4,  5,   0,  2,  2,  0,  1,  3,  1,  0,  -2, 2,
    4,  -3, 4,  0,  1,  0,  -8, -1, -1, 5,  4,  -3, 1,  3,   -4, -3, 3,  -2, -1, 0,  3,  -2, -5, 1,  3,  -3,  0,  2,   3,  4,  -2, 0,  1,  1,  -1, -2, 1,  0,
    1,  0,  1,  1,  3,  2,  0,  0,  1,  0,  3,  -1, 3,  0,   1,  2,  1,  0,  2,  2,  0,  1,  -1, -1, 4,  2,   -3, -5,  -4, 1,  0,  -1, 0,  3,  2,  1,  0,  -1,
    3,  -1, 0,  0,  -1, 2,  -1, 1,  1,  1,  0,  3,  4,  0,   1,  5,  3,  0,  1,  4,  2,  1,  0,  -3, -5, 0,   -1, 3,   2,  -3, -1, 1,  1,  -1, -4, -2, -2, 2,
    0,  4,  2,  3,  0,  0,  1,  1,  1,  3,  2,  1,  -2, 3,   3,  -1, 2,  0,  -1, -1, -5, -1, -2, 2,  0,  0,   -2, 1,   -6, -2, 0,  -2, -2, 4,  -2, -3, -5, 2,
    2,  -2, -2, 3,  2,  4,  0,  2,  1,  0,  -1, 3,  0,  4,   4,  1,  1,  3,  2,  3,  1,  1,  0,  3,  4,  0,   1,  4,   2,  4,  3,  0,  3,  2,  0,  0,  0,  3,
    0,  4,  -1, -4, 0,  0,  1,  0,  2,  3,  0,  3,  0,  0,   2,  -1, 2,  -1, -1, 3,  0,  0,  -2, 2,  1,  3,   3,  0,   3,  5,  4,  1,  1,  4,  3,  0,  0,  -3,
    -7, -3, 0,  0,  0,  0,  0,  -3, 0,  -2, -2, -2, -2, 0,   -4, 1,  1,  4,  0,  1,  1,  2,  0,  3,  2,  0,   -3, 0,   3,  0,  0,  0,  -1, 1,  -5, -1, 0,  1,
    0,  2,  5,  3,  1,  -2, 3,  1,  1,  2,  4,  3,  -2, 2,   0,  -3, -2, 2,  3,  -3, 0,  -4, 4,  -3, -2, -1,  0,  1,   0,  3,  2,  3,  4,  1,  -2, 2,  -1, -3,
    1,  2,  -2, 0,  6,  -1, 0,  -3, -1, 1,  2,  -2, 0,  4,   2,  1,  0,  -3, 4,  2,  -3, 0,  1,  -4, 3,  1,   -2, -1,  -3, 2,  6,  1,  2,  1,  1,  4,  -3, -2,
    -2, 2,  3,  -1, -1, -1, 1,  0,  -1, 1,  -5, 2,  2,  0,   -3, 2,  0,  4,  1,  1,  5,  -2, -1, 2,  -1, 0,   0,  -5,  2,  4,  1,  3,  4,  5,  2,  -1, 3,  -3,
    -2, 0,  -1, -1, 5,  4,  0,  2,  3,  0,  0,  7,  0,  2,   0,  0,  1,  3,  -1, -2, 5,  2,  1,  -1, 2,  0,   -2, 1,   -2, 0,  0,  -2, 2,  -2, 1,  -2, 4,  -2,
    -1, 1,  -1, 1,  -1, 4,  2,  2,  4,  3,  -3, 3,  -3, -2,  1,  5,  -1, 1,  7,  0,  0,  -1, -2, 2,  4,  -2,  2,  1,   1,  -1, 2,  -5, 1,  3,  -2, -1, 0,  -3,
    2,  3,  -2, 1,  1,  3,  4,  -1, 2,  3,  4,  7,  0,  -4,  -1, 2,  4,  -1, 1,  -2, 0,  -1, 1,  -1, -4, 3,   3,  -1,  -1, -1, -2, 3,  2,  1,  2,  1,  -1, 2,
    -4, -2, 1,  -7, 0,  7,  2,  1,  3,  5,  1,  1,  -2, -3,  -2, -1, -1, -2, 1,  1,  2,  2,  4,  4,  4,  6,   0,  -3,  -1, 0,  5,  2,  -1, -1, 4,  1,  1,  0,
    0,  -1, 2,  0,  -1, 0,  -1, -1, 2,  -2, 1,  -1, 1,  0,   -1, 1,  -3, 2,  1,  4,  3,  2,  2,  0,  -3, 2,   0,  -2,  2,  3,  2,  2,  10, -1, -1, -1, 1,  2,
    4,  -1, 1,  2,  3,  -3, 1,  -4, 2,  -2, -1, -2, 0,  -3,  4,  1,  -2, 0,  1,  1,  0,  1,  -1, 0,  3,  5,   3,  -3,  -1, 3,  4,  0,  -2, 0,  -1, 0,  0,  -2,
    -3, -1, 3,  1,  -1, 2,  2,  1,  3,  1,  -2, 0,  0,  -2,  -3, 1,  2,  -4, 2,  6,  0,  2,  3,  3,  1,  4,   0,  -3,  1,  0,  3,  -2, 1,  1,  2,  -1, 3,  3,
    0,  8,  -1, -3, 0,  1,  4,  3,  3,  0,  4,  3,  0,  3,   1,  -1, 1,  0,  -2, -1, 0,  1,  3,  -1, -2, 0,   1,  0,   0,  1,  -2, 2,  0,  2,  3,  5,  1,  2,
    0,  0,  0,  -2, -1, 4,  1,  4,  7,  -2, -1, -1, 1,  2,   1,  -1, 1,  1,  1,  -1, -1, -1, 1,  4,  -2, 0,   -1, -1,  3,  -1, -3, 2,  1,  1,  1,  3,  -1, 2,
    0,  4,  3,  -3, -1, 0,  3,  1,  -1, -3, -2, 0,  -1, -6,  -2, 1,  4,  0,  3,  1,  1,  1,  -1, -2, 2,  2,   2,  0,   -4, -1, -1, -2, 0,  5,  -3, 3,  3,  4,
    1,  1,  3,  -4, 0,  3,  2,  -3, -1, 0,  2,  -3, 1,  3,   -3, 8,  -2, -2, 0,  0,  4,  5,  4,  1,  2,  -1,  -3, 1,   2,  -1, 0,  -2, -2, -2, -4, -1, 4,  -1,
    -4, -2, 1,  -1, 1,  2,  -2, 4,  1,  0,  5,  5,  0,  0,   -1, 0,  -2, 1,  1,  4,  1,  3,  5,  0,  2,  1,   2,  3,   4,  1,  1,  3,  1,  4,  4,  -4, 3,  2,
    -3, -1, -1, 0,  2,  0,  -3, 2,  0,  3,  1,  3,  1,  2,   3,  5,  2,  -3, -1, 4,  6,  3,  -1, -1, -6, 1,   -3, -5,  -1, 1,  4,  -4, 2,  0,  2,  3,  2,  -2,
    3,  1,  3,  4,  -3, -1, 3,  0,  2,  4,  -1, 2,  3,  4,   2,  3,  5,  -1, 4,  3,  4,  1,  1,  -1, 4,  -3,  2,  4,   1,  5,  3,  2,  2,  6,  -1, 4,  -1, -4,
    1,  3,  0,  4,  1,  -1, 6,  -2, 1,  0,  0,  -5, -4, 4,   -2, 2,  3,  0,  3,  0,  0,  -2, 1,  2,  0,  0,   1,  -4,  0,  -2, 2,  1,  4,  6,  3,  1,  1,  4,
    1,  1,  2,  -1, -1, 0,  0,  -1, 0,  4,  0,  2,  0,  1,   1,  -3, 0,  0,  2,  0,  1,  1,  1,  2,  -3, 3,   0,  -2,  -5, 0,  1,  1,  0,  0,  2,  2,  2,  3,
    1,  1,  5,  2,  3,  1,  -3, 0,  1,  0,  -1, -3, 0,  0,   3,  0,  -2, -5, -3, -1, 3,  0,  -4, 3,  0,  0,   0,  4,   3,  1,  4,  1,  1,  0,  -1, 1,  -1, 4,
    0,  0,  1,  4,  1,  -2, 2,  4,  0,  5,  0,  3,  -3, -3,  3,  6,  3,  0,  1,  2,  6,  1,  1,  1,  1,  -3,  -4, 4,   -1, 2,  2,  1,  2,  4,  -2, -5, 2,  4,
    0,  0,  -2, -4, 0,  -2, 1,  0,  5,  4,  3,  0,  -1, 5,   0,  -1, 1,  0,  1,  2,  2,  0,  -3, 4,  -4, 3,   0,  0,   1,  -4, 0,  0,  3,  1,  1,  1,  4,  1,
    -3, 2,  -1, -2, -4, 3,  0,  -1, -1, 1,  1,  -2, 4,  1,   -1, 0,  4,  0,  2,  1,  -3, -2, 2,  2,  1,  -3,  -3, -1,  1,  -1, -2, -1, -2, 0,  2,  1,  -3, 2,
    0,  1,  1,  2,  2,  1,  4,  1,  1,  1,  0,  2,  -1, 2,   1,  0,  2,  4,  -1, -2, 1,  6,  1,  7,  -1, 5,   -2, -3,  3,  3,  2,  2,  1,  4,  5,  0,  2,  1,
    2,  -4, -4, 4,  0,  -1, 1,  2,  2,  1,  0,  -3, 2,  3,   -2, -1, -2, 1,  0,  -2, 4,  0,  5,  5,  2,  1,   -2, 5,   1,  0,  -3, -1, 0,  1,  -1, 0,  -1, 2,
    -1, 1,  1,  1,  0,  -2, 0,  1,  2,  -2, 1,  0,  4,  2,   -1, 3,  0,  -2, -2, 1,  1,  -1, -2, 1,  2,  2,   0,  1,   0,  1,  4,  1,  2,  1,  -3, -4, 0,  0,
    0,  -2, -2, 1,  2,  -3, 1,  1,  1,  0,  4,  2,  0,  3,   0,  1,  4,  3,  1,  1,  5,  1,  1,  1,  0,  1,   -2, 3,   3,  -1, 2,  4,  0,  -1, 1,  6,  1,  5,
    3,  4,  -1, -1, 5,  3,  2,  3,  -1, 4,  1,  1,  2,  4,   0,  -4, -4, 2,  -2, 0,  2,  2,  2,  -1, -3, -2,  4,  1,   -3, 0,  -2, 0,  0,  0,  3,  1,  4,  4,
    2,  0,  -1, 5,  2,  3,  -4, -1, 0,  0,  0,  -1, -2, 2,   -3, 0,  1,  0,  0,  -2, 0,  1,  2,  -2, 1,  1,   1,  2,   -1, 0,  -3, -1, 0,  1,  -1, 0,  0,  0,
    1,  0,  2,  2,  -1, 1,  4,  0,  1,  1,  -4, -2, 1,  0,   0,  0,  -1, -3, 4,  -2, -1, 1,  -3, 1,  4,  -2,  -2, 4,   1,  2,  1,  3,  2,  1,  3,  -1, 3,  1,
    1,  -1, -4, 1,  1,  -2, 4,  3,  2,  -1, 0,  6,  1,  5,   0,  2,  1,  0,  1,  4,  2,  2,  0,  2,  2,  2,   2,  5,   0,  -3, -2, 2,  -1, -1, 3,  3,  2,  -2,
    -1, -1, 3,  2,  -5, 0,  -2, 2,  0,  -1, 3,  1,  2,  3,   0,  0,  0,  4,  2,  1,  1,  1,  0,  2,  1,  0,   -1, 0,   -3, 0,  2,  1,  1,  -2, 0,  2,  2,  0,
    3,  0,  3,  2,  0,  2,  -4, -1, 0,  -4, 1,  -1, -1, 1,   1,  1,  0,  3,  -2, 2,  3,  -3, 2,  2,  -4, -3,  3,  -1,  1,  1,  -1, -3, 3,  -1, 0,  3,  -2, 1,
    3,  0,  -2, 3,  0,  2,  1,  3,  1,  1,  4,  -1, 1,  1,   1,  0,  0,  1,  1,  1,  1,  1,  3,  2,  1,  1,   3,  4,   3,  -4, 1,  2,  -2, 0,  3,  1,  -2, 2,
    -2, 5,  -1, -4, 5,  2,  0,  2,  -5, 2,  4,  -1, 0,  2,   0,  -1, 2,  -1, 0,  2,  4,  1,  -2, -2, 4,  2,   -1, 0,   1,  0,  -2, 0,  0,  1,  1,  -1, -1, -1,
    0,  1,  -1, 3,  4,  1,  -1, -1, -1, 1,  -1, -1, 0,  0,   2,  1,  -3, -1, 5,  -1, 2,  0,  -1, 2,  2,  0,   3,  2,   -2, -2, -1, -1, -4, -6, -1, 4,  3,  -1,
    3,  -1, 0,  -3, 0,  -1, 0,  6,  5,  0,  -2, 3,  -1, 7,   9,  11, 2,  0,  -1, 1,  -3, 2,  -1, 2,  0,  2,   1,  2,   2,  0,  -1, 2,  6,  -2, 0,  -2, 2,  -4,
    -1, 0,  2,  2,  3,  -1, 2,  2,  -1, -3, 1,  -2, 3,  0,   -3, 4,  -2, -2, 2,  4,  4,  -1, -1, -1, 2,  -1,  -2, -1,  0,  0,  1,  0,  1,  1,  0,  1,  -2, -3,
    3,  0,  0,  -2, 0,  2,  -6, -2, 1,  1,  0,  -2, -2, -1,  -1, 3,  -3, 4,  5,  2,  0,  -2, -1, 2,  0,  -2,  -3, -1,  3,  1,  -1, 1,  2,  1,  1,  2,  3,  1,
    1,  0,  1,  1,  -2, -4, -2, 0,  0,  -3, -3, 1,  3,  1,   2,  -1, 2,  -1, -1, 0,  0,  5,  5,  0,  -4, 4,   -1, 2,   9,  11, 3,  -2, 1,  2,  1,  4,  1,  2,
    -2, 1,  -3, 1,  3,  -1, -1, 2,  3,  -3, 0,  0,  0,  -4,  -3, -2, 5,  4,  3,  -2, 1,  0,  -4, -2, 1,  0,   4,  -1,  1,  1,  -3, 0,  5,  0,  3,  -2, 2,  1,
    2,  -3, -5, 0,  3,  1,  -2, 2,  -1, 1,  1,  2,  -1, -1,  2,  1,  -1, 1,  4,  3,  -4, -4, -3, -1, 0,  0,   -2, -3,  -2, 3,  0,  2,  8,  5,  1,  0,  1,  3,
    1,  0,  -4, -4, 3,  2,  -3, -1, 7,  0,  2,  2,  1,  0,   -1, 1,  1,  2,  -6, -4, 0,  -2, 2,  -6, -3, 2,   1,  2,   2,  -2, 1,  1,  0,  -1, -2, 3,  4,  -1,
    -3, 3,  2,  2,  8,  8,  0,  -7, 1,  6,  1,  1,  2,  4,   0,  4,  -3, 0,  6,  -1, -1, 1,  4,  -1, -2, -1,  3,  -3,  -1, 2,  5,  2,  2,  -1, 1,  2,  -1, -1,
    -3, 0,  0,  1,  -1, 1,  -3, -2, 4,  3,  2,  0,  -2, 1,   1,  2,  2,  2,  0,  2,  -1, -1, 2,  4,  0,  1,   0,  0,   3,  -4, -1, 0,  2,  1,  -3, -3, 1,  2,
    0,  1,  -3, 0,  0,  2,  1,  1,  6,  4,  1,  0,  2,  3,   1,  -1, 0,  -4, 3,  3,  1,  -4, 8,  -1, 3,  3,   1,  3,   2,  1,  1,  3,  -7, -4, -2, -3, -1, -6,
    -1, -1, 2,  0,  -2, -1, 1,  -1, 0,  -2, 0,  2,  4,  0,   1,  5,  -1, 3,  8,  10, 1,  -2, 4,  5,  2,  3,   5,  2,   -1, 2,  -1, 2,  7,  -2, -1, 2,  4,  -1,
    -1, 1,  5,  -4, -3, 3,  3,  3,  1,  1,  0,  2,  -3, -3,  -3, 0,  1,  1,  -4, 1,  -2, 0,  1,  1,  0,  0,   2,  -2,  0,  0,  5,  2,  -1, 2,  1,  0,  1,  3,
    1,  -1, 0,  1,  2,  -2, 1,  2,  2,  0,  -2, -3, 0,  0,   -3, 2,  -1, 0,  -1, 1,  -1, 1,  5,  7,  -1, 1,   2,  2,   1,  -1, -1, -1, 1,  4,  0,  -2, 7,  0,
    0,  6,  2,  1,  1,  2,  2,  1,  -6, -6, -2, -3, -2, -6,  -2, 2,  5,  0,  -1, -1, 0,  1,  0,  0,  1,  2,   2,  1,   0,  4,  -1, 1,  8,  11, 1,  -6, 3,  6,
    0,  3,  7,  3,  -2, 3,  -1, 0,  7,  1,  2,  2,  8,  -3,  -1, 1,  3,  -3, 2,  3,  2,  -3, -1, 1,  2,  2,   4,  -3,  -5, 1,  -1, -1, -1, 5,  -1, 0,  -2, 1,
    0,  -1, 3,  1,  -1, -2, 0,  0,  0,  -3, 0,  -1, -1, 2,   0,  2,  0,  1,  0,  2,  2,  0,  -3, -1, 2,  -1,  -2, 0,   4,  -2, 0,  -2, -1, -2, 1,  0,  0,  1,
    3,  1,  3,  0,  -1, 0,  0,  0,  0,  1,  2,  4,  1,  -1,  2,  1,  1,  2,  -3, 5,  3,  2,  1,  1,  3,  2,   3,  2,   1,  2,  2,  -1, 0,  0,  3,  0,  5,  3,
    -1, 4,  1,  1,  4,  -3, -4, -2, 2,  -1, 0,  0,  1,  2,   0,  3,  2,  -3, 0,  -1, -2, -1, -3, -2, -1, -6,  -4, 0,   -1, 2,  -1, -3, 2,  2,  3,  -3, 0,  0,
    2,  2,  4,  -2, -2, 1,  -2, 1,  -1, 1,  1,  -1, -4, 4,   0,  0,  2,  0,  0,  0,  1,  3,  -1, -1, -2, 0,   -1, 2,   0,  2,  2,  0,  -2, 3,  1,  2,  -3, 1,
    1,  -2, -3, 1,  0,  -1, -2, -1, -1, -1, 0,  0,  2,  -1,  1,  0,  2,  1,  1,  -1, 2,  0,  1,  1,  1,  3,   2,  0,   2,  1,  4,  3,  -2, 4,  2,  0,  3,  0,
    2,  2,  1,  3,  3,  1,  0,  -2, -1, -1, 2,  0,  0,  1,   0,  3,  0,  -3, 3,  -2, -3, -3, 0,  -1, 1,  -1,  1,  4,   1,  1,  1,  -1, -1, -2, -2, -3, -3, 2,
    2,  -1, -4, 1,  -2, 3,  0,  -3, 1,  4,  0,  -3, 1,  -1,  2,  2,  3,  -4, -3, 4,  -1, -1, 1,  5,  0,  -3,  -2, 1,   4,  0,  1,  0,  1,  0,  0,  0,  0,  0,
    -2, -1, 0,  0,  0,  2,  3,  0,  -2, 3,  0,  1,  -2, -1,  2,  -1, -4, -1, -1, -2, 2,  -3, -1, 0,  -1, 0,   1,  1,   0,  3,  1,  0,  -1, -1, 2,  -1, 2,  2,
    2,  3,  1,  0,  -1, 2,  3,  1,  -2, 5,  2,  2,  2,  0,   1,  1,  0,  3,  2,  1,  4,  -1, 2,  1,  2,  0,   1,  1,   -1, 3,  1,  -1, 3,  0,  -2, -2, 0,  -3,
    3,  3,  6,  2,  1,  3,  2,  0,  0,  -2, -3, 0,  -1, 3,   1,  -2, -3, 1,  1,  1,  -2, 0,  2,  3,  1,  -1,  0,  0,   7,  0,  1,  -2, -2, 1,  0,  -1, 0,  0,
    2,  -1, -2, -2, 2,  2,  5,  -2, 0,  0,  -2, 0,  -2, -2,  1,  -1, 1,  2,  0,  1,  2,  -1, -1, 1,  1,  2,   -1, 1,   2,  -4, -5, 1,  0,  0,  0,  -2, -1, 2,
    -1, 0,  -1, 0,  1,  1,  -1, -2, -1, -1, 1,  0,  1,  2,   4,  6,  0,  0,  -1, -1, 4,  3,  0,  4,  3,  2,   2,  1,   4,  -1, -1, -1, 2,  0,  3,  -1, -1, 1,
    1,  -1, 2,  1,  0,  4,  1,  1,  0,  -2, -2, 1,  -1, 1,   4,  3,  5,  2,  0,  0,  2,  0,  0,  -1, -2, 0,   -2, 0,   1,  -1, -3, 2,  -1, 2,  -3, -1, 1,  2,
    0,  -2, 1,  3,  4,  2,  2,  -1, 0,  1,  2,  -3, -2, 0,   2,  1,  -1, 2,  -1, 0,  6,  1,  -1, -2, 0,  2,   -2, 0,   1,  1,  1,  1,  3,  0,  3,  -1, 0,  -1,
    2,  2,  -1, -1, 3,  -2, -6, 0,  3,  1,  0,  0,  0,  2,   -1, 1,  0,  2,  2,  1,  1,  0,  -1, -1, 0,  3,   0,  2,   4,  0,  0,  1,  -1, -1, 3,  4,  0,  6,
    2,  1,  5,  1,  0,  2,  1,  0,  2,  0,  5,  -3, -1, 1,   2,  -1, 0,  1,  0,  4,  0,  2,  3,  0,  -5, 1,   -2, 0,   4,  2,  4,  4,  2,  2,  1,  0,  -1, 1,
    -2, 1,  -1, 0,  1,  -1, -2, 4,  0,  3,  -1, -1, -5, -5,  -4, 0,  -1, -2, 1,  1,  -4, -1, -4, -1, -1, 4,   0,  1,   3,  1,  0,  -2, 1,  -1, -2, -2, 0,  3,
    1,  1,  3,  4,  2,  6,  0,  3,  1,  2,  -1, -1, 2,  0,   -3, -3, -1, -3, 2,  5,  2,  5,  3,  1,  7,  1,   -2, -1,  0,  -4, 0,  2,  2,  -1, 1,  0,  0,  2,
    -4, -1, -1, 2,  1,  2,  1,  4,  -1, -2, 0,  -1, -4, -6,  -3, 2,  4,  3,  2,  -1, 0,  -2, -2, -2, -2, 2,   4,  6,   0,  2,  0,  -7, 1,  1,  -3, 1,  2,  0,
    -2, 0,  -1, 3,  -3, 1,  1,  5,  3,  3,  1,  4,  3,  -5,  -4, -2, 0,  -2, 1,  1,  2,  6,  0,  1,  -1, 1,   -4, -2,  -3, 0,  1,  -2, 0,  2,  -2, -1, -4, 0,
    -1, 4,  -1, -1, 2,  0,  0,  -3, 0,  0,  -1, -2, 2,  3,   2,  0,  3,  7,  3,  3,  2,  3,  2,  0,  1,  -1,  -1, -1,  1,  -3, 0,  -5, 3,  4,  5,  2,  4,  3,
    2,  0,  -3, 0,  -1, -3, 1,  -1, 0,  1,  0,  1,  -1, 1,   -3, 3,  3,  4,  1,  3,  1,  2,  -1, 1,  1,  -2,  -3, -6,  -7, 1,  5,  0,  1,  -1, 0,  -1, -6, -2,
    -1, 0,  1,  5,  2,  -1, 0,  -6, -1, 1,  -1, 0,  2,  0,   -4, 0,  -2, 2,  -1, 2,  3,  2,  3,  0,  2,  6,   2,  -3,  -2, -1, 0,  -1, -1, 1,  4,  0,  0,  2,
    -2, 1,  -3, -3, -1, -2, 1,  -1, 3,  2,  1,  -1, -5, -3,  1,  3,  -1, -1, 3,  0,  1,  -1, 1,  0,  -1, -3,  3,  4,   -3, 3,  2,  5,  0,  3,  0,  3,  0,  2,
    1,  0,  1,  2,  0,  2,  -3, -2, 0,  3,  2,  0,  3,  4,   2,  2,  -3, -2, 0,  1,  0,  1,  1,  -1, 2,  1,   -1, 1,   -3, 1,  2,  3,  0,  4,  4,  2,  -2, -3,
    -3, -3, -1, -5, -4, -2, 4,  2,  1,  1,  -2, -2, -1, 0,   1,  2,  1,  4,  3,  2,  2,  -5, 1,  3,  -2, -3,  -2, 1,   -2, 1,  -1, 4,  1,  2,  1,  5,  0,  -2,
    1,  4,  3,  -1, -1, -2, -1, 1,  0,  0,  5,  4,  3,  -1,  -3, 2,  -4, -1, 1,  0,  2,  -2, 3,  0,  -2, -4,  -2, 0,   1,  4,  1,  -1, 6,  1,  -1, 0,  1,  -1,
    -2, -1, 3,  2,  -2, 2,  1,  2,  -1, 2,  0,  3,  0,  3,   -1, 0,  0,  1,  2,  1,  -2, -4, -3, 3,  3,  5,   4,  5,   1,  2,  -1, -1, -1, 0,  1,  -1, 1,  1,
    0,  2,  -1, 1,  -4, 2,  1,  1,  0,  0,  3,  1,  -2, -1,  0,  -2, 0,  -2, -6, 1,  5,  4,  -1, 1,  -2, -2,  -6, -3,  -1, 0,  1,  5,  4,  1,  1,  -2, -2, 2,
    -1, -1, -3, 0,  3,  3,  -2, 3,  0,  2,  4,  5,  -1, 1,   2,  4,  3,  1,  -1, -1, -1, 1,  -3, 3,  6,  3,   3,  3,   -3, 1,  -4, -4, 0,  0,  4,  -2, 3,  0,
    -1, -2, -3, -2, 0,  3,  1,  4,  1,  2,  -4, -3, 2,  -2,  -5, 3,  1,  1,  -1, 2,  2,  4,  0,  2,  -1, -2,  -1, 2,   0,  -2, 2,  0,  4,  1,  -5, -2, 3,  4,
    3,  4,  3,  3,  1,  1,  -1, 0,  0,  0,  -1, 3,  4,  -2,  -1, 1,  -1, 2,  0,  -1, 3,  2,  0,  2,  5,  1,   0,  -4,  -1, -2, -3, -3, -2, 0,  6,  3,  -4, -3,
    -3, -1, -6, -1, -2, -4, 1,  7,  2,  1,  0,  -1, 1,  0,   1,  1,  -1, 0,  -1, 3,  3,  5,  3,  1,  4,  6,   2,  0,   1,  4,  4,  1,  -1, -2, 1,  1,  -2, 0,
    4,  0,  3,  2,  -2, 0,  1,  -1, 2,  0,  -1, 0,  -1, 4,   -4, -2, 2,  -1, 0,  0,  1,  1,  2,  -1, 2,  1,   -1, 0,   0,  0,  -2, -1, 4,  2,  5,  3,  3,  4,
    1,  -3, 1,  3,  4,  2,  0,  -1, 3,  5,  0,  -4, 1,  -2,  0,  1,  1,  1,  1,  0,  1,  0,  0,  1,  3,  -1,  3,  0,   1,  0,  1,  -6, 2,  1,  -1, 1,  1,  -1,
    2,  1,  1,  -2, -6, 0,  0,  1,  1,  2,  4,  -2, 1,  3,   2,  -2, -1, 5,  3,  6,  -1, -2, -2, -8, -4, -1,  -1, 2,   -3, 4,  3,  -6, -1, 0,  -3, 1,  2,  3,
    1,  1,  7,  4,  1,  0,  4,  1,  1,  0,  -4, 1,  -2, -4,  4,  -2, -3, 0,  -4, -3, 1,  -2, -2, -1, -1, 0,   -3, 4,   -2, 0,  1,  -2, -1, 3,  -2, 1,  1,  0,
    3,  2,  1,  -2, -1, 0,  -1, 0,  3,  -1, 3,  1,  1,  5,   0,  -3, 1,  5,  4,  1,  1,  -1, 3,  2,  1,  -2,  2,  -1,  0,  2,  -1, -1, -1, 4,  -1, 2,  0,  0,
    2,  1,  3,  -1, 0,  1,  1,  -5, 1,  -1, 0,  1,  0,  0,   3,  1,  1,  -2, -3, 0,  0,  3,  1,  3,  2,  -2,  2,  0,   0,  0,  -1, 5,  2,  4,  1,  -2, -1, -7,
    0,  -1, -2, 0,  -2, 3,  2,  -7, -2, -1, -3, 0,  3,  4,   3,  4,  7,  5,  1,  1,  6,  1,  1,  -2, -7, 3,   -1, -4,  5,  -2, 2,  1,  -4, 1,  3,  -4, 1,  1,
    1,  1,  -2, 4,  -2, 0,  -1, 1,  1,  2,  3,  1,  -2, 1,   4,  2,  1,  -4, -1, -1, 1,  2,  2,  1,  1,  4,   -2, 4,   1,  -3, 0,  4,  -1, 0,  -1, -1, 0,  3,
    1,  -1, 0,  -1, 1,  1,  3,  -3, -2, 3,  -2, 3,  -1, 0,   4,  -1, 1,  0,  -1, 1,  1,  -4, 0,  0,  1,  0,   0,  0,   1,  3,  0,  1,  -1, 5,  4,  0,  2,  6,
    5,  -1, 5,  0,  -3, -2, 0,  2,  4,  4,  1,  0,  -1, -6,  1,  0,  -1, -1, -1, 4,  2,  -4, -1, 0,  -3, 1,   3,  4,   5,  5,  6,  3,  2,  0,  2,  1,  0,  -2,
    -6, 3,  3,  -4, 2,  -1, 1,  -2, -5, -3, 3,  -1, 1,  -1,  2,  3,  0,  3,  -3, -3, 3,  -3, 0,  3,  2,  -1,  -4, 0,   2,  3,  2,  -3, -2, -2, -1, 2,  1,  1,
    2,  7,  3,  3,  2,  -4, 1,  3,  1,  -2, -2, -1, -1, 2,   1,  -1, 1,  -1, 2,  -1, 0,  -1, 0,  3,  0,  2,   0,  0,   2,  1,  -1, 0,  0,  -1, 1,  -2, 1,  -1,
    0,  0,  1,  -1, 0,  1,  1,  1,  -1, 7,  4,  3,  1,  5,   3,  -2, 0,  2,  -2, -2, -2, 5,  2,  1,  1,  -2,  -1, -3,  -1, 0,  -2, -4, 0,  4,  2,  -2, -3, -1,
    -5, 2,  2,  4,  3,  5,  6,  3,  2,  1,  4,  0,  1,  -2,  -7, 2,  -2, -5, 3,  -2, 2,  -1, -6, -4, 0,  -3,  3,  1,   2,  4,  -2, 1,  -3, 0,  1,  -1, 0,  4,
    -1, 0,  -3, 1,  2,  3,  2,  -3, 0,  0,  -1, 3,  2,  1,   2,  6,  1,  2,  0,  -5, 2,  1,  1,  -1, 0,  -1,  1,  3,   1,  1,  2,  -1, -1, 1,  -1, 0,  -1, 1,
    0,  3,  3,  0,  3,  1,  -1, 2,  2,  -1, 1,  -3, 0,  3,   4,  0,  0,  1,  1,  2,  -2, 2,  0,  4,  0,  3,   0,  4,   4,  -2, 2,  4,  -2, -1, -3, 4,  3,  4,
    1,  -2, -2, -5, -1, 0,  -3, -4, -3, 3,  0,  -1, 1,  0,   -5, 0,  1,  4,  6,  5,  9,  1,  3,  2,  2,  0,   0,  -2,  -5, 2,  -1, -2, 4,  0,  1,  0,  -2, -5,
    4,  -5, -3, -3, -1, 3,  -3, 4,  1,  0,  0,  3,  1,  1,   3,  -2, 2,  4,  -1, -2, 7,  4,  -1, 1,  -2, 1,   2,  -2,  0,  2,  2,  1,  1,  0,  2,  0,  -1, 1,
    1,  0,  1,  3,  4,  3,  0,  0,  0,  1,  0,  2,  3,  1,   5,  1,  2,  1,  0,  2,  0,  4,  2,  4,  2,  -1,  4,  1,   4,  0,  2,  -1, 0,  -2, 2,  4,  3,  3,
    3,  0,  3,  1,  -3, 0,  -1, 0,  -1, -1, -5, 0,  -3, 2,   -3, 1,  -1, 2,  2,  4,  -1, 0,  0,  -3, -1, 3,   5,  1,   2,  -5, 0,  -1, -2, -2, 0,  5,  1,  2,
    0,  1,  2,  -1, -1, -1, 1,  1,  -1, -5, -4, -1, -1, -1,  3,  -4, -2, -2, 2,  1,  -1, 3,  1,  -1, -2, 1,   1,  0,   0,  -1, 2,  4,  0,  -1, 6,  2,  -2, 2,
    1,  2,  -1, -2, -2, 1,  3,  1,  0,  0,  1,  0,  -1, 3,   0,  0,  -2, 1,  2,  0,  2,  1,  -1, 2,  3,  0,   2,  3,   5,  2,  0,  -2, 1,  4,  1,  3,  2,  2,
    1,  3,  2,  3,  2,  -1, 2,  0,  -1, -2, 1,  3,  2,  5,   3,  -3, 2,  1,  -1, 2,  -2, 0,  -1, -2, -2, 0,   0,  2,   0,  0,  0,  4,  3,  3,  0,  2,  -1, -1,
    -2, 3,  3,  2,  1,  -5, 0,  -1, 2,  -1, 2,  5,  0,  0,   -2, 1,  4,  -1, 1,  -2, 2,  6,  -3, -6, -3, 0,   3,  2,   2,  -5, 2,  -3, 1,  3,  -1, 3,  -3, 1,
    -2, -1, 2,  1,  -1, -1, -2, 3,  1,  -1, 4,  -1, 1,  1,   3,  1,  0,  -4, -5, 4,  -1, 2,  -1, -1, 1,  1,   0,  1,   2,  -3, -2, 4,  2,  -4, 1,  0,  -1, -1,
    0,  -2, 5,  3,  3,  0,  -1, -1, 0,  -2, 2,  2,  -1, 2,   0,  0,  1,  1,  0,  -1, 0,  1,  -1, -2, 3,  1,   5,  5,   3,  -4, 3,  1,  0,  1,  2,  0,  -3, -2,
    -2, 3,  1,  -3, 1,  0,  -2, 5,  1,  1,  -1, 0,  -4, -2,  -1, 2,  6,  1,  0,  -2, -1, 1,  -5, -4, 0,  3,   -1, 0,   1,  0,  2,  -2, 0,  -5, 0,  1,  -3, -2,
    -1, 1,  2,  0,  4,  -4, 1,  -1, 2,  3,  0,  3,  -1, -3,  -2, 1,  4,  1,  0,  -1, 0,  2,  0,  1,  2,  -1,  2,  1,   2,  1,  -2, -2, -3, 2,  -1, 0,  -1, 1,
    0,  -1, 1,  0,  1,  0,  -1, 3,  1,  -2, -1, 1,  -1, 1,   0,  -1, 4,  2,  4,  2,  3,  -1, -1, -1, 2,  3,   2,  1,   0,  0,  1,  0,  0,  -1, -1, 4,  -2, 1,
    3,  2,  6,  4,  3,  -4, 3,  0,  1,  1,  0,  3,  -4, 0,   -2, 2,  -1, -3, -1, -1, -3, 4,  4,  2,  -1, 3,   -1, 0,   -1, 4,  5,  2,  3,  0,  0,  2,  -1, 0,
    1,  3,  0,  1,  0,  1,  3,  -1, 2,  -1, 1,  5,  0,  -3,  -2, -4, 4,  3,  4,  -6, -1, -2, 1,  3,  -1, 4,   -1, -2,  0,  -2, 4,  1,  0,  0,  2,  -1, 0,  -1,
    2,  2,  0,  1,  2,  1,  -3, -2, -1, 2,  -1, 0,  0,  -2,  -2, 0,  1,  2,  4,  -1, -1, 2,  3,  1,  0,  1,   -1, 2,   -1, -1, 5,  1,  2,  4,  -1, -1, 0,  -2,
    2,  1,  -1, 1,  0,  0,  3,  1,  -1, -1, 1,  -1, -3, -1,  3,  4,  5,  3,  4,  -1, 3,  3,  2,  4,  2,  5,   0,  -2,  -1, 1,  2,  -3, 2,  2,  2,  6,  1,  3,
    0,  4,  0,  -3, -1, 3,  4,  3,  -1, 0,  2,  3,  -5, -4,  -2, 1,  0,  5,  1,  0,  0,  -1, 1,  -2, -1, 3,   -1, -4,  -3, -3, 0,  2,  1,  1,  2,  0,  1,  1,
    -5, -2, -1, 1,  -2, 2,  2,  3,  -3, 0,  5,  2,  2,  2,   3,  2,  0,  0,  1,  -1, 2,  1,  2,  0,  1,  0,   3,  -3,  1,  -2, 2,  -3, 0,  2,  -2, -1, 2,  -1,
    1,  -2, -3, -2, -3, 0,  0,  2,  0,  -1, 1,  3,  1,  -1,  1,  1,  -3, 1,  1,  -2, 0,  -1, -1, -3, 1,  0,   0,  4,   3,  -2, 0,  -1, 0,  -1, 0,  0,  4,  2,
    3,  3,  2,  1,  1,  1,  2,  3,  0,  0,  -1, -4, -1, 0,   -1, 0,  0,  1,  2,  -3, -4, -4, 0,  0,  -2, 2,   2,  0,   1,  2,  0,  -6, -1, 6,  4,  0,  -1, -2,
    -2, -5, -6, 4,  -4, -4, -2, -2, 1,  1,  2,  3,  -1, 2,   -4, -3, 0,  -3, 0,  2,  1,  4,  0,  0,  3,  1,   2,  3,   3,  1,  1,  0,  -2, 1,  2,  3,  0,  0,
    2,  -1, 2,  -2, 1,  -2, 3,  -1, 0,  1,  -2, -2, 1,  -1,  0,  0,  -2, -3, -3, 2,  0,  2,  -1, -2, -3, 3,   0,  1,   0,  1,  -1, 0,  1,  0,  2,  1,  -1, 2,
    1,  1,  2,  3,  4,  1,  -1, 2,  -1, 1,  1,  -1, 1,  1,   4,  2,  3,  -3, 3,  -1, 4,  4,  0,  2,  -1, -4,  0,  -2,  -2, 0,  2,  -1, 1,  -2, -3, -2, 1,  2,
    0,  2,  -1, -1, 2,  0,  -1, -2, 1,  7,  3,  3,  0,  -1,  -2, -2, -2, 0,  -3, -4, -2, 1,  -1, 1,  2,  2,   -2, 4,   -3, -1, 1,  0,  1,  2,  1,  4,  -2, -1,
    1,  1,  3,  1,  3,  0,  1,  1,  -2, 2,  0,  3,  3,  0,   2,  1,  1,  -2, 0,  0,  3,  -2, -2, 1,  -1, -1,  3,  1,   -2, 1,  1,  -4, -1, -2, 0,  2,  -2, -2,
    -1, 3,  1,  1,  2,  1,  0,  0,  0,  -1, 1,  0,  0,  0,   1,  -1, 1,  3,  1,  3,  -1, 0,  0,  0,  3,  1,   1,  3,   3,  0,  2,  -1, 3,  1,  4,  4,  -1, 0,
    -1, -2, 0,  0,  -1, 1,  1,  -1, 1,  -2, -1, -1, -1, -1,  2,  2,  -2, -1, 3,  -2, 0,  -2, 3,  7,  3,  2,   0,  1,   -1, -2, -2, 2,  -2, 1,  -1, 3,  2,  0,
    4,  -2, -1, 6,  -2, -2, -2, 3,  0,  3,  3,  4,  -2, -2,  1,  1,  0,  2,  3,  0,  1,  0,  -1, 3,  0,  -1,  1,  1,   2,  0,  0,  -3, -1, 1,  2,  -3, -1, 1,
    0,  -2, 2,  1,  -1, -1, -2, -4, -1, -1, -1, 2,  0,  -4,  -1, 4,  -1, 2,  2,  2,  -1, -1, 1,  0,  -1, 2,   1,  0,   1,  1,  -1, 2,  3,  1,  -2, -1, -1, -4,
    3,  3,  0,  -1, 1,  -1, 3,  1,  4,  -1, 3,  3,  0,  -1,  -2, -2, 0,  0,  -1, -1, 0,  0,  1,  -1, 0,  0,   -1, 1,   0,  4,  -1, 1,  2,  0,  1,  -1, 3,  8,
    6,  2,  3,  0,  0,  -3, -1, 3,  -1, 1,  3,  4,  -1, 0,   3,  -3, -1, 3,  -2, -3, -1, 3,  -1, 0,  3,  4,   -4, -2,  1,  0,  1,  1,  4,  0,  -1, 3,  0,  1,
    -1, 2,  2,  1,  3,  1,  2,  0,  0,  -1, 1,  -2, -1, 1,   0,  -1, 2,  1,  1,  -2, 0,  -1, 0,  -1, 1,  2,   1,  -2,  -3, 3,  0,  1,  2,  0,  0,  0,  1,  0,
    1,  0,  1,  2,  2,  -1, -1, 0,  1,  1,  2,  2,  -1, -3,  2,  3,  1,  0,  3,  0,  -1, 3,  3,  -2, 5,  6,   0,  2,   2,  -2, -3, 1,  1,  0,  2,  2,  2,  3,
    1,  1,  0,  1,  1,  2,  0,  -1, 2,  1,  3,  -1, 6,  9,   5,  2,  3,  1,  0,  -2, -2, 4,  0,  3,  3,  4,   -1, -1,  0,  -2, -5, 2,  -1, 5,  1,  2,  0,  2,
    2,  -2, 3,  2,  3,  -1, -2, 1,  1,  2,  -2, 0,  4,  0,   -3, -5, 1,  4,  0,  1,  -1, -2, 1,  1,  1,  -1,  -1, 0,   3,  0,  0,  3,  -2, 0,  3,  0,  2,  1,
    1,  1,  2,  2,  0,  -3, -2, 5,  -1, -3, 1,  0,  1,  3,   3,  1,  4,  -3, 2,  -2, 1,  0,  0,  -1, 1,  2,   -2, 2,   -3, 1,  -3, -3, 3,  0,  0,  -2, 2,  4,
    -1, -2, -2, 1,  2,  1,  -1, 3,  0,  3,  0,  3,  2,  0,   -3, -4, -1, -4, 0,  0,  3,  6,  2,  5,  3,  4,   0,  -2,  3,  2,  2,  4,  0,  -7, 0,  -5, -5, -3,
    2,  1,  0,  1,  2,  -3, -6, 4,  1,  2,  2,  -1, -1, 3,   0,  -3, 2,  2,  2,  1,  -3, 0,  0,  0,  2,  -2,  6,  3,   -5, -4, 3,  2,  2,  -1, 0,  -1, 3,  2,
    3,  0,  -1, 3,  2,  1,  0,  0,  0,  -1, 1,  4,  2,  1,   1,  2,  1,  2,  1,  -1, 1,  4,  2,  -1, 0,  0,   -1, 2,   3,  2,  2,  -1, 3,  -4, -2, -1, 4,  -1,
    0,  2,  -2, 3,  -3, 0,  -1, 0,  0,  -2, -1, -4, 3,  5,   1,  -2, -1, 1,  4,  0,  -2, 3,  -1, 3,  2,  4,   1,  -2,  -2, -4, -3, -5, 2,  1,  1,  5,  4,  5,
    2,  3,  3,  -1, 0,  2,  3,  5,  0,  -7, -1, -6, -2, -5,  3,  -1, -1, 2,  1,  -4, -2, 4,  3,  3,  2,  1,   0,  3,   0,  -5, 0,  2,  4,  1,  0,  -1, 1,  3,
    -2, -2, 5,  1,  -3, -5, 4,  3,  1,  0,  -2, -4, 1,  2,   1,  -2, -2, 3,  4,  1,  1,  2,  -1, -1, 0,  3,   2,  1,   1,  1,  2,  3,  1,  0,  0,  5,  3,  1,
    -2, -2, 0,  4,  2,  0,  2,  -1, 2,  -1, 0,  -1, 3,  1,   -1, 0,  -3, 1,  -2, 1,  -2, 1,  1,  -2, 4,  -3,  1,  2,   1,  -2, -4, -1, 6,  3,  -2, 1,  2,  2,
    1,  1,  1,  0,  -2, -1, -3, -2, 3,  1,  1,  7,  6,  2,   2,  1,  1,  -1, 1,  0,  3,  4,  0,  -8, 0,  -4,  -1, -5,  2,  -1, -2, 1,  3,  0,  -3, 6,  3,  2,
    0,  -1, 0,  1,  0,  -2, 1,  2,  2,  1,  0,  2,  -1, 2,   1,  -1, 4,  0,  -3, -3, 4,  4,  1,  -1, -1, -2,  -1, 2,   2,  1,  -1, 3,  5,  -1, 2,  2,  1,  -2,
    1,  3,  3,  1,  0,  3,  2,  1,  1,  -1, 1,  6,  4,  -1,  1,  -1, 0,  4,  4,  -1, 3,  0,  2,  1,  0,  0,   4,  2,   -1, -2, -4, 5,  -2, -2, -1, 1,  0,  -2,
    4,  -1, 4,  2,  -1, 0,  -5, 1,  5,  1,  -1, 1,  6,  0,   -1, 2,  -1, -2, -1, -2, 0,  -2, -2, 1,  3,  6,   5,  2,   2,  3,  3,  -1, 1,  0,  2,  9,  0,  -9,
    0,  -5, 1,  -9, -1, -2, -2, 2,  2,  -2, -4, 4,  1,  2,   3,  1,  1,  2,  -1, 1,  1,  0,  1,  1,  -1, 3,   -3, 0,   -1, 1,  2,  0,  -3, -2, 2,  5,  2,  -1,
    -2, -1, -1, 1,  2,  0,  -3, 1,  5,  -1, 2,  -1, 2,  -3,  1,  0,  2,  2,  0,  1,  1,  1,  -1, -1, 3,  3,   1,  -1,  3,  -1, -3, 3,  3,  -1, 1,  -1, 0,  0,
    0,  2,  2,  0,  -2, -1, -3, 4,  -1, 1,  -5, 1,  1,  -2,  1,  -2, 1,  1,  -1, 0,  -8, 4,  4,  3,  -1, 1,   3,  0,   -2, 1,  0,  -3, -4, -2, -1, 0,  -2, -1,
    4,  6,  3,  3,  0,  2,  4,  0,  0,  -1, 3,  6,  -2, -7,  1,  -5, 0,  -5, -1, -2, -1, -1, 0,  2,  1,  -5,  -4, 2,   6,  0,  7,  2,  -1, 2,  -2, 0,  -2, -1,
    1,  -1, -2, 3,  0,  2,  2,  0,  -2, 2,  3,  2,  1,  1,   -3, -2, 3,  1,  2,  1,  -1, 1,  -3, -2, 0,  6,   5,  0,   1,  1,  0,  4,  6,  3,  4,  -3, -3, -1,
    -1, -2, -1, -1, -2, -2, -1, 1,  -3, -2, -2, -1, 0,  0,   3,  1,  1,  -3, 5,  -1, -2, 1,  0,  2,  3,  2,   1,  -1,  -1, 2,  0,  3,  -5, -4, -6, 0,  1,  -1,
    -2, 3,  0,  2,  0,  3,  2,  0,  -2, 0,  -1, 0,  3,  7,   4,  4,  3,  -1, -1, 1,  3,  1,  0,  4,  4,  2,   0,  0,   -2, -2, 1,  0,  1,  2,  0,  -2, 1,  0,
    3,  -4, -2, 0,  2,  1,  7,  1,  2,  3,  1,  -1, -2, 0,   1,  1,  1,  1,  -1, 2,  1,  -2, -1, 0,  1,  4,   1,  -1,  -1, 0,  0,  1,  1,  3,  0,  0,  -3, -2,
    0,  7,  6,  1,  2,  -1, -1, 4,  4,  3,  1,  -3, -2, -2,  0,  -4, -4, 0,  1,  -1, -1, -1, -2, 0,  -3, 2,   0,  4,   0,  1,  1,  -2, 3,  -1, -3, 0,  -1, 1,
    5,  0,  2,  0,  -3, 0,  0,  -2, -3, -4, -6, -2, -2, -1,  1,  2,  -1, 3,  -1, 3,  1,  4,  -1, 0,  1,  2,   2,  6,   4,  5,  3,  -3, -1, 2,  3,  2,  2,  3,
    3,  2,  -2, -1, -3, -1, 0,  0,  0,  2,  -1, 3,  0,  3,   5,  -2, -3, 3,  -1, -1, 9,  1,  0,  1,  3,  -2,  0,  1,   0,  -1, 0,  -2, 2,  3,  -1, -2, -1, -1,
    1,  3,  2,  -1, -2, 0,  -2, 0,  1,  5,  1,  -1, -2, -2,  -2, 6,  5,  0,  3,  3,  -1, 0,  0,  2,  2,  -1,  -4, 1,   -2, -4, -2, 3,  0,  0,  0,  1,  -2, 0,
    -1, 2,  -3, 2,  3,  0,  0,  -2, 4,  -1, -3, 3,  1,  3,   4,  0,  -1, 2,  -3, 4,  -3, -1, -4, -5, -2, 0,   -4, 2,   2,  1,  0,  2,  0,  2,  2,  2,  -1, 1,
    4,  -1, -1, 4,  -1, 4,  4,  -4, -1, -1, 1,  1,  1,  5,   4,  2,  0,  0,  -2, 1,  0,  -2, 0,  0,  1,  3,   0,  1,   4,  -3, -3, 1,  1,  2,  6,  2,  0,  2,
    0,  4,  0,  1,  -1, 1,  0,  -1, 1,  1,  -2, -3, 1,  -3,  0,  2,  2,  0,  -1, -2, 0,  1,  3,  4,  3,  0,   -4, -3,  -3, 5,  5,  2,  2,  5,  1,  1,  -3, 2,
    4,  -1, -2, -1, -3, -3, -3, 1,  -1, 0,  2,  2,  -1, 4,   0,  -2, -3, 2,  4,  1,  0,  -3, 4,  0,  -1, 1,   -1, 1,   4,  1,  2,  1,  -1, 1,  -1, -2, -4, -2,
    0,  2,  -1, 1,  0,  0,  4,  4,  -3, 1,  1,  2,  -4, 1,   0,  -1, 1,  6,  3,  4,  3,  -2, -1, 1,  1,  -1,  -1, 2,   3,  3,  1,  0,  -2, 1,  1,  -2, 0,  2,
    0,  1,  0,  -2, 4,  -2, 0,  2,  -3, 4,  4,  -1, 1,  1,   -3, -1, -2, 1,  -1, 3,  3,  2,  3,  1,  -2, -3,  1,  -1,  0,  3,  3,  -1, -1, -3, -1, 2,  -1, 0,
    0,  1,  -4, 0,  -3, 1,  4,  3,  3,  1,  0,  -1, 0,  1,   3,  -3, -1, -1, -1, 0,  -2, 1,  -2, -1, 1,  2,   -2, 0,   -1, -1, -1, 2,  4,  1,  -2, -1, 1,  -1,
    -2, 2,  -1, 3,  4,  2,  1,  4,  -2, 0,  1,  0,  -2, -1,  0,  2,  1,  -1, 1,  1,  2,  -1, -5, -3, 0,  2,   -3, 2,   1,  0,  1,  2,  1,  1,  1,  -4, -3, 0,
    0,  0,  -2, 1,  4,  3,  0,  0,  -2, 1,  2,  0,  0,  1,   1,  4,  3,  -1, -1, 3,  0,  2,  -3, -1, -3, -2,  -1, 0,   3,  0,  4,  0,  -2, 2,  4,  2,  2,  -1,
    2,  0,  -2, 2,  1,  -1, 0,  0,  1,  -3, 4,  5,  1,  1,   1,  2,  0,  -2, -4, 0,  0,  1,  0,  0,  -1, -3,  1,  1,   0,  -2, -2, -1, 0,  0,  -2, 2,  -1, -1,
    0,  0,  1,  2,  1,  -1, 3,  2,  0,  2,  3,  1,  2,  0,   1,  3,  1,  1,  1,  2,  4,  3,  2,  -1, 1,  1,   2,  -2,  -1, -3, 3,  -2, 0,  3,  3,  -1, -1, 2,
    -2, -2, -2, -5, 4,  3,  2,  4,  4,  1,  0,  0,  -4, -6,  1,  3,  2,  1,  3,  -3, -2, 2,  -5, -2, -3, 0,   -1, -1,  1,  4,  1,  -1, -3, 2,  -1, -1, -1, -3,
    1,  3,  -1, 1,  -3, 0,  2,  2,  -2, 2,  6,  2,  1,  -1,  -2, -1, 1,  2,  2,  0,  -2, 2,  -1, -5, 2,  2,   2,  0,   3,  2,  -1, -2, -1, 2,  0,  0,  -1, 0,
    -1, -2, 0,  2,  2,  1,  -2, 2,  -1, 0,  0,  1,  2,  0,   -1, -3, 0,  3,  0,  -1, 1,  1,  -2, 3,  3,  3,   1,  1,   3,  2,  1,  2,  -2, -1, 5,  2,  2,  1,
    1,  2,  1,  -3, 1,  0,  3,  -5, 2,  2,  3,  -1, 2,  1,   -3, -5, -1, -4, 1,  -1, 4,  4,  2,  -2, 1,  -1,  -3, -4,  1,  6,  5,  5,  3,  -1, -4, -1, -3, -3,
    -3, 0,  -2, 3,  1,  4,  1,  0,  -3, 3,  -4, -1, 2,  -3,  1,  3,  2,  1,  -1, 1,  5,  1,  1,  1,  4,  -1,  -1, 1,   -4, -2, 0,  1,  0,  -2, -1, 1,  -2, -2,
    1,  3,  3,  0,  3,  4,  1,  0,  -1, 2,  -1, -3, -2, 1,   -3, 0,  -2, 1,  4,  1,  -2, 2,  -2, 0,  0,  2,   -2, 1,   0,  -2, 0,  3,  2,  -1, 1,  2,  1,  4,
    4,  1,  0,  2,  3,  2,  -1, 1,  0,  -3, 3,  1,  3,  1,   2,  0,  4,  -2, 0,  -1, 3,  -3, 1,  0,  1,  -1,  2,  3,   1,  -3, -2, -6, 2,  0,  5,  5,  3,  -1,
    0,  1,  -2, -7, 3,  6,  4,  5,  2,  -2, 0,  1,  1,  -1,  -5, -1, 2,  4,  0,  4,  1,  0,  0,  2,  -1, 1,   0,  -1,  0,  2,  0,  2,  0,  0,  2,  0,  1,  1,
    6,  1,  0,  -1, -1, 0,  2,  2,  0,  1,  0,  1,  0,  -2,  1,  3,  2,  1,  3,  1,  1,  -2, -3, 3,  -3, -1,  0,  -1,  -3, -3, -1, 1,  1,  0,  -2, 1,  -1, 2,
    0,  1,  0,  1,  0,  -1, 1,  4,  0,  1,  3,  1,  2,  4,   2,  3,  -2, 3,  5,  -1, 2,  -2, 1,  -2, -1, 1,   5,  1,   3,  3,  5,  -1, 1,  -2, -2, -2, -2, -2,
    3,  -1, 3,  1,  2,  1,  -2, 0,  -1, -1, 3,  2,  4,  -3,  4,  -2, -2, -4, 0,  6,  2,  4,  3,  -1, -1, 0,   -6, -1,  -4, -1, 4,  6,  0,  8,  1,  -1, 0,  5,
    -4, 1,  0,  0,  3,  3,  3,  0,  -2, -1, 0,  2,  1,  -1,  4,  0,  -1, 1,  -3, -1, 2,  3,  -1, -1, 1,  0,   1,  0,   -1, 4,  4,  0,  1,  3,  5,  -2, -3, 3,
    -3, -1, -2, 0,  -3, -4, 0,  1,  2,  -2, -1, 3,  0,  0,   -1, 0,  -1, -1, 1,  2,  3,  1,  0,  1,  -1, 3,   0,  2,   1,  2,  -2, 6,  3,  1,  0,  -3, -2, 0,
    0,  -2, 2,  -1, 1,  2,  3,  -3, -1, -1, -1, -4, 2,  -2,  1,  1,  2,  3,  0,  -2, -2, -2, 2,  3,  6,  4,   3,  -3,  1,  0,  1,  -4, 2,  4,  2,  3,  4,  -1,
    1,  1,  -6, -2, -2, 0,  2,  6,  2,  -3, 2,  -1, 0,  1,   2,  -2, 0,  -1, 2,  1,  -1, 0,  -1, 5,  -1, 0,   0,  1,   -1, 2,  0,  1,  0,  -4, 2,  1,  0,  -1,
    0,  1,  -2, -1, 1,  -1, -1, 2,  -4, -7, 0,  2,  0,  -1,  2,  -1, 3,  3,  2,  -1, 5,  2,  -1, 2,  0,  -1,  5,  0,   2,  0,  2,  0,  -1, -2, -1, -1, -1, -2,
    -2, 2,  1,  -1, -1, 0,  5,  0,  0,  1,  2,  -1, 2,  6,   8,  3,  0,  2,  0,  3,  -3, 0,  -4, 0,  2,  -2,  2,  -1,  -2, 0,  1,  1,  2,  1,  0,  1,  0,  4,
    3,  -3, -4, 1,  0,  -2, -1, 1,  -3, -3, -3, 1,  1,  -3,  1,  2,  1,  1,  1,  -1, 2,  3,  1,  1,  3,  -1,  0,  1,   6,  -1, 0,  -2, 5,  2,  -3, -2, -2, 2,
    2,  -3, -2, -2, -1, -1, 2,  0,  -2, -5, 4,  2,  2,  -5,  -2, -1, 0,  -2, 0,  0,  -3, 1,  -3, -6, -5, 0,   1,  2,   -1, -2, 3,  2,  3,  2,  3,  1,  0,  -1,
    1,  2,  4,  -1, 1,  -1, -1, -2, 0,  -1, 0,  -1, -2, -3,  -3, 1,  -1, 0,  0,  1,  1,  0,  0,  1,  0,  0,   3,  5,   4,  4,  0,  0,  1,  1,  1,  1,  -3, -3,
    1,  -1, -2, -1, 0,  3,  5,  2,  4,  0,  0,  0,  2,  0,   4,  -3, -5, 0,  2,  -1, -2, 1,  -1, -2, -3, -1,  2,  -1,  -1, 2,  2,  3,  2,  0,  3,  -1, 2,  0,
    3,  0,  1,  -2, 4,  1,  0,  2,  2,  2,  -3, -1, 2,  4,   1,  -2, 0,  0,  -1, 0,  1,  -2, -1, -6, 3,  2,   4,  -4,  0,  1,  -1, -1, 1,  1,  -4, 4,  -1, -6,
    1,  0,  -1, 1,  0,  -2, 3,  2,  2,  2,  3,  1,  0,  -1,  -1, -1, 5,  -1, 1,  -2, -1, -2, 0,  -1, -1, -3,  -2, -3,  -3, 0,  0,  1,  1,  2,  2,  1,  1,  -1,
    0,  -2, 4,  6,  4,  3,  1,  -1, 1,  3,  2,  0,  -2, -2,  1,  -2, 2,  -1, 3,  3,  3,  1,  4,  0,  2,  2,   4,  0,   2,  0,  -1, 0,  3,  -1, -1, 1,  -2, -2,
    -4, 1,  3,  -2, -2, 4,  3,  2,  1,  1,  2,  1,  2,  0,   2,  0,  3,  -2, 3,  2,  -3, 1,  4,  0,  -3, -2,  4,  2,   3,  2,  -1, -1, 1,  -2, 3,  -2, -2, -5,
    0,  2,  2,  -3, -2, -2, 1,  2,  0,  -2, -3, 0,  1,  -7,  2,  0,  -2, 3,  -1, -3, 1,  2,  2,  3,  0,  4,   1,  -2,  -2, 0,  3,  0,  1,  0,  0,  -2, 1,  0,
    -3, -3, 0,  -2, 0,  0,  0,  2,  -2, 0,  4,  -2, 2,  0,   0,  -1, 5,  4,  5,  4,  -2, 0,  2,  2,  2,  2,   -2, -2,  1,  -5, 1,  0,  6,  4,  5,  1,  6,  2,
    4,  0,  0,  0,  2,  0,  -3, 1,  2,  -2, 0,  2,  -1, -1,  -3, 1,  3,  -1, -4, 3,  3,  0,  0,  -1, 2,  0,   0,  -4,  2,  1,  1,  0,  1,  0,  2,  -1, 7,  -2,
    -3, 0,  0,  3,  3,  0,  1,  2,  -3, -2, 1,  0,  -3, -6,  1,  0,  0,  -2, -2, -1, 4,  1,  2,  -1, -5, -1,  1,  -6,  1,  -1, 1,  2,  -1, -1, 1,  4,  2,  3,
    -1, 4,  1,  -3, 0,  0,  5,  1,  -1, -2, -2, -2, -1, 2,   -2, -2, -2, -2, -2, 0,  -2, 1,  -3, -2, 4,  -3,  0,  -2,  -2, -5, 3,  3,  8,  1,  1,  -2, 0,  1,
    1,  0,  -2, -2, 1,  -1, 2,  -1, 2,  5,  4,  3,  6,  1,   4,  -2, 1,  3,  3,  -2, -1, 2,  0,  -2, 3,  1,   -1, -2,  -1, 1,  2,  -1, -3, 4,  4,  0,  1,  0,
    1,  0,  5,  2,  3,  2,  -1, 2,  2,  2,  3,  0,  -2, -1,  4,  2,  -2, 5,  5,  -2, -1, -1, 1,  5,  1,  0,   0,  -2,  -7, 2,  0,  0,  2,  1,  1,  3,  1,  1,
    2,  4,  2,  5,  0,  0,  1,  1,  -2, -3, 2,  1,  -1, -4,  0,  1,  1,  1,  -2, 2,  2,  -2, 2,  3,  1,  3,   1,  1,   3,  1,  -1, 2,  1,  0,  1,  3,  0,  1,
    -1, 2,  -1, 4,  -2, 1,  2,  0,  1,  0,  -3, -1, 1,  3,   4,  0,  2,  -4, -3, 1,  -2, 0,  2,  2,  0,  7,   5,  1,   -4, -2, -5, 4,  5,  1,  3,  3,  2,  1,
    -1, 6,  3,  2,  3,  3,  3,  3,  -2, -8, -4, -1, 1,  -3,  5,  0,  3,  1,  2,  3,  0,  -1, -1, 3,  1,  1,   -1, -1,  6,  3,  -1, -2, 6,  -2, -3, -1, 2,  2,
    0,  -2, 0,  -2, -8, 1,  0,  -2, -1, 0,  2,  2,  3,  -1,  1,  1,  5,  5,  -1, 0,  0,  2,  0,  -2, 0,  -1,  2,  0,   0,  1,  -1, 3,  -2, 2,  6,  -1, 1,  0,
    1,  0,  1,  2,  3,  0,  -1, 0,  2,  2,  -1, -2, -3, -1,  3,  -1, 0,  3,  -2, -1, 0,  0,  0,  1,  -1, -1,  -3, -4,  0,  -1, 3,  -3, -1, 1,  -4, 1,  2,  1,
    -1, 6,  4,  4,  -2, 0,  -5, 3,  2,  -1, 2,  7,  7,  4,   1,  5,  7,  3,  0,  3,  2,  4,  -5, -9, -2, -1,  -2, -3,  4,  -2, 3,  4,  1,  6,  -1, -1, -2, 2,
    1,  3,  -4, 1,  5,  2,  -1, 2,  3,  2,  2,  1,  1,  3,   4,  -5, 0,  0,  -5, 1,  -1, -4, 1,  -1, 2,  3,   2,  0,   -1, 2,  5,  1,  2,  0,  2,  1,  2,  -2,
    -1, 2,  3,  0,  1,  1,  1,  -1, -1, -1, 3,  1,  0,  3,   1,  0,  1,  1,  3,  -2, -2, -1, 4,  3,  4,  -1,  0,  -2,  1,  1,  2,  3,  -2, 0,  -1, -1, 0,  4,
    -2, -1, 0,  -1, -1, -3, 1,  -3, -3, 2,  -4, 2,  4,  2,   -3, 7,  6,  3,  -1, 1,  -4, -1, 2,  -3, 4,  5,   4,  3,   0,  5,  8,  0,  0,  1,  4,  7,  -4, -8,
    0,  -2, -3, -2, 3,  0,  1,  -1, 3,  2,  -2, -2, -4, 2,   0,  2,  -4, -1, 1,  -1, 0,  2,  5,  0,  1,  -3,  -1, 1,   1,  -5, 1,  -1, -3, 0,  -3, -4, -1, 0,
    2,  3,  2,  1,  2,  3,  2,  1,  0,  -2, 2,  3,  -1, -2,  -3, 3,  2,  -1, 1,  1,  0,  2,  -3, 2,  2,  4,   2,  0,   1,  0,  3,  2,  4,  -1, 1,  -5, 5,  2,
    0,  -5, -2, -5, -3, 2,  2,  4,  2,  3,  -2, -3, 3,  1,   -5, -3, -4, -2, 2,  -5, 0,  -4, -3, 3,  1,  0,   0,  5,   2,  5,  4,  1,  -6, 0,  -6, 0,  0,  -4,
    0,  6,  4,  2,  0,  5,  4,  0,  0,  2,  4,  5,  -5, -10, -2, -3, -2, -3, 3,  0,  -1, -1, 4,  2,  -1, -2,  -5, -2,  3,  1,  -3, 2,  0,  1,  3,  5,  4,  1,
    0,  -3, -3, 3,  1,  -1, 4,  2,  -3, 1,  0,  0,  1,  0,   2,  2,  0,  3,  2,  2,  2,  4,  -1, -1, 2,  4,   0,  -1,  -2, 1,  0,  -1, 2,  2,  1,  2,  0,  0,
    1,  2,  0,  3,  2,  0,  0,  1,  1,  -1, 1,  -2, 3,  4,   4,  -2, 0,  -1, 1,  2,  -1, 1,  0,  1,  -2, -2,  0,  1,   -2, -1, -2, 1,  1,  -7, -1, -3, -3, 6,
    -1, 1,  1,  3,  3,  10, 4,  2,  -5, 4,  -4, 2,  5,  -1,  2,  5,  3,  2,  1,  5,  6,  2,  0,  1,  6,  1,   -4, -9,  -5, -6, 0,  -2, 4,  4,  1,  0,  2,  0,
    2,  0,  -1, 1,  -3, 1,  1,  1,  -1, 1,  -1, 3,  2,  3,   1,  1,  1,  1,  1,  2,  -1, -1, 2,  0,  -2, -1,  -1, 1,   3,  -2, 0,  -1, -1, 4,  -2, -3, 2,  0,
    0,  0,  -2, -1, -1, 0,  -1, 1,  1,  1,  1,  1,  0,  2,   1,  0,  1,  5,  2,  0,  2,  2,  -1, 1,  2,  2,   2,  2,   2,  1,  -3, 2,  3,  3,  1,  -2, -4, 2,
    -1, 1,  1,  -2, -2, -2, 1,  -3, 0,  -2, -1, 1,  3,  -1,  -1, 0,  1,  -1, 1,  2,  -3, 2,  -1, 1,  2,  -1,  2,  0,   0,  -1, 1,  -2, -1, 0,  4,  0,  1,  2,
    1,  2,  0,  -2, 1,  2,  2,  0,  -1, 1,  2,  -1, 2,  -1,  1,  -2, -1, 2,  -2, -2, 2,  -1, -1, 3,  -3, 1,   4,  0,   1,  3,  4,  -3, 2,  3,  -2, -1, 1,  -1,
    0,  0,  -1, 3,  0,  -1, -1, 0,  2,  1,  -1, -2, 0,  1,   0,  0,  2,  -1, 0,  3,  0,  0,  -1, 0,  1,  3,   -2, 1,   0,  2,  1,  5,  0,  1,  1,  1,  2,  3,
    0,  2,  6,  2,  3,  -1, -4, 3,  3,  3,  -1, -3, -4, 2,   -2, 1,  4,  -2, 0,  -2, 2,  -2, -2, -3, 1,  1,   -1, -1,  -1, 0,  0,  -1, -2, 2,  -1, 0,  -1, 3,
    5,  -4, 3,  1,  0,  -3, -1, 1,  -1, 2,  5,  -1, 1,  3,   2,  -1, 0,  -2, -1, 3,  2,  1,  -1, -1, 2,  -1,  0,  2,   0,  -1, 1,  0,  -1, -1, 2,  0,  0,  2,
    0,  4,  4,  2,  4,  1,  2,  -1, 4,  2,  -1, 3,  2,  -2,  -3, -1, 4,  3,  2,  0,  2,  1,  0,  2,  -1, -4,  2,  4,   2,  1,  3,  0,  1,  4,  3,  3,  1,  2,
    5,  1,  -1, -2, 0,  2,  2,  6,  2,  2,  3,  1,  2,  2,   2,  3,  4,  4,  5,  1,  -1, 3,  2,  1,  -1, -1,  -3, 1,   -1, 0,  3,  0,  -1, -1, 1,  1,  -4, -3,
    0,  3,  2,  0,  2,  1,  0,  -1, 1,  3,  -2, 0,  -2, 0,   5,  2,  3,  1,  1,  -1, 0,  0,  0,  1,  5,  0,   1,  1,   3,  3,  3,  5,  4,  5,  4,  2,  1,  2,
    5,  0,  1,  -1, 1,  -2, 2,  3,  -1, 1,  2,  -3, -1, 1,   -1, 1,  3,  2,  1,  1,  2,  -1, 3,  2,  -1, 1,   0,  -2,  -3, 0,  0,  0,  2,  -2, 0,  2,  0,  1,
    -4, -2, 0,  2,  1,  -3, 1,  1,  -3, -1, 1,  -3, 2,  0,   0,  0,  -5, -1, -2, -1, 0,  3,  1,  0,  2,  -1,  1,  0,   0,  2,  2,  3,  1,  -1, -1, 0,  3,  2,
    -1, -3, -4, 2,  -1, 0,  3,  0,  0,  -3, -2, 1,  -2, -4,  1,  2,  1,  -1, 2,  -3, -2, 1,  0,  1,  -2, -1,  1,  -1,  2,  -1, 1,  1,  -2, -3, -3, 0,  -1, 2,
    0,  -1, 0,  -1, 0,  -3, -2, 1,  0,  2,  2,  0,  -2, -2,  2,  3,  1,  -1, -1, -1, 0,  2,  3,  -1, 1,  -1,  0,  2,   -1, 1,  3,  1,  3,  1,  3,  2,  2,  0,
    1,  0,  -1, 0,  -3, -3, 0,  0,  -1, 0,  -2, 2,  2,  0,   -2, -1, -1, 2,  2,  -2, 1,  2,  -2, 1,  1,  3,   0,  1,   2,  2,  0,  -1, -1, 4,  -2, 3,  1,  0,
    4,  2,  0,  4,  2,  2,  2,  2,  1,  -2, -1, -1, 2,  2,   0,  1,  -3, 1,  0,  0,  0,  -1, 1,  0,  1,  2,   -3, -2,  0,  1,  3,  -3, 0,  1,  -3, -3, 1,  3,
    -5, -1, 0,  0,  2,  0,  0,  1,  2,  0,  1,  -2, -1, 0,   2,  0,  1,  2,  2,  0,  6,  4,  4,  3,  3,  -1,  1,  1,   -1, 3,  0,  -3, 0,  2,  -1, 0,  -1, -1,
    0,  4,  3,  1,  2,  1,  -3, 0,  -2, -3, 4,  1,  -1, 0,   -2, 4,  3,  5,  3,  -1, 2,  -1, 0,  3,  3,  -2,  1,  2,   -2, -2, 1,  0,  1,  4,  2,  0,  -3, -3,
    -2, 4,  1,  -1, 0,  -1, 0,  1,  -3, 2,  -2, 4,  0,  2,   0,  1,  -1, 4,  -2, 2,  -1, 1,  0,  2,  0,  -1,  2,  3,   3,  1,  2,  1,  3,  1,  1,  4,  -2, 3,
    1,  2,  0,  1,  0,  0,  0,  -1, 1,  2,  -1, 0,  3,  1,   0,  1,  -5, 0,  3,  5,  5,  8,  4,  5,  1,  1,   2,  5,   3,  0,  -1, 3,  5,  5,  0,  -5, 4,  3,
    1,  1,  3,  1,  0,  0,  0,  -2, 0,  1,  -2, 1,  -2, 1,   0,  3,  1,  1,  1,  0,  -4, -1, -2, -2, 4,  3,   0,  -1,  -3, 6,  2,  2,  3,  -1, -1, 0,  0,  4,
    1,  -1, 1,  2,  -2, -1, 1,  2,  3,  1,  2,  0,  -1, 0,   1,  2,  2,  -1, -1, -2, 0,  1,  -2, 1,  1,  5,   -1, 1,   0,  3,  -2, 3,  0,  -1, 0,  2,  -2, 2,
    0,  1,  1,  2,  3,  -3, 0,  0,  1,  -2, 0,  4,  -4, 2,   1,  3,  3,  2,  3,  -1, 2,  1,  2,  3,  4,  -1,  2,  1,   0,  2,  -3, 0,  -1, 4,  3,  5,  4,  3,
    3,  2,  1,  0,  3,  -1, 2,  2,  2,  1,  -4, -3, 0,  3,   1,  3,  3,  1,  2,  -1, 0,  -1, 0,  4,  -1, 1,   -1, 4,   2,  1,  2,  -1, 0,  0,  -5, -2, 1,  0,
    3,  3,  1,  1,  -4, 4,  0,  2,  2,  0,  1,  -2, -1, 3,   2,  0,  1,  2,  -1, 1,  0,  0,  1,  1,  0,  0,   0,  -1,  2,  1,  3,  -4, -1, -1, 1,  2,  1,  -3,
    -1, 3,  -2, 0,  -1, 3,  -2, 2,  -1, -1, -1, 1,  -3, 1,   -2, 0,  0,  3,  2,  -2, 0,  1,  5,  0,  1,  3,   -3, 6,   5,  4,  2,  1,  3,  0,  3,  1,  0,  3,
    2,  0,  1,  0,  2,  -1, -3, 2,  -2, 7,  2,  5,  4,  5,   0,  3,  1,  -1, 0,  2,  2,  2,  2,  1,  -2, -2,  1,  4,   3,  2,  3,  2,  1,  -1, -2, -1, 1,  0,
    -2, 0,  1,  2,  -1, 2,  4,  -2, 4,  1,  -4, -2, 2,  1,   4,  3,  1,  -1, -3, 1,  -3, 4,  3,  -1, 1,  0,   1,  5,   2,  -1, 2,  2,  0,  0,  -2, 1,  3,  0,
    1,  1,  -1, 0,  3,  2,  4,  -4, -1, -3, 1,  2,  -2, 1,   -1, 1,  1,  2,  0,  0,  -1, 6,  -3, 1,  2,  2,   -2, 0,   1,  -2, 0,  3,  1,  -1, -1, 0,  4,  -1,
    1,  1,  0,  5,  3,  3,  3,  2,  2,  -2, 2,  3,  0,  1,   4,  1,  -1, 2,  1,  2,  -2, 2,  1,  5,  2,  4,   3,  6,   1,  3,  2,  2,  2,  2,  3,  1,  3,  2,
    -1, -1, 3,  5,  1,  2,  1,  3,  -1, -1, 0,  -1, -3, 0,   -1, 1,  1,  3,  1,  1,  2,  3,  1,  -1, -1, -1,  1,  0,   3,  1,  2,  0,  -4, 2,  0,  1,  0,  -1,
    2,  0,  2,  4,  1,  0,  1,  4,  -1, 1,  -5, -1, 3,  1,   2,  1,  -4, -1, 3,  0,  4,  -1, 1,  -1, 3,  0,   -2, 1,   -2, 1,  -1, 2,  0,  2,  0,  3,  -2, -1,
    -1, 3,  -1, 1,  -3, -1, 0,  4,  -1, -3, -2, 1,  5,  0,   3,  3,  -1, 6,  0,  1,  0,  4,  1,  -1, 1,  5,   0,  -2,  0,  3,  3,  -1, 1,  1,  -2, 1,  1,  4,
    1,  6,  2,  5,  0,  2,  2,  1,  0,  1,  3,  0,  3,  1,   -2, -2, 5,  2,  1,  3,  0,  3,  -1, 0,  0,  1,   4,  0,   0,  -1, -2, 3,  1,  2,  0,  5,  4,  6,
    3,  -3, 0,  -1, 6,  0,  4,  1,  1,  -2, 1,  -1, -1, -1,  -1, -2, 0,  -1, 1,  2,  2,  0,  0,  5,  2,  0,   -2, -1,  1,  0,  2,  1,  0,  -1, 2,  2,  5,  -3,
    3,  2,  3,  0,  1,  2,  0,  -1, 0,  5,  3,  0,  -1, 1,   3,  1,  4,  2,  -1, 4,  -1, 1,  -1, 0,  0,  2,   2,  1,   -1, 0,  -2, 4,  6,  4,  4,  -1, 0,  -1,
    -1, -3, -2, 2,  -1, 1,  0,  2,  0,  -5, 3,  1,  2,  5,   -2, 2,  0,  0,  4,  0,  -3, -4, -2, 4,  3,  -2,  -1, 0,   2,  1,  0,  2,  5,  5,  5,  4,  -1, 1,
    3,  0,  1,  3,  -1, 0,  -1, 2,  1,  0,  1,  4,  3,  1,   1,  -5, -3, 0,  6,  0,  4,  2,  -1, -1, 0,  0,   -1, 0,   0,  -4, -2, 0,  -1, 2,  1,  1,  1,  4,
    6,  -3, -1, 0,  2,  -2, 1,  0,  -3, -4, 1,  0,  2,  -3,  2,  1,  1,  2,  2,  3,  2,  2,  2,  6,  2,  -1,  0,  1,   3,  1,  1,  1,  -2, 1,  -1, -3, -1, -2,
    0,  3,  2,  2,  1,  -2, -3, 3,  4,  2,  4,  -1, -1, 1,   0,  -1, 1,  2,  -1, -1, -3, -1, -2, 1,  2,  -1,  3,  2,   3,  2,  -3, 0,  2,  -1, -1, -3, -1, 3,
    1,  -2, -1, 1,  -1, 1,  0,  0,  2,  4,  2,  1,  2,  -1,  0,  1,  0,  1,  -1, -4, 1,  2,  -1, -1, 4,  1,   3,  2,   1,  -2, -5, 2,  4,  0,  -1, 4,  1,  0,
    0,  -1, -1, 1,  2,  0,  -2, 2,  1,  -1, 3,  -1, 2,  5,   6,  -1, -2, 3,  4,  -1, 0,  1,  0,  -5, -2, 1,   5,  1,   2,  -1, 1,  -1, 1,  1,  1,  0,  2,  3,
    0,  -2, 0,  3,  2,  0,  4,  2,  0,  0,  -1, 0,  -3, 0,   1,  3,  3,  1,  1,  1,  -1, 0,  1,  4,  5,  -2,  -2, -1,  0,  0,  2,  1,  1,  3,  0,  -1, 1,  0,
    -1, 2,  2,  2,  3,  4,  -3, 1,  2,  -1, 0,  -3, 0,  2,   -1, -4, -3, -1, -1, 1,  -2, -1, 2,  2,  4,  1,   3,  -1,  1,  -3, 0,  4,  -3, -2, 1,  0,  0,  -1,
    3,  0,  2,  0,  2,  -1, -4, 2,  3,  -1, -2, 3,  1,  -3,  3,  -1, -2, 1,  1,  0,  0,  2,  4,  0,  1,  -2,  0,  6,   1,  -2, -5, 0,  3,  2,  -1, -1, -3, -3,
    1,  0,  6,  -1, 1,  0,  -1, 1,  2,  0,  0,  0,  3,  4,   0,  1,  -2, 2,  0,  0,  4,  2,  1,  0,  0,  -4,  0,  0,   3,  2,  3,  0,  -1, -1, 0,  -1, 3,  4,
    3,  0,  0,  1,  1,  1,  3,  2,  3,  2,  2,  1,  4,  -3,  0,  -1, 2,  2,  5,  3,  0,  2,  1,  1,  0,  -3,  0,  2,   -3, -2, -4, -3, -3, 1,  -2, 2,  1,  1,
    3,  -1, 3,  -1, 0,  -2, -2, 4,  2,  -1, 3,  0,  0,  1,   2,  -1, 4,  -2, 1,  -1, -2, -1, 3,  -1, -1, 1,   2,  3,   4,  -1, -3, 3,  2,  1,  -2, 4,  4,  0,
    1,  -2, 1,  4,  1,  1,  -1, 2,  1,  0,  -2, 1,  -2, 0,   0,  3,  5,  -1, -1, 4,  -3, -1, 3,  1,  0,  1,   2,  3,   3,  -1, 0,  3,  -2, -2, 4,  3,  0,  3,
    0,  -6, 1,  3,  1,  1,  1,  -1, -4, -1, 2,  3,  0,  5,   2,  -1, 1,  2,  2,  3,  5,  2,  -1, 0,  -2, -1,  3,  -1,  -2, -1, 1,  4,  4,  1,  -2, 0,  0,  -2,
    -1, -4, -1, 1,  -2, -4, -2, -1, -2, 1,  -1, -1, 3,  0,   2,  -1, -3, 0,  5,  0,  0,  2,  -2, -2, -3, 2,   -1, 0,   0,  0,  0,  0,  -1, 1,  -1, -4, 2,  -4,
    3,  1,  2,  5,  4,  4,  1,  1,  -2, -3, 0,  -2, -1, 3,   1,  4,  -1, 2,  3,  2,  -1, 2,  3,  3,  2,  -2,  -2, 0,   1,  -2, 0,  0,  3,  -2, -1, 0,  0,  0,
    1,  -2, -1, 0,  3,  5,  0,  0,  -1, 3,  5,  1,  0,  -3,  -2, -2, -1, 2,  -2, 0,  0,  2,  -1, 0,  0,  -1,  2,  1,   1,  2,  -1, 2,  2,  4,  5,  5,  1,  3,
    0,  4,  3,  4,  2,  1,  1,  2,  0,  -1, 3,  -1, -1, 1,   1,  2,  -2, -2, 1,  3,  0,  -1, -5, -6, -2, 1,   -1, 1,   2,  3,  0,  1,  2,  2,  2,  -2, 4,  0,
    -3, -2, 0,  3,  -1, 1,  1,  0,  0,  0,  -1, -2, 2,  -3,  3,  0,  -1, 6,  5,  3,  -1, 0,  -2, -1, 1,  -1,  -1, 2,   3,  2,  -4, 0,  0,  2,  -2, 4,  4,  2,
    3,  -3, -2, 0,  1,  -3, 1,  1,  4,  -3, -1, 3,  1,  1,   2,  0,  -2, 2,  2,  2,  2,  1,  0,  4,  4,  3,   2,  -2,  -1, -2, -1, 4,  -2, 3,  -1, 0,  -1, -1,
    -1, -1, 3,  2,  3,  2,  -2, 0,  1,  4,  4,  5,  -1, 2,   0,  3,  1,  1,  0,  4,  1,  2,  1,  2,  3,  1,   0,  -5,  -1, 4,  -1, -1, 0,  2,  -1, 0,  -2, -6,
    -6, 0,  1,  2,  2,  4,  -2, -1, 3,  2,  0,  -1, 0,  2,   -5, -3, 0,  3,  0,  3,  1,  3,  1,  0,  0,  1,   3,  -5,  2,  0,  -2, 6,  5,  6,  -3, 0,  -4, 0,
    -1, -2, -1, 4,  4,  2,  -2, 2,  0,  2,  -2, 3,  4,  1,   3,  -3, -1, -1, -1, 0,  -2, 2,  1,  -2, 0,  2,   1,  0,   0,  0,  -2, 1,  6,  2,  -2, 0,  -2, 3,
    3,  2,  1,  -2, -5, -2, 2,  2,  -2, 1,  1,  0,  0,  0,   0,  0,  3,  2,  2,  1,  -2, 0,  3,  2,  5,  5,   -2, 1,   -3, 5,  2,  2,  2,  0,  2,  2,  0,  1,
    0,  1,  -1, -4, 1,  0,  1,  -1, 1,  1,  -1, 0,  -2, -6,  -6, 1,  2,  2,  1,  3,  -2, 1,  -1, 2,  0,  -1,  0,  1,   -6, -2, 0,  3,  -2, 3,  2,  1,  3,  0,
    -1, 1,  3,  -7, 1,  2,  0,  4,  6,  5,  -1, -1, -2, -4,  1,  -3, 0,  2,  3,  2,  -3, 2,  1,  4,  -3, 2,   3,  3,   4,  -4, -1, -1, 0,  0,  -1, 2,  0,  -2,
    1,  0,  -1, 1,  3,  -1, -3, 3,  5,  1,  1,  -1, -1, 3,   3,  2,  2,  -2, -4, -3, 0,  3,  -1, -1, 0,  1,   0,  4,   0,  -1, 4,  4,  2,  3,  1,  -1, 3,  4,
    3,  4,  -2, -1, -4, 4,  2,  0,  -3, 2,  1,  0,  1,  -1,  0,  2,  -1, -2, -1, 1,  1,  -1, -2, -1, 2,  2,   -5, -6,  -3, 1,  2,  0,  0,  3,  -3, 0,  0,  -1,
    0,  -3, 3,  0,  -6, -3, 0,  2,  -3, 2,  2,  1,  2,  1,   -3, 0,  4,  -6, 2,  -1, -1, 4,  2,  7,  1,  1,   -2, -1,  -2, -2, 3,  3,  5,  -1, 0,  5,  3,  2,
    -2, 1,  3,  1,  3,  -2, -2, -3, 1,  -2, 1,  1,  3,  -5,  0,  0,  -1, 1,  1,  0,  -1, 3,  3,  2,  1,  1,   0,  2,   3,  1,  1,  -3, -2, -5, 3,  2,  -4, 1,
    -1, 1,  1,  3,  2,  0,  4,  1,  1,  3,  -1, 0,  2,  4,   3,  2,  -1, -2, -1, 5,  1,  1,  -1, 1,  -3, -1,  -1, 0,   2,  2,  -4, -1, -2, 0,  -1, -2, -1, 2,
    4,  0,  -4, -8, -2, -1, 0,  -3, -3, 0,  1,  5,  2,  0,   2,  -2, 1,  -1, 3,  3,  2,  1,  -2, 0,  -2, -1,  -2, -1,  -3, -2, -2, 0,  3,  0,  0,  4,  -4, 0,
    -3, 1,  -1, 2,  -1, -1, 0,  1,  -3, -1, 1,  2,  0,  2,   1,  -1, 1,  -1, 0,  3,  0,  -2, 0,  0,  1,  1,   0,  4,   1,  5,  2,  2,  2,  3,  2,  3,  1,  0,
    3,  3,  2,  0,  1,  2,  -2, 1,  1,  0,  2,  0,  0,  -3,  0,  1,  0,  1,  0,  -4, 3,  0,  2,  0,  2,  3,   1,  1,   -1, 0,  0,  -1, 0,  1,  0,  1,  2,  -5,
    -3, 3,  0,  4,  1,  4,  4,  1,  1,  1,  0,  0,  2,  3,   5,  3,  -3, -3, -3, -4, -4, 0,  -1, 1,  0,  3,   0,  0,   2,  -2, -1, 0,  1,  5,  1,  -1, -1, 2,
    -1, 0,  -3, 0,  -1, -1, -1, 0,  1,  2,  4,  3,  -2, -1,  0,  3,  0,  1,  -1, -1, 3,  -1, -4, -2, 1,  3,   3,  1,   3,  0,  -2, 0,  1,  1,  1,  -4, -1, 1,
    0,  0,  1,  0,  2,  3,  2,  1,  1,  3,  2,  4,  0,  0,   2,  0,  0,  -1, 3,  2,  1,  2,  2,  -2, 1,  -1,  3,  -6,  -2, 1,  0,  2,  0,  -2, 4,  -3, 1,  -1,
    0,  3,  0,  3,  4,  -3, -1, 0,  1,  4,  0,  -1, -1, -6,  0,  3,  -1, 1,  3,  3,  3,  1,  0,  1,  2,  0,   2,  2,   4,  2,  -2, -4, -4, -2, -5, 0,  -3, 2,
    0,  1,  0,  1,  0,  -3, 1,  1,  1,  4,  0,  -2, 0,  2,   0,  -2, -1, 2,  -2, 1,  -3, 0,  0,  3,  1,  0,   -1, 1,   2,  5,  1,  1,  -1, 0,  1,  -2, -3, -1,
    -1, 2,  0,  2,  3,  0,  -2, 1,  0,  2,  -2, -1, 1,  0,   0,  0,  -1, 0,  1,  4,  3,  2,  0,  2,  1,  4,   -1, 2,   4,  0,  1,  1,  1,  3,  2,  1,  -2, 0,
    1,  -3, 1,  -3, 0,  4,  2,  0,  -2, -1, 2,  0,  3,  0,   1,  4,  -1, 2,  3,  -1, 1,  0,  3,  6,  1,  0,   -2, -5,  -1, -1, -2, 1,  3,  3,  3,  2,  -1, 2,
    2,  1,  1,  0,  3,  0,  -3, -5, -4, -2, -6, -3, -3, -1,  -1, 3,  1,  1,  0,  -3, 1,  0,  1,  2,  0,  -3,  2,  0,   -1, 0,  -3, 1,  -1, 1,  -2, 1,  0,  -1,
    2,  2,  0,  1,  -1, 6,  -1, 1,  -2, 0,  0,  0,  -3, -1,  -2, 3,  3,  1,  0,  0,  -3, 2,  0,  3,  -1, -1,  0,  -1,  0,  2,  1,  2,  1,  3,  1,  2,  2,  2,
    2,  3,  0,  0,  3,  0,  -1, -1, -1, 4,  3,  1,  -1, 1,   1,  -1, 0,  -2, 0,  2,  1,  0,  0,  -1, 2,  1,   1,  1,   1,  2,  -1, 4,  4,  0,  0,  2,  2,  4,
    2,  4,  0,  -5, 0,  4,  -2, 2,  4,  3,  3,  1,  -1, 2,   2,  -1, 0,  2,  3,  -1, -2, -4, -4, -1, -8, -3,  -4, -2,  1,  1,  1,  3,  0,  -2, 0,  0,  0,  2,
    -1, -2, 1,  2,  -1, 0,  0,  0,  1,  4,  -2, -2, 2,  -1,  2,  2,  -1, 3,  0,  4,  3,  1,  -1, 2,  1,  -1,  -4, -3,  0,  1,  2,  2,  0,  3,  -3, 3,  1,  2,
    -3, -3, 1,  -1, 0,  3,  1,  2,  3,  4,  2,  0,  1,  3,   2,  2,  1,  2,  2,  -1, -2, 0,  -1, 1,  3,  0,   -1, -2,  1,  -1, 1,  0,  -2, -1, 0,  3,  0,  -2,
    3,  2,  -1, 2,  1,  1,  -3, 6,  3,  0,  0,  -1, 1,  3,   3,  2,  0,  -6, 0,  0,  -2, 1,  5,  5,  1,  2,   1,  0,   2,  0,  0,  2,  1,  1,  0,  -3, -5, -2,
    -5, -4, -2, -6, -6, 1,  1,  0,  2,  2,  0,  5,  0,  0,   2,  1,  0,  -1, -1, 1,  4,  0,  0,  2,  3,  2,   -3, 6,   -2, 0,  0,  2,  -3, -2, -3, -4, 0,  3,
    1,  0,  0,  -3, 3,  2,  0,  0,  0,  -1, -1, -3, -1, 0,   -1, 4,  -2, 0,  2,  0,  2,  -1, -2, -4, -2, -1,  0,  -1,  1,  0,  1,  -1, -1, 3,  3,  1,  -3, -1,
    1,  2,  3,  1,  0,  4,  4,  7,  2,  6,  -2, 0,  1,  -2,  3,  -1, 2,  1,  -1, 1,  0,  -2, 1,  1,  2,  -1,  0,  -1,  0,  -1, 1,  2,  2,  0,  2,  4,  2,  5,
    -1, 4,  5,  2,  1,  3,  2,  -1, 1,  0,  1,  2,  5,  -1,  -5, -3, -3, -3, -2, 0,  2,  2,  -2, 3,  -2, 4,   1,  1,   1,  2,  1,  0,  1,  1,  4,  -2, 0,  4,
    4,  1,  -2, 6,  1,  -2, -3, -3, -2, -3, -2, -1, 1,  1,   3,  2,  0,  -2, 2,  0,  -2, 0,  -3, 1,  -1, -2,  -2, 1,   -2, 4,  -4, 0,  -1, 0,  4,  -4, -3, -4,
    0,  -1, -1, 0,  1,  0,  1,  3,  0,  1,  2,  -2, -3, 0,   1,  2,  1,  -2, -2, 5,  1,  1,  3,  6,  -2, 1,   0,  -3,  2,  1,  3,  -2, -1, 0,  1,  -4, 2,  3,
    -1, -1, 0,  -2, -1, 2,  2,  -1, 3,  -2, 2,  4,  2,  5,   2,  7,  6,  2,  3,  3,  2,  -1, 0,  0,  -1, 1,   4,  1,   -4, -2, -3, -1, -5, 2,  2,  2,  -3, 2,
    0,  3,  0,  0,  0,  3,  2,  -2, 1,  2,  3,  -2, -1, 2,   7,  0,  -3, 5,  0,  -2, -4, 2,  2,  -1, -1, -2,  2,  2,   1,  1,  0,  3,  1,  -2, -4, -1, -1, -3,
    1,  -2, 0,  1,  1,  4,  -3, -2, 1,  0,  3,  0,  -4, -1,  0,  -3, -2, -1, 1,  2,  2,  3,  -1, 0,  3,  -4,  -1, 2,   1,  1,  1,  -1, -1, 3,  0,  1,  0,  2,
    -2, -1, -1, -5, 2,  -4, 3,  1,  1,  -1, 0,  -2, -1, 0,   -1, -2, -1, 0,  0,  6,  3,  2,  1,  -1, 6,  2,   5,  8,   3,  5,  4,  3,  2,  3,  5,  -1, 0,  1,
    0,  1,  0,  1,  -1, 0,  -1, -3, -6, 3,  3,  1,  0,  3,   0,  2,  -3, 0,  2,  2,  4,  0,  -1, 1,  0,  -2,  0,  0,   6,  3,  -3, 5,  -2, -1, 0,  0,  2,  0,
    -2, -1, 1,  3,  1,  -2, 1,  1,  3,  -1, -2, -1, -3, -1,  2,  -1, -2, -1, 0,  4,  -1, -1, 0,  0,  2,  0,   -2, -3,  -2, -3, -2, 1,  2,  0,  1,  3,  1,  -1,
    3,  -5, -1, 3,  -2, 1,  1,  -1, -2, 2,  1,  -2, 1,  6,   -3, -2, 0,  -4, 6,  -3, 2,  0,  1,  0,  0,  -3,  -1, -1,  -5, 0,  1,  -1, -2, 1,  -2, 1,  3,  -2,
    4,  2,  4,  8,  1,  5,  4,  3,  3,  5,  5,  2,  0,  3,   3,  4,  3,  1,  -2, 0,  -4, -3, -2, 3,  1,  1,   -3, 1,   1,  1,  -1, -2, -1, 1,  1,  0,  0,  -1,
    2,  -2, 2,  2,  8,  4,  -2, 4,  -3, -2, -1, 0,  2,  0,   -2, -2, 3,  3,  0,  -1, 1,  0,  1,  0,  -2, 0,   -2, -3,  3,  0,  -3, -3, -3, 2,  -1, 1,  1,  0,
    4,  -2, 2,  -1, 2,  0,  -3, 1,  1,  0,  0,  4,  1,  1,   2,  -2, 0,  1,  1,  3,  -1, -3, -2, 2,  1,  0,   1,  3,   0,  0,  3,  -5, 5,  -1, 1,  -3, 1,  -1,
    -4, -4, 3,  -2, -5, 1,  3,  0,  -1, 1,  0,  1,  3,  -4,  0,  3,  3,  3,  -2, 4,  3,  1,  2,  5,  4,  1,   -1, 1,   3,  3,  3,  3,  -1, 0,  -2, -5, 1,  0,
    -1, 2,  -1, 1,  4,  1,  1,  1,  0,  -1, 1,  0,  0,  3,   2,  -1, 4,  0,  1,  1,  3,  -1, 0,  1,  0,  2,   1,  2,   -1, 1,  1,  3,  3,  4,  1,  -1, 0,  3,
    2,  2,  0,  3,  5,  2,  0,  3,  2,  2,  0,  2,  3,  2,   1,  1,  2,  1,  5,  6,  2,  1,  2,  2,  1,  0,   1,  3,   2,  4,  2,  1,  0,  1,  0,  2,  3,  -2,
    -2, 2,  1,  1,  0,  -3, 1,  2,  0,  0,  -1, 1,  1,  0,   2,  1,  1,  4,  6,  4,  -2, 2,  0,  -1, 0,  1,   2,  2,   4,  4,  5,  5,  3,  2,  2,  3,  2,  -1,
    2,  4,  4,  4,  3,  3,  0,  1,  1,  -2, 5,  5,  1,  -2,  0,  2,  0,  0,  2,  1,  -1, 2,  2,  -1, -1, -5,  -2, 1,   -1, -1, 0,  -2, -2, -1, -1, 1,  -3, -4,
    -3, -1, -1, 0,  0,  -2, 1,  -2, -2, -2, -3, 0,  -1, -5,  -1, -3, 0,  -4, -4, -2, -4, -4, -2, -5, 1,  0,   -3, -1,  -4, -3, -1, -3, -3, -3, -2, -1, -1, -1,
    -1, 0,  -2, 0,  -3, -4, -3, 0,  -1, -3, 0,  -3, 1,  -1,  1,  0,  0,  1,  3,  0,  0,  2,  -1, -8, -4, 0,   0,  1,   -1, 0,  -3, -4, -2, 1,  0,  -2, 0,  -3,
    0,  -2, 0,  2,  0,  -2, -1, -3, -2, -3, -4, -2, -1, 3,   -1, 0,  -4, -6, -5, -4, -5, -3, -1, -5, 0,  -2,  0,  -1,  1,  1,  0,  1,  2,  2,  2,  2,  -1, 1,
    3,  3,  2,  4,  3,  -1, 1,  1,  3,  -1, 1,  1,  3,  2,   -1, 3,  2,  2,  -1, 0,  0,  -1, 2,  2,  3,  0,   1,  1,   2,  1,  4,  0,  3,  0,  2,  1,  1,  3,
    3,  0,  0,  0,  2,  1,  -1, 1,  2,  5,  1,  0,  2,  1,   1,  2,  -1, 3,  3,  2,  3,  1,  -1, -1, 0,  2,   -1, 1,   0,  1,  0,  0,  -1, -1, 2,  -2, 2,  2,
    -2, -1, 0,  1,  4,  -1, 3,  1,  2,  0,  1,  2,  0,  -1,  1,  3,  0,  2,  0,  -1, 2,  4,  4,  1,  -1, 3,   4,  2,   2,  2,  4,  4,  2,  2,  3,  1,  1,  -2,
    7,  4,  1,  -1, 2,  2,  1,  2,  0,  1,  -3, 3,  3,  -1,  0,  1,  1,  3,  3,  2,  -1, 1,  1,  -1, 4,  -2,  1,  2,   3,  0,  1,  0,  2,  -2, 1,  1,  2,  2,
    -3, 2,  0,  -2, 2,  1,  -1, 3,  0,  2,  -1, 3,  1,  1,   1,  2,  -1, 0,  1,  -1, 0,  1,  2,  3,  1,  0,   2,  1,   2,  1,  1,  2,  3,  2,  3,  0,  2,  0,
    0,  2,  4,  0,  -1, 1,  0,  2,  1,  2,  -2, -1, 2,  -2,  -3, 0,  -2, 0,  1,  0,  0,  1,  2,  1,  0,  3,   2,  -2,  0,  2,  2,  0,  1,  0,  2,  3,  1,  2,
    2,  4,  2,  3,  2,  3,  3,  1,  -1, -2, 3,  -2, -1, -3,  2,  3,  1,  0,  -2, 1,  1,  1,  1,  0,  1,  0,   1,  1,   0,  0,  1,  2,  2,  2,  1,  1,  1,  1,
    0,  1,  1,  1,  -1, 0,  0,  2,  1,  0,  1,  -2, 3,  -1,  0,  0,  -1, -3, 0,  2,  -1, -1, -2, 1,  1,  2,   0,  -3,  1,  1,  0,  -2, 0,  -1, 2,  2,  1,  3,
    1,  -2, 2,  -1, -2, 0,  2,  0,  0,  2,  0,  1,  -2, -3,  -2, 2,  -1, 2,  1,  0,  -1, 1,  2,  2,  0,  1,   -1, -1,  -1, 1,  1,  -1, 2,  2,  -1, 1,  1,  0,
    0,  1,  -1, 1,  1,  2,  -1, 1,  0,  1,  0,  0,  -3, 0,   -2, 1,  0,  3,  0,  0,  0,  0,  -2, -1, -1, 1,   -2, 2,   2,  1,  -1, 0,  2,  3,  -1, 1,  -2, 4,
    -3, -2, 2,  2,  0,  -2, 1,  -3, 0,  8,  2,  1,  -1, 2,   -5, -1, -1, -2, 0,  3,  -3, -4, 1,  1,  0,  0,   1,  2,   0,  2,  -1, 0,  -3, 1,  1,  1,  2,  0,
    5,  4,  0,  2,  1,  0,  1,  -2, 2,  -1, 2,  2,  0,  -3,  -2, 3,  4,  1,  0,  2,  1,  0,  -2, -5, -3, 1,   5,  0,   -3, 0,  -2, 2,  -1, 2,  3,  0,  2,  1,
    1,  1,  1,  1,  0,  0,  -1, -1, 0,  2,  -1, -2, -1, 2,   2,  -1, 2,  2,  -2, -3, 1,  -2, 4,  6,  3,  6,   9,  5,   4,  4,  3,  4,  0,  0,  -7, 2,  1,  -8,
    3,  3,  -1, 1,  -2, 0,  -1, 1,  1,  0,  1,  1,  0,  3,   -2, -1, -1, 0,  2,  2,  0,  -3, -1, 4,  2,  1,   1,  2,   -1, 0,  -1, -1, -1, 0,  1,  0,  3,  3,
    2,  -1, 3,  1,  0,  0,  -1, -1, -4, 3,  2,  3,  2,  1,   2,  3,  -2, 0,  2,  2,  0,  -2, 0,  -3, 1,  1,   -1, -2,  -2, 3,  3,  1,  -2, 2,  1,  -1, -1, -4,
    0,  2,  5,  1,  -3, 1,  -3, 3,  0,  0,  2,  1,  -1, 2,   0,  1,  1,  2,  0,  0,  0,  -2, 0,  0,  0,  0,   2,  3,   4,  -1, 1,  0,  -2, 0,  -1, -1, 0,  5,
    4,  3,  6,  6,  3,  2,  1,  3,  2,  -2, -7, 3,  0,  -11, 1,  2,  -1, -1, -5, -1, -1, 1,  2,  -1, -3, 3,   0,  4,   1,  1,  1,  1,  2,  0,  -3, 1,  3,  5,
    4,  4,  1,  -2, 1,  3,  -1, 0,  0,  3,  1,  -2, 3,  1,   2,  -2, 3,  1,  0,  0,  -2, -1, -1, 2,  3,  1,   4,  2,   2,  3,  -2, -1, 1,  3,  -1, -2, 0,  -2,
    2,  0,  -1, -1, -2, 2,  3,  1,  -2, 1,  0,  -1, -2, -2,  1,  0,  3,  1,  -2, -1, -2, 1,  -3, 1,  0,  3,   0,  2,   -1, 1,  1,  2,  3,  0,  -1, 0,  -2, 2,
    -2, -3, -1, 2,  5,  0,  3,  1,  -3, 2,  0,  -2, 2,  6,   4,  4,  6,  7,  4,  2,  3,  2,  3,  -3, -8, 4,   -2, -11, 4,  4,  -1, 0,  -3, -4, 0,  4,  2,  -2,
    -2, 1,  0,  2,  3,  1,  0,  1,  1,  0,  -2, 1,  1,  3,   1,  3,  2,  -4, -2, 1,  2,  1,  -1, 4,  -2, -3,  2,  2,   1,  -2, 0,  -1, 2,  -1, -2, -2, -2, 1,
    1,  0,  2,  2,  2,  4,  -1, -3, 2,  2,  -1, -5, 1,  -1,  3,  0,  0,  -2, -3, 2,  4,  0,  0,  1,  -1, -2,  -2, -3,  2,  0,  3,  4,  -3, 0,  -1, 2,  -2, 1,
    -1, 1,  1,  3,  -2, 2,  2,  1,  2,  -2, 1,  0,  1,  1,   -2, 0,  0,  2,  3,  -2, 5,  3,  -2, 1,  -3, 0,   4,  4,   3,  4,  6,  5,  3,  2,  2,  1,  2,  -3,
    -5, 5,  -4, -8, 2,  4,  -1, 0,  -3, -3, 2,  2,  3,  -3,  1,  1,  2,  2,  0,  0,  -2, -1, 2,  1,  -1, 0,   3,  1,   1,  4,  3,  -3, -4, 2,  3,  -1, 0,  1,
    1,  -2, 0,  2,  -1, -3, 2,  1,  2,  2,  1,  0,  0,  -1,  0,  -1, 1,  1,  1,  3,  -1, -1, 2,  3,  0,  -3,  0,  -1,  3,  0,  -2, 1,  0,  3,  3,  0,  2,  -1,
    3,  1,  -5, -3, 1,  -1, 3,  3,  2,  -3, -1, -2, -1, 0,   -2, 0,  1,  4,  -2, -2, 0,  1,  1,  -4, 0,  2,   0,  0,   -1, 1,  1,  0,  3,  -1, 4,  3,  2,  -2,
    -2, -3, 4,  6,  3,  3,  3,  6,  6,  4,  3,  3,  2,  -3,  -6, 4,  -3, -6, 1,  3,  -1, -1, -2, -3, 0,  2,   2,  -1,  0,  5,  -2, 1,  -4, -1, -1, 0,  1,  -1,
    0,  4,  -3, 0,  1,  2,  -1, 0,  -3, 8,  4,  5,  0,  0,   -2, -2, 0,  -2, 0,  -1, 1,  1,  5,  1,  2,  2,   -1, -2,  0,  1,  0,  1,  3,  0,  -1, 3,  4,  1,
    0,  0,  -2, 1,  1,  -2, -3, 2,  -2, 1,  -1, -2, -1, 3,   0,  0,  1,  -2, 2,  2,  1,  1,  3,  4,  2,  -1,  -3, 2,   1,  1,  3,  2,  4,  7,  0,  5,  3,  -1,
    1,  3,  1,  0,  -3, -1, 2,  6,  4,  1,  3,  2,  3,  1,   -2, -1, -1, -1, -2, -1, 4,  5,  3,  -2, 0,  4,   4,  6,   6,  -3, 1,  1,  1,  -2, 0,  -3, -4, -2,
    1,  2,  1,  -2, -2, 1,  -1, 1,  -3, -2, 0,  0,  1,  -7,  5,  8,  -4, -4, 3,  4,  -1, 0,  0,  4,  3,  5,   1,  1,   -2, -3, -3, -2, 2,  0,  -1, -1, 6,  2,
    3,  3,  1,  -1, 1,  2,  2,  1,  1,  0,  -3, -1, 1,  0,   0,  3,  -4, 0,  3,  -1, -2, -1, -1, 0,  -1, -2,  -1, 1,   1,  1,  -1, 0,  3,  1,  -1, 3,  1,  5,
    1,  -2, -2, 2,  1,  3,  2,  1,  4,  10, 2,  3,  3,  -2,  2,  3,  2,  -2, -1, -3, -3, 4,  0,  -2, 2,  -1,  3,  -2,  -2, -2, -2, -1, -1, 2,  5,  4,  0,  -3,
    2,  5,  6,  6,  6,  -1, 4,  0,  1,  -3, 0,  -2, -5, 1,   0,  2,  1,  2,  4,  2,  -2, 1,  -1, -1, 1,  -1,  2,  -5,  2,  9,  0,  -2, 3,  6,  0,  0,  1,  3,
    4,  3,  -1, -2, -4, -3, 0,  -2, -1, -2, 0,  1,  6,  0,   2,  2,  -3, 2,  2,  0,  3,  0,  0,  3,  0,  -2,  0,  -1,  -2, 0,  -1, -2, -1, -3, -2, 0,  -2, 0,
    -1, -3, -2, 3,  -1, 1,  1,  0,  4,  2,  -3, 2,  2,  0,   1,  0,  -2, 2,  1,  -2, 3,  2,  3,  9,  0,  -1,  1,  -4,  1,  2,  1,  -4, 0,  -2, -3, 1,  2,  0,
    2,  -1, 1,  1,  -3, -1, -4, -3, 1,  0,  6,  5,  1,  -5,  1,  4,  5,  5,  6,  0,  1,  1,  2,  -4, 2,  -2,  -7, -2,  1,  3,  3,  3,  4,  0,  -2, 2,  1,  -1,
    1,  -1, 2,  -3, 3,  7,  -1, -1, 1,  6,  1,  1,  3,  1,   4,  1,  -2, -2, -1, -4, 0,  -2, -5, -3, -1, 4,   7,  0,   2,  1,  -2, -1, -2, 1,  2,  2,  0,  2,
    1,  1,  -3, 0,  -2, -1, -4, 0,  0,  -2, -2, 1,  -1, 1,   -3, -4, 0,  4,  -1, -2, 0,  -2, 4,  1,  -1, 2,   1,  2,   -1, 0,  0,  -2, 2,  1,  2,  2,  1,  4,
    1,  0,  0,  -1, -1, 2,  1,  -2, -3, -3, -3, 2,  2,  -1,  -1, -1, -3, -1, -2, 0,  -5, -4, -2, -1, 6,  5,   1,  -3,  -1, 5,  4,  6,  5,  0,  1,  2,  -2, -3,
    0,  -4, -3, 0,  3,  0,  3,  3,  4,  2,  0,  -1, 1,  -2,  -1, -1, 1,  -4, 5,  0,  2,  -3, 1,  6,  1,  2,   1,  -1,  0,  1,  0,  0,  2,  0,  -3, -1, -2, -2,
    2,  4,  7,  -1, 6,  -1, -4, -2, -1, 1,  3,  0,  0,  2,   -2, 3,  -2, 0,  0,  -1, -5, 1,  -1, 1,  0,  1,   1,  2,   -2, -3, 1,  5,  0,  -3, -1, 0,  2,  1,
    2,  3,  1,  3,  0,  1,  -1, -2, 2,  1,  1,  4,  0,  5,   1,  2,  0,  -1, -4, 0,  -1, -4, -3, -2, -3, 1,   2,  1,   1,  -1, -3, 0,  -2, 0,  -2, 0,  -2, -1,
    5,  4,  -2, -3, 2,  2,  7,  5,  5,  1,  2,  4,  1,  0,   0,  -3, -3, 3,  0,  1,  0,  0,  -2, 0,  -2, 0,   2,  -3,  2,  2,  0,  0,  0,  1,  1,  1,  3,  0,
    2,  3,  2,  -1, 1,  3,  3,  0,  1,  2,  -1, -1, -1, 0,   4,  1,  3,  1,  3,  1,  -1, 1,  0,  1,  4,  4,   -2, 3,   0,  1,  1,  1,  2,  2,  1,  5,  3,  4,
    -1, 2,  1,  0,  -1, 2,  3,  2,  4,  -1, 5,  1,  3,  -1,  -3, 3,  0,  -3, 3,  2,  1,  2,  1,  0,  -1, 0,   1,  3,   1,  -1, 0,  1,  5,  2,  6,  4,  4,  3,
    5,  2,  0,  1,  5,  3,  -2, -2, -4, -2, 1,  -4, 1,  0,   -1, 1,  3,  0,  0,  0,  -1, 2,  0,  -2, -1, 1,   5,  4,   -2, -4, -2, 4,  3,  2,  -1, 2,  2,  1,
    0,  -2, 0,  2,  4,  0,  2,  -1, 1,  -3, 1,  0,  4,  1,   5,  2,  1,  0,  2,  3,  4,  0,  -1, 2,  2,  3,   -1, 0,   1,  1,  3,  -1, 4,  3,  1,  1,  0,  2,
    3,  2,  -2, 0,  -1, 0,  1,  1,  1,  2,  1,  3,  2,  2,   1,  0,  2,  0,  -1, 0,  2,  0,  0,  0,  5,  3,   2,  0,   0,  -1, -2, -1, 3,  2,  1,  1,  1,  3,
    -2, -1, 0,  1,  0,  -2, 2,  3,  1,  0,  2,  0,  1,  5,   4,  -1, -3, -1, 4,  0,  -1, -4, -4, -1, 2,  -4,  -1, -1,  3,  1,  1,  3,  -1, -1, 0,  5,  -1, -1,
    2,  -1, 5,  6,  -1, -2, -2, 6,  1,  1,  2,  0,  0,  1,   0,  0,  2,  0,  0,  -1, -1, -2, -3, -1, 2,  0,   2,  -1,  1,  -2, -2, -1, -1, 3,  1,  2,  2,  -1,
    -1, 1,  0,  -3, 1,  2,  2,  -3, 3,  1,  0,  0,  0,  2,   3,  1,  -3, -2, 0,  -3, -1, -1, 0,  0,  -2, 0,   2,  0,   -1, -1, 0,  -1, 0,  0,  1,  1,  1,  0,
    1,  2,  -1, 2,  0,  -1, -1, -1, 2,  1,  0,  -1, -2, -1,  -1, -2, -2, -2, -1, -3, 0,  2,  1,  1,  2,  3,   0,  2,   0,  -3, -2, 0,  3,  1,  4,  -1, -4, 0,
    2,  -7, -3, 1,  0,  0,  2,  2,  -2, -1, 1,  5,  2,  -2,  -2, 3,  4,  6,  1,  -1, 1,  5,  1,  4,  0,  3,   2,  2,   3,  -1, 0,  -1, 0,  0,  -1, -3, 0,  -1,
    1,  1,  -3, -4, 3,  -6, -1, 1,  -3, 2,  0,  -1, 1,  1,   -1, -3, -3, -2, -1, 2,  0,  -2, 1,  1,  -2, 0,   2,  -2,  1,  0,  -1, -1, 2,  -2, -1, 0,  -3, -1,
    0,  -1, -1, -3, -3, -1, 0,  -3, -1, -1, 4,  1,  -2, -3,  1,  -2, 1,  -1, 0,  -1, 0,  -3, 1,  1,  0,  -2,  -3, -5,  -4, -3, -5, -2, -1, -5, 0,  -1, -2, 1,
    2,  0,  -1, 0,  1,  -7, -2, -3, 2,  4,  2,  -2, -5, -1,  -1, -5, -2, -1, -2, -2, 0,  1,  -1, -3, 2,  4,   1,  -1,  -1, 0,  0,  6,  0,  -2, -1, 1,  -2, 2,
    1,  1,  1,  1,  1,  0,  3,  -1, 0,  -1, 2,  0,  2,  2,   1,  1,  -1, -2, 6,  -3, 3,  1,  0,  5,  6,  0,   3,  1,   -2, 1,  -4, 0,  2,  2,  3,  -2, 4,  2,
    -1, 3,  2,  1,  4,  1,  1,  0,  1,  1,  0,  -1, 2,  -1,  1,  -1, 3,  -1, 2,  6,  1,  1,  0,  1,  2,  1,   2,  0,   2,  3,  3,  3,  -1, -3, 1,  -1, -1, 3,
    0,  -1, -1, -4, -4, -4, -2, -2, -2, -4, 0,  -1, 0,  2,   1,  1,  2,  4,  1,  -3, -2, 0,  1,  3,  2,  0,   -3, -1,  0,  -4, -1, 0,  2,  1,  2,  4,  1,  -2,
    2,  5,  4,  3,  3,  4,  4,  6,  0,  -1, 6,  2,  4,  1,   0,  3,  3,  3,  0,  0,  3,  4,  -1, 3,  0,  0,   -3, 1,   3,  1,  1,  4,  3,  2,  5,  0,  -3, 2,
    1,  0,  -2, 3,  0,  2,  3,  0,  2,  0,  1,  -1, 1,  1,   0,  1,  1,  1,  -1, -6, -4, 0,  1,  1,  1,  0,   -1, 1,   2,  0,  2,  3,  1,  -2, -1, -3, 1,  -1,
    0,  2,  -1, 0,  1,  4,  1,  3,  0,  3,  2,  1,  -2, -1,  2,  0,  1,  0,  1,  4,  2,  4,  4,  1,  -1, 1,   -2, -2,  2,  -2, -1, 1,  3,  3,  -4, 2,  2,  3,
    -1, 1,  -1, -1, 3,  1,  0,  3,  0,  -3, 2,  1,  4,  2,   -1, 3,  -3, 0,  -2, -5, 1,  4,  1,  0,  -2, -1,  5,  -1,  -2, 2,  -2, 2,  0,  -2, 4,  -4, 2,  0,
    2,  3,  -2, 2,  4,  3,  3,  3,  2,  3,  4,  0,  -1, 3,   2,  -1, -2, 0,  2,  -1, 2,  -3, 2,  -1, 2,  -3,  0,  1,   2,  1,  2,  2,  -1, -4, -3, -1, 1,  2,
    -1, -1, 0,  2,  2,  0,  3,  -1, 1,  -2, -1, -3, 0,  -2,  -2, 1,  0,  0,  0,  3,  0,  3,  2,  2,  -1, 0,   -3, -1,  4,  0,  3,  -1, 0,  1,  1,  1,  4,  0,
    -2, -1, 0,  -2, 4,  -4, 0,  1,  1,  4,  0,  1,  0,  2,   1,  0,  -1, -3, 3,  1,  1,  5,  1,  0,  3,  0,   2,  3,   -1, 2,  -2, -1, -3, -3, 1,  3,  2,  1,
    0,  -5, 3,  -3, -1, 3,  -2, 2,  1,  -3, 6,  -5, 2,  -1,  1,  0,  0,  3,  4,  0,  2,  2,  4,  3,  3,  1,   -2, 4,   0,  2,  -1, 0,  3,  -1, 2,  -2, 2,  -4,
    0,  -2, -2, -1, 0,  1,  0,  2,  -2, -3, -1, -1, 1,  3,   -1, -2, 0,  2,  2,  0,  1,  -1, 0,  -1, 1,  -3,  -1, -1,  -2, 2,  -1, -1, 1,  5,  2,  2,  1,  0,
    0,  0,  -3, -1, 4,  -1, 4,  -4, 1,  3,  5,  2,  3,  0,   -3, -1, 0,  -2, 2,  -2, 3,  3,  1,  2,  -3, 1,   0,  2,   2,  0,  1,  -1, 1,  2,  1,  4,  1,  1,
    2,  2,  4,  1,  -3, 3,  -3, 1,  -3, -1, 2,  2,  4,  3,   -3, -5, 4,  -1, -2, 3,  -1, 1,  4,  -3, 2,  -5,  2,  1,   1,  2,  -2, 2,  1,  1,  3,  3,  3,  1,
    6,  5,  -2, 3,  1,  -1, -4, 0,  1,  0,  3,  0,  2,  -1,  0,  -1, -2, 0,  2,  1,  2,  0,  0,  -2, -1, -4,  1,  2,   -3, 1,  0,  1,  3,  -2, -1, -1, 0,  -1,
    -1, -3, -2, 0,  -1, 2,  -1, -1, 1,  5,  2,  2,  1,  1,   1,  1,  -3, 0,  0,  -3, 0,  -4, 1,  3,  5,  2,   5,  1,   0,  -2, 1,  -4, 3,  0,  3,  2,  2,  -1,
    -3, 2,  3,  1,  4,  1,  -1, 1,  2,  2,  3,  4,  1,  3,   3,  1,  0,  2,  -3, 3,  -1, 0,  -3, -3, 3,  1,   3,  3,   0,  -5, 4,  -3, -2, 2,  -1, -2, 4,  -1,
    0,  -5, 2,  1,  4,  1,  -1, 1,  1,  3,  2,  2,  2,  3,   4,  3,  0,  5,  -1, -3, -2, 0,  0,  0,  3,  -2,  1,  -2,  -1, -2, -1, 0,  1,  4,  0,  0,  2,  -2,
    3,  -3, -1, 4,  0,  -2, 1,  2,  3,  -2, 0,  1,  0,  0,   -1, -2, -1, -2, 0,  3,  2,  2,  2,  3,  2,  1,   4,  0,   -2, 1,  -1, 3,  -1, -3, 0,  -5, 0,  6,
    5,  1,  4,  -2, 0,  1,  0,  -3, 1,  0,  3,  2,  4,  -1,  -3, 1,  1,  0,  5,  -2, 0,  0,  1,  2,  4,  3,   1,  3,   3,  1,  0,  2,  -3, 1,  -3, 0,  0,  -2,
    3,  1,  2,  1,  0,  -5, 2,  2,  1,  -1, 2,  -1, 1,  2,   3,  0,  -2, 3,  4,  -2, 0,  2,  -1, -1, 2,  1,   -4, 4,   2,  1,  -1, 1,  4,  2,  -1, 0,  1,  -2,
    1,  1,  0,  2,  0,  0,  -2, -3, -1, 0,  -2, 0,  0,  -2,  2,  3,  0,  -1, 0,  3,  1,  -1, -2, -1, -1, 4,   0,  1,   2,  1,  -1, -1, 0,  2,  0,  3,  2,  -2,
    -3, -2, 0,  -1, -1, -1, -2, -2, -5, -4, 0,  2,  0,  -2,  3,  5,  3,  4,  1,  6,  0,  1,  1,  0,  5,  -1,  2,  4,   4,  4,  0,  0,  -2, -1, -1, -4, -1, -3,
    -4, 1,  4,  0,  -1, -3, -1, -1, 2,  -2, 3,  1,  -2, 2,   -1, -3, -2, -1, 2,  3,  4,  3,  2,  -4, -1, 1,   -1, 2,   3,  -2, -1, 2,  2,  1,  0,  1,  -2, 1,
    1,  3,  -1, 2,  0,  2,  1,  2,  1,  0,  1,  -1, -2, 0,   2,  2,  1,  2,  2,  1,  -4, -3, 0,  0,  -1, 0,   -1, -3,  1,  0,  0,  -1, 2,  1,  3,  0,  -1, 1,
    0,  3,  2,  4,  3,  4,  0,  1,  1,  1,  1,  3,  0,  -1,  0,  -3, -1, 2,  2,  -2, -2, -3, -2, -3, -1, 1,   -1, -3,  2,  5,  5,  3,  2,  2,  -1, 2,  3,  -1,
    3,  1,  6,  4,  3,  4,  0,  -1, -3, 0,  -1, -2, -1, -3,  -1, 2,  4,  1,  -1, 1,  0,  -1, 2,  2,  4,  4,   3,  1,   0,  -1, 0,  -1, 3,  1,  5,  4,  1,  0,
    0,  0,  -4, 3,  4,  -2, 0,  3,  0,  1,  3,  1,  2,  2,   3,  2,  2,  3,  2,  1,  -2, 2,  3,  2,  1,  -1,  -2, 2,   1,  2,  1,  2,  2,  -2, -2, -1, -1, 1,
    -1, 0,  1,  0,  0,  3,  0,  -4, 3,  -1, 5,  0,  1,  1,   -1, 5,  1,  3,  3,  5,  1,  1,  -1, 2,  -2, 4,   0,  -2,  -1, 0,  0,  -1, -2, 2,  -2, -4, -3, -6,
    -5, -1, -3, -2, 4,  4,  5,  1,  0,  1,  0,  1,  4,  3,   2,  -1, 5,  4,  2,  4,  -1, 0,  -3, 2,  0,  -1,  -1, -1,  0,  0,  4,  2,  1,  -1, -1, -2, 2,  1,
    3,  1,  3,  2,  0,  -4, 0,  -1, 2,  0,  3,  5,  0,  -1,  -3, 2,  -3, 2,  4,  3,  -1, 1,  1,  2,  1,  1,   0,  -1,  3,  1,  -1, -2, 1,  4,  1,  0,  5,  3,
    -2, 0,  -1, 1,  0,  -1, -1, 2,  2,  0,  -5, -3, -2, 0,   0,  -1, 1,  -3, 2,  0,  3,  -1, 0,  2,  5,  0,   1,  0,   0,  3,  1,  2,  1,  2,  1,  -1, 1,  1,
    1,  3,  -1, 2,  -1, -3, -4, -3, 0,  -2, -4, -3, -1, -8,  -5, -3, -3, -3, 7,  3,  4,  1,  1,  3,  -2, -1,  2,  3,   1,  -2, 3,  4,  2,  3,  1,  0,  -3, -1,
    0,  1,  0,  -1, 2,  0,  2,  1,  2,  -1, -1, -4, 0,  -1,  2,  2,  1,  1,  0,  -4, -2, 0,  2,  2,  4,  4,   0,  -2,  0,  1,  1,  1,  4,  3,  -1, 1,  0,  2,
    2,  3,  1,  5,  1,  0,  2,  0,  2,  1,  0,  -1, 3,  2,   2,  -3, 0,  0,  -2, -2, 0,  0,  0,  -2, -1, -1,  1,  0,   -2, 2,  1,  0,  2,  0,  3,  3,  1,  1,
    4,  2,  4,  1,  -1, 1,  1,  7,  3,  3,  2,  0,  1,  -1,  -2, 5,  2,  6,  -1, 1,  -1, 1,  1,  -3, -5, -1,  -4, -8,  -7, -3, -4, -4, 4,  2,  5,  0,  -2, 1,
    -1, -2, 1,  3,  1,  1,  5,  2,  2,  4,  0,  -1, -2, -2,  -1, 3,  1,  0,  -2, 1,  2,  3,  1,  0,  -1, -1,  4,  0,   3,  2,  1,  1,  4,  -4, 0,  1,  7,  4,
    2,  1,  6,  0,  0,  -2, 2,  0,  -3, 4,  1,  -2, -2, 0,   -1, -4, -1, 2,  2,  2,  0,  0,  3,  -4, -2, 2,   3,  0,   0,  0,  1,  2,  -1, 0,  0,  -2, 1,  1,
    1,  4,  0,  0,  -1, 3,  -2, 1,  2,  1,  1,  -1, -1, 3,   0,  0,  -2, -2, 3,  2,  -1, -2, 2,  0,  0,  2,   1,  -3,  2,  2,  0,  -1, -1, 1,  1,  0,  0,  -1,
    3,  2,  2,  2,  -2, -1, 0,  1,  2,  0,  -3, 3,  1,  0,   -1, 1,  1,  -3, 2,  4,  5,  2,  2,  -1, 3,  5,   3,  1,   -2, 3,  1,  -1, -1, -1, 2,  4,  1,  -1,
    0,  5,  5,  4,  4,  0,  3,  2,  1,  3,  6,  0,  4,  -1,  0,  2,  4,  1,  0,  -1, 0,  2,  -3, 4,  2,  -3,  -2, -1,  -1, -8, 1,  3,  1,  0,  1,  0,  4,  -4,
    -2, 1,  1,  1,  1,  1,  3,  0,  0,  2,  -1, -2, -1, 3,   1,  4,  2,  2,  3,  1,  -2, 1,  2,  1,  2,  -2,  1,  5,   -1, 1,  0,  1,  1,  2,  -3, -3, 1,  1,
    1,  -1, -1, -3, 1,  3,  1,  0,  0,  -2, -2, 3,  1,  0,   0,  3,  4,  2,  -4, 1,  1,  1,  -2, 2,  -4, -1,  1,  -2,  -1, 0,  1,  -1, 1,  3,  3,  2,  2,  2,
    2,  3,  1,  -1, -2, 0,  -1, -2, 0,  -1, 2,  -1, -1, -1,  -1, 6,  4,  5,  4,  1,  3,  0,  -2, 0,  3,  3,   4,  0,   1,  2,  3,  -1, 2,  -2, 0,  2,  -4, 4,
    2,  -4, 1,  -2, -2, -6, 3,  1,  2,  0,  2,  1,  2,  -1,  -2, 2,  3,  2,  1,  0,  1,  2,  0,  1,  0,  -1,  -2, -1,  1,  6,  1,  2,  4,  3,  -4, 3,  4,  -1,
    2,  1,  -3, 2,  1,  -1, 2,  0,  1,  0,  -2, -4, 2,  0,   0,  1,  -3, -2, 2,  3,  2,  -1, 0,  0,  -1, -1,  1,  0,   -1, 1,  4,  1,  -2, 0,  -2, 1,  1,  0,
    -1, 1,  1,  2,  0,  0,  3,  0,  1,  3,  -1, 0,  5,  1,   3,  3,  -2, -2, -3, 0,  0,  -1, -1, 0,  2,  -2,  0,  0,   0,  4,  3,  3,  2,  0,  3,  2,  -2, 1,
    3,  1,  2,  -2, 1,  2,  2,  0,  -1, -3, 1,  0,  -3, 4,   1,  -2, 0,  1,  -3, -6, 1,  3,  -1, 0,  0,  2,   1,  -1,  -2, 1,  1,  3,  1,  -1, -1, 0,  1,  1,
    -1, -1, 0,  -2, 1,  4,  1,  2,  5,  2,  -4, 1,  1,  2,   3,  1,  -1, 3,  2,  0,  3,  -2, 1,  2,  -1, -5,  2,  1,   0,  1,  -3, 1,  2,  0,  2,  0,  -1, -1,
    -2, 2,  -1, -2, 2,  3,  1,  2,  -2, -1, 0,  1,  3,  0,   0,  -1, 1,  1,  -1, 1,  3,  1,  3,  3,  -1, 3,   3,  1,   2,  2,  -1, -1, -4, -1, 0,  -1, -2, 0,
    0,  -2, -2, -3, 0,  3,  3,  1,  2,  -1, -1, 0,  -4, 2,   1,  2,  2,  -2, 1,  1,  2,  0,  -4, -4, -1, 0,   -5, 3,   1,  -2, 2,  -2, -1, -4, 3,  2,  0,  -2,
    -2, 0,  1,  -2, -3, 0,  0,  1,  5,  1,  0,  0,  0,  1,   0,  0,  -1, 0,  1,  6,  4,  3,  2,  -1, -2, 4,   1,  1,   3,  0,  -3, 3,  3,  1,  1,  0,  0,  2,
    0,  -3, 3,  0,  0,  0,  -2, -3, 0,  2,  -1, -1, 0,  -1,  -1, 2,  2,  1,  2,  1,  1,  2,  -4, 0,  -1, 2,   3,  -1,  -3, -1, 1,  -2, 0,  3,  2,  0,  1,  -1,
    -1, -1, 1,  0,  1,  2,  -3, -2, 0,  0,  2,  -3, 0,  0,   -1, -1, 1,  -3, -1, 2,  3,  0,  1,  -1, -1, 2,   -5, 0,   2,  -2, -3, -2, 0,  4,  0,  0,  5,  -1,
    -2, 0,  -3, -2, 6,  -5, 1,  -1, 3,  2,  2,  2,  -2, 1,   -1, -1, 0,  1,  -1, 2,  -3, 0,  0,  -2, -2, -1,  0,  5,   1,  -1, 0,  2,  4,  2,  2,  -2, 6,  -2,
    -3, 3,  1,  0,  0,  1,  0,  1,  1,  4,  3,  0,  -1, 0,   -3, -1, 1,  2,  3,  1,  3,  0,  4,  1,  -1, -1,  -1, 4,   0,  3,  2,  -2, 0,  0,  -1, -7, 1,  8,
    4,  5,  -2, 2,  -1, 0,  -2, 5,  -2, -2, -1, 0,  -1, -1,  3,  2,  2,  0,  -3, -1, -3, 3,  5,  -2, 1,  4,   -2, -3,  0,  -3, 2,  -3, 1,  4,  -1, -2, 2,  -1,
    4,  0,  -4, 4,  -2, 4,  3,  1,  0,  3,  -1, -2, 5,  2,   -1, 0,  -3, -4, 7,  -4, -2, 3,  3,  1,  4,  4,   -2, 3,   1,  0,  -1, -1, 3,  1,  -4, -2, 1,  -2,
    -6, -3, -2, 4,  2,  -3, 0,  0,  5,  1,  2,  1,  9,  0,   -3, 3,  2,  -1, 1,  7,  0,  2,  0,  3,  2,  1,   -1, 1,   -1, -1, 0,  0,  3,  -2, 1,  1,  3,  0,
    -1, 0,  0,  2,  -1, 4,  0,  -2, -2, -1, -2, -7, 1,  8,   2,  3,  0,  3,  0,  2,  2,  4,  0,  0,  -1, 3,   0,  -2,  3,  3,  4,  -2, -4, -1, 0,  -1, 3,  -3,
    -2, 3,  1,  -5, 1,  0,  4,  -3, 4,  5,  -1, -1, 0,  -1,  3,  2,  0,  4,  -2, 2,  4,  3,  -3, 4,  -1, -1,  3,  -1,  -1, 0,  -2, -1, 3,  -3, -1, 0,  7,  1,
    3,  5,  -1, 1,  4,  2,  0,  -2, 0,  1,  1,  -2, 0,  -1,  -4, -4, -1, 0,  0,  -3, 3,  2,  3,  0,  3,  2,   6,  2,   -2, 3,  3,  -2, 1,  7,  1,  4,  -1, 4,
    2,  0,  -2, 2,  -1, -2, -1, 0,  2,  -2, 2,  1,  2,  -1,  -2, -2, 3,  1,  -1, 0,  2,  -6, 0,  -2, 0,  -5,  -1, 9,   4,  4,  0,  3,  -1, 1,  -2, 2,  -2, 0,
    1,  -1, 4,  -2, -2, 7,  3,  -6, -1, 1,  -1, -1, 1,  0,   0,  1,  -2, -2, 0,  2,  5,  1,  0,  3,  0,  -1,  -1, 0,   5,  3,  -1, 3,  0,  0,  2,  3,  -1, 2,
    -3, 0,  3,  -1, 1,  -1, -4, 1,  1,  -4, 0,  0,  4,  3,   1,  4,  -1, 1,  2,  3,  1,  1,  -1, 1,  -2, -2,  1,  -1,  -3, -2, 0,  3,  1,  -3, 2,  1,  2,  2,
    4,  -1, 6,  5,  0,  5,  2,  -1, 4,  6,  2,  3,  2,  3,   3,  1,  -2, 0,  -1, -1, 2,  0,  1,  1,  3,  -1,  4,  1,   0,  1,  2,  -3, -4, 2,  0,  -8, 1,  1,
    0,  -5, -1, 10, 9,  3,  -2, 2,  2,  0,  -2, 0,  -1, -1,  0,  2,  2,  2,  -2, 3,  0,  -4, -3, 1,  2,  -1,  2,  -3,  1,  3,  2,  -1, 0,  3,  4,  2,  -1, 3,
    1,  -6, 0,  -3, 6,  5,  0,  6,  -1, 0,  3,  1,  -2, 1,   -3, 0,  -1, -4, 1,  0,  -1, 1,  2,  -2, -3, 1,   4,  1,   2,  3,  -1, 2,  1,  0,  0,  0,  -2, 3,
    -2, -2, 0,  -3, -3, -4, 0,  1,  2,  -3, 0,  3,  2,  2,   1,  -4, 8,  2,  1,  2,  2,  3,  2,  6,  2,  5,   0,  2,   0,  0,  1,  0,  1,  -4, 1,  1,  1,  -1,
    3,  3,  1,  2,  -2, 0,  0,  -2, 1,  0,  2,  -3, 0,  2,   0,  -4, 0,  8,  4,  3,  2,  4,  5,  1,  -1, 1,   2,  1,   2,  0,  -1, 0,  2,  3,  3,  -1, -3, 1,
    1,  0,  2,  1,  2,  2,  3,  -2, 4,  1,  2,  5,  -3, 3,   1,  -4, 0,  -1, 7,  0,  1,  3,  0,  5,  3,  -3,  2,  1,   -1, 0,  2,  4,  5,  1,  1,  0,  1,  1,
    -1, -1, -1, -3, 1,  2,  4,  -2, 3,  0,  0,  1,  2,  4,   5,  4,  1,  -2, 3,  4,  2,  1,  3,  3,  -1, 0,   3,  -1,  0,  2,  -3, -1, 2,  1,  1,  1,  3,  2,
    -1, 0,  0,  2,  1,  2,  1,  -3, 1,  4,  3,  2,  1,  0,   3,  0,  -3, 0,  1,  2,  1,  2,  0,  0,  -1, 4,   1,  2,   -2, -2, 1,  -3, 0,  1,  -1, 2,  2,  2,
    0,  -2, 2,  -3, -5, -1, 1,  3,  -1, 0,  0,  0,  -3, -2,  3,  0,  3,  -1, -3, -2, 0,  -2, 0,  1,  3,  4,   1,  2,   0,  3,  2,  1,  1,  1,  -2, 1,  0,  1,
    1,  6,  -2, 2,  0,  2,  2,  0,  1,  -2, 1,  2,  0,  3,   -3, -2, -1, 0,  0,  2,  1,  1,  -1, -1, 0,  3,   1,  4,   5,  5,  1,  0,  3,  1,  3,  0,  -1, 3,
    0,  0,  0,  -1, 2,  2,  -5, -2, 1,  1,  1,  0,  1,  1,   -3, 2,  1,  3,  3,  5,  2,  -2, 0,  3,  3,  1,   2,  3,   4,  1,  -2, 1,  3,  3,  2,  1,  -1, 1,
    -1, 6,  2,  1,  -3, -3, 2,  -1, -1, -1, -1, 1,  1,  -1,  0,  1,  1,  -3, -5, -1, 0,  4,  -1, -2, -1, 0,   -3, 0,   4,  2,  -1, -1, -4, -2, 4,  2,  0,  -1,
    3,  4,  1,  1,  0,  2,  2,  1,  -1, 0,  0,  -3, -1, 0,   1,  4,  -1, 2,  -1, 2,  1,  2,  0,  -1, 6,  -2,  1,  3,   -2, -1, 0,  0,  -2, 0,  2,  1,  -2, 1,
    0,  3,  2,  3,  4,  5,  2,  -4, 1,  3,  0,  -1, 1,  3,   1,  -3, -3, -2, -1, 5,  -1, -1, 0,  2,  1,  -2,  1,  1,   -1, 2,  0,  1,  3,  4,  2,  -1, 2,  3,
    3,  1,  1,  2,  1,  0,  -1, 1,  3,  0,  3,  2,  -1, 1,   0,  2,  0,  1,  -1, -4, 2,  -3, -2, -3, -1, 1,   1,  0,   -1, 1,  0,  -5, -7, -2, 0,  3,  -1, -4,
    -3, 2,  -2, -2, 3,  1,  -2, -3, -4, -4, 4,  0,  1,  3,   2,  3,  1,  3,  -3, 2,  2,  1,  2,  1,  -3, -2,  -1, -3,  1,  6,  -2, 0,  1,  1,  -2, 3,  1,  0,
    3,  -1, 3,  2,  0,  2,  -1, 2,  -3, 0,  0,  0,  0,  0,   1,  1,  2,  5,  4,  4,  3,  -3, 2,  4,  2,  -2,  0,  1,   3,  -2, -1, -1, -1, 2,  0,  0,  2,  -1,
    -1, -5, 0,  3,  1,  2,  1,  3,  4,  4,  2,  -1, 2,  2,   4,  0,  1,  2,  2,  1,  2,  -1, 4,  -2, 1,  0,   -1, 2,   -2, 2,  0,  5,  -1, -2, -1, 0,  1,  -2,
    -2, 1,  1,  0,  1,  0,  -2, -4, -4, 1,  3,  1,  0,  0,   -1, 0,  -3, -2, 6,  1,  -1, -2, -2, -4, 4,  2,   2,  1,   2,  4,  2,  0,  -2, 1,  2,  0,  2,  1,
    -3, -3, 0,  -2, -2, 8,  0,  2,  -1, 0,  0,  2,  1,  0,   1,  -2, 2,  5,  1,  1,  -1, 0,  -3, -2, -1, -1,  1,  0,   0,  1,  0,  5,  3,  6,  5,  -2, 3,  2,
    -1, -1, 3,  1,  1,  1,  -4, 1,  -2, 1,  1,  0,  1,  0,   1,  -2, 2,  1,  -2, 3,  4,  3,  2,  4,  2,  -1,  2,  0,   3,  2,  1,  1,  1,  0,  3,  -2, 3,  2,
    5,  -2, -1, 2,  -1, 4,  1,  2,  -3, 0,  1,  0,  1,  -2,  -2, 2,  1,  0,  1,  0,  1,  -3, -4, 2,  0,  3,   -1, 2,   -2, 0,  1,  -1, 3,  2,  -1, -3, -3, -3,
    3,  1,  1,  0,  0,  3,  1,  2,  0,  0,  2,  1,  1,  1,   -4, -1, 2,  -1, 0,  9,  -1, 4,  0,  1,  0,  3,   -4, -2,  -1, 3,  3,  2,  -1, 2,  1,  2,  0,  0,
    -1, 0,  -2, -8, -3, 2,  2,  2,  2,  -2, 0,  3,  -1, 1,   1,  1,  -1, -1, -1, 1,  -3, 0,  2,  2,  0,  -2,  0,  -1,  2,  0,  1,  5,  -2, -2, 1,  -1, 2,  2,
    5,  -2, 3,  1,  3,  0,  1,  3,  1,  4,  3,  -2, 3,  3,   2,  3,  2,  -3, -2, -2, -1, 0,  -1, 2,  0,  0,   2,  0,   0,  -3, -3, 0,  -1, -3, 1,  1,  0,  -1,
    -1, -1, -3, -2, 5,  0,  0,  -2, 0,  -2, -1, 2,  1,  -1,  2,  5,  2,  -1, 1,  1,  1,  0,  -2, 1,  0,  0,   -1, 2,   0,  2,  6,  8,  6,  5,  -2, 2,  1,  0,
    -5, 3,  -2, -1, 0,  0,  3,  0,  0,  0,  3,  1,  0,  0,   1,  0,  0,  -7, -4, 1,  0,  -1, -1, -2, -1, 2,   2,  0,   2,  2,  -1, -3, -1, 0,  -3, -2, 1,  2,
    -2, -3, -2, 2,  1,  -1, 1,  2,  1,  -2, 1,  1,  2,  1,   5,  -1, 1,  2,  2,  1,  1,  5,  2,  3,  2,  0,   3,  3,   3,  4,  -1, 1,  2,  2,  -4, 1,  1,  3,
    1,  1,  0,  1,  -2, -2, -3, 1,  -1, -2, 0,  1,  -3, -1,  -2, -2, -2, -1, 4,  0,  1,  -4, 0,  -1, -1, 4,   1,  0,   3,  5,  2,  0,  1,  1,  0,  -1, -2, 2,
    2,  -1, -1, 2,  3,  0,  4,  7,  4,  9,  -1, 3,  1,  1,   -1, 2,  0,  -1, -1, 3,  3,  -1, -2, 4,  3,  3,   0,  0,   0,  -1, 1,  -5, -3, 2,  2,  -1, 0,  -2,
    0,  1,  1,  0,  3,  3,  1,  -1, 0,  1,  -4, -3, -1, 1,   1,  -2, 0,  1,  1,  0,  1,  -1, 3,  0,  0,  0,   -1, -1,  5,  2,  3,  2,  3,  0,  1,  6,  2,  1,
    2,  -1, 3,  5,  2,  3,  -1, -3, -2, 2,  -2, 0,  1,  3,   0,  0,  3,  2,  1,  -3, -4, -4, -1, 0,  3,  1,   -3, -2,  -3, -1, 0,  -2, 3,  1,  2,  -4, -1, -1,
    -1, 1,  1,  -3, 3,  2,  4,  0,  -2, 0,  1,  -2, 2,  2,   1,  -3, -2, 2,  2,  -2, 3,  7,  7,  9,  -3, 2,   0,  2,   3,  1,  0,  1,  1,  0,  1,  0,  1,  2,
    2,  3,  -1, 0,  3,  -1, 0,  -5, -4, 4,  2,  -1, -2, -1,  2,  2,  0,  0,  1,  2,  1,  -1, 2,  0,  -2, 0,   0,  1,   2,  -2, -1, 0,  0,  0,  -1, 0,  0,  0,
    -1, 1,  1,  2,  3,  4,  3,  3,  0,  0,  1,  5,  1,  3,   1,  -1, -2, 5,  2,  4,  1,  -4, -3, -3, -2, 1,   2,  4,   1,  -1, 3,  2,  2,  -2, -3, 1,  0,  -2,
    2,  3,  1,  -2, -1, -1, -1, -2, 3,  2,  2,  -3, -4, -1,  1,  4,  0,  2,  0,  2,  4,  1,  0,  -1, 0,  -3,  -1, 3,   -1, -3, 1,  4,  5,  0,  5,  5,  7,  6,
    0,  2,  0,  0,  1,  1,  -4, -1, 0,  1,  0,  1,  0,  2,   2,  2,  -2, 0,  1,  -2, -1, -9, -4, 2,  3,  -3,  -1, 2,   0,  2,  0,  -1, 2,  -2, 0,  0,  1,  -4,
    -2, 1,  1,  0,  1,  -3, -3, 1,  0,  -1, 2,  1,  1,  -1,  -1, 0,  0,  -1, 4,  4,  2,  1,  0,  1,  1,  2,   0,  3,   1,  -2, -2, 0,  2,  3,  2,  -1, -2, -2,
    -3, 2,  2,  4,  0,  -1, 2,  2,  -3, -4, -3, 3,  3,  -3,  -1, 1,  0,  -1, -2, 0,  -1, -1, 3,  5,  2,  0,   -3, -1,  -1, 7,  0,  1,  0,  1,  4,  0,  1,  -1,
    3,  0,  0,  5,  0,  -2, -1, 2,  5,  0,  5,  5,  7,  9,   2,  -1, -2, -2, 0,  1,  -3, -3, 4,  -4, 4,  -6,  4,  4,   3,  -4, 2,  0,  -6, -1, -3, 2,  2,  -2,
    2,  0,  -1, 3,  2,  -5, 4,  4,  1,  1,  1,  -1, -4, 3,   1,  -2, -2, -2, -2, 1,  3,  -2, 2,  1,  -4, -2,  -1, 0,   1,  -1, 2,  -1, 0,  -2, -2, 1,  0,  -2,
    1,  1,  1,  3,  0,  1,  1,  -2, 6,  3,  3,  3,  5,  1,   -2, 2,  -1, -1, 1,  -1, -1, 0,  3,  3,  1,  1,   -2, -1,  1,  -2, -2, 1,  3,  5,  0,  -2, 0,  4,
    0,  1,  1,  3,  1,  -4, 2,  3,  -3, -1, 3,  2,  2,  1,   3,  -1, -2, 4,  2,  0,  -2, 0,  -2, -1, 0,  2,   4,  -1,  2,  3,  -1, -4, 1,  1,  -1, 1,  1,  0,
    -1, -5, 3,  3,  2,  -2, 2,  1,  -5, -2, 0,  2,  1,  0,   1,  -2, -3, 1,  2,  -1, 3,  3,  2,  0,  3,  0,   -1, 5,   1,  -3, 1,  0,  1,  1,  1,  -1, 2,  0,
    -7, -4, -3, -2, 2,  1,  0,  0,  1,  -1, -1, -1, 1,  -1,  -2, 0,  -1, 4,  0,  -1, -2, -1, 4,  2,  2,  0,   5,  2,   -1, 6,  -2, 0,  3,  -1, -2, 2,  2,  1,
    4,  0,  -2, 0,  2,  -2, 0,  3,  2,  5,  -1, -4, 0,  4,   -1, -1, 0,  1,  0,  -1, 0,  1,  -1, 1,  2,  3,   2,  2,   3,  -2, -1, 3,  0,  -1, -1, -1, -2, -4,
    0,  2,  3,  -1, 3,  1,  -2, -2, 2,  2,  0,  3,  3,  -1,  -2, -4, 4,  3,  -1, -2, 4,  1,  -3, -1, -1, 1,   3,  1,   1,  -2, 0,  0,  2,  0,  3,  0,  -1, 1,
    1,  0,  2,  2,  2,  1,  -1, 0,  -1, 1,  1,  -1, 1,  -2,  -5, 0,  -2, -2, 2,  -2, 3,  3,  2,  -1, 0,  0,   1,  -1,  -2, -2, 0,  2,  0,  -1, -1, 1,  3,  4,
    4,  -3, 4,  1,  1,  5,  -1, 0,  4,  1,  -2, 1,  4,  1,   3,  2,  0,  0,  1,  -4, -3, 2,  0,  5,  0,  -3,  2,  3,   2,  0,  1,  2,  -1, 1,  -2, -1, -3, 1,
    3,  4,  3,  3,  1,  -3, -1, 3,  -1, -1, 0,  -1, 0,  -4,  -4, 0,  3,  -1, 5,  1,  -1, -4, 0,  4,  2,  2,   4,  -1,  -3, -3, 1,  7,  -1, 1,  3,  -2, -1, 0,
    -1, 2,  2,  -1, 1,  -2, -2, 1,  2,  1,  2,  -2, -1, 0,   4,  1,  3,  1,  3,  1,  -3, 1,  -2, -2, -2, 1,   3,  -4,  -1, 1,  -3, -1, 0,  -3, 3,  2,  1,  -6,
    2,  1,  -1, 0,  -1, -1, 1,  1,  0,  -1, -2, 1,  4,  2,   5,  2,  3,  0,  2,  5,  -2, 2,  3,  0,  -2, 0,   3,  4,   5,  2,  -2, -3, 3,  -4, -5, 1,  4,  2,
    2,  2,  1,  3,  3,  3,  4,  3,  3,  0,  -4, 0,  -2, 3,   2,  2,  1,  4,  1,  -1, -1, 0,  0,  -2, -1, 1,   -2, -3,  -3, 1,  2,  -3, 4,  1,  1,  -3, 0,  2,
    -2, 5,  5,  -4, -2, -2, 0,  3,  0,  3,  1,  -1, -1, 1,   -1, 1,  2,  0,  -2, -2, -2, -1, 2,  3,  4,  -1,  0,  1,   3,  1,  -1, 0,  2,  2,  0,  -1, 0,  1,
    -1, 2,  2,  -1, -2, -3, -2, -2, 1,  -3, -1, 2,  2,  -2,  2,  0,  -1, 0,  -1, -3, 1,  0,  -3, -1, -1, -1,  5,  1,   2,  2,  6,  -1, 0,  2,  0,  -2, 3,  1,
    0,  0,  2,  0,  4,  2,  -1, -1, 3,  -3, -3, 3,  1,  3,   1,  -2, 2,  3,  4,  3,  3,  0,  2,  0,  -2, -1,  -1, 3,   1,  1,  0,  3,  0,  -2, 1,  3,  -2, 0,
    -1, -2, -4, -3, 0,  -1, 4,  -3, 0,  0,  2,  1,  -3, 1,   2,  0,  1,  1,  -2, 0,  -3, 0,  2,  0,  -3, 4,   -1, -2,  -1, -4, 4,  -1, -1, 2,  1,  -2, -1, 5,
    3,  -1, 1,  0,  -1, 0,  0,  -1, -1, -1, 0,  0,  0,  -2,  2,  3,  1,  4,  -1, 2,  -2, -3, -3, -3, 0,  3,   -1, 0,   -1, 0,  0,  3,  1,  -1, 3,  2,  2,  1,
    3,  -2, -1, -1, 5,  4,  0,  2,  4,  -1, 4,  4,  3,  4,   0,  3,  -2, 1,  -7, 1,  -2, 1,  0,  -2, -5, -1,  -3, -2,  -1, -2, 0,  3,  2,  2,  2,  -2, -3, -1,
    -3, 2,  -2, -5, -3, -1, 0,  4,  2,  -3, 2,  0,  4,  2,   3,  2,  -4, -6, -3, -4, -2, 4,  4,  -1, 1,  1,   -1, 0,   0,  0,  -1, 1,  -3, 1,  -1, 0,  1,  3,
    0,  5,  -1, -1, 0,  -1, 4,  0,  1,  2,  1,  1,  1,  5,   2,  1,  2,  1,  1,  1,  0,  -2, -1, 0,  1,  -1,  1,  0,   3,  2,  -1, 1,  0,  2,  0,  -2, -2, -1,
    3,  1,  -1, -2, 0,  0,  1,  3,  1,  2,  3,  3,  -1, 1,   3,  0,  2,  1,  7,  4,  3,  3,  3,  1,  4,  5,   4,  0,   1,  2,  -1, 0,  -4, -3, -2, -1, -1, -2,
    -5, -1, -1, -3, -3, -2, 1,  1,  3,  3,  1,  1,  1,  -2,  -3, 0,  0,  -6, -4, 0,  1,  4,  2,  -2, 1,  2,   4,  3,   6,  3,  -4, -7, -2, -2, 0,  2,  1,  -1,
    2,  2,  -2, 1,  1,  1,  0,  2,  -2, 2,  -2, -3, 2,  3,   1,  2,  -1, 0,  -2, 1,  1,  3,  2,  1,  1,  0,   1,  4,   4,  1,  3,  1,  -1, 2,  -2, 0,  -1, 0,
    0,  -3, 1,  2,  2,  2,  1,  0,  1,  3,  0,  -3, -2, 0,   2,  0,  -1, -4, 1,  1,  2,  3,  0,  3,  5,  5,   0,  3,   3,  0,  0,  1,  6,  5,  3,  2,  2,  1,
    3,  4,  1,  2,  -1, -1, 0,  -1, -5, -3, -3, -1, 2,  -4,  -5, 0,  -3, 0,  -4, -5, 0,  1,  6,  3,  4,  0,   -2, -2,  -3, 0,  1,  -5, -3, 0,  0,  3,  0,  -2,
    2,  3,  5,  3,  2,  3,  -2, -7, -5, -2, -4, 3,  1,  -4,  2,  -1, -2, 2,  0,  0,  0,  1,  0,  1,  -4, 0,   3,  2,   3,  5,  0,  2,  -1, -2, 1,  0,  2,  2,
    -1, -1, -3, 4,  3,  0,  2,  0,  -1, 2,  -1, 1,  -2, 2,   -2, -2, 3,  0,  1,  1,  0,  1,  0,  1,  1,  -2,  -1, 0,   0,  0,  -1, -1, 1,  2,  5,  2,  1,  1,
    5,  4,  -2, 3,  2,  3,  -2, 0,  3,  6,  2,  2,  0,  2,   4,  4,  0,  0,  -3, 1,  1,  -2, -4, -2, -3, -1,  1,  -3,  -4, -1, 0,  0,  -2, -3, -1, 1,  4,  0,
    2,  -1, -2, -3, -5, 1,  1,  -6, -3, 1,  1,  2,  0,  -2,  0,  -1, 2,  4,  4,  6,  -4, -5, -2, -1, -4, 3,   1,  -3,  1,  0,  -4, -1, -1, 0,  -1, 1,  -1, 0,
    -3, 2,  3,  0,  0,  5,  -1, 1,  0,  1,  0,  1,  3,  0,   -2, 2,  -2, 2,  3,  0,  1,  1,  1,  2,  -1, -1,  0,  1,   0,  -3, 1,  2,  1,  3,  1,  0,  0,  2,
    -2, -1, -3, 1,  2,  0,  -1, 0,  0,  3,  6,  1,  0,  -1,  3,  3,  -1, 3,  3,  2,  1,  -1, 5,  4,  1,  3,   2,  3,   5,  3,  -1, 0,  -1, 2,  -1, 1,  -2, -1,
    -2, 1,  1,  -3, -3, -3, -2, 0,  0,  -5, -1, -1, 3,  1,   0,  2,  -2, -3, -5, 3,  1,  -5, -2, 0,  2,  2,   0,  -1,  0,  -2, 3,  6,  4,  5,  -4, -6, -2, -2,
    -4, 1,  2,  3,  2,  1,  2,  3,  5,  1,  9,  5,  1,  1,   2,  -4, 2,  3,  1,  3,  4,  0,  -1, 1,  -1, 2,   5,  -1,  0,  2,  6,  1,  1,  2,  -1, 0,  0,  0,
    2,  1,  -3, 2,  1,  -3, -1, 2,  1,  1,  3,  2,  3,  -2,  -1, 4,  2,  1,  1,  1,  1,  0,  -3, 0,  0,  2,   0,  -1,  2,  1,  -2, 0,  0,  -3, 1,  -1, 1,  -3,
    0,  -2, -4, 1,  2,  -1, 2,  -2, 2,  0,  -1, -1, 2,  2,   3,  4,  -2, 0,  0,  -1, -5, 0,  3,  -3, -2, -3,  -1, -4,  -3, 2,  0,  -1, 0,  0,  1,  1,  3,  3,
    3,  0,  4,  4,  0,  -3, -1, 4,  1,  2,  6,  0,  3,  -1,  0,  -3, 3,  5,  3,  0,  4,  1,  3,  1,  7,  2,   1,  1,   1,  -3, 4,  2,  1,  4,  4,  -1, 0,  0,
    2,  3,  4,  -1, 3,  3,  6,  -1, -2, 3,  1,  0,  1,  1,   0,  3,  -1, 1,  3,  -7, -1, 2,  -3, -1, 3,  1,   2,  -1,  -2, 3,  0,  1,  3,  0,  1,  3,  0,  -1,
    -2, 4,  0,  0,  0,  2,  1,  0,  -1, -2, 2,  -2, -1, -4,  1,  -3, -7, 1,  -2, -1, -2, 0,  2,  0,  3,  -2,  1,  0,   3,  0,  0,  -1, 0,  -1, -5, 2,  6,  -2,
    -3, -5, 0,  -4, -2, 2,  0,  2,  1,  0,  3,  1,  0,  0,   5,  1,  6,  3,  2,  0,  -3, 3,  -2, 1,  6,  1,   0,  0,   0,  -1, 5,  4,  5,  0,  4,  1,  4,  0,
    6,  4,  -2, 1,  3,  -5, 2,  2,  2,  1,  3,  1,  0,  2,   2,  2,  5,  -3, 2,  4,  4,  -2, -4, 2,  0,  0,   0,  0,   0,  0,  0,  3,  -3, -6, 0,  0,  -3, -1,
    2,  1,  -1, 0,  2,  3,  3,  0,  0,  0,  0,  1,  0,  -2,  0,  1,  0,  0,  4,  2,  -3, -1, -2, -3, 0,  0,   0,  0,   -1, -1, -7, 4,  -2, -1, 0,  -2, 0,  2,
    3,  2,  -1, 3,  4,  2,  -1, 0,  2,  1,  -4, 1,  2,  -1,  -2, -4, -1, -3, 0,  1,  -3, -1, -1, 0,  -2, 2,   -2, -1,  5,  -1, 3,  4,  3,  -1, -4, 0,  -3, 2,
    6,  4,  3,  1,  0,  -3, -1, 7,  5,  2,  3,  2,  4,  0,   8,  4,  -2, 2,  1,  -5, 3,  3,  3,  3,  3,  1,   -1, 0,   -1, 4,  3,  -4, 4,  3,  5,  0,  -3, 3,
    0,  0,  0,  4,  2,  -1, 0,  0,  1,  -1, 0,  1,  -2, -1,  3,  0,  3,  0,  3,  -1, 3,  -1, -1, -1, 0,  1,   2,  -1,  2,  0,  0,  0,  1,  2,  0,  -2, -1, -2,
    1,  -2, 0,  0,  -1, -3, -8, 4,  -2, -2, 1,  -1, 0,  2,   3,  1,  -3, 2,  3,  0,  0,  3,  5,  2,  -5, 1,   1,  0,   -2, -2, 1,  -2, 0,  5,  2,  -3, -2, 3,
    0,  2,  2,  0,  1,  1,  1,  2,  3,  0,  -2, 3,  0,  2,   4,  -1, 5,  1,  1,  -1, 2,  4,  6,  2,  4,  2,   5,  1,   8,  3,  -1, 0,  -3, -4, 4,  1,  -1, 8,
    2,  1,  -1, 4,  0,  3,  4,  1,  7,  3,  5,  2,  -1, -1,  1,  -3, 0,  3,  1,  0,  0,  2,  4,  -3, 1,  2,   -1, -1,  4,  1,  2,  2,  2,  -1, 2,  0,  -1, -1,
    0,  2,  1,  -2, 1,  0,  1,  1,  2,  1,  0,  -1, 1,  -2,  0,  -1, -2, 1,  -2, -2, -8, 2,  0,  -1, 0,  -2,  1,  1,   2,  3,  -1, 2,  5,  -1, 1,  6,  5,  5,
    -6, -2, -1, 0,  -1, -5, 1,  2,  1,  5,  2,  0,  0,  2,   1,  1,  2,  -2, -2, 1,  -1, 1,  0,  -2, -4, 2,   0,  4,   3,  2,  4,  3,  0,  2,  2,  0,  -1, 0,
    -1, 0,  3,  3,  -1, 1,  -1, 3,  2,  -1, 1,  0,  -3, -2,  -3, 0,  1,  -3, 1,  3,  1,  -1, 1,  -1, -4, -1,  -1, 0,   4,  2,  1,  3,  3,  -1, -1, 2,  2,  -3,
    -4, 0,  1,  -1, 2,  2,  0,  -2, -1, 1,  -3, -2, 0,  -2,  2,  -1, -2, -1, 1,  -3, -2, 3,  1,  2,  3,  2,   0,  1,   2,  1,  2,  1,  1,  5,  1,  3,  -4, 1,
    4,  0,  1,  4,  4,  2,  3,  1,  2,  -4, -2, 1,  -2, 1,   3,  0,  0,  1,  3,  2,  -1, -2, 0,  2,  0,  0,   -3, 1,   -4, 1,  2,  -2, -1, -1, 3,  3,  -1, 3,
    -2, 0,  -1, 1,  1,  3,  2,  5,  5,  2,  2,  0,  -2, 2,   -1, 1,  2,  1,  -2, 2,  1,  1,  0,  -1, 2,  -2,  -2, -4,  -3, -2, 2,  -3, 0,  3,  0,  -3, 1,  0,
    -2, -5, -3, 1,  3,  -3, 1,  2,  4,  -2, 1,  2,  0,  -3,  -6, 1,  1,  -2, 1,  0,  2,  -3, 1,  1,  0,  -2,  0,  0,   3,  0,  -2, -1, -1, -2, -1, 2,  1,  2,
    2,  2,  -1, 1,  1,  2,  1,  0,  0,  4,  1,  0,  -3, 1,   3,  1,  3,  2,  2,  5,  4,  -2, 2,  0,  2,  5,   -1, 2,   1,  -1, 0,  -1, 0,  0,  -3, -3, 3,  3,
    1,  1,  -1, 1,  -3, 1,  2,  -1, 0,  1,  4,  4,  1,  2,   -3, 0,  -1, 2,  4,  5,  0,  3,  3,  -2, 0,  0,   -3, 2,   -2, -3, 3,  3,  -2, 1,  3,  1,  2,  0,
    2,  3,  1,  -5, 1,  1,  3,  1,  0,  2,  2,  -4, -1, 2,   -3, -1, -2, -1, 3,  -1, 1,  3,  4,  -1, 2,  0,   3,  -3,  -5, -1, 2,  -1, 1,  -1, -1, -1, -1, 1,
    -1, -3, -1, -1, 3,  1,  -1, -3, 0,  -1, 1,  3,  2,  2,   0,  -1, 0,  0,  1,  2,  1,  -1, -1, 5,  1,  0,   -5, -1,  4,  1,  4,  2,  5,  7,  2,  -1, 3,  1,
    0,  4,  0,  2,  5,  2,  0,  -3, 2,  1,  -1, 3,  4,  4,   4,  1,  -1, -1, -4, 2,  4,  -2, -2, 1,  2,  4,   1,  2,   -2, 2,  -1, 3,  3,  3,  3,  2,  3,  0,
    -3, 1,  -1, 0,  -4, -2, 6,  1,  0,  -2, 8,  2,  2,  1,   3,  -1, 1,  -4, 1,  2,  2,  -1, 1,  4,  0,  -1,  0,  -1,  -3, -1, 0,  3,  2,  1,  2,  4,  4,  -1,
    -1, 0,  0,  0,  -3, 0,  0,  -4, 1,  0,  -2, 3,  -2, 1,   -1, -1, -1, 0,  1,  1,  0,  0,  0,  0,  -1, 1,   2,  3,   -1, 1,  3,  0,  2,  2,  1,  -1, 0,  2,
    2,  2,  -5, -1, 1,  -1, 2,  2,  2,  2,  0,  -2, 4,  1,   2,  4,  1,  -2, 3,  2,  2,  0,  3,  1,  0,  2,   3,  1,   4,  -1, -5, -1, -5, 0,  0,  -3, -4, 2,
    2,  4,  2,  0,  -2, 1,  -1, -1, 1,  0,  3,  4,  4,  0,   0,  1,  1,  0,  -4, -3, 7,  2,  -1, 0,  2,  4,   -1, 1,   2,  3,  0,  -1, 0,  2,  1,  -2, 0,  1,
    2,  -1, 0,  2,  -3, -2, -1, 3,  3,  -2, 1,  2,  0,  0,   -1, 1,  1,  1,  -1, 2,  0,  -2, -2, 1,  2,  1,   -2, 0,   0,  -1, -1, 1,  1,  3,  0,  -2, 0,  0,
    -1, 1,  1,  3,  -3, 0,  0,  1,  2,  2,  1,  4,  1,  3,   3,  1,  -3, -2, 2,  0,  1,  2,  1,  3,  1,  0,   3,  3,   3,  1,  0,  1,  2,  0,  2,  -1, 1,  2,
    1,  1,  4,  5,  -1, -2, 1,  0,  -4, 0,  1,  -1, -3, 3,   2,  2,  3,  1,  -3, -1, -3, -2, 0,  1,  4,  3,   3,  -1,  5,  -2, 3,  0,  0,  3,  0,  1,  1,  -2,
    0,  7,  3,  -2, -2, -3, 2,  0,  -1, -3, 5,  3,  -1, -2,  -2, -2, -4, 1,  -3, 0,  -2, -3, 1,  0,  0,  3,   1,  -1,  4,  1,  2,  7,  2,  3,  2,  0,  -1, -1,
    4,  -1, -4, -1, 1,  0,  -1, -2, 0,  -1, 0,  2,  1,  2,   1,  2,  -1, 2,  1,  1,  -3, -1, 4,  3,  5,  2,   -1, -5,  -3, 0,  -2, -1, 1,  0,  -2, 3,  2,  -1,
    2,  -2, 5,  1,  -1, -1, 1,  2,  0,  4,  5,  -7, -2, 1,   0,  2,  2,  -2, -3, 3,  -3, 2,  0,  1,  -3, 4,   1,  1,   3,  0,  0,  2,  -1, 2,  0,  -3, 2,  0,
    5,  0,  -4, -1, 1,  -2, 0,  0,  0,  2,  -1, 0,  0,  1,   -1, 6,  3,  -3, -2, -1, 1,  0,  0,  -6, 4,  -2,  1,  -2,  -3, -3, -2, 0,  -2, 0,  -2, -2, 1,  1,
    -1, 1,  2,  0,  1,  0,  1,  5,  4,  3,  2,  -1, 0,  0,   2,  -1, -5, 1,  0,  0,  0,  -1, -1, 0,  -2, 1,   1,  2,   2,  3,  -2, 2,  0,  -1, 0,  0,  4,  1,
    6,  2,  -1, -6, -2, 4,  -1, -2, 2,  0,  -1, 4,  0,  1,   1,  1,  3,  4,  0,  1,  0,  -3, 3,  4,  5,  -2,  -2, 1,   1,  2,  0,  1,  -4, -1, -3, 1,  -1, 0,
    -1, 5,  1,  2,  1,  1,  1,  2,  -3, 3,  1,  -4, 1,  2,   5,  2,  -2, -1, 6,  -2, 3,  1,  -2, 2,  1,  -2,  -1, -1,  1,  3,  4,  2,  -2, 0,  3,  1,  1,  -2,
    4,  0,  1,  -1, -2, -2, 1,  0,  -6, 2,  -3, 2,  2,  1,   1,  2,  3,  0,  0,  1,  2,  3,  2,  4,  2,  -3,  1,  -1,  0,  -1, -4, 3,  0,  0,  -1, -2, -2, 0,
    -2, 0,  1,  1,  1,  4,  -2, 2,  1,  0,  -4, -2, 4,  4,   7,  1,  -1, -4, -3, 3,  -1, 0,  2,  0,  -1, 5,   1,  -1,  0,  4,  4,  4,  -1, 0,  -1, 0,  -1, 2,
    4,  -3, -1, -2, 0,  3,  -1, 0,  -2, 1,  -4, 0,  -3, -1,  0,  3,  4,  3,  -1, 1,  0,  1,  -3, 4,  3,  -2,  3,  3,   5,  2,  -2, 3,  6,  0,  3,  -2, 0,  4,
    0,  -2, -2, 1,  1,  3,  2,  4,  -2, -3, 0,  0,  -2, -1,  5,  0,  3,  1,  -3, -4, -2, 3,  -5, 0,  -2, 0,   1,  1,   -1, 3,  1,  -1, -1, -3, 1,  4,  2,  2,
    2,  0,  0,  -3, -2, -2, -5, 2,  1,  3,  0,  -5, -2, 1,   -1, 1,  -1, 1,  2,  4,  2,  3,  0,  2,  0,  -2,  4,  2,   6,  1,  -2, -4, -5, 1,  -4, 1,  4,  -2,
    -2, 5,  1,  2,  0,  4,  6,  6,  0,  0,  2,  -1, 2,  4,   4,  -4, -1, -2, -1, 3,  1,  0,  -1, 1,  -3, -2,  -3, 0,   -1, 4,  2,  -3, -1, 0,  1,  0,  -2, 2,
    2,  -2, 5,  2,  4,  2,  -1, 1,  4,  -2, 2,  0,  -1, 1,   1,  -2, 0,  3,  -1, 3,  5,  3,  -3, -1, -1, 0,   0,  0,   3,  -4, 3,  0,  -4, -3, -2, 3,  -1, 0,
    -1, 1,  -1, -1, -2, 2,  0,  0,  2,  -3, 0,  4,  3,  0,   2,  -1, 0,  0,  -3, -4, -5, 2,  1,  4,  1,  -3,  -2, 0,   -1, 1,  2,  2,  3,  2,  1,  0,  1,  0,
    -1, -3, 2,  2,  5,  0,  -3, -3, -8, 3,  -1, 2,  2,  -1,  0,  4,  2,  2,  1,  3,  6,  4,  -2, 0,  1,  -3,  -1, 4,   -1, -4, -5, -2, 1,  2,  -1, -3, -1, 0,
    -2, 0,  -2, -1, -2, 2,  2,  -1, 0,  -1, -2, 1,  -4, 0,   2,  -2, 6,  2,  4,  1,  1,  1,  -2, -1, 0,  -3,  -2, 1,   2,  1,  -1, 1,  2,  -2, -1, -3, 1,  -3,
    0,  -5, 0,  0,  4,  -2, -6, -2, 1,  3,  3,  3,  4,  1,   1,  1,  -1, -3, -1, 0,  1,  1,  3,  5,  0,  -1,  2,  -2,  -2, 0,  2,  4,  -2, -3, -2, -1, 3,  -1,
    -3, -1, 3,  2,  -1, -1, 3,  0,  1,  3,  1,  1,  0,  2,   1,  -2, 2,  3,  3,  -1, -2, 3,  2,  3,  2,  1,   2,  3,   0,  2,  3,  6,  5,  1,  3,  3,  0,  2,
    -4, 0,  3,  1,  -1, -2, 1,  2,  0,  0,  -1, -4, -1, -1,  2,  1,  -1, 3,  -1, 0,  2,  0,  1,  -1, 0,  -1,  2,  3,   -4, -2, -3, 2,  0,  -6, -1, -2, -4, -1,
    0,  0,  -1, 1,  4,  2,  -3, 1,  0,  -1, -2, -2, 2,  -5,  1,  0,  -5, 0,  1,  1,  -3, 0,  0,  1,  2,  3,   6,  2,   0,  1,  -2, -3, -3, -1, 4,  0,  2,  3,
    1,  -1, 0,  0,  -5, 1,  0,  2,  0,  -5, -3, 0,  5,  -2,  -3, -2, 1,  3,  0,  1,  0,  -1, -1, 2,  2,  0,   0,  3,   3,  4,  1,  1,  5,  -1, -4, 0,  2,  1,
    1,  5,  3,  1,  -1, 1,  5,  2,  0,  2,  1,  2,  2,  2,   -5, -2, 0,  0,  1,  1,  -1, 2,  1,  1,  -1, -3,  -1, 0,   0,  2,  0,  -1, 1,  2,  1,  1,  2,  -2,
    1,  0,  2,  2,  -1, -1, 0,  1,  1,  -4, 0,  -1, -3, 1,   0,  -2, 2,  3,  2,  2,  0,  2,  3,  -1, -2, 0,   2,  -5,  -2, 0,  -1, -1, 2,  1,  -3, -4, 5,  2,
    -2, 3,  2,  3,  -1, 2,  1,  -1, -1, 2,  3,  1,  0,  4,   -1, 1,  1,  -2, -3, 0,  1,  3,  -2, -3, -4, 0,   4,  -2,  -2, -2, 0,  2,  -2, 1,  0,  0,  0,  2,
    1,  0,  0,  4,  2,  1,  1,  1,  3,  -2, -4, 3,  2,  0,   2,  3,  0,  0,  0,  3,  2,  3,  2,  1,  -1, -1,  1,  3,   -1, 0,  2,  1,  0,  2,  0,  3,  3,  0,
    0,  -2, 2,  1,  -1, 2,  1,  0,  4,  0,  1,  0,  2,  -2,  -1, 1,  1,  2,  1,  0,  2,  2,  -1, -2, 3,  -1,  -5, 1,   2,  1,  3,  1,  -1, 3,  0,  0,  5,  -3,
    0,  0,  1,  -2, -2, 2,  2,  -2, 3,  -2, 0,  -2, 0,  3,   0,  2,  0,  2,  1,  -1, 0,  -3, 0,  2,  3,  3,   2,  1,   0,  -2, 1,  2,  -3, -2, 0,  3,  -2, -4,
    -3, 0,  3,  -1, -2, -2, -2, 4,  -2, 0,  1,  0,  1,  1,   1,  3,  0,  4,  0,  3,  -1, -1, 4,  -1, -2, 0,   1,  2,   1,  1,  -2, 3,  1,  -2, 1,  3,  1,  2,
    0,  -1, 0,  1,  -1, 3,  3,  3,  3,  1,  2,  2,  1,  1,   0,  1,  4,  2,  1,  1,  1,  1,  1,  -1, 1,  0,   3,  -1,  -2, 0,  0,  2,  1,  0,  -2, 2,  -2, -4,
    0,  -1, -1, 1,  3,  1,  2,  4,  1,  3,  -2, -1, 3,  -1,  0,  2,  -1, -3, 0,  0,  2,  -1, 4,  0,  -2, -2,  -1, 1,   0,  0,  0,  2,  -3, -2, 0,  -2, 0,  3,
    -1, 3,  0,  2,  2,  -4, 1,  4,  -2, 0,  0,  2,  2,  0,   -4, -1, 2,  -1, -3, -4, 0,  1,  -1, -1, 2,  0,   0,  2,   1,  1,  -1, 3,  1,  1,  0,  0,  3,  -1,
    1,  0,  2,  1,  0,  1,  1,  1,  2,  2,  0,  3,  1,  2,   2,  3,  0,  0,  -1, 1,  2,  2,  1,  2,  -1, 1,   3,  1,   -1, 3,  5,  3,  -2, 0,  0,  1,  4,  0,
    1,  0,  4,  -1, -3, 3,  1,  3,  0,  -1, -3, 1,  1,  -5,  2,  2,  0,  -1, 1,  0,  -3, 3,  1,  0,  3,  4,   -1, -1,  4,  -1, 1,  4,  3,  0,  -1, 1,  -1, 1,
    1,  0,  1,  2,  3,  -2, -2, 3,  3,  -1, -2, 2,  2,  0,   1,  1,  -1, 3,  2,  0,  1,  4,  1,  2,  1,  1,   -2, 3,   1,  2,  1,  -2, 1,  -1, 1,  -1, 0,  -4,
    0,  1,  0,  1,  1,  0,  -1, -2, -4, 1,  0,  0,  1,  -2,  2,  1,  -4, 1,  6,  1,  4,  4,  4,  1,  -2, 0,   -3, -4,  -3, 1,  3,  2,  -2, 0,  -1, 5,  2,  -3,
    -1, 0,  2,  1,  2,  2,  3,  2,  1,  -2, -1, 2,  -2, 0,   1,  4,  -3, 0,  3,  2,  4,  2,  0,  0,  -1, 1,   0,  -1,  1,  -1, 0,  -1, 0,  -1, -2, 3,  1,  1,
    3,  4,  -2, 0,  2,  0,  2,  2,  2,  1,  0,  1,  1,  1,   0,  -1, 2,  0,  2,  -1, -3, 0,  2,  1,  -2, 1,   2,  1,   2,  0,  0,  4,  3,  2,  0,  4,  -1, 1,
    2,  0,  -2, 5,  2,  1,  1,  -1, 3,  -1, 1,  -2, -2, -3,  0,  1,  1,  1,  1,  -1, 0,  -2, -2, 1,  -3, -2,  1,  -1,  1,  1,  -1, 0,  5,  1,  4,  4,  3,  -2,
    -4, -1, -3, -2, -1, -1, 3,  4,  1,  0,  -2, 5,  2,  -2,  2,  0,  0,  3,  1,  2,  3,  3,  0,  -4, 0,  2,   -3, -2,  1,  4,  -2, -2, 1,  2,  3,  4,  2,  1,
    0,  0,  -2, 0,  0,  0,  1,  1,  -1, 0,  -4, 1,  2,  0,   4,  2,  0,  -3, 3,  0,  2,  0,  3,  2,  0,  0,   0,  0,   2,  0,  0,  0,  0,  -1, -1, 2,  2,  0,
    -1, 1,  2,  1,  1,  -1, 1,  1,  2,  1,  1,  3,  0,  1,   2,  0,  -1, 4,  1,  1,  2,  0,  3,  0,  1,  -5,  -1, -2,  0,  0,  0,  0,  -1, 0,  -2, 0,  -3, -2,
    -4, -1, 0,  -2, 1,  1,  -1, 2,  6,  3,  1,  3,  2,  -1,  -3, 0,  -2, -3, -2, 0,  3,  3,  2,  2,  2,  3,   2,  -3,  1,  0,  0,  3,  0,  2,  0,  1,  -1, -3,
    -2, 0,  -3, -4, -1, 3,  -1, -4, -1, 1,  3,  4,  2,  0,   -1, 0,  -1, 0,  0,  1,  1,  0,  1,  2,  -4, 4,   1,  3,   2,  3,  0,  -2, 1,  2,  1,  0,  2,  0,
    0,  -1, 2,  -1, 0,  0,  2,  -1, 0,  0,  -1, 2,  2,  1,   0,  2,  1,  -1, 2,  0,  1,  3,  1,  3,  1,  4,   -2, 1,   0,  2,  -3, 2,  0,  0,  3,  0,  3,  1,
    0,  -3, -2, -1, -1, -1, 0,  1,  -1, -1, -1, 1,  -4, 1,   -2, -2, 0,  -1, 1,  -2, -2, 3,  5,  2,  2,  2,   1,  0,   -2, 0,  -3, -2, -1, -1, 1,  2,  3,  0,
    2,  1,  0,  -1, 2,  -1, 1,  5,  3,  4,  1,  2,  0,  -1,  1,  0,  -4, -2, 0,  1,  -1, -1, 0,  2,  3,  3,   0,  -1,  0,  -1, -3, -1, 1,  0,  2,  1,  2,  1,
    -2, 2,  -1, 3,  2,  2,  2,  -1, 1,  3,  1,  1,  1,  1,   2,  1,  0,  0,  2,  -2, 1,  1,  2,  1,  -1, 1,   2,  3,   -1, 2,  2,  3,  -1, -1, 0,  4,  1,  2,
    0,  2,  -1, 3,  1,  0,  0,  4,  0,  2,  5,  0,  4,  -1,  2,  -2, -2, -1, 1,  0,  0,  0,  1,  0,  0,  0,   -3, 0,   -1, -4, -1, 1,  -2, -1, -3, 2,  5,  0,
    3,  2,  1,  -1, -2, 1,  -1, -4, 1,  3,  2,  4,  2,  2,   6,  1,  -1, 0,  2,  -3, 2,  4,  3,  5,  4,  2,   1,  -3,  1,  1,  -3, -4, -1, 0,  -1, -2, -1, 3,
    4,  2,  -1, 1,  1,  0,  -3, -2, 2,  1,  -1, 1,  1,  -2,  2,  2,  2,  1,  2,  -2, -1, 0,  -2, 1,  2,  1,   0,  -3,  0,  -1, -2, 0,  2,  2,  -2, -1, -1, 1,
    -4, -2, -1, -1, 1,  1,  2,  -1, 3,  0,  3,  2,  3,  2,   -1, 4,  -3, 1,  2,  -1, 2,  -1, 1,  0,  -1, -1,  1,  -3,  0,  2,  -1, 0,  0,  0,  0,  0,  0,  1,
    1,  0,  -5, 2,  3,  0,  -2, -3, -1, -1, 1,  -4, 2,  0,   -5, -3, -4, -2, -2, 1,  1,  1,  0,  -4, 0,  1,   4,  0,   -2, 1,  3,  -2, -5, -3, 1,  1,  -3, 0,
    1,  -2, -2, 0,  0,  4,  2,  0,  1,  3,  0,  -1, 2,  5,   2,  4,  2,  6,  3,  5,  3,  0,  0,  -1, 0,  0,   1,  0,   1,  0,  1,  2,  2,  -1, 1,  -1, -6, -2,
    5,  -1, 1,  -5, 2,  2,  -3, -1, 1,  2,  2,  -2, 1,  -1,  -5, -2, -1, 2,  -1, 0,  -2, -2, 2,  4,  6,  1,   3,  3,   -1, 3,  -1, 0,  2,  -1, 1,  0,  0,  1,
    0,  2,  0,  -3, 1,  1,  1,  -1, -1, 2,  -1, 1,  2,  0,   1,  0,  -2, 3,  1,  -2, -2, -1, 0,  1,  0,  -3,  1,  1,   -3, -3, 0,  -1, -2, -2, 2,  1,  0,  -1,
    0,  3,  5,  1,  1,  -1, 2,  -3, -3, 0,  2,  0,  0,  1,   4,  -1, -1, 1,  0,  2,  0,  1,  3,  5,  1,  -3,  1,  2,   2,  1,  0,  2,  3,  3,  1,  0,  1,  -1,
    -2, 1,  2,  -1, 1,  2,  -3, 0,  3,  0,  1,  -1, -3, -3,  2,  0,  1,  1,  0,  6,  -1, 1,  0,  3,  2,  0,   -1, -2,  -4, -5, 1,  3,  -2, 3,  0,  2,  1,  5,
    4,  3,  2,  0,  3,  1,  1,  2,  1,  1,  1,  2,  -3, 1,   2,  2,  0,  0,  1,  4,  0,  1,  -2, 0,  1,  2,   2,  0,   2,  -1, 0,  3,  2,  0,  -1, -2, -1, -1,
    0,  -4, -1, 2,  -1, -1, -2, 1,  -2, -1, -1, 0,  2,  0,   -2, 1,  3,  2,  0,  2,  2,  -1, -2, 1,  1,  -1,  2,  1,   5,  1,  1,  2,  1,  2,  2,  1,  4,  4,
    -2, -1, 3,  1,  3,  2,  3,  2,  1,  2,  1,  1,  -1, -2,  0,  0,  1,  0,  -1, 1,  -1, -1, -2, 2,  2,  -3,  -2, -1,  -1, 1,  0,  -3, 2,  2,  -2, 1,  1,  2,
    3,  -2, 0,  -1, -4, -2, -2, 2,  3,  -1, 1,  1,  0,  5,   1,  1,  4,  2,  3,  0,  2,  0,  1,  1,  -2, -1,  0,  2,   0,  2,  1,  -1, 1,  1,  3,  2,  -1, -1,
    0,  0,  1,  2,  1,  -1, -1, 0,  1,  -2, -1, -2, 1,  0,   -4, -3, 0,  2,  0,  -2, -2, 1,  1,  -1, -2, 3,   4,  1,   -1, 1,  4,  3,  1,  -2, 2,  -1, -2, -1,
    1,  0,  1,  2,  1,  -1, 0,  2,  -2, -1, 0,  -2, 4,  1,   -2, -2, -1, 0,  0,  0,  1,  -1, 0,  1,  -2, 0,   -3, -1,  0,  -1, 1,  -2, 3,  -1, 2,  0,  -3, 4,
    3,  1,  -2, 0,  -2, 0,  2,  -1, 4,  2,  1,  3,  0,  2,   0,  -2, 0,  -2, -1, 1,  -2, 3,  4,  2,  1,  3,   0,  5,   0,  1,  4,  3,  3,  3,  -1, 1,  0,  3,
    2,  0,  2,  1,  1,  2,  3,  3,  2,  3,  2,  1,  0,  1,   0,  1,  -2, 0,  1,  2,  1,  0,  5,  -1, 2,  0,   2,  1,   -1, -3, 2,  2,  3,  -2, 1,  3,  1,  1,
    2,  1,  3,  3,  0,  5,  1,  1,  -1, 0,  2,  1,  -2, 0,   2,  2,  2,  1,  -1, 2,  2,  3,  1,  0,  4,  0,   5,  3,   1,  -3, 2,  4,  4,  5,  4,  3,  1,  3,
    1,  -1, 1,  -2, -1, 2,  2,  -1, 0,  -1, 1,  0,  -2, 2,   -1, -1, -1, -1, 2,  0,  2,  0,  2,  3,  1,  0,   0,  -1,  -1, 2,  3,  1,  2,  0,  1,  4,  3,  0,
    -1, -1, -1, 1,  1,  -2, 2,  -1, 2,  -2, -3, 2,  2,  0,   -2, -2, -1, 2,  2,  1,  0,  1,  -4, -1, 1,  0,   0,  1,   -1, 1,  2,  4,  1,  2,  2,  -1, 0,  1,
    1,  1,  -2, -1, 3,  -3, 0,  0,  0,  1,  2,  1,  -1, 0,   1,  0,  -4, 2,  1,  0,  -1, 3,  -2, 1,  1,  1,   2,  3,   1,  -1, -4, -1, -2, 0,  2,  5,  3,  1,
    5,  6,  1,  2,  0,  0,  1,  3,  3,  1,  -3, -4, -1, -2,  -1, 0,  -5, -7, -2, 0,  3,  0,  1,  0,  1,  -1,  -3, 4,   -1, -1, -1, 0,  1,  1,  3,  -2, 3,  2,
    1,  0,  -2, 1,  -1, 0,  1,  2,  3,  -1, 1,  2,  3,  2,   0,  -1, -1, 2,  2,  0,  1,  2,  4,  -3, -4, 4,   2,  1,   1,  -2, 0,  1,  3,  1,  0,  2,  -4, 0,
    -1, -1, 3,  0,  -1, 1,  2,  2,  2,  1,  3,  -1, -1, 2,   0,  2,  1,  -2, 1,  -3, 1,  1,  0,  0,  2,  2,   -1, 0,   1,  -3, -2, 5,  0,  2,  0,  2,  0,  0,
    2,  3,  2,  2,  2,  -5, -2, -2, -4, -4, 3,  6,  2,  -1,  3,  2,  4,  2,  3,  1,  0,  0,  3,  1,  -2, -5,  -4, -3,  -3, -4, -5, -9, -2, 0,  4,  2,  -1, 0,
    0,  1,  -1, 6,  -1, -1, -2, 3,  1,  1,  1,  1,  4,  3,   -1, 2,  1,  0,  -1, 0,  2,  1,  4,  -1, 0,  1,   4,  1,   -2, -2, -3, 3,  0,  -1, 1,  0,  2,  0,
    -1, 4,  2,  2,  0,  -3, -1, 1,  4,  0,  2,  4,  -3, 0,   0,  -1, 0,  1,  0,  1,  0,  1,  3,  3,  2,  0,   -1, 2,   -1, 1,  0,  -3, -1, -3, 1,  -1, -1, -1,
    0,  0,  1,  0,  3,  -2, -2, 2,  2,  2,  -1, 2,  -2, -1,  2,  3,  3,  2,  2,  -4, -2, -2, -5, -1, 1,  4,   2,  0,   4,  2,  3,  2,  2,  -1, 1,  1,  0,  2,
    -3, -5, -1, 0,  -3, -4, -5, -6, -4, 1,  4,  1,  -1, 1,   1,  1,  0,  2,  0,  1,  -1, 1,  4,  2,  1,  3,   1,  3,   1,  0,  2,  0,  0,  1,  1,  1,  4,  0,
    2,  2,  4,  4,  -1, -2, -1, 2,  1,  -1, 1,  0,  3,  -2,  -3, 2,  1,  -1, 0,  -4, 0,  2,  0,  1,  1,  0,   0,  2,   -2, -1, 1,  1,  -1, 0,  1,  2,  1,  2,
    4,  -1, 2,  4,  -1, 1,  1,  -3, -1, -2, 0,  -3, 3,  3,   0,  0,  0,  -1, 3,  0,  1,  2,  0,  1,  -1, 1,   -1, -2,  2,  0,  2,  2,  2,  -3, -1, -3, -4, -1,
    2,  5,  4,  1,  4,  2,  2,  0,  0,  -2, 2,  0,  0,  1,   -1, -5, 0,  0,  -2, -2, -3, -4, -1, 2,  4,  1,   1,  1,   1,  0,  -1, 3,  3,  1,  -1, 1,  2,  3,
    0,  1,  2,  4,  1,  0,  2,  0,  -1, 4,  2,  1,  4,  0,   1,  2,  3,  4,  1,  0,  -2, 0,  -1, -1, -2, -1,  1,  -1,  0,  3,  0,  -1, 0,  -2, -2, 2,  0,  0,
    1,  -1, 0,  3,  1,  -4, 1,  1,  -1, 0,  1,  1,  2,  2,   2,  -1, 0,  3,  -1, 1,  -2, -3, 1,  -1, 0,  -3,  2,  3,   0,  -1, 1,  -2, -1, 2,  2,  2,  -2, 1,
    2,  -1, -2, -1, 4,  1,  2,  4,  3,  -3, -1, -3, -7, -1,  1,  4,  3,  0,  2,  4,  3,  -2, 2,  -1, 1,  -1,  0,  2,   -2, -4, 0,  1,  0,  -1, -1, -5, 1,  0,
    -1, 3,  -1, 1,  0,  -3, 2,  4,  1,  -1, 1,  0,  1,  0,   2,  3,  -3, 3,  0,  0,  3,  -2, -2, -3, 2,  1,   1,  1,   3,  0,  0,  -2, 2,  1,  0,  0,  -1, -3,
    -3, 2,  1,  1,  1,  -1, 0,  2,  -1, -1, -1, -3, -3, 0,   -1, 0,  2,  3,  1,  1,  1,  -2, 0,  -1, 2,  0,   2,  4,   2,  -1, 0,  1,  5,  4,  2,  1,  2,  4,
    4,  1,  1,  0,  0,  1,  0,  -2, -4, 3,  1,  -1, -2, 0,   0,  2,  -2, 0,  0,  2,  3,  1,  1,  1,  3,  4,   -1, -3,  -2, 2,  1,  1,  -1, 0,  1,  -3, -3, 2,
    -1, 2,  1,  2,  2,  -1, 0,  0,  0,  0,  0,  3,  -1, -2,  0,  0,  1,  2,  0,  -2, 1,  2,  3,  -2, 1,  1,   2,  2,   1,  1,  -1, 2,  1,  -1, 2,  -3, 1,  -2,
    1,  1,  0,  1,  2,  0,  0,  0,  2,  2,  2,  1,  -1, -3,  -4, 5,  1,  0,  4,  0,  1,  0,  0,  -5, -2, -3,  0,  0,   1,  -2, 0,  4,  4,  1,  2,  0,  2,  1,
    1,  -1, 0,  2,  3,  1,  1,  2,  5,  2,  0,  5,  1,  4,   4,  3,  3,  -1, 1,  2,  -2, -3, -3, 3,  2,  2,   -1, 1,   0,  1,  -1, -1, 1,  0,  4,  1,  0,  1,
    2,  4,  3,  -3, -2, -3, -1, -1, 0,  0,  0,  0,  -1, -1,  0,  0,  5,  0,  1,  2,  3,  1,  1,  0,  -1, 1,   2,  -3,  -1, 1,  -1, 3,  1,  -1, 2,  2,  3,  0,
    1,  0,  1,  2,  0,  1,  1,  3,  1,  1,  1,  -3, 1,  -4,  3,  2,  1,  2,  2,  -2, 3,  -2, 0,  -1, 0,  0,   -3, -4,  1,  3,  1,  3,  3,  1,  0,  3,  -1, -3,
    -3, -2, -1, 0,  -1, 0,  -1, 1,  3,  3,  2,  0,  2,  1,   0,  -1, 0,  2,  2,  -1, 1,  2,  5,  2,  2,  2,   0,  4,   3,  2,  3,  -1, 0,  2,  -1, -2, -2, 3,
    2,  3,  -1, 0,  2,  1,  -2, -1, 0,  2,  4,  2,  -1, 2,   3,  4,  0,  -3, -1, 0,  1,  0,  1,  1,  1,  -1,  -2, 1,   -1, 2,  4,  -1, 1,  3,  3,  1,  1,  -1,
    -1, 2,  -1, -5, -1, 3,  -1, 4,  2,  -3, 1,  0,  1,  -1,  0,  0,  0,  2,  -2, 1,  1,  4,  -1, 1,  -1, -3,  0,  -4,  2,  1,  1,  0,  2,  -1, 1,  -3, 0,  -1,
    2,  0,  -6, -3, -1, 3,  0,  0,  1,  0,  1,  2,  1,  -3,  -1, -1, 1,  0,  -2, 1,  -2, 0,  0,  0,  1,  0,   2,  -1,  -1, -3, 2,  2,  1,  0,  1,  0,  6,  4,
    1,  2,  0,  2,  1,  3,  2,  1,  0,  0,  -2, -2, -2, 3,   3,  1,  -1, 0,  2,  0,  -2, -1, 1,  2,  4,  1,   -2, 2,   3,  1,  0,  -2, -3, -1, 1,  -1, 1,  -1,
    -1, 0,  -1, 0,  -2, -1, 2,  -1, -1, 0,  2,  -1, 0,  -2,  -3, 1,  1,  -7, 0,  2,  -1, 3,  4,  -2, -1, -2,  3,  0,   1,  0,  2,  3,  -1, 0,  2,  2,  2,  3,
    1,  -6, 1,  -5, 3,  1,  2,  3,  3,  0,  0,  -2, 1,  -1,  2,  0,  -5, -2, 1,  1,  -1, 0,  3,  0,  0,  1,   -1, -4,  0,  0,  0,  1,  0,  3,  1,  1,  2,  2,
    1,  2,  2,  0,  1,  0,  2,  3,  2,  2,  1,  0,  7,  5,   2,  0,  2,  5,  4,  2,  4,  1,  2,  0,  0,  -5,  1,  -1,  1,  3,  0,  -1, 3,  2,  -1, -2, 1,  2,
    3,  1,  1,  1,  0,  2,  2,  -1, -2, -3, 1,  0,  0,  1,   3,  2,  -1, 1,  1,  1,  3,  1,  0,  2,  3,  -1,  -1, -3,  -4, 1,  -3, -2, -3, -1, 0,  3,  0,  -2,
    1,  1,  3,  3,  3,  2,  1,  0,  -2, -3, -3, -4, 1,  1,   3,  1,  -1, 1,  0,  -1, -2, 2,  3,  3,  2,  4,   2,  -3,  0,  2,  2,  3,  1,  0,  -3, 1,  2,  2,
    -1, -3, -4, 4,  -2, -1, 2,  4,  5,  -4, -3, 3,  3,  2,   -1, 1,  1,  -2, 1,  3,  -2, 0,  0,  3,  -4, -1,  1,  1,   -2, 1,  -1, 0,  1,  -1, -1, -1, 0,  0,
    -2, 2,  0,  -2, 1,  2,  3,  4,  4,  -3, -5, 2,  2,  -1,  -2, -2, 2,  1,  -2, -5, 0,  3,  5,  2,  0,  3,   3,  1,   2,  -1, 2,  4,  0,  -1, 2,  1,  -3, -2,
    2,  1,  1,  -1, -2, -4, -5, 4,  -2, -2, -2, 3,  1,  -1,  0,  1,  5,  3,  4,  1,  2,  -2, -5, -3, -2, -4,  -2, 1,   0,  0,  1,  1,  0,  0,  1,  2,  4,  1,
    4,  5,  1,  -1, 1,  1,  1,  2,  0,  0,  -2, -1, 2,  3,   -1, -6, -3, 3,  1,  -1, 2,  5,  1,  -1, -5, 3,   2,  0,   -1, 2,  0,  -2, 3,  5,  -3, 0,  -2, 5,
    -1, 1,  2,  0,  1,  0,  0,  4,  1,  -1, -2, -1, 0,  0,   2,  4,  3,  -2, 2,  2,  3,  5,  3,  -3, -2, -2,  3,  -3,  -2, 0,  1,  3,  -4, -4, 3,  1,  7,  4,
    2,  3,  3,  4,  3,  -3, 2,  1,  -2, 0,  2,  1,  -4, -2,  0,  -1, 0,  -1, 0,  0,  -5, 0,  -3, -2, -5, 4,   1,  -2,  0,  4,  5,  2,  3,  0,  0,  -1, -3, -3,
    -2, -5, -1, -2, 0,  0,  1,  1,  0,  0,  -1, 1,  4,  3,   4,  3,  0,  0,  1,  0,  2,  3,  0,  0,  -2, 1,   4,  3,   -2, -6, -5, 5,  -1, 1,  3,  3,  -1, -3,
    -2, 5,  1,  1,  -1, 2,  0,  -3, 3,  5,  0,  0,  -3, 4,   -3, -1, -1, 2,  -2, 2,  -1, 4,  0,  3,  -2, -1,  1,  0,   1,  3,  2,  -1, 1,  2,  3,  1,  0,  -4,
    3,  -1, 1,  -2, -2, 0,  -1, -2, -6, -5, 2,  1,  3,  5,   4,  1,  3,  6,  4,  -2, 0,  3,  0,  -2, 3,  3,   0,  -3,  -5, 0,  2,  -1, -3, -2, -5, -1, -3, -4,
    -7, 3,  -2, 1,  -3, 2,  6,  5,  2,  -2, 1,  0,  -2, -5,  -1, -4, -2, -1, 0,  -2, 2,  0,  1,  1,  -2, -2,  1,  1,   4,  4,  -1, 0,  -1, 1,  1,  3,  2,  -1,
    -3, 0,  3,  2,  0,  -7, -2, 5,  -3, 1,  2,  3,  2,  -5,  -2, 3,  0,  2,  -1, 2,  0,  -1, 0,  8,  0,  3,   -2, 4,   1,  -2, 2,  1,  0,  3,  0,  -1, -2, 4,
    1,  -2, 1,  1,  3,  2,  1,  -3, 4,  4,  5,  3,  0,  -2,  2,  1,  3,  -2, -5, 0,  2,  -2, -2, -7, -1, 0,   3,  3,   2,  0,  3,  6,  4,  -3, 3,  1,  0,  -2,
    5,  2,  -1, -1, -3, 0,  2,  0,  -3, -4, -5, -3, -2, -5,  -6, 0,  -2, -1, -1, 0,  3,  4,  2,  -4, -1, 0,   1,  -4,  1,  -4, -2, 0,  1,  -1, -1, 1,  3,  2,
    -2, 0,  2,  2,  2,  1,  0,  0,  0,  4,  0,  3,  4,  -1,  -3, 2,  3,  -1, -1, -4, 1,  5,  -1, 0,  2,  2,   0,  -1,  -2, 4,  -1, 2,  0,  1,  0,  0,  2,  4,
    0,  2,  2,  4,  0,  -2, -2, 1,  -1, 2,  -2, 1,  -1, 1,   3,  -3, -1, -2, 3,  2,  0,  -2, 1,  2,  0,  1,   2,  2,   2,  0,  5,  -3, -2, 0,  4,  -2, -3, -6,
    1,  2,  3,  4,  3,  1,  3,  4,  4,  -1, 4,  3,  0,  -3,  1,  2,  -2, -3, -7, -3, 2,  -4, -3, -3, 4,  1,   -2, 0,   2,  5,  2,  -1, 4,  4,  3,  -2, -1, -1,
    -3, -1, 3,  2,  1,  0,  -2, 0,  3,  0,  1,  5,  -1, 4,   4,  -2, 0,  4,  4,  2,  -2, 2,  -1, 1,  0,  4,   1,  2,   0,  1,  3,  -1, -1, 0,  0,  -3, -3, 0,
    2,  -1, 0,  -2, 0,  -1, -2, 2,  1,  2,  0,  3,  2,  0,   3,  1,  0,  3,  2,  0,  0,  3,  -1, -1, 5,  1,   1,  2,   -2, 2,  -2, 0,  0,  2,  0,  2,  0,  -1,
    1,  1,  1,  3,  4,  1,  4,  2,  2,  2,  -1, 3,  4,  -1,  1,  1,  -2, 1,  3,  1,  0,  3,  3,  0,  -1, 1,   2,  3,   2,  -4, 0,  1,  -7, -2, -4, -2, -2, -8,
    4,  1,  -1, 0,  2,  4,  -1, -1, 4,  2,  3,  -1, 1,  2,   -4, -2, 2,  1,  -1, 0,  -2, -3, 1,  0,  2,  6,   -1, 3,   3,  -3, 1,  2,  0,  3,  -2, -1, -1, 2,
    0,  3,  2,  2,  2,  2,  1,  0,  0,  -1, 0,  -1, -4, -1,  1,  0,  0,  -2, -1, -1, 1,  1,  1,  1,  2,  1,   2,  1,   0,  3,  1,  1,  -1, 1,  1,  1,  -2, -3,
    4,  3,  -2, 0,  -3, 0,  -2, 1,  -1, -1, 0,  6,  1,  -1,  3,  -1, 1,  2,  4,  1,  3,  1,  0,  3,  2,  5,   3,  -2,  -1, 1,  -2, 4,  0,  1,  -1, 2,  3,  2,
    1,  -1, 2,  3,  2,  -3, 1,  1,  -5, -1, -3, -4, -2, -5,  3,  1,  -3, 0,  3,  3,  1,  1,  5,  2,  1,  -1,  1,  0,   -1, -1, 1,  2,  -1, 1,  0,  -5, 3,  1,
    2,  1,  -1, 2,  0,  -2, -1, 1,  3,  1,  -3, 0,  -2, 3,   2,  3,  5,  5,  0,  -1, 2,  0,  -1, 0,  1,  -2,  -2, 1,   -2, 0,  0,  -2, -2, 2,  1,  0,  3,  1,
    1,  2,  1,  -1, 1,  1,  0,  2,  1,  1,  1,  1,  -3, -1,  4,  1,  0,  2,  1,  2,  -3, -1, 1,  1,  2,  1,   -1, 0,   1,  -2, 0,  5,  4,  -1, 3,  2,  1,  4,
    2,  3,  1,  0,  -2, 1,  -1, 2,  1,  2,  2,  -1, 3,  1,   -2, 0,  1,  3,  3,  -1, 2,  2,  -4, 1,  -2, -2,  0,  -7,  3,  0,  0,  0,  2,  4,  0,  -1, 2,  1,
    2,  -2, 1,  0,  -3, -1, 2,  1,  -2, 0,  0,  -3, 1,  1,   0,  2,  1,  1,  1,  -3, 1,  3,  2,  1,  -3, -1,  -1, 4,   4,  2,  3,  2,  0,  -1, 0,  1,  -4, 0,
    -1, 2,  -2, 1,  1,  2,  -1, -2, -1, 1,  0,  3,  3,  1,   1,  2,  0,  0,  -1, 3,  1,  0,  -1, 1,  -1, 0,   -1, 0,   5,  2,  2,  -1, -1, 1,  -1, -2, 0,  0,
    1,  1,  2,  -1, 1,  -1, 0,  2,  5,  0,  4,  2,  0,  5,   1,  3,  3,  -3, 0,  1,  -1, 2,  1,  -1, 2,  1,   3,  1,   -1, 2,  0,  1,  2,  -1, 1,  2,  -4, 0,
    -2, -4, -4, -8, 1,  3,  0,  0,  1,  3,  0,  0,  1,  2,   1,  -1, -1, 4,  -3, 0,  1,  0,  -4, -1, 0,  -2,  -2, 3,   2,  -2, 0,  0,  -1, -3, 2,  2,  4,  1,
    -2, -2, 0,  2,  4,  1,  3,  0,  -1, -1, 0,  1,  0,  -1,  -3, 0,  -4, 0,  1,  3,  0,  -2, 0,  2,  2,  2,   3,  0,   1,  1,  0,  0,  -1, 3,  1,  1,  1,  1,
    -1, 2,  -1, 2,  1,  1,  2,  -1, 0,  -1, 0,  -2, -1, -1,  0,  3,  2,  -2, 1,  0,  2,  1,  7,  2,  3,  0,   2,  4,   2,  2,  2,  -2, -2, 1,  0,  2,  2,  0,
    1,  1,  4,  1,  -2, 0,  1,  2,  1,  -2, 0,  2,  -2, -1,  -2, -3, -4, -6, 3,  -4, -4, -1, 1,  0,  0,  1,   -3, -4,  2,  3,  0,  -2, 3,  2,  3,  3,  0,  -2,
    -3, 2,  1,  -3, 1,  0,  3,  0,  4,  -1, 1,  0,  -1, 0,   1,  2,  3,  0,  3,  2,  1,  2,  -1, 0,  3,  -2,  1,  -3,  1,  5,  -1, -4, -2, 2,  4,  1,  -1, 0,
    5,  3,  -1, -1, -2, 1,  0,  3,  3,  -2, 1,  0,  1,  -3,  1,  -4, -8, -1, -4, 2,  1,  -3, 4,  2,  5,  4,   1,  5,   0,  -1, 1,  1,  -3, -3, 2,  3,  3,  -1,
    1,  5,  -1, -1, -4, 2,  3,  0,  1,  -4, -1, 3,  -1, 2,   4,  4,  3,  1,  1,  -1, -2, -3, -1, -1, 3,  2,   -1, 2,   1,  6,  0,  0,  5,  0,  -2, 1,  -1, -5,
    -1, 2,  -3, -4, 3,  3,  -2, -5, 3,  0,  3,  -1, 1,  0,   1,  0,  0,  -2, 0,  1,  0,  0,  4,  -2, 0,  0,   -4, 0,   2,  2,  6,  1,  3,  2,  3,  5,  -1, -4,
    -3, -1, -1, 1,  2,  5,  0,  -1, 0,  0,  1,  3,  3,  -1,  1,  3,  0,  -2, -2, 1,  5,  2,  2,  -2, 4,  0,   1,  0,   -1, -1, -7, 1,  -3, 3,  1,  -1, 1,  3,
    4,  1,  2,  7,  2,  1,  2,  -1, -3, -1, -2, 3,  4,  -4,  2,  6,  0,  -1, -2, -1, 1,  0,  2,  -5, 2,  6,   3,  6,   1,  3,  1,  1,  1,  -1, -4, -2, -3, -3,
    3,  2,  -2, 2,  4,  5,  0,  -1, 6,  -4, -1, -2, 0,  -2,  -3, 2,  -2, -2, -1, 2,  -1, -3, 5,  1,  0,  -1,  2,  -2,  0,  0,  -1, -5, 1,  0,  4,  1,  2,  -1,
    -1, 0,  -3, -1, 1,  4,  5,  2,  4,  3,  3,  4,  -1, -3,  -2, -2, -2, -1, 1,  2,  2,  -1, -2, 0,  -2, 1,   0,  1,   3,  1,  0,  0,  -4, 1,  4,  1,  2,  -4,
    4,  0,  1,  2,  0,  1,  -5, 0,  -4, 0,  1,  0,  1,  1,   2,  0,  5,  4,  0,  -1, 0,  4,  0,  -2, 2,  0,   2,  -1,  1,  3,  3,  2,  -4, -3, 1,  1,  -2, -6,
    5,  3,  3,  4,  1,  2,  0,  -2, -3, -1, -4, -3, -4, -2,  3,  2,  0,  -2, 2,  3,  -1, 1,  1,  -1, 0,  -1,  -1, -3,  0,  5,  -1, -3, -2, -2, -1, 1,  5,  2,
    -1, -2, 3,  -3, 0,  -1, -2, -3, 1,  5,  6,  -3, 0,  -1,  4,  0,  -2, 1,  2,  3,  4,  3,  4,  3,  3,  3,   -2, -4,  1,  0,  -1, -1, 5,  1,  0,  0,  2,  -1,
    3,  3,  1,  0,  2,  0,  1,  -2, -3, 1,  2,  3,  1,  -2,  1,  0,  1,  2,  3,  0,  -2, -1, -6, 2,  -2, -2,  -1, 0,   4,  0,  2,  2,  1,  2,  2,  3,  -3, -2,
    1,  1,  4,  0,  4,  5,  3,  -1, -4, -3, 2,  0,  -1, -4,  4,  4,  1,  5,  0,  1,  3,  -1, -3, 2,  -5, -3,  -2, -2,  5,  2,  0,  1,  1,  1,  2,  0,  3,  -2,
    0,  1,  -2, -7, -2, 5,  0,  -3, 0,  0,  -4, 1,  1,  1,   0,  0,  3,  -2, 1,  0,  -3, -4, -1, 5,  4,  3,   0,  0,   1,  2,  -1, -1, 4,  2,  4,  0,  4,  4,
    3,  2,  1,  -3, 1,  -3, -2, -4, 2,  1,  -1, -2, 0,  0,   2,  3,  0,  -1, 0,  1,  2,  -1, 0,  -1, -2, 3,   0,  -3,  3,  3,  0,  2,  4,  -1, -5, -1, -5, 1,
    1,  -1, -1, 0,  2,  4,  1,  1,  4,  3,  1,  5,  0,  -1,  3,  -1, 1,  1,  3,  2,  0,  -3, -5, -2, 1,  2,   -2, -4,  1,  2,  0,  7,  2,  2,  5,  1,  0,  2,
    -4, 1,  -4, -3, 1,  4,  -1, 1,  0,  1,  2,  2,  -2, -3,  0,  -3, -4, 3,  -3, 2,  -2, -3, 1,  -1, 0,  1,   3,  0,   -1, 0,  0,  1,  -4, 2,  1,  0,  0,  0,
    4,  -1, -3, 2,  -3, -3, 1,  2,  1,  -2, 2,  -1, -2, 2,   0,  2,  -1, 0,  2,  0,  0,  2,  1,  1,  2,  3,   0,  1,   -1, 2,  7,  1,  0,  0,  5,  -1, -1, 1,
    -3, -1, 3,  0,  3,  2,  -1, 3,  3,  -1, -1, -1, 2,  2,   1,  3,  3,  2,  1,  -1, 6,  4,  3,  2,  -1, -2,  -3, 1,   -1, -1, 1,  2,  0,  -1, 1,  4,  2,  0,
    -3, 1,  -2, 3,  3,  5,  2,  -2, 3,  -1, -4, -4, 4,  3,   0,  4,  -2, -6, -2, -3, -3, -1, 1,  -1, -4, -4,  0,  -2,  0,  -3, -5, 3,  -5, 2,  -2, -1, 3,  -2,
    0,  -2, 5,  -2, 0,  0,  0,  0,  1,  4,  1,  1,  -2, 2,   1,  -2, -4, 1,  -4, -2, 2,  0,  -1, -3, 3,  0,   -2, -1,  -1, 4,  -3, 0,  3,  -1, -1, 1,  1,  2,
    3,  0,  1,  1,  1,  2,  6,  0,  1,  0,  2,  0,  1,  5,   0,  -1, 3,  0,  2,  0,  0,  1,  0,  -3, -1, -1,  2,  2,   1,  1,  4,  0,  -1, 1,  5,  3,  1,  1,
    -2, -2, -1, 0,  -1, -1, 2,  2,  -1, -3, -3, 3,  1,  1,   -2, 1,  -2, 1,  3,  4,  2,  -1, 1,  -2, -4, -2,  6,  1,   -1, 2,  -2, -2, -3, -1, -2, 1,  2,  0,
    -3, -2, 1,  -2, 1,  -2, -6, 3,  -3, 1,  -1, -3, 3,  -1,  0,  0,  4,  -1, 2,  -1, -1, 1,  0,  3,  -1, 1,   1,  0,   1,  -2, -3, 2,  -2, 0,  0,  -1, 0,  -1,
    4,  -1, -1, -1, -1, 4,  0,  0,  4,  0,  -1, 0,  2,  4,   4,  0,  0,  0,  2,  0,  4,  2,  0,  0,  3,  0,   1,  0,   1,  0,  3,  3,  1,  1,  1,  1,  -2, -2,
    2,  0,  -3, 0,  -1, 2,  3,  0,  0,  1,  4,  1,  -1, 0,   -2, -2, 1,  0,  3,  1,  3,  1,  -1, -3, -4, 0,   1,  2,   -2, 1,  1,  2,  5,  3,  1,  -1, 2,  -2,
    -2, -1, 5,  2,  2,  3,  -1, -2, -2, -3, 1,  1,  -1, 4,   -1, 0,  1,  -1, 2,  -3, -4, 2,  -2, 1,  0,  -3,  4,  0,   -4, 2,  4,  -1, 1,  0,  -1, 3,  0,  -1,
    -1, 2,  0,  -1, -1, -1, 0,  1,  -3, -4, 0,  0,  -2, -1,  2,  -1, 0,  -2, -3, 2,  2,  1,  2,  0,  0,  0,   2,  3,   1,  0,  0,  0,  4,  -1, 5,  4,  0,  -1,
    0,  0,  0,  -2, 4,  2,  4,  3,  -1, 0,  2,  -1, -2, -3,  2,  -1, -2, 0,  -2, 4,  4,  -1, 0,  -1, 1,  0,   -3, 3,   1,  -5, -1, 0,  0,  -3, 1,  4,  -1, -3,
    -2, 3,  0,  2,  -3, 1,  0,  1,  6,  3,  -1, 1,  2,  -2,  -3, -1, 5,  3,  -1, 5,  0,  2,  2,  -2, 0,  2,   0,  0,   0,  -1, 3,  -1, 0,  -5, -6, 3,  -1, 1,
    -1, -1, 3,  -1, -5, 1,  1,  -3, 3,  -2, 1,  3,  2,  4,   2,  2,  -3, 2,  -1, 0,  -2, 1,  -5, -2, 2,  0,   -1, -1,  2,  0,  0,  1,  -2, 2,  0,  1,  4,  1,
    0,  0,  3,  0,  4,  0,  0,  -2, 4,  -1, 5,  4,  -1, -1,  1,  1,  -1, -1, 3,  0,  4,  5,  0,  1,  1,  1,   -1, 0,   0,  -1, 0,  1,  1,  5,  6,  2,  1,  0,
    2,  1,  -1, 5,  -1, -1, 0,  1,  7,  -3, 0,  3,  0,  0,   -1, 1,  0,  5,  -2, -1, 2,  1,  3,  1,  -2, 0,   4,  -4,  -2, 3,  6,  3,  1,  4,  1,  1,  0,  -2,
    3,  3,  -2, -1, 0,  -1, 3,  -1, 2,  4,  0,  0,  -6, 0,   1,  -6, 2,  3,  0,  -1, 0,  3,  -2, 3,  0,  -4,  2,  2,   -2, 1,  0,  1,  0,  0,  0,  3,  6,  3,
    3,  0,  1,  3,  0,  3,  1,  -4, 0,  1,  2,  2,  4,  1,   1,  1,  0,  1,  1,  0,  -2, -2, 6,  1,  -1, 1,   2,  1,   -2, 1,  -1, 1,  0,  4,  -1, -2, -1, -1,
    2,  1,  3,  3,  0,  0,  -1, -2, -5, -4, -3, 2,  0,  -1,  1,  -2, 1,  -3, -1, 2,  6,  4,  0,  -2, 3,  3,   1,  -3,  -1, -2, -2, 0,  2,  -3, -1, -2, -1, 3,
    1,  2,  1,  3,  3,  2,  -4, -3, 2,  -1, 3,  4,  0,  -2,  3,  0,  -2, -4, 0,  2,  0,  0,  0,  2,  1,  0,   0,  1,   1,  -4, 3,  3,  3,  3,  0,  -1, 2,  2,
    3,  0,  1,  5,  -3, 3,  2,  2,  -1, -2, 1,  4,  3,  4,   3,  1,  0,  4,  1,  0,  1,  -1, 1,  2,  0,  2,   0,  2,   1,  0,  1,  1,  -1, 0,  -1, -2, 5,  3,
    0,  1,  3,  0,  -1, 2,  0,  1,  -2, 2,  1,  0,  -3, -2,  0,  0,  1,  3,  0,  1,  0,  -1, -2, -1, 3,  2,   0,  1,   2,  1,  3,  -3, 0,  4,  6,  3,  -3, 0,
    2,  2,  6,  1,  -5, -4, -3, 0,  3,  -2, -1, 0,  -2, 1,   -1, 1,  4,  2,  4,  2,  -3, 0,  4,  2,  2,  5,   4,  1,   2,  2,  -2, -3, 2,  1,  2,  1,  -1, 4,
    -3, -2, -4, -1, 0,  -5, 3,  6,  1,  4,  -3, 3,  2,  5,   0,  -4, -2, 3,  0,  -2, 2,  0,  1,  0,  0,  3,   3,  1,   3,  -1, 1,  3,  0,  -1, 3,  -2, -1, 2,
    -1, 1,  0,  2,  1,  1,  1,  0,  2,  2,  0,  0,  3,  3,   -1, 1,  0,  -1, -1, 0,  -1, -2, -2, -2, 1,  0,   0,  -6,  -2, 2,  0,  1,  -1, -1, -1, 0,  -3, -2,
    1,  4,  0,  2,  2,  2,  1,  -5, 0,  4,  6,  5,  -3, 3,   4,  4,  4,  -2, -3, -6, 1,  0,  -1, -2, 2,  -1,  -2, 2,   0,  1,  2,  0,  2,  4,  -1, 1,  2,  2,
    1,  2,  0,  -5, 0,  3,  -1, -4, -1, 2,  -1, -1, 0,  -1,  -2, -3, -3, 0,  -3, -2, 3,  3,  1,  2,  0,  1,   4,  2,   -1, -3, 1,  2,  1,  1,  2,  0,  0,  2,
    0,  2,  4,  1,  3,  4,  2,  -1, 0,  0,  1,  -3, 2,  -2,  0,  3,  0,  2,  2,  0,  2,  0,  -2, 0,  0,  1,   5,  2,   1,  2,  3,  3,  2,  1,  0,  -1, -2, 2,
    2,  -3, 2,  -4, 2,  3,  1,  0,  -1, 0,  -2, -2, 0,  -3,  4,  2,  1,  0,  0,  2,  -1, 0,  -2, 1,  4,  3,   -3, 3,   6,  3,  4,  1,  -1, -1, 0,  -2, -1, -1,
    0,  1,  -1, 3,  0,  0,  2,  0,  0,  4,  0,  0,  4,  4,   2,  1,  2,  1,  0,  -1, -5, -1, 2,  3,  2,  -3,  0,  3,   0,  -2, -5, 0,  -4, -5, 2,  5,  2,  2,
    -2, 3,  -2, 1,  2,  3,  4,  2,  2,  3,  -1, 3,  1,  2,   3,  1,  3,  1,  1,  3,  2,  3,  2,  -1, 1,  -1,  2,  1,   2,  2,  1,  4,  3,  3,  1,  0,  -1, 1,
    2,  0,  6,  4,  1,  2,  2,  0,  1,  0,  -1, 1,  -1, 1,   2,  -1, 2,  -1, 2,  2,  3,  -1, -2, 1,  0,  0,   -1, -4,  -1, 1,  2,  1,  1,  -2, 3,  -2, -5, 1,
    4,  5,  -1, 3,  2,  3,  3,  -2, -2, -5, -2, -1, -2, -1,  0,  2,  0,  -1, -1, 3,  3,  -2, 0,  5,  -2, 0,   4,  4,   4,  3,  0,  -1, 1,  -1, -4, -5, 0,  3,
    0,  0,  1,  -4, 1,  1,  2,  0,  -4, 1,  6,  0,  5,  1,   -1, 1,  4,  3,  2,  3,  3,  1,  1,  0,  -2, 0,   -2, 2,   2,  -1, -3, -2, 1,  2,  0,  2,  1,  3,
    0,  2,  2,  4,  1,  3,  -2, 1,  1,  0,  6,  7,  3,  1,   1,  -1, 4,  3,  0,  -1, -1, 3,  -1, 1,  0,  -3,  1,  2,   -2, 0,  1,  -1, -1, 1,  -3, 1,  -4, -4,
    0,  -1, 0,  3,  -1, 0,  2,  2,  2,  -1, 2,  -3, -1, -1,  -2, 1,  -2, 0,  0,  1,  -1, -3, -2, 1,  -5, -4,  -3, 2,   0,  0,  3,  3,  4,  0,  6,  1,  1,  -2,
    2,  -1, 3,  2,  5,  2,  -1, -9, -2, -1, -4, -5, -1, 0,   1,  0,  0,  -4, 3,  1,  2,  -1, -2, -1, 5,  2,   6,  -1,  -1, 0,  4,  3,  4,  4,  1,  2,  -1, 1,
    -3, 1,  -2, 2,  0,  -2, -3, 0,  -2, 4,  2,  4,  0,  1,   1,  3,  3,  2,  -1, 5,  -1, 1,  1,  0,  6,  8,   0,  2,   -1, 0,  2,  0,  0,  2,  -3, 5,  2,  0,
    1,  0,  2,  0,  -1, -2, 1,  1,  -3, -2, -3, 1,  -4, -2,  1,  -1, 1,  2,  -2, 2,  -1, 0,  1,  -3, 2,  -2,  -3, 0,   2,  3,  -1, -1, 0,  2,  1,  0,  -3, 3,
    -2, -4, -2, 1,  0,  2,  -1, 3,  1,  -2, 6,  3,  2,  -1,  4,  1,  1,  0,  6,  -1, 3,  -5, 0,  1,  -1, -1,  1,  3,   1,  2,  -1, -3, -2, 2,  2,  1,  -4, 3,
    1,  -3, 4,  -2, 0,  1,  4,  0,  0,  3,  3,  5,  -1, 0,   -2, 0,  -3, 3,  1,  -3, -3, 0,  -1, 2,  2,  3,   0,  4,   2,  2,  2,  6,  0,  1,  1,  1,  -2, 0,
    6,  5,  -1, 2,  1,  0,  1,  -1, -2, 1,  -1, 4,  2,  1,   -1, 1,  3,  2,  0,  -2, -2, 3,  -1, 2,  -6, -2,  -2, -5,  0,  -1, -2, 1,  -2, 0,  0,  1,  2,  -1,
    1,  -4, -1, 0,  4,  4,  -2, 1,  1,  3,  -1, -1, -3, 0,   -3, -2, -2, 0,  -4, 1,  -1, 3,  3,  0,  4,  2,   2,  1,   2,  -1, 4,  2,  5,  2,  3,  -5, 2,  2,
    -1, -2, 2,  2,  -1, 3,  -2, -3, 2,  1,  -1, 1,  0,  0,   2,  -2, 6,  -2, -1, 4,  5,  0,  1,  3,  2,  3,   -2, -1,  0,  0,  2,  3,  -1, -1, -4, -3, 1,  -1,
    3,  4,  -2, 4,  4,  2,  0,  6,  1,  2,  1,  2,  -2, -1,  5,  4,  2,  2,  2,  1,  4,  0,  -3, 1,  -1, 0,   0,  1,   0,  1,  0,  2,  2,  -3, -1, 1,  2,  3,
    -2, -2, 1,  -1, 1,  -3, -1, -3, 1,  -1, -2, 0,  3,  -2,  1,  -5, -2, -2, 0,  3,  1,  0,  1,  5,  1,  0,   -1, 0,   -4, -6, 1,  3,  -1, 1,  -1, -1, 2,  2,
    5,  1,  0,  2,  1,  -1, 1,  4,  5,  5,  2,  -3, 1,  0,   -4, 1,  4,  3,  -1, 1,  -3, -1, 1,  1,  -1, 2,   -3, 1,   0,  2,  3,  0,  -2, 5,  5,  -2, 0,  3,
    2,  4,  0,  0,  1,  0,  -1, 3,  1,  -2, -3, -4, -2, 2,   4,  4,  0,  1,  4,  -1, 2,  2,  1,  0,  2,  3,   0,  2,   6,  3,  2,  0,  -2, 1,  5,  0,  -1, 3,
    -2, 3,  4,  0,  0,  2,  0,  5,  3,  -4, 0,  0,  1,  1,   -2, -3, 1,  1,  1,  -4, 1,  1,  1,  -3, -2, 1,   3,  0,   0,  -3, -1, -3, -1, 2,  0,  -3, -2, 3,
    3,  1,  -3, -1, -5, -5, 4,  -1, -2, -3, -2, 2,  7,  3,   3,  4,  2,  1,  3,  -2, 2,  1,  4,  4,  2,  -2,  3,  -2,  -2, -2, 3,  4,  -1, 2,  1,  1,  3,  -1,
    -2, 0,  1,  -2, -3, -3, 1,  0,  -4, -2, 3,  2,  3,  0,   1,  4,  -1, 5,  -1, 3,  4,  -4, 3,  -1, -1, 2,   0,  2,   -4, -2, 5,  -4, 2,  5,  0,  0,  -5, 1,
    2,  3,  3,  1,  -3, 1,  3,  2,  2,  -2, -2, 2,  1,  1,   3,  2,  2,  1,  0,  -2, -3, 0,  3,  1,  1,  -4,  2,  -1,  3,  2,  1,  0,  -1, 3,  6,  -1, 4,  1,
    -1, -1, 0,  1,  4,  3,  0,  2,  0,  1,  2,  -3, 2,  3,   1,  -3, 0,  -1, 4,  3,  1,  6,  2,  0,  1,  1,   1,  3,   0,  0,  2,  2,  3,  -3, -5, -5, -5, 2,
    1,  1,  -1, 3,  0,  5,  2,  -4, 2,  4,  2,  -2, -1, -2,  -3, 0,  3,  -1, -2, -4, 1,  -2, -1, 0,  3,  3,   -1, 0,   0,  -2, 0,  2,  2,  2,  1,  -1, 2,  -1,
    1,  1,  2,  0,  1,  1,  3,  -2, -1, 3,  4,  -4, -5, 2,   3,  4,  0,  0,  1,  -1, 2,  2,  3,  -3, -7, 1,   1,  0,   1,  1,  2,  1,  0,  1,  -3, -1, 2,  -1,
    0,  -5, 0,  0,  4,  2,  0,  1,  1,  2,  3,  1,  4,  0,   -1, 2,  -2, 1,  0,  1,  -1, 2,  0,  3,  3,  -2,  2,  0,   -1, -4, 2,  1,  2,  5,  1,  5,  4,  1,
    0,  -1, 2,  5,  0,  1,  1,  2,  2,  -2, -5, -3, -6, 0,   4,  4,  2,  -2, 4,  3,  2,  -1, 0,  3,  2,  0,   1,  -1,  -4, 1,  1,  0,  2,  1,  0,  1,  -2, 0,
    1,  4,  -2, 0,  2,  0,  -2, 1,  -1, 1,  1,  -3, 2,  -1,  1,  -3, 2,  -1, -1, 1,  5,  -1, 2,  1,  1,  -4,  -4, 3,   0,  0,  -1, 1,  -1, 2,  0,  2,  4,  -1,
    -5, -1, 1,  2,  4,  3,  2,  0,  -3, 0,  -4, -1, 3,  -3,  -1, -3, 0,  3,  4,  3,  3,  -1, 1,  1,  4,  0,   4,  -1,  0,  0,  -1, 1,  0,  3,  2,  0,  1,  1,
    0,  -7, -1, 0,  -4, -1, 3,  1,  3,  5,  -1, 5,  5,  -1,  -1, 0,  3,  2,  -2, -1, 1,  0,  2,  -1, -6, -4,  -6, 2,   2,  3,  3,  -1, 5,  3,  -1, -3, 0,  4,
    4,  -2, 4,  1,  -3, 1,  2,  -3, 1,  0,  -1, -1, -2, 0,   3,  3,  -2, -1, 1,  2,  0,  4,  0,  2,  5,  -4,  -1, 3,   1,  -5, 2,  1,  -4, 0,  4,  2,  2,  5,
    2,  -6, 0,  4,  1,  2,  3,  2,  -2, -1, -1, 2,  4,  0,   -7, 1,  2,  0,  0,  2,  2,  1,  -1, -2, -3, -2,  5,  -2,  0,  -1, 1,  4,  3,  4,  1,  0,  -2, 2,
    3,  -1, 2,  0,  0,  -1, 2,  0,  -5, 4,  1,  -1, 1,  5,   0,  -7, -3, -1, 1,  -3, 3,  5,  6,  2,  -3, 4,   4,  1,   -1, 0,  5,  6,  0,  -3, 4,  2,  1,  -2,
    -5, -1, -4, -1, 4,  1,  0,  2,  4,  5,  -2, -1, 0,  6,   3,  -6, 2,  3,  -2, 1,  1,  -4, -2, 1,  -1, -2,  0,  -2,  3,  3,  -2, 1,  1,  3,  -1, 3,  3,  4,
    1,  -4, 0,  3,  1,  -4, 3,  2,  -4, 1,  4,  0,  2,  3,   3,  -5, -2, 4,  3,  1,  1,  2,  -3, -5, 1,  -2,  6,  -3,  -7, -2, 3,  1,  4,  1,  3,  1,  0,  0,
    -1, -1, 0,  0,  2,  0,  0,  2,  2,  3,  1,  -1, -2, 3,   4,  -1, 2,  1,  1,  0,  1,  1,  -2, 2,  4,  -3,  -1, -1,  -1, -9, 4,  2,  1,  -1, -1, 3,  5,  0,
    -3, 3,  5,  4,  2,  -1, 2,  5,  1,  1,  4,  1,  0,  -4,  -2, -3, -5, 4,  5,  3,  5,  6,  4,  5,  -2, -1,  2,  -3,  -2, 0,  3,  1,  -6, -5, -1, 1,  3,  -1,
    1,  1,  0,  3,  -2, -3, 3,  -1, 1,  4,  0,  2,  3,  1,   2,  -2, 0,  5,  -3, -2, -1, 1,  -2, 0,  -1, -1,  0,  5,   2,  -5, -2, -1, 1,  2,  -2, -2, -2, 4,
    0,  2,  1,  -2, 1,  -1, -4, -2, -2, 2,  2,  4,  0,  5,   -2, 3,  1,  1,  4,  0,  1,  2,  -1, 1,  3,  0,   0,  -2,  2,  1,  3,  0,  -2, 2,  5,  2,  0,  4,
    0,  2,  1,  5,  0,  0,  1,  -2, -3, 4,  2,  1,  1,  -2,  2,  -2, -1, 5,  -1, 0,  2,  3,  3,  3,  1,  -1,  1,  1,   -2, -1, 1,  -3, -2, 2,  -3, -3, -1, 0,
    -2, 3,  2,  -6, -5, 2,  4,  2,  -6, -4, -2, -3, 3,  2,   1,  0,  4,  3,  -2, -4, 2,  4,  2,  1,  0,  3,   2,  2,   0,  -2, 3,  2,  -4, -3, 0,  0,  2,  2,
    1,  2,  0,  3,  1,  0,  -1, -2, -2, 0,  1,  -4, -5, 3,   -3, 1,  -1, -1, -1, -2, -3, -1, -2, 1,  3,  5,   0,  5,   -3, 2,  2,  -1, 2,  -1, 6,  2,  2,  1,
    1,  1,  1,  -2, 1,  0,  5,  0,  -3, -2, 4,  6,  1,  5,   -2, 1,  2,  3,  3,  -1, -1, -6, -3, 1,  2,  -1,  2,  -3,  5,  2,  -2, 4,  -2, 1,  0,  3,  1,  0,
    0,  0,  2,  0,  -3, 1,  -1, -6, -4, 3,  -2, -3, -4, -2,  -2, -2, 1,  -2, -4, 1,  5,  -1, -4, -5, -4, -1,  1,  2,   6,  3,  3,  3,  -3, -1, 1,  2,  5,  1,
    -2, 5,  2,  2,  2,  -3, 4,  0,  0,  -2, 1,  -1, 3,  1,   -1, -2, 0,  0,  0,  0,  2,  -2, -3, 1,  0,  -4,  -2, 3,   -6, 1,  1,  -2, -3, -2, -1, -1, -3, 1,
    2,  4,  1,  3,  -3, 3,  4,  -1, 3,  1,  4,  0,  2,  1,   2,  -2, 1,  -3, -2, 1,  4,  2,  -3, -1, 0,  5,   -3, 5,   2,  0,  1,  2,  2,  0,  -2, -4, -4, 4,
    5,  1,  -1, -1, 7,  2,  -3, 0,  0,  2,  5,  2,  4,  0,   3,  0,  1,  2,  -1, 1,  0,  -6, -3, 3,  -2, -2,  -1, -3,  -2, -1, 4,  -1, -5, -2, 2,  1,  -5, -3,
    -3, -1, 2,  3,  6,  1,  1,  2,  -4, 0,  0,  3,  5,  1,   0,  4,  3,  1,  3,  -2, 1,  -1, -1, -1, 2,  -3,  2,  2,   0,  -1, 1,  1,  -2, -2, 1,  -1, 0,  0,
    3,  -2, -5, 1,  -6, 4,  1,  -2, -4, -2, -2, 1,  -3, 4,   2,  2,  0,  6,  -2, 4,  3,  -1, 2,  0,  1,  -1,  0,  3,   2,  -1, 1,  -4, -1, -3, 4,  -1, -6, -2,
    0,  7,  -1, 3,  1,  -2, 1,  4,  5,  0,  -2, -3, -5, 5,   7,  1,  -1, 2,  7,  4,  -3, 2,  3,  2,  4,  2,   2,  2,   1,  0,  1,  3,  0,  1,  0,  -1, -2, 1,
    -4, -3, -3, 0,  1,  1,  1,  -2, -5, 0,  1,  1,  0,  -4,  -1, 2,  2,  2,  1,  1,  1,  3,  -2, 0,  1,  3,   4,  2,   3,  5,  3,  0,  7,  0,  3,  -1, -2, 0,
    2,  -1, 2,  0,  -1, 1,  1,  -1, 0,  1,  -1, -3, 0,  2,   1,  -3, -4, 1,  -6, 0,  -1, 0,  -4, 1,  -4, 0,   -2, 1,   1,  1,  0,  2,  -2, 2,  3,  4,  1,  0,
    5,  4,  4,  4,  0,  0,  4,  -1, -4, -1, 1,  -3, -4, 0,   2,  9,  2,  2,  1,  -3, 1,  3,  5,  0,  -1, -3,  -4, 4,   1,  3,  4,  2,  6,  3,  -2, 1,  4,  0,
    0,  1,  2,  5,  1,  -2, 4,  4,  1,  1,  2,  -4, 0,  1,   -5, -2, -3, 0,  4,  0,  -1, -4, 0,  5,  -4, -2,  -1, 5,   2,  -3, 1,  0,  1,  2,  -1, 3,  -1, 4,
    1,  3,  2,  0,  3,  7,  -2, -3, -1, 0,  7,  -4, 0,  -5,  -1, 3,  7,  -2, 0,  4,  -3, 0,  1,  1,  2,  0,   1,  0,   1,  0,  2,  3,  -2, -2, -1, 1,  -3, -3,
    -2, 0,  -1, 1,  -1, 0,  2,  3,  3,  4,  1,  0,  -2, -4,  1,  2,  2,  1,  -1, 0,  2,  1,  -3, 0,  0,  -2,  -3, -1,  3,  1,  4,  -2, -3, -2, 0,  -4, 0,  4,
    3,  3,  3,  4,  2,  1,  0,  3,  1,  3,  -6, -1, 2,  3,   3,  2,  1,  -2, 3,  -1, 0,  3,  1,  2,  0,  -1,  0,  0,   -2, 2,  3,  2,  3,  4,  -2, -5, -2, 1,
    -3, 0,  3,  5,  2,  -5, 0,  -1, 2,  1,  -2, 1,  -1, 4,   -1, 4,  3,  0,  1,  4,  3,  -3, 1,  -1, 5,  -1,  -3, -5,  -3, 1,  3,  -1, 2,  7,  -2, 0,  -1, 3,
    1,  1,  0,  0,  3,  2,  2,  1,  2,  -1, -1, 0,  -1, -4,  -2, -1, -2, 0,  -2, -1, 2,  3,  2,  2,  2,  0,   0,  -2,  2,  3,  3,  -2, 1,  2,  2,  1,  -3, -1,
    1,  -4, -3, 1,  2,  -1, 4,  1,  -1, 1,  -1, -5, -1, 5,   2,  2,  5,  3,  1,  2,  0,  -2, -1, 2,  -3, -3,  0,  4,   1,  2,  2,  0,  1,  -2, -3, 4,  0,  2,
    0,  2,  -1, -1, -1, 1,  1,  -1, 2,  2,  -3, -3, 0,  3,   -1, -4, 2,  0,  4,  -6, -1, 0,  2,  -3, -7, 5,   -3, 2,   -1, 2,  0,  1,  6,  6,  4,  -2, 2,  -2,
    3,  0,  -4, -3, -4, 0,  4,  -2, 0,  1,  -2, 1,  3,  3,   1,  1,  1,  0,  3,  -1, -1, 3,  2,  -1, 1,  0,   1,  -3,  2,  -2, -2, 2,  -1, -1, 1,  1,  0,  4,
    -1, 0,  0,  0,  4,  2,  3,  3,  -2, -1, 2,  -2, 0,  1,   -2, -4, 0,  -2, 1,  -4, 2,  -2, -3, 0,  -1, -3,  1,  3,   3,  1,  4,  5,  2,  1,  3,  0,  -2, 3,
    -3, -4, 4,  2,  3,  -1, 1,  4,  3,  -1, -3, 3,  -1, 1,   -1, 0,  1,  0,  -2, 3,  3,  0,  1,  1,  -3, 0,   2,  2,   -2, -3, 4,  0,  2,  -3, -6, -1, 0,  -3,
    -5, 5,  1,  2,  2,  2,  3,  2,  5,  6,  2,  1,  -1, 1,   2,  -3, -4, -2, -2, -1, 6,  -3, 3,  -1, -1, 4,   1,  1,   2,  0,  -1, -1, 3,  0,  -2, 4,  1,  -1,
    1,  -2, -1, -1, 2,  0,  -1, 0,  -1, -1, -1, 3,  4,  4,   -2, -1, 2,  1,  0,  1,  1,  0,  0,  4,  2,  -1,  0,  0,   -4, -4, 0,  -3, 0,  -2, 4,  2,  -3, -3,
    0,  -5, -1, 5,  4,  1,  4,  4,  2,  2,  2,  -4, -4, 1,   -5, -5, 5,  3,  0,  1,  5,  4,  1,  -2, -1, 4,   0,  -1,  0,  -1, 2,  0,  -3, 1,  2,  -2, 1,  4,
    -1, -3, 1,  6,  1,  -4, 2,  5,  3,  -3, -4, 1,  3,  -6,  -2, 4,  3,  3,  4,  2,  6,  1,  5,  8,  2,  1,   -2, 5,   3,  -2, -4, -1, -2, 2,  5,  -1, 1,  1,
    0,  1,  2,  3,  0,  -2, 0,  3,  2,  1,  1,  2,  0,  -2,  1,  -1, 2,  1,  3,  2,  0,  3,  2,  1,  -1, 1,   2,  1,   -2, -1, -1, -2, 1,  1,  1,  1,  -2, 3,
    1,  2,  1,  -2, -1, -3, 1,  1,  -1, -4, 3,  1,  -3, -1,  1,  -2, 2,  4,  4,  4,  2,  3,  -1, 0,  2,  -3,  -3, 3,   -1, -1, 5,  1,  1,  -1, 0,  4,  2,  0,
    2,  3,  -1, 1,  -1, -3, 2,  -3, 0,  2,  3,  -1, -2, 3,   2,  1,  1,  2,  3,  -2, 2,  0,  -4, 4,  -1, -2,  -1, 0,   -2, 1,  1,  0,  0,  -1, 2,  -1, -1, 4,
    2,  -1, 3,  0,  0,  1,  -4, 2,  1,  0,  1,  -2, 1,  -1,  0,  0,  0,  0,  0,  -4, -1, 0,  1,  0,  -2, -2,  -4, 1,   -1, 1,  -1, -1, 0,  0,  -1, 2,  2,  1,
    -1, 1,  0,  2,  -1, -1, 4,  3,  0,  0,  -1, -1, 1,  2,   -2, 2,  0,  -2, 1,  0,  0,  -3, -2, 4,  1,  1,   3,  2,   1,  0,  1,  -4, -1, 2,  1,  3,  -3, 0,
    0,  0,  1,  3,  2,  -2, -1, -1, 0,  4,  1,  1,  -1, 3,   3,  -2, 0,  3,  1,  1,  -1, 0,  2,  0,  2,  -1,  3,  1,   0,  0,  1,  1,  1,  2,  2,  0,  -1, 2,
    3,  2,  0,  2,  3,  1,  4,  2,  0,  2,  0,  -4, 0,  3,   1,  0,  2,  3,  2,  2,  0,  1,  -1, 3,  5,  1,   0,  0,   3,  1,  1,  1,  1,  1,  2,  -1, 2,  0,
    -1, 2,  -3, -2, 3,  0,  3,  -2, 2,  3,  2,  6,  2,  1,   1,  2,  2,  1,  -3, 2,  3,  3,  2,  1,  0,  0,   0,  1,   1,  2,  0,  -1, -3, 0,  -2, -4, -1, 1,
    2,  1,  2,  0,  -1, 0,  0,  1,  1,  1,  2,  2,  -1, 5,   -3, -2, 3,  3,  0,  0,  1,  2,  -1, 2,  2,  2,   2,  4,   -1, -1, 0,  3,  2,  5,  3,  3,  2,  2,
    4,  1,  7,  4,  0,  2,  0,  1,  0,  -1, 1,  2,  -1, 2,   0,  -1, -1, -3, -1, -3, 1,  2,  0,  -2, -3, -3,  -1, 4,   -3, -4, 0,  1,  -1, -3, -1, 0,  0,  -1,
    0,  -3, -3, 2,  0,  -1, -2, 0,  1,  -5, -3, 0,  1,  -3,  -2, 0,  -1, 0,  -1, -1, -1, 0,  -2, -3, -4, -3,  0,  -1,  0,  -3, -1, -1, -5, 0,  -1, 0,  1,  -3,
    -4, -2, 2,  -3, 0,  -1, -1, 0,  -5, 0,  0,  -1, -2, 3,   2,  -1, 0,  0,  0,  0,  1,  -2, -1, -2, -5, 0,   0,  -2,  0,  0,  1,  0,  0,  -1, -1, 0,  -2, -3,
    -1, -1, -2, 1,  1,  -3, -2, 0,  -2, -3, -6, -5, 0,  -2,  -1, -3, -2, -1, 0,  0,  4,  0,  -2, 1,  1,  -1,  0,  4,   -1, 3,  1,  6,  2,  3,  1,  1,  3,  1,
    2,  2,  1,  4,  2,  1,  1,  1,  1,  2,  1,  2,  -1, 1,   4,  4,  3,  1,  3,  3,  1,  0,  2,  0,  5,  1,   4,  1,   1,  3,  0,  1,  2,  3,  3,  -1, 2,  4,
    2,  7,  1,  0,  0,  2,  3,  1,  2,  4,  5,  5,  6,  2,   -4, 2,  2,  0,  2,  2,  0,  1,  -3, -1, -2, -4,  0,  3,   2,  5,  1,  0,  1,  0,  3,  0,  2,  4,
    3,  1,  -1, 2,  2,  0,  1,  4,  3,  2,  -1, 2,  3,  2,   3,  2,  1,  4,  3,  -1, 2,  3,  3,  4,  3,  2,   3,  1,   4,  1,  6,  2,  2,  2,  1,  0,  -1, 1,
    0,  0,  1,  1,  0,  1,  -1, -1, -1, 0,  0,  1,  0,  1,   -1, -2, 1,  1,  0,  -3, 1,  3,  1,  -2, 2,  1,   0,  1,   2,  1,  -1, 2,  0,  -2, 1,  -3, 0,  0,
    1,  1,  0,  -3, -1, -1, -2, 1,  -1, 1,  -1, -1, -1, -2,  -2, 0,  1,  0,  0,  -1, 2,  2,  0,  0,  2,  0,   0,  -2,  -3, 0,  -2, -1, 2,  -1, 0,  1,  -2, 0,
    0,  -1, -1, 3,  1,  1,  2,  1,  0,  1,  1,  -1, 0,  0,   -3, -1, 1,  1,  1,  1,  0,  1,  1,  0,  1,  -1,  2,  0,   0,  2,  1,  3,  1,  -3, 0,  2,  1,  2,
    -1, -3, 3,  0,  1,  -1, 2,  1,  -1, 3,  -1, -2, 2,  1,   3,  0,  0,  -1, 1,  2,  -1, 0,  3,  2,  0,  4,   2,  -2,  3,  3,  1,  1,  3,  3,  1,  1,  2,  -2,
    -1, 1,  2,  -2, 1,  2,  0,  2,  0,  2,  5,  -1, -3, -3,  0,  -1, 1,  -2, 1,  -1, -1, 1,  2,  0,  1,  0,   -3, 1,   3,  -1, -4, 2,  2,  3,  1,  1,  1,  2,
    7,  -1, 2,  4,  2,  -1, -3, -1, 2,  2,  3,  0,  2,  0,   -3, 2,  2,  -2, -1, 0,  -1, 0,  -2, -2, 0,  3,   -3, -2,  1,  5,  5,  3,  -2, -2, -6, -2, 2,  6,
    3,  2,  5,  4,  3,  0,  -3, 1,  3,  -3, 0,  0,  1,  4,   -2, -1, 2,  -7, -1, 2,  6,  5,  3,  2,  -1, -2,  -2, 1,   2,  0,  -2, 2,  0,  1,  -4, 0,  4,  6,
    0,  3,  1,  -1, 2,  3,  1,  2,  1,  1,  1,  2,  2,  -5,  -2, 3,  0,  -2, 2,  2,  0,  1,  2,  4,  1,  -3,  -3, -4,  2,  -1, 1,  -2, -2, -2, 0,  3,  3,  0,
    -2, 1,  -4, -2, 0,  0,  -1, 2,  3,  3,  1,  2,  3,  2,   4,  3,  3,  2,  4,  -1, -3, 0,  4,  1,  0,  -2,  -1, 0,   -3, 3,  1,  -1, -3, -3, -2, 1,  2,  -2,
    1,  4,  -6, 0,  1,  6,  2,  5,  -2, -3, -6, -3, -1, 6,   4,  5,  6,  5,  5,  -1, -2, 2,  4,  -1, 0,  -4,  0,  4,   0,  2,  2,  -5, -1, 0,  5,  4,  1,  3,
    1,  -1, -1, -1, 4,  1,  -1, 1,  -2, 0,  -4, -1, 2,  5,   2,  5,  -1, 1,  0,  1,  1,  1,  0,  3,  1,  1,   5,  -4,  -2, 2,  2,  -1, -1, 3,  1,  1,  0,  2,
    -1, -4, -3, -4, 4,  -2, 0,  -4, -2, -2, 1,  -1, 4,  -1,  -2, -3, -2, -1, -1, 2,  -2, 2,  2,  2,  2,  6,   1,  4,   4,  2,  3,  5,  5,  0,  -2, 1,  2,  1,
    1,  1,  1,  -1, -1, 5,  2,  -2, -2, -2, 0,  -4, 1,  -1,  -1, 0,  -3, 1,  0,  3,  1,  5,  -1, -2, -5, -1,  1,  6,   7,  1,  4,  8,  3,  -3, -3, 2,  2,  0,
    -1, -1, 3,  4,  0,  3,  3,  -5, -1, -1, 5,  2,  2,  5,   0,  -1, -1, 3,  1,  1,  1,  0,  0,  1,  -4, 1,   3,  3,   -1, 2,  2,  -1, 0,  4,  1,  1,  2,  2,
    2,  3,  6,  -4, 0,  1,  1,  1,  0,  0,  2,  -3, 0,  4,   1,  -3, -2, -2, 1,  1,  -2, -2, 0,  1,  -2, 2,   2,  -2,  -1, -2, -2, -4, -3, 2,  -1, 4,  2,  2,
    0,  6,  1,  2,  3,  4,  3,  6,  6,  0,  -2, 0,  2,  1,   -1, 2,  0,  -1, 1,  2,  -1, -2, 0,  -4, -2, 1,   2,  -1,  0,  1,  -3, 3,  4,  3,  2,  4,  -1, -4,
    -5, 0,  -1, 6,  4,  2,  5,  7,  4,  -3, -1, 3,  3,  -2,  1,  -2, 1,  1,  -1, 3,  3,  -3, -1, -1, 4,  6,   1,  8,   -2, -2, -3, -3, 2,  2,  -3, 4,  0,  1,
    -3, 0,  4,  -1, -3, 0,  3,  1,  1,  2,  2,  -1, 1,  2,   -1, 2,  6,  -3, -1, 3,  2,  0,  -2, 0,  0,  0,   0,  3,   1,  -3, -1, 0,  1,  -2, -4, -4, -1, 1,
    -2, 0,  1,  1,  -2, -3, 2,  -1, 1,  -1, -2, 5,  3,  3,   -2, 5,  3,  1,  2,  2,  -2, 3,  3,  -1, -4, 2,   1,  3,   3,  3,  -1, -1, 2,  1,  1,  -1, 3,  -3,
    -1, -1, 2,  -1, -1, 1,  -7, 4,  1,  2,  2,  2,  -2, -5,  -5, -4, -1, 7,  4,  3,  4,  7,  7,  -1, -2, 4,   1,  -2,  1,  -3, 2,  3,  -2, -2, 2,  -4, -1, 0,
    4,  5,  0,  0,  0,  -1, 2,  1,  1,  0,  1,  4,  0,  -1,  1,  2,  -3, -2, 1,  2,  1,  1,  0,  0,  -1, 0,   7,  3,   2,  -2, -3, -1, 1,  1,  2,  3,  -2, 1,
    4,  1,  1,  -2, 2,  0,  -2, -1, -2, 0,  -1, 2,  0,  0,   0,  0,  2,  0,  2,  3,  2,  3,  -2, 1,  0,  1,   0,  0,   0,  2,  -2, -2, 0,  0,  0,  4,  2,  -2,
    1,  1,  0,  -1, -3, -2, -1, 3,  -2, -3, -4, 0,  2,  -4,  1,  -1, 0,  3,  -1, 1,  -1, -1, 0,  0,  -1, 0,   -3, 0,   -1, -1, 3,  -3, -3, -1, -5, 2,  3,  3,
    1,  1,  1,  2,  3,  4,  1,  1,  1,  3,  1,  1,  -1, -1,  4,  0,  1,  2,  0,  0,  1,  -1, -2, 2,  -1, 1,   0,  -2,  2,  1,  -1, 1,  -1, -1, 0,  2,  1,  -1,
    -1, 0,  5,  4,  1,  0,  -1, 0,  -1, -1, 2,  2,  -1, 3,   5,  2,  2,  -1, 0,  0,  1,  -2, -1, 0,  -1, 3,   1,  2,   0,  1,  2,  1,  3,  4,  -1, 3,  -1, 3,
    0,  0,  -1, 1,  0,  3,  -1, -3, 1,  -1, 1,  3,  1,  -2,  -1, 3,  -3, 0,  -3, -1, -2, 1,  -1, -4, -5, 3,   0,  0,   -2, -3, -1, 2,  1,  -1, 2,  1,  2,  1,
    0,  0,  0,  1,  -1, -2, 2,  -2, -2, -1, -4, 0,  2,  4,   0,  3,  3,  -1, 0,  4,  0,  0,  2,  2,  2,  0,   -1, 0,   3,  2,  3,  1,  1,  0,  -2, -1, 1,  -1,
    0,  3,  1,  1,  2,  0,  -2, 3,  2,  -1, 1,  2,  2,  -1,  0,  2,  7,  4,  2,  1,  2,  1,  -1, 3,  3,  5,   3,  2,   4,  2,  3,  0,  2,  2,  -1, 0,  1,  0,
    1,  2,  1,  5,  3,  3,  3,  1,  2,  3,  1,  5,  1,  6,   0,  1,  0,  3,  1,  4,  0,  -2, 3,  2,  3,  4,   1,  -1,  0,  1,  1,  3,  -2, -2, 0,  1,  -1, -3,
    -6, 3,  1,  -2, 0,  -1, 0,  0,  0,  0,  3,  2,  2,  3,   -3, 1,  0,  2,  -2, 1,  1,  0,  -2, 0,  -3, 1,   3,  6,   3,  3,  3,  -1, 1,  5,  1,  1,  0,  3,
    3,  -1, 1,  0,  4,  3,  1,  3,  0,  0,  -1, -1, -1, 2,   0,  2,  1,  -1, 2,  -1, -2, 0,  0,  -2, 0,  1,   0,  -4,  -2, 2,  5,  1,  0,  0,  -2, -2, 0,  1,
    2,  1,  1,  0,  1,  2,  2,  0,  2,  -1, -2, -1, 0,  0,   -2, 1,  1,  3,  3,  1,  2,  3,  1,  3,  0,  4,   1,  2,   0,  0,  0,  0,  -2, 2,  -2, -3, 1,  -1,
    1,  3,  0,  -2, 0,  2,  3,  3,  -3, -3, -3, 2,  1,  -1,  -3, 1,  -3, -3, 0,  -2, -2, 1,  0,  2,  2,  0,   -1, 0,   -2, -1, 0,  1,  -2, -3, 0,  2,  0,  -1,
    -4, 0,  3,  4,  0,  4,  0,  -2, 2,  4,  1,  -1, -4, 0,   1,  -2, 0,  -1, 1,  0,  3,  3,  -1, 1,  -1, -2,  -3, 2,   1,  2,  2,  0,  2,  1,  -1, 1,  -4, -1,
    0,  1,  -1, -3, -1, -2, 5,  3,  0,  -1, 0,  -2, 0,  2,   3,  1,  1,  3,  2,  0,  -1, 1,  1,  0,  -1, 2,   2,  2,   -2, 4,  3,  0,  5,  2,  0,  2,  1,  4,
    3,  1,  2,  1,  0,  1,  0,  0,  1,  2,  -2, -2, -1, -1,  1,  3,  0,  0,  1,  1,  5,  2,  -2, -4, -3, 1,   0,  -4,  -4, -2, -1, -1, -2, -5, -1, 4,  2,  -2,
    -1, 0,  0,  1,  -1, -1, -1, 2,  -1, -1, 0,  -2, -3, 0,   -2, 1,  1,  3,  3,  3,  1,  -1, 2,  3,  0,  0,   -1, 0,   1,  0,  1,  -1, 2,  3,  -3, 1,  1,  0,
    0,  2,  -2, 0,  2,  0,  3,  -1, 0,  2,  3,  1,  1,  1,   1,  -2, 2,  4,  3,  -2, 1,  -4, 1,  3,  2,  -1,  0,  1,   0,  -1, -1, 0,  0,  0,  2,  3,  1,  1,
    2,  1,  3,  2,  0,  2,  2,  0,  0,  -1, 0,  -1, 0,  0,   3,  -2, -1, -3, -1, -1, -1, 1,  -2, -3, -2, -1,  1,  1,   0,  2,  1,  1,  1,  1,  -2, 3,  0,  1,
    2,  1,  1,  -3, -2, 2,  1,  2,  -1, 1,  1,  2,  3,  2,   1,  3,  -1, 0,  -2, 0,  1,  0,  0,  2,  2,  3,   0,  0,   -2, -1, 3,  -2, -1, 3,  4,  -2, -2, 2,
    9,  6,  -2, -1, 0,  0,  -2, -1, 1,  0,  -1, 2,  1,  0,   -1, -2, -1, 2,  2,  1,  0,  0,  2,  3,  3,  1,   0,  0,   1,  -1, 2,  4,  4,  -3, 1,  -6, 0,  2,
    0,  1,  2,  1,  1,  1,  -2, -2, -1, 1,  1,  4,  4,  0,   -1, 0,  5,  2,  -1, 3,  3,  1,  -3, -2, 0,  -3,  -1, -3,  0,  -1, -1, -3, 0,  0,  -1, 0,  -2, -2,
    -1, 0,  1,  1,  -1, 3,  0,  2,  2,  -1, 0,  3,  -2, 1,   2,  1,  3,  -2, -3, 1,  1,  3,  -4, 2,  3,  0,   3,  2,   3,  4,  -1, 0,  -2, -2, -2, 0,  1,  0,
    0,  4,  0,  -2, -5, -1, 1,  -2, 0,  3,  2,  0,  -1, 3,   6,  5,  -2, -3, 0,  -1, 0,  0,  2,  0,  -2, 1,   1,  0,   -1, -1, 0,  1,  3,  3,  0,  0,  0,  2,
    2,  1,  0,  -1, 0,  1,  3,  6,  2,  -1, 3,  -4, 1,  1,   2,  1,  -1, 0,  1,  2,  -2, 1,  1,  -1, 1,  3,   4,  2,   0,  2,  5,  2,  2,  3,  0,  4,  -2, -1,
    0,  -3, -1, -1, -1, 2,  1,  2,  -1, 0,  1,  0,  -1, -4,  -1, -2, 3,  2,  1,  1,  1,  0,  1,  0,  2,  1,   0,  2,   3,  3,  4,  -4, -3, 2,  3,  5,  -2, 2,
    3,  1,  4,  1,  2,  3,  1,  0,  -2, -1, -3, 2,  1,  -1,  1,  5,  1,  0,  -2, 0,  0,  -2, 1,  3,  5,  1,   -1, 3,   7,  5,  0,  -3, 2,  2,  -2, 1,  2,  0,
    -2, 2,  0,  0,  0,  2,  0,  1,  1,  1,  0,  -1, 1,  1,   3,  0,  0,  -1, 0,  2,  3,  4,  3,  -2, 1,  -4,  1,  -1,  3,  -1, 1,  1,  2,  1,  -4, 0,  -1, -1,
    2,  2,  3,  0,  -2, -1, 5,  3,  0,  3,  0,  3,  -1, 0,   -1, -3, -1, 0,  -2, -1, -1, -2, -2, 0,  0,  -2,  -2, -4,  -2, 0,  0,  0,  1,  3,  1,  -1, -1, 2,
    1,  0,  -1, 2,  2,  2,  2,  0,  -3, -1, 2,  1,  -1, 2,   2,  1,  0,  1,  1,  6,  -3, -2, -4, 2,  -3, 1,   0,  0,   1,  3,  1,  -2, -4, -1, 1,  -1, 0,  5,
    4,  0,  -1, 1,  6,  3,  -2, -3, 3,  1,  0,  -3, 0,  0,   -1, 0,  1,  1,  -3, 0,  1,  2,  -1, 2,  -1, -1,  0,  2,   2,  3,  0,  0,  -1, 0,  3,  5,  3,  0,
    1,  -3, -1, 1,  3,  -1, 2,  0,  1,  1,  0,  0,  2,  0,   1,  3,  5,  1,  -2, 1,  6,  1,  0,  1,  1,  0,   -2, -1,  0,  -3, 0,  0,  -2, -2, 0,  -1, -1, 0,
    0,  1,  -2, -2, 1,  0,  -1, 2,  1,  1,  0,  -1, -1, 2,   2,  2,  -2, 0,  1,  -1, 3,  -2, -3, 1,  -1, 2,   -1, 0,   3,  1,  2,  2,  1,  5,  -3, -1, -3, 0,
    -1, 2,  1,  1,  0,  2,  -1, -3, -1, -2, 0,  -1, 0,  4,   4,  0,  -1, 3,  8,  5,  0,  -1, 1,  2,  1,  0,   2,  1,   0,  1,  3,  3,  0,  -1, 0,  3,  1,  2,
    1,  2,  -3, 1,  3,  5,  0,  -2, -1, -1, 1,  -2, 5,  3,   5,  0,  -6, -1, 3,  -1, 3,  -2, 2,  1,  0,  -4,  0,  -3,  -2, 3,  -3, 4,  2,  -2, 4,  0,  0,  -2,
    1,  1,  1,  2,  -1, 0,  -1, 1,  3,  -3, -1, -1, 1,  0,   0,  1,  2,  -1, 0,  3,  0,  -3, 0,  1,  0,  -1,  0,  4,   -3, -1, -2, -2, 2,  1,  3,  5,  2,  3,
    -1, 2,  0,  0,  0,  -1, 0,  -1, -1, 4,  1,  1,  -2, 2,   2,  -5, 0,  1,  -2, 1,  -3, -1, 0,  1,  2,  4,   0,  5,   4,  -1, -2, 0,  4,  4,  -4, -7, -3, -7,
    -1, -2, 0,  -2, 1,  0,  2,  1,  -1, 2,  0,  -1, 1,  4,   2,  4,  -2, 4,  2,  4,  3,  -4, -1, 6,  3,  2,   3,  2,   2,  1,  -3, 1,  3,  0,  3,  -1, 3,  0,
    0,  -2, -2, -3, -2, 0,  0,  3,  4,  -3, 4,  2,  1,  -4,  0,  2,  -1, 3,  1,  -2, 3,  -1, 2,  -1, 3,  3,   1,  0,   1,  0,  2,  3,  2,  6,  1,  2,  2,  3,
    1,  1,  3,  5,  0,  4,  2,  -2, 2,  0,  2,  3,  2,  0,   0,  3,  0,  -2, -1, 0,  1,  3,  0,  5,  3,  2,   -4, 2,   5,  -2, -3, 2,  -1, 4,  -1, 2,  3,  2,
    3,  4,  3,  6,  3,  -2, -2, 1,  5,  7,  0,  -5, 3,  -4,  4,  -2, 5,  0,  1,  2,  2,  1,  0,  3,  1,  -2,  0,  1,   4,  4,  -1, -1, 0,  2,  1,  -4, -2, -1,
    1,  -2, 4,  3,  4,  -1, -2, 0,  3,  1,  0,  0,  4,  -2,  -3, -4, -3, -3, 0,  0,  0,  0,  1,  -4, -3, -3,  0,  -3,  -4, 1,  0,  3,  -2, -3, -3, -1, 1,  -3,
    0,  0,  2,  -1, 2,  -1, 2,  0,  0,  2,  -3, -3, 0,  3,   0,  1,  0,  2,  1,  0,  -2, 1,  1,  4,  1,  5,   2,  1,   -2, 0,  0,  0,  -2, -1, 2,  1,  -2, 2,
    3,  1,  -3, 4,  3,  -3, -3, -3, 0,  5,  0,  -1, 0,  0,   0,  0,  0,  1,  2,  1,  -2, -2, 1,  1,  -3, -7,  -2, -8,  -1, -4, 1,  -1, 1,  -1, 2,  2,  -1, 2,
    4,  -1, 1,  0,  0,  1,  0,  2,  2,  3,  2,  -3, -2, 1,   4,  0,  3,  3,  7,  1,  -2, 1,  0,  2,  2,  -2,  3,  -2,  -2, -1, -1, -1, -1, 2,  1,  3,  3,  -3,
    -1, 1,  0,  -1, 1,  5,  1,  1,  0,  -1, -1, 1,  2,  1,   -1, 2,  2,  1,  1,  -1, 2,  1,  2,  4,  2,  0,   3,  5,   0,  2,  4,  5,  -1, 0,  -4, -2, 2,  2,
    0,  4,  1,  2,  1,  6,  0,  -2, -1, 0,  -1, 2,  0,  4,   4,  1,  1,  2,  1,  0,  -3, -2, 0,  6,  1,  1,   1,  2,   2,  2,  0,  3,  4,  2,  0,  1,  5,  5,
    1,  -5, -1, -5, 1,  -1, 4,  1,  1,  0,  6,  3,  0,  2,   1,  2,  0,  3,  5,  3,  2,  1,  -1, 4,  1,  1,   -1, -1,  3,  1,  3,  4,  5,  -1, -1, 1,  1,  1,
    1,  -1, 1,  -2, -2, 1,  -1, -2, -1, 1,  -2, 2,  3,  -1,  -2, 1,  -2, -1, 2,  5,  -3, 1,  -2, -2, -5, 1,   2,  -4,  1,  -1, 0,  1,  0,  2,  4,  3,  0,  5,
    2,  1,  -1, 1,  -2, 4,  0,  1,  0,  -1, -4, -2, -1, 3,   1,  3,  1,  2,  -1, 2,  1,  0,  -2, -1, 1,  0,   1,  3,   2,  0,  -4, 3,  1,  -2, -5, 1,  -2, 4,
    -4, 2,  3,  3,  1,  3,  0,  4,  3,  -1, -1, -1, 5,  4,   -2, -8, -1, -4, -1, -4, 2,  -1, -3, 1,  0,  -1,  -2, -1,  3,  1,  3,  -1, -1, 0,  -1, 2,  -6, -3,
    3,  0,  -1, 4,  1,  -1, -1, 1,  3,  4,  -2, 0,  0,  0,   -1, 2,  1,  -3, -2, 1,  1,  -1, 0,  1,  -2, -1,  1,  0,   2,  0,  1,  1,  0,  -1, 2,  1,  3,  1,
    -1, 0,  -1, 3,  0,  0,  1,  0,  2,  1,  3,  -5, -1, -1,  -3, 1,  0,  1,  -2, -4, -4, 0,  -2, -4, -6, 1,   -2, 0,   -5, -7, -4, 3,  2,  0,  2,  4,  2,  0,
    2,  5,  3,  -4, 3,  2,  4,  4,  0,  3,  -2, -3, 3,  0,   -1, -1, 4,  -3, -1, -4, -4, -3, -4, -5, -1, 0,   0,  1,   3,  6,  -1, 2,  -2, -1, 5,  2,  -2, 2,
    1,  -1, 0,  -1, -1, 2,  5,  -2, -1, -1, 2,  -2, -5, -3,  3,  0,  0,  2,  0,  -1, 1,  2,  4,  2,  -2, 1,   5,  -2,  2,  0,  -1, -2, -3, 3,  2,  -1, -1, 1,
    -2, -3, -1, 0,  2,  3,  1,  -1, -1, 0,  1,  2,  4,  1,   0,  0,  -1, 1,  -2, 2,  1,  0,  0,  1,  3,  -3,  3,  -1,  -5, 0,  1,  2,  -1, -3, -2, 3,  1,  -3,
    -3, 1,  0,  -3, -3, -3, -7, 4,  1,  0,  7,  3,  1,  0,   0,  4,  3,  -1, 2,  1,  3,  5,  -1, 3,  0,  -2,  1,  -3,  -3, -2, 3,  -2, -1, -4, -4, -1, -3, -5,
    0,  0,  -1, 4,  2,  6,  0,  3,  -3, 2,  3,  0,  -3, 2,   -1, 1,  -1, 0,  3,  1,  2,  0,  1,  0,  2,  -1,  -1, -2,  3,  0,  1,  2,  3,  2,  1,  2,  2,  4,
    -2, -1, 4,  -3, 3,  1,  -2, 0,  0,  2,  0,  -1, -1, 1,   0,  -2, -1, -1, 2,  0,  1,  1,  1,  1,  3,  1,   4,  -1,  -1, 1,  0,  3,  -2, 2,  1,  -1, 0,  1,
    1,  -1, 1,  -2, -2, 0,  3,  2,  -4, -1, 1,  1,  -1, -2,  -4, 1,  -3, 2,  -3, -5, -4, 4,  1,  1,  6,  6,   0,  1,   -1, 2,  1,  1,  1,  -2, 5,  4,  0,  2,
    1,  -2, 2,  -1, -1, 0,  0,  -2, 0,  0,  -1, -1, -1, -5,  -1, -4, -1, 2,  2,  6,  -2, 2,  -2, 2,  3,  2,   -3, 2,   -1, 1,  0,  0,  1,  0,  0,  1,  1,  0,
    2,  0,  1,  -2, 3,  -2, 1,  2,  2,  3,  3,  3,  4,  4,   -1, -2, 2,  2,  0,  1,  0,  -2, 1,  0,  0,  0,   1,  2,   1,  -3, -1, 0,  0,  -2, 3,  -1, 4,  1,
    2,  1,  3,  -2, 0,  2,  -1, 2,  -1, -1, -1, 0,  -2, 1,   2,  1,  5,  -2, 0,  0,  1,  1,  0,  -3, 0,  4,   -2, -2,  -2, 1,  -1, 2,  -1, -3, -5, 4,  2,  0,
    4,  4,  0,  3,  -4, 3,  0,  2,  1,  -2, 5,  3,  0,  0,   -1, 0,  5,  -2, 0,  0,  -1, -2, 1,  -3, -4, -1,  0,  -2,  0,  -2, -1, 3,  1,  4,  -1, 2,  -2, 1,
    5,  1,  -5, 1,  0,  1,  2,  2,  2,  2,  2,  2,  0,  2,   3,  0,  -1, -1, -1, -1, 1,  0,  0,  4,  -1, 1,   3,  4,   -1, 0,  2,  0,  -1, 0,  0,  -2, 1,  1,
    -1, 1,  1,  0,  2,  -3, -2, 1,  1,  0,  2,  0,  4,  0,   2,  2,  3,  0,  -3, 3,  0,  3,  -2, -2, -4, 0,   1,  0,   3,  1,  1,  0,  -1, -1, 1,  0,  -1, -2,
    0,  4,  -2, 0,  -2, 2,  0,  3,  -4, 0,  -5, 4,  0,  3,   3,  4,  -2, 4,  -2, 4,  0,  1,  3,  1,  4,  3,   -1, 1,   1,  -2, 2,  -5, -1, 1,  -2, -3, 2,  -2,
    -4, -4, -1, -3, -3, -2, -1, 3,  3,  3,  -3, 5,  -2, 0,   4,  3,  -1, 1,  2,  -1, 1,  1,  2,  0,  -3, -1,  -2, 3,   3,  -1, -2, 1,  -2, 0,  1,  1,  2,  2,
    0,  -1, -1, 1,  1,  1,  -1, 4,  0,  0,  1,  2,  3,  -1,  0,  -1, 3,  0,  1,  -1, 1,  2,  -3, -1, -1, 0,   -1, -2,  4,  0,  -1, 4,  3,  1,  3,  5,  1,  0,
    2,  3,  -1, 2,  1,  -2, -1, -1, -1, -5, -7, -5, 2,  0,   -1, 1,  2,  -2, 5,  3,  0,  2,  0,  0,  -2, 0,   -6, -4,  1,  4,  2,  -1, -1, 2,  -2, -1, 0,  1,
    2,  1,  -2, 3,  0,  1,  -1, 1,  1,  1,  4,  -2, -4, -2,  -1, 3,  -4, -4, -3, -4, 1,  -6, -2, -3, 1,  4,   3,  3,   6,  6,  -2, 3,  1,  -1, 1,  0,  3,  1,
    -2, -2, 0,  2,  3,  1,  1,  -1, -3, 0,  2,  3,  1,  3,   2,  1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  1,   5,  -2,  0,  0,  5,  0,  -2, -2, 2,  2,  0,  2,
    0,  1,  -1, -3, 4,  1,  -1, 5,  4,  3,  2,  2,  0,  -1,  2,  0,  -1, 1,  3,  -1, 0,  0,  0,  -5, -5, -7,  1,  0,   0,  -2, 2,  -3, 5,  2,  1,  2,  1,  2,
    0,  0,  -6, -4, -1, 3,  3,  1,  0,  4,  -2, -1, -2, 2,   2,  4,  0,  0,  0,  0,  -6, 1,  4,  0,  4,  1,   -3, 0,   0,  2,  0,  -1, -3, -6, -3, -6, -5, -2,
    -1, 4,  2,  1,  7,  5,  -2, 4,  2,  -4, 3,  0,  0,  1,   -1, -1, 0,  2,  3,  1,  0,  0,  -1, -1, 2,  3,   1,  0,   1,  -1, -1, 2,  1,  0,  0,  2,  1,  3,
    -1, 0,  2,  -1, -1, 1,  6,  1,  0,  -3, 3,  1,  0,  0,   2,  1,  -1, 0,  0,  1,  1,  2,  3,  1,  3,  3,   1,  -1,  3,  -1, -1, 0,  0,  -1, 0,  -1, 1,  -2,
    -6, -6, 2,  1,  1,  1,  3,  -2, 5,  1,  1,  1,  1,  0,   -1, 1,  -4, -6, 2,  1,  2,  1,  0,  1,  0,  0,   -2, 4,   0,  3,  1,  3,  2,  -1, -3, 2,  5,  0,
    3,  1,  -5, -1, 0,  1,  0,  -1, -1, -4, -1, -4, -5, 1,   2,  3,  3,  2,  6,  5,  -3, 4,  1,  -2, 5,  -2,  3,  1,   -1, -3, 2,  3,  5,  4,  -1, 2,  0,  1,
    1,  2,  1,  1,  0,  0,  -2, 2,  1,  1,  1,  2,  3,  2,   0,  0,  1,  0,  -1, 1,  6,  1,  0,  -1, 0,  -1,  2,  1,   -1, -1, 0,  0,  1,  0,  -1, 1,  4,  3,
    2,  3,  2,  -1, 3,  -1, 0,  -1, -2, -1, -1, 0,  1,  -4,  -3, -6, 2,  3,  0,  -1, 4,  -1, 3,  2,  -1, 1,   3,  0,   -1, 1,  -4, -5, 0,  1,  1,  0,  0,  0,
    0,  -2, -3, 3,  1,  0,  1,  1,  3,  0,  -1, 1,  4,  1,   2,  -1, -3, -1, 1,  1,  -2, -2, -2, -2, -2, -5,  -6, 3,   3,  2,  5,  4,  5,  7,  -2, 3,  0,  -3,
    4,  0,  2,  0,  0,  -3, 0,  2,  5,  1,  -2, 4,  -1, 2,   2,  0,  0,  -1, 0,  -1, -2, 1,  -1, 0,  2,  2,   1,  2,   -1, 0,  1,  -2, -1, -2, 6,  2,  0,  -3,
    1,  1,  -1, 0,  0,  -2, -2, 1,  3,  0,  -1, 2,  2,  4,   0,  0,  1,  -2, 1,  -1, -1, 3,  1,  -1, -1, 1,   1,  -2,  -5, -4, 1,  1,  -1, -1, 4,  -2, 2,  2,
    -3, -1, 0,  -1, -3, 1,  -6, -1, 1,  1,  2,  0,  -3, 0,   0,  -1, -1, 0,  4,  -1, 0,  3,  3,  0,  -2, 1,   4,  -2,  1,  -1, -2, -2, 2,  1,  0,  -3, -5, -4,
    -2, -6, -6, -3, 0,  4,  5,  4,  6,  5,  1,  2,  0,  0,   1,  2,  1,  0,  1,  3,  -2, -2, -3, 3,  2,  0,   3,  4,   3,  -1, 0,  2,  2,  5,  1,  -1, 3,  2,
    -1, 2,  -1, 0,  2,  -2, 0,  0,  3,  -2, 2,  -2, 3,  -1,  0,  1,  1,  1,  0,  -1, 2,  1,  0,  2,  -1, 0,   2,  1,   2,  -1, 0,  1,  3,  0,  -1, -4, 2,  0,
    0,  4,  1,  -3, -1, 3,  1,  0,  0,  -2, 0,  1,  -2, -1,  4,  2,  3,  0,  1,  2,  -1, -2, 3,  -3, 1,  -6,  0,  -4,  -3, -1, 5,  -1, 2,  1,  -2, 3,  -4, -1,
    -1, -2, -1, 0,  3,  2,  1,  1,  4,  2,  0,  4,  1,  -1,  -2, -2, 1,  1,  5,  3,  4,  2,  -4, -1, 1,  3,   0,  1,   -1, 2,  2,  0,  4,  3,  -2, -2, -1, 3,
    4,  1,  0,  5,  1,  2,  1,  0,  3,  6,  2,  -2, 5,  2,   0,  0,  -1, 0,  0,  0,  1,  -2, 4,  -1, 1,  0,   5,  -1,  -1, 3,  1,  0,  -1, 1,  0,  2,  0,  0,
    1,  -3, 1,  -1, 0,  0,  0,  -1, 2,  -1, -1, -3, 2,  0,   1,  5,  1,  -2, 1,  0,  1,  1,  0,  -2, -1, 2,   -2, -1,  1,  2,  1,  2,  0,  -1, 1,  -1, 2,  0,
    -2, -3, 0,  -3, -1, -1, 4,  1,  1,  2,  -1, 1,  -2, -1,  -1, -1, 2,  -2, 3,  3,  1,  3,  3,  -1, 1,  2,   0,  -1,  -3, -1, 1,  0,  6,  2,  4,  2,  -1, 1,
    3,  4,  1,  0,  -3, 1,  1,  2,  3,  1,  0,  -2, 2,  1,   4,  2,  1,  5,  2,  3,  -2, -1, 2,  4,  4,  -2,  1,  0,   -1, 0,  2,  -1, 1,  1,  -1, -2, 2,  1,
    1,  -2, 2,  -2, 2,  2,  0,  -1, 0,  2,  -1, -1, 0,  1,   1,  0,  0,  0,  0,  -1, -2, 0,  0,  -2, 0,  -1,  1,  0,   0,  1,  0,  -3, 1,  1,  0,  3,  -3, 0,
    -1, 0,  -3, -2, 0,  2,  3,  3,  2,  -1, 2,  0,  1,  -1,  -2, -2, -2, -1, 0,  0,  3,  0,  -2, 1,  0,  -2,  -3, 1,   0,  -1, 0,  1,  3,  3,  -1, 5,  3,  0,
    1,  4,  0,  -4, -2, 1,  1,  1,  3,  2,  5,  4,  -2, 0,   4,  3,  2,  0,  -3, -2, 3,  -2, 2,  1,  2,  0,   3,  4,   1,  3,  3,  2,  2,  3,  0,  0,  3,  3,
    3,  -4, 1,  -1, -2, -2, 1,  0,  1,  1,  -2, -4, 4,  1,   -1, -2, 2,  -3, -2, 1,  2,  1,  -2, 2,  0,  -3,  1,  1,   0,  -2, 1,  -1, 2,  -2, -1, -1, 0,  -3,
    0,  -3, -1, 1,  1,  2,  0,  -4, 2,  0,  2,  1,  0,  2,   1,  0,  1,  -1, -2, 1,  2,  3,  4,  -2, 0,  0,   0,  -1,  0,  -1, -2, -1, -1, -1, 6,  2,  1,  1,
    -2, 0,  -2, 0,  -1, -2, -1, 2,  0,  2,  3,  3,  2,  1,   0,  3,  0,  -1, -4, -1, 3,  1,  3,  1,  5,  2,   -2, 1,   4,  2,  1,  2,  -3, -1, 1,  2,  3,  1,
    5,  -1, -2, 4,  1,  1,  5,  2,  3,  2,  0,  1,  2,  1,   1,  -4, 1,  1,  -3, 1,  -2, 2,  2,  -2, 3,  -4,  3,  -1,  -1, -4, 3,  -2, -1, 1,  1,  1,  0,  2,
    0,  0,  -1, 0,  0,  0,  -1, 1,  1,  -1, 0,  0,  0,  -3,  -1, -2, -2, 0,  2,  1,  2,  -1, 2,  1,  5,  0,   0,  3,   -1, 1,  -1, -2, -3, 1,  3,  3,  2,  -2,
    0,  0,  1,  -1, -2, 0,  -6, 1,  -3, 1,  5,  2,  -1, 0,   -4, 0,  -3, 0,  -1, 0,  1,  3,  2,  2,  1,  1,   3,  0,   1,  4,  0,  -3, -4, -1, 0,  0,  2,  0,
    3,  0,  -1, 1,  3,  0,  2,  0,  -5, 1,  4,  2,  4,  1,   0,  -1, 5,  3,  -2, 4,  1,  0,  -1, 1,  2,  -1,  1,  -4,  -2, 1,  0,  -3, -3, -4, -1, 0,  -1, 0,
    -2, 2,  5,  -2, 0,  3,  3,  3,  2,  -1, -2, 1,  1,  2,   1,  -1, 2,  2,  0,  0,  0,  -2, 0,  2,  1,  1,   1,  -3,  -1, 1,  2,  -1, 2,  1,  5,  -1, 2,  3,
    0,  4,  3,  3,  0,  5,  -1, 0,  2,  -2, -1, 3,  5,  1,   -1, 1,  -1, -2, 0,  -3, -5, -1, 0,  2,  -2, 0,   -2, -2,  0,  -1, 0,  1,  1,  -2, 3,  -3, -2, 0,
    3,  -2, -1, 0,  4,  -1, -3, 0,  0,  2,  -3, 0,  1,  3,   -2, 3,  1,  -2, 2,  1,  2,  2,  -6, 0,  3,  1,   5,  -3,  0,  -1, 3,  -1, -1, 3,  4,  -2, 0,  2,
    -1, -2, 0,  -3, -2, 1,  -1, -4, -1, -4, -1, -1, -1, 2,   0,  2,  2,  -2, 3,  1,  3,  3,  1,  -4, -2, 1,   -1, 0,   -1, -3, 3,  2,  -1, -2, -1, -1, 4,  -1,
    1,  0,  2,  -1, -1, 1,  1,  0,  0,  2,  4,  -1, 4,  3,   -2, 4,  0,  2,  0,  4,  1,  1,  -2, -2, -3, 2,   6,  1,   0,  4,  0,  1,  2,  1,  -1, -1, 1,  1,
    -1, 1,  0,  1,  1,  1,  1,  3,  0,  0,  2,  0,  0,  0,   1,  1,  0,  0,  3,  2,  -1, -1, 0,  0,  -1, -2,  0,  2,   0,  3,  0,  2,  1,  1,  2,  1,  -5, 1,
    4,  2,  5,  -3, 2,  -3, 0,  -1, -2, 1,  4,  0,  0,  -1,  1,  -2, 2,  -1, -1, 0,  -1, -2, -3, -4, 0,  -1,  -2, 2,   -1, 3,  2,  1,  0,  2,  3,  1,  1,  -4,
    -2, 0,  -2, 4,  1,  -3, 3,  2,  1,  1,  0,  0,  2,  1,   1,  0,  1,  -2, -2, 1,  0,  1,  1,  2,  5,  1,   3,  2,   -2, 0,  2,  0,  0,  4,  1,  1,  0,  -2,
    0,  2,  4,  1,  -1, 4,  -1, -1, 2,  0,  -1, 0,  3,  0,   -2, 1,  1,  0,  -2, 2,  0,  3,  2,  1,  1,  0,   4,  4,   1,  2,  -1, 2,  3,  3,  1,  -2, 1,  1,
    -2, -2, 2,  1,  0,  0,  -2, -1, 2,  -1, 1,  0,  -6, 1,   2,  2,  4,  -1, 0,  -1, 1,  0,  -1, -1, 2,  -1,  -3, -2,  0,  -2, 2,  -4, 1,  0,  -4, -2, -1, -3,
    0,  -1, 1,  2,  -1, 2,  3,  2,  0,  1,  3,  0,  2,  -3,  -2, 1,  -2, 2,  1,  -2, 1,  4,  -3, -1, -2, 1,   2,  3,   1,  1,  1,  -2, -2, 1,  0,  0,  -1, -1,
    3,  1,  2,  2,  0,  4,  1,  2,  -1, 5,  0,  0,  -1, 2,   2,  2,  4,  -1, 0,  5,  -2, -4, 0,  -1, -2, 1,   4,  0,   -5, 0,  0,  -1, 1,  1,  4,  4,  3,  2,
    2,  0,  4,  2,  1,  2,  -1, 1,  3,  1,  0,  0,  2,  2,   -3, 0,  0,  2,  1,  2,  -2, 0,  -2, 2,  2,  0,   -3, -3,  2,  2,  4,  0,  -2, 0,  2,  -1, 1,  0,
    3,  0,  -6, -2, 3,  -2, 0,  -3, -2, 2,  -1, -1, -1, -1,  -2, -3, -2, 1,  0,  2,  1,  2,  2,  -1, -1, -2,  -2, -3,  -4, 2,  1,  1,  0,  -1, 1,  0,  -1, 1,
    -1, 0,  2,  1,  1,  0,  2,  0,  0,  1,  0,  0,  -1, -1,  3,  3,  2,  3,  -2, 1,  1,  1,  -1, 6,  -3, 0,   0,  0,   2,  2,  3,  -2, 2,  4,  1,  -3, 0,  0,
    0,  1,  5,  4,  -2, -2, 0,  0,  -2, 1,  3,  7,  3,  2,   3,  1,  4,  2,  1,  2,  0,  2,  2,  0,  0,  0,   -1, 0,   0,  -1, 2,  1,  0,  3,  -3, 0,  0,  -4,
    5,  1,  0,  0,  -3, 2,  2,  2,  3,  -5, -1, -2, 2,  1,   -1, 3,  1,  -1, 1,  0,  0,  2,  1,  4,  1,  2,   1,  3,   -1, -2, 0,  -1, 0,  2,  3,  -4, 4,  4,
    4,  4,  3,  -6, -1, 1,  0,  2,  0,  -4, -2, 2,  0,  0,   0,  -3, -3, 3,  -5, -3, 2,  0,  0,  0,  -1, -1,  0,  1,   -1, -1, 2,  -1, 1,  0,  1,  -2, 2,  1,
    2,  -1, 0,  -2, 0,  4,  2,  2,  4,  1,  2,  0,  4,  1,   2,  -1, -1, -1, 3,  -2, 4,  4,  3,  2,  -2, -2,  -1, 0,   1,  1,  3,  3,  2,  -1, 2,  1,  3,  3,
    0,  0,  3,  0,  0,  0,  -2, 0,  -1, -2, 2,  -2, 0,  -2,  5,  3,  2,  0,  -1, 1,  0,  3,  -1, -6, -1, -1,  2,  0,   -2, 5,  2,  0,  4,  -2, 4,  2,  1,  5,
    -2, 0,  -2, 0,  -2, -3, 0,  -2, -2, 3,  3,  -3, 5,  7,   1,  -1, 1,  -4, 0,  -1, 1,  2,  2,  -3, 0,  -1,  1,  1,   2,  -2, -1, 2,  -4, -2, 0,  0,  1,  -1,
    2,  2,  0,  0,  -1, -1, 4,  0,  2,  -1, -2, -3, -1, 1,   1,  2,  1,  -1, 1,  4,  2,  5,  3,  0,  4,  2,   1,  1,   0,  1,  -1, 2,  -1, -1, 4,  2,  2,  0,
    0,  0,  -1, 0,  2,  2,  3,  2,  3,  0,  2,  -1, 2,  2,   -1, 0,  0,  3,  1,  -2, -2, 1,  1,  0,  0,  0,   2,  -1,  4,  3,  2,  0,  -1, 2,  -1, 3,  -2, -7,
    -4, 2,  3,  -1, 0,  1,  2,  2,  3,  -1, 5,  3,  1,  2,   -1, 1,  -3, -1, -2, -2, 2,  -2, -2, 0,  3,  -1,  3,  9,   3,  1,  2,  -2, -3, 0,  2,  2,  -1, -3,
    0,  0,  -1, 2,  -2, -1, 0,  0,  -1, -2, -1, 0,  1,  1,   2,  2,  0,  2,  1,  -1, 1,  1,  2,  -1, 0,  -1,  -3, 2,   4,  -3, 1,  0,  1,  3,  1,  5,  6,  3,
    1,  2,  1,  1,  -1, 1,  -2, 2,  0,  1,  4,  1,  2,  0,   -1, 0,  1,  -2, 1,  -1, 4,  3,  3,  0,  2,  1,   3,  2,   1,  2,  2,  2,  0,  -1, -3, 2,  -1, 1,
    0,  -1, 3,  -4, 1,  3,  1,  1,  3,  0,  -3, 2,  2,  -2,  -3, 3,  -1, -3, 1,  5,  0,  3,  4,  -2, 3,  3,   0,  2,   -2, 1,  0,  -1, 0,  -2, 2,  -1, 0,  1,
    1,  -3, 2,  11, 2,  0,  0,  -5, -5, 0,  0,  3,  0,  -1,  0,  -1, 1,  1,  -2, -1, -3, 0,  0,  1,  1,  0,   0,  -1,  1,  1,  0,  2,  0,  1,  1,  1,  2,  1,
    -4, 0,  0,  4,  1,  -1, 1,  0,  3,  3,  2,  7,  5,  6,   3,  3,  4,  0,  -3, -2, -2, 1,  -2, -1, 1,  0,   1,  3,   0,  3,  1,  0,  -1, 3,  7,  2,  2,  1,
    1,  0,  3,  1,  0,  -1, 0,  0,  -2, -1, -2, 1,  -1, -1,  1,  -1, 3,  -6, -3, 4,  0,  0,  4,  0,  0,  0,   3,  1,   -2, 0,  -1, -2, -2, 5,  2,  1,  -1, -2,
    3,  2,  3,  1,  4,  3,  0,  0,  0,  -1, 1,  -1, -2, 1,   4,  -3, 0,  7,  2,  2,  1,  -4, -6, 2,  1,  4,   0,  -1,  0,  -1, -2, 0,  0,  -2, -1, 1,  -1, 0,
    0,  0,  -1, -2, 1,  0,  -1, -1, 0,  -1, -2, 0,  2,  0,   -2, 1,  -2, 3,  2,  -2, 2,  1,  2,  1,  1,  4,   3,  3,   3,  2,  6,  2,  -2, 0,  -1, 1,  -2, -1,
    0,  -2, 1,  2,  1,  1,  0,  1,  0,  1,  3,  2,  4,  1,   -1, 1,  2,  1,  -1, 0,  1,  2,  0,  -3, -4, 2,   -1, -1,  0,  -2, -3, 2,  3,  -1, 0,  1,  1,  0,
    1,  3,  -2, 1,  1,  0,  -2, -1, 0,  -1, -3, 0,  0,  -2,  1,  -2, 3,  3,  3,  2,  0,  0,  -2, 0,  1,  1,   1,  2,   1,  -2, -3, -2, 1,  2,  2,  1,  -1, 0,
    2,  3,  1,  0,  1,  2,  2,  0,  0,  1,  -1, 1,  2,  4,   4,  2,  2,  1,  3,  1,  1,  2,  4,  0,  0,  0,   -1, -2,  -2, 0,  -2, -1, -2, -5, -3, 1,  -1, 2,
    -3, -2, -2, -2, -5, -6, -3, -3, -1, -4, 0,  -1, -1, -3,  0,  1,  2,  0,  -2, 1,  7,  1,  3,  1,  1,  2,   0,  0,   -1, 5,  3,  0,  -1, 0,  3,  0,  0,  -1,
    -2, -6, -1, -1, 1,  -2, -2, 0,  2,  0,  -1, 1,  1,  -1,  3,  3,  -1, 1,  1,  0,  -3, -1, 0,  3,  -4, -2,  0,  -3,  2,  -1, 4,  3,  3,  -1, 1,  0,  -1, -1,
    1,  1,  1,  1,  0,  -2, -5, -1, 1,  1,  1,  2,  0,  0,   0,  2,  1,  0,  -1, 3,  2,  1,  -1, 0,  1,  1,   2,  3,   3,  3,  3,  0,  2,  2,  -1, 2,  3,  0,
    -1, -1, 1,  -3, -1, -1, -1, 2,  -1, -4, -5, 1,  -3, -3,  -3, -1, -2, -4, -6, -6, -3, -2, 2,  -5, -2, -2,  1,  1,   0,  0,  0,  1,  -1, 0,  6,  3,  1,  4,
    3,  3,  1,  2,  1,  2,  1,  0,  -1, 2,  4,  0,  1,  -1,  0,  -1, 1,  0,  2,  1,  0,  1,  4,  -1, -1, 1,   -2, -1,  1,  1,  -2, 2,  -1, 5,  -1, -3, -1, 2,
    -3, 0,  2,  -1, 1,  -1, 2,  2,  1,  3,  0,  0,  -2, 0,   -1, 1,  2,  -1, 2,  -3, -3, -1, 2,  2,  4,  -1,  -2, 0,   -1, 3,  4,  0,  -2, 3,  3,  0,  1,  1,
    1,  3,  4,  6,  3,  1,  3,  0,  0,  0,  0,  3,  1,  -1,  -1, 1,  4,  -4, -1, 1,  1,  2,  -2, -2, -2, 2,   1,  -3,  -3, -1, 0,  -2, -3, -3, -3, -2, 0,  -3,
    -4, 2,  3,  1,  1,  -1, 0,  1,  3,  -1, 3,  1,  1,  3,   2,  4,  1,  3,  0,  1,  4,  0,  0,  0,  3,  1,   2,  1,   -1, -2, 2,  0,  -1, -2, 1,  1,  4,  -1,
    0,  0,  -1, 0,  0,  -1, -2, 3,  0,  4,  3,  -4, 3,  1,   -2, -1, 0,  0,  -1, 2,  4,  1,  1,  3,  2,  -2,  -1, -1,  0,  0,  2,  -1, 2,  -1, -2, -2, -1, 1,
    3,  -2, 0,  1,  1,  1,  4,  -1, -2, 1,  1,  -1, 1,  1,   1,  2,  2,  3,  1,  1,  1,  1,  0,  1,  0,  3,   3,  2,   -1, 2,  2,  0,  -1, 0,  -3, 1,  -1, 1,
    -3, 2,  -2, -3, -2, -2, 0,  -1, -3, -6, -3, -1, 0,  0,   -3, 0,  3,  1,  3,  3,  0,  4,  0,  1,  2,  1,   1,  4,   0,  2,  3,  3,  -1, 1,  3,  0,  0,  0,
    1,  5,  2,  0,  -2, -1, 0,  -4, -3, -4, 0,  4,  4,  -1,  -2, 1,  2,  -2, 1,  1,  -1, 6,  0,  1,  1,  -1,  2,  -2,  -1, 0,  1,  0,  -1, 1,  2,  0,  3,  3,
    -1, 1,  0,  -2, 1,  1,  1,  0,  1,  2,  -1, 1,  2,  2,   1,  -2, 0,  2,  0,  1,  5,  -3, 2,  1,  3,  2,   2,  1,   1,  2,  0,  2,  1,  0,  1,  0,  0,  0,
    2,  1,  4,  1,  0,  0,  -1, -1, 0,  1,  -2, 1,  -1, -1,  -2, 2,  0,  -2, 0,  1,  3,  0,  -1, -2, -2, -1,  2,  -2,  -1, 1,  4,  1,  0,  2,  2,  3,  -1, 0,
    3,  -1, 2,  4,  1,  2,  3,  2,  -1, 2,  2,  -2, 1,  1,   2,  4,  0,  -2, -1, -1, 1,  -3, 0,  -3, -1, 1,   -2, 3,   0,  -3, -2, 0,  1,  3,  -1, -2, 0,  0,
    0,  2,  4,  2,  -1, -2, 0,  1,  -1, -2, -3, 6,  1,  2,   6,  1,  1,  0,  1,  1,  4,  2,  -2, -2, 0,  7,   -3, 4,   0,  1,  2,  2,  1,  -2, 4,  3,  -2, -3,
    0,  0,  -2, -1, 2,  0,  0,  -4, -2, 1,  3,  0,  -2, 3,   1,  1,  7,  2,  0,  1,  2,  0,  1,  -4, 1,  3,   2,  4,   3,  2,  -1, -2, 0,  -2, -4, -1, 0,  -3,
    1,  -4, -9, 0,  -1, 2,  0,  -1, 3,  0,  -6, 1,  3,  0,   -1, 0,  1,  6,  6,  3,  -3, -1, -3, 0,  1,  2,   2,  0,   -2, 0,  0,  -3, 3,  5,  3,  1,  1,  0,
    1,  1,  1,  4,  -2, -6, 0,  0,  -1, 2,  1,  -3, 1,  -2,  0,  1,  3,  5,  -1, -4, 1,  4,  -4, -2, -2, 3,   1,  2,   4,  -3, 0,  0,  0,  3,  5,  -1, -2, -2,
    0,  8,  -1, 2,  2,  0,  0,  1,  2,  2,  4,  3,  1,  -1,  0,  -1, -1, 0,  0,  2,  -2, -4, -3, 2,  4,  2,   -1, 2,   2,  1,  5,  4,  1,  2,  2,  0,  0,  -3,
    0,  4,  0,  6,  4,  3,  1,  1,  1,  -4, -1, 1,  -2, -4,  1,  -1, -4, 1,  1,  1,  0,  -1, 4,  1,  -5, 3,   3,  0,   -3, 0,  4,  5,  7,  2,  0,  1,  -3, 0,
    2,  0,  -1, -4, -5, -1, -1, -2, 3,  3,  0,  1,  -1, 1,   2,  1,  0,  5,  2,  -7, 1,  1,  -1, 4,  -1, -1,  3,  -1,  -3, 1,  -3, 4,  -3, -2, 3,  1,  -1, -1,
    -7, 4,  0,  1,  0,  -1, -3, -1, 2,  -1, 5,  -1, -3, -3,  3,  8,  -1, 0,  -3, 0,  0,  2,  3,  -1, 3,  2,   4,  1,   3,  1,  0,  0,  3,  4,  -2, -4, -4, 3,
    3,  1,  1,  2,  3,  -1, 7,  1,  -1, 2,  2,  0,  -5, -2,  -2, 4,  3,  3,  4,  2,  0,  4,  1,  -3, 1,  -1,  -3, -1,  0,  -2, -5, 0,  3,  0,  -2, 2,  4,  0,
    0,  1,  3,  -4, -2, 2,  2,  1,  4,  0,  0,  2,  -2, 2,   5,  0,  -2, -4, -6, -1, 1,  0,  4,  1,  -1, 1,   -3, 1,   3,  0,  2,  3,  2,  -8, 1,  1,  0,  1,
    2,  1,  0,  0,  -3, 1,  -3, 2,  -1, -1, 1,  2,  -3, -2,  -6, 0,  -1, 0,  5,  -1, -1, -1, 1,  0,  2,  1,   -2, -1,  4,  7,  0,  2,  3,  0,  1,  2,  4,  -4,
    4,  0,  1,  -2, 3,  2,  -3, -3, 2,  0,  1,  -2, -6, 3,   3,  3,  2,  0,  4,  2,  3,  2,  -1, 2,  1,  0,   0,  -4,  1,  4,  4,  2,  3,  4,  1,  3,  1,  -3,
    1,  0,  0,  -3, -3, -2, -6, 0,  0,  1,  3,  -1, 1,  4,   -3, 2,  5,  0,  0,  3,  3,  7,  8,  2,  4,  2,   0,  0,   3,  0,  -3, -3, -5, 0,  -2, -2, 4,  0,
    0,  1,  4,  2,  1,  3,  0,  1,  3,  -1, -1, 1,  -3, 1,   1,  0,  -1, -2, 1,  -3, -4, 2,  -2, 1,  0,  3,   -2, -7,  -5, 3,  3,  2,  2,  0,  -1, 1,  0,  0,
    3,  0,  -3, -1, 1,  5,  1,  2,  1,  1,  2,  3,  3,  -1,  3,  -1, 4,  -2, -1, 2,  0,  -1, 0,  2,  1,  -1,  -4, 3,   1,  2,  2,  0,  5,  2,  2,  4,  1,  1,
    1,  2,  -2, -5, 1,  4,  3,  1,  5,  3,  2,  3,  3,  -4,  0,  -3, 0,  -3, -4, -4, -6, 2,  -4, 2,  7,  2,   5,  -1,  -2, 0,  3,  -2, -3, 0,  3,  6,  8,  -2,
    1,  0,  -3, 2,  2,  -1, -3, -4, -6, 1,  -2, -1, 3,  4,   1,  0,  3,  2,  3,  0,  3,  3,  1,  1,  1,  -2,  -2, -2,  -2, 4,  1,  -3, 4,  7,  1,  -2, 2,  -1,
    2,  3,  -1, 3,  -1, 1,  4,  3,  -3, 2,  -1, 1,  -1, -2,  1,  3,  1,  2,  -2, -1, -2, -2, 4,  2,  2,  0,   -2, 7,   6,  0,  5,  -1, 1,  2,  -4, 1,  2,  0,
    -1, 1,  1,  1,  0,  -3, 1,  -2, 0,  3,  2,  0,  -1, 1,   2,  -1, 2,  1,  4,  0,  3,  0,  -1, -1, 0,  1,   2,  -3,  -1, -2, 5,  2,  -1, 1,  1,  1,  1,  0,
    -1, -1, 0,  -2, -1, -1, -3, -1, -1, -2, 2,  4,  6,  2,   -1, 1,  0,  -4, 0,  3,  0,  1,  -2, -1, 0,  1,   -1, 1,   -1, -4, -4, -2, 1,  1,  3,  4,  -1, 2,
    -2, -2, 0,  -1, -2, 1,  0,  -3, 5,  3,  0,  0,  2,  -2,  1,  4,  -1, 3,  0,  1,  1,  3,  -3, 1,  1,  -2,  -2, 0,   1,  3,  2,  0,  -2, -2, -1, -3, 2,  3,
    0,  0,  0,  4,  3,  2,  3,  -1, 0,  1,  -3, 2,  3,  0,   0,  1,  1,  1,  0,  -3, 1,  1,  1,  1,  0,  1,   -1, 0,   2,  -2, 1,  2,  2,  1,  2,  3,  0,  0,
    0,  2,  1,  -2, -2, 0,  4,  0,  0,  0,  3,  4,  1,  4,   -2, -1, -1, -1, 2,  1,  -3, -2, -1, -2, 1,  5,   5,  2,   0,  2,  1,  -2, -1, 3,  1,  2,  -1, 0,
    1,  -1, 0,  0,  -1, -3, -4, -1, 1,  1,  4,  2,  1,  2,   -1, 0,  0,  -4, -4, 4,  0,  -3, 4,  4,  0,  2,   -1, -3,  1,  1,  0,  4,  -1, 1,  4,  2,  -4, 1,
    -2, 1,  -1, 1,  2,  2,  1,  2,  -2, -2, -2, -3, 1,  3,   1,  -2, 1,  4,  2,  2,  3,  1,  2,  1,  -2, 0,   4,  -2,  -1, -1, -2, 0,  0,  0,  3,  3,  0,  1,
    -1, 1,  0,  2,  1,  0,  -1, 4,  2,  0,  2,  2,  1,  1,   1,  2,  1,  1,  1,  -2, 3,  3,  3,  1,  1,  4,   1,  1,   -1, 1,  -1, -2, 0,  3,  -1, -6, -2, -1,
    2,  4,  3,  5,  1,  3,  1,  -1, -2, 4,  0,  2,  0,  0,   2,  -1, -1, 0,  -2, -2, -1, -2, 1,  1,  4,  0,   0,  1,   -2, -1, -1, 0,  -4, 3,  -1, 0,  0,  5,
    1,  0,  -1, -2, -1, 0,  1,  2,  -1, 2,  3,  3,  -5, 2,   -1, -1, 0,  3,  1,  2,  1,  0,  -1, -2, -1, -4,  1,  1,   1,  -1, 0,  3,  2,  2,  4,  0,  1,  0,
    2,  -1, 4,  0,  -1, -1, -2, 0,  -1, 0,  3,  3,  3,  -1,  0,  1,  0,  1,  2,  1,  1,  2,  3,  -3, 2,  3,   2,  0,   -1, 1,  0,  2,  2,  0,  5,  2,  1,  0,
    2,  3,  -1, 1,  1,  -2, 0,  -1, 0,  3,  2,  -2, 0,  -1,  2,  5,  6,  6,  1,  4,  0,  0,  -1, 5,  2,  2,   0,  -2,  1,  -1, -1, 1,  -3, -3, -2, -1, -1, 2,
    4,  2,  2,  -1, -1, -2, -1, -2, -3, 4,  0,  -2, 0,  4,   -2, 1,  -1, -2, 1,  0,  -1, 2,  0,  0,  4,  1,   -5, 2,   -1, -1, 2,  2,  2,  4,  0,  -1, -1, 1,
    -2, -3, 1,  0,  1,  -1, -2, 1,  1,  2,  1,  1,  1,  1,   1,  1,  4,  -2, 1,  -1, -2, -1, 0,  1,  0,  2,   1,  0,   1,  2,  1,  2,  1,  1,  1,  1,  3,  -3,
    2,  1,  1,  -1, -3, 0,  3,  0,  -1, -1, 5,  2,  1,  1,   4,  1,  1,  1,  2,  -3, -1, -1, 0,  3,  2,  -3,  -2, -2,  1,  3,  4,  4,  2,  4,  1,  0,  1,  4,
    1,  2,  -1, -1, 2,  -1, -3, -1, -3, -2, -4, -2, -2, -2,  1,  1,  1,  0,  5,  1,  7,  1,  0,  2,  -2, -2,  2,  3,   1,  1,  2,  4,  -3, 5,  0,  0,  -4, -5,
    0,  0,  1,  1,  0,  -1, -1, 3,  3,  -2, 1,  -1, 2,  -2,  1,  2,  -2, 3,  4,  -1, -1, -2, 0,  -3, -2, -2,  -2, -1,  0,  -2, 2,  -1, 0,  -2, 2,  0,  0,  -1,
    -4, -2, 1,  5,  0,  -1, -1, 1,  0,  -5, -1, 5,  1,  1,   0,  -2, -3, 0,  4,  0,  2,  1,  -1, -2, 0,  -2,  -1, 1,   2,  1,  2,  1,  -2, 5,  1,  2,  -2, 2,
    1,  -1, -1, 4,  5,  3,  5,  0,  3,  2,  -2, 4,  1,  1,   2,  -1, 3,  1,  0,  1,  4,  -2, -2, 4,  5,  1,   -1, -4,  3,  4,  0,  2,  1,  1,  7,  4,  -3, 2,
    0,  -3, 1,  1,  -2, 3,  3,  1,  1,  1,  0,  -2, -1, -3,  0,  2,  1,  1,  -1, -2, -4, 3,  4,  0,  3,  -2,  1,  -3,  4,  2,  1,  0,  2,  -2, -1, 0,  1,  -1,
    -2, -2, 1,  0,  -1, -1, 1,  1,  -2, 0,  1,  0,  0,  1,   -2, -3, 3,  2,  1,  -1, 0,  2,  0,  -5, -1, 4,   3,  -1,  1,  -2, -1, 0,  2,  -1, 4,  0,  1,  -2,
    -2, 0,  -2, 1,  3,  -1, 2,  1,  -1, 5,  2,  3,  0,  1,   -2, -3, 3,  3,  4,  2,  2,  1,  4,  1,  -2, 1,   4,  0,   1,  -2, 1,  3,  -2, 0,  3,  -5, 3,  2,
    3,  0,  -2, -3, 2,  0,  2,  -2, 1,  1,  7,  2,  -1, 2,   0,  -3, 2,  1,  -2, 3,  2,  3,  1,  2,  -1, -2,  0,  -6,  1,  2,  2,  2,  2,  0,  0,  2,  3,  -2,
    2,  -2, 2,  -2, 3,  3,  -4, 1,  3,  -3, -2, 1,  0,  -3,  -1, 0,  0,  0,  -2, -2, 2,  0,  -5, -1, 0,  0,   1,  -2,  -3, -1, 3,  1,  2,  -4, -2, -1, -3, -5,
    1,  2,  3,  2,  -1, 1,  -1, 0,  4,  -2, 5,  0,  -1, -1,  -2, 1,  0,  3,  2,  2,  1,  0,  -2, 3,  1,  4,   1,  3,   -5, -4, 3,  3,  1,  1,  5,  0,  2,  0,
    -1, 3,  2,  3,  -1, -3, 1,  4,  -1, 1,  2,  -2, -1, 1,   2,  2,  -3, -4, 0,  0,  -1, 2,  2,  -1, 6,  5,   0,  -1,  -2, -3, 0,  1,  -3, -2, 2,  3,  0,  2,
    1,  -1, 0,  -6, -1, 3,  2,  4,  2,  1,  1,  2,  2,  1,   3,  -1, 2,  -2, 2,  2,  -1, 3,  1,  -1, -1, 1,   0,  -4,  0,  -1, -1, 0,  1,  0,  4,  0,  -1, 4,
    0,  1,  1,  0,  -4, 0,  2,  0,  2,  -5, -2, -2, 0,  -2,  -1, 3,  2,  1,  0,  1,  1,  2,  5,  2,  6,  -2,  1,  0,   -2, 1,  1,  0,  1,  0,  2,  1,  -5, 0,
    2,  2,  0,  3,  -4, -5, 0,  3,  2,  1,  6,  3,  0,  1,   -1, 1,  2,  2,  0,  -1, 2,  3,  0,  -1, 1,  -2,  -1, 2,   1,  -1, 0,  -2, 0,  -1, 2,  6,  0,  2,
    5,  2,  3,  -2, -3, -1, 1,  3,  -3, 1,  -1, 2,  2,  3,   0,  0,  -2, -6, 0,  2,  3,  2,  0,  0,  1,  0,   2,  1,   2,  -2, 3,  -2, 3,  2,  -1, 3,  1,  0,
    -2, 0,  1,  -1, -1, 1,  -1, 0,  -1, -2, 1,  0,  -3, 3,   2,  -1, 1,  -1, -5, 3,  4,  1,  2,  -5, 1,  -3,  2,  1,   -1, 3,  3,  1,  1,  2,  -1, 0,  7,  3,
    5,  -2, 2,  1,  1,  -1, 2,  2,  1,  1,  2,  0,  -4, 3,   2,  3,  -3, 2,  -3, -1, 1,  4,  1,  2,  5,  4,   3,  0,   -1, 2,  3,  2,  2,  0,  2,  1,  0,  0,
    0,  -3, 1,  2,  0,  2,  3,  2,  0,  1,  -1, -2, 2,  1,   2,  -3, 1,  1,  -2, 0,  3,  -1, 1,  2,  -2, -2,  1,  1,   3,  1,  3,  1,  3,  0,  3,  0,  -2, 0,
    -2, 1,  6,  0,  0,  -3, 0,  6,  2,  0,  2,  -4, 1,  1,   0,  3,  -1, 2,  1,  5,  0,  -1, 2,  2,  1,  -1,  1,  0,   -1, -1, 0,  1,  1,  0,  0,  4,  0,  -1,
    1,  2,  -3, -1, 0,  -3, 0,  1,  0,  0,  -2, 2,  -1, 2,   3,  0,  0,  -3, -3, -3, 0,  -4, 0,  5,  3,  8,   8,  7,   6,  2,  2,  2,  1,  -1, -8, -4, -3, 2,
    -1, 1,  2,  2,  7,  9,  1,  5,  1,  -1, -5, 5,  9,  -2,  7,  3,  1,  -1, 0,  -1, 5,  2,  0,  1,  1,  -1,  -1, 2,   0,  -1, -1, 2,  -3, 0,  4,  -3, -1, 1,
    -1, 0,  -1, -1, 3,  0,  1,  2,  0,  2,  3,  -3, 1,  1,   -3, 1,  3,  1,  -1, -1, 2,  5,  -2, 1,  3,  -4,  -1, 1,   0,  -1, 2,  2,  1,  3,  0,  -2, 2,  3,
    1,  1,  -1, -1, 0,  1,  0,  0,  2,  1,  1,  2,  -1, 0,   3,  0,  -3, -3, -1, -2, 2,  3,  5,  1,  -2, 0,   -3, -1,  1,  -1, -5, -2, -2, -4, 0,  -4, 0,  5,
    4,  6,  9,  7,  3,  0,  3,  3,  1,  -3, -5, -6, -4, 1,   -5, 0,  4,  3,  7,  10, 3,  5,  2,  -3, -5, 2,   5,  -1,  6,  1,  -1, -2, -2, -4, 5,  0,  1,  -1,
    3,  -2, 0,  2,  0,  0,  -2, 1,  0,  -4, 4,  0,  0,  2,   -2, 2,  1,  -2, 1,  4,  1,  3,  2,  1,  -1, -2,  1,  2,   -2, 2,  4,  5,  -1, 1,  1,  2,  2,  4,
    4,  -4, -1, 3,  -2, -2, 3,  -1, 1,  3,  -1, 0,  2,  2,   -1, 0,  0,  -1, 0,  -1, 0,  1,  2,  2,  2,  2,   -1, -2,  0,  -1, -2, -3, -3, -1, -3, 1,  5,  2,
    1,  4,  -1, 2,  1,  0,  -1, 0,  -3, -6, 2,  -1, -1, 4,   3,  9,  8,  2,  4,  2,  3,  2,  1,  -2, -4, -6,  -1, 3,   -3, 0,  3,  3,  6,  7,  1,  1,  2,  -1,
    -3, 3,  4,  -1, 8,  2,  3,  2,  -1, -1, 7,  0,  -1, 1,   5,  -5, -1, 1,  1,  2,  -3, 2,  2,  -2, 0,  -2,  -1, 2,   -4, 0,  0,  -4, 4,  4,  1,  2,  3,  -3,
    -1, -1, 2,  1,  -2, 1,  2,  0,  1,  0,  1,  4,  2,  3,   3,  -2, -3, 1,  -1, -3, 0,  3,  0,  4,  -2, 0,   4,  1,   0,  -1, -1, -1, -2, 0,  0,  -1, 3,  1,
    -1, 2,  0,  -3, 0,  0,  -2, -1, -1, -1, 0,  -1, 2,  4,   1,  1,  -4, 2,  1,  0,  -1, -1, -3, -5, 1,  -2,  -2, 0,   1,  6,  3,  4,  3,  -2, 4,  0,  3,  -6,
    1,  -6, -3, 4,  0,  -1, 1,  4,  6,  11, 3,  0,  2,  -3,  -4, 0,  4,  1,  7,  0,  1,  -2, -1, 0,  3,  1,   2,  2,   5,  -3, 0,  0,  -2, -2, -2, 3,  1,  -1,
    3,  0,  2,  2,  -6, 0,  2,  -5, 3,  5,  -1, 2,  3,  1,   -1, 0,  0,  3,  -1, 1,  2,  4,  1,  1,  1,  4,   2,  1,   3,  0,  -2, 1,  -1, -1, -1, 1,  -1, 2,
    1,  1,  3,  4,  -2, -1, -2, -2, 0,  0,  -1, -2, 4,  1,   1,  2,  0,  -2, 0,  -1, 0,  -2, -4, -1, 2,  3,   2,  6,   1,  1,  -4, -1, 3,  1,  1,  -2, 0,  -7,
    -1, -3, -3, 1,  1,  6,  3,  5,  3,  2,  3,  2,  3,  -4,  -2, -4, -1, 3,  3,  -1, 0,  4,  6,  9,  5,  1,   0,  -1,  -2, 0,  8,  3,  5,  4,  4,  0,  1,  1,
    -3, -2, -1, -1, 1,  1,  0,  3,  3,  2,  1,  -2, 0,  -2,  -1, 6,  3,  1,  2,  1,  1,  2,  -1, 1,  2,  4,   0,  0,   3,  2,  -2, 0,  2,  -1, 2,  0,  1,  2,
    -2, -4, 1,  -1, 1,  -2, 0,  1,  -2, -1, -1, 0,  0,  -1,  0,  -1, 0,  1,  -1, 1,  0,  0,  -1, -2, 0,  2,   2,  1,   -1, 3,  2,  0,  1,  2,  0,  1,  0,  -1,
    -1, 1,  0,  1,  1,  -2, 1,  3,  4,  1,  -1, 0,  2,  1,   0,  1,  2,  2,  1,  0,  1,  -1, -2, 0,  0,  -2,  1,  1,   -1, 1,  1,  4,  3,  1,  2,  1,  0,  -1,
    -2, 4,  -1, 1,  -1, -1, 1,  -1, 5,  4,  3,  4,  -2, 0,   -2, -2, 1,  0,  2,  0,  0,  2,  1,  3,  2,  0,   2,  -2,  1,  6,  1,  0,  4,  2,  1,  0,  -1, 5,
    1,  1,  -1, 0,  0,  0,  1,  0,  3,  2,  1,  2,  3,  2,   -4, -3, 1,  -1, 0,  -2, 1,  0,  -1, -1, -1, 1,   -2, -2,  -2, -1, 0,  0,  -1, 0,  -2, 1,  0,  0,
    0,  2,  1,  -1, 0,  3,  2,  2,  3,  1,  1,  1,  1,  -2,  -1, 0,  0,  0,  2,  0,  0,  1,  3,  1,  -3, 0,   0,  0,   2,  1,  1,  1,  -1, 0,  -1, -2, 1,  0,
    0,  -3, -1, 3,  -2, 1,  0,  3,  3,  0,  1,  2,  0,  -2,  -1, 3,  -1, 1,  -1, 0,  1,  0,  3,  4,  3,  5,   0,  -2,  -3, -3, 0,  1,  2,  0,  0,  0,  2,  1,
    2,  2,  -1, -1, 2,  5,  1,  0,  4,  1,  1,  0,  0,  6,   0,  1,  -1, 1,  0,  1,  0,  -1, 2,  0,  4,  2,   4,  2,   -3, -5, 0,  -2, 1,  -1, 1,  0,  -2, -3,
    -2, 1,  -2, 1,  -2, 0,  1,  -1, -1, 1,  0,  0,  -1, -1,  1,  3,  -1, 0,  -2, 2,  0,  2,  3,  2,  4,  2,   1,  -2,  0,  2,  0,  1,  3,  1,  0,  6,  2,  0,
    -1, 2,  2,  -1, 0,  1,  2,  2,  0,  -2, 0,  0,  -1, 0,   2,  -2, 1,  2,  -3, 2,  3,  5,  2,  1,  1,  2,   1,  -2,  -2, 2,  -2, 0,  -1, -2, 2,  0,  3,  4,
    2,  4,  1,  -2, -2, -1, 1,  1,  1,  -1, 3,  1,  2,  3,   2,  2,  -1, 0,  3,  3,  2,  -1, 2,  2,  1,  1,   -1, 6,   2,  1,  2,  0,  1,  3,  -1, -2, 2,  2,
    3,  1,  2,  0,  -1, -2, 0,  -3, 0,  -1, 2,  -3, -4, -2,  -1, 0,  0,  -1, -4, 0,  1,  0,  -2, -2, 1,  3,   -2, 0,   1,  2,  0,  1,  -1, 4,  1,  1,  4,  1,
    3,  1,  1,  -2, -1, -1, 1,  1,  0,  -1, 0,  4,  2,  2,   -3, 2,  2,  -3, -1, 1,  1,  2,  1,  -3, 0,  -2,  -1, 1,   1,  -3, 0,  0,  -4, 0,  2,  5,  0,  0,
    1,  2,  1,  -4, -3, 1,  -2, 0,  0,  -2, 5,  0,  4,  3,   3,  3,  0,  -3, -2, 0,  0,  -1, 3,  -1, 3,  0,   2,  0,   5,  2,  2,  -1, 3,  6,  3,  0,  1,  1,
    1,  2,  -2, 5,  1,  1,  3,  2,  1,  1,  -2, 0,  1,  2,   5,  1,  1,  0,  -2, 0,  -1, -3, -1, 0,  4,  -2,  -3, -4,  1,  0,  -3, -3, -2, -1, 1,  2,  -1, 0,
    0,  1,  -1, 0,  1,  2,  2,  1,  -1, 5,  1,  0,  2,  3,   3,  1,  -1, -1, -1, 2,  2,  2,  1,  1,  1,  5,   2,  1,   1,  1,  3,  -2, -2, 1,  0,  1,  -1, -2,
    0,  0,  -2, -1, 2,  -3, -3, 0,  -2, 2,  1,  3,  2,  2,   1,  2,  0,  -3, -1, 2,  -2, 2,  0,  -1, 2,  -1,  3,  4,   1,  3,  2,  -4, 2,  0,  2,  0,  1,  1,
    1,  -2, -1, 0,  5,  1,  -1, -3, -1, 0,  4,  -1, 0,  3,   1,  0,  2,  0,  3,  1,  -3, 1,  3,  2,  2,  1,   1,  2,   3,  3,  0,  -2, 3,  3,  2,  -1, 3,  2,
    2,  1,  -1, 4,  -2, 3,  2,  3,  2,  0,  4,  0,  1,  3,   1,  3,  0,  0,  0,  1,  0,  2,  0,  0,  4,  0,   0,  1,   0,  2,  2,  1,  1,  0,  2,  2,  0,  2,
    5,  1,  4,  1,  -1, 7,  2,  1,  -1, 2,  2,  1,  4,  2,   0,  5,  3,  4,  3,  -1, -2, 1,  1,  6,  2,  4,   6,  2,   4,  0,  1,  3,  1,  -1, -1, 2,  2,  4,
    -2, -1, 1,  0,  4,  4,  6,  3,  1,  2,  2,  0,  2,  0,   1,  -1, 0,  -1, 0,  -3, 0,  -3, -3, 0,  2,  -1,  -1, -1,  1,  -2, 0,  0,  1,  -3, -3, 2,  1,  0,
    3,  3,  2,  -2, 2,  -1, -1, 0,  2,  1,  -1, 0,  1,  0,   0,  -1, -2, 0,  -2, -1, 0,  -1, -1, -1, -2, 1,   -1, -4,  0,  0,  0,  -1, 0,  -1, -2, -3, -1, -2,
    1,  -3, -2, -2, -2, -2, 1,  1,  -2, 1,  0,  1,  -2, 2,   5,  3,  1,  -1, -3, 3,  2,  0,  2,  2,  -2, -2,  2,  0,   -3, 4,  -1, 1,  1,  -3, 1,  -1, -1, 2,
    -2, 1,  3,  2,  2,  -2, -1, 1,  -1, -2, -2, -1, -1, 1,   -5, -2, -2, -5, 1,  1,  2,  1,  -1, 0,  3,  0,   1,  2,   2,  0,  -2, 0,  -1, 2,  0,  -1, -1, -2,
    3,  0,  1,  1,  2,  2,  -1, 0,  0,  -2, -2, 1,  0,  2,   2,  0,  1,  -1, 3,  0,  1,  -2, 2,  2,  0,  -1,  0,  1,   3,  0,  0,  1,  0,  2,  1,  0,  2,  2,
    2,  2,  1,  -3, 2,  1,  -1, 3,  0,  -1, 0,  -3, 0,  1,   0,  2,  1,  0,  0,  1,  0,  2,  -1, 0,  -1, 1,   0,  1,   1,  1,  -3, 3,  -3, 1,  1,  3,  1,  3,
    -2, -2, 0,  -1, 1,  3,  2,  3,  -1, -3, 0,  2,  -2, 3,   2,  3,  8,  4,  3,  1,  0,  3,  1,  -1, 1,  -1,  2,  2,   2,  1,  3,  0,  2,  4,  8,  3,  2,  4,
    2,  -2, 1,  0,  1,  1,  1,  -1, 1,  3,  2,  -1, -2, -2,  1,  2,  0,  -2, 1,  2,  3,  0,  -3, -4, -1, -1,  0,  2,   -1, 0,  0,  -4, -1, -2, -1, 2,  1,  1,
    0,  -3, -3, -2, -3, -1, 0,  -1, -1, 0,  -2, -1, 0,  -2,  -1, 0,  1,  -3, -1, -3, -2, 0,  1,  0,  0,  -4,  -3, 0,   -1, -2, -3, 1,  -1, 1,  2,  0,  -1, -4,
    -2, 4,  -1, 1,  0,  -1, -2, 2,  0,  2,  4,  1,  0,  2,   -2, -1, -2, -3, -3, 0,  2,  0,  0,  -2, -1, 1,   0,  5,   2,  2,  3,  4,  0,  1,  -1, 3,  0,  -3,
    0,  0,  -1, -2, -4, -4, 0,  -6, 2,  0,  4,  0,  -1, 2,   3,  2,  0,  -2, 3,  0,  2,  2,  1,  3,  2,  -2,  -1, 1,   2,  1,  -1, 2,  4,  0,  3,  0,  -1, -1,
    3,  2,  -2, 1,  -1, 2,  2,  2,  3,  -1, 0,  0,  4,  2,   2,  -4, -1, 2,  1,  0,  3,  3,  2,  0,  0,  0,   2,  4,   0,  1,  1,  -1, 2,  0,  0,  0,  1,  2,
    -1, -2, -1, 3,  1,  2,  1,  0,  -1, 2,  1,  1,  1,  -1,  1,  1,  0,  2,  -1, -1, -2, 1,  1,  3,  2,  -1,  1,  2,   -1, -1, 0,  2,  2,  2,  2,  1,  0,  -2,
    -3, 3,  4,  8,  4,  1,  4,  5,  3,  0,  -2, 3,  1,  0,   0,  -2, 0,  2,  -1, -2, 2,  -2, 0,  1,  4,  0,   -4, 3,   1,  1,  -5, 2,  -4, -1, 1,  1,  2,  3,
    0,  1,  2,  0,  -1, -2, -1, 2,  3,  -1, 2,  4,  0,  -1,  1,  0,  2,  0,  -2, 2,  2,  1,  0,  0,  0,  -3,  -1, 4,   3,  3,  -3, 1,  4,  2,  1,  -1, -3, -3,
    -2, 1,  1,  1,  3,  1,  5,  1,  0,  1,  -1, 3,  1,  4,   -2, -3, 2,  0,  1,  3,  0,  3,  0,  2,  -6, 1,   -2, 1,   2,  5,  0,  3,  -6, -3, 0,  -1, -2, 2,
    1,  -2, 2,  5,  6,  -3, -2, 3,  -1, 2,  4,  0,  -2, -2,  4,  5,  3,  1,  1,  -7, 5,  6,  3,  0,  -1, -4,  -1, 1,   6,  1,  0,  -2, -5, -3, 1,  -3, -1, 3,
    -2, 2,  -4, 4,  5,  0,  -5, 4,  -2, -2, 2,  2,  4,  3,   1,  2,  1,  0,  -3, -3, -1, -1, 4,  0,  0,  4,   2,  -3,  0,  1,  2,  0,  -2, -1, 2,  0,  1,  0,
    3,  0,  -2, 2,  4,  4,  -2, -1, -1, 1,  2,  0,  0,  -3,  2,  4,  0,  0,  1,  0,  5,  2,  -1, 1,  0,  2,   0,  3,   -1, -3, 0,  -1, 1,  5,  2,  1,  -1, 0,
    -3, 2,  -1, 2,  2,  1,  1,  2,  -3, -4, -1, 0,  0,  -2,  1,  -2, 3,  2,  2,  -1, -2, 4,  1,  1,  2,  -1,  -2, -4,  2,  3,  2,  -1, 2,  -6, 5,  2,  2,  1,
    2,  -2, 5,  4,  7,  1,  1,  2,  -4, -2, 1,  -1, -2, 1,   -1, 3,  -6, 2,  2,  3,  -1, 4,  -2, -1, 3,  1,   3,  3,   2,  1,  4,  2,  -3, -3, -4, -1, 4,  0,
    0,  4,  1,  -4, 1,  3,  -2, 0,  -2, -1, 1,  3,  2,  0,   2,  1,  0,  2,  5,  1,  1,  0,  1,  -1, 3,  -1,  -1, -1,  -1, 6,  2,  0,  1,  -1, 4,  3,  0,  0,
    0,  3,  0,  5,  0,  2,  1,  -3, 1,  4,  2,  1,  0,  -1,  -1, 1,  -1, 0,  1,  2,  -2, 0,  -2, -4, -1, -2,  0,  1,   -1, -2, 2,  3,  4,  -2, -3, 5,  3,  0,
    3,  -4, 1,  0,  2,  2,  1,  1,  5,  -5, 6,  2,  3,  -1,  2,  -5, 0,  2,  7,  3,  2,  1,  -2, -2, -1, 0,   0,  -1,  -1, 5,  -4, 3,  2,  0,  -3, 2,  0,  0,
    2,  3,  3,  6,  4,  1,  2,  5,  -1, -1, -6, 1,  2,  0,   0,  2,  -1, -3, 0,  0,  0,  1,  0,  -1, 1,  1,   1,  2,   1,  -3, 1,  1,  4,  1,  0,  4,  4,  0,
    2,  0,  -2, -1, -2, 3,  1,  0,  0,  2,  4,  5,  -2, 0,   -1, 3,  -2, 5,  2,  3,  -1, -3, 2,  1,  -1, 2,   0,  -1,  1,  1,  -3, 0,  5,  0,  -2, 1,  -1, -5,
    -3, 0,  -1, 0,  -1, 2,  3,  1,  6,  1,  -2, 4,  4,  0,   1,  -5, 1,  1,  2,  2,  1,  2,  0,  -1, 3,  3,   1,  0,   3,  -6, 0,  0,  8,  4,  2,  1,  -3, -2,
    1,  -1, 1,  -1, 1,  3,  -3, 2,  3,  -2, -4, 2,  2,  3,   3,  4,  1,  5,  2,  1,  4,  4,  -2, -1, -5, 3,   3,  -1,  0,  1,  0,  -4, 1,  3,  -3, 1,  0,  -3,
    3,  2,  0,  1,  3,  0,  2,  2,  5,  0,  -2, 3,  5,  1,   2,  2,  -5, 0,  0,  2,  1,  0,  1,  0,  4,  3,   1,  1,   0,  2,  -2, 5,  2,  2,  0,  -3, -1, 3,
    -3, 5,  -1, 1,  0,  2,  -5, -2, 7,  1,  0,  2,  -3, -4,  -2, -1, -1, -3, -4, 0,  2,  3,  5,  0,  -2, 4,   4,  0,   0,  -2, 1,  4,  -1, 6,  5,  1,  4,  -2,
    3,  4,  -2, -1, 1,  -6, 1,  1,  8,  3,  0,  0,  -2, -3,  2,  -2, 3,  2,  0,  3,  0,  4,  -1, 1,  1,  4,   2,  3,   5,  2,  4,  2,  1,  1,  1,  6,  3,  4,
    0,  2,  2,  -2, 0,  -1, -3, 0,  0,  -1, 2,  3,  1,  -2,  -1, 0,  0,  3,  2,  -1, 5,  1,  0,  -1, 1,  4,   0,  0,   3,  0,  3,  2,  1,  3,  2,  0,  0,  -1,
    -2, 1,  4,  0,  -1, 1,  0,  1,  3,  0,  1,  3,  -1, -4,  0,  -2, 1,  0,  -2, 0,  -1, 3,  -5, 1,  0,  -1,  0,  -1,  2,  3,  3,  -2, 3,  -3, 1,  1,  2,  -1,
    0,  0,  -3, -1, -4, -2, -4, 0,  1,  3,  1,  -1, 0,  -1,  2,  -1, -3, -3, -2, 0,  2,  6,  8,  0,  -1, 1,   3,  1,   -1, 2,  1,  -1, -3, 4,  1,  4,  0,  -2,
    0,  2,  2,  4,  5,  1,  4,  0,  2,  2,  -1, 6,  3,  -1,  2,  1,  1,  -4, 2,  -1, -4, -1, -1, -5, -1, 4,   2,  0,   -2, -1, 3,  3,  4,  -1, 4,  -1, 0,  -1,
    -1, 5,  2,  0,  4,  1,  4,  1,  -1, 2,  1,  0,  0,  -2,  0,  0,  2,  3,  1,  -1, -1, 3,  3,  -1, 1,  1,   -1, -3,  -2, -1, 0,  0,  -5, 1,  0,  2,  -3, -1,
    1,  -3, -1, 1,  -2, 3,  4,  0,  1,  -2, 1,  2,  4,  -2,  1,  2,  -3, -4, -1, -2, 1,  2,  1,  6,  2,  1,   2,  2,   3,  3,  0,  0,  0,  0,  5,  4,  8,  2,
    2,  1,  4,  1,  -2, 1,  1,  5,  3,  2,  1,  6,  -1, 0,   -1, 0,  3,  2,  5,  0,  1,  0,  1,  1,  1,  3,   2,  1,   -1, 2,  1,  -3, 3,  -1, -4, 2,  -2, -5,
    0,  2,  2,  0,  -1, 1,  3,  4,  3,  0,  2,  -1, 2,  -3,  1,  4,  3,  -1, 3,  1,  1,  3,  1,  2,  2,  0,   3,  -4,  0,  -1, 1,  2,  2,  -1, -1, 4,  3,  -1,
    0,  3,  0,  -4, 2,  1,  -1, 0,  -5, -1, 0,  1,  -1, 2,   0,  -2, 2,  0,  0,  4,  1,  1,  2,  -2, 0,  0,   6,  -1,  0,  0,  -5, -5, -1, -1, 1,  3,  1,  5,
    5,  -1, 2,  1,  1,  0,  0,  -1, -1, 1,  3,  6,  9,  3,   2,  1,  1,  0,  0,  0,  2,  3,  2,  4,  0,  2,   1,  2,   0,  0,  2,  4,  6,  0,  4,  0,  -2, 3,
    -3, 3,  1,  -2, -3, 2,  0,  -4, 1,  4,  -3, 2,  0,  -1,  -2, 3,  4,  -1, -1, -1, 2,  3,  2,  -1, 2,  2,   2,  -2,  0,  3,  3,  -2, 1,  1,  1,  5,  -1, 2,
    -2, 1,  1,  -7, 3,  -1, 0,  4,  2,  -3, 0,  2,  4,  -1,  0,  0,  -1, -5, 1,  -1, 0,  0,  -6, 0,  0,  1,   -3, 4,   3,  -1, 1,  1,  -4, 4,  6,  4,  1,  0,
    1,  3,  4,  -1, 1,  -1, -4, -6, 0,  0,  -2, 3,  0,  2,   4,  0,  3,  -1, 1,  -1, 0,  -2, -1, -1, 3,  5,   5,  3,   3,  2,  0,  0,  -2, 3,  2,  1,  2,  5,
    3,  3,  2,  0,  1,  3,  1,  3,  4,  0,  5,  1,  0,  3,   -2, 6,  1,  -1, -3, 4,  2,  -4, 0,  1,  1,  2,   -2, -3,  -1, 1,  6,  -3, -1, -2, 2,  4,  1,  0,
    2,  3,  -1, 0,  0,  3,  3,  0,  0,  -1, 2,  5,  -1, 1,   -1, 2,  3,  -3, 1,  -2, 0,  3,  1,  -1, 2,  4,   2,  0,   -2, -1, 0,  -2, 3,  2,  -4, 2,  -1, 2,
    0,  -2, -2, 4,  0,  -1, 0,  1,  -2, 3,  3,  2,  0,  0,   2,  1,  5,  0,  -2, -1, -3, -4, -1, -3, 3,  3,   1,  2,   1,  -1, 2,  -3, -2, -3, 2,  -1, -4, -2,
    4,  3,  6,  0,  -1, 1,  -3, 2,  -3, 2,  1,  2,  2,  4,   -2, 2,  -2, 1,  -2, 0,  -1, 2,  3,  -1, -2, 4,   1,  4,   2,  -1, 1,  2,  3,  2,  -2, 3,  1,  1,
    0,  -2, 2,  1,  0,  0,  3,  6,  2,  -2, 2,  3,  2,  0,   -3, -3, -2, 1,  0,  0,  2,  3,  1,  2,  0,  -1,  2,  1,   -1, 0,  3,  -1, 3,  1,  1,  2,  1,  0,
    2,  0,  0,  1,  -1, -1, 3,  -1, -1, -4, 0,  4,  1,  1,   3,  3,  1,  2,  -1, 2,  -1, -1, 2,  0,  -1, 0,   0,  0,   2,  2,  3,  2,  -2, 3,  -2, -2, 3,  0,
    -2, -2, 2,  1,  2,  1,  1,  -1, -3, 0,  -2, 0,  3,  -2,  -1, 1,  0,  1,  4,  0,  1,  2,  3,  -2, 3,  2,   2,  2,   2,  0,  -1, 3,  -4, 2,  -1, 3,  2,  -2,
    -2, 1,  1,  5,  2,  -1, 2,  1,  1,  1,  -2, -1, 1,  -3,  1,  -3, -1, 2,  0,  0,  -1, 2,  1,  -4, 2,  -1,  1,  -3,  -4, -3, -3, 3,  1,  0,  2,  -1, 1,  0,
    -3, -1, 1,  0,  -3, 0,  0,  1,  1,  -1, 0,  -1, 1,  0,   3,  1,  1,  1,  2,  1,  1,  -4, -1, -2, -3, 1,   0,  2,   1,  2,  1,  2,  -5, -3, -2, 2,  1,  2,
    -1, -2, 0,  1,  -1, 0,  1,  1,  -4, 1,  -2, -1, 3,  1,   0,  1,  1,  -1, 2,  -1, 3,  0,  -2, -2, -3, 0,   3,  1,   0,  0,  -1, -2, 1,  -1, 3,  -1, 3,  -3,
    3,  3,  -1, 0,  3,  -3, -4, 0,  1,  1,  2,  4,  3,  1,   -1, 1,  2,  3,  5,  1,  0,  1,  0,  -1, 2,  -2,  4,  -1,  2,  -2, -3, 4,  -1, -1, 3,  4,  -2, -1,
    4,  1,  3,  -1, -2, -1, -2, 6,  0,  -1, 2,  3,  0,  2,   0,  2,  1,  0,  1,  2,  4,  3,  2,  1,  1,  2,   3,  2,   1,  1,  0,  4,  2,  3,  2,  1,  2,  0,
    -1, 4,  0,  2,  0,  2,  0,  0,  -3, -3, -1, 1,  0,  2,   -4, -2, 1,  0,  1,  -1, -1, 1,  -1, 0,  2,  1,   1,  1,   0,  1,  -2, -1, 4,  3,  7,  4,  0,  0,
    2,  2,  1,  2,  2,  0,  1,  1,  2,  1,  3,  1,  3,  0,   0,  4,  -2, 2,  4,  -2, -3, 3,  -2, -3, 1,  0,   5,  0,   -2, 3,  2,  0,  1,  2,  1,  0,  0,  -3,
    -1, -2, 1,  0,  0,  -4, -4, -1, -2, 0,  3,  2,  -2, -4,  -1, -2, 0,  -3, -2, -3, -4, 3,  0,  2,  3,  -1,  1,  1,   0,  -1, 2,  -1, 0,  -1, 0,  2,  3,  0,
    -2, 0,  3,  1,  1,  0,  2,  3,  0,  2,  2,  0,  -2, -1,  -2, 1,  3,  3,  0,  1,  -1, 2,  -5, -4, -1, 1,   -3, 1,   -3, 1,  1,  3,  2,  -1, -2, -1, -3, 2,
    -3, -3, 2,  0,  0,  0,  0,  -1, 1,  2,  6,  2,  -1, -1,  -1, -2, 2,  1,  0,  0,  0,  0,  1,  -1, -1, -1,  2,  -3,  1,  3,  1,  1,  3,  -1, 0,  2,  -2, 1,
    0,  0,  2,  0,  1,  2,  1,  0,  0,  -2, -2, -1, -1, 3,   3,  -1, 1,  2,  3,  -4, -3, 0,  -2, 1,  5,  4,   1,  0,   1,  0,  1,  -2, -5, -1, 0,  0,  2,  1,
    4,  2,  1,  2,  0,  0,  3,  -1, 2,  0,  -1, 4,  3,  0,   1,  1,  3,  0,  1,  2,  2,  2,  1,  2,  2,  0,   -1, -1,  -2, 5,  2,  1,  0,  1,  -2, 2,  1,  -3,
    -3, 2,  -2, 2,  0,  3,  0,  0,  1,  -3, -2, -1, 2,  3,   -1, -2, -2, -1, 1,  -1, 0,  3,  4,  -1, 6,  2,   1,  -1,  0,  1,  2,  1,  1,  1,  1,  1,  2,  2,
    3,  1,  2,  -2, 3,  3,  1,  1,  0,  2,  5,  -1, -4, -4,  4,  0,  0,  -1, -1, 0,  4,  -1, 1,  5,  2,  2,   2,  1,   3,  2,  -1, 1,  4,  0,  5,  2,  -3, 0,
    3,  3,  0,  0,  0,  5,  2,  2,  3,  4,  1,  0,  2,  3,   1,  -5, 1,  4,  -2, -1, 0,  2,  -2, 7,  -1, 3,   -3, -1,  0,  0,  -3, 0,  0,  3,  0,  -3, 1,  -2,
    1,  2,  -1, 3,  -3, 0,  -1, 0,  1,  0,  -2, -2, 1,  3,   2,  4,  5,  5,  1,  -1, -5, -8, -2, 0,  5,  -1,  4,  2,   0,  4,  1,  4,  2,  -3, -3, 1,  3,  1,
    -3, 0,  4,  1,  2,  -2, -2, -3, 0,  2,  1,  3,  -2, 1,   -2, -1, 2,  5,  6,  -1, 2,  -3, 0,  0,  2,  0,   -2, -2,  4,  -1, 2,  2,  -3, -1, 3,  -3, 1,  8,
    2,  1,  0,  1,  1,  1,  -2, -1, 4,  0,  3,  0,  2,  -1,  1,  1,  0,  2,  -1, 6,  0,  2,  1,  5,  1,  1,   0,  1,   -3, -1, 3,  4,  -2, -1, 0,  2,  -2, 4,
    -2, 3,  -1, -1, 2,  -1, -3, -1, 3,  1,  2,  0,  1,  -4,  -3, 3,  0,  3,  -1, -1, -2, 3,  -1, 3,  1,  -1,  2,  1,   -1, 3,  5,  2,  2,  1,  -1, -6, 0,  -1,
    3,  0,  3,  3,  0,  2,  -5, 3,  -2, -5, -2, 0,  0,  1,   -2, 4,  1,  2,  0,  -2, -1, -3, 1,  -2, -2, 3,   1,  -1,  0,  0,  3,  4,  5,  0,  0,  -2, 0,  -2,
    0,  1,  0,  -3, 3,  2,  -1, 4,  -6, -2, 2,  -4, 1,  5,   3,  -1, 0,  2,  1,  0,  -3, 1,  2,  3,  2,  2,   -1, 1,   1,  3,  1,  2,  -1, 4,  1,  3,  2,  3,
    0,  -2, 2,  0,  0,  -3, 0,  3,  -3, 0,  1,  1,  -2, 2,   -3, 2,  0,  0,  0,  -1, -4, 1,  1,  1,  0,  3,   4,  -2,  -3, 0,  -3, 3,  -5, -2, -5, 0,  1,  0,
    -1, 0,  0,  2,  1,  0,  6,  4,  0,  1,  -5, -4, -3, 1,   0,  -2, 1,  1,  2,  6,  -2, 4,  -2, -4, -4, -1,  -2, -2,  0,  5,  -1, 0,  -3, -3, 0,  -2, -1, -4,
    -2, 3,  2,  2,  -1, 0,  4,  4,  2,  1,  1,  -4, -1, 3,   -1, 1,  -1, -1, 4,  0,  -1, 3,  -5, -3, 4,  0,   -2, 4,   5,  0,  -4, 1,  2,  0,  -3, 4,  3,  2,
    2,  3,  -1, 1,  -2, 3,  1,  5,  0,  1,  3,  1,  3,  1,   0,  0,  1,  0,  1,  -4, 0,  4,  -2, 1,  2,  2,   0,  2,   0,  4,  -4, 1,  1,  -3, -2, 1,  2,  0,
    -1, 0,  2,  -2, -1, 0,  1,  4,  -1, 0,  -4, 0,  3,  2,   2,  -1, 0,  4,  2,  2,  2,  2,  1,  1,  -2, -5,  -4, 1,   -2, -3, 5,  3,  3,  5,  -2, 1,  -1, -4,
    0,  3,  -3, 0,  -1, 6,  0,  0,  -1, -1, 1,  -2, 0,  -2,  -1, 4,  3,  0,  3,  -1, 2,  7,  5,  -1, 2,  -1,  -2, 4,   2,  1,  -3, 1,  3,  1,  1,  1,  -4, -1,
    3,  3,  -1, 6,  5,  0,  -2, 6,  3,  1,  -2, 0,  5,  1,   3,  2,  1,  4,  0,  4,  1,  2,  -2, 2,  2,  4,   5,  1,   1,  2,  0,  0,  -2, -1, 0,  4,  1,  -2,
    0,  3,  0,  1,  -1, 5,  0,  0,  1,  -4, -2, 2,  2,  -2,  1,  1,  1,  -1, 3,  0,  1,  3,  0,  1,  -3, -4,  -3, 1,   2,  0,  1,  5,  3,  3,  1,  2,  1,  1,
    -1, -4, -4, -1, 0,  -2, 3,  2,  2,  4,  -3, -2, 0,  -5,  -4, 4,  0,  -1, -4, 6,  2,  -1, 2,  -1, 0,  0,   0,  2,   0,  3,  2,  2,  1,  0,  -1, 7,  10, 2,
    2,  -2, 0,  2,  1,  -1, 5,  2,  3,  0,  -2, 1,  1,  1,   3,  3,  3,  4,  -2, 1,  0,  4,  -2, -1, 1,  -1,  3,  1,   1,  0,  -1, -2, -4, 1,  2,  0,  -1, 2,
    -2, 2,  0,  -1, -1, 4,  2,  2,  2,  0,  1,  2,  3,  -1,  2,  1,  -1, 2,  -4, 1,  3,  1,  3,  4,  0,  -2,  -1, -1,  -2, -3, -2, -2, 1,  3,  1,  3,  2,  -3,
    -2, 0,  -3, -4, -3, 0,  0,  1,  -1, -4, -2, -3, 2,  1,   0,  0,  1,  1,  1,  0,  2,  -2, 5,  5,  3,  2,   -1, 5,   0,  -4, 2,  1,  0,  7,  5,  3,  1,  2,
    0,  2,  -2, 2,  1,  2,  7,  6,  3,  2,  1,  -1, 3,  2,   2,  2,  2,  2,  2,  0,  1,  2,  2,  2,  0,  -2,  1,  1,   3,  5,  1,  3,  -2, 1,  -2, 3,  1,  -1,
    -2, 0,  4,  -1, 3,  0,  -1, -4, -2, 2,  4,  0,  0,  1,   2,  2,  0,  -3, 1,  7,  0,  -1, 0,  0,  1,  2,   2,  -2,  4,  1,  -1, 4,  -4, 1,  2,  2,  3,  0,
    -1, 0,  -1, -2, -3, -2, -1, -1, 2,  2,  1,  4,  0,  -4,  -3, 1,  0,  -6, -4, 0,  0,  -1, 3,  -3, -5, 1,   2,  4,   1,  -3, -1, 1,  -3, -2, 2,  -3, 5,  5,
    2,  -1, 0,  3,  1,  0,  0,  -1, 0,  4,  2,  4,  2,  3,   -1, 3,  -3, 2,  -1, -1, 3,  4,  1,  0,  2,  1,   2,  1,   3,  0,  2,  3,  2,  1,  0,  4,  3,  0,
    -3, -3, -2, 0,  1,  1,  3,  1,  -3, -3, 0,  5,  2,  1,   -2, 0,  5,  -2, 0,  0,  -2, -2, -2, 1,  3,  0,   4,  2,   4,  3,  2,  -1, 3,  4,  2,  -2, 0,  1,
    3,  5,  2,  -2, 4,  -2, 0,  1,  -2, 0,  2,  0,  2,  -2,  1,  -1, 0,  0,  -2, -1, 0,  -2, 1,  2,  -2, 3,   2,  -5,  -3, 0,  -1, -2, -2, 0,  0,  -2, 0,  -2,
    -3, 0,  0,  3,  2,  -1, -1, 2,  -1, -1, 4,  -1, 3,  4,   3,  3,  -1, 1,  -1, 1,  -1, -1, 1,  5,  3,  2,   4,  4,   0,  3,  -1, 2,  -1, 1,  4,  2,  2,  2,
    2,  -1, 5,  2,  1,  0,  1,  4,  0,  -1, -3, 1,  2,  -1,  -4, -4, 1,  -1, 2,  -1, 1,  -3, -3, -1, -1, 1,   3,  -2,  -1, 1,  5,  -5, -1, -1, -1, 0,  -4, 2,
    0,  2,  3,  -1, 3,  -1, 0,  2,  4,  6,  4,  -1, 1,  0,   4,  3,  1,  -1, 1,  1,  -2, 1,  -4, -3, 1,  1,   -1, 1,   -1, -3, -1, 1,  0,  -4, 1,  -1, 0,  2,
    -4, 1,  2,  -3, 2,  -2, -5, -1, -5, 0,  2,  -2, -1, -3,  -4, 0,  0,  1,  4,  -1, -1, 1,  -4, -1, 1,  0,   3,  4,   2,  0,  -2, 0,  0,  -1, -2, -1, 2,  4,
    3,  1,  1,  0,  -1, 2,  3,  4,  -1, 2,  5,  2,  2,  2,   3,  0,  4,  3,  1,  0,  -1, 3,  -2, 0,  -2, 1,   3,  0,   -3, -6, -1, -1, 0,  0,  0,  -2, -3, -4,
    -1, 2,  2,  -3, -1, -1, 2,  -1, -1, -3, -2, -1, -6, 2,   1,  2,  1,  -1, 5,  -1, 0,  0,  3,  5,  1,  0,   -2, -2,  1,  3,  2,  3,  3,  2,  -1, 2,  -2, -4,
    1,  2,  1,  -2, -4, -3, 0,  0,  -1, -3, 0,  0,  -1, 0,   0,  0,  1,  -2, -5, -2, -4, -3, -5, 1,  0,  -1,  1,  -2,  -4, 1,  0,  -1, 4,  3,  5,  1,  -4, 2,
    0,  0,  3,  4,  -1, 3,  1,  2,  0,  1,  1,  1,  2,  4,   1,  1,  3,  1,  2,  3,  4,  2,  0,  1,  5,  5,   2,  2,   4,  -1, 5,  5,  5,  2,  1,  0,  1,  -1,
    1,  3,  2,  1,  -1, 0,  2,  3,  2,  4,  2,  -2, 0,  1,   0,  -1, 2,  -3, -3, 1,  1,  2,  2,  -2, -5, -2,  0,  -4,  1,  4,  7,  2,  0,  2,  -2, -2, -2, 1,
    0,  -2, -1, 1,  -2, 1,  4,  0,  1,  2,  2,  0,  4,  0,   -2, 1,  1,  1,  0,  2,  0,  4,  -1, 2,  2,  0,   -4, -2,  -1, 2,  -2, -2, -1, 2,  0,  0,  -1, -3,
    0,  0,  0,  0,  0,  1,  2,  1,  -1, 3,  -1, 3,  0,  -2,  -1, 1,  3,  -1, 2,  3,  1,  0,  -3, -1, -2, -2,  0,  0,   2,  3,  2,  1,  0,  -1, 0,  0,  5,  5,
    -1, -1, 0,  0,  2,  -2, 1,  1,  3,  4,  0,  0,  3,  0,   0,  1,  1,  -1, -2, 0,  2,  3,  3,  4,  2,  -3,  -2, 3,   1,  1,  1,  -1, -5, 3,  1,  3,  5,  -1,
    -6, -2, 0,  -1, 1,  4,  5,  2,  1,  4,  -1, -3, -2, 1,   -2, -1, -2, -1, 1,  3,  3,  -1, 0,  0,  1,  0,   4,  -1,  -1, 2,  2,  2,  1,  3,  1,  3,  1,  3,
    -1, 2,  -3, -2, 0,  3,  -2, -3, -3, 0,  1,  2,  -1, -3,  1,  -1, -1, 0,  3,  0,  -1, 1,  0,  1,  -2, 3,   1,  -2,  -1, 2,  2,  0,  2,  3,  0,  2,  -3, 0,
    -1, -2, 1,  0,  3,  1,  0,  2,  2,  -1, -1, -1, 2,  4,   -3, -2, 1,  0,  1,  -4, 0,  0,  3,  3,  1,  -1,  4,  2,   -3, -1, 1,  0,  -2, 0,  4,  3,  3,  3,
    0,  -3, -2, 1,  2,  -1, 0,  -1, -1, 4,  2,  6,  4,  -2,  -3, 0,  1,  0,  3,  6,  4,  2,  1,  4,  1,  -4,  0,  -1,  -1, -2, -1, 0,  1,  2,  3,  -1, -1, 0,
    2,  2,  5,  -1, -2, 3,  2,  4,  0,  3,  0,  2,  3,  4,   -1, 1,  -3, -3, 0,  3,  -1, -3, -2, 1,  3,  2,   0,  -2,  -1, 2,  -2, 2,  3,  0,  -1, 2,  -1, -1,
    -2, 2,  1,  -1, 0,  3,  -1, 1,  4,  1,  2,  1,  -2, 0,   1,  -4, 1,  1,  1,  0,  -1, 3,  1,  1,  1,  1,   4,  4,   -1, 0,  2,  0,  -1, -2, 0,  3,  7,  2,
    2,  2,  3,  2,  -2, -3, 3,  -1, -3, -1, 2,  1,  -1, 0,   -1, -3, -3, 2,  0,  0,  0,  -3, -2, 3,  1,  6,   3,  0,   -2, 0,  -3, 1,  4,  3,  4,  6,  0,  2,
    -1, -4, -3, -1, 1,  -3, -1, -2, 0,  0,  3,  -1, 0,  2,   0,  0,  4,  0,  -2, 1,  1,  2,  1,  3,  2,  1,   3,  3,   0,  0,  -2, -2, -2, 2,  -1, -4, -1, 0,
    1,  2,  -3, -3, 0,  3,  1,  -1, 3,  3,  -4, 2,  2,  -1,  -1, 1,  2,  -1, 0,  2,  0,  1,  5,  2,  0,  2,   -3, -1,  1,  -1, 3,  1,  0,  0,  1,  0,  1,  -2,
    1,  -1, 2,  4,  -2, -3, 1,  1,  -2, -1, 0,  3,  4,  1,   -3, 0,  0,  0,  1,  -3, 5,  1,  -4, -3, 4,  3,   0,  1,   0,  -5, -2, 3,  1,  -2, -2, -2, -3, 3,
    0,  6,  5,  -2, -1, 1,  -1, -1, 2,  4,  6,  4,  0,  1,   -2, -2, -4, 0,  -1, 0,  -2, 0,  0,  -1, 4,  0,   -1, 0,   1,  0,  3,  -3, 0,  2,  1,  2,  1,  3,
    2,  0,  3,  4,  0,  -1, -2, -2, -2, 2,  -1, -3, -1, -2,  -1, 1,  -1, -3, -1, 1,  1,  -1, 3,  3,  -1, 0,   2,  -3,  -3, 1,  1,  0,  -2, 0,  2,  1,  3,  3,
    1,  0,  -2, 0,  0,  0,  2,  0,  0,  -1, 2,  2,  -1, -1,  0,  0,  1,  3,  1,  -1, 2,  1,  -1, 0,  0,  3,   5,  4,   3,  1,  -2, 2,  -1, 0,  1,  1,  1,  -2,
    -1, -1, -1, -5, -1, -2, 2,  -2, 0,  -2, -2, -1, 0,  -1,  -3, -5, 0,  0,  -1, 0,  -1, 0,  1,  -3, -2, -1,  -3, 0,   0,  -3, -1, -1, 1,  -3, -1, 0,  -1, -3,
    -1, -2, 0,  0,  -1, 0,  -2, 0,  -2, -1, -1, 0,  0,  0,   0,  1,  -1, 0,  -1, 0,  0,  -3, 1,  -2, -1, -2,  -2, -3,  2,  -1, 1,  3,  -2, 0,  2,  0,  2,  -3,
    0,  1,  -2, 0,  0,  1,  0,  1,  -3, 0,  -1, 2,  2,  1,   0,  1,  -1, 3,  3,  4,  -1, -1, 3,  1,  -1, -1,  -2, 2,   2,  1,  0,  -4, -1, 2,  -3, -2, 0,  -3,
    -1, -1, 0,  1,  2,  -1, -2, 3,  1,  1,  3,  0,  4,  2,   0,  2,  2,  1,  2,  1,  4,  3,  0,  1,  2,  2,   3,  -1,  1,  1,  1,  3,  4,  1,  0,  1,  1,  3,
    4,  3,  1,  1,  0,  3,  2,  0,  2,  1,  4,  1,  0,  3,   5,  3,  3,  0,  0,  2,  2,  1,  2,  1,  2,  4,   0,  1,   2,  1,  1,  3,  2,  3,  6,  4,  4,  1,
    2,  -1, -1, -2, 0,  2,  0,  4,  0,  -1, 1,  0,  2,  0,   1,  4,  0,  1,  0,  1,  1,  0,  1,  2,  2,  0,   2,  1,   -1, -1, -2, 3,  2,  5,  4,  3,  6,  5,
    2,  1,  0,  4,  5,  0,  0,  0,  4,  5,  1,  2,  3,  -1,  1,  -1, 4,  3,  2,  -1, -1, 0,  2,  -1, 3,  0,   0,  1,   0,  0,  -1, 0,  -2, 3,  3,  1,  1,  3,
    1,  0,  4,  -2, 0,  -2, 0,  1,  0,  0,  1,  0,  1,  0,   -1, -1, -2, -1, -3, -1, 0,  0,  3,  3,  0,  3,   -1, 2,   -2, -2, 1,  1,  -1, 0,  2,  -1, 0,  0,
    2,  1,  -1, -1, 0,  1,  0,  0,  3,  2,  1,  0,  -1, -1,  3,  1,  -1, -1, 0,  0,  2,  1,  0,  2,  0,  2,   0,  0,   -2, 0,  1,  2,  0,  1,  2,  0,  -2, 1,
    1,  -1, 1,  3,  -3, -2, -1, 1,  3,  4,  3,  2,  1,  3,   1,  -1, 2,  3,  -1, 0,  -1, -2, 0,  1,  -1, -3,  0,  -3,  0,  -2, 3,  3,  1,  0,  0,  1,  0,  0,
    2,  1,  -1, 0,  1,  2,  -1, -1, -1, 1,  -3, 2,  1,  -1,  0,  1,  0,  1,  -1, -2, -2, 0,  -1, 0,  1,  1,   0,  -3,  0,  -1, -2, -2, -1, 0,  -2, -1, 1,  -1,
    -2, -2, 0,  1,  0,  -1, 0,  3,  -3, 0,  2,  0,  1,  1,   0,  -1, 0,  0,  -1, -1, -2, -1, 0,  1,  3,  -2,  -2, -2,  1,  -1, -3, -1, 0,  1,  1,  1,  1,  1,
    2,  1,  3,  -1, -1, -2, -3, 1,  1,  1,  -3, -1, -3, 1,   -3, -1, 0,  1,  -2, -2, -3, 1,  2,  3,  1,  -1,  2,  2,   0,  -1, -4, 1,  0,  0,  -2, -2, 1,  1,
    -2, -5, 2,  -2, -2, -1, 3,  2,  2,  0,  2,  -1, 4,  0,   4,  -1, 1,  -1, -2, 1,  3,  4,  0,  2,  3,  -1,  2,  3,   3,  2,  2,  2,  2,  -1, -1, 1,  0,  2,
    1,  1,  1,  3,  4,  2,  1,  0,  0,  1,  -2, 1,  3,  1,   1,  3,  0,  1,  -1, 2,  1,  1,  0,  1,  2,  -1,  2,  4,   0,  5,  1,  1,  0,  1,  0,  0,  0,  3,
    6,  2,  2,  4,  2,  0,  -2, 0,  -1, 0,  2,  3,  -1, -1,  1,  0,  2,  0,  1,  0,  -3, 1,  1,  0,  -2, -1,  0,  2,   3,  0,  2,  1,  0,  0,  1,  0,  3,  3,
    3,  0,  5,  3,  1,  1,  0,  5,  4,  2,  -1, -1, 3,  5,   1,  -2, 4,  -1, 2,  0,  3,  4,  -2, -4, 1,  2,   -2, 0,   -5, 1,  -3, -5, -1, 1,  0,  2,  1,  -3,
    -2, 3,  0,  -3, 3,  0,  2,  2,  0,  2,  1,  1,  5,  1,   3,  -1, -1, -1, 3,  1,  1,  3,  0,  2,  3,  4,   -2, 2,   0,  -1, -1, 1,  4,  2,  4,  -2, -2, 0,
    2,  0,  0,  -1, 2,  3,  1,  -1, 0,  0,  -1, -3, 2,  3,   0,  -2, -2, 0,  -3, -1, 2,  4,  1,  1,  2,  3,   0,  -1,  0,  -2, 1,  -2, 0,  2,  0,  -2, -1, 0,
    1,  0,  0,  3,  -2, 2,  2,  1,  1,  -3, -2, -1, -1, 1,   2,  2,  0,  -2, 2,  3,  2,  -1, 2,  3,  4,  4,   1,  -1,  -2, -3, -4, 4,  2,  3,  0,  1,  -3, -7,
    0,  -1, -3, -2, -4, 2,  -4, -2, -2, 0,  0,  1,  2,  -3,  -3, 3,  0,  -1, 0,  -1, 3,  2,  1,  4,  3,  4,   3,  -1,  2,  0,  -1, -4, 2,  2,  -1, 3,  1,  0,
    2,  2,  0,  2,  1,  -3, -1, -1, 4,  2,  1,  -1, -1, -1,  2,  1,  3,  1,  2,  3,  -1, 0,  2,  2,  -1, -1,  1,  4,   1,  -3, -3, 1,  -1, -1, 1,  3,  1,  0,
    2,  2,  1,  2,  -2, 0,  2,  0,  2,  2,  2,  0,  -2, -1,  2,  0,  -1, 0,  -1, 1,  -3, 1,  3,  -1, -1, -2,  0,  1,   1,  4,  2,  1,  3,  0,  3,  -1, 2,  3,
    2,  3,  0,  -1, 0,  -2, -3, 5,  0,  3,  0,  1,  -1, 1,   2,  -2, -3, 0,  -7, 0,  -2, -1, -3, 1,  2,  1,   2,  0,   -2, -1, -2, 0,  2,  2,  2,  2,  1,  2,
    2,  0,  4,  -2, 0,  0,  1,  -3, 4,  1,  -1, 3,  0,  1,   2,  3,  -2, 2,  3,  1,  0,  2,  3,  3,  0,  -2,  0,  0,   2,  0,  3,  1,  1,  2,  -1, 0,  0,  0,
    0,  0,  2,  4,  -1, 0,  -3, -1, -2, 0,  1,  3,  0,  -2,  0,  1,  0,  1,  -3, 0,  4,  2,  2,  2,  1,  -1,  -1, 0,   0,  2,  2,  0,  0,  1,  0,  2,  0,  -3,
    0,  -2, -1, 2,  1,  3,  0,  1,  0,  0,  2,  -3, 4,  6,   2,  2,  0,  -1, 0,  -1, -3, 2,  1,  3,  1,  3,   -3, 0,   3,  -2, -4, 0,  -7, 1,  -3, 2,  -7, -2,
    0,  -2, 3,  -2, -3, -4, 0,  1,  2,  1,  -3, 2,  1,  1,   -1, -2, 1,  -1, -1, -1, 0,  -2, 1,  2,  0,  3,   0,  4,   1,  -1, 0,  0,  1,  1,  -1, 0,  1,  4,
    0,  -2, 0,  -1, 2,  1,  1,  -2, 2,  2,  0,  0,  1,  -1,  -1, -1, 3,  3,  0,  2,  0,  1,  0,  0,  -1, 1,   -1, 2,   2,  2,  1,  0,  -4, 0,  2,  2,  1,  0,
    -2, -2, -3, 0,  3,  1,  2,  3,  -2, 0,  0,  3,  1,  -4,  -1, -1, -4, 0,  1,  3,  -3, -2, 1,  0,  1,  0,   2,  4,   3,  1,  0,  -3, -1, -2, -3, 3,  -1, 1,
    0,  3,  -1, -1, 5,  -2, -2, 1,  -4, 0,  -2, -1, -5, 0,   -4, -3, 6,  -3, -4, -2, -2, -1, 2,  0,  0,  -1,  3,  3,   0,  -1, -2, -1, -1, -1, 0,  -2, 1,  -1,
    2,  5,  2,  1,  3,  2,  -1, 0,  1,  3,  -4, 2,  0,  2,   4,  -2, -1, -2, 2,  0,  0,  2,  0,  2,  0,  0,   -1, 0,   1,  0,  1,  3,  1,  2,  0,  1,  2,  -2,
    -1, 2,  0,  4,  3,  4,  2,  0,  -4, 2,  4,  1,  1,  3,   1,  0,  0,  1,  4,  4,  0,  3,  0,  2,  -1, 1,   0,  -1,  -2, 0,  -3, 2,  1,  1,  0,  -2, 1,  2,
    0,  0,  2,  3,  4,  2,  -1, -3, -3, -3, -4, 2,  0,  2,   2,  3,  -2, 1,  -2, 1,  -1, 2,  2,  0,  -3, -1,  2,  1,   4,  -1, 0,  1,  1,  6,  2,  -2, -5, 0,
    5,  2,  2,  0,  -3, -1, 5,  5,  1,  0,  -1, 3,  0,  4,   -2, -4, 1,  -1, 0,  2,  6,  -6, -3, 3,  -1, 0,   -4, 0,   -1, 1,  3,  3,  2,  2,  -2, -1, -2, -4,
    0,  2,  0,  -2, -2, 0,  -1, -1, -4, 3,  2,  -2, -1, 1,   6,  3,  2,  1,  2,  1,  4,  1,  -2, 0,  1,  2,   1,  3,   0,  -1, 0,  -1, 3,  1,  1,  3,  -2, 2,
    5,  -1, 3,  4,  9,  1,  10, 3,  2,  -3, 3,  -1, -3, 1,   2,  0,  2,  2,  5,  2,  2,  2,  4,  1,  -1, -1,  -4, 3,   -1, -2, -1, 1,  -1, -1, -1, 1,  5,  -1,
    -4, -2, 3,  0,  6,  0,  -2, -1, 0,  6,  1,  -1, -5, 1,   4,  1,  3,  1,  0,  -3, 1,  3,  1,  2,  -2, 1,   2,  3,   -5, -2, 0,  -4, -5, 4,  4,  -7, -4, 2,
    0,  0,  -3, 0,  0,  1,  3,  3,  4,  4,  -2, 0,  -2, -5,  2,  2,  1,  0,  -3, -2, 0,  -2, -4, 1,  3,  0,   0,  -1,  3,  1,  1,  3,  5,  2,  0,  2,  -2, 2,
    0,  2,  -1, 0,  -1, 0,  0,  0,  0,  3,  4,  2,  -3, 0,   3,  -2, 1,  4,  5,  3,  8,  7,  0,  -2, 2,  -1,  -2, 1,   3,  0,  2,  3,  4,  0,  3,  1,  3,  4,
    0,  -1, -1, 3,  -3, 0,  -3, 1,  -3, 1,  -2, -1, 7,  0,   -3, -1, 1,  -1, 4,  -2, -2, 1,  2,  5,  3,  -2,  -6, 3,   2,  1,  3,  3,  -1, -3, 1,  2,  -2, 0,
    2,  0,  1,  0,  -1, -4, 1,  -4, -2, 2,  2,  -4, -3, 1,   0,  -2, -1, 0,  0,  0,  0,  3,  5,  4,  0,  -2,  -2, -5,  1,  3,  1,  0,  -3, -1, 1,  -2, -4, 1,
    4,  3,  -3, -1, 5,  3,  2,  0,  1,  2,  0,  3,  0,  -1,  2,  2,  -1, -2, 1,  1,  0,  -2, 1,  2,  5,  0,   -2, 2,   2,  -1, 1,  2,  2,  5,  7,  6,  -1, 0,
    1,  -1, 0,  0,  2,  0,  0,  2,  8,  2,  2,  0,  5,  1,   1,  0,  -2, 2,  0,  2,  -3, 1,  -2, 1,  -4, 1,   6,  2,   -1, 0,  0,  0,  6,  -2, 0,  3,  0,  4,
    3,  0,  -5, -2, 3,  1,  0,  1,  -2, -2, 3,  3,  -2, 0,   1,  3,  0,  1,  1,  0,  2,  -1, -2, -1, 1,  -3,  -4, 0,   -1, -3, -1, 3,  0,  1,  1,  1,  4,  4,
    0,  0,  -4, -6, 0,  1,  1,  1,  -2, -1, 1,  -1, -3, 1,   4,  3,  -1, -1, 5,  1,  4,  3,  3,  4,  -1, 5,   0,  -3,  2,  1,  4,  1,  5,  -2, -2, -1, 0,  3,
    3,  -1, -1, 1,  -1, -2, 5,  1,  3,  2,  6,  6,  0,  0,   0,  -2, -1, 0,  4,  1,  0,  1,  5,  3,  3,  1,   2,  0,   -3, 3,  -3, 1,  -4, 2,  -1, -1, -2, -1,
    -1, 1,  6,  1,  -1, 2,  2,  3,  5,  0,  -3, 4,  1,  2,   6,  2,  -2, 0,  5,  3,  4,  3,  1,  -2, 3,  3,   0,  2,   0,  2,  0,  3,  -1, -1, 0,  -3, -1, 1,
    4,  -3, -1, -1, -1, -1, -1, 3,  2,  1,  1,  2,  3,  3,   1,  2,  -3, -5, 1,  1,  3,  1,  -1, -1, 3,  0,   -3, 0,   0,  -1, -2, -1, 2,  3,  2,  2,  4,  6,
    0,  2,  1,  0,  2,  0,  5,  1,  0,  -2, -2, -2, 0,  3,   3,  0,  0,  0,  -2, -3, 3,  1,  1,  -1, 4,  4,   -1, -2,  1,  -4, -2, -1, 3,  2,  -1, 2,  5,  1,
    2,  1,  2,  0,  -2, 1,  -2, 1,  -5, -1, 1,  -3, 0,  -3,  -1, 2,  1,  -1, 2,  0,  -1, 1,  -1, 6,  0,  0,   1,  -2,  2,  1,  3,  3,  0,  2,  -1, 1,  0,  -1,
    1,  0,  1,  2,  -3, 1,  2,  2,  -1, 0,  3,  3,  4,  -1,  0,  1,  4,  2,  3,  7,  -1, 2,  3,  -1, 4,  2,   -3, -1,  8,  3,  1,  -1, -1, 2,  3,  0,  4,  0,
    3,  1,  2,  -1, 0,  0,  3,  0,  -3, 5,  4,  -2, -1, -3,  -1, 0,  -2, 1,  2,  -1, -4, -2, 1,  1,  2,  3,   3,  0,   0,  -1, -6, -2, 4,  -5, -1, 1,  2,  0,
    5,  0,  2,  -1, 4,  4,  -1, 1,  1,  0,  2,  4,  -4, 1,   -4, 3,  2,  4,  6,  2,  3,  0,  0,  1,  1,  -5,  1,  2,   1,  3,  3,  1,  2,  -1, -2, 2,  2,  8,
    -1, -4, 1,  -7, 2,  1,  2,  1,  -3, -3, -1, 3,  2,  -4,  -2, -1, 3,  2,  -1, 3,  -3, 1,  -1, -1, 2,  1,   0,  -2,  -2, 0,  3,  0,  3,  10, 1,  1,  1,  -2,
    5,  2,  -4, 2,  7,  1,  1,  -2, -1, 0,  1,  3,  4,  -1,  1,  -1, 1,  0,  1,  2,  1,  3,  -1, 5,  2,  -2,  -3, 1,   0,  2,  -1, 1,  1,  2,  -4, 4,  6,  1,
    -1, 2,  -2, -1, -3, -3, -5, -3, 4,  -2, -4, 2,  3,  1,   6,  -4, 1,  2,  3,  5,  2,  2,  1,  1,  2,  2,   -3, 1,   -2, -1, 3,  4,  3,  1,  0,  2,  -2, 0,
    0,  -2, -1, 0,  1,  2,  -1, 0,  1,  -3, -1, -1, 1,  2,   -1, -1, 2,  -3, 1,  2,  3,  1,  -3, -2, -2, 1,   1,  -1,  1,  0,  2,  2,  -2, -2, -3, -2, 0,  6,
    1,  2,  2,  -4, 1,  4,  3,  0,  2,  7,  1,  1,  1,  -2,  4,  1,  -3, 5,  9,  1,  2,  -1, -1, -1, -1, 0,   3,  -2,  1,  -3, 2,  -2, 0,  2,  0,  1,  0,  6,
    4,  3,  -4, 2,  -1, 2,  -3, -1, 1,  -1, -3, 0,  2,  3,   1,  0,  -5, -2, -3, -2, -4, -4, 1,  -2, -6, 2,   2,  -2,  1,  -3, -2, 1,  2,  2,  2,  1,  0,  1,
    4,  3,  0,  2,  -3, 2,  1,  5,  3,  1,  0,  1,  0,  5,   3,  -2, -3, -2, 1,  3,  0,  0,  -3, -2, 0,  -1,  2,  1,   -1, -1, 1,  0,  -1, -1, 1,  2,  2,  -1,
    -3, -1, 1,  1,  0,  3,  1,  2,  -1, -2, -1, -3, 1,  1,   -1, 2,  4,  -4, -1, 0,  4,  0,  -1, 7,  0,  -1,  0,  1,   3,  2,  -3, 0,  7,  3,  1,  0,  -2, -2,
    -1, 1,  2,  -3, -4, -1, 3,  -4, 3,  2,  1,  -1, -3, 5,   4,  5,  -2, 3,  1,  -3, 1,  2,  2,  -1, -2, 2,   2,  4,   -1, 2,  0,  -1, -3, -7, -6, -6, 1,  0,
    -6, 0,  1,  -4, 1,  -2, 0,  2,  2,  3,  2,  2,  2,  2,   4,  2,  0,  4,  -3, 0,  3,  5,  6,  1,  0,  -1,  -1, 3,   3,  -3, -3, -3, 1,  2,  1,  1,  -4, -1,
    -2, 1,  3,  -3, -1, 1,  -1, 0,  0,  -2, 0,  2,  1,  0,   -1, -2, 0,  -1, -2, 0,  -4, 0,  -2, -5, -1, -2,  0,  -1,  0,  1,  0,  0,  0,  -2, 4,  1,  -3, 4,
    -3, -4, -3, 4,  0,  0,  0,  2,  6,  4,  -1, -2, 0,  0,   -1, 1,  1,  1,  0,  1,  1,  -5, 0,  1,  2,  3,   2,  2,   0,  4,  -2, -3, 1,  1,  1,  -1, 3,  4,
    -4, 4,  1,  0,  0,  2,  2,  -1, -5, 1,  -5, -3, -1, -2,  -7, -5, 1,  -5, -1, -2, -2, 3,  2,  4,  -1, 3,   4,  2,   1,  1,  0,  4,  -5, 3,  2,  7,  6,  2,
    -1, 0,  -2, 0,  -1, 1,  1,  0,  -1, 2,  -1, 2,  -5, -1,  4,  1,  1,  0,  0,  2,  -2, 0,  -1, 0,  2,  5,   2,  0,   0,  -2, 2,  -1, 0,  -2, -4, -1, 2,  0,
    0,  2,  1,  2,  -1, -3, 4,  -1, -1, -1, 2,  1,  2,  4,   2,  -1, 1,  2,  0,  -2, -4, 0,  -1, -3, 0,  1,   1,  0,   1,  0,  0,  1,  -2, 3,  0,  0,  3,  3,
    0,  5,  3,  5,  4,  1,  -2, 4,  7,  2,  -1, 1,  -2, 0,   2,  -4, -2, 1,  0,  -2, 0,  -4, 2,  3,  1,  -1,  0,  0,   -2, -4, -3, -2, 2,  3,  2,  2,  3,  4,
    1,  -1, -3, 1,  4,  0,  0,  0,  -1, -1, -6, -2, 3,  0,   1,  -1, 2,  0,  0,  0,  0,  -1, -2, -1, -2, 1,   -4, -2,  3,  1,  0,  0,  0,  1,  0,  1,  -1, 1,
    4,  4,  0,  0,  1,  -4, 1,  -2, -2, -1, -2, -1, 2,  1,   1,  3,  0,  2,  -1, -1, 3,  1,  -1, -1, -1, 0,   2,  4,   2,  2,  1,  1,  2,  0,  -1, 1,  -1, -2,
    1,  3,  1,  1,  2,  3,  2,  1,  0,  0,  -3, 0,  5,  7,   0,  1,  1,  3,  4,  4,  -2, 5,  4,  -2, 1,  1,   -2, 1,   2,  0,  -1, -1, 1,  1,  2,  0,  2,  2,
    -1, -1, 2,  0,  -3, -2, -7, 0,  3,  4,  2,  4,  3,  2,   4,  -1, -3, 1,  5,  -1, 1,  -1, 0,  1,  -3, -3,  2,  0,   1,  -1, 2,  1,  2,  1,  1,  -1, -1, 1,
    -4, 0,  0,  -1, 2,  1,  1,  0,  -1, 0,  1,  0,  0,  1,   4,  6,  -2, 1,  -1, -4, 3,  -1, -4, -3, -3, -1,  2,  0,   1,  2,  1,  0,  -1, 0,  2,  1,  1,  -2,
    -1, 0,  0,  1,  4,  1,  0,  0,  1,  0,  -2, -1, -2, -2,  -2, 3,  -1, 1,  1,  2,  0,  1,  0,  0,  -1, -1,  3,  1,   0,  0,  2,  3,  4,  5,  -1, 4,  6,  0,
    1,  -2, -2, -2, -2, -1, -2, -2, 0,  1,  -2, 0,  0,  0,   1,  0,  0,  0,  -2, -5, -2, -2, 4,  5,  1,  3,   5,  4,   3,  0,  -2, 1,  5,  -1, 0,  2,  0,  1,
    -5, -6, 2,  -1, 2,  -3, -2, 2,  3,  -3, 1,  -2, -3, -1,  -2, 1,  -2, 0,  4,  2,  1,  0,  0,  0,  -1, 1,   0,  1,   3,  3,  1,  2,  -2, -1, 3,  -1, 0,  -2,
    0,  -3, 0,  1,  2,  2,  2,  -1, -2, 1,  4,  -1, 0,  1,   0,  1,  2,  1,  2,  4,  2,  1,  3,  1,  0,  1,   -2, 0,   -1, 4,  0,  0,  2,  2,  1,  -1, 2,  -1,
    -2, -1, 4,  2,  0,  1,  3,  0,  3,  5,  -1, 3,  3,  -2,  0,  0,  0,  -1, 0,  -1, -2, -3, 0,  1,  1,  -1,  -1, 1,   2,  0,  1,  0,  -4, -1, -1, -1, 3,  3,
    3,  4,  7,  5,  3,  1,  -4, 2,  5,  1,  2,  0,  -1, 3,   -3, -2, 1,  1,  1,  -3, 1,  3,  3,  1,  0,  0,   0,  -1,  0,  -3, -1, -2, 2,  2,  -1, 0,  -2, -1,
    -1, 0,  -1, 1,  1,  2,  -1, 1,  -3, -2, 1,  -1, -4, 1,   -1, 0,  3,  2,  3,  -1, 2,  1,  1,  0,  3,  0,   -1, 1,   2,  0,  1,  3,  1,  1,  3,  2,  1,  -2,
    0,  -1, -2, 0,  1,  2,  1,  -1, 1,  1,  1,  2,  1,  0,   -2, -1, 5,  -1, 0,  0,  3,  2,  4,  2,  -2, 3,   5,  -2,  2,  1,  0,  0,  1,  -4, -3, -1, -2, -1,
    1,  -1, 4,  0,  2,  1,  -1, 3,  -1, -5, -2, 1,  4,  4,   3,  1,  4,  5,  2,  -4, -4, 1,  5,  0,  1,  -1,  -1, 3,   -2, -5, 3,  -1, 0,  -3, 1,  1,  2,  2,
    1,  0,  -4, -2, 0,  -1, 0,  1,  -1, -2, 2,  -1, 4,  1,   -2, 5,  3,  2,  0,  -4, -1, 2,  0,  4,  1,  1,   3,  2,   0,  -2, -2, 3,  1,  -1, -2, -2, -1, 2,
    -4, 4,  5,  3,  -3, 1,  -1, -2, 3,  2,  5,  4,  1,  2,   3,  1,  -1, 1,  -1, -3, 2,  1,  -1, -2, 0,  1,   0,  0,   0,  2,  3,  -2, 1,  0,  0,  -1, 3,  -2,
    2,  1,  -1, 0,  -3, -2, -2, 3,  -1, 4,  3,  4,  2,  1,   -3, -1, -1, 2,  0,  2,  2,  2,  4,  2,  0,  1,   -1, 4,   -3, 2,  4,  0,  2,  1,  4,  1,  -1, -1,
    -1, -4, 0,  0,  1,  1,  -2, 2,  2,  -2, -6, -3, 2,  2,   1,  1,  -3, -2, 0,  -2, 0,  2,  1,  0,  -1, 1,   5,  0,   0,  4,  2,  -1, 1,  -3, -2, 2,  0,  5,
    2,  4,  5,  3,  -1, -2, -3, 3,  -1, 0,  0,  -6, 2,  4,   -3, 3,  5,  2,  -2, -3, 0,  -1, 5,  4,  2,  5,   0,  1,   1,  2,  0,  2,  0,  -3, 0,  2,  1,  0,
    -1, 2,  0,  0,  0,  3,  1,  -1, 1,  0,  -1, -3, 3,  -1,  3,  0,  0,  1,  -4, 0,  2,  2,  0,  1,  3,  2,   1,  0,   -4, -1, -1, 2,  1,  -1, 0,  2,  1,  0,
    0,  0,  -1, 6,  0,  1,  3,  2,  2,  2,  1,  2,  -1, 0,   -1, -2, -2, 0,  1,  2,  -3, 3,  -2, -3, -4, -3,  -1, 0,   2,  1,  -4, -2, 0,  -2, 1,  1,  6,  -1,
    -2, 1,  3,  -3, 2,  3,  1,  2,  0,  -3, -2, -1, 3,  6,   5,  3,  5,  4,  -3, -2, 0,  2,  -2, -1, 2,  -4,  1,  4,   -3, 2,  2,  4,  0,  -1, 2,  0,  1,  4,
    1,  3,  0,  -1, 2,  -2, -1, -1, 1,  -5, 1,  1,  1,  0,   1,  2,  -1, -1, 1,  2,  2,  1,  1,  0,  -1, -3,  1,  0,   3,  0,  0,  0,  -3, -2, 0,  3,  1,  2,
    4,  2,  1,  1,  -4, -2, -2, 1,  2,  -2, 0,  1,  0,  1,   2,  0,  -3, 2,  -1, 2,  1,  4,  3,  1,  2,  6,   -2, 1,   0,  -2, -1, 3,  2,  4,  -3, 2,  -1, -2,
    -4, -1, 0,  4,  4,  2,  -3, -2, 0,  -2, 3,  1,  4,  -2,  -3, 2,  0,  0,  -1, 7,  2,  4,  0,  -1, 0,  -1,  0,  6,   1,  4,  7,  2,  -1, -2, -4, 4,  0,  -1,
    0,  -3, 1,  6,  -1, -2, 0,  3,  -2, -1, 1,  -1, 0,  3,   1,  3,  0,  0,  0,  -4, -2, 1,  1,  -3, 0,  1,   1,  -1,  1,  2,  0,  1,  0,  5,  3,  0,  1,  0,
    0,  -1, 0,  1,  1,  1,  1,  1,  -4, 1,  3,  2,  0,  1,   2,  3,  1,  0,  -4, -2, -2, -1, 3,  -1, -1, 3,   1,  2,   -2, 1,  -2, 2,  2,  1,  4,  2,  2,  1,
    2,  4,  -1, 0,  2,  -2, -1, 4,  1,  3,  -2, 0,  0,  -2,  -3, -2, 1,  3,  2,  0,  0,  -3, 0,  -2, 0,  0,   1,  -2,  -3, 2,  3,  2,  1,  5,  2,  4,  1,  -1,
    2,  0,  -1, 6,  -1, 3,  6,  2,  -1, 1,  -3, 4,  0,  -1,  0,  -4, 2,  4,  -1, -2, 0,  2,  -3, -3, 0,  -1,  1,  4,   0,  2,  3,  0,  3,  1,  -2, 0,  1,  -2,
    1,  0,  0,  0,  1,  3,  -1, 0,  0,  5,  3,  1,  2,  -1,  -2, -1, 0,  1,  1,  2,  1,  -2, -3, 1,  0,  2,   2,  3,   3,  3,  0,  2,  -4, -1, -3, -1, 3,  -1,
    0,  1,  2,  2,  -2, 2,  -1, 2,  1,  -1, 6,  3,  4,  0,   2,  3,  0,  -1, 1,  -2, -2, 4,  2,  3,  -1, 1,   0,  -3,  -3, -4, -1, 1,  3,  -3, 1,  3,  3,  1,
    0,  -3, 1,  0,  1,  1,  2,  3,  2,  1,  1,  1,  0,  0,   1,  3,  3,  -1, 3,  3,  1,  0,  -2, 1,  2,  -1,  -1, 0,   3,  1,  5,  4,  -2, -2, 1,  5,  5,  1,
    2,  1,  0,  1,  3,  2,  0,  5,  0,  -1, -1, 3,  2,  1,   1,  0,  0,  1,  3,  3,  0,  -1, 1,  1,  4,  -1,  -3, 0,   1,  1,  0,  3,  3,  1,  1,  0,  0,  2,
    -2, -1, 1,  1,  0,  -2, -1, 2,  1,  0,  -1, 5,  0,  1,   1,  1,  2,  0,  1,  2,  5,  2,  5,  1,  2,  0,   2,  0,   0,  2,  3,  0,  -2, -2, 1,  2,  0,  -1,
    -1, -2, -3, -2, 2,  4,  -2, 2,  0,  1,  4,  2,  0,  4,   3,  -3, -1, 1,  -1, 0,  1,  3,  -1, 2,  0,  3,   -3, -2,  0,  3,  4,  -3, 3,  3,  0,  0,  -2, 1,
    1,  -2, -2, 0,  3,  2,  3,  3,  0,  -1, 2,  6,  2,  3,   1,  1,  1,  -2, 0,  -1, 0,  2,  1,  1,  0,  2,   4,  2,   1,  2,  0,  1,  2,  2,  1,  0,  -1, 1,
    2,  1,  0,  1,  2,  -1, -2, 1,  5,  3,  -2, -4, -2, -2,  -4, -2, 2,  2,  -3, 0,  -3, 1,  1,  1,  -2, 3,   1,  1,   1,  1,  0,  0,  3,  -2, 2,  -1, 4,  -1,
    5,  2,  1,  -1, 0,  3,  3,  3,  0,  0,  3,  3,  0,  1,   1,  -2, 0,  1,  1,  4,  -1, 0,  1,  0,  3,  -3,  -1, 1,   3,  -5, 0,  3,  -2, -2, 2,  1,  -3, 0,
    1,  -3, -4, -2, 0,  1,  1,  -5, -1, 0,  -1, -1, -2, -1,  1,  -3, -1, -1, 1,  4,  2,  3,  -3, -1, -1, 0,   0,  -1,  0,  -1, 1,  -2, -2, -2, -1, 1,  -1, -1,
    0,  -1, 0,  0,  1,  0,  -1, 0,  2,  0,  1,  -1, -1, 0,   2,  -2, -1, 0,  2,  0,  1,  -1, -1, 2,  -3, -5,  -2, -3,  -3, 1,  1,  0,  -1, -2, -2, 1,  -2, -4,
    -1, -3, 0,  0,  0,  -2, 1,  -1, 2,  0,  -2, 2,  4,  -1,  2,  1,  0,  -1, -2, 1,  0,  2,  0,  -1, -2, 0,   -3, -1,  -2, -4, -2, 1,  0,  0,  -1, -1, 1,  0,
    2,  0,  1,  1,  2,  0,  -1, 0,  -2, 0,  4,  -1, 0,  1,   2,  0,  -4, -1, 0,  0,  3,  -3, 1,  0,  1,  -4,  -3, 2,   1,  3,  -2, 1,  0,  3,  1,  3,  0,  0,
    2,  4,  3,  2,  1,  1,  3,  0,  -2, 0,  1,  -1, 1,  2,   2,  3,  4,  4,  2,  2,  2,  1,  3,  0,  0,  -1,  0,  5,   5,  1,  -1, 0,  -1, 2,  2,  0,  0,  -2,
    -1, -2, -4, -4, -4, -2, -2, 1,  0,  -3, -2, -1, 0,  -3,  -1, 5,  4,  -1, 2,  0,  1,  -3, -2, 1,  1,  5,   4,  0,   6,  1,  2,  2,  2,  3,  1,  1,  0,  1,
    3,  7,  0,  4,  -1, -1, -2, 2,  4,  2,  1,  -2, 1,  2,   1,  -1, -1, 2,  1,  -2, 4,  1,  -2, -2, 0,  -1,  -1, 2,   -1, -2, -4, 0,  -1, 0,  -1, 0,  -1, 0,
    0,  -4, -4, 3,  1,  0,  -4, 0,  0,  1,  0,  1,  -1, 0,   -1, 3,  1,  0,  1,  1,  1,  1,  -1, -1, 2,  -1,  0,  -1,  2,  2,  2,  1,  0,  1,  1,  2,  2,  0,
    0,  1,  0,  2,  4,  1,  1,  1,  1,  2,  2,  0,  -2, 0,   -2, -4, -3, -5, -4, -1, -2, -1, 0,  0,  -3, 1,   1,  -1,  -1, 0,  3,  2,  1,  1,  -1, -2, 1,  3,
    1,  2,  3,  -1, 3,  -1, 1,  2,  0,  1,  2,  1,  2,  0,   0,  2,  0,  0,  1,  -1, -2, 2,  0,  0,  -1, 0,   0,  3,   1,  4,  0,  3,  2,  1,  2,  2,  1,  -3,
    2,  2,  0,  -2, 2,  -1, 2,  1,  3,  2,  -4, 0,  4,  0,   -2, -1, 3,  -1, 1,  2,  3,  2,  -1, 2,  1,  -1,  0,  0,   0,  -4, -3, 0,  0,  0,  1,  1,  2,  1,
    1,  2,  3,  1,  -3, -1, 1,  -1, 0,  0,  1,  2,  0,  0,   2,  0,  2,  -3, 1,  4,  2,  0,  -2, 0,  2,  2,   -1, 3,   0,  0,  0,  1,  -1, 2,  -1, 3,  2,  -2,
    1,  2,  2,  2,  1,  0,  1,  -2, -1, 2,  -2, -2, 2,  2,   -4, -2, 3,  2,  0,  3,  2,  3,  -3, -1, 1,  1,   -2, 0,   -2, 1,  0,  -3, 1,  0,  1,  0,  0,  -1,
    -3, 0,  1,  2,  1,  1,  -1, 4,  1,  3,  4,  -1, 0,  0,   2,  1,  -1, -3, 1,  -1, 2,  1,  1,  2,  -2, 1,   1,  -2,  -4, -1, 1,  0,  -2, 3,  3,  1,  -2, 1,
    1,  -1, 3,  -1, -1, -4, -4, 0,  1,  1,  0,  -1, 2,  -1,  0,  2,  2,  1,  -2, -2, -1, -1, -1, -1, 1,  3,   1,  4,   2,  1,  1,  -3, 2,  3,  0,  1,  0,  -1,
    3,  1,  1,  3,  1,  -1, -1, 0,  0,  2,  -1, 0,  2,  -3,  1,  2,  0,  2,  0,  1,  2,  -2, 1,  2,  -2, -2,  2,  0,   -5, -4, -1, 4,  2,  4,  2,  2,  -1, 0,
    0,  2,  -1, -2, -3, 1,  -1, -1, 2,  1,  1,  1,  0,  -2,  -4, 0,  0,  0,  1,  1,  -4, 1,  4,  1,  0,  -2,  3,  1,   5,  -1, -3, -1, 1,  2,  2,  0,  2,  0,
    -5, -1, -1, -3, 0,  -3, 1,  2,  -1, 1,  3,  0,  0,  1,   2,  0,  4,  1,  -1, -3, -2, -1, -1, 0,  2,  0,   2,  0,   0,  0,  1,  0,  -2, -1, -1, 1,  0,  -1,
    1,  1,  2,  4,  1,  1,  2,  -3, 1,  4,  6,  2,  2,  0,   0,  2,  3,  1,  1,  1,  -1, 1,  1,  2,  -3, 2,   -2, -2,  4,  3,  1,  1,  1,  0,  2,  -1, -1, 1,
    -2, 0,  4,  -1, -6, -7, 2,  4,  1,  5,  3,  2,  0,  1,   0,  0,  0,  -2, -3, 0,  4,  0,  3,  -1, -2, 0,   -2, 0,   -3, -4, 0,  1,  1,  4,  -3, 1,  4,  3,
    2,  -3, 1,  2,  2,  1,  3,  -3, -2, 1,  1,  2,  0,  0,   -5, 1,  -3, 0,  0,  0,  1,  1,  0,  2,  0,  0,   0,  1,   3,  2,  4,  0,  -1, -3, -3, -1, -1, 1,
    2,  0,  1,  1,  0,  1,  2,  2,  0,  -1, -1, -2, 1,  0,   -1, -1, 1,  3,  1,  -1, 1,  -4, 1,  2,  2,  1,   2,  0,   3,  5,  4,  1,  1,  3,  1,  1,  0,  -1,
    3,  4,  -3, -2, 0,  1,  -2, -2, -4, 1,  2,  0,  -1, 1,   -1, 0,  2,  -3, -6, -2, 1,  2,  0,  2,  4,  2,   1,  0,   1,  2,  1,  -3, -5, 1,  1,  -2, 5,  3,
    -1, 1,  2,  0,  -2, -3, 0,  -1, 1,  3,  0,  5,  5,  1,   2,  -3, 0,  0,  1,  0,  0,  -3, 0,  0,  1,  5,   -2, 1,   -3, 0,  1,  -1, -1, 0,  -1, 1,  2,  1,
    3,  -2, 1,  3,  1,  5,  4,  1,  0,  -1, -2, -2, 0,  3,   1,  1,  -1, 0,  0,  0,  -1, -3, 0,  1,  -1, -2,  1,  -1,  -1, -2, 1,  4,  1,  0,  -1, -4, 2,  4,
    4,  3,  3,  0,  1,  3,  0,  5,  1,  1,  1,  2,  5,  2,   4,  1,  -1, 0,  1,  -1, 0,  0,  -2, -1, -1, 5,   -2, 3,   0,  -1, 2,  -4, -3, -2, 1,  3,  -4, 2,
    4,  2,  0,  -2, 1,  2,  1,  -2, -4, -2, 2,  0,  3,  0,   -4, 1,  2,  2,  3,  -3, -1, -1, 2,  6,  2,  0,   1,  2,   1,  2,  -2, -3, 0,  4,  1,  -2, -4, -9,
    -2, 3,  -1, -2, 0,  4,  -3, -2, 2,  1,  -5, 1,  -1, -4,  -1, 0,  2,  0,  1,  1,  -1, 0,  -2, 0,  -3, 1,   2,  0,   -2, 6,  2,  2,  2,  1,  2,  2,  0,  2,
    -1, -3, -3, 0,  0,  -1, 2,  1,  3,  -1, -1, 5,  -2, 5,   3,  2,  1,  3,  0,  -2, 3,  0,  -5, 3,  0,  -1,  0,  2,   4,  2,  0,  3,  1,  1,  3,  -2, 0,  0,
    -3, -4, 2,  -6, -2, 0,  -1, 0,  -2, -1, 4,  0,  3,  2,   2,  2,  2,  -1, 1,  2,  0,  0,  -1, 1,  3,  2,   0,  2,   0,  2,  2,  0,  4,  -3, 0,  0,  0,  2,
    2,  -1, 2,  3,  1,  2,  0,  -2, 0,  5,  -1, -2, -3, -7,  0,  2,  -3, -2, 1,  6,  -1, 1,  2,  1,  -4, 1,   1,  0,   -1, -1, 0,  1,  -4, 4,  -1, 1,  -3, -2,
    -5, 1,  -1, -1, 1,  4,  2,  0,  2,  1,  2,  1,  -4, 1,   1,  -1, 0,  0,  -1, -2, 3,  1,  2,  -1, 2,  6,   -1, 5,   4,  -1, 1,  4,  1,  -2, 1,  -1, -6, 5,
    0,  -1, 1,  1,  2,  4,  -1, 1,  2,  1,  2,  -2, -3, 2,   0,  -3, 1,  -4, 0,  0,  -2, 2,  2,  1,  3,  1,   1,  1,   1,  3,  2,  -1, 2,  3,  0,  0,  -1, 0,
    2,  3,  -1, 3,  1,  1,  2,  0,  3,  -3, 0,  2,  1,  4,   2,  2,  2,  -1, -1, 0,  -3, -1, 0,  6,  0,  -3,  -2, -5,  0,  2,  -1, -4, 2,  5,  -1, 0,  2,  0,
    0,  2,  1,  1,  1,  -3, 1,  1,  -2, 2,  0,  0,  -2, -3,  -3, 1,  -1, -1, 1,  1,  1,  1,  1,  -1, 2,  3,   -4, 1,   -1, -3, -1, 0,  0,  0,  1,  3,  2,  -2,
    0,  3,  2,  6,  4,  1,  1,  5,  0,  -1, 0,  -2, -3, 3,   3,  0,  2,  2,  3,  6,  0,  1,  5,  2,  0,  0,   -3, 0,   5,  -2, 0,  -4, -1, -1, 0,  0,  0,  1,
    4,  1,  0,  3,  2,  4,  2,  0,  0,  3,  1,  1,  -1, 1,   2,  0,  -3, 4,  0,  -1, -1, -1, 4,  -3, -1, 1,   0,  4,   0,  0,  2,  1,  1,  2,  -1, -3, 4,  6,
    2,  -4, 0,  -6, 1,  1,  0,  0,  1,  1,  -2, -1, 1,  2,   -2, -1, 1,  0,  -1, -2, 0,  -1, -4, 2,  -1, -2,  -5, -5,  -3, 3,  -2, -4, -2, 0,  2,  -1, 2,  0,
    2,  3,  -4, 2,  2,  -1, -3, -1, -2, -1, 0,  1,  0,  -3,  -2, 1,  -1, 5,  1,  2,  0,  5,  1,  -3, 2,  -2,  -2, 2,   3,  1,  3,  2,  4,  4,  -3, 2,  2,  1,
    0,  -1, -2, 3,  1,  -2, 1,  -3, 0,  -4, 2,  -4, -1, 0,   3,  1,  3,  0,  0,  3,  4,  -2, 0,  0,  -1, -2,  -2, 1,   2,  -2, -2, 4,  0,  2,  0,  0,  4,  -3,
    -1, 0,  0,  2,  2,  0,  4,  1,  2,  2,  0,  -3, 3,  4,   1,  -3, -5, -5, 1,  3,  -1, 0,  0,  3,  -1, -2,  -1, 3,   -1, -1, 1,  2,  -2, -2, -1, -2, -3, 1,
    1,  0,  -2, -4, -1, 1,  0,  -5, -1, 0,  1,  0,  3,  0,   3,  1,  -4, 2,  0,  -1, -2, -3, -1, 0,  0,  1,   1,  -5,  -2, 0,  -2, 2,  1,  3,  1,  4,  -1, -2,
    0,  -2, -3, 3,  2,  -2, 3,  6,  3,  7,  -3, -2, 1,  2,   1,  3,  0,  1,  2,  -1, -2, -3, 0,  -2, 2,  -1,  1,  0,   3,  3,  0,  1,  0,  3,  2,  -1, 1,  1,
    -1, -1, -1, 2,  2,  0,  -4, 6,  1,  2,  -1, 1,  5,  3,   4,  1,  -3, -1, 0,  0,  0,  1,  1,  -1, 3,  0,   0,  -1,  -2, 2,  1,  3,  -1, -2, -3, -1, 3,  0,
    -2, 0,  -5, -2, 1,  3,  1,  0,  -1, 0,  3,  2,  4,  2,   2,  4,  0,  -2, 4,  1,  -1, 1,  3,  2,  2,  3,   5,  3,   -1, 0,  -3, 4,  3,  0,  1,  1,  -1, 1,
    1,  1,  -3, 3,  4,  2,  1,  1,  -1, 1,  3,  -3, -1, -1,  -7, 2,  -1, -2, 0,  -4, -1, 1,  -2, 3,  1,  -1,  0,  -1,  1,  0,  0,  1,  1,  -2, -2, 0,  -2, -1,
    -4, -1, -4, -3, 0,  -1, -3, 1,  1,  3,  1,  5,  1,  1,   0,  3,  2,  2,  3,  1,  5,  0,  0,  1,  1,  3,   2,  3,   4,  1,  -2, -3, 3,  0,  -1, 5,  -2, -2,
    5,  2,  -2, -2, 2,  1,  4,  1,  1,  5,  -1, -1, 1,  -2,  3,  -1, -4, -1, 0,  3,  -1, 2,  0,  -1, 4,  2,   4,  0,   3,  3,  1,  -1, -1, -2, -2, 0,  3,  2,
    4,  3,  5,  -2, 0,  2,  -2, 0,  -1, 0,  1,  -1, 0,  0,   -2, 1,  -2, 1,  2,  1,  1,  3,  0,  -1, 3,  0,   2,  -2,  -5, 0,  -2, 0,  -1, -3, 0,  3,  1,  -1,
    0,  1,  0,  2,  -1, 1,  3,  3,  1,  -1, 0,  1,  0,  1,   1,  1,  -3, 1,  3,  3,  -1, 4,  1,  2,  1,  4,   -1, -1,  4,  3,  3,  4,  3,  2,  2,  0,  0,  2,
    3,  2,  3,  -2, 1,  0,  0,  -1, 5,  2,  3,  0,  -4, -3,  2,  -2, 0,  -3, 0,  -1, 1,  -1, 1,  1,  -3, 0,   0,  -4,  1,  -3, -6, 0,  3,  -2, -1, 0,  -4, -1,
    2,  -1, 3,  2,  0,  0,  -3, -4, -1, -2, -2, 1,  1,  -3,  -2, 3,  0,  -2, -2, 4,  -3, -1, -2, -2, 0,  2,   -1, -1,  0,  0,  -3, -1, 0,  0,  -4, 1,  -2, 0,
    0,  -1, 2,  1,  -5, -1, -4, 0,  0,  -5, 3,  2,  -2, -3,  1,  3,  1,  1,  -3, 2,  -1, 1,  0,  -1, 0,  1,   0,  1,   2,  1,  -3, -1, 2,  -3, -4, -1, 1,  0,
    -1, 3,  -2, -2, 3,  0,  1,  1,  -2, -1, 1,  -1, -1, 1,   0,  0,  1,  -2, 3,  2,  -4, 0,  3,  1,  3,  1,   -5, 1,   2,  1,  1,  4,  0,  0,  0,  3,  2,  3,
    1,  1,  1,  2,  5,  0,  -1, 0,  5,  -2, 0,  2,  -1, 1,   3,  0,  4,  2,  4,  2,  0,  -3, 5,  0,  -1, 1,   3,  4,   2,  0,  5,  1,  1,  3,  2,  2,  2,  3,
    2,  2,  1,  2,  1,  2,  -1, -1, 1,  0,  0,  4,  2,  4,   3,  1,  1,  -1, -1, 0,  -2, -2, -1, -3, 1,  6,   2,  5,   2,  2,  2,  0,  -3, 0,  4,  1,  4,  1,
    -1, 1,  2,  -1, 2,  2,  -2, -1, 0,  0,  -3, 1,  3,  2,   2,  3,  0,  -2, 3,  0,  2,  6,  1,  -1, 2,  0,   0,  0,   2,  -1, 6,  -3, 3,  0,  0,  4,  3,  1,
    0,  -2, 0,  -2, 3,  2,  1,  1,  -2, 3,  0,  2,  4,  -1,  0,  3,  2,  2,  -1, -1, -1, 2,  4,  1,  2,  2,   -1, 2,   0,  3,  2,  1,  4,  2,  -2, -5, 2,  -1,
    -2, -1, 1,  -1, 1,  1,  4,  0,  -2, 3,  0,  1,  2,  1,   0,  -1, 0,  -3, 0,  2,  -2, -1, 5,  3,  -1, 3,   2,  3,   1,  2,  3,  -2, -3, 1,  -2, 1,  -3, -2,
    -1, 5,  0,  4,  4,  3,  3,  3,  -1, 2,  2,  1,  4,  2,   1,  3,  1,  0,  -2, 0,  -2, -1, 0,  -1, -4, -1,  3,  2,   0,  3,  -2, -2, 2,  -1, -2, 2,  -1, 0,
    1,  0,  0,  1,  -2, 1,  3,  1,  0,  -1, 5,  0,  3,  3,   0,  1,  3,  2,  -3, 1,  4,  -1, 3,  0,  4,  3,   -2, 3,   5,  -1, 1,  2,  -2, 1,  -2, 1,  2,  2,
    3,  -1, 0,  5,  1,  4,  3,  1,  0,  -1, -1, 3,  0,  0,   -1, -1, 2,  0,  2,  1,  0,  -2, 1,  2,  1,  0,   1,  -3,  0,  2,  3,  3,  3,  -1, 1,  0,  2,  0,
    3,  -2, -1, -3, 1,  -3, 0,  0,  0,  -1, -2, 0,  1,  -1,  -3, -2, 1,  -3, -4, -1, 0,  2,  3,  -3, 6,  -1,  4,  5,   2,  -3, 1,  -2, 4,  -3, 3,  0,  8,  -1,
    -1, -1, -2, 1,  -4, 0,  0,  2,  -4, -6, -4, -4, 3,  3,   -4, -6, 1,  3,  2,  5,  2,  1,  1,  0,  2,  -1,  1,  1,   -2, 4,  4,  1,  -2, 0,  4,  -1, 1,  0,
    2,  4,  -2, 4,  0,  1,  3,  0,  1,  0,  -1, -2, 0,  4,   2,  -2, 0,  4,  2,  2,  1,  2,  -2, 0,  0,  0,   -2, 0,   -3, -2, 3,  0,  0,  -2, 0,  -3, -2, 1,
    1,  4,  -1, -3, 1,  2,  2,  4,  3,  1,  1,  2,  4,  0,   3,  -3, -1, -3, 1,  -2, 0,  -4, -1, -1, -5, 2,   0,  -5,  -4, -2, -1, 1,  -3, -2, 0,  1,  2,  -2,
    6,  0,  3,  3,  0,  -3, 3,  2,  2,  -1, 1,  4,  6,  1,   4,  -3, -2, 1,  -6, -1, 0,  0,  -5, -4, 1,  -4,  0,  5,   -3, -5, 0,  2,  0,  4,  0,  -1, 1,  -2,
    4,  -1, 3,  1,  1,  0,  0,  1,  -1, 2,  4,  2,  2,  0,   2,  3,  0,  2,  1,  -1, 1,  3,  1,  0,  -1, 0,   0,  2,   1,  -3, 0,  4,  3,  1,  2,  1,  0,  -1,
    3,  2,  -1, -1, -2, -2, 1,  0,  2,  -2, 2,  0,  1,  -1,  0,  2,  -1, -3, -1, 1,  3,  3,  4,  0,  3,  3,   0,  0,   2,  -3, -3, -3, -3, -2, 2,  -2, 0,  -2,
    -4, 1,  -1, -5, -1, -4, 2,  1,  -5, -2, 2,  3,  5,  1,   2,  -1, 3,  4,  3,  0,  2,  1,  2,  -3, 4,  2,   7,  1,   3,  -1, -1, 0,  -3, -1, 1,  3,  -5, -3,
    -3, -6, -2, 5,  -3, -3, -2, 0,  4,  3,  -1, 1,  4,  -4,  1,  0,  1,  0,  0,  -1, 1,  1,  -1, 1,  4,  2,   -1, 1,   1,  2,  1,  1,  1,  -2, 0,  0,  3,  0,
    -1, 0,  0,  1,  1,  -2, 1,  4,  1,  1,  2,  1,  -2, 2,   1,  1,  -3, -4, -1, -2, 2,  -2, 1,  1,  2,  1,   -1, 0,   2,  1,  1,  -3, -2, 1,  2,  4,  3,  3,
    1,  3,  1,  -1, 1,  -4, -3, -2, -5, -1, 0,  -2, -1, -2,  -5, 2,  1,  -4, -3, -3, 2,  -2, -5, -1, -1, 3,   4,  2,   1,  0,  2,  3,  4,  2,  0,  2,  5,  -3,
    4,  2,  6,  3,  2,  1,  0,  2,  0,  -1, 3,  2,  -5, -5,  -2, -3, 0,  3,  -4, -2, -3, 1,  3,  2,  1,  2,   4,  -2,  1,  1,  2,  1,  1,  -2, 1,  0,  0,  0,
    4,  0,  -1, 2,  5,  0,  3,  4,  -1, 0,  2,  1,  2,  -2,  0,  1,  1,  2,  0,  0,  2,  5,  4,  -2, 2,  1,   0,  3,   2,  0,  0,  -1, 0,  -1, 2,  0,  2,  1,
    1,  1,  2,  -1, 2,  4,  4,  -2, 0,  2,  1,  4,  3,  1,   2,  -1, 2,  1,  0,  -1, -1, -5, -3, -4, -1, -3,  -2, -1,  -3, 4,  1,  -4, -2, -6, 2,  -4, -4, 0,
    0,  0,  3,  0,  4,  1,  1,  4,  1,  -1, -2, -1, 7,  0,   3,  4,  8,  3,  0,  -1, 0,  -1, -1, -3, 2,  1,   -3, -3,  -2, -4, -1, 3,  -7, 1,  -2, 4,  4,  5,
    1,  4,  -1, 1,  7,  1,  -1, -4, -3, 2,  2,  -2, -4, 2,   5,  -2, 3,  4,  6,  -2, 0,  -4, 0,  1,  0,  3,   -2, -3,  -1, -3, 2,  0,  -1, 2,  -1, -1, 0,  2,
    -1, 3,  2,  -3, 2,  3,  2,  0,  -1, 2,  1,  0,  2,  1,   -3, 0,  -2, 0,  1,  0,  -1, 0,  -1, 0,  -1, 2,   4,  3,   -2, 1,  -1, 5,  2,  2,  -1, -2, 2,  0,
    -2, 3,  1,  -1, 1,  -2, -3, 5,  2,  1,  2,  -1, -2, -5,  4,  3,  2,  0,  2,  4,  1,  -5, 1,  -4, -3, 2,   4,  1,   4,  -2, 5,  4,  9,  4,  1,  1,  0,  2,
    1,  -2, 1,  -1, -1, 5,  -3, -5, 2,  3,  1,  2,  -3, 1,   -2, 1,  2,  1,  4,  4,  -5, -5, -1, 1,  2,  -1,  -2, -1,  6,  -1, -2, 4,  1,  -4, 1,  0,  1,  0,
    -1, 2,  -3, 0,  -5, -3, 0,  3,  0,  -2, 0,  2,  1,  2,   -2, 5,  3,  -2, 2,  2,  0,  -2, 3,  2,  -1, 0,   1,  -2,  -1, 0,  -1, -1, 2,  -4, -4, -3, -1, 0,
    0,  2,  4,  3,  -3, 2,  0,  4,  3,  -1, 1,  -3, 1,  -1,  -1, 5,  3,  0,  4,  0,  -2, 4,  3,  -2, 3,  -1,  -1, -6,  2,  2,  4,  2,  3,  7,  4,  -2, 1,  -4,
    -4, -2, 1,  1,  1,  -1, 2,  1,  6,  3,  1,  3,  0,  0,   -1, -1, -1, 0,  -3, 3,  -3, -9, 2,  1,  1,  1,   -2, 6,   0,  4,  2,  3,  4,  3,  -2, -4, -2, 2,
    0,  1,  -2, 0,  7,  1,  0,  2,  0,  0,  1,  1,  4,  -1,  1,  5,  -4, -1, 0,  2,  4,  1,  -2, -2, -3, 1,   1,  0,   -4, 4,  3,  -1, -1, 2,  0,  2,  3,  1,
    2,  0,  2,  1,  -1, 0,  1,  -3, -2, 2,  -2, -1, -2, 1,   2,  1,  2,  4,  -1, 1,  2,  4,  2,  -3, -2, -3,  -1, 1,   0,  8,  2,  -1, 2,  -3, -1, 6,  3,  -1,
    -2, -3, 1,  -3, 4,  3,  2,  1,  1,  8,  4,  -1, 1,  -1,  -4, 2,  2,  -2, 1,  -2, 0,  2,  8,  3,  0,  0,   0,  1,   0,  -4, 0,  -2, -5, 3,  0,  -3, 3,  -1,
    2,  0,  -1, 4,  1,  0,  0,  2,  6,  5,  1,  -4, -3, 0,   -2, 1,  -1, -1, 4,  -1, 0,  6,  1,  -4, 3,  1,   0,  2,   2,  0,  -2, 0,  1,  4,  3,  2,  1,  0,
    -4, 1,  0,  -1, -2, 4,  3,  1,  1,  0,  0,  1,  0,  1,   5,  -4, 0,  1,  -2, -4, -1, 0,  -2, 2,  -2, -1,  -2, -1,  2,  1,  1,  5,  -2, 1,  1,  5,  0,  -3,
    -4, -2, 2,  -1, 1,  5,  0,  1,  1,  -3, -1, 7,  6,  -4,  3,  -4, 0,  -6, 4,  1,  2,  4,  1,  6,  3,  -1,  0,  -1,  -1, 2,  3,  -2, 0,  -4, -1, 4,  6,  5,
    2,  -1, 0,  0,  -2, -1, 1,  -2, -5, 3,  0,  -5, 3,  1,   2,  0,  -3, 3,  2,  0,  0,  2,  5,  7,  1,  -3,  -1, 1,   -3, 2,  -3, 1,  -1, 3,  0,  2,  2,  -2,
    5,  -2, -1, -2, 1,  4,  0,  -3, -3, 2,  3,  5,  2,  0,   -5, 2,  -1, -3, 1,  3,  0,  1,  2,  -1, 2,  2,   2,  2,   1,  0,  -3, 0,  0,  -1, -2, 0,  0,  3,
    -2, -1, 0,  1,  3,  2,  3,  6,  2,  1,  1,  5,  2,  0,   -2, 2,  1,  -1, 0,  3,  2,  -3, -1, -2, 1,  3,   3,  -3,  3,  -5, -1, -5, 2,  3,  5,  4,  0,  5,
    5,  1,  2,  -4, 1,  2,  5,  -2, -1, -1, 2,  4,  5,  4,   2,  4,  -2, 3,  -2, -2, 3,  0,  -3, 4,  0,  -3,  1,  -3,  0,  3,  1,  2,  1,  3,  2,  3,  3,  0,
    -5, -4, 4,  2,  -1, -1, -1, -2, 3,  -2, 0,  -5, 1,  1,   0,  5,  1,  0,  2,  -2, -1, -2, -1, 0,  1,  -2,  0,  -1,  4,  -2, 2,  4,  -3, 3,  3,  0,  5,  4,
    1,  -1, -1, 0,  0,  -3, 2,  -2, -3, 1,  -3, -2, 1,  -2,  3,  0,  2,  1,  0,  -3, 7,  -1, 2,  3,  1,  -2,  0,  0,   3,  0,  0,  -5, 1,  2,  1,  -3, -4, 2,
    1,  0,  5,  0,  -1, 1,  2,  1,  -1, 0,  0,  -1, -2, 2,   -1, -1, -1, 0,  -1, -2, 2,  1,  -2, 2,  1,  1,   -1, 2,   5,  0,  -3, -1, 4,  3,  1,  0,  0,  2,
    1,  -1, -5, 0,  1,  -2, -3, 0,  2,  2,  4,  0,  1,  3,   -1, -4, 3,  1,  -2, -4, -5, 1,  5,  -1, -1, -5,  0,  2,   1,  4,  3,  0,  3,  -2, -1, -1, 1,  2,
    -1, -1, -2, -1, 2,  1,  0,  3,  -1, 4,  2,  -1, 3,  3,   0,  -3, 1,  3,  -1, -1, 3,  -2, -3, -1, -3, 0,   4,  0,   2,  3,  1,  2,  -2, -1, 5,  1,  2,  3,
    0,  -2, 0,  -2, 2,  -1, -3, -3, -2, 2,  2,  -4, -1, 1,   -2, -1, 4,  2,  -2, 1,  2,  1,  -3, 0,  2,  2,   0,  3,   -1, 2,  2,  0,  -1, -3, 0,  4,  -2, 1,
    1,  1,  -3, 3,  3,  1,  -2, -1, 4,  5,  2,  -1, -2, 0,   1,  1,  -1, 1,  1,  2,  1,  -2, 0,  2,  3,  2,   0,  2,   -3, 0,  0,  -2, 0,  -3, -5, 4,  2,  1,
    -2, -6, 2,  2,  2,  1,  4,  -2, 4,  1,  -3, 1,  0,  -1,  4,  -1, -2, 0,  0,  0,  1,  2,  0,  5,  5,  1,   3,  4,   0,  0,  2,  3,  -1, -1, 5,  0,  -1, 0,
    -2, -1, 4,  1,  2,  4,  0,  2,  -2, -1, 5,  -1, 4,  5,   3,  -1, -3, 1,  4,  1,  -2, 3,  1,  3,  4,  -3,  -4, 0,   -3, -3, 4,  0,  -2, 4,  1,  1,  -1, 2,
    2,  3,  -2, 2,  -3, 2,  3,  3,  -2, -1, 0,  2,  -2, 4,   3,  1,  -2, 4,  5,  -1, -2, 0,  3,  5,  1,  -2,  0,  -1,  2,  -1, -3, -1, -1, 4,  0,  -2, -1, 3,
    3,  1,  2,  0,  -4, -2, 0,  -4, 3,  -4, -6, 2,  0,  0,   -4, -3, 1,  0,  -2, 1,  1,  0,  -1, -1, -3, 1,   0,  -3,  2,  0,  -3, -2, -1, 2,  -1, 4,  -1, 5,
    7,  3,  5,  -1, 0,  2,  2,  2,  -1, -3, 3,  2,  -1, -1,  -4, -1, 3,  -1, 2,  4,  6,  2,  0,  -1, 5,  0,   3,  3,   2,  0,  -3, -2, 2,  0,  -2, 2,  2,  2,
    3,  -4, 0,  1,  -2, -1, 3,  1,  -4, 5,  2,  1,  1,  3,   0,  1,  -4, 0,  -3, -1, 0,  0,  -2, -2, 0,  -1,  -5, 0,   3,  2,  2,  3,  4,  1,  -2, -2, 3,  3,
    0,  -2, -1, -1, 2,  -1, -3, 2,  1,  1,  0,  2,  -3, -4,  3,  -1, 0,  0,  -2, -2, -2, 2,  1,  -4, -3, 1,   0,  -1,  -3, -4, 1,  2,  0,  1,  2,  -4, 2,  -1,
    0,  2,  -1, -2, 1,  -1, 0,  -1, -1, 4,  -1, 1,  1,  5,   5,  0,  4,  1,  0,  1,  1,  1,  -4, -2, 1,  4,   1,  -3,  -3, 0,  3,  -1, 2,  4,  4,  3,  1,  0,
    4,  1,  5,  3,  2,  2,  -1, -1, 1,  1,  -1, -2, -1, 2,   3,  -4, 0,  0,  -2, 1,  1,  1,  2,  4,  4,  2,   0,  2,   1,  1,  -6, 0,  -1, -1, 1,  2,  -1, -3,
    0,  1,  -2, 1,  -2, 0,  4,  6,  4,  0,  -1, -1, 1,  4,   0,  1,  -2, -1, 1,  -1, -3, 1,  3,  0,  2,  -2,  2,  4,   2,  -1, 2,  3,  -1, 1,  0,  -3, 2,  2,
    0,  0,  -1, 1,  -2, 0,  3,  0,  0,  3,  2,  1,  -2, 4,   2,  -3, 3,  1,  2,  -1, -2, 1,  -3, -2, -1, 2,   2,  0,   2,  1,  2,  -1, -4, 3,  0,  0,  1,  0,
    0,  1,  -2, 3,  2,  -1, 1,  -1, -3, 2,  1,  -1, -1, -2,  3,  4,  1,  1,  1,  3,  2,  4,  1,  0,  1,  2,   -3, -2,  -1, 0,  2,  3,  0,  3,  6,  -2, 1,  7,
    -5, 2,  1,  -5, -4, -3, 1,  -3, -2, -2, -1, 0,  -2, 2,   0,  2,  -2, -2, 4,  -3, 4,  3,  -2, 1,  0,  0,   0,  1,   2,  -2, -6, -3, 1,  2,  9,  3,  4,  3,
    2,  1,  1,  2,  3,  1,  1,  3,  -4, -1, 3,  -2, 0,  4,   1,  -2, -1, 1,  0,  -2, 2,  0,  -1, 2,  3,  0,   -1, 3,   1,  -3, 5,  -3, 2,  -2, 1,  0,  -2, -1,
    0,  -1, 2,  -1, 3,  4,  1,  -1, -3, 1,  0,  0,  4,  -2,  1,  1,  -1, 3,  1,  0,  1,  0,  -1, 2,  1,  -1,  -2, -1,  1,  6,  2,  2,  -1, 0,  3,  4,  0,  -2,
    0,  2,  -1, 0,  2,  -3, 3,  0,  -1, 1,  8,  0,  -2, 2,   -6, 2,  -1, -4, -3, -1, -1, -1, -2, -2, 1,  2,   -1, 3,   2,  1,  1,  -1, 4,  -2, 6,  0,  -1, 2,
    0,  -1, -2, -1, 3,  -2, -6, -2, 4,  2,  6,  0,  5,  3,   1,  0,  -1, -1, 4,  -1, 1,  2,  -4, 2,  5,  -1,  2,  0,   -3, 0,  -1, -1, -2, 1,  2,  2,  2,  3,
    2,  1,  -3, -1, -1, -3, 3,  -2, 2,  -4, 1,  -1, 0,  -1,  -4, 2,  1,  -3, 2,  3,  3,  -3, -3, 2,  0,  0,   -2, -1,  0,  0,  -2, 3,  1,  2,  1,  0,  -2, 0,
    3,  -1, -1, -2, 1,  5,  3,  1,  -1, 1,  2,  3,  1,  -3,  2,  4,  0,  0,  -1, -1, 2,  1,  0,  2,  8,  0,   1,  4,   -2, 1,  -1, -4, -1, -3, 3,  0,  -1, -1,
    -1, 1,  -2, 2,  1,  -2, 1,  1,  2,  -1, 5,  -1, -1, 1,   -4, -5, 0,  2,  2,  -1, -5, -4, 5,  3,  7,  -2,  2,  5,   0,  1,  -1, 1,  4,  -2, -1, 1,  -3, 3,
    3,  -4, 3,  4,  -1, 3,  -1, 2,  1,  -2, 3,  2,  0,  3,   1,  1,  -2, 2,  -1, -2, 5,  -4, -1, -4, 0,  0,   0,  -1,  -4, -2, 0,  -1, 1,  3,  2,  -3, 0,  0,
    1,  2,  1,  0,  1,  0,  0,  3,  3,  1,  0,  1,  -1, 1,   3,  1,  0,  -1, 0,  5,  1,  0,  0,  0,  5,  2,   1,  -3,  1,  6,  -1, -1, 0,  -1, 3,  0,  -1, 1,
    8,  2,  5,  5,  -3, -2, -1, -4, 0,  -2, 1,  0,  0,  2,   -2, 0,  -1, -1, 0,  0,  5,  1,  1,  1,  3,  1,   -1, -1,  -1, -1, 1,  -2, 0,  -3, -6, -5, 2,  0,
    9,  -2, 3,  4,  1,  0,  -2, 3,  2,  -2, 0,  3,  -4, 4,   -1, -1, 2,  4,  -3, 3,  0,  0,  1,  0,  2,  5,   -1, 1,   3,  0,  -2, 0,  -4, -4, 4,  -3, 0,  -4,
    1,  -1, 1,  0,  -3, 0,  -1, -1, 3,  1,  2,  -2, -1, 2,   2,  -1, 2,  -1, 0,  1,  1,  3,  1,  2,  0,  0,   -1, 0,   2,  1,  -2, -2, 1,  4,  1,  -1, -2, -1,
    2,  0,  -1, -1, 2,  4,  -2, -1, -3, 2,  3,  0,  -2, 2,   8,  0,  2,  1,  1,  1,  0,  -6, -4, -5, -1, 0,   -3, -1,  0,  0,  -1, 2,  2,  1,  6,  1,  1,  3,
    6,  1,  -1, 0,  1,  -1, 1,  -1, 0,  0,  -5, -4, 2,  1,   8,  -3, 3,  4,  3,  3,  0,  1,  -1, 0,  4,  -2,  2,  1,   -4, 0,  0,  2,  -1, 2,  0,  -1, -2, -1,
    -5, -2, 3,  0,  2,  -1, 2,  2,  0,  1,  -1, 2,  -2, 0,   -1, -1, -2, -2, 2,  1,  -1, 6,  2,  1,  4,  -1,  -2, 1,   2,  1,  -1, -2, -2, 2,  1,  1,  -1, -1,
    2,  -2, -3, -2, 2,  0,  0,  3,  1,  1,  4,  1,  1,  4,   0,  2,  2,  3,  1,  0,  -1, 0,  0,  -1, -1, 1,   1,  1,   2,  4,  3,  2,  -1, -2, 0,  -2, 0,  0,
    -1, -1, 0,  3,  -2, 0,  1,  0,  3,  2,  3,  1,  1,  -2,  -2, 1,  0,  0,  3,  1,  0,  1,  -2, -2, 0,  2,   2,  2,   0,  0,  1,  0,  3,  -2, 1,  0,  1,  -1,
    4,  -1, 2,  0,  -3, -2, 1,  2,  -1, 1,  3,  -1, -5, -2,  -4, -1, 0,  -3, 1,  0,  2,  2,  -1, 1,  -2, 3,   -2, 1,   0,  -3, 1,  -1, 3,  1,  0,  5,  2,  0,
    7,  -1, -3, -1, 2,  2,  0,  -1, -1, 3,  0,  -1, 2,  -1,  1,  -1, -3, -1, 2,  0,  0,  3,  1,  0,  2,  1,   2,  5,   -1, 1,  4,  4,  2,  1,  0,  -2, 1,  0,
    -2, 0,  3,  3,  5,  4,  3,  0,  -1, 0,  0,  0,  0,  -1,  1,  0,  1,  1,  1,  2,  -1, 1,  3,  2,  4,  0,   0,  0,   1,  0,  3,  -1, 1,  1,  0,  -1, -2, -3,
    -1, 2,  1,  4,  1,  2,  0,  1,  2,  -2, 1,  2,  1,  2,   2,  -3, 2,  -1, -2, -3, 2,  1,  -1, 1,  1,  -4,  -2, 0,   -3, -3, 2,  -1, 1,  1,  2,  0,  0,  2,
    2,  2,  -1, 0,  0,  -2, -1, -2, 3,  1,  -3, 6,  3,  1,   2,  0,  -4, -1, -1, 0,  2,  -1, 0,  4,  0,  0,   2,  -2,  -2, -1, -3, 0,  2,  0,  0,  2,  1,  0,
    3,  -1, 4,  3,  -3, 0,  0,  4,  1,  0,  0,  -3, 0,  0,   0,  2,  3,  3,  3,  4,  2,  -1, 1,  -4, 1,  0,   -1, -2,  0,  2,  4,  1,  -3, 1,  1,  1,  3,  2,
    2,  0,  0,  0,  -2, -2, 2,  0,  2,  2,  0,  0,  -1, -1,  1,  2,  1,  3,  2,  0,  0,  0,  3,  -3, 1,  4,   3,  0,   2,  2,  3,  1,  -2, -5, 3,  1,  -2, 2,
    -1, -4, -5, -2, -4, -3, 1,  1,  2,  2,  0,  -1, 2,  1,   0,  4,  0,  0,  0,  -4, 1,  0,  4,  5,  -2, 3,   3,  1,   3,  -2, -4, 2,  -1, 1,  3,  -1, 0,  1,
    0,  -1, 3,  -1, -1, -2, -2, -1, 1,  1,  0,  2,  2,  2,   2,  1,  3,  -1, -1, 0,  2,  2,  0,  1,  2,  -1,  0,  2,   1,  0,  2,  2,  5,  6,  1,  -2, -1, -2,
    -2, 0,  0,  -1, 0,  1,  5,  2,  0,  1,  1,  0,  2,  4,   1,  0,  0,  0,  2,  -1, 1,  -1, 2,  2,  0,  0,   -3, -1,  1,  2,  1,  5,  2,  1,  0,  2,  1,  -1,
    -1, 6,  1,  2,  4,  0,  3,  2,  -2, -2, 2,  0,  -1, 3,   -1, -3, -7, -5, -3, -2, 0,  2,  -2, 3,  1,  0,   1,  1,   0,  2,  0,  0,  0,  -3, -1, 3,  4,  4,
    -1, 3,  2,  1,  2,  -1, -4, 1,  -1, -1, 0,  1,  2,  3,   0,  0,  1,  -2, -2, 0,  -3, -1, 2,  1,  0,  1,   -1, 2,   2,  0,  3,  0,  -1, 1,  0,  1,  1,  2,
    2,  -1, 2,  3,  1,  -2, 3,  4,  2,  3,  0,  -3, -1, -2,  -2, 0,  1,  -1, -1, 2,  3,  1,  0,  1,  2,  2,   2,  2,   -2, 2,  -1, 0,  2,  1,  0,  1,  3,  0,
    1,  1,  -1, -1, 1,  3,  0,  4,  2,  1,  3,  3,  3,  -2,  3,  -4, 2,  1,  -1, 1,  2,  2,  1,  1,  1,  -1,  2,  1,   1,  2,  5,  -4, 2,  5,  0,  -2, 3,  1,
    -1, -2, 1,  -2, 1,  1,  3,  -3, -2, -1, 1,  2,  3,  4,   -3, -2, 2,  -1, -3, 3,  0,  0,  2,  0,  -2, -1,  -2, 1,   2,  2,  -2, 0,  3,  3,  -1, 0,  -1, 1,
    2,  2,  1,  2,  0,  1,  4,  4,  0,  0,  1,  0,  -4, 0,   -1, 2,  1,  1,  1,  -2, -2, -1, 1,  -4, -1, 4,   1,  3,   4,  0,  -1, -1, 2,  -1, 1,  1,  2,  -1,
    -1, -3, -1, 1,  0,  -1, -1, 1,  6,  2,  1,  1,  2,  3,   -3, -2, 0,  0,  3,  5,  0,  2,  5,  3,  2,  3,   0,  5,   0,  -5, 3,  2,  -2, 3,  1,  2,  1,  -1,
    3,  3,  3,  -1, 2,  2,  1,  1,  3,  4,  3,  0,  5,  2,   -2, 0,  2,  -2, 2,  -1, 0,  -2, -1, -2, 1,  4,   0,  0,   -1, 1,  1,  0,  -2, 5,  0,  0,  1,  -3,
    -5, -2, 0,  -1, 2,  1,  1,  -1, 2,  2,  -2, -1, 0,  0,   1,  0,  0,  1,  -1, 2,  3,  2,  2,  0,  1,  0,   1,  -2,  -1, 2,  3,  0,  1,  -1, -2, -4, -3, -4,
    -4, 3,  -1, 2,  6,  2,  1,  3,  -2, 2,  1,  -1, 1,  0,   -2, -1, -1, 1,  4,  -2, -1, -1, 4,  2,  2,  -1,  0,  2,   -2, -1, -3, -2, 1,  3,  1,  0,  3,  3,
    2,  -1, 1,  5,  2,  -2, 5,  0,  1,  3,  -1, 2,  -2, 0,   0,  2,  2,  1,  1,  2,  0,  1,  3,  5,  0,  -2,  2,  6,   1,  -2, 0,  -2, 0,  0,  -1, 0,  -5, -2,
    2,  3,  0,  -2, 0,  -1, 0,  2,  0,  3,  3,  2,  0,  0,   -1, -3, 0,  -1, 1,  3,  2,  -1, 1,  1,  -1, 0,   0,  1,   1,  1,  -1, 3,  2,  2,  3,  0,  2,  -1,
    0,  1,  0,  2,  2,  1,  2,  1,  2,  -2, -2, -2, 0,  0,   2,  1,  -2, 2,  3,  1,  -1, 0,  -2, 0,  0,  0,   3,  1,   -1, 0,  -3, 2,  2,  -5, -1, 0,  4,  2,
    0,  2,  3,  2,  -2, -4, -2, 0,  2,  0,  1,  2,  3,  3,   1,  0,  -2, 3,  0,  -3, 6,  3,  2,  0,  -3, 0,   -2, 2,   1,  2,  3,  -1, 3,  1,  0,  3,  5,  3,
    -2, 0,  3,  0,  0,  -2, -1, -3, -2, -2, 0,  1,  -4, 0,   3,  2,  2,  0,  -1, -1, -2, 0,  1,  0,  2,  -1,  -1, 0,   0,  -1, 2,  -1, 2,  3,  3,  1,  1,  0,
    0,  1,  1,  1,  2,  1,  0,  3,  0,  3,  4,  1,  3,  0,   1,  -2, 0,  5,  6,  2,  0,  2,  1,  -2, -4, -4,  1,  -1,  1,  2,  0,  3,  2,  -2, 0,  -2, -1, 2,
    2,  2,  2,  0,  -1, -1, -3, -1, 1,  1,  0,  0,  6,  2,   -1, 2,  3,  3,  -1, 0,  0,  -1, 2,  2,  1,  1,   4,  2,   2,  1,  0,  2,  2,  -2, 6,  3,  4,  0,
    -3, -1, -3, 2,  3,  2,  3,  2,  3,  0,  2,  2,  2,  5,   0,  -1, 2,  0,  1,  -3, 1,  -2, 0,  -2, 1,  2,   -3, 0,   0,  1,  0,  1,  0,  0,  -2, -1, 2,  3,
    4,  1,  -1, 0,  1,  -1, -2, -1, 1,  1,  3,  -1, -1, 1,   1,  3,  2,  1,  2,  0,  -1, 3,  1,  3,  3,  2,   1,  0,   1,  -3, 0,  2,  3,  2,  0,  0,  0,  -3,
    -6, -6, -1, -4, -2, 1,  0,  0,  1,  -1, 0,  -2, -2, 4,   1,  1,  2,  -1, -1, 0,  1,  -2, -2, 2,  2,  2,   2,  3,   1,  1,  2,  4,  2,  0,  1,  -1, 1,  1,
    3,  3,  2,  2,  -1, 2,  0,  1,  1,  4,  3,  1,  1,  -2,  4,  3,  3,  3,  0,  1,  4,  2,  -3, -2, 3,  1,   -4, -1,  4,  4,  -2, 0,  3,  2,  -3, -2, 0,  -2,
    3,  1,  -1, 2,  2,  -1, 2,  -3, 2,  1,  3,  5,  -4, 0,   -3, 1,  1,  1,  0,  -3, 0,  2,  1,  1,  2,  -2,  0,  0,   -2, 3,  1,  0,  1,  4,  1,  -1, 3,  2,
    -5, 2,  3,  3,  1,  4,  3,  1,  -2, -4, -5, -9, 2,  -1,  -1, 0,  1,  2,  0,  0,  -3, -3, 2,  -2, -6, 1,   0,  5,   -1, -1, 2,  4,  2,  4,  0,  -1, 1,  4,
    3,  -2, 6,  0,  -3, -3, 3,  2,  -1, 1,  4,  0,  1,  0,   2,  -1, -1, 4,  -3, 1,  -1, -3, 1,  4,  5,  0,   1,  2,   1,  3,  3,  5,  0,  0,  0,  4,  -3, -1,
    1,  1,  -2, -4, 3,  3,  0,  3,  -1, 4,  -4, -4, 2,  -1,  4,  2,  -1, -1, 2,  0,  1,  -1, 3,  0,  4,  3,   -2, -1,  -5, -3, 3,  -4, -2, 1,  -1, 2,  1,  1,
    1,  -1, 0,  -3, -3, 4,  4,  0,  1,  3,  -2, 0,  2,  2,   -5, 3,  4,  5,  4,  0,  3,  2,  0,  -2, -3, -10, 3,  1,   -2, -1, 2,  2,  2,  -2, -5, -3, 2,  -2,
    -2, 3,  1,  3,  -1, -1, 0,  1,  3,  1,  -2, -2, 0,  3,   3,  0,  7,  -1, -2, -1, 4,  2,  -4, 2,  2,  1,   2,  3,   2,  0,  -2, 3,  -3, 2,  -2, -3, 3,  4,
    3,  2,  1,  2,  -1, 1,  2,  7,  -2, 1,  1,  4,  -1, -2,  2,  1,  -3, -2, 1,  2,  1,  0,  -3, 3,  -3, -4,  -1, 2,   2,  1,  0,  0,  4,  1,  1,  0,  2,  1,
    4,  5,  -3, -1, -3, 0,  2,  -3, -1, 0,  0,  5,  1,  2,   0,  -1, 2,  -5, -2, 0,  0,  -3, 1,  3,  0,  0,   2,  0,   -4, 3,  2,  4,  4,  0,  4,  2,  1,  -2,
    -3, -7, 3,  0,  0,  1,  1,  1,  6,  -1, -5, -2, 0,  -2,  -2, 3,  2,  3,  2,  -2, -2, 3,  3,  1,  -1, 0,   1,  5,   2,  -2, 2,  -1, -3, -1, 2,  0,  -4, 3,
    4,  -1, 1,  1,  3,  2,  -3, 2,  0,  0,  -2, -1, 0,  4,   3,  1,  3,  3,  -3, 2,  2,  6,  -3, 1,  1,  0,   2,  -2,  1,  4,  -4, -4, 3,  1,  2,  -2, -1, 4,
    -2, -4, -2, 0,  3,  -1, 2,  -3, 2,  -1, 1,  -1, 1,  3,   3,  4,  -1, 1,  -5, -2, 1,  -2, -3, 0,  1,  2,   3,  1,   -1, -4, 1,  -4, -1, 0,  2,  -2, 0,  3,
    -1, 3,  5,  -3, -4, 6,  5,  3,  7,  2,  2,  1,  3,  -2,  -3, -7, 3,  2,  2,  1,  3,  1,  3,  0,  -3, -1,  2,  -1,  -2, 3,  1,  2,  1,  -2, -1, 2,  1,  1,
    1,  -2, 2,  5,  3,  -1, 1,  -3, -2, -1, 1,  2,  -4, 0,   2,  -2, -1, 2,  2,  1,  -3, 0,  -4, 1,  -2, -4,  1,  3,   1,  0,  2,  3,  0,  3,  4,  7,  -5, 0,
    2,  0,  -2, 0,  1,  3,  -3, -2, 2,  4,  1,  -1, -1, 0,   0,  -1, 0,  -2, -1, 1,  0,  -2, 2,  2,  1,  -2,  1,  1,   2,  2,  -2, 1,  -4, 0,  0,  -4, 0,  1,
    0,  0,  3,  4,  -1, -2, 0,  -3, -3, 0,  2,  -1, 0,  1,   0,  2,  8,  0,  -2, 4,  3,  3,  5,  0,  2,  -2,  1,  -1,  -3, -7, 4,  1,  4,  4,  3,  1,  3,  -1,
    -1, -2, 3,  0,  -2, 2,  2,  3,  1,  -1, 0,  2,  1,  3,   -2, -1, 4,  4,  2,  0,  1,  -2, -2, -3, 2,  1,   -3, 1,   3,  2,  0,  1,  3,  3,  -3, -1, -3, -1,
    1,  -1, 2,  2,  3,  3,  -4, -1, 1,  -1, -1, 4,  1,  -5,  -3, 0,  4,  2,  3,  1,  1,  -2, -1, 1,  -1, 1,   2,  5,   0,  1,  1,  0,  2,  -1, -1, -3, 0,  5,
    4,  2,  1,  5,  3,  -1, 0,  0,  -2, 1,  1,  2,  2,  -3,  -4, 2,  1,  3,  0,  3,  -4, -3, 0,  2,  3,  4,   1,  2,   -1, 2,  1,  1,  0,  1,  1,  -1, 0,  -2,
    -2, -1, 1,  0,  -5, 0,  -2, -1, 1,  3,  -1, 1,  1,  5,   3,  -2, 2,  2,  -3, 1,  0,  2,  1,  2,  -1, 1,   1,  -3,  6,  5,  2,  3,  -1, -2, 5,  4,  -1, 1,
    1,  0,  1,  0,  -4, 0,  0,  -3, -4, -2, -2, 1,  2,  2,   -3, -1, 1,  1,  1,  1,  -2, 1,  3,  -2, -2, 3,   0,  -4,  -4, 2,  4,  2,  2,  2,  0,  -1, -1, -2,
    1,  2,  3,  4,  -1, 0,  1,  0,  2,  -2, -2, -2, -2, 5,   4,  4,  1,  6,  1,  3,  -2, 0,  -3, 1,  1,  3,   2,  -3,  -3, 0,  0,  1,  0,  1,  -4, -3, -2, 4,
    1,  2,  0,  4,  1,  1,  1,  0,  2,  0,  0,  2,  2,  1,   -2, -1, 1,  0,  -3, 1,  -1, -2, -1, 3,  1,  0,   -1, 3,   5,  -6, 0,  2,  -1, 1,  0,  2,  3,  3,
    -1, 4,  0,  0,  4,  4,  -1, 1,  -4, -2, 3,  4,  -2, 1,   0,  1,  0,  -4, -5, -2, -2, -3, -4, -1, -3, 2,   3,  1,   -3, -1, 0,  2,  1,  1,  -1, 2,  3,  -1,
    0,  1,  -1, -2, -7, 3,  5,  2,  2,  -3, -2, -4, -3, -1,  3,  2,  2,  3,  1,  0,  1,  -1, 3,  0,  -2, -1,  -2, 3,   6,  1,  0,  7,  3,  -1, -4, -1, -4, 3,
    2,  5,  1,  -4, -3, 1,  -1, 1,  1,  3,  -1, -3, -1, 2,   0,  0,  -1, 4,  2,  2,  0,  1,  4,  0,  -1, 3,   0,  0,   -1, -1, 4,  0,  -2, 1,  -1, -2, -1, 5,
    1,  0,  0,  3,  5,  -3, 2,  1,  -2, 3,  -1, 2,  3,  1,   -1, 3,  -2, 1,  4,  6,  1,  4,  0,  -1, 1,  4,   1,  2,   0,  1,  0,  0,  -3, -3, 0,  -4, -5, -3,
    0,  0,  -2, -1, 0,  3,  0,  2,  1,  2,  -1, 2,  3,  -2,  -1, 4,  0,  -3, -5, 0,  6,  0,  3,  -1, -2, -4,  -1, 0,   2,  1,  0,  2,  -1, -4, 1,  3,  4,  -3,
    0,  0,  -1, 2,  4,  2,  0,  5,  0,  0,  -4, 0,  -2, 3,   2,  3,  1,  -4, -4, 1,  0,  0,  2,  2,  -2, -1,  -2, 3,   -1, 1,  0,  2,  1,  2,  3,  0,  2,  1,
    -1, 1,  2,  1,  0,  -2, 3,  0,  -3, -2, 0,  -2, 1,  7,   0,  2,  -1, 2,  3,  -2, 5,  2,  -1, 2,  0,  2,   3,  0,   -1, 3,  -3, 3,  1,  2,  -1, 4,  0,  1,
    1,  3,  2,  0,  -1, 1,  3,  1,  -2, -3, 1,  -3, -4, -4,  0,  0,  1,  1,  1,  0,  1,  2,  0,  1,  -1, 2,   3,  -1,  -3, 4,  0,  -3, -3, 2,  3,  4,  1,  -1,
    0,  -2, -2, 0,  2,  2,  2,  0,  0,  -3, 0,  2,  4,  -1,  -1, 0,  -1, 3,  3,  2,  2,  5,  0,  -1, -4, -3,  -3, 3,   2,  4,  -1, -4, -3, 1,  2,  -2, 1,  1,
    -3, 0,  0,  3,  -1, 2,  1,  6,  -2, 2,  1,  2,  0,  1,   -1, 0,  2,  -3, 0,  0,  4,  -1, -3, 1,  1,  -2,  4,  6,   2,  3,  1,  5,  5,  2,  4,  1,  0,  4,
    0,  3,  4,  2,  1,  2,  -3, 2,  3,  3,  0,  3,  1,  1,   1,  4,  2,  1,  -1, 1,  2,  -1, -2, -1, 1,  -4,  -3, -3,  0,  -2, -3, 0,  -1, 1,  -1, -2, -1, 1,
    1,  2,  4,  1,  1,  2,  -3, -1, 0,  -2, 3,  3,  0,  -3,  0,  0,  2,  5,  1,  1,  1,  -2, 1,  3,  1,  -1,  3,  1,   0,  1,  -1, -2, -2, -2, 4,  5,  1,  -4,
    -1, 3,  -1, 1,  1,  -3, 2,  -1, -1, 1,  1,  0,  1,  2,   -3, -1, 0,  2,  0,  -2, 0,  -1, -2, -1, -1, -1,  -1, 0,   0,  -3, 3,  3,  0,  0,  2,  1,  4,  -2,
    1,  -4, -4, -2, 2,  1,  3,  3,  2,  5,  -2, -2, -3, 1,   1,  1,  -1, -1, 0,  3,  2,  -1, 2,  2,  0,  0,   0,  -2,  0,  1,  1,  0,  -1, 3,  -1, 1,  2,  1,
    0,  4,  0,  -2, 0,  0,  1,  1,  0,  3,  2,  -2, 2,  2,   -1, 0,  4,  0,  -2, 1,  3,  -3, 1,  1,  0,  0,   -2, -4,  -1, -1, 3,  5,  1,  1,  -1, -1, 2,  3,
    2,  -3, 1,  1,  1,  1,  -2, -1, -2, -2, 4,  5,  -1, -2,  -1, 2,  1,  4,  1,  -2, 0,  -1, 0,  0,  0,  1,   0,  2,   -1, -1, -1, 1,  2,  0,  0,  -1, -2, -1,
    -3, 0,  1,  -1, 0,  -1, 2,  2,  2,  -3, 2,  0,  4,  -2,  0,  -2, -6, -4, 2,  1,  1,  3,  2,  4,  0,  -1,  -1, 1,   0,  -1, 0,  -1, -4, 1,  3,  -1, 0,  1,
    0,  1,  0,  -2, 0,  1,  0,  -2, -2, 3,  1,  -1, 1,  1,   0,  4,  -2, -1, 3,  -1, 0,  -2, -1, 1,  -1, -3,  1,  0,   -1, -2, 5,  0,  1,  1,  1,  -3, 0,  4,
    4,  0,  1,  0,  -1, 1,  3,  5,  1,  0,  2,  -1, 3,  1,   1,  2,  5,  3,  2,  3,  -1, -2, 2,  0,  3,  5,   2,  -1,  -1, 3,  3,  3,  0,  -2, 2,  0,  1,  1,
    2,  1,  0,  2,  -1, 3,  1,  1,  3,  1,  0,  -1, 0,  1,   1,  0,  1,  2,  3,  3,  3,  4,  3,  1,  1,  0,   4,  0,   2,  -3, -4, -1, 0,  2,  2,  4,  -1, 3,
    0,  -2, -3, 3,  3,  2,  2,  1,  -1, 4,  2,  -1, -2, 2,   -1, 1,  0,  0,  3,  1,  3,  2,  0,  3,  3,  1,   2,  4,   2,  6,  1,  0,  2,  -1, 1,  1,  2,  4,
    1,  -2, 0,  1,  -1, -1, 4,  0,  0,  -2, 2,  -2, -1, 2,   -1, -3, -1, -2, -2, -1, -1, 3,  2,  0,  0,  -1,  3,  -1,  1,  0,  2,  1,  2,  -1, -2, -3, -3, -1,
    3,  2,  1,  -3, -2, 0,  -2, 0,  0,  -5, 1,  -4, -1, -1,  0,  0,  -3, 2,  0,  0,  -2, -3, 2,  -1, -1, -1,  -3, 1,   -1, -3, 0,  -1, 1,  -3, -1, 2,  2,  0,
    1,  2,  4,  0,  1,  -2, -2, -1, 0,  2,  -1, 2,  -2, 1,   0,  1,  -3, 1,  1,  0,  -2, 2,  -3, 0,  2,  -3,  2,  -1,  0,  2,  -1, -2, 1,  -1, 1,  -1, -2, 2,
    1,  1,  2,  0,  0,  5,  -2, -4, 1,  -2, -1, -1, -4, 2,   -3, -3, 1,  1,  -1, 1,  4,  1,  1,  -1, 1,  -3,  1,  2,   2,  1,  1,  -2, 0,  0,  1,  2,  0,  3,
    0,  -3, 3,  0,  -2, 2,  4,  0,  3,  0,  1,  0,  -2, -1,  5,  4,  2,  -2, -1, 4,  1,  3,  2,  -2, 1,  -1,  -1, 1,   3,  -2, -2, 6,  0,  2,  1,  0,  2,  1,
    1,  0,  -1, 1,  0,  -1, 3,  0,  2,  -1, 1,  3,  0,  2,   2,  4,  4,  0,  3,  -3, -3, -1, 0,  1,  0,  3,   0,  0,   -1, 0,  -3, -1, 3,  4,  1,  1,  -1, 2,
    4,  0,  0,  3,  0,  2,  0,  -1, 1,  2,  2,  1,  -1, 3,   1,  0,  3,  1,  2,  6,  2,  2,  3,  -1, 2,  1,   0,  2,   -2, -1, 1,  -1, -1, -1, 3,  0,  3,  0,
    -3, 1,  5,  2,  0,  3,  6,  0,  -2, 1,  2,  -5, 2,  3,   -1, 0,  0,  0,  1,  -4, 0,  0,  3,  2,  2,  -1,  -2, -2,  0,  1,  1,  -2, 1,  -3, 1,  0,  -3, 1,
    0,  4,  3,  1,  0,  2,  2,  -2, 1,  3,  2,  0,  1,  1,   2,  3,  0,  2,  -1, 0,  1,  -2, -2, 3,  2,  0,   2,  0,   3,  3,  -3, 4,  -2, -1, 1,  1,  0,  2,
    2,  3,  3,  -4, 2,  1,  4,  2,  3,  -2, 1,  1,  -2, 1,   1,  -3, 1,  -1, -3, -1, -2, 3,  2,  0,  3,  -2,  0,  0,   -1, 0,  0,  -1, -1, -2, 0,  -3, -3, -1,
    1,  0,  -3, -1, -4, -1, 3,  1,  -2, -1, 2,  4,  6,  2,   -3, 3,  3,  1,  1,  5,  5,  4,  -1, -3, 1,  0,   1,  3,   -1, -1, 1,  1,  1,  -2, 2,  0,  3,  0,
    0,  -2, -4, 0,  -1, 3,  -1, -1, -1, -2, 1,  -2, -3, 0,   1,  5,  0,  1,  0,  2,  1,  -2, 2,  3,  1,  1,   3,  0,   0,  1,  2,  3,  -2, 1,  0,  -4, -1, 1,
    4,  0,  2,  3,  5,  1,  1,  4,  -3, -2, 3,  -2, -1, 0,   0,  2,  4,  -1, 3,  0,  2,  1,  3,  -3, 1,  2,   0,  1,   0,  -2, 0,  -1, -3, -2, 0,  2,  4,  2,
    2,  -1, 0,  0,  1,  1,  3,  1,  1,  0,  2,  0,  1,  -1,  -3, 1,  -4, -3, -1, 0,  5,  0,  -2, 0,  0,  1,   7,  2,   0,  0,  5,  0,  3,  4,  8,  3,  -4, -2,
    2,  -1, -2, 5,  1,  -1, 0,  3,  1,  -2, 1,  2,  3,  3,   1,  -2, -2, -2, 1,  1,  -2, -2, -2, -1, 1,  -2,  -3, 0,   0,  4,  0,  -1, -2, 4,  -1, 0,  4,  3,
    3,  3,  2,  0,  0,  -1, 0,  2,  0,  1,  0,  -2, 1,  4,   1,  -1, 1,  3,  4,  0,  3,  5,  -2, -3, 3,  2,   2,  2,   1,  4,  5,  -5, 1,  0,  3,  -1, 1,  -1,
    0,  3,  -1, 0,  -4, 1,  0,  -1, -2, 0,  0,  3,  4,  1,   2,  0,  1,  -1, 0,  2,  2,  1,  -1, 0,  3,  -2,  2,  3,   -2, 1,  -2, -1, -1, 3,  3,  1,  -3, -2,
    0,  2,  7,  0,  1,  2,  4,  3,  2,  2,  6,  3,  -1, -1,  3,  0,  0,  5,  3,  -1, 1,  1,  2,  0,  0,  1,   4,  2,   1,  -3, -2, -3, -1, 2,  -1, 0,  -2, -2,
    0,  1,  -2, 4,  1,  3,  -2, 0,  -2, 1,  -1, -1, 1,  1,   2,  1,  2,  0,  -1, -3, -2, 3,  -2, -1, 0,  -4,  0,  5,   1,  -1, 2,  4,  4,  2,  1,  5,  -1, -2,
    2,  2,  1,  0,  1,  1,  0,  -4, 1,  1,  0,  0,  1,  0,   -1, 4,  -4, 0,  1,  0,  1,  -2, -4, -2, 2,  3,   2,  0,   -1, 2,  1,  -1, 1,  2,  0,  0,  1,  0,
    3,  -2, -1, 2,  -4, 0,  -4, 0,  0,  2,  5,  -2, -5, -3,  2,  4,  3,  -1, 1,  0,  4,  2,  0,  6,  5,  -1,  -5, 2,   3,  1,  0,  3,  -1, 3,  0,  2,  2,  1,
    0,  0,  3,  3,  0,  1,  1,  -4, -3, 0,  0,  0,  1,  1,   0,  0,  -2, 2,  -1, 3,  0,  2,  -2, 1,  -2, 2,   -1, 2,   1,  0,  4,  0,  -1, -1, -2, 3,  0,  0,
    0,  -4, -1, 7,  2,  1,  4,  2,  2,  1,  0,  4,  -4, -2,  0,  4,  2,  1,  0,  3,  4,  -4, 0,  2,  1,  -1,  2,  3,   0,  6,  -1, 2,  0,  0,  1,  -1, -5, -2,
    1,  3,  5,  -1, 0,  1,  1,  0,  1,  1,  0,  -2, 0,  1,   2,  -1, -2, 2,  -4, -1, -4, -2, 1,  1,  2,  -1,  -1, 2,   -2, 1,  0,  -2, 0,  -3, 1,  3,  2,  -3,
    2,  0,  5,  4,  -1, -2, -2, 2,  3,  2,  4,  -1, 2,  1,   3,  1,  -2, -1, -1, 1,  1,  0,  -2, -5, 4,  1,   -2, -3,  2,  -1, 1,  1,  -1, 4,  -1, 1,  0,  1,
    1,  -1, -3, 3,  1,  0,  1,  0,  1,  -1, 3,  3,  5,  0,   3,  2,  2,  -1, 2,  0,  -1, -1, -2, 4,  4,  4,   1,  3,   3,  2,  1,  0,  1,  0,  -1, 3,  2,  3,
    -1, 0,  2,  1,  3,  -2, 2,  0,  3,  1,  3,  1,  -2, -1,  -1, 1,  -2, -2, -1, 2,  1,  2,  -1, 0,  2,  2,   -1, -1,  2,  0,  1,  3,  1,  -5, -3, 2,  -2, -1,
    2,  1,  1,  1,  -2, 2,  2,  0,  -1, 0,  2,  2,  1,  -2,  3,  0,  4,  3,  -2, -3, 1,  2,  1,  0,  3,  -1,  2,  1,   2,  3,  -3, -2, 0,  3,  -1, 1,  -2, -3,
    3,  3,  -1, -2, 0,  0,  1,  0,  2,  3,  0,  2,  3,  -1,  0,  -1, 1,  2,  0,  2,  1,  1,  0,  0,  0,  2,   3,  0,   1,  2,  0,  0,  -1, 1,  -1, -1, -3, 1,
    2,  6,  2,  3,  3,  1,  3,  -1, 0,  4,  1,  3,  2,  3,   -1, -1, 4,  3,  2,  0,  0,  2,  0,  2,  3,  2,   -2, -1,  3,  0,  -3, -2, -1, 3,  2,  3,  -1, 0,
    2,  4,  -1, -1, 0,  -1, 4,  4,  -1, -4, -1, 4,  1,  -2,  2,  2,  0,  2,  0,  0,  1,  -2, 1,  1,  0,  2,   0,  0,   2,  1,  4,  2,  -4, -4, -1, 2,  3,  -2,
    -1, -2, 2,  4,  2,  2,  -4, -1, -3, 0,  1,  0,  -4, -7,  1,  3,  -2, -1, -2, -1, 0,  -2, -1, 3,  0,  1,   1,  -2,  -1, 0,  0,  2,  0,  0,  -1, -3, -1, 1,
    1,  3,  2,  0,  1,  -1, -1, 0,  -4, 1,  2,  -1, 0,  -3,  0,  3,  2,  2,  4,  2,  1,  -2, 1,  -1, -1, 2,   -1, 3,   -1, -2, 2,  -1, 0,  2,  -1, 1,  1,  2,
    3,  1,  1,  0,  1,  0,  -3, -2, -4, 1,  5,  1,  -1, 1,   2,  1,  -1, -2, 1,  0,  2,  1,  -1, -4, -1, 1,   0,  -3,  2,  1,  0,  3,  -1, 0,  2,  0,  0,  2,
    0,  6,  0,  0,  2,  3,  1,  3,  -3, -1, 0,  1,  4,  1,   0,  -1, 4,  4,  5,  1,  -3, -1, 0,  1,  3,  1,   -2, -6,  3,  2,  -4, -1, 1,  0,  1,  0,  1,  3,
    3,  0,  2,  2,  -1, 1,  0,  2,  -1, 3,  1,  -1, 2,  2,   1,  4,  3,  1,  3,  0,  -1, 1,  -3, 1,  3,  1,   1,  -2,  0,  1,  0,  2,  0,  1,  1,  -1, 1,  -1,
    -2, 2,  0,  3,  -2, -3, 2,  1,  4,  -1, -5, 0,  2,  1,   2,  -3, 2,  1,  1,  0,  1,  0,  -1, 1,  4,  2,   0,  1,   3,  1,  1,  0,  2,  3,  5,  1,  -1, -2,
    1,  4,  0,  -1, 4,  1,  -1, 3,  -1, 0,  2,  1,  0,  -2,  -1, -1, 1,  0,  2,  3,  -2, 0,  -2, -1, -1, 0,   -1, 0,   1,  -3, 4,  2,  3,  0,  -2, -2, 2,  1,
    2,  1,  -2, -5, 2,  5,  -2, -1, 1,  0,  1,  0,  1,  0,   3,  2,  1,  0,  -3, -1, -1, 1,  -2, -1, -1, -1,  1,  2,   0,  2,  3,  2,  1,  2,  -1, -1, -2, 1,
    4,  0,  -1, -3, -2, 1,  -1, 0,  -1, -1, -2, -2, -1, -3,  -5, 0,  0,  0,  -1, -1, 3,  -1, 2,  0,  -3, 1,   -1, 2,   6,  -3, 2,  1,  2,  -1, -1, -1, -2, 0,
    4,  1,  -1, 1,  1,  4,  1,  -3, 0,  2,  4,  0,  -2, -2,  2,  2,  1,  -3, 2,  2,  -3, 1,  -1, -1, 0,  -3,  2,  1,   2,  -1, 2,  0,  1,  3,  -2, 0,  0,  -1,
    -1, 1,  5,  0,  -2, 4,  2,  3,  1,  -3, 0,  2,  0,  1,   3,  3,  1,  3,  4,  2,  1,  -1, -2, 1,  2,  -3,  3,  4,   -1, 0,  2,  1,  0,  1,  -1, 3,  -1, 0,
    -1, 0,  2,  -1, -3, -3, 2,  1,  0,  5,  2,  0,  0,  2,   4,  3,  2,  3,  0,  -3, -2, 1,  0,  2,  2,  -1,  2,  0,   -1, -1, 2,  -1, -4, 0,  0,  4,  3,  1,
    -4, -2, -1, -3, 3,  -3, -4, 0,  2,  2,  4,  3,  5,  2,   4,  4,  2,  -4, -1, 3,  4,  2,  -5, -5, -3, -5,  -2, 0,   2,  -1, -1, 0,  1,  3,  2,  3,  -5, -1,
    1,  -3, 1,  4,  1,  0,  -1, 0,  0,  -1, 3,  -2, -1, -1,  -4, 3,  5,  -2, -1, 2,  3,  3,  -2, -2, 0,  1,   2,  0,   3,  2,  1,  4,  3,  1,  3,  0,  -7, -1,
    -3, 0,  2,  4,  -1, -1, 3,  0,  -2, 0,  -1, 3,  -2, 1,   -2, -3, 1,  -2, -2, 0,  2,  -1, -2, 1,  0,  1,   -1, 0,   5,  4,  2,  2,  0,  -2, -1, -1, 1,  1,
    2,  4,  1,  2,  1,  -1, 2,  0,  -1, 1,  0,  2,  3,  0,   -4, -3, 1,  -5, 0,  2,  -2, 1,  1,  5,  3,  4,   6,  3,   2,  3,  4,  -1, 2,  4,  4,  2,  -7, -4,
    -1, -5, -2, 1,  1,  -3, 0,  1,  1,  5,  1,  0,  -3, 1,   2,  -2, 1,  3,  2,  -1, 3,  1,  -2, 0,  0,  2,   -1, -1,  -1, 2,  4,  -1, 0,  2,  3,  4,  1,  1,
    -1, 0,  0,  2,  4,  4,  2,  3,  4,  3,  2,  2,  -1, 0,   -2, -3, 3,  1,  -2, -1, 6,  -1, 1,  2,  -2, 5,   1,  0,   -1, 0,  3,  2,  1,  -4, 3,  1,  -1, 5,
    -2, -1, -1, 2,  2,  2,  2,  2,  2,  0,  0,  -1, -2, 1,   3,  5,  2,  0,  0,  -2, 0,  0,  1,  0,  1,  3,   4,  2,   -4, -1, 1,  -3, 0,  1,  -1, -2, 3,  3,
    6,  4,  5,  4,  2,  5,  1,  -3, 0,  3,  4,  4,  -4, -4,  0,  -2, 0,  -1, 3,  -1, -3, -1, 0,  3,  0,  3,   -1, 1,   1,  -1, 2,  7,  -1, 0,  1,  1,  0,  1,
    1,  0,  -3, -2, -6, 1,  3,  1,  -1, 3,  4,  1,  1,  0,   -1, -2, -1, 1,  2,  2,  1,  0,  1,  0,  3,  1,   -2, -2,  -2, -3, 3,  -3, -4, 0,  1,  0,  -2, 0,
    -2, 0,  1,  -3, -3, -2, 0,  0,  0,  -2, 2,  -2, -4, 1,   -1, 0,  -1, 3,  3,  -2, 0,  2,  0,  -2, -4, 2,   -2, 2,   2,  5,  4,  -3, -2, -2, 0,  0,  1,  -1,
    -1, -2, 2,  -1, -6, -3, 1,  -1, -3, 0,  -2, -3, -1, 0,   2,  2,  5,  0,  -1, 4,  2,  -2, 0,  -2, -1, 1,   -4, -7,  -2, -4, 1,  -3, 0,  -1, -2, -3, 1,  4,
    1,  5,  -1, 1,  5,  -2, 1,  1,  -1, -3, -1, -1, 2,  0,   2,  3,  -2, -2, -4, 0,  3,  1,  0,  1,  3,  2,   -1, 1,   -2, 0,  1,  4,  3,  -1, 2,  5,  4,  3,
    3,  1,  -1, 1,  1,  -1, 0,  3,  1,  -1, 4,  -1, 0,  0,   -2, 3,  2,  1,  0,  -1, 1,  3,  1,  -2, 0,  0,   -1, 1,   3,  2,  0,  0,  3,  0,  -1, 3,  -2, -1,
    -3, 3,  -1, -1, 4,  2,  4,  0,  -2, -2, 2,  -3, -1, -2,  -1, 4,  2,  -3, -4, 0,  1,  -3, 0,  -1, -2, -2,  0,  0,   1,  -1, 5,  2,  0,  6,  4,  -4, -1, 0,
    3,  1,  -2, -4, 0,  -4, -3, -2, 5,  -1, 0,  -1, -2, 1,   -2, -3, 1,  2,  2,  1,  3,  3,  0,  -2, 3,  3,   -1, -2,  -3, 0,  -3, 4,  1,  -2, -1, 1,  1,  1,
    3,  -3, 0,  -1, 1,  -2, 1,  2,  4,  1,  -1, 1,  2,  -1,  -2, 0,  1,  1,  0,  1,  0,  -2, -1, 1,  -3, -4,  0,  -3,  -4, -1, 5,  6,  3,  1,  4,  3,  -3, 2,
    1,  5,  4,  1,  -1, -2, 3,  1,  0,  -2, -2, 1,  2,  2,   4,  0,  -3, 1,  3,  1,  -3, 4,  4,  0,  0,  1,   -6, -5,  -5, -2, -4, -4, 2,  3,  4,  2,  0,  -3,
    -6, 1,  -3, -5, -1, 1,  6,  2,  7,  0,  4,  10, 8,  3,   8,  4,  8,  8,  -2, 2,  5,  1,  0,  -1, -1, 1,   -2, 0,   -2, -1, 1,  4,  1,  0,  3,  2,  3,  -3,
    1,  3,  0,  1,  -3, 1,  -4, 0,  0,  -2, -1, 2,  0,  0,   0,  0,  -1, 0,  4,  -1, 1,  2,  5,  0,  -2, 1,   3,  -3,  -4, 0,  2,  0,  0,  1,  0,  1,  3,  1,
    -3, -2, 0,  -2, -2, 0,  3,  6,  2,  2,  2,  0,  -2, 3,   5,  4,  1,  1,  -1, -1, 0,  2,  3,  -4, -2, -1,  -2, 3,   2,  0,  -2, 0,  2,  1,  -3, 8,  3,  -2,
    0,  1,  -3, -6, -7, -3, -3, -4, 1,  1,  4,  1,  1,  -3,  -5, 0,  -5, -2, -2, 3,  3,  2,  6,  -1, 5,  4,   8,  4,   5,  2,  7,  8,  -3, 2,  2,  0,  0,  -1,
    -3, 2,  0,  3,  -3, 0,  2,  4,  1,  1,  1,  -1, 0,  -3,  -1, 3,  -2, 0,  -1, 1,  -3, 0,  -1, -1, -2, 1,   3,  0,   -1, -2, -2, -2, 2,  0,  1,  4,  4,  -3,
    1,  3,  0,  -1, -2, 0,  2,  0,  0,  -1, -1, -1, 1,  -1,  -3, -4, -2, -2, -1, 0,  4,  6,  4,  1,  1,  3,   -3, 2,   5,  4,  -1, 1,  2,  -4, 2,  1,  2,  0,
    0,  -1, -1, 5,  2,  -3, -1, 2,  5,  3,  -4, 7,  5,  2,   2,  2,  -2, -6, -4, -4, -2, -2, 0,  1,  3,  -1,  0,  0,   -8, -1, -5, -5, -2, 1,  3,  3,  4,  1,
    3,  3,  7,  6,  7,  1,  6,  6,  -4, 0,  2,  -1, 2,  1,   0,  2,  -1, 1,  0,  -3, 2,  4,  1,  2,  3,  0,   0,  -3,  0,  2,  -1, 0,  1,  1,  -3, 1,  -1, -1,
    0,  4,  1,  2,  2,  0,  1,  -1, 3,  2,  1,  2,  2,  1,   -2, 4,  3,  0,  -2, 0,  -1, 1,  1,  0,  -1, -3,  2,  0,   -1, -4, 0,  -1, -2, 3,  6,  6,  3,  0,
    0,  2,  -1, 2,  4,  4,  0,  -1, 1,  -1, -1, 0,  2,  -5,  -1, 0,  1,  3,  1,  -4, 3,  3,  2,  3,  -4, 5,   4,  4,   -1, 2,  -4, -3, -5, -4, -2, -4, -2, 0,
    3,  -2, -1, -2, -9, -1, -7, -5, -1, 1,  4,  1,  5,  3,   2,  7,  8,  7,  9,  1,  5,  5,  -3, 2,  1,  -4,  0,  2,   0,  2,  0,  3,  0,  0,  2,  3,  4,  0,
    2,  -1, 0,  -4, 1,  0,  1,  -1, 1,  2,  -5, 1,  -3, -1,  -2, 2,  1,  -1, 3,  1,  0,  2,  1,  2,  0,  2,   4,  1,   -1, 5,  2,  -3, 3,  -1, 0,  0,  3,  -1,
    -2, -2, 2,  -1, -1, -4, -4, -1, -3, 0,  4,  7,  3,  2,   0,  1,  -1, 1,  5,  3,  0,  -2, -2, 0,  0,  1,   4,  -4,  -1, -1, 0,  3,  1,  -2, 1,  5,  3,  2,
    -6, 3,  4,  5,  2,  1,  -4, -3, -3, 0,  -3, -4, -1, -1,  2,  1,  -1, -3, -7, -3, -5, -7, 4,  -3, 3,  3,   3,  3,   5,  5,  10, 6,  8,  3,  5,  7,  1,  0,
    3,  0,  0,  1,  5,  1,  -3, -1, 6,  3,  -2, -1, -1, -2,  0,  3,  3,  -2, 3,  -1, 0,  0,  -1, 3,  3,  0,   0,  -2,  -3, -2, -6, -3, -4, 0,  -2, 0,  1,  -4,
    0,  2,  0,  1,  -1, 4,  -1, -4, 1,  1,  -1, 2,  -2, -1,  0,  -4, -2, 1,  4,  1,  1,  0,  0,  0,  1,  2,   0,  0,   2,  2,  3,  0,  -1, 4,  -1, 2,  0,  -1,
    0,  3,  -5, -3, 2,  5,  2,  -2, 6,  1,  -2, 4,  -1, 1,   3,  -2, -3, -4, -3, -1, 3,  1,  -2, 2,  2,  1,   2,  1,   3,  4,  -4, -2, -4, 3,  2,  5,  4,  4,
    4,  -1, -2, 0,  -1, -1, -1, 6,  2,  -1, 2,  -3, -3, 2,   6,  3,  6,  -4, 4,  2,  -1, 1,  2,  1,  -4, -1,  -2, 0,   0,  2,  4,  -3, 1,  1,  1,  0,  2,  1,
    3,  -2, 3,  -3, 0,  0,  -3, -3, -5, 0,  -3, 2,  2,  -3,  1,  2,  0,  3,  2,  5,  0,  -4, 0,  2,  -2, 0,   -2, -3,  0,  -3, -1, 0,  4,  1,  2,  0,  -2, 2,
    2,  2,  1,  1,  0,  2,  2,  0,  -3, 4,  0,  2,  0,  0,   1,  1,  -5, -2, 2,  6,  -1, -5, 6,  1,  -1, 2,   -1, 0,   3,  -2, -4, -1, -4, -2, 2,  0,  -3, 3,
    1,  -2, 3,  1,  1,  8,  -5, -1, -4, 1,  2,  3,  3,  3,   3,  1,  -2, 1,  0,  -2, 3,  3,  3,  -3, 1,  0,   -1, -1,  5,  3,  3,  -3, 3,  -3, -2, 2,  6,  -2,
    1,  0,  -3, -4, 3,  3,  6,  -1, 2,  0,  1,  3,  -2, 0,   1,  -2, 1,  0,  -2, 1,  -4, -5, -8, 2,  1,  -1,  -1, -1,  1,  3,  2,  3,  3,  3,  0,  -3, 2,  0,
    1,  -1, -2, 0,  -1, -6, -1, 0,  3,  1,  0,  0,  0,  2,   5,  0,  -2, 1,  0,  1,  3,  0,  -3, 2,  0,  2,   0,  1,   1,  2,  -2, -1, 3,  6,  4,  -4, 6,  2,
    -1, 3,  1,  0,  0,  0,  -3, -5, -3, 0,  3,  -2, -1, 2,   5,  3,  5,  5,  3,  5,  -2, -2, 1,  1,  -1, 4,   4,  2,   4,  -1, -3, 0,  1,  -2, 1,  3,  1,  -2,
    -1, 0,  -2, -1, 6,  4,  5,  -4, 3,  -2, 0,  -1, 5,  0,   -1, -3, -2, -2, 2,  1,  5,  2,  2,  -2, 1,  2,   -3, -1,  5,  -3, -2, 0,  -3, -1, -4, -5, -7, 0,
    0,  2,  -1, -2, 2,  2,  3,  2,  2,  5,  1,  -4, 1,  -1,  0,  -2, -2, 1,  -2, -6, -1, 2,  4,  1,  3,  0,   0,  4,   2,  -3, 0,  2,  1,  2,  4,  0,  -3, 1,
    2,  1,  4,  1,  0,  1,  -1, -1, 3,  4,  2,  -4, 5,  0,   1,  1,  2,  0,  0,  0,  -4, -3, -3, -2, 2,  -1,  -1, -1,  2,  1,  4,  2,  0,  3,  0,  -2, -3, -1,
    -2, 2,  5,  0,  3,  -1, -4, 1,  1,  -1, 1,  4,  3,  1,   0,  1,  0,  2,  6,  4,  6,  -2, 0,  -1, -1, -1,  5,  -1,  -2, -1, 0,  0,  1,  3,  4,  0,  1,  -1,
    0,  3,  -5, -1, 5,  -3, -1, -1, -2, 2,  -2, -2, -4, -3,  -1, 1,  0,  -3, 0,  2,  1,  1,  1,  6,  1,  -4,  1,  -1,  -4, -2, -2, 3,  -1, -6, -1, 0,  5,  1,
    3,  4,  -1, 4,  2,  -1, -3, 1,  2,  2,  2,  2,  -2, 0,   2,  1,  3,  2,  0,  2,  0,  0,  5,  5,  0,  0,   4,  -3,  -3, -1, -1, 0,  5,  -1, -3, -5, -3, -2,
    2,  -1, -1, 2,  3,  1,  2,  3,  1,  1,  0,  -2, -6, 0,   1,  1,  7,  0,  2,  -1, -4, 3,  1,  -3, 1,  4,   2,  2,   -2, -1, -1, -1, 8,  4,  2,  -4, 2,  5,
    2,  1,  0,  -4, 1,  -1, -2, 0,  -2, 4,  0,  -4, 0,  -1,  2,  -1, -1, 5,  2,  -2, 0,  7,  -1, -1, -1, -1,  0,  1,   1,  2,  0,  1,  0,  -2, 2,  1,  2,  -2,
    -1, 1,  -1, -2, 2,  0,  -2, 0,  3,  3,  3,  3,  -1, 3,   0,  2,  4,  -2, 0,  -2, 1,  1,  2,  -2, 0,  -1,  -2, 0,   -2, -2, 1,  3,  -1, 3,  3,  5,  3,  -1,
    -2, -1, -1, -1, 0,  1,  -1, -1, 1,  -1, 0,  -1, 0,  0,   -1, 1,  1,  -1, 0,  0,  -2, 0,  2,  5,  1,  2,   4,  -1,  4,  -1, 0,  0,  -1, 1,  1,  4,  5,  4,
    -3, -4, -3, -2, 0,  4,  0,  -1, 2,  -2, 4,  7,  1,  2,   1,  2,  -1, -2, -1, 0,  -5, 1,  0,  4,  1,  -1,  1,  1,   1,  -1, -1, 1,  4,  0,  0,  4,  -1, -4,
    -2, 0,  0,  -2, -1, 3,  0,  1,  1,  -1, 0,  0,  0,  -2,  -1, 1,  0,  2,  5,  2,  -2, 0,  1,  2,  4,  4,   -1, 0,   1,  3,  4,  -2, 0,  -2, 0,  0,  2,  0,
    1,  -1, -2, -2, -2, -1, 0,  3,  1,  1,  1,  3,  1,  -3,  -1, 1,  -2, 1,  2,  -2, -1, -2, 2,  -2, -1, -3,  0,  0,   0,  -1, -1, -2, 0,  2,  -1, 0,  0,  2,
    4,  3,  3,  -1, 3,  -1, -1, -2, 0,  2,  2,  1,  4,  3,   -1, -3, -3, -3, -1, 3,  -2, 0,  0,  -1, 3,  6,   0,  2,   3,  3,  1,  -2, 1,  -1, -4, 1,  0,  5,
    1,  -2, -2, -1, 2,  0,  2,  2,  3,  3,  0,  3,  1,  -3,  0,  -1, 1,  1,  2,  2,  3,  0,  2,  1,  2,  1,   -1, -2,  -2, 2,  0,  0,  4,  0,  -1, 1,  -2, 2,
    3,  6,  1,  1,  3,  2,  4,  -3, -1, 0,  0,  -2, 1,  -2,  1,  -2, -2, -1, -2, 0,  -1, 3,  0,  1,  1,  2,   2,  -2,  -2, 1,  -1, 1,  4,  -1, 0,  -2, 1,  -4,
    0,  -1, 1,  -1, 2,  1,  3,  -2, 2,  3,  2,  1,  0,  3,   3,  2,  1,  0,  1,  0,  0,  0,  -2, 1,  2,  3,   5,  3,   -2, -4, -4, -3, 0,  3,  -3, 0,  1,  1,
    5,  6,  0,  3,  2,  2,  0,  -1, 0,  -1, -2, 1,  1,  3,   1,  -2, 0,  2,  1,  2,  1,  1,  3,  3,  2,  2,   1,  -2,  0,  0,  1,  -1, 2,  0,  2,  1,  3,  2,
    1,  1,  1,  -2, -2, 2,  2,  0,  0,  3,  1,  0,  2,  1,   3,  4,  1,  1,  1,  3,  4,  -1, 2,  1,  -1, -1,  1,  0,   3,  -2, -3, 1,  -2, 1,  -1, 1,  -1, 1,
    1,  4,  3,  -2, -1, -1, -2, 1,  3,  -1, 0,  0,  2,  -3,  1,  0,  2,  -3, 1,  1,  1,  0,  1,  2,  2,  1,   0,  2,   4,  1,  0,  2,  2,  1,  0,  0,  -2, 1,
    4,  2,  6,  4,  -3, -4, -4, -2, -2, 3,  -4, 0,  1,  0,   3,  5,  -1, 1,  0,  4,  2,  -1, 1,  -1, -5, 1,   -1, 4,   2,  -1, 0,  2,  0,  0,  1,  2,  4,  1,
    2,  4,  3,  -4, 2,  2,  1,  1,  2,  2,  2,  1,  2,  2,   -2, 2,  -1, -1, -1, 2,  0,  -1, 1,  2,  0,  3,   3,  1,   2,  2,  -3, 0,  0,  3,  3,  -1, 4,  3,
    -1, -1, 1,  1,  3,  -3, -1, -1, -2, 0,  1,  0,  0,  -1,  1,  -1, 2,  -2, 0,  0,  -1, 3,  2,  -1, 2,  0,   4,  -4,  0,  -2, 2,  -1, 0,  -1, 1,  0,  -1, 0,
    1,  1,  0,  1,  1,  0,  -1, 1,  2,  -2, 1,  -1, -2, -1,  2,  1,  4,  4,  -3, -4, -5, -2, 0,  5,  -3, -2,  4,  1,   4,  4};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer0_bias_tensor[128] = {
    -853196190,  898256768,   -810721055,  1675913537,  1057076838,  -652447458, 293632455,   1208140579,  1271299258, -532601258,  -1423840395, 396680826,
    228257682,   1448217429,  10711121,    -1393923126, 332044751,   630848092,  955875212,   -535186701,  -559563735, 740544745,   -449497733,  1246183526,
    1659485057,  -231581823,  -291785710,  1924485414,  -67960216,   529646466,  1069265355,  -645622052,  2130405032, 1182655498,  1414606670,  -414409578,
    -1049689858, -964370239,  93814646,    -338323684,  -1012016260, -988377924, 16990054,    -1215158210, 601300172,  12557866,    898079644,   930390131,
    -429552887,  -1146459296, -526691674,  122623868,   247833179,   195016272,  -586156863,  239707501,   757165450,  -134812385,  1852654584,  367871604,
    1870013987,  -382645564,  1698074477,  460578203,   1151260833,  -635649629, -303974227,  -876095828,  80887431,   1492716433,  352728295,   -1995592647,
    -575445742,  494558311,   -779326390,  799640585,   -106003163,  1201492297, -606471058,  -346449362,  91598552,   -1869821762, -303604878,  -80518082,
    2216094,     1632330355,  2140554579,  -479784351,  -67221518,   795577746,  -1960135143, 316532093,   26962477,   119299727,   -1754584874, -348665456,
    949226930,   1095858483,  2092731434,  -225672239,  -41736437,   1379887864, 32502712,    563626574,   -460947552, 1636954768,  43952531,    -164729654,
    -51708860,   -696222865,  -1574904136, -746454329,  233059219,   2100295538, 624569159,   -585048816,  -507116177, 880528016,   842854418,   -1033069153,
    -271471515,  -202033903,  -847655955,  -12557866,   -1720427642, 1608884244, -147370251,  339801080};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer0_mul_tensor[128] = {
    369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349,
    369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349,
    369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349,
    369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349,
    369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349,
    369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349,
    369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349, 369349};

static PI_L2 int8_t AnomalyDetection_heavy_6_layer1_weight_tensor[16384] = {
    15,  -11, -2,  -25, 7,   -1,  12,  14,  -1,  2,   -24, 5,   -17, 4,   3,   1,   -16, 11,  -6,  -12, -4,  7,   14,  -1,  6,   14,  27,  -25, -7,  3,   -4,
    -11, 7,   3,   12,  -14, 25,  -1,  9,   -18, -5,  -14, 3,   10,  -7,  4,   6,   -9,  6,   7,   5,   -5,  5,   14,  -23, 22,  -9,  -5,  -16, 6,   -14, -2,
    -15, 12,  -5,  -13, 12,  -20, 6,   14,  -7,  6,   11,  -1,  6,   -4,  -23, 12,  -17, 3,   -4,  6,   1,   -13, -10, -22, -25, -8,  -17, -11, 21,  -8,  13,
    11,  -12, -1,  -5,  13,  13,  -14, 3,   16,  14,  2,   -7,  -12, 16,  -9,  -2,  11,  22,  -14, -4,  5,   2,   27,  5,   -7,  -3,  7,   13,  -15, -8,  -16,
    -4,  0,   9,   18,  11,  -1,  12,  4,   21,  -12, -11, 9,   -12, 13,  13,  12,  14,  19,  -5,  7,   -13, -1,  -9,  -1,  -2,  5,   -2,  -6,  -5,  -2,  -22,
    -11, 4,   4,   13,  13,  -9,  22,  9,   -11, -11, 15,  -25, 2,   0,   2,   -7,  -14, -5,  -13, -3,  -17, 9,   11,  0,   -13, 16,  0,   -22, 1,   20,  -5,
    -5,  7,   -9,  -10, -9,  -5,  14,  6,   -5,  -17, 7,   29,  -17, 8,   -2,  14,  4,   3,   6,   -15, -1,  -3,  -4,  0,   -6,  -29, -1,  -10, -2,  4,   5,
    13,  -5,  0,   16,  -26, 4,   -10, -10, -22, -6,  -5,  7,   14,  -9,  -13, -4,  4,   23,  9,   24,  10,  -5,  14,  -2,  -21, 7,   -16, 3,   11,  17,  3,
    -8,  4,   8,   3,   13,  -5,  9,   -1,  4,   6,   -11, 17,  -13, 13,  -8,  1,   12,  -18, 10,  -1,  -12, 4,   19,  -10, 2,   -24, -4,  19,  -25, 12,  -3,
    -1,  -29, -22, -3,  -16, 1,   3,   -2,  -17, -4,  -1,  0,   12,  13,  -17, 3,   -9,  -7,  -8,  23,  -6,  13,  -13, -3,  5,   32,  -6,  -21, 0,   1,   -20,
    2,   2,   -8,  17,  0,   11,  6,   6,   -24, 9,   -1,  -6,  -3,  -21, -31, -8,  28,  -14, 29,  5,   19,  -18, -9,  2,   22,  8,   -15, 3,   2,   -5,  -4,
    18,  0,   -6,  15,  -12, 12,  21,  -6,  19,  20,  14,  -3,  9,   6,   -9,  10,  -21, 5,   -12, 15,  1,   -3,  -16, -1,  20,  0,   -11, -33, 7,   0,   -8,
    -9,  -11, 10,  -6,  35,  -4,  16,  -22, -3,  -6,  10,  11,  3,   1,   12,  4,   13,  -23, 8,   -9,  11,  33,  -7,  -30, -24, -10, 15,  -1,  -10, 14,  13,
    13,  -17, -9,  -9,  -4,  -14, -21, -9,  -37, 2,   4,   -1,  15,  14,  -36, -18, -26, -1,  -27, 14,  2,   1,   25,  -20, -11, 6,   6,   -15, 0,   -3,  3,
    0,   18,  -9,  -43, 11,  -25, 18,  14,  -10, -12, -12, -16, 38,  14,  -11, 42,  -15, 3,   15,  34,  17,  11,  9,   -20, 18,  11,  9,   8,   18,  -14, 4,
    -20, -6,  43,  33,  -9,  -28, -9,  18,  -6,  8,   -2,  -3,  -6,  8,   4,   14,  10,  15,  -18, 6,   11,  -9,  0,   -14, -25, -22, -2,  -1,  12,  2,   -10,
    -7,  -11, 26,  9,   -7,  20,  14,  8,   -50, 17,  7,   14,  -25, 4,   -24, 8,   -11, 1,   -6,  12,  1,   -9,  12,  11,  13,  12,  -11, -12, -8,  -2,  -15,
    4,   -11, 5,   4,   10,  5,   -8,  -5,  -10, -5,  -15, -3,  -37, 6,   -9,  -8,  1,   -8,  -7,  10,  6,   6,   -11, 7,   19,  19,  4,   -15, 5,   -3,  -13,
    5,   0,   -2,  -15, -9,  23,  7,   -23, 4,   4,   -5,  -12, 6,   0,   0,   5,   -2,  -8,  -1,  14,  0,   12,  15,  -11, 5,   0,   -27, -14, 9,   9,   -12,
    14,  -3,  2,   4,   16,  -12, -10, 14,  -17, 20,  17,  16,  -24, 1,   -5,  12,  2,   -1,  8,   9,   7,   14,  -2,  -20, -8,  11,  7,   -9,  8,   -2,  21,
    11,  29,  20,  -5,  9,   10,  -1,  -11, 8,   -1,  16,  -3,  0,   12,  -12, 14,  -4,  3,   6,   -17, -7,  -7,  -36, 0,   2,   6,   2,   18,  9,   -10, -8,
    14,  11,  10,  13,  3,   14,  -11, 16,  -10, 3,   4,   14,  5,   -16, -30, 11,  3,   -7,  -17, -12, -12, 14,  -2,  -2,  -15, -14, 7,   3,   -7,  3,   -5,
    19,  -11, -7,  3,   3,   -8,  16,  18,  4,   -14, 17,  -6,  -6,  -10, -34, -24, -3,  -8,  -32, 28,  1,   2,   0,   17,  25,  5,   -2,  2,   -6,  11,  -3,
    11,  24,  7,   3,   -7,  1,   -11, -3,  -15, 3,   -4,  17,  11,  -4,  4,   12,  -7,  -6,  13,  25,  16,  8,   -7,  8,   -7,  -18, 5,   -14, 11,  -10, 2,
    3,   -4,  17,  -2,  3,   -20, 5,   -2,  10,  -14, 3,   2,   -1,  -3,  13,  -7,  14,  4,   -6,  -30, -6,  -7,  -2,  1,   3,   -4,  12,  1,   4,   -9,  22,
    5,   3,   13,  3,   -11, -31, 7,   9,   -21, 10,  10,  7,   10,  0,   -13, 5,   5,   -9,  15,  1,   -14, 18,  15,  -4,  -37, -17, -15, -3,  -16, 3,   -6,
    8,   -31, 22,  -6,  5,   5,   -22, 21,  0,   17,  16,  19,  -8,  26,  -8,  -15, 17,  -8,  10,  5,   7,   -10, -6,  -3,  -18, 17,  5,   17,  -16, -7,  7,
    0,   -5,  -14, 5,   15,  -24, -4,  -11, -22, 12,  16,  7,   -5,  12,  -2,  -15, 2,   -1,  5,   -10, -16, -14, -7,  24,  14,  -4,  -11, -4,  -19, -5,  8,
    1,   19,  10,  12,  -16, -11, 6,   -11, 11,  -18, -26, 23,  -5,  10,  8,   -26, 10,  1,   4,   1,   -4,  29,  9,   15,  5,   -18, -35, -1,  -3,  4,   -11,
    16,  19,  -4,  8,   24,  4,   4,   2,   10,  -10, -13, -4,  3,   -4,  6,   -1,  -14, -5,  -19, 16,  -4,  10,  17,  -13, -13, -11, -29, 0,   -18, -3,  -1,
    0,   14,  19,  -9,  -5,  9,   14,  -7,  1,   -22, 7,   -2,  10,  -10, 12,  5,   -10, -5,  11,  10,  -10, 16,  -7,  3,   2,   -7,  -3,  0,   -19, 25,  8,
    -6,  6,   4,   -15, -11, -28, 17,  11,  -22, -1,  -6,  10,  11,  12,  -6,  -3,  6,   14,  -1,  -11, -10, -32, 8,   4,   17,  -15, -2,  -11, 12,  -11, 8,
    -4,  -11, 17,  -24, -21, -29, -16, -14, 0,   6,   7,   -15, 12,  -16, 3,   8,   -8,  -5,  23,  -13, 3,   1,   -23, 19,  -2,  2,   11,  -7,  -7,  -5,  -1,
    -1,  10,  13,  -4,  -24, 6,   -2,  -8,  -9,  -31, -7,  -29, -10, 16,  25,  0,   -6,  14,  -19, 26,  11,  9,   1,   15,  -6,  3,   0,   -8,  2,   10,  17,
    14,  5,   -24, -17, -16, -7,  -13, -30, -22, 13,  -45, -1,  15,  17,  -36, 2,   -3,  -8,  -8,  -8,  37,  -8,  0,   44,  -31, -12, -28, 3,   5,   4,   -34,
    13,  -6,  -9,  15,  2,   6,   -23, -38, 14,  7,   -2,  7,   -8,  -10, -30, -6,  -8,  17,  -12, -16, -1,  -16, -11, -17, -1,  12,  -11, -13, -17, -50, 6,
    5,   -16, -8,  23,  -9,  -4,  12,  -23, 24,  -7,  -10, -16, 20,  17,  9,   2,   -2,  -3,  -12, 10,  -10, 1,   11,  -6,  0,   17,  -18, -22, -3,  8,   -5,
    8,   -6,  20,  11,  -4,  4,   -5,  -12, -15, 7,   8,   -11, 10,  -8,  1,   -12, 14,  16,  11,  8,   0,   -6,  -34, -3,  -7,  -2,  -3,  -17, -1,  -7,  7,
    6,   -4,  4,   17,  16,  9,   10,  -1,  -9,  13,  7,   1,   15,  -1,  -12, -9,  1,   -21, 12,  5,   -3,  -8,  -8,  1,   -11, 9,   3,   32,  21,  -8,  9,
    -26, -14, -1,  4,   -5,  6,   -19, 16,  -5,  -1,  -1,  12,  10,  -10, 18,  6,   13,  -16, -7,  9,   10,  24,  5,   -2,  -4,  -7,  -14, 15,  3,   -2,  -16,
    -1,  20,  1,   6,   9,   10,  8,   5,   3,   7,   -12, 8,   -6,  -1,  15,  6,   12,  13,  -1,  3,   19,  -24, 20,  5,   0,   -7,  4,   -14, 4,   -2,  -2,
    4,   16,  4,   15,  -6,  -16, 12,  -12, -13, 7,   11,  14,  5,   6,   -11, 9,   -11, 10,  16,  4,   -19, -8,  5,   -1,  -3,  13,  6,   -6,  -8,  -10, 15,
    -5,  11,  -3,  9,   1,   11,  -9,  5,   -3,  -7,  -4,  10,  8,   -7,  7,   9,   3,   13,  -3,  -3,  11,  -6,  1,   23,  6,   9,   -9,  -24, -1,  3,   8,
    -5,  2,   4,   5,   0,   23,  8,   7,   -11, -12, 16,  -10, 22,  -22, 1,   -1,  -2,  -7,  4,   -20, 7,   9,   -1,  -7,  -12, -5,  -19, 4,   12,  6,   14,
    0,   -8,  10,  11,  -3,  -1,  15,  1,   -24, -2,  6,   -10, 5,   6,   19,  -2,  6,   11,  7,   22,  -17, 6,   -20, -6,  -12, -5,  14,  -10, -7,  0,   6,
    13,  16,  -7,  9,   6,   -21, -15, -13, 11,  -3,  -21, 18,  16,  14,  10,  10,  0,   -12, 5,   -14, -37, -2,  12,  8,   1,   -8,  -16, 3,   -1,  3,   -31,
    13,  -5,  1,   -25, -2,  -7,  -4,  2,   -1,  11,  -7,  -6,  -30, 27,  4,   -24, -37, 19,  18,  -11, -10, -18, 6,   -25, -2,  -25, 2,   -3,  4,   -10, 9,
    14,  10,  -4,  -7,  14,  15,  9,   -8,  -4,  -20, -27, 30,  -10, -19, -3,  -14, -2,  13,  -25, 18,  11,  14,  -26, -10, 3,   10,  -2,  6,   -9,  11,  35,
    -7,  10,  -4,  -38, 6,   16,  -4,  9,   0,   17,  -11, 32,  -9,  -5,  -52, 34,  -25, 16,  -9,  6,   12,  -13, 21,  -3,  30,  -10, 3,   20,  -6,  -15, 4,
    -17, 10,  -8,  2,   4,   23,  -11, 23,  10,  -2,  -11, 4,   -13, 7,   -25, 5,   -18, 14,  -2,  -13, 12,  19,  -7,  3,   11,  3,   -37, -33, 14,  8,   2,
    -1,  -15, -29, 5,   -50, -9,  9,   -9,  17,  -20, -18, -2,  12,  -5,  8,   -21, -12, 0,   -4,  -35, 5,   -1,  6,   35,  5,   14,  8,   -2,  5,   15,  -7,
    -26, 1,   -40, -20, -8,  -1,  -3,  11,  18,  2,   6,   14,  11,  -2,  23,  5,   12,  19,  -45, 9,   -25, -14, 3,   -8,  21,  -9,  0,   9,   17,  19,  9,
    11,  -9,  17,  -4,  0,   3,   -3,  -2,  -16, -10, -1,  -13, -4,  22,  6,   -5,  -26, 14,  -1,  -1,  -19, 15,  34,  -5,  -6,  -1,  -25, -12, 9,   -7,  0,
    2,   31,  15,  -6,  -8,  -19, -7,  -32, 7,   -20, -4,  -47, 1,   5,   17,  0,   -7,  -23, -15, 5,   6,   -26, 0,   6,   -12, -8,  -5,  20,  -4,  -5,  -5,
    -6,  -19, 20,  15,  6,   -9,  -12, -5,  -19, -4,  9,   1,   8,   18,  -9,  13,  7,   9,   10,  -20, -8,  9,   1,   15,  -3,  11,  8,   -1,  8,   6,   -2,
    13,  0,   5,   -2,  24,  13,  -24, -21, -19, -13, 7,   -11, -3,  -13, 2,   17,  4,   -18, 17,  12,  0,   -5,  7,   11,  -15, 0,   7,   0,   -9,  -7,  1,
    -8,  -13, 4,   30,  1,   -33, 0,   12,  11,  22,  -10, 10,  14,  8,   -11, 10,  11,  -2,  5,   12,  -18, 17,  -38, 0,   7,   -17, 12,  -14, 14,  -1,  -25,
    13,  1,   -25, -1,  16,  -2,  -4,  -20, 1,   0,   -12, 1,   27,  3,   -2,  7,   9,   -18, -19, 3,   -12, -4,  -24, 4,   12,  -3,  3,   -4,  12,  -7,  6,
    -9,  11,  3,   7,   -4,  4,   -15, 14,  6,   7,   15,  -4,  11,  5,   -1,  4,   14,  5,   -9,  2,   13,  4,   -22, -16, -4,  19,  -20, -22, 12,  6,   -4,
    4,   2,   -1,  17,  1,   -22, -5,  3,   0,   8,   -5,  27,  -14, 1,   -5,  16,  -6,  3,   12,  -36, 12,  9,   -12, -10, 6,   1,   8,   13,  6,   6,   -31,
    -2,  -11, -16, -17, -21, 4,   0,   1,   -15, 17,  4,   1,   -4,  -8,  -3,  7,   -4,  -9,  -2,  19,  4,   -1,  -9,  18,  9,   9,   8,   -9,  -1,  0,   -13,
    -17, -5,  -9,  -21, -10, 0,   2,   -1,  9,   -2,  -28, 3,   9,   3,   -23, 6,   -13, 13,  10,  5,   13,  8,   1,   -24, 8,   -7,  -24, 12,  -26, -7,  -17,
    9,   -4,  13,  -8,  -8,  44,  -12, 7,   -4,  32,  -7,  -27, -2,  -7,  -6,  6,   -14, -37, 24,  -8,  -6,  2,   -27, 20,  12,  10,  -1,  11,  14,  -43, -27,
    12,  11,  -12, -2,  -10, -13, 11,  -18, 18,  -13, 15,  -16, -14, -30, 1,   5,   -25, 15,  -15, -1,  1,   -11, 16,  -17, 19,  11,  7,   15,  14,  -21, -12,
    -19, -5,  -30, 13,  14,  19,  -7,  -3,  13,  4,   1,   12,  -15, 21,  14,  0,   11,  -32, 7,   21,  -10, 14,  1,   -14, -1,  22,  0,   -26, -5,  5,   12,
    -14, 7,   22,  28,  3,   0,   -4,  0,   -16, 17,  -17, -15, 10,  13,  -7,  6,   11,  -1,  8,   -1,  -21, 1,   18,  0,   11,  -6,  -11, -5,  14,  -2,  -40,
    -35, 13,  -26, 1,   14,  -30, -16, 23,  -26, 15,  1,   -6,  -30, -1,  21,  27,  6,   28,  -4,  -8,  11,  -2,  -6,  -9,  -28, -21, -1,  -14, -4,  5,   9,
    -23, -2,  -6,  -7,  13,  -6,  -4,  -24, 13,  -35, 12,  30,  -8,  14,  -13, 22,  8,   12,  -29, 8,   -5,  -16, -17, 16,  -7,  -11, -12, -2,  14,  -14, -16,
    7,   -1,  -51, 28,  7,   -25, -7,  -39, 14,  8,   6,   -6,  -6,  -8,  3,   -26, 12,  7,   9,   -7,  -6,  17,  1,   12,  25,  -2,  -20, -1,  11,  24,  -1,
    3,   0,   -4,  -3,  13,  12,  10,  -9,  -44, 6,   10,  8,   -37, 16,  10,  3,   -5,  5,   8,   16,  5,   17,  -9,  11,  -23, 9,   -2,  -1,  -6,  -6,  0,
    -15, -15, -8,  -14, -2,  -8,  -14, -14, -4,  30,  -11, 7,   -8,  -8,  11,  2,   -4,  9,   -5,  6,   3,   0,   3,   0,   -2,  3,   -12, -10, 22,  8,   14,
    9,   4,   -1,  14,  7,   -10, 12,  7,   -13, -1,  3,   -3,  4,   -8,  -15, 22,  14,  -10, 1,   -1,  -9,  6,   -7,  -11, 14,  7,   -11, 9,   -10, 5,   11,
    7,   8,   8,   -7,  -14, 4,   -10, 13,  -2,  2,   15,  4,   10,  2,   0,   -10, 7,   -2,  6,   1,   5,   -5,  2,   11,  4,   6,   -1,  -5,  16,  -13, 8,
    5,   -16, 10,  -5,  -1,  -20, 26,  2,   -3,  4,   -3,  4,   18,  9,   -6,  -27, 0,   -3,  4,   3,   14,  0,   -23, 1,   -10, -27, -3,  17,  -9,  -11, 16,
    7,   2,   -20, 19,  6,   -6,  11,  -22, 28,  -2,  1,   26,  -18, 4,   1,   8,   -11, 10,  13,  -21, 15,  -10, 6,   -13, 15,  -7,  18,  0,   3,   22,  -62,
    -8,  -13, 27,  9,   12,  -21, -15, -3,  -1,  1,   -8,  7,   3,   9,   -36, -19, -27, 5,   -8,  -18, -3,  -63, 3,   1,   2,   -1,  15,  -13, 28,  -31, 26,
    1,   14,  -14, -20, -4,  7,   -8,  -4,  -13, -39, 12,  -28, -10, 30,  2,   -23, -8,  10,  -13, -9,  15,  -16, -5,  -2,  -1,  1,   -29, 11,  -4,  -4,  8,
    28,  24,  -37, -27, 33,  -49, -6,  7,   -28, 16,  7,   3,   -8,  5,   -7,  -3,  -5,  -5,  3,   -3,  -4,  13,  6,   -7,  -29, -13, 3,   7,   6,   25,  -3,
    6,   16,  -9,  16,  0,   14,  5,   8,   -3,  10,  -13, -2,  -19, 15,  1,   -6,  -5,  17,  -15, 14,  11,  0,   0,   -4,  18,  7,   11,  -7,  6,   -34, -11,
    12,  10,  13,  -13, -11, -4,  20,  -21, -11, 3,   -11, -19, 0,   -15, -1,  13,  23,  6,   -3,  -4,  12,  -4,  15,  -32, -12, 2,   5,   -41, 4,   12,  3,
    13,  2,   16,  -6,  -2,  -16, -8,  -25, -19, 24,  9,   -7,  -8,  17,  6,   -8,  -1,  -17, -1,  15,  5,   -21, -37, -21, -2,  -3,  17,  2,   -22, -2,  5,
    -12, 0,   7,   -12, 0,   11,  10,  -5,  6,   15,  6,   -17, -9,  7,   -10, -9,  14,  30,  -10, 1,   3,   -6,  -16, 0,   7,   15,  10,  -15, -21, 5,   -11,
    4,   -21, -14, -8,  23,  15,  23,  -2,  -5,  7,   1,   2,   -10, -17, -7,  -4,  -12, 9,   -19, -16, -9,  26,  -22, -6,  -3,  -14, 1,   14,  13,  5,   -14,
    -11, 9,   24,  -26, 16,  -26, -12, -24, -9,  14,  -28, -18, 1,   -13, 6,   2,   7,   -12, 13,  -55, -2,  -1,  -7,  2,   15,  -14, -14, -6,  1,   30,  -3,
    -5,  2,   13,  -25, 8,   1,   0,   4,   -3,  -17, 31,  7,   8,   12,  -2,  -23, 15,  -18, -15, -11, -1,  1,   23,  -5,  7,   -9,  -2,  7,   -2,  -6,  -2,
    -21, 22,  -17, -6,  4,   11,  16,  -7,  9,   -1,  -10, 1,   -9,  2,   -5,  1,   18,  -25, -4,  -14, -6,  5,   25,  -14, 0,   -8,  -17, -6,  17,  3,   3,
    8,   -8,  19,  22,  15,  -20, 3,   -49, 8,   24,  3,   11,  -1,  -27, 2,   27,  -13, 5,   4,   9,   -1,  -45, 12,  2,   10,  15,  2,   -1,  -22, -3,  -2,
    -13, -5,  1,   0,   1,   -2,  20,  2,   7,   20,  11,  21,  22,  -5,  -32, 3,   2,   5,   -24, 6,   -7,  -23, -12, -2,  -4,  6,   21,  -9,  -9,  -9,  -11,
    3,   38,  -18, -3,  18,  -8,  -36, -6,  19,  -11, -10, 1,   12,  -12, -9,  -5,  -11, 3,   -1,  20,  -22, 8,   -14, 20,  -3,  -12, 24,  -3,  -9,  -11, -1,
    -7,  4,   14,  -17, -25, 8,   -10, -37, 11,  6,   1,   11,  -18, -11, -18, -1,  2,   -2,  15,  9,   -8,  -10, -2,  3,   17,  18,  -16, 2,   -7,  22,  6,
    8,   5,   12,  10,  -25, -3,  -19, -2,  19,  -21, 8,   -2,  -19, 11,  -24, 13,  16,  -29, -7,  22,  9,   -10, 1,   -16, 4,   -8,  -29, -9,  18,  -30, -9,
    11,  20,  -10, 14,  16,  -9,  30,  1,   11,  -7,  -5,  10,  -16, -10, -11, -11, -10, 7,   7,   -3,  -3,  -27, -11, -9,  -6,  -12, -2,  -7,  -15, 7,   3,
    2,   2,   -5,  0,   -5,  -17, 10,  23,  21,  -3,  7,   -11, -36, -11, 23,  2,   21,  -3,  -26, 3,   -12, -5,  -5,  -12, 8,   -5,  -17, -6,  -18, -6,  4,
    10,  2,   -7,  0,   9,   -15, -6,  -18, 4,   -8,  -3,  -7,  18,  7,   29,  4,   -16, 21,  -11, -8,  -12, -10, 14,  14,  10,  4,   25,  12,  4,   12,  4,
    0,   -9,  -7,  5,   -12, 15,  6,   -12, 0,   9,   -2,  10,  -27, -19, -7,  -9,  1,   7,   -15, -3,  -1,  6,   8,   2,   -22, -20, 23,  5,   1,   -4,  -11,
    0,   0,   -56, -21, -12, -10, 5,   -5,  3,   23,  22,  -33, -15, -10, 25,  -7,  -14, -1,  -5,  -8,  6,   -25, -17, 6,   10,  15,  0,   -35, 12,  -11, -12,
    -2,  -6,  -2,  0,   18,  20,  20,  -14, -54, -8,  11,  -2,  11,  6,   11,  1,   -9,  8,   -31, -8,  0,   18,  -19, -1,  23,  -5,  9,   -26, -14, -66, 7,
    10,  -47, -26, 5,   -7,  7,   8,   -3,  7,   -1,  -22, -14, -5,  12,  11,  3,   -24, -18, 26,  -1,  4,   -23, 6,   -42, -28, -2,  9,   -15, -5,  -18, 13,
    13,  -8,  -1,  2,   11,  -8,  -10, -7,  -12, -10, 18,  0,   2,   3,   13,  0,   -1,  -2,  -11, 0,   16,  7,   26,  2,   17,  19,  15,  -17, -92, -19, -22,
    22,  23,  2,   -6,  17,  8,   2,   13,  -4,  -4,  46,  -5,  18,  -11, -14, -24, -1,  8,   23,  -6,  4,   -20, 17,  -15, 3,   4,   -7,  21,  11,  3,   -6,
    -23, -6,  7,   10,  -1,  -20, -15, -7,  -4,  -20, 6,   19,  16,  11,  0,   -8,  7,   -4,  0,   4,   -2,  -2,  7,   30,  2,   -10, 28,  -26, -5,  -23, 6,
    19,  -27, -8,  -14, 2,   13,  2,   15,  14,  -39, -8,  -1,  5,   -23, -10, -1,  7,   -26, 20,  2,   -15, 20,  12,  33,  -31, 11,  -7,  -17, 3,   27,  -17,
    34,  -26, -11, -8,  -5,  5,   -12, -18, -14, 4,   17,  8,   -5,  9,   12,  -17, -17, 2,   14,  -18, -17, 7,   18,  -35, 3,   18,  -39, -8,  -11, -20, -3,
    11,  -6,  -1,  14,  -4,  8,   9,   9,   25,  -12, -2,  -7,  -2,  1,   -26, 27,  17,  -9,  -1,  -6,  -24, -22, 14,  -3,  8,   12,  10,  -14, -10, -4,  16,
    -17, 15,  -9,  -13, -1,  -11, 2,   -26, -9,  3,   -3,  -31, -4,  21,  -2,  29,  -28, -48, 30,  -16, -5,  -5,  -4,  14,  -14, 6,   2,   8,   -25, 6,   10,
    8,   6,   0,   11,  11,  22,  -9,  -17, -1,  5,   0,   0,   -12, 0,   17,  5,   16,  2,   -10, 12,  11,  -7,  -3,  -22, -28, 1,   -17, 11,  -13, -17, -1,
    11,  15,  3,   -5,  9,   -5,  -4,  -3,  13,  18,  -2,  7,   9,   10,  0,   -27, 21,  -5,  -7,  -14, 18,  -15, -10, 10,  9,   1,   3,   11,  -18, 20,  10,
    -3,  -7,  -4,  -37, -13, -24, -11, -5,  12,  -1,  25,  13,  -2,  -23, -9,  -22, 0,   -12, 13,  -16, 5,   -4,  7,   -19, 9,   -18, -10, -11, 15,  -7,  17,
    -48, 1,   -29, -9,  12,  -8,  -3,  -36, -16, -7,  12,  -22, 2,   -4,  2,   21,  4,   -7,  -2,  -10, -4,  -4,  4,   -22, 10,  -6,  9,   3,   19,  13,  -11,
    -4,  -3,  -16, 0,   3,   4,   2,   -15, -13, 5,   -10, -8,  -4,  -10, 8,   23,  4,   8,   20,  4,   17,  18,  -25, 10,  6,   19,  -1,  -7,  16,  -7,  17,
    -10, 9,   9,   -37, -22, -47, -10, 12,  2,   -13, -6,  -29, -9,  28,  -12, 10,  -7,  2,   -13, -11, -2,  -7,  21,  -3,  -14, 10,  10,  5,   3,   5,   1,
    -63, -8,  9,   -36, 0,   5,   2,   6,   -15, -5,  -2,  9,   18,  14,  -14, 2,   22,  7,   6,   12,  -34, 8,   8,   -2,  4,   14,  -12, -3,  14,  -9,  1,
    -14, -1,  -12, -8,  -10, -5,  13,  -29, 6,   27,  -24, 12,  -3,  -22, 2,   -8,  1,   -20, -32, 5,   15,  21,  -3,  18,  5,   20,  -11, 4,   16,  -31, -40,
    -17, -11, 4,   31,  -11, 42,  20,  -13, 1,   -26, -8,  -28, -16, -1,  -14, -11, -6,  11,  9,   10,  -28, -7,  20,  22,  -2,  -55, -10, -15, -18, 18,  7,
    8,   1,   -4,  -5,  5,   -23, 12,  -4,  -1,  6,   1,   23,  1,   11,  16,  31,  16,  16,  21,  5,   -14, 4,   -12, 17,  -10, -4,  -11, -30, -13, 2,   -9,
    6,   -2,  -10, -4,  -2,  8,   -1,  -12, 4,   -6,  -21, -5,  -4,  -3,  -19, 18,  -4,  -9,  -19, -6,  -14, 1,   5,   -19, 13,  12,  -7,  9,   -10, 10,  -25,
    11,  4,   15,  21,  4,   4,   -9,  24,  -30, 13,  -16, -8,  2,   -19, 6,   7,   -7,  -6,  -5,  20,  -2,  -1,  7,   17,  -12, -4,  -8,  -23, 13,  -5,  -5,
    -16, -18, 2,   9,   3,   4,   8,   -33, 19,  -12, 16,  7,   -5,  3,   -3,  9,   4,   -12, -19, 8,   3,   -19, -2,  8,   -1,  -8,  -5,  17,  -24, -27, 9,
    -3,  11,  6,   1,   -5,  -4,  32,  -9,  11,  13,  16,  -10, -10, -4,  12,  12,  17,  11,  -2,  -21, -16, -17, -1,  7,   2,   -9,  2,   -4,  8,   6,   9,
    8,   8,   6,   -2,  3,   12,  -4,  24,  -15, 10,  34,  22,  -9,  26,  9,   -16, 11,  -4,  -3,  11,  -7,  -4,  -22, 9,   -3,  -24, 0,   9,   14,  10,  0,
    -7,  -6,  7,   -25, 0,   11,  3,   15,  -14, -6,  -3,  15,  -10, -9,  -16, -8,  1,   -13, 7,   -13, -5,  3,   3,   26,  12,  15,  20,  3,   -19, -7,  0,
    -13, 4,   -4,  -6,  -10, 11,  12,  5,   -14, 9,   0,   -33, 4,   23,  11,  15,  10,  -5,  -16, 5,   16,  1,   -14, 12,  -13, 9,   21,  10,  -3,  2,   12,
    -2,  -1,  2,   -21, 5,   -13, 9,   8,   -15, -1,  -8,  -3,  7,   13,  -12, -4,  24,  8,   1,   -4,  9,   1,   -9,  5,   -5,  8,   0,   4,   -21, 12,  -13,
    1,   40,  1,   -26, 11,  8,   -4,  -13, 3,   12,  -21, -2,  -16, 4,   -10, -11, -7,  7,   3,   -8,  -8,  -21, -5,  -1,  16,  -3,  -16, 13,  -25, 2,   15,
    -6,  9,   -2,  0,   0,   6,   2,   -21, 2,   -13, -2,  -5,  1,   -11, 7,   2,   -4,  14,  14,  -8,  10,  -33, -3,  21,  -4,  -23, 6,   17,  15,  -1,  -20,
    -12, -6,  -7,  -34, 25,  -13, -11, 9,   1,   -2,  -17, 21,  4,   -10, -18, 0,   -2,  20,  -14, 7,   1,   -4,  -9,  22,  -27, -5,  7,   0,   -6,  3,   16,
    14,  -20, -9,  -5,  7,   -3,  34,  -5,  23,  -2,  -28, 16,  -6,  14,  -12, 3,   -31, 2,   -5,  3,   8,   14,  1,   3,   -8,  -14, -10, -10, 14,  18,  -19,
    -3,  -6,  -11, -8,  -6,  -18, 2,   15,  10,  -6,  6,   4,   9,   12,  -15, -10, 13,  -7,  18,  3,   -5,  16,  4,   21,  -5,  4,   -14, 9,   9,   19,  -4,
    -6,  7,   -1,  8,   -10, -3,  -16, 3,   0,   -11, -7,  8,   -16, 16,  -11, 18,  4,   10,  -7,  17,  5,   12,  14,  -9,  3,   -11, 2,   2,   0,   -23, 11,
    18,  8,   -10, 11,  9,   10,  -9,  -16, -22, 15,  -2,  8,   6,   23,  -11, -14, -15, 10,  11,  -2,  -26, -1,  17,  -29, 2,   -3,  -22, 5,   5,   -6,  2,
    -20, 18,  24,  -2,  20,  -10, -17, 7,   -6,  1,   1,   19,  -9,  -19, -14, -18, 13,  18,  -21, -12, -23, -13, 12,  -13, -8,  10,  23,  8,   -22, -2,  -14,
    2,   26,  -1,  -4,  -8,  12,  -3,  -5,  -9,  -27, -23, 1,   -3,  -2,  -28, -4,  -17, -3,  -1,  11,  10,  1,   20,  -7,  4,   -14, -4,  7,   -8,  8,   -17,
    4,   7,   -11, -38, -11, -14, 10,  14,  6,   -9,  3,   11,  20,  2,   2,   -16, -1,  -6,  15,  0,   4,   6,   6,   14,  0,   0,   -6,  13,  5,   10,  1,
    2,   -11, 2,   -3,  13,  10,  8,   -2,  -24, -2,  -3,  -8,  -7,  13,  -6,  4,   -16, 5,   14,  5,   6,   -20, 9,   -9,  14,  -10, -11, 22,  -5,  -13, -11,
    12,  13,  6,   -6,  1,   10,  14,  -3,  -12, -2,  4,   -23, -2,  -24, -8,  -3,  4,   5,   -5,  -4,  2,   9,   17,  9,   13,  6,   21,  -9,  -14, 8,   7,
    -8,  17,  10,  7,   3,   8,   -9,  19,  -7,  -10, 9,   0,   4,   -6,  17,  2,   11,  11,  9,   11,  -14, -1,  -4,  8,   -22, 4,   9,   12,  -17, 20,  1,
    -18, 4,   -16, -1,  27,  -9,  3,   7,   14,  -13, -17, 1,   17,  -10, -14, 13,  23,  -17, 1,   4,   3,   3,   -12, -9,  -24, -2,  11,  13,  20,  9,   -3,
    -17, -4,  6,   -31, -5,  -19, -25, 0,   8,   2,   1,   4,   1,   12,  7,   -30, 6,   9,   1,   9,   -17, 7,   -6,  8,   10,  1,   2,   5,   -1,  -6,  -10,
    1,   -6,  0,   9,   -11, -15, 8,   -5,  -10, -2,  -23, -6,  15,  -13, -20, -15, -10, -4,  -3,  7,   0,   -2,  2,   -16, 5,   18,  -7,  9,   11,  -20, -14,
    3,   6,   1,   14,  11,  4,   1,   7,   -19, 2,   13,  -16, 7,   -20, -5,  8,   -1,  4,   -2,  13,  -4,  1,   0,   -1,  0,   7,   13,  14,  -3,  23,  -7,
    14,  16,  -1,  14,  14,  6,   1,   -5,  15,  -8,  -5,  -11, -27, 11,  -19, -38, 25,  -7,  -16, 0,   -15, -2,  21,  15,  24,  12,  1,   -3,  6,   -14, -10,
    -2,  0,   22,  -14, -8,  28,  10,  14,  12,  -1,  -6,  5,   -1,  7,   25,  -6,  18,  -31, -1,  5,   30,  -12, 15,  -8,  6,   -39, -9,  -8,  -5,  -6,  10,
    -8,  -17, 3,   -11, -8,  -15, -17, 16,  17,  4,   13,  -7,  20,  13,  11,  -5,  15,  -63, -27, -9,  9,   -3,  18,  -13, 0,   -9,  6,   -19, -1,  -19, 3,
    -14, -23, -19, 11,  -13, -24, -17, 20,  17,  24,  -1,  27,  -25, -5,  5,   -9,  -18, 8,   6,   3,   -5,  -1,  -20, 8,   7,   3,   -18, 12,  16,  10,  -2,
    -56, -26, 5,   17,  7,   20,  -29, -6,  -9,  -5,  -27, -3,  -11, 9,   4,   -4,  15,  -1,  9,   -26, -8,  8,   -7,  -9,  22,  2,   -19, -4,  13,  -12, 10,
    -21, -3,  -18, 8,   0,   24,  1,   6,   19,  5,   12,  0,   -20, -2,  -5,  -3,  -14, -1,  11,  9,   21,  -27, -15, -1,  20,  -18, -12, 14,  10,  1,   -5,
    -5,  10,  12,  -2,  13,  -2,  -8,  -8,  -7,  -16, 8,   -1,  4,   5,   5,   -24, 13,  12,  -4,  -20, -3,  -25, 12,  -17, -4,  1,   23,  -12, 11,  -13, -9,
    8,   6,   7,   -3,  4,   -18, -18, -29, -39, -32, -1,  24,  1,   19,  21,  -17, -7,  12,  -8,  -4,  6,   16,  18,  0,   -2,  -15, 2,   -4,  0,   -20, -28,
    15,  -9,  -9,  19,  -7,  -71, -6,  2,   15,  0,   -2,  15,  -2,  -14, 16,  4,   -9,  9,   -12, -5,  -27, 1,   25,  -29, -1,  -24, -10, -1,  10,  3,   -7,
    -30, -13, 1,   14,  -16, 16,  -10, -8,  4,   -44, 1,   -5,  16,  12,  15,  -17, -1,  -9,  5,   -3,  -20, -36, -6,  28,  5,   24,  -17, 16,  -11, -15, -4,
    27,  13,  -1,  39,  8,   15,  -16, -34, -4,  2,   7,   8,   28,  -5,  -30, -28, -22, 0,   -8,  11,  -13, -10, 18,  -5,  -10, -33, -12, -13, 14,  -20, -10,
    -2,  30,  -42, -29, -6,  15,  -5,  6,   -12, -10, -21, -2,  13,  -1,  -27, 42,  -18, -13, -28, -25, 7,   -2,  -9,  -3,  18,  -27, 13,  -33, -15, 10,  12,
    16,  -14, 4,   15,  -23, 12,  -32, 7,   -4,  -23, -14, 3,   34,  5,   3,   23,  -8,  8,   -9,  10,  -27, 16,  23,  -53, 3,   -37, -12, 1,   3,   2,   15,
    9,   -5,  -10, 6,   -2,  20,  -25, -6,  -11, 9,   -10, -23, 2,   8,   -1,  12,  1,   3,   -11, 9,   -7,  -21, -4,  9,   3,   15,  -23, 19,  -17, 2,   21,
    -24, -17, 0,   -22, -20, -21, 8,   -2,  7,   14,  9,   -16, 9,   -36, -14, -4,  9,   -6,  -19, 0,   -23, 6,   18,  -18, 8,   10,  21,  -11, 5,   -23, -1,
    29,  -45, 10,  14,  -4,  15,  10,  -20, 6,   -10, 11,  1,   -40, -4,  -4,  -8,  -10, 10,  12,  -1,  -13, -44, -69, 37,  6,   0,   3,   -2,  -2,  -12, -10,
    11,  22,  -4,  -28, 14,  -6,  23,  38,  -31, -18, 4,   26,  -2,  -8,  30,  16,  12,  13,  -8,  16,  -27, 9,   30,  -1,  8,   -2,  -21, 16,  -16, 2,   14,
    4,   -16, 10,  -6,  14,  -6,  -17, -16, 1,   -39, 8,   -12, 10,  11,  -3,  9,   9,   4,   7,   -4,  2,   1,   -12, 9,   -9,  10,  11,  -7,  -32, 6,   8,
    -3,  -11, 14,  -6,  6,   12,  -10, 16,  20,  -2,  -16, 9,   13,  2,   12,  -5,  16,  11,  -22, 1,   13,  -9,  -3,  0,   6,   12,  -10, -13, 10,  5,   -3,
    21,  0,   0,   -8,  1,   12,  -20, -9,  6,   -34, -8,  6,   -22, 29,  12,  18,  20,  22,  26,  -14, 0,   -8,  7,   20,  -20, 23,  -3,  -3,  5,   2,   9,
    -3,  12,  27,  8,   19,  -4,  7,   27,  -16, 1,   4,   15,  -13, 19,  -7,  -20, 13,  -9,  5,   -8,  -12, 3,   6,   -6,  -1,  17,  -13, 0,   7,   -22, -21,
    -7,  -8,  -3,  -1,  -2,  20,  10,  18,  -3,  4,   -6,  -7,  5,   -23, 16,  5,   -16, -8,  10,  7,   2,   -21, 24,  -10, -4,  -15, 17,  10,  -23, -13, -9,
    -4,  12,  -4,  5,   0,   12,  -22, -11, 8,   2,   9,   -23, 6,   -14, 5,   3,   -11, 5,   13,  -2,  -5,  -2,  -3,  13,  -12, 3,   25,  -31, 3,   -12, -15,
    23,  26,  -2,  11,  -29, -5,  -1,  -11, 2,   6,   -12, 2,   1,   4,   8,   13,  -6,  10,  -8,  30,  2,   1,   19,  2,   16,  -14, -22, -14, 13,  0,   10,
    8,   13,  -22, -4,  17,  -3,  5,   -13, 7,   -16, 5,   15,  -1,  5,   -22, 9,   -3,  6,   3,   -4,  4,   -16, 6,   7,   0,   8,   0,   25,  14,  38,  11,
    -4,  -1,  -6,  -7,  -39, -33, 11,  30,  7,   -10, 7,   13,  -6,  4,   -6,  4,   6,   -1,  6,   0,   -17, -6,  7,   17,  12,  14,  -31, 7,   0,   0,   -9,
    -28, -1,  19,  1,   -11, 15,  4,   2,   -7,  -22, 9,   -11, -9,  -12, -5,  -10, -32, 10,  -2,  1,   8,   8,   -13, -15, -8,  29,  -10, 4,   7,   9,   -5,
    2,   -20, 10,  4,   -7,  -33, 4,   20,  11,  -24, 2,   3,   -8,  -20, 15,  -8,  -12, 6,   14,  19,  20,  16,  11,  27,  27,  -3,  14,  -9,  12,  -26, -5,
    -4,  6,   -4,  3,   -7,  -4,  20,  21,  18,  10,  -8,  6,   1,   20,  -14, 15,  10,  -13, 4,   -6,  13,  12,  23,  -12, 7,   -19, -7,  9,   0,   18,  17,
    -21, -8,  29,  -14, -15, 10,  -13, 12,  -4,  18,  -14, 5,   25,  -8,  -30, -11, -2,  -9,  2,   0,   -4,  5,   -8,  2,   6,   13,  6,   13,  -12, 12,  6,
    0,   14,  0,   -15, 1,   2,   -12, -15, 3,   2,   2,   2,   -19, 2,   -19, 1,   4,   -3,  1,   13,  -3,  15,  33,  -12, -11, 13,  10,  3,   4,   -2,  -1,
    10,  -8,  1,   -2,  14,  15,  -21, -15, 6,   -11, -4,  26,  14,  3,   -2,  -4,  7,   -1,  -6,  -5,  -7,  -2,  -11, 6,   5,   -5,  24,  4,   11,  -21, -7,
    8,   -20, -11, -1,  -3,  -4,  10,  12,  4,   0,   -4,  9,   -9,  -14, -6,  2,   -11, 8,   7,   -14, -8,  -1,  -6,  -20, -2,  14,  4,   0,   -6,  -8,  5,
    14,  9,   13,  -2,  13,  9,   -1,  -10, -10, 10,  9,   -7,  -17, -51, 10,  -11, -6,  3,   -14, 5,   30,  -20, 22,  9,   -1,  -5,  8,   -21, 18,  -11, 9,
    -3,  4,   7,   -25, 16,  -12, 20,  -31, 21,  0,   -20, 4,   -3,  33,  -3,  4,   -20, -26, -21, 8,   -33, 7,   12,  -1,  -4,  -4,  33,  -14, -1,  8,   -33,
    -13, 35,  -2,  28,  -11, -7,  8,   7,   -16, 10,  -1,  0,   14,  -1,  6,   -1,  14,  1,   7,   -11, -17, 6,   -12, -29, -17, 2,   32,  12,  27,  -8,  14,
    -10, -10, 12,  5,   29,  12,  -9,  0,   -16, 21,  2,   35,  10,  -24, 26,  3,   17,  15,  -11, 7,   7,   10,  -24, -7,  2,   -39, 15,  2,   3,   11,  10,
    11,  17,  3,   40,  6,   -9,  -14, 3,   6,   -14, 5,   5,   -8,  5,   -33, -7,  -14, 14,  15,  -1,  -9,  -9,  4,   8,   -17, 20,  5,   -17, -21, -17, -10,
    -47, -8,  -7,  -14, 2,   1,   11,  2,   -23, 2,   -21, -8,  12,  -7,  -6,  4,   -24, 0,   2,   11,  9,   9,   -9,  28,  -2,  -14, -10, -12, 13,  1,   -35,
    13,  1,   -15, 1,   2,   20,  -20, 16,  -30, -2,  5,   9,   5,   3,   -23, 9,   -6,  -3,  8,   -4,  -11, -16, 6,   26,  17,  -6,  -22, 19,  4,   13,  -7,
    -16, 2,   -27, -8,  0,   7,   -17, 1,   -3,  -19, -26, 13,  -16, -7,  8,   1,   -2,  -23, 15,  11,  20,  4,   -3,  25,  20,  -9,  -19, 15,  11,  9,   5,
    11,  27,  19,  -20, 13,  -13, -34, -5,  -18, 43,  -8,  7,   27,  -20, 14,  -15, 9,   -19, 14,  14,  -5,  5,   -11, 16,  19,  -29, -3,  -19, -18, -1,  -5,
    11,  -34, 8,   -14, 15,  9,   -18, 8,   7,   1,   1,   20,  -1,  8,   -3,  11,  22,  -12, 18,  -3,  -49, 3,   5,   12,  -53, 2,   -5,  21,  -3,  -25, 4,
    -10, -7,  15,  -8,  -31, -11, -21, -4,  -1,  -7,  0,   -17, 22,  -28, 4,   -21, -42, -16, -19, -12, -1,  4,   10,  -10, 5,   -11, -12, -27, -9,  -1,  12,
    -3,  -6,  23,  -6,  5,   -12, 6,   -2,  -38, -51, -27, 5,   20,  7,   6,   -7,  20,  -4,  -20, 18,  4,   -14, 0,   -10, -10, -4,  -18, -8,  -10, -23, -6,
    17,  3,   24,  9,   -30, 6,   -21, 12,  19,  -14, -6,  -7,  8,   24,  8,   -14, -12, -2,  -42, -3,  19,  9,   17,  -17, -6,  -14, -11, 6,   11,  10,  4,
    14,  -11, 12,  4,   15,  15,  -26, 17,  -4,  -8,  -13, 5,   2,   7,   7,   3,   -2,  1,   -6,  1,   -10, -11, 2,   6,   -4,  8,   -10, -17, 13,  -10, 5,
    4,   8,   -12, 15,  2,   -31, -14, 6,   17,  -3,  8,   7,   -2,  2,   18,  -10, -5,  1,   -13, 2,   -25, -11, 4,   10,  18,  7,   9,   -6,  13,  17,  0,
    13,  -1,  -16, 7,   -2,  -3,  22,  6,   11,  -4,  -4,  13,  10,  -12, 7,   5,   -6,  13,  -8,  13,  -1,  9,   15,  1,   0,   -6,  18,  -21, 28,  0,   13,
    2,   3,   -12, 2,   21,  8,   -17, -12, 11,  6,   0,   -6,  -7,  22,  -14, 8,   5,   5,   -15, -21, 8,   3,   9,   5,   0,   2,   -3,  6,   -3,  4,   21,
    -3,  -9,  14,  -1,  -23, -19, -4,  2,   -12, 18,  5,   5,   -22, -16, -12, -13, -1,  -14, 5,   -9,  26,  0,   23,  -20, 15,  -8,  -15, 0,   15,  -9,  3,
    7,   11,  3,   -8,  9,   1,   18,  12,  -7,  9,   -14, 7,   7,   -15, 3,   -6,  11,  12,  10,  3,   -3,  12,  6,   -10, -6,  1,   -23, 17,  22,  -35, -8,
    -11, 13,  18,  -36, -14, 0,   -19, 1,   5,   8,   0,   0,   -9,  -26, -25, 20,  4,   -1,  -28, -10, 0,   -3,  8,   13,  11,  -8,  14,  11,  0,   -3,  -23,
    -3,  -26, 11,  7,   25,  0,   7,   17,  0,   12,  -10, -10, 10,  -1,  -3,  6,   19,  -12, -8,  29,  -26, 17,  -31, 4,   0,   -22, -8,  -2,  32,  6,   -17,
    15,  9,   14,  14,  4,   -4,  -4,  2,   -4,  6,   -21, 16,  -6,  34,  -10, -6,  -17, 16,  1,   -22, 20,  -1,  8,   -8,  2,   5,   12,  -7,  -8,  -10, 11,
    5,   -15, -6,  3,   -24, -16, 3,   0,   16,  6,   -13, 24,  13,  4,   -8,  31,  8,   9,   9,   -6,  -2,  28,  -6,  -4,  -5,  3,   7,   -3,  -12, -5,  -2,
    5,   18,  -34, 21,  -16, -20, -2,  -16, -3,  -8,  -1,  -8,  13,  5,   18,  -8,  -3,  -16, -7,  12,  3,   -13, 6,   -12, -11, -2,  -13, 16,  -15, 4,   -7,
    9,   -20, -7,  18,  13,  -8,  13,  -6,  -2,  11,  8,   -6,  -16, 14,  3,   -10, -8,  -3,  -17, -2,  -10, -5,  2,   -1,  -15, 18,  -9,  -5,  -17, 23,  2,
    16,  0,   -17, 18,  18,  2,   7,   1,   7,   -6,  -5,  -10, 8,   -7,  13,  0,   -3,  39,  -7,  -10, -4,  -2,  26,  1,   -8,  -7,  3,   17,  6,   -11, -29,
    5,   -5,  -10, 14,  5,   -13, -1,  -1,  13,  24,  -22, -1,  -13, 12,  3,   -34, 19,  8,   18,  -42, 17,  -25, 10,  18,  6,   13,  -6,  -12, -16, 4,   -6,
    21,  21,  1,   -1,  7,   -3,  -12, 22,  -31, -4,  -25, 13,  -1,  -18, -2,  -21, -17, 16,  7,   21,  2,   -3,  -29, 33,  -3,  -8,  -8,  34,  -33, -2,  9,
    -7,  5,   -2,  -19, -17, -12, 6,   -7,  8,   24,  10,  3,   7,   13,  -11, 13,  -6,  6,   15,  -9,  -5,  -11, 10,  -31, -11, -20, 4,   -18, -28, -8,  -25,
    31,  -6,  -17, 4,   4,   36,  -9,  -12, -34, 20,  3,   7,   -9,  -11, 16,  0,   -13, -3,  -17, 2,   -5,  1,   5,   9,   2,   7,   13,  3,   -3,  10,  21,
    8,   -14, 13,  -5,  -13, 12,  -25, 5,   4,   18,  3,   1,   8,   -12, 6,   -21, 3,   1,   -11, 13,  -3,  22,  3,   5,   -12, 1,   13,  17,  10,  12,  6,
    -15, -8,  -1,  7,   -4,  -21, -11, 18,  15,  2,   -7,  -12, 3,   7,   0,   0,   -34, -2,  -8,  -1,  -2,  3,   6,   -1,  -5,  16,  -1,  10,  -1,  3,   -4,
    -1,  10,  -35, -14, 12,  14,  12,  -29, -9,  -4,  16,  -11, -15, -9,  3,   8,   -22, -1,  5,   14,  3,   -5,  -12, -12, -3,  4,   -13, 15,  -15, -16, -38,
    0,   -6,  12,  -2,  2,   11,  1,   -6,  -22, -7,  11,  7,   -10, 5,   8,   13,  1,   2,   6,   8,   3,   13,  -3,  12,  -14, -11, -2,  -27, -5,  -9,  -13,
    7,   11,  10,  -8,  -10, -25, 4,   -2,  -14, 1,   -11, -1,  -15, 13,  -11, -13, -16, 19,  -21, -8,  -16, 1,   -27, 8,   19,  -1,  -8,  12,  14,  5,   -4,
    20,  -10, 14,  11,  2,   4,   0,   -9,  10,  8,   -10, -20, 3,   14,  -10, -3,  7,   5,   -10, 18,  9,   -17, -1,  13,  -14, -14, 18,  -2,  5,   20,  -22,
    18,  11,  -4,  10,  -10, -6,  11,  7,   -15, 9,   -3,  -14, 8,   -8,  -6,  7,   -21, -10, 6,   -12, 2,   -5,  0,   12,  16,  3,   -1,  -6,  24,  -3,  5,
    11,  -15, 9,   3,   15,  -13, 0,   -17, 2,   -5,  5,   23,  -19, 4,   14,  9,   21,  0,   9,   13,  22,  4,   9,   -4,  -13, -9,  5,   -41, 2,   7,   19,
    -48, -19, -39, -1,  -24, 7,   -1,  17,  16,  -14, -10, -6,  -12, 5,   -10, 1,   -14, -13, -1,  -11, -17, -14, -29, -14, -23, -42, -31, -21, 1,   -5,  -9,
    12,  18,  24,  11,  -1,  5,   15,  -5,  -3,  -22, 5,   -21, 27,  -7,  12,  30,  -3,  -39, 20,  -2,  10,  -6,  -29, -15, 3,   -10, -17, -34, -6,  25,  -38,
    -14, -2,  32,  15,  7,   -19, -24, -3,  -1,  8,   10,  -11, 23,  -31, 1,   -19, 13,  -9,  7,   6,   -4,  -10, -37, -12, -16, 9,   3,   -22, -23, -7,  -11,
    31,  -2,  -6,  16,  -17, 11,  7,   -13, -2,  -16, 22,  -1,  27,  11,  44,  -5,  -21, -13, 11,  2,   14,  11,  11,  9,   0,   -37, -5,  -2,  -5,  -14, 19,
    16,  8,   15,  9,   14,  8,   13,  -16, -1,  4,   9,   -9,  -7,  -16, -18, 4,   10,  -39, -16, -25, 17,  17,  -4,  -12, -10, -2,  -6,  2,   20,  -6,  1,
    -64, -2,  -17, 9,   8,   10,  -8,  7,   -7,  2,   -7,  10,  7,   9,   -4,  -15, 8,   8,   -3,  11,  -12, -4,  6,   -20, 18,  -2,  2,   -2,  7,   21,  10,
    48,  2,   -54, 6,   -1,  13,  4,   -13, 0,   -3,  1,   -10, 5,   -2,  -29, -49, 3,   3,   0,   -2,  26,  -23, -12, 7,   -90, -12, -8,  10,  3,   -5,  -33,
    -29, 10,  -21, -6,  -8,  9,   9,   12,  -34, 11,  -10, 22,  -4,  11,  -49, 19,  -8,  -10, 13,  -10, -51, 9,   -7,  1,   5,   -18, -2,  -2,  -10, -5,  6,
    8,   -14, 2,   3,   -16, -23, -1,  5,   8,   -33, -28, 9,   -7,  -2,  7,   -3,  -3,  17,  13,  -8,  19,  -6,  -7,  0,   -5,  -26, -12, 20,  19,  -12, -1,
    -20, -8,  -1,  5,   12,  -9,  1,   -6,  16,  -20, 11,  4,   22,  8,   9,   -6,  -1,  16,  -4,  8,   -9,  2,   -23, 2,   0,   -2,  1,   -9,  2,   -8,  -11,
    -12, -25, -6,  -13, 2,   7,   0,   4,   -3,  -12, -11, -10, -13, 6,   6,   -14, -8,  4,   -11, -12, 8,   18,  -6,  10,  -8,  14,  11,  -1,  -8,  -1,  12,
    4,   20,  4,   12,  9,   0,   28,  2,   13,  3,   -28, -10, 6,   20,  2,   -38, 22,  -15, 3,   -10, 12,  23,  19,  -1,  6,   -26, -11, 6,   4,   10,  20,
    -10, -6,  4,   0,   -11, -1,  5,   1,   23,  -6,  -18, -1,  15,  -12, -6,  -20, -16, 10,  3,   9,   -6,  -6,  3,   0,   6,   1,   -2,  0,   9,   15,  -10,
    -2,  4,   -16, -2,  -3,  13,  15,  -4,  2,   12,  12,  -4,  12,  14,  10,  5,   19,  -27, 2,   7,   19,  -18, 9,   12,  24,  -2,  -28, 3,   14,  15,  6,
    2,   13,  -8,  -1,  1,   10,  -22, -20, 21,  -18, 8,   13,  -25, 26,  -3,  6,   -4,  15,  -23, 3,   10,  4,   -5,  -16, -19, -2,  -1,  -4,  -7,  -12, -8,
    12,  10,  14,  -8,  -4,  5,   3,   7,   9,   -6,  9,   -32, 7,   24,  2,   5,   -7,  8,   12,  2,   -7,  10,  -10, 7,   8,   -3,  29,  2,   14,  1,   16,
    5,   -4,  8,   8,   -4,  5,   -3,  13,  6,   5,   4,   -5,  -3,  -6,  12,  19,  16,  9,   -1,  10,  4,   -8,  6,   -7,  6,   -14, -8,  9,   -7,  15,  -13,
    2,   -8,  14,  -9,  -9,  7,   -7,  6,   11,  20,  14,  -5,  2,   4,   -5,  0,   8,   5,   7,   -3,  -5,  0,   -4,  5,   9,   7,   -20, -4,  15,  -27, -12,
    -9,  -7,  6,   4,   -5,  9,   0,   -14, 12,  2,   -4,  5,   -1,  6,   13,  -1,  11,  -6,  12,  -11, -8,  -5,  -3,  -18, 1,   5,   5,   14,  -3,  -5,  -4,
    20,  7,   4,   17,  -2,  5,   15,  6,   4,   15,  -1,  -2,  26,  -6,  11,  8,   2,   3,   13,  3,   -3,  -3,  -4,  -2,  7,   -9,  8,   -16, 3,   9,   -6,
    -18, 4,   18,  9,   5,   -10, 3,   8,   5,   -3,  18,  -20, -10, -9,  6,   15,  19,  -4,  11,  13,  12,  -16, 15,  -16, 1,   14,  -10, 5,   16,  -20, -25,
    -1,  4,   -5,  -10, -11, -31, 5,   14,  -6,  7,   2,   -2,  7,   11,  -13, 4,   8,   2,   -16, -17, 4,   8,   -6,  -13, 3,   3,   0,   4,   2,   11,  -1,
    15,  21,  7,   8,   12,  -16, -11, -1,  -9,  7,   13,  -14, -7,  -10, 11,  -17, 10,  8,   0,   -7,  8,   3,   -17, -2,  -9,  3,   -3,  -4,  -13, 3,   0,
    11,  -9,  -4,  -29, -5,  14,  14,  -8,  18,  0,   -25, -13, -21, -13, -4,  14,  -3,  5,   -8,  7,   -18, -10, -12, -7,  17,  9,   5,   -3,  18,  -8,  -1,
    14,  10,  10,  -7,  4,   11,  16,  0,   -5,  0,   -13, 3,   -6,  -16, -43, 20,  6,   8,   5,   -8,  12,  0,   -25, -7,  29,  18,  11,  -19, 9,   -22, -10,
    4,   6,   -1,  -9,  5,   20,  -8,  -13, -15, 2,   -7,  -19, 18,  4,   13,  10,  -1,  5,   -1,  -13, -2,  2,   -13, 0,   -1,  17,  -9,  -7,  -3,  -43, 10,
    -26, -9,  12,  -5,  -4,  -7,  -5,  9,   5,   13,  18,  30,  5,   -11, 14,  -9,  -13, 11,  -21, -3,  -21, -41, -6,  -62, -29, -6,  12,  10,  1,   2,   -3,
    13,  -2,  1,   -17, 15,  -21, 14,  -2,  -10, 12,  -16, -10, -7,  10,  17,  2,   17,  16,  6,   6,   4,   -6,  -15, 6,   10,  -15, -25, 9,   15,  11,  19,
    4,   5,   -16, -12, 25,  -9,  -9,  -6,  10,  0,   19,  26,  6,   3,   1,   -9,  3,   -4,  3,   2,   -60, -11, 23,  -15, -26, -14, -9,  0,   0,   -9,  13,
    6,   15,  -7,  -7,  -13, -20, -18, 3,   -5,  -1,  -20, -15, -8,  -5,  12,  -11, 10,  0,   -10, -2,  3,   -3,  16,  -37, -11, -4,  9,   -33, 8,   -14, -23,
    7,   -4,  -4,  -15, -4,  -15, -8,  -8,  15,  -12, -4,  -7,  14,  -1,  8,   -12, -11, -28, 3,   0,   20,  9,   0,   5,   -7,  -3,  -5,  -3,  -2,  24,  -10,
    -5,  -9,  -15, -26, 12,  1,   -6,  19,  20,  12,  -2,  -13, 4,   3,   -5,  14,  -5,  18,  7,   12,  9,   8,   -10, 20,  -19, -13, -14, -7,  18,  5,   22,
    -14, 5,   -18, 11,  4,   -19, 10,  17,  20,  -16, -7,  19,  -7,  12,  -5,  -4,  34,  20,  22,  18,  16,  16,  -7,  13,  9,   -3,  3,   -25, -9,  -23, -12,
    5,   6,   -1,  6,   -2,  -24, -2,  -13, -16, -3,  -17, -12, 11,  -6,  16,  -19, 7,   -15, -1,  26,  -6,  -29, 6,   10,  11,  26,  -5,  15,  -7,  -53, -12,
    -9,  -32, 39,  11,  7,   4,   14,  7,   -5,  -10, 0,   -13, -20, -34, -9,  -28, -4,  18,  4,   12,  12,  -1,  12,  20,  9,   5,   -3,  -3,  19,  19,  13,
    -22, -2,  2,   8,   -4,  -12, -38, -34, -36, -22, 5,   -7,  12,  -16, -14, 11,  -11, -46, 12,  15,  -23, 0,   13,  -14, 6,   9,   -8,  -1,  -6,  -4,  18,
    -11, -21, 5,   -24, -5,  4,   -5,  -25, 11,  12,  -2,  9,   0,   5,   -10, 11,  0,   30,  7,   -26, 26,  -13, 14,  10,  3,   -26, 6,   2,   -9,  5,   27,
    -21, 13,  11,  -18, 9,   -12, 10,  -4,  -5,  23,  6,   6,   37,  0,   6,   -7,  13,  -4,  -18, -12, -25, -21, 1,   26,  -8,  -5,  3,   -5,  -16, 30,  14,
    4,   -1,  -19, -15, -11, -3,  -19, 9,   -7,  -39, -2,  21,  11,  5,   -21, -11, 24,  -13, 16,  17,  1,   -21, 0,   -21, -14, 2,   14,  -9,  -5,  20,  -7,
    -2,  -32, 0,   1,   -14, 16,  0,   12,  -14, 0,   -9,  -22, 0,   -26, -12, -22, -31, 10,  -20, -1,  -11, -20, 17,  8,   -2,  0,   -3,  -4,  -1,  9,   15,
    -24, 38,  13,  -3,  -3,  4,   -33, -26, -12, 17,  -32, -5,  -22, -1,  37,  -17, 8,   -43, 4,   -29, -12, 0,   9,   10,  7,   4,   -24, -13, 15,  -8,  -32,
    -5,  -4,  26,  13,  -5,  -22, 0,   5,   -28, -33, -30, 1,   -18, 3,   8,   11,  13,  -41, -3,  -11, -17, 13,  4,   -11, 2,   17,  -9,  0,   -1,  9,   13,
    10,  -25, -2,  -12, 9,   -22, 1,   11,  2,   -8,  -19, -6,  32,  -20, 11,  15,  -10, 3,   19,  6,   5,   4,   17,  20,  1,   0,   14,  -9,  12,  4,   -14,
    -25, 5,   12,  -17, -25, 31,  18,  -16, 2,   -8,  15,  9,   -17, -2,  -14, 0,   -22, 1,   -2,  -21, -21, 15,  -26, -18, 14,  17,  -18, 0,   -18, 21,  10,
    15,  3,   -4,  -3,  -2,  5,   10,  2,   -9,  2,   -9,  -5,  9,   14,  8,   5,   -25, 3,   15,  -22, -22, 15,  -39, -41, 21,  -5,  -26, -17, 21,  13,  -8,
    -9,  14,  8,   -8,  17,  -2,  4,   20,  9,   1,   -15, 23,  -10, 19,  -10, 26,  22,  -7,  -10, -4,  27,  -15, -17, -8,  8,   -7,  -23, -16, -3,  5,   -13,
    -30, 6,   20,  7,   15,  -27, -1,  15,  -23, 21,  21,  4,   4,   12,  -9,  20,  6,   -18, 7,   -3,  31,  2,   4,   -18, 10,  15,  -4,  -9,  -31, 12,  -8,
    -19, 15,  -21, 1,   -13, 8,   -6,  16,  -13, -8,  4,   -6,  -6,  18,  -5,  -21, -2,  -11, 0,   15,  27,  3,   7,   -19, -2,  -6,  1,   -13, -33, -1,  13,
    -12, 14,  -10, 6,   16,  -22, -17, -23, -2,  15,  16,  -2,  -18, 15,  6,   -3,  17,  -14, 16,  39,  -23, -7,  16,  -6,  4,   1,   -3,  3,   0,   -7,  -24,
    24,  0,   15,  15,  22,  -4,  10,  -11, 16,  -17, -1,  13,  -2,  -8,  -3,  -25, -1,  13,  19,  15,  -9,  -3,  -6,  -19, -9,  -8,  -5,  28,  -16, 1,   7,
    6,   1,   -3,  -9,  5,   -1,  -3,  23,  9,   -9,  8,   10,  1,   6,   4,   1,   -10, 8,   6,   14,  1,   -3,  -7,  -21, -8,  14,  -17, -1,  -14, -4,  13,
    9,   -15, -6,  -20, 0,   0,   20,  1,   -15, 6,   -14, 4,   -3,  1,   6,   12,  -19, -4,  -1,  -8,  -4,  -20, 10,  3,   4,   -10, 13,  7,   2,   0,   0,
    -6,  -2,  1,   -23, 2,   -26, 14,  15,  18,  22,  7,   8,   23,  8,   3,   -33, 4,   -1,  5,   18,  17,  -3,  2,   -12, 9,   5,   -29, 4,   -4,  4,   -6,
    -12, 19,  -5,  -15, 17,  -26, -6,  0,   -15, -6,  -13, -1,  -17, 3,   -15, 6,   0,   3,   9,   9,   3,   -7,  -10, -10, -17, 21,  1,   29,  7,   -27, 27,
    -14, 8,   17,  -28, -35, -28, 8,   16,  -10, -4,  -40, 3,   -24, -8,  27,  -16, 9,   -9,  11,  -10, -19, 15,  6,   2,   1,   21,  -7,  6,   -18, -11, -20,
    4,   -23, -5,  16,  12,  6,   -17, -18, 15,  -2,  -22, -19, 4,   16,  1,   19,  -15, -4,  11,  -30, 3,   0,   23,  -12, -6,  -20, -23, 8,   -8,  4,   -13,
    -44, 11,  -38, 2,   22,  2,   0,   -3,  -36, 1,   -29, -3,  -13, -18, 5,   13,  -7,  2,   -19, 6,   12,  -4,  -2,  14,  8,   3,   16,  20,  20,  13,  -3,
    -3,  23,  -6,  -4,  0,   15,  -8,  0,   -2,  -5,  8,   -8,  -5,  -10, 10,  11,  9,   3,   1,   14,  16,  17,  7,   -11, -13, 15,  -16, -13, 8,   -1,  -8,
    6,   -6,  -12, 18,  -8,  20,  -8,  -11, -5,  -18, 10,  13,  11,  6,   8,   -8,  0,   20,  15,  -10, -18, 5,   -1,  -8,  -10, 7,   11,  10,  2,   12,  1,
    -2,  6,   -21, 0,   4,   -11, 13,  2,   -9,  -1,  4,   -23, -15, 12,  -15, 15,  1,   -13, 11,  -6,  -12, 6,   1,   14,  14,  -10, 13,  1,   15,  -13, 8,
    -4,  7,   13,  13,  14,  0,   -13, -8,  7,   -6,  -15, -12, 4,   1,   9,   9,   8,   1,   9,   -7,  0,   14,  24,  -4,  -13, 19,  2,   7,   4,   -19, 3,
    5,   5,   -4,  7,   12,  -16, 6,   -15, 25,  -11, -19, 11,  -7,  14,  -3,  -10, 13,  -35, -1,  6,   -17, -82, 16,  -27, -25, 8,   -9,  9,   -21, 11,  9,
    -12, 19,  -8,  -25, 11,  -17, 23,  -2,  -25, -88, 29,  5,   16,  -38, -25, -6,  -4,  9,   -9,  -19, 3,   13,  -2,  -39, -4,  -17, -1,  -6,  12,  48,  -4,
    -24, -8,  -7,  -12, 14,  -20, 24,  -19, -4,  -16, 5,   0,   12,  -34, -4,  8,   -3,  5,   -23, 20,  -2,  -16, 5,   11,  -2,  -14, -23, 10,  -6,  14,  -36,
    -37, -8,  -3,  4,   0,   -2,  -14, -39, 7,   -4,  12,  8,   9,   -17, 13,  7,   -3,  -37, 18,  -5,  -16, -4,  -11, -16, 20,  5,   -8,  -17, 2,   8,   8,
    4,   -22, 14,  14,  26,  -2,  6,   -17, -3,  -46, -15, 10,  0,   -5,  -14, -13, -28, -25, -28, -45, 29,  -22, 27,  14,  -4,  27,  14,  -48, -5,  -22, -27,
    -40, -26, -21, 5,   -32, -2,  -22, 9,   12,  7,   -19, -30, -14, -19, -15, -30, -4,  -23, 19,  -5,  9,   -48, -20, 10,  -6,  -34, 16,  -28, 2,   9,   -8,
    13,  -13, -38, 7,   -8,  -73, -53, -3,  30,  -8,  -3,  40,  -6,  11,  7,   -7,  -5,  5,   14,  -31, -7,  4,   11,  0,   2,   -45, 11,  -11, -13, -18, -18,
    -12, 35,  -3,  3,   9,   29,  -4,  13,  25,  3,   -15, -4,  -46, -11, -14, 35,  -1,  26,  -6,  -14, 0,   16,  -15, 0,   11,  2,   21,  -21, -71, 30,  21,
    15,  -6,  3,   -5,  -1,  -6,  -18, -9,  16,  -7,  30,  5,   1,   -16, -4,  -15, -31, -25, 16,  -9,  -3,  -15, -25, -26, -10, 23,  -7,  -10, 6,   -18, -16,
    -6,  23,  15,  -39, -4,  -6,  21,  22,  -9,  -4,  11,  -28, 18,  13,  -15, 0,   -6,  0,   18,  -10, -23, -7,  -5,  4,   -7,  -7,  5,   -7,  13,  3,   26,
    -28, -7,  15,  -11, 12,  -8,  16,  -17, -1,  -11, 8,   17,  16,  12,  11,  -13, -12, 18,  8,   -11, 3,   0,   11,  0,   -28, 28,  9,   7,   -14, 1,   -15,
    0,   -18, 1,   -10, 4,   4,   2,   -7,  4,   1,   -6,  -10, -1,  -11, -3,  13,  -7,  -30, -5,  -4,  10,  2,   3,   14,  4,   12,  10,  -5,  -8,  19,  -17,
    -5,  8,   26,  12,  19,  1,   -12, -27, -9,  -1,  4,   -1,  -6,  -9,  3,   8,   -1,  8,   -15, 2,   -1,  -2,  19,  11,  11,  0,   13,  -5,  -19, -10, -7,
    0,   9,   0,   -11, 11,  8,   10,  -15, 7,   1,   17,  29,  -1,  -17, -5,  0,   23,  14,  -20, 22,  -5,  22,  16,  -8,  6,   -23, -4,  15,  -10, 15,  16,
    0,   -4,  -14, 22,  0,   -14, 2,   -14, -17, 7,   4,   8,   14,  -3,  -4,  -7,  -1,  19,  -3,  23,  6,   3,   5,   -1,  -1,  0,   0,   -6,  -26, 8,   0,
    9,   -3,  18,  -8,  -9,  22,  -7,  14,  5,   -15, -18, 15,  -7,  -5,  4,   26,  9,   13,  15,  8,   -2,  -8,  7,   5,   -9,  6,   -8,  -18, -9,  5,   -5,
    10,  3,   4,   -8,  9,   10,  9,   1,   -11, -6,  -9,  4,   -9,  -5,  2,   8,   8,   -20, -15, 8,   -11, -18, -9,  3,   -21, 19,  -16, -4,  0,   8,   20,
    -4,  14,  -5,  1,   5,   7,   -10, -2,  3,   9,   20,  21,  -3,  -13, -5,  -15, -5,  -1,  -11, -11, -10, -1,  1,   -5,  9,   -2,  -5,  3,   12,  4,   -7,
    5,   9,   -10, -14, 11,  22,  5,   4,   10,  18,  2,   -15, 13,  13,  3,   -18, 2,   9,   7,   -20, -18, 20,  4,   -7,  4,   11,  -20, 16,  -2,  11,  0,
    -15, 1,   16,  -19, 12,  -17, 0,   5,   -7,  19,  -5,  -28, 3,   -2,  7,   4,   2,   -11, -8,  11,  17,  -3,  -7,  -14, 4,   11,  7,   6,   10,  13,  12,
    -15, 8,   -5,  -3,  11,  10,  5,   -18, 0,   0,   11,  16,  4,   -6,  -6,  13,  -15, -10, -9,  -9,  2,   -3,  -15, -10, -10, 9,   1,   3,   -3,  13,  7,
    -19, -7,  -10, 9,   18,  -1,  -11, -7,  -14, -5,  -6,  -3,  17,  -1,  -15, 1,   15,  9,   3,   0,   21,  7,   -16, -3,  -2,  13,  -5,  18,  -5,  -8,  -11,
    -9,  10,  22,  2,   5,   2,   11,  -9,  0,   -2,  -11, 21,  3,   1,   17,  2,   -2,  -8,  9,   -16, -6,  -15, 15,  9,   -7,  -11, 30,  -7,  -19, 5,   -10,
    -4,  7,   -3,  -16, 10,  -19, 10,  -1,  -14, 17,  -6,  -4,  -3,  5,   -7,  -3,  11,  14,  15,  -14, 13,  0,   -10, -7,  3,   -11, -6,  -6,  -37, -40, 24,
    13,  5,   -7,  17,  3,   -2,  14,  -11, -2,  -5,  8,   -12, -11, 8,   6,   -16, 0,   -2,  20,  10,  14,  -12, 4,   6,   9,   -29, -8,  2,   3,   6,   12,
    -31, -3,  11,  -13, -17, -8,  -23, 3,   12,  3,   3,   -20, 11,  -21, -4,  -5,  -19, 38,  -2,  -2,  -3,  16,  -9,  -4,  6,   -30, -8,  5,   8,   13,  4,
    -5,  5,   -16, 7,   9,   2,   9,   -34, 14,  23,  10,  11,  -6,  -20, 11,  5,   -13, 27,  -9,  10,  -4,  -3,  18,  4,   2,   -1,  -9,  -7,  17,  -11, -2,
    36,  -6,  16,  5,   -9,  24,  -14, 6,   -12, -19, 9,   5,   13,  -13, 4,   15,  -1,  -25, 8,   -14, 2,   -8,  5,   -16, 13,  -16, -15, 5,   7,   -1,  13,
    21,  -13, 11,  13,  -17, 10,  -5,  10,  23,  -1,  16,  0,   -8,  -33, -13, -3,  6,   -22, 12,  -2,  20,  16,  1,   -8,  -18, -31, 2,   18,  5,   -10, 16,
    -13, -6,  -26, -30, 35,  -7,  1,   7,   0,   1,   2,   -7,  -5,  0,   13,  -1,  -5,  12,  -3,  17,  7,   -9,  14,  8,   1,   -28, 18,  13,  19,  -3,  -7,
    16,  18,  18,  -7,  -9,  -1,  8,   -4,  4,   -3,  5,   19,  10,  -10, 6,   1,   -16, 18,  2,   -22, -7,  5,   15,  -6,  -4,  7,   11,  -13, 5,   -8,  -7,
    -14, 15,  10,  1,   9,   6,   12,  23,  -10, -6,  -12, -20, -21, -4,  -21, 0,   -5,  -5,  6,   -4,  -60, 3,   10,  22,  -15, 9,   10,  -5,  -23, -6,  20,
    16,  2,   10,  8,   0,   -4,  8,   -16, 24,  6,   -4,  21,  38,  7,   12,  -29, -11, 10,  -24, -5,  7,   -10, 12,  14,  -11, 4,   11,  11,  -2,  5,   -18,
    -42, 15,  -7,  14,  -21, -4,  -14, -4,  -8,  4,   10,  25,  -3,  1,   2,   5,   4,   -2,  11,  -2,  1,   -16, 4,   4,   -47, -22, 7,   15,  -15, 7,   34,
    11,  0,   -7,  -8,  -4,  -37, 11,  11,  0,   -3,  15,  10,  -6,  -1,  2,   2,   -4,  2,   13,  -5,  -1,  -22, 23,  3,   -6,  -24, 15,  6,   -4,  0,   2,
    5,   0,   -10, 10,  0,   -14, 28,  19,  2,   -32, -17, 17,  -4,  5,   5,   -16, -15, 0,   26,  -8,  25,  8,   7,   -1,  -7,  6,   -43, -3,  -21, -18, -7,
    -15, 16,  11,  -10, 1,   -6,  8,   16,  -11, -11, -9,  0,   -2,  37,  -6,  30,  2,   19,  -1,  -7,  -32, -4,  -8,  19,  7,   -10, -42, 4,   -30, -18, 2,
    26,  -7,  -2,  -18, -26, 13,  -10, 18,  5,   13,  -14, 20,  -20, 5,   5,   -7,  14,  6,   20,  10,  -9,  -1,  -6,  -4,  -3,  29,  -2,  2,   21,  13,  4,
    -5,  0,   -23, 7,   3,   13,  6,   -2,  3,   9,   -38, -21, 12,  13,  19,  -4,  -7,  -13, -10, 14,  -1,  -24, 0,   0,   -2,  -10, 9,   -39, 7,   3,   -16,
    -16, 1,   -3,  -33, 7,   -2,  7,   5,   0,   -11, 16,  -1,  -24, -23, -23, 3,   -3,  10,  26,  -6,  -36, -4,  -32, -4,  -22, -7,  -11, -9,  3,   -15, 9,
    -18, 4,   -36, 2,   2,   -7,  -23, 20,  3,   3,   29,  10,  9,   6,   -20, -14, -16, -11, -7,  12,  15,  -11, 14,  -34, 34,  -10, -2,  -14, 5,   -26, 20,
    12,  6,   0,   17,  0,   3,   -1,  24,  18,  21,  -12, -9,  1,   -38, -46, -10, 7,   -14, 2,   -7,  15,  11,  13,  7,   9,   -8,  -6,  0,   -18, 2,   -5,
    -19, -33, -3,  -9,  11,  13,  9,   -16, 12,  -9,  12,  0,   8,   9,   6,   7,   -2,  -9,  -2,  6,   16,  11,  15,  -4,  -4,  4,   -5,  24,  4,   8,   -13,
    -32, -8,  28,  -5,  -25, 12,  -4,  -8,  -12, -10, 14,  -6,  -17, -9,  4,   -39, -22, 11,  -3,  -22, 29,  7,   -8,  16,  2,   -19, 13,  -2,  5,   -7,  -14,
    0,   -9,  -18, -4,  -16, 9,   11,  -6,  27,  6,   6,   22,  -1,  0,   20,  -11, -13, 12,  8,   11,  18,  -4,  -12, 22,  0,   -13, 18,  4,   -4,  23,  -6,
    -6,  -5,  13,  -5,  6,   5,   -8,  -13, -10, -32, 11,  -23, -2,  -4,  1,   6,   18,  17,  11,  -16, 2,   -14, 2,   -12, 0,   -15, 2,   23,  16,  12,  6,
    1,   7,   24,  -8,  -27, -2,  -15, 16,  16,  -11, 12,  -2,  7,   -7,  5,   -12, -3,  2,   17,  -19, -9,  -13, 11,  1,   12,  -3,  -5,  -12, -9,  -3,  -2,
    8,   -12, 7,   7,   -3,  -1,  3,   12,  14,  16,  -7,  11,  -10, -12, 3,   -6,  22,  11,  -5,  2,   22,  -26, -10, 3,   10,  -26, -14, -13, 3,   -14, 4,
    9,   11,  5,   -16, -20, -12, 0,   -3,  -23, -4,  -3,  -2,  28,  10,  6,   11,  4,   -13, -10, -17, 7,   -6,  -1,  -7,  13,  36,  -3,  25,  -4,  -3,  10,
    -9,  -18, 2,   3,   -10, -1,  0,   0,   -25, 17,  -23, 16,  7,   7,   -21, -4,  -6,  -11, -18, 11,  13,  1,   -27, -7,  -27, 6,   14,  -13, -13, -5,  -14,
    15,  10,  1,   14,  4,   -10, -3,  -2,  -10, 15,  13,  4,   3,   -11, -23, 3,   10,  5,   8,   -11, -3,  -20, 1,   -7,  5,   8,   23,  -5,  -12, -1,  3,
    15,  6,   -4,  17,  4,   -37, 0,   -14, 18,  0,   2,   22,  7,   1,   -19, 16,  10,  -13, -20, 25,  24,  -4,  7,   6,   -8,  -5,  5,   -2,  0,   -27, 8,
    7,   4,   5,   -12, -12, -3,  19,  -2,  -12, 18,  -37, -3,  5,   -12, -8,  -23, 16,  -36, 22,  -4,  -4,  14,  -13, 11,  14,  7,   9,   -6,  -18, 2,   9,
    5,   1,   13,  -4,  2,   -18, 9,   -20, -9,  3,   27,  -31, -10, -11, -20, -7,  7,   -9,  18,  -2,  -19, -13, 0,   -21, -18, -28, -34, -20, -28, 11,  7,
    20,  -44, 13,  2,   3,   -2,  9,   7,   -7,  -6,  -17, 25,  -2,  12,  10,  4,   10,  12,  13,  12,  4,   -18, -35, -17, -22, 26,  -4,  16,  -6,  -25, 9,
    7,   -2,  -5,  -18, 3,   -12, 9,   -7,  -9,  9,   5,   -2,  -13, -19, 16,  15,  7,   10,  -1,  4,   -1,  -1,  -35, -17, -6,  -3,  -8,  -7,  -22, -14, 3,
    12,  3,   -12, -4,  -19, 0,   -5,  15,  -25, -7,  -22, 13,  -1,  4,   -21, 12,  -9,  26,  16,  -8,  -8,  -23, -7,  16,  6,   -10, 1,   -3,  -8,  9,   -4,
    -7,  1,   -16, -9,  -11, 5,   13,  18,  5,   -5,  11,  7,   1,   19,  0,   -20, 10,  16,  19,  -28, -13, -4,  -24, 8,   11,  25,  4,   1,   5,   5,   -5,
    12,  6,   6,   3,   20,  -9,  9,   -6,  2,   -11, 15,  3,   4,   -39, 19,  -12, -7,  10,  13,  -11, -1,  9,   4,   -4,  6,   -17, 4,   25,  7,   14,  4,
    10,  21,  2,   13,  -20, 3,   -20, -9,  -22, -12, 6,   1,   8,   0,   7,   1,   18,  5,   16,  -1,  7,   20,  8,   -3,  1,   -17, -4,  20,  15,  -5,  -13,
    -5,  -14, -12, 15,  -13, -2,  -12, 7,   21,  -2,  -12, -11, -13, 22,  -2,  5,   -18, 17,  7,   -18, -9,  -17, 10,  0,   -15, 11,  21,  -25, -10, -10, -21,
    2,   -6,  -7,  12,  5,   2,   19,  -17, 12,  20,  -2,  -7,  4,   11,  22,  15,  -7,  -8,  -16, 10,  -3,  -14, 18,  1,   -10, 15,  1,   2,   6,   2,   -4,
    1,   0,   20,  -6,  -1,  1,   -8,  -4,  4,   18,  2,   -7,  17,  11,  -2,  7,   15,  -4,  -11, -9,  -5,  13,  -10, 2,   -12, -12, 17,  -3,  -27, 20,  1,
    15,  4,   12,  22,  9,   2,   11,  3,   -2,  -18, 14,  -16, 8,   7,   -32, -3,  0,   14,  -3,  14,  -2,  2,   -4,  -9,  12,  -6,  8,   24,  8,   4,   2,
    -16, -4,  5,   -6,  6,   -3,  17,  11,  -9,  15,  -16, -21, -13, -1,  -21, 3,   -3,  -2,  2,   19,  6,   -11, 10,  11,  -7,  3,   -17, -31, -18, 6,   5,
    -21, 10,  23,  1,   22,  -13, -5,  8,   9,   15,  -1,  12,  -7,  0,   12,  -20, -25, 22,  10,  -10, 3,   -11, -2,  -38, 15,  5,   10,  8,   22,  1,   12,
    -9,  -10, 13,  -9,  15,  0,   17,  -17, -13, -11, -4,  4,   -3,  -27, 0,   -22, -5,  25,  12,  -2,  24,  -12, 8,   -13, 5,   -20, 12,  18,  -7,  4,   3,
    -17, 16,  -2,  -7,  -2,  13,  -22, -3,  3,   -21, -1,  -23, 14,  1,   -41, -27, -16, -8,  -24, 4,   6,   5,   24,  -17, 2,   -21, 15,  -34, -2,  5,   8,
    -20, -5,  8,   5,   14,  5,   -8,  1,   -18, 10,  -19, 4,   -1,  -13, 9,   2,   9,   -16, -4,  -13, 1,   -44, -2,  -6,  8,   -2,  10,  16,  -5,  24,  1,
    -25, -12, 6,   -5,  28,  -10, -9,  -12, 13,  -8,  -9,  -9,  -33, 8,   17,  18,  -1,  1,   3,   -9,  0,   10,  14,  6,   -4,  23,  5,   -10, -2,  -1,  -8,
    25,  -17, 16,  -7,  5,   -15, 1,   -28, -5,  22,  1,   3,   12,  -35, 14,  -19, -3,  12,  -8,  -7,  -3,  13,  7,   -13, 5,   5,   -2,  -1,  -5,  -8,  -9,
    2,   -1,  -6,  -5,  7,   -12, -12, -3,  -13, 0,   -4,  1,   -9,  5,   3,   -11, -3,  -4,  8,   -13, -8,  -5,  -2,  -1,  0,   4,   3,   2,   -1,  -12, -5,
    1,   -9,  3,   4,   -11, -8,  -5,  -3,  0,   0,   3,   -3,  -2,  -10, -3,  -13, 4,   4,   6,   7,   4,   -7,  -8,  -16, -11, 2,   2,   -7,  1,   -12, -9,
    -4,  -2,  2,   2,   0,   -7,  3,   6,   4,   -2,  8,   5,   -7,  -13, -6,  -5,  -7,  3,   -3,  -7,  -13, -10, -9,  -6,  -12, 6,   3,   -7,  7,   9,   -6,
    -1,  -11, 1,   -4,  -17, 1,   -2,  -13, -11, 3,   -5,  -7,  2,   -10, 1,   5,   -10, -1,  -16, 0,   -2,  -2,  -5,  -9,  -6,  -7,  -4,  10,  -11, 3,   -29,
    -25, 20,  6,   21,  16,  2,   2,   3,   -10, -8,  -18, -24, 36,  -4,  18,  12,  22,  -5,  20,  -12, 20,  6,   2,   -28, 21,  10,  -24, 23,  -16, 0,   0,
    -2,  -9,  2,   -27, -11, -9,  -3,  8,   40,  19,  -16, 16,  11,  -13, 7,   23,  -6,  -23, 5,   -4,  31,  -17, -6,  -1,  -7,  14,  0,   -33, 6,   2,   -20,
    21,  19,  14,  0,   -6,  21,  8,   -2,  11,  -12, -4,  14,  27,  26,  -15, 10,  22,  -4,  -2,  -10, 7,   -3,  -30, -7,  -14, -19, 2,   1,   7,   5,   0,
    1,   11,  -17, -2,  -21, 20,  13,  5,   -3,  -31, -44, 6,   3,   -16, -5,  -4,  2,   27,  -7,  -23, 1,   -12, -8,  -4,  0,   17,  -5,  20,  4,   -6,  -17,
    20,  10,  -14, 6,   1,   -18, 4,   -10, -15, -9,  26,  7,   -17, -2,  0,   2,   11,  -4,  8,   -1,  0,   17,  -9,  19,  6,   0,   -25, 12,  -5,  -15, 13,
    -2,  4,   -14, -2,  3,   -36, -9,  -17, 12,  -9,  -18, -22, 14,  -9,  10,  6,   -14, 8,   15,  -12, 15,  -24, -9,  -3,  -27, -2,  17,  11,  -1,  -10, -7,
    20,  4,   11,  17,  -8,  -12, 2,   2,   -5,  24,  6,   -6,  -23, 11,  17,  -11, -2,  -10, -8,  -4,  -25, 8,   -27, -6,  12,  5,   -13, 4,   2,   -17, 10,
    1,   12,  11,  -5,  20,  -4,  20,  -7,  10,  -15, 22,  -1,  16,  -4,  -17, -14, 4,   22,  9,   14,  27,  3,   -2,  5,   -4,  -15, 10,  -2,  3,   -2,  -4,
    4,   8,   -12, -30, -14, -10, 12,  6,   0,   -2,  1,   -9,  5,   -18, 10,  1,   11,  -4,  -1,  3,   24,  5,   -20, 3,   11,  0,   10,  0,   3,   3,   4,
    0,   -13, -4,  -18, 10,  1,   -3,  -8,  18,  2,   -14, -3,  23,  1,   -23, 20,  -2,  -7,  4,   -13, -33, 3,   6,   -13, 14,  15,  7,   14,  8,   7,   0,
    -6,  9,   -7,  -8,  -5,  -2,  -6,  -5,  8,   -10, 5,   -3,  -13, -29, 7,   -10, -7,  11,  7,   9,   -1,  1,   6,   11,  3,   6,   6,   7,   -5,  -4,  7,
    -2,  8,   -5,  19,  -12, -4,  3,   4,   -13, 1,   15,  17,  -16, 3,   -10, 7,   15,  -4,  2,   10,  -7,  9,   -3,  1,   -2,  -10, -7,  3,   2,   -12, -17,
    4,   1,   -6,  19,  1,   13,  6,   -17, 0,   6,   1,   20,  6,   7,   10,  -19, -25, -6,  12,  6,   3,   11,  10,  2,   -11, 10,  3,   9,   11,  0,   13,
    25,  7,   37,  -11, 8,   -14, 2,   -11, 12,  16,  -11, 4,   9,   -5,  -18, -7,  13,  -5,  -15, 7,   -9,  -14, 15,  5,   10,  1,   -2,  0,   3,   -6,  -17,
    18,  -3,  4,   -13, 16,  -18, 6,   -13, -12, 2,   -11, 21,  -22, -1,  -10, -29, -2,  5,   3,   4,   4,   4,   0,   3,   9,   1,   -21, 8,   2,   20,  -14,
    -1,  1,   17,  23,  20,  -21, 2,   17,  10,  -19, -1,  -4,  -28, -7,  1,   4,   -5,  7,   -13, -12, -13, -45, -20, 9,   -8,  1,   -14, 9,   -12, -27, 9,
    4,   -8,  0,   -10, -18, -3,  -6,  27,  -64, 1,   18,  -7,  -1,  -13, 10,  -9,  1,   -9,  2,   -11, 4,   -6,  11,  0,   -6,  -5,  -11, 0,   -11, 7,   -7,
    -5,  -2,  -3,  4,   -4,  3,   1,   4,   -14, -2,  3,   -8,  0,   -1,  -11, 5,   2,   -9,  3,   -11, -7,  4,   8,   -10, 6,   -3,  -1,  3,   -8,  2,   7,
    -8,  -6,  3,   -13, -10, -5,  -1,  -10, -1,  -7,  9,   0,   -8,  -9,  1,   -1,  -7,  -7,  -4,  -8,  -8,  -6,  2,   -6,  3,   -2,  7,   -11, -2,  -10, -9,
    -7,  -7,  0,   -4,  -4,  11,  3,   6,   -9,  8,   9,   1,   -4,  2,   3,   -4,  -3,  -9,  -11, -10, -7,  0,   6,   -15, -2,  6,   -3,  2,   -8,  0,   2,
    -2,  -8,  -3,  8,   7,   7,   -1,  2,   5,   -4,  1,   -8,  -16, -12, -2,  -2,  7,   -26, -15, -5,  -34, -7,  -16, -62, -3,  -13, -10, 4,   -5,  2,   27,
    -5,  -14, 16,  23,  -22, -25, 4,   9,   7,   -16, -14, -30, -23, -4,  -46, -10, 6,   -48, 4,   12,  -12, -34, 7,   12,  7,   5,   21,  -23, 8,   8,   -14,
    -22, 3,   -35, -28, 19,  -34, -7,  -9,  -11, -6,  15,  -4,  8,   2,   12,  1,   9,   5,   26,  22,  -6,  -16, -4,  -16, -6,  13,  19,  -33, -11, -27, -5,
    -27, -30, -5,  5,   -31, -36, -11, -15, 2,   -21, 29,  -51, 3,   25,  -1,  -11, 24,  -15, 8,   -1,  3,   19,  22,  -34, 17,  -16, -10, 16,  -37, -30, -32,
    -2,  14,  11,  11,  13,  -4,  -13, -5,  14,  -31, 5,   -32, -6,  -10, 9,   -7,  -25, 28,  -33, 2,   5,   1,   -4,  -14, -15, -6,  -10, -19, 8,   15,  -38,
    -5,  21,  15,  21,  13,  -8,  -30, -17, -1,  13,  -5,  -5,  -2,  9,   4,   5,   5,   -16, -52, -10, -4,  -5,  -47, 13,  14,  -1,  -7,  1,   -14, 12,  0,
    -3,  4,   9,   -15, 14,  -3,  29,  -20, -31, 20,  -27, -3,  19,  -31, 7,   -13, -6,  9,   17,  3,   -5,  -21, 1,   -14, 7,   -41, 13,  2,   -5,  -10, 16,
    9,   1,   -5,  4,   -1,  -34, -33, -14, -12, -21, 4,   15,  7,   7,   1,   -19, -9,  18,  -18, -10, -28, -3,  -1,  -8,  -2,  16,  -17, -19, 15,  -31, -18,
    -9,  10,  21,  19,  -4,  6,   -1,  -32, -3,  -31, -29, 7,   25,  -15, 9,   -17, -1,  -16, 1,   4,   16,  2,   2,   -5,  -18, -1,  -26, -10, -25, 4,   19,
    -3,  4,   24,  9,   1,   -17, -27, 5,   8,   -3,  3,   -7,  -10, 19,  -3,  -27, -52, 11,  -23, -10, 29,  -14, 5,   16,  22,  1,   -1,  8,   9,   -29, -28,
    -8,  0,   -5,  14,  -14, 8,   -27, -27, -22, 3,   -13, 17,  -2,  11,  -5,  -6,  8,   -5,  -23, 7,   -11, -6,  0,   -4,  14,  2,   0,   -11, 2,   21,  -7,
    -30, -5,  -30, 11,  -1,  9,   6,   -11, -8,  13,  -7,  -22, 7,   -6,  -30, -8,  -25, -14, -1,  11,  18,  20,  27,  -12, 16,  -7,  5,   -13, 15,  12,  -8,
    22,  -33, 3,   -12, -27, 11,  15,  11,  14,  -22, 11,  5,   12,  13,  -1,  19,  0,   7,   -15, 6,   -8,  -12, -1,  -4,  -12, 3,   3,   7,   -33, 5,   -18,
    -35, -14, -13, -55, 3,   6,   -22, -38, -1,  34,  25,  -29, -15, -19, 8,   -7,  -7,  -13, -6,  -14, 8,   -27, -21, 9,   -23, 2,   2,   -20, 12,  -26, 10,
    -18, -16, -21, -14, -6,  -17, -31, -24, 11,  -37, 12,  1,   -52, -4,  -18, -13, 4,   16,  0,   -13, 22,  1,   10,  -65, 29,  30,  13,  -3,  -3,  -21, -32,
    12,  -3,  -5,  -44, -19, -40, 8,   23,  -8,  1,   10,  14,  3,   14,  18,  -28, 0,   1,   -17, -2,  6,   6,   9,   36,  22,  1,   -20, 9,   -15, -6,  10,
    -37, -19, 27,  -11, -17, -3,  -10, 16,  13,  1,   -3,  -21, 13,  -42, 20,  -9,  17,  -15, -21, -15, -7,  -38, 13,  15,  4,   -3,  1,   -28, -5,  18,  4,
    -3,  -5,  -19, -10, 1,   17,  0,   -4,  17,  2,   -7,  5,   -20, 19,  -15, 16,  12,  -1,  10,  9,   11,  7,   -9,  5,   7,   -15, 8,   0,   7,   12,  -23,
    -6,  4,   -5,  -12, -7,  5,   -19, -20, -4,  -30, -7,  0,   -8,  12,  -16, 4,   -4,  31,  -6,  -5,  2,   26,  0,   1,   8,   -4,  11,  -13, 5,   1,   15,
    -19, 8,   6,   -18, 5,   -12, -9,  1,   -16, 1,   5,   19,  -3,  11,  -20, -15, 2,   -5,  1,   21,  -3,  -23, -11, 7,   -23, 17,  -3,  9,   -16, 9,   6,
    4,   7,   10,  7,   -12, 5,   -9,  16,  -4,  1,   -3,  -7,  17,  5,   -3,  -15, -4,  22,  -11, -4,  -9,  7,   0,   12,  22,  -19, 14,  20,  20,  -6,  -19,
    -10, 5,   6,   11,  -4,  16,  -7,  -28, 12,  -3,  -13, -23, -19, -1,  -3,  -21, -13, -8,  -13, -4,  -28, -20, 14,  14,  -18, 17,  4,   -13, -3,  8,   30,
    -3,  1,   3,   16,  13,  11,  -34, -2,  0,   9,   -21, 22,  -7,  6,   -3,  -15, -21, 31,  -9,  -14, -12, -11, 29,  4,   -3,  9,   -7,  -5,  -24, -11, -2,
    -1,  21,  -2,  2,   -9,  -4,  -7,  12,  1,   9,   0,   5,   6,   -4,  8,   2,   5,   13,  -4,  -18, -25, 12,  6,   -9,  -5,  2,   -34, 4,   11,  -14, 3,
    -16, 10,  -8,  -11, -2,  17,  5,   -24, 11,  -8,  2,   -15, -37, -1,  10,  -22, 7,   -1,  -1,  -1,  6,   0,   17,  -9,  22,  18,  -14, 6,   -7,  13,  13,
    26,  10,  -22, -6,  -22, -1,  -22, -26, -22, 5,   -19, -13, -20, -9,  0,   15,  -21, -4,  14,  24,  -1,  -19, 3,   4,   -21, 20,  -8,  21,  -13, 2,   -14,
    12,  -33, -10, 0,   -20, 23,  1,   -1,  -1,  -55, 0,   -35, -11, 9,   -44, -7,  -11, 1,   -5,  2,   12,  -4,  -1,  -34, 0,   5,   11,  -40, -32, 1,   17,
    23,  -17, 26,  -6,  1,   -8,  7,   -12, 0,   -3,  9,   13,  -3,  -15, 20,  -9,  -4,  4,   -4,  -18, 22,  -80, -16, 20,  -6,  17,  2,   -11, -34, 16,  -12,
    11,  -22, 2,   0,   15,  -13, 4,   -14, 9,   -12, -25, -5,  -32, 18,  22,  6,   -12, -14, -24, 28,  -10, 17,  -21, -27, 13,  -38, -15, -22, -35, -47, 28,
    4,   25,  -8,  19,  9,   6,   -9,  8,   -30, -18, -5,  -12, 5,   -3,  14,  11,  24,  6,   20,  13,  2,   -8,  10,  -20, -10, 9,   8,   -6,  7,   17,  -17,
    9,   -5,  33,  -12, 8,   4,   -5,  -10, -9,  -17, 21,  -1,  -14, -8,  12,  -12, -5,  -17, 11,  4,   -12, -15, -6,  15,  -15, -9,  1,   13,  -19, 6,   3,
    -6,  7,   -1,  2,   -5,  4,   7,   -20, 8,   14,  -5,  1,   1,   -10, -24, 0,   14,  -5,  4,   -10, 5,   -11, 3,   -16, -4,  3,   14,  -20, 11,  8,   -10,
    10,  7,   15,  17,  13,  -6,  6,   -14, 4,   8,   5,   0,   20,  -17, -19, 16,  -10, 0,   5,   2,   -27, 15,  -2,  -13, -20, 1,   -11, -7,  12,  7,   17,
    21,  1,   1,   -12, -4,  -4,  -6,  17,  6,   -6,  14,  -12, 17,  16,  11,  6,   13,  -9,  4,   -3,  9,   -40, -2,  6,   -22, 5,   12,  6,   -15, 2,   -34,
    -4,  -18, -11, 7,   17,  4,   -12, -20, 6,   -23, 4,   -6,  -4,  17,  3,   30,  27,  -16, 26,  23,  -11, -9,  -6,  -18, 9,   13,  -1,  11,  -14, 2,   -22,
    -9,  -9,  -12, -18, -3,  -3,  -9,  2,   -10, 2,   16,  -19, 4,   23,  5,   -1,  4,   8,   -48, 0,   18,  -7,  -10, 11,  -32, -22, -32, -11, -1,  13,  -15,
    2,   2,   -6,  -3,  -5,  14,  5,   6,   13,  -11, 20,  -15, -10, 14,  -59, 6,   24,  38,  3,   -5,  11,  -11, -44, -1,  -8,  -11, -11, 1,   1,   4,   3,
    13,  -8,  -27, -12, 9,   5,   -29, 21,  8,   17,  -11, 3,   -6,  -24, -32, 4,   13,  -49, -22, 5,   18,  -10, 3,   14,  -1,  -3,  -9,  11,  7,   -38, 7,
    -8,  13,  3,   -9,  -14, 15,  2,   -1,  13,  -29, -11, 23,  26,  3,   -10, -39, -16, 1,   -23, 14,  2,   2,   21,  6,   12,  2,   -14, 16,  15,  11,  -11,
    4,   6,   32,  -7,  12,  -18, -8,  0,   -4,  6,   -9,  -22, 0,   -16, -1,  -37, -1,  -2,  9,   8,   -24, -13, 17,  0,   14,  0,   -4,  -9,  -10, -12, -3,
    1,   15,  -1,  -22, -19, 32,  15,  6,   6,   1,   18,  6,   0,   8,   -39, 0,   6,   -35, -8,  -4,  -10, 6,   -18, -6,  -20, 0,   3,   2,   -20, 19,  12,
    6,   -9,  -11, 17,  4,   -24, 25,  4,   0,   14,  2,   -10, 7,   -16, -30, -9,  16,  -19, 6,   9,   0,   -9,  -18, -5,  -53, -60, 1,   -50, 3,   -56, 19,
    -6,  -29, 1,   6,   -3,  5,   7,   -20, 7,   29,  -19, -38, 17,  -10, -1,  18,  2,   -49, -2,  0,   12,  -16, -17, -19, -26, 20,  2,   -3,  -1,  -1,  -27,
    -16, -1,  7,   3,   -24, -16, -9,  1,   -61, -8,  15,  6,   -36, 10,  5,   -2,  5,   13,  -17, 2,   -29, 3,   -5,  13,  -3,  21,  4,   15,  -25, 4,   -8,
    1,   -10, -7,  -14, -14, -3,  -16, 3,   -35, 10,  17,  3,   -51, -4,  -1,  -3,  -9,  -14, 0,   20,  10,  -7,  14,  -5,  -24, 1,   6,   6,   15,  -24, 9,
    -39, 0,   -22, 24,  21,  11,  -3,  13,  21,  1,   1,   12,  7,   12,  -3,  22,  -8,  13,  -2,  6,   -9,  0,   -18, 1,   20,  26,  -40, -5,  -36, -2,  -1,
    -33, 14,  12,  19,  9,   -3,  -6,  3,   22,  -11, 5,   -1,  -4,  -3,  0,   -5,  16,  -14, 3,   14,  -8,  12,  18,  6,   0,   9,   13,  6,   7,   -5,  30,
    -4,  -17, -4,  -4,  6,   -21, -3,  8,   -4,  -7,  17,  9,   14,  15,  1,   2,   -10, 8,   -5,  -9,  12,  5,   13,  -13, 16,  -8,  5,   11,  21,  -3,  7,
    -9,  -20, 4,   1,   -32, 2,   -23, -1,  -2,  2,   11,  3,   -2,  0,   -7,  13,  0,   9,   8,   -3,  5,   4,   2,   -4,  -19, 8,   -16, -6,  14,  15,  -15,
    -6,  -4,  -3,  18,  -4,  7,   -10, 4,   -4,  14,  -6,  9,   -9,  15,  -2,  19,  8,   -26, -5,  2,   -24, -6,  0,   -6,  19,  4,   -7,  1,   21,  -9,  2,
    -3,  14,  -2,  -9,  -22, -30, -13, 14,  -11, -6,  21,  18,  -13, 7,   -3,  6,   16,  -11, 10,  1,   -9,  4,   15,  -23, 0,   22,  -15, 7,   11,  3,   -5,
    18,  16,  5,   10,  -2,  -13, -27, -34, 10,  1,   -10, 10,  -23, -5,  9,   -2,  18,  15,  -27, 7,   -9,  -1,  5,   28,  -6,  2,   13,  -1,  11,  -13, 6,
    -5,  -8,  -13, -4,  -15, -16, -31, 1,   -2,  -15, 7,   -30, 4,   3,   2,   0,   -5,  -29, -6,  -16, 0,   0,   -18, 20,  -37, -27, 32,  2,   17,  -3,  17,
    -14, -1,  -3,  4,   21,  -14, -9,  12,  -9,  13,  2,   10,  6,   -11, 18,  19,  -2,  12,  23,  -5,  -13, -36, -9,  -24, -14, -17, -4,  0,   -1,  11,  15,
    -2,  -55, 1,   28,  -14, -15, 27,  -8,  7,   2,   22,  -11, 26,  12,  8,   -12, -33, -6,  -4,  -12, 1,   18,  -5,  -3,  -4,  -6,  -3,  4,   23,  13,  16,
    -7,  5,   15,  -26, 9,   -5,  24,  0,   -4,  2,   -7,  12,  -2,  3,   8,   20,  4,   -16, 37,  -4,  -6,  9,   -15, 1,   -6,  -1,  -5,  11,  -19, -7,  -4,
    4,   -6,  -34, 3,   -17, -43, 10,  7,   12,  21,  0,   -6,  16,  6,   -22, -4,  -3,  14,  10,  -22, -17, -17, -25, 8,   -13, 19,  12,  9,   2,   -19, 0,
    -16, 6,   8,   8,   10,  -7,  7,   13,  1,   9,   9,   -22, 14,  8,   -6,  -39, -25, 0,   4,   6,   2,   2,   -7,  -14, -4,  -12, 22,  -4,  -34, 8,   9,
    -24, -15, -4,  -14, 3,   1,   -7,  24,  -18, 2,   -11, 8,   -7,  3,   -12, 9,   -4,  -11, -9,  6,   -3,  -11, -2,  -10, -8,  -4,  6,   7,   -5,  3,   1,
    3,   -9,  1,   -4,  -12, -9,  0,   -12, 1,   -4,  1,   -11, -11, 2,   3,   4,   -7,  3,   -4,  1,   5,   0,   -7,  -9,  3,   8,   -7,  -11, 0,   -3,  -4,
    1,   -13, -9,  9,   -9,  2,   -6,  -12, -2,  -10, 4,   -9,  -12, -3,  -7,  -4,  5,   -5,  6,   3,   -12, 8,   -10, 0,   -6,  -12, -5,  5,   2,   -11, -13,
    -4,  -1,  -3,  -3,  -7,  8,   3,   -5,  6,   3,   -11, 4,   4,   7,   -12, 4,   -10, -5,  8,   -5,  -1,  7,   -11, -2,  -5,  -9,  2,   -11, -8,  -10, 1,
    -4,  -3,  -9,  -9,  -7,  -2,  -9,  -11, -13, -8,  -4,  -5,  -3,  -24, -10, 15,  -7,  -22, 17,  15,  7,   4,   4,   -29, -15, -16, -38, 16,  -16, 11,  5,
    1,   13,  0,   -34, -19, 0,   -11, 0,   -19, 8,   0,   9,   -4,  -6,  -1,  -7,  -11, -19, 16,  -1,  -25, -9,  2,   -4,  -4,  1,   15,  12,  1,   10,  -16,
    25,  8,   -12, -22, 13,  -5,  3,   13,  4,   18,  -32, -11, 12,  -5,  16,  -11, -38, 8,   4,   3,   29,  -5,  -14, 17,  -52, 16,  18,  -5,  12,  -2,  14,
    -15, 19,  7,   2,   3,   5,   -25, -52, -21, 11,  -19, -24, -10, -12, 12,  -10, -13, 23,  9,   -11, -2,  -21, -15, -26, -16, 14,  -12, -2,  0,   18,  -23,
    6,   1,   10,  15,  -30, 25,  -37, -6,  17,  3,   17,  -7,  -11, -8,  9,   -5,  -1,  3,   12,  -12, 9,   10,  0,   -1,  -3,  -14, -24, 2,   17,  -19, 8,
    8,   -1,  -10, -18, -1,  -6,  9,   2,   4,   -16, -1,  -11, 11,  -17, -9,  -8,  -18, 9,   3,   5,   7,   15,  10,  -20, -3,  7,   -6,  -20, -2,  1,   -7,
    7,   19,  -8,  -1,  -2,  11,  1,   -4,  1,   -12, -4,  -4,  25,  -4,  2,   4,   5,   -15, 11,  4,   5,   6,   -16, -7,  4,   3,   5,   4,   11,  1,   -6,
    11,  -5,  19,  9,   -7,  -2,  4,   -12, 1,   -42, -35, 21,  -16, -13, -5,  -8,  -7,  -11, -3,  -7,  18,  10,  0,   -27, -3,  -5,  -14, 9,   22,  -2,  15,
    12,  7,   -7,  11,  -2,  -4,  -44, 18,  8,   -8,  -3,  11,  5,   1,   -3,  2,   -11, -9,  4,   8,   23,  10,  -11, -7,  3,   7,   -2,  -5,  -12, 11,  -8,
    5,   -2,  16,  -13, 12,  -12, -11, 2,   -27, -31, 14,  -3,  -14, 27,  -19, -33, 3,   11,  15,  -11, -14, 2,   7,   -6,  7,   12,  -16, -6,  -2,  5,   15,
    -3,  16,  6,   -21, 0,   -2,  8,   15,  -35, -3,  10,  16,  10,  -6,  14,  20,  13,  -16, 30,  -10, 12,  -26, 10,  -4,  -27, -8,  -9,  -5,  6,   -7,  -19,
    21,  -6,  14,  10,  -4,  -7,  18,  18,  13,  10,  20,  -1,  13,  -8,  2,   1,   -34, -3,  -18, -3,  -21, -14, -7,  9,   -6,  -1,  -8,  -7,  19,  2,   -16,
    -2,  17,  17,  -8,  27,  -15, 16,  -7,  5,   0,   -17, -1,  8,   1,   4,   15,  -1,  10,  -3,  1,   -8,  12,  -84, 18,  -7,  -12, 4,   7,   3,   8,   25,
    -1,  -28, -39, 2,   11,  -17, -11, -19, -8,  15,  -3,  -39, 7,   12,  4,   -12, -30, 11,  13,  -42, -4,  -29, 9,   2,   -28, -15, -10, 11,  -24, 0,   18,
    -15, -7,  -3,  -18, 6,   17,  -18, -9,  -19, -13, 5,   -32, 5,   -10, 2,   7,   -15, -1,  12,  -17, -28, -16, 0,   -6,  6,   15,  22,  12,  27,  -12, 11,
    3,   -28, -9,  -1,  17,  -12, 4,   17,  -24, 12,  21,  -9,  -13, 19,  6,   11,  10,  -5,  -14, -14, 16,  7,   -29, -18, -4,  -9,  -1,  12,  12,  13,  3,
    -33, 25,  -12, -12, 12,  -6,  13,  -21, 27,  14,  -21, 9,   -10, -5,  -7,  24,  19,  17,  1,   -30, -2,  5,   -17, 5,   16,  11,  -31, 5,   -9,  -4,  27,
    -47, -13, -11, 12,  11,  -23, -3,  1,   -3,  -3,  -1,  -6,  -5,  8,   -1,  -57, -3,  -6,  -2,  -32, 17,  -13, 17,  -23, 0,   3,   -5,  6,   -19, 0,   -8,
    2,   -50, 5,   -8,  1,   15,  -19, 10,  12,  3,   7,   -9,  -11, -13, 21,  0,   9,   -8,  -18, -11, 23,  10,  1,   8,   -39, -72, -20, 10,  -4,  -24, 1,
    -6,  11,  16,  11,  -22, -7,  -30, -28, 10,  -1,  -13, 5,   14,  -20, 37,  29,  -22, 10,  8,   5,   -8,  -5,  -5,  -2,  19,  -6,  -17, 2,   27,  -9,  -1,
    -6,  -16, -24, -32, -10, 7,   1,   14,  5,   4,   17,  8,   -3,  -41, -15, 20,  -6,  14,  14,  -12, -53, -1,  14,  33,  17,  21,  3,   12,  -6,  8,   12,
    -35, 0,   7,   -12, 16,  -15, 30,  -6,  -3,  -2,  -2,  11,  -19, -42, 9,   8,   9,   -28, 0,   -5,  0,   -1,  10,  -9,  -17, 17,  -2,  -15, -25, 7,   11,
    1,   -29, -27, -17, 14,  -13, -37, 13,  -39, 3,   -7,  10,  13,  16,  12,  16,  30,  -17, -23, -40, -4,  14,  14,  16,  -9,  3,   -1,  13,  5,   15,  -9,
    1,   9,   -28, 12,  11,  9,   23,  -15, 0,   -6,  -12, 0,   12,  -13, -21, -14, -16, -31, -10, -25, 1,   -7,  12,  0,   15,  -1,  -7,  1,   13,  5,   3,
    -9,  7,   22,  20,  -3,  18,  17,  -3,  8,   7,   32,  15,  18,  -7,  24,  -18, 7,   -8,  -22, -6,  -11, -4,  -45, -6,  8,   7,   -15, -4,  14,  0,   -25,
    -25, 17,  -29, 5,   8,   24,  4,   -9,  -2,  -1,  8,   -22, 10,  -11, -11, 26,  -3,  -24, 2,   10,  8,   0,   -6,  22,  -18, 3,   18,  -10, -29, 10,  -3,
    2,   -7,  5,   15,  11,  11,  1,   -6,  2,   -4,  -9,  1,   24,  18,  14,  13,  1,   -1,  1,   -31, -6,  1,   -13, -8,  4,   -1,  8,   19,  8,   -4,  17,
    14,  -2,  5,   13,  -18, -9,  -32, -5,  34,  1,   24,  -13, -9,  -11, -25, 18,  -12, -16, 11,  18,  31,  -7,  -34, 6,   5,   16,  4,   10,  -14, -6,  -16,
    2,   9,   -5,  10,  9,   -10, 0,   12,  15,  -13, 13,  9,   16,  -6,  -3,  -5,  17,  6,   -20, 5,   -9,  16,  19,  4,   -11, 6,   -4,  17,  19,  -10, 6,
    2,   25,  -18, 22,  -4,  2,   0,   1,   6,   2,   -3,  -5,  -24, 21,  7,   13,  -23, -8,  -12, -20, 15,  7,   -3,  19,  -16, 15,  -19, 14,  -8,  -24, -16,
    12,  -16, -6,  10,  -5,  -6,  -13, 13,  10,  -17, 10,  9,   14,  -2,  2,   -11, 8,   6,   -4,  8,   -5,  0,   3,   1,   -34, 14,  -6,  -21, -24, -7,  9,
    -8,  -2,  0,   10,  22,  1,   -27, 7,   -27, 4,   -6,  0,   4,   5,   0,   -27, -34, 8,   28,  -18, -10, -4,  3,   5,   -4,  -9,  -1,  16,  19,  7,   -4,
    -21, 6,   13,  -1,  11,  7,   -16, -22, -23, 22,  22,  8,   -11, 8,   2,   -1,  3,   -12, -19, 18,  5,   31,  13,  -8,  2,   -50, -11, 7,   11,  9,   -19,
    -6,  12,  29,  11,  0,   13,  -16, 0,   19,  -21, 5,   6,   -26, -30, -10, -36, 8,   12,  7,   8,   27,  -2,  16,  -3,  16,  16,  2,   -3,  9,   7,   -1,
    3,   -14, -16, -14, -12, 5,   -23, 3,   15,  1,   -17, 1,   7,   -15, 0,   6,   3,   18,  14,  -18, -3,  -3,  -14, -6,  7,   15,  3,   -2,  9,   -8,  -2,
    -12, 5,   7,   20,  -19, 6,   12,  10,  -5,  10,  -7,  6,   15,  5,   -10, 22,  -12, -14, 8,   3,   26,  14,  -22, 18,  1,   -11, -10, 2,   8,   -12, 8,
    13,  7,   -2,  15,  7,   -5,  -5,  -2,  5,   7,   -15, -5,  10,  3,   -12, 6,   -16, -1,  -2,  18,  -6,  0,   3,   5,   -13, -4,  0,   -1,  9,   -18, 14,
    13,  14,  9,   4,   9,   -4,  -13, 9,   2,   -1,  -9,  33,  21,  8,   1,   0,   11,  -6,  3,   -11, 4,   -11, 5,   -7,  11,  3,   -15, -11, 30,  4,   -1,
    7,   3,   24,  18,  13,  -23, 2,   17,  2,   10,  10,  12,  20,  -13, -29, 1,   -14, 3,   -12, 0,   -10, -21, 2,   6,   8,   -4,  -12, -10, 8,   6,   1,
    8,   -7,  8,   12,  -2,  2,   -8,  -10, 5,   1,   -10, 12,  -11, -15, -20, -5,  4,   -4,  2,   -13, 14,  -9,  -14, 11,  1,   18,  -4,  10,  10,  -6,  8,
    9,   1,   12,  -19, 8,   -30, 7,   -3,  0,   -11, -4,  -18, 13,  8,   -6,  -40, -8,  3,   -8,  -4,  -9,  7,   16,  6,   12,  25,  -15, 0,   1,   12,  3,
    21,  13,  6,   -10, -2,  -2,  -11, -35, 3,   12,  0,   -9,  -1,  -18, -4,  -22, -4,  8,   -30, 16,  -13, 2,   6,   -5,  -5,  6,   2,   -27, -6,  26,  -52,
    11,  -15, 0,   8,   27,  -13, 6,   5,   -11, 22,  2,   -19, -17, -5,  -16, 15,  -15, 1,   5,   16,  -7,  -24, 10,  -14, 12,  13,  -8,  1,   1,   -6,  -2,
    9,   10,  -5,  9,   -7,  15,  -1,  -5,  -2,  9,   -21, 0,   12,  -1,  -1,  23,  11,  -27, 21,  -4,  -17, 21,  -12, 11,  4,   21,  -4,  -13, -7,  -2,  -18,
    -17, -23, -10, -25, -26, 5,   6,   -1,  15,  -4,  -14, 11,  -1,  8,   10,  -7,  14,  1,   1,   2,   -2,  25,  -6,  20,  5,   -10, -2,  16,  -10, -7,  3,
    -3,  -2,  18,  18,  -21, -10, -27, -12, -5,  -15, 23,  -4,  1,   -4,  7,   1,   12,  27,  25,  -13, 7,   -41, -8,  1,   -19, -8,  -2,  5,   -10, 5,   -9,
    22,  14,  -6,  -12, 9,   9,   -3,  6,   1,   8,   0,   -9,  -2,  13,  -3,  -14, 0,   10,  14,  20,  19,  2,   16,  7,   2,   7,   -20, 20,  6,   14,  15,
    -25, -2,  0,   -5,  4,   5,   -8,  14,  -17, -6,  -6,  21,  26,  -3,  19,  -20, -4,  8,   -22, -10, 10,  -3,  17,  -10, -2,  23,  -15, -19, 6,   -10, 10,
    -14, -9,  -2,  9,   26,  14,  19,  -5,  11,  -36, 8,   20,  31,  4,   -6,  -20, 11,  -27, 13,  -1,  -7,  6,   0,   9,   -11, 30,  1,   -8,  -8,  15,  8,
    -12, -6,  -34, -1,  12,  1,   -10, 0,   -16, -1,  -22, 3,   3,   18,  5,   7,   26,  -11, -8,  -12, 1,   -9,  19,  -11, -39, -4,  9,   -31, -19, -22, -2,
    -7,  2,   -1,  -16, 17,  -14, 16,  -18, -8,  13,  0,   12,  -2,  10,  -11, -6,  -27, -48, 6,   -9,  6,   -8,  7,   15,  0,   -2,  19,  -16, 1,   9,   -12,
    12,  12,  -12, -6,  21,  44,  -30, -10, 8,   33,  -13, 22,  2,   -26, 15,  29,  -16, 1,   -13, 3,   18,  4,   -11, 4,   17,  5,   -10, -14, 16,  -26, -4,
    -20, -10, -4,  9,   -3,  18,  13,  -26, 4,   7,   3,   21,  16,  -10, -2,  -16, 1,   0,   24,  -18, -33, -15, -17, -13, 18,  5,   -10, 26,  -27, -3,  -4,
    2,   22,  5,   7,   3,   -19, -1,  11,  19,  5,   7,   -31, -9,  37,  11,  -2,  15,  14,  2,   -5,  -19, 7,   19,  43,  -12, -25, -13, -12, -15, -23, -1,
    -20, 5,   -18, -6,  6,   15,  0,   -31, -14, 10,  6,   -5,  -4,  1,   -4,  -2,  -10, 20,  -3,  9,   -7,  13,  20,  5,   20,  3,   -34, -49, -14, -17, 11,
    -3,  0,   -39, 3,   -7,  30,  32,  6,   8,   -6,  23,  9,   0,   10,  12,  -11, -27, 3,   -27, 6,   6,   19,  10,  -1,  -32, 25,  12,  -13, 10,  9,   -22,
    -31, 3,   -9,  15,  -5,  -1,  11,  -18, 4,   -23, -8,  -9,  4,   22,  0,   -4,  -28, -8,  -18, 9,   12,  -1,  -14, 29,  -8,  -32, 10,  3,   9,   -1,  -18,
    8,   -2,  22,  14,  -17, -1,  2,   -17, 11,  10,  -7,  4,   -4,  10,  15,  9,   9,   -4,  -12, -4,  -15, -1,  12,  0,   4,   -15, 3,   -28, 9,   -9,  12,
    -17, 2,   -2,  -20, 4,   -23, 9,   -22, 24,  16,  -13, 5,   6,   -2,  -6,  -10, 25,  3,   16,  5,   19,  9,   16,  23,  -11, -2,  -8,  -16, 22,  -41, 1,
    -1,  -28, -22, 8,   3,   14,  -3,  8,   -2,  21,  9,   -2,  11,  -8,  -3,  6,   -9,  -5,  -15, -11, -19, 7,   -14, 3,   -3,  21,  6,   -19, -15, 15,  -14,
    -26, 2,   1,   13,  14,  -4,  4,   -28, 7,   4,   2,   -1,  6,   5,   15,  -2,  -23, 4,   8,   37,  16,  -32, 8,   -17, -12, -4,  15,  -1,  -7,  28,  7,
    14,  -6,  16,  -18, -33, -21, -6,  17,  -5,  -7,  0,   -10, 10,  7,   -36, -23, -32, 1,   11,  -11, 39,  -9,  7,   -12, -14, -8,  22,  -7,  -12, 23,  4,
    10,  14,  35,  5,   -13, -30, -7,  -11, 19,  4,   -1,  -2,  12,  -1,  13,  -12, -6,  -14, -2,  14,  4,   12,  5,   -27, 7,   -20, 11,  21,  17,  9,   -13,
    -4,  5,   -2,  -18, 7,   4,   16,  7,   -9,  14,  -2,  -4,  20,  8,   10,  30,  -13, 11,  -5,  -18, 18,  5,   -9,  -6,  24,  -26, -12, 7,   16,  2,   -5,
    -14, 12,  -6,  -10, -34, 6,   3,   12,  -17, 22,  4,   3,   9,   -20, -8,  0,   -5,  2,   15,  6,   10,  -9,  12,  -12, 8,   4,   16,  -8,  -10, 1,   27,
    -1,  -23, -15, 7,   5,   3,   -5,  -23, 2,   -3,  2,   -2,  -7,  -16, -7,  11,  8,   -4,  -1,  2,   23,  2,   -13, -7,  11,  -5,  -10, 4,   -4,  1,   -13,
    11,  0,   19,  -11, -10, 0,   -15, 0,   -4,  -4,  -34, 11,  -18, -4,  -29, 1,   -15, -17, -7,  25,  19,  -7,  -12, 1,   -27, 15,  10,  3,   7,   11,  5,
    -11, 3,   8,   8,   8,   20,  -22, -4,  -8,  -10, 11,  -7,  17,  -14, -8,  -4,  19,  -6,  -12, 6,   9,   4,   -13, -13, -7,  2,   1,   6,   15,  -29, -5,
    3,   1,   33,  25,  -13, -17, -2,  6,   -18, -6,  -41, -20, 6,   7,   -2,  -14, 11,  0,   20,  27,  -6,  -14, -17, -5,  -33, -1,  -10, 15,  16,  -23, 2,
    -19, -21, 12,  8,   22,  12,  -3,  -3,  -1,  8,   26,  2,   -17, -3,  5,   5,   -7,  0,   6,   11,  -19, 1,   7,   -26, 3,   -8,  10,  -16, 19,  1,   -22,
    -1,  5,   -5,  11,  -1,  18,  -10, 1,   2,   21,  -15, 6,   2,   -8,  12,  14,  -21, 6,   3,   10,  -7,  4,   -17, 1,   18,  5,   -9,  -19, 2,   35,  -1,
    -16, -24, 0,   -16, -12, 10,  23,  -3,  -17, -25, -1,  11,  -9,  12,  2,   -4,  1,   4,   -11, 0,   -13, 15,  20,  2,   9,   -2,  -10, 45,  -23, 10,  18,
    12,  -10, 8,   -7,  14,  -13, -4,  -9,  17,  -14, 9,   -9,  -19, 9,   13,  -4,  -5,  3,   1,   10,  -23, 8,   -13, -1,  -1,  21,  5,   9,   27,  4,   -36,
    19,  7,   0,   -4,  -11, -30, 9,   -21, -4,  -7,  0,   -15, -27, 2,   8,   -3,  0,   -6,  -5,  -1,  5,   0,   8,   15,  19,  -12, 13,  12,  20,  -3,  15,
    8,   -9,  -11, 0,   -6,  -23, -3,  -2,  -10, -4,  10,  1,   -12, 12,  17,  -13, -20, -14, 8,   4,   12,  -9,  5,   19,  -28, -27, -3,  -10, -7,  2,   -21,
    -4,  -16, 8,   -16, -9,  14,  25,  -23, -4,  -31, 29,  -1,  0,   7,   -16, 4,   -16, 5,   7,   -9,  15,  -18, -34, 17,  19,  -2,  5,   -10, 13,  5,   -14,
    0,   -5,  -12, -21, 20,  8,   -16, -26, 0,   -3,  1,   -5,  -14, -29, -1,  -27, -1,  0,   -13, -21, -37, -15, -13, -8,  18,  22,  -20, -27, 9,   0,   -3,
    14,  -20, -25, -16, 26,  4,   8,   -10, 14,  -30, -7,  -5,  9,   13,  5,   11,  6,   3,   15,  -38, 0,   6,   19,  -21, 20,  8,   15,  -24, -33, 1,   4,
    14,  -11, 0,   6,   7,   13,  -2,  26,  -30, 8,   -38, 7,   2,   3,   16,  -11, -14, -15, -13, 24,  -5,  6,   -15, -2,  22,  -3,  -5,  -6,  9,   -13, 10,
    -11, -7,  10,  11,  -5,  -4,  -7,  -25, -15, 6,   -7,  -6,  23,  5,   -25, 14,  2,   0,   13,  8,   -12, -5,  -17, -2,  6,   13,  -11, 4,   10,  -5,  7,
    11,  -3,  4,   -3,  -15, -15, 7,   1,   2,   0,   10,  -8,  -4,  -23, -1,  -4,  -6,  10,  4,   -7,  -13, -11, 14,  10,  -6,  3,   -9,  29,  4,   0,   -9,
    8,   -32, 8,   1,   3,   15,  -1,  14,  13,  -16, -7,  11,  0,   19,  4,   -1,  -23, -23, -9,  -29, -49, 20,  13,  16,  -12, -9,  5,   -4,  12,  4,   -3,
    8,   -10, 9,   10,  -9,  -1,  -4,  4,   13,  -3,  12,  0,   -3,  -7,  -6,  -22, -48, -5,  2,   -9,  8,   10,  -7,  -10, 4,   5,   3,   0,   0,   10,  14,
    -8,  -21, 10,  -14, -72, -45, -6,  -2,  -47, 0,   -6,  -8,  4,   -16, 5,   18,  -41, 17,  6,   40,  -20, -14, -1,  -8,  1,   11,  -6,  -20, -12, 21,  -24,
    9,   -1,  18,  23,  -9,  15,  20,  -24, 17,  1,   -11, -35, 8,   7,   -29, -14, 14,  -36, -18, 24,  1,   -11, -18, -9,  15,  -45, -2,  -12, 6,   2,   -10,
    -1,  -27, 11,  26,  7,   16,  5,   -9,  6,   -1,  8,   26,  -16, 11,  27,  2,   -7,  4,   -9,  -6,  14,  -18, -8,  11,  -10, -7,  0,   -17, 6,   -2,  -37,
    -12, 0,   -2,  -8,  14,  -23, 5,   -14, 15,  -18, 3,   -17, 6,   -8,  -25, 7,   -9,  -10, 12,  1,   -13, -11, 0,   -11, 5,   -2,  11,  -40, -6,  -44, 9,
    -1,  -15, -18, -24, 0,   9,   -4,  -41, -26, -22, 0,   -23, 18,  10,  -14, 25,  -19, -53, 23,  -20, 14,  3,   -21, 1,   -40, 11,  -7,  24,  -10, 5,   -15,
    10,  27,  0,   1,   -23, -35, 2,   -5,  11,  9,   -9,  -4,  18,  -20, 4,   1,   -9,  6,   -6,  -12, -1,  -13, 22,  -14, 6,   -5,  -9,  -18, -31, -21, 2,
    -8,  -16, -4,  -22, 2,   -9,  5,   -24, 7,   -32, 27,  -3,  4,   6,   11,  -4,  0,   18,  20,  14,  -7,  18,  -12, 10,  13,  -2,  -42, -46, -15, 28,  -4,
    -14, -10, 18,  13,  -6,  -32, -25, 4,   -8,  -11, -4,  1,   -44, -33, 8,   26,  11,  -8,  4,   -11, -12, 6,   -13, 3,   -19, 13,  1,   47,  -32, -20, 8,
    0,   -6,  1,   -4,  12,  -23, 6,   16,  11,  17,  -5,  7,   29,  -14, -3,  -42, 1,   13,  22,  -7,  -11, -16, 14,  -19, 0,   -3,  15,  3,   -5,  0,   -3,
    -9,  -13, 2,   18,  14,  -22, 3,   13,  -6,  1,   9,   14,  17,  3,   14,  10,  4,   -12, -18, -6,  7,   -2,  -9,  -14, -9,  4,   14,  -5,  8,   -27, -4,
    -2,  5,   -1,  -17, 10,  3,   12,  -1,  3,   18,  19,  -6,  8,   -7,  -2,  5,   2,   20,  -16, -14, 7,   -5,  5,   5,   6,   10,  16,  1,   -19, 1,   -4,
    13,  -1,  -15, 5,   -12, -29, -4,  0,   -2,  7,   23,  -7,  -7,  -5,  18,  -34, -28, -20, 6,   -20, -7,  14,  -12, 1,   -2,  6,   6,   -23, 3,   8,   7,
    3,   -14, -6,  -9,  -7,  19,  5,   10,  -7,  -17, -4,  -27, -6,  -20, 2,   -13, 20,  18,  8,   -3,  -35, -10, 8,   17,  18,  -48, 21,  1,   11,  24,  -5,
    -3,  -4,  9,   21,  -14, 9,   -30, 2,   5,   12,  -2,  -21, 4,   -20, 3,   -9,  6,   7,   19,  2,   -12, 7,   -16, 19,  9,   -17, 7,   20,  4,   -9,  12,
    17,  -20, 14,  29,  -40, -7,  -4,  0,   9,   -8,  -35, -6,  -4,  6,   14,  -13, -21, -15, 16,  24,  -48, 9,   2,   18,  1,   -46, 13,  16,  14,  -3,  2,
    -32, -9,  18,  -14, -2,  -24, -16, -16, -16, 4,   16,  -8,  -6,  17,  5,   1,   -22, -18, 4,   8,   -14, 14,  7,   -2,  -13, 15,  4,   3,   3,   19,  11,
    0,   -1,  -11, -14, -16, -10, -20, -2,  8,   3,   4,   -26, 1,   -23, -10, -9};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer1_bias_tensor[128] = {
    13316712, 12782152, -1081456, -1019776,  -1778440,  7323472,  -2938024,  6316032,   -2056000, 664088,   -13279704, -3754256, 2970920,  -10349904, 15899048,
    182984,   13491472, -1301448, -2512432,  6731344,   -28784,   2635792,   -24912552, 11937136, 11713032, 3560992,   10171032, -2644016, 5867824,   -6996568,
    2631680,  -5209904, -5376440, 15878488,  -10933808, 8505672,  -14079488, -3324552,  -8657816, 929312,   -2635792,  2999704,  8910704,  20362624,  9698152,
    -9599464, 24565088, -1669472, -10808392, 12599168,  -240552,  -11028384, 6764240,   5797920,  -596240,  -7650376,  14206960, -4356664, 542784,    -14128832,
    7054136,  4858328,  5773248,  -318680,   3316328,   -1480320, -7747008,  -8226056,  9229384,  -182984,  -4377224,  5645776,  1455648,  -162424,   9321904,
    359800,   6971896,  3046992,  629136,    1163696,   11692472, -1472096,  -207656,   2382904,  10837176, 12089280,  -2113568, 1661248,  -555120,   2619344,
    -6116600, -579792,  -3435576, 12231144,  937536,    -261112,  -3717248,  -3859112,  6155664,  1060896,  -19176312, 5059816,  3096336,  41120,     4957016,
    17665152, 11423136, 5263360,  -28784,    -4480024,  27316016, 18717824,  -15311032, 8849024,  2308888,  -12761592, 5810256,  -7196000, 571568,    941648,
    11791160, -4044152, -3610336, 7601032,   10970816,  3783040,  6188560,   1685920};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer1_mul_tensor[128] = {
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056};

static PI_L2 int8_t AnomalyDetection_heavy_6_layer2_weight_tensor[16384] = {
    -22, -24, 28,  -24, 2,   12,  -4,  -19, 1,   -8,  -15, 2,   -52, -50, -7,  -14, -22, -35, -28, -1,  2,   5,   17,  -46, 11,  41,  2,   -54, 19,  20,  2,
    -13, -15, 41,  -27, -7,  -8,  -31, -25, -23, -19, 8,   -11, 13,  4,   -5,  7,   -16, -17, 1,   -21, 15,  -31, -2,  -3,  -3,  -21, 7,   -28, -6,  -6,  0,
    2,   21,  -22, 16,  2,   16,  -14, -34, -32, -30, -36, -2,  -23, -9,  -4,  0,   -5,  3,   3,   -22, 2,   -5,  4,   -12, -2,  29,  1,   33,  2,   -28, -49,
    -15, -35, 54,  -21, -41, 25,  -38, -31, 25,  0,   -10, -4,  -35, 20,  -8,  -26, -11, -1,  1,   -63, 32,  -28, 16,  -33, -42, 28,  7,   -52, 4,   6,   13,
    -32, 26,  -6,  -38, -12, 22,  -13, 22,  -3,  -13, -5,  1,   1,   12,  -14, -3,  3,   1,   4,   -13, 19,  -12, 10,  12,  -28, 14,  -52, -18, 4,   28,  25,
    23,  -7,  -4,  -28, 3,   10,  0,   -12, 11,  -14, 1,   -5,  -26, 5,   7,   -5,  2,   8,   -20, -14, -14, 0,   10,  0,   19,  -1,  11,  12,  -5,  -22, 11,
    35,  -41, 26,  27,  9,   -21, -5,  -2,  31,  -27, 16,  14,  -22, 13,  -5,  -5,  6,   -7,  -12, 4,   30,  1,   -3,  -16, 4,   -5,  5,   1,   -12, -1,  -8,
    -3,  9,   16,  2,   -5,  -51, -9,  2,   20,  -1,  -1,  0,   -8,  -6,  5,   -53, 9,   -5,  -1,  25,  -25, 9,   3,   -15, 26,  -20, -13, -1,  -34, -5,  6,
    16,  7,   -28, -6,  -17, 0,   20,  8,   14,  2,   5,   4,   10,  12,  16,  8,   7,   -20, 9,   -21, -13, 2,   -2,  6,   -25, 6,   2,   -18, 6,   1,   -25,
    -9,  -37, 10,  17,  -27, 18,  0,   5,   2,   -9,  -15, -16, -11, 23,  7,   21,  -8,  -1,  29,  0,   -21, -8,  -25, -7,  5,   21,  15,  -9,  -32, 11,  11,
    -6,  9,   4,   -3,  -7,  -18, 6,   15,  -28, 7,   -5,  -9,  -3,  -18, -17, 11,  16,  4,   -1,  3,   -29, -19, -3,  22,  12,  13,  -5,  13,  12,  9,   6,
    -10, -4,  29,  -1,  -11, -26, 20,  -31, -8,  -20, -36, -19, 9,   -25, -13, 7,   34,  -22, 10,  3,   13,  -30, 23,  2,   9,   -14, -28, 5,   20,  30,  10,
    -13, 6,   -7,  -15, 23,  16,  31,  2,   -27, -2,  13,  -40, -7,  -12, 13,  21,  7,   -7,  -2,  -22, 35,  -5,  1,   -16, -26, 11,  8,   1,   -12, -8,  8,
    4,   -5,  22,  9,   -21, -11, 5,   -17, 2,   13,  23,  20,  4,   10,  0,   18,  -16, 1,   5,   -6,  -8,  25,  10,  10,  8,   -5,  3,   -13, 2,   -33, -10,
    -10, -8,  -3,  -1,  6,   -3,  -14, -2,  9,   -3,  2,   -5,  13,  -7,  -8,  9,   -19, 18,  -10, 6,   6,   -34, -13, 8,   9,   2,   -15, -8,  -30, -30, -15,
    17,  14,  3,   -10, -14, 13,  -12, 8,   22,  -9,  25,  23,  -28, 16,  32,  -10, -10, 1,   -15, -8,  13,  6,   -4,  -26, -10, 15,  -29, 14,  -26, 6,   6,
    12,  14,  -7,  9,   4,   -8,  21,  16,  -44, -6,  -1,  -10, 7,   23,  -14, 6,   -8,  10,  -10, -4,  4,   18,  3,   17,  -17, -2,  -2,  -10, 15,  -25, 3,
    -20, 1,   -4,  -10, -5,  -21, 16,  -1,  11,  -6,  23,  19,  13,  -18, -4,  -6,  5,   5,   13,  -12, -5,  -10, -14, 4,   -15, -5,  -5,  -13, 7,   14,  7,
    2,   -1,  2,   5,   18,  -10, 8,   13,  -4,  9,   -2,  -15, 20,  -12, 7,   -14, 12,  14,  -3,  12,  4,   -24, 3,   1,   -8,  31,  -1,  -15, 9,   -15, -7,
    12,  3,   7,   6,   6,   17,  5,   -2,  1,   0,   -15, -9,  -7,  -10, 29,  17,  -5,  -8,  -11, -19, 8,   31,  -14, -10, -9,  -6,  7,   -22, 17,  11,  -2,
    22,  1,   11,  16,  0,   11,  2,   -9,  12,  3,   3,   10,  21,  13,  6,   -1,  4,   -31, 10,  1,   -30, -38, 32,  5,   14,  -2,  12,  31,  -5,  8,   -20,
    -26, -1,  -17, -12, -16, -45, 7,   -8,  31,  -45, -47, -9,  -5,  -14, -9,  -66, 34,  -26, 20,  -19, 23,  -15, -21, 4,   -18, 31,  10,  -8,  4,   -10, -14,
    -35, -6,  -4,  -29, -13, -80, -13, -26, -10, 26,  28,  -15, -9,  -1,  -15, 27,  18,  17,  22,  -19, -27, 9,   19,  -14, -4,  -10, -19, -9,  -9,  -31, -10,
    2,   -10, 58,  -25, -43, -15, -13, 3,   1,   -2,  6,   4,   7,   -6,  8,   -9,  -6,  13,  -34, -20, -7,  -1,  -4,  3,   40,  10,  6,   -16, -7,  -1,  2,
    -7,  -25, 5,   29,  25,  -20, -11, 33,  -1,  -17, -6,  13,  -4,  -5,  3,   -9,  6,   28,  24,  6,   40,  -57, 17,  4,   -21, 8,   -14, -26, 4,   10,  2,
    9,   0,   15,  -7,  8,   16,  -10, 14,  -22, 0,   -3,  -11, 12,  6,   17,  -3,  12,  9,   13,  4,   -3,  11,  19,  -13, -13, -2,  14,  2,   12,  -33, 18,
    7,   14,  3,   -3,  1,   10,  6,   -1,  4,   11,  7,   3,   8,   -13, 17,  11,  6,   -17, 1,   15,  22,  5,   -11, -5,  14,  -15, -11, 1,   14,  -10, 5,
    -10, -23, 8,   1,   -6,  -4,  -20, 19,  4,   11,  -8,  13,  -13, 9,   -10, 14,  4,   -12, -8,  4,   -8,  -13, 19,  -11, 5,   15,  -17, -7,  13,  4,   -6,
    -8,  -2,  5,   -6,  -10, 0,   8,   16,  -10, -24, -5,  10,  -21, 11,  1,   -21, 14,  -21, -10, 6,   21,  5,   8,   6,   -18, -27, 22,  12,  12,  -62, -13,
    18,  -8,  -36, 18,  -2,  6,   -26, 2,   -14, -23, 13,  -21, -14, 10,  3,   12,  25,  6,   10,  -42, -23, 29,  -31, -18, -16, -7,  3,   -10, -29, 7,   -22,
    -1,  3,   0,   -15, 10,  -16, -4,  -40, -8,  -3,  -2,  1,   -13, -4,  4,   -5,  -12, -15, -18, 6,   6,   -18, -51, 17,  22,  -27, -10, -5,  11,  16,  6,
    8,   21,  30,  -20, 5,   12,  -24, 10,  -1,  3,   -24, 4,   -45, -39, -13, -18, 5,   -9,  -10, 9,   -13, 10,  -16, -12, 54,  7,   -10, 29,  15,  -1,  50,
    -22, -12, -15, 12,  5,   -16, 10,  10,  5,   5,   26,  -7,  -43, 7,   3,   42,  1,   13,  9,   -19, -27, -15, -29, -19, 16,  -6,  -7,  15,  37,  8,   13,
    28,  23,  -5,  4,   22,  -16, 1,   0,   5,   10,  -27, 9,   3,   -24, -2,  24,  8,   -16, 3,   -4,  3,   -36, -16, -29, 7,   -21, 42,  35,  -26, -24, -19,
    -16, -2,  -10, 7,   -19, -33, 20,  2,   8,   10,  -26, 2,   -25, -46, 17,  8,   6,   -20, 8,   8,   6,   -9,  7,   15,  -9,  5,   -11, 7,   1,   0,   -4,
    -5,  -3,  16,  14,  3,   -19, -23, 18,  -19, 16,  29,  -2,  13,  -4,  -17, 7,   38,  1,   12,  -7,  17,  -10, -2,  6,   -48, -2,  8,   1,   11,  7,   27,
    -53, 16,  -34, -20, -10, 19,  -21, -15, 11,  7,   -3,  -12, -8,  -2,  -21, 25,  12,  5,   5,   -46, 11,  16,  -4,  0,   4,   14,  7,   -11, 9,   -3,  4,
    7,   -26, -22, 0,   -22, 15,  -15, 19,  33,  -36, 36,  -18, -51, -13, 1,   4,   -20, 14,  -10, -17, -23, -14, -43, 7,   11,  11,  -15, 6,   -15, -45, 9,
    -42, -8,  -11, -28, -13, 0,   4,   9,   18,  11,  6,   3,   4,   -4,  12,  6,   -8,  14,  -31, 7,   22,  -32, -7,  -8,  -16, -32, 1,   -2,  14,  -13, 9,
    -65, 5,   26,  41,  14,  -12, -13, -40, -3,  -20, 5,   6,   17,  -30, -57, -49, -38, 34,  -7,  -5,  -64, -12, 9,   -27, 21,  -61, 3,   -10, -33, 4,   -10,
    7,   4,   -55, -25, 54,  6,   -13, 21,  -4,  -62, -37, 11,  -11, -48, -1,  -2,  6,   -34, -25, -13, -30, 15,  -4,  -11, -2,  -14, 22,  11,  23,  -35, 32,
    1,   -10, 2,   17,  -4,  19,  -38, -19, -38, -4,  12,  16,  4,   -1,  -15, 1,   21,  -35, -10, -10, -8,  -15, -16, -1,  -9,  -17, -13, 11,  3,   12,  0,
    -8,  15,  -6,  19,  16,  21,  0,   -9,  -13, 13,  7,   15,  -1,  1,   -6,  -47, 6,   -20, -22, -9,  -2,  20,  7,   -13, 2,   1,   -4,  -4,  1,   14,  -8,
    4,   -6,  7,   -6,  -14, 5,   1,   0,   -15, -14, 13,  -1,  -2,  12,  16,  -9,  -10, 1,   0,   8,   -6,  18,  -9,  -16, 5,   8,   -2,  1,   -9,  20,  -8,
    1,   -1,  12,  14,  -3,  5,   6,   34,  -4,  16,  -26, -13, -36, -2,  9,   7,   4,   11,  3,   -3,  16,  9,   -19, 18,  -30, -8,  1,   24,  12,  11,  6,
    -7,  -6,  17,  3,   5,   5,   -9,  18,  8,   9,   -14, 9,   -2,  3,   22,  18,  20,  -7,  2,   -19, 5,   -31, -37, 10,  -3,  -1,  4,   -47, 4,   -38, 4,
    -13, -20, -3,  1,   -6,  30,  7,   31,  18,  -20, -14, 21,  -16, -43, 12,  3,   -46, -10, 21,  -39, 10,  -19, -14, -2,  -3,  10,  8,   -17, -39, -21, 16,
    -14, -27, -11, 13,  26,  -4,  -2,  -21, 3,   -2,  2,   30,  -9,  20,  4,   -5,  -6,  20,  -11, 25,  -47, 3,   38,  -25, -21, -29, 6,   9,   5,   16,  3,
    -9,  -18, 11,  2,   -26, -32, 6,   50,  0,   6,   -44, 15,  -38, -24, 0,   -8,  -11, 30,  -57, -12, 2,   -30, -17, 2,   24,  35,  12,  -9,  14,  14,  -4,
    -12, 7,   -16, 5,   -45, -14, 6,   12,  16,  54,  -8,  -19, 0,   5,   -3,  -13, 0,   1,   6,   5,   25,  -23, -5,  15,  0,   -4,  4,   11,  -2,  -19, 11,
    12,  -9,  -15, 7,   -3,  -9,  29,  -2,  20,  -6,  -5,  -9,  -9,  17,  15,  6,   -16, 10,  1,   4,   6,   -14, 35,  5,   -10, 7,   18,  1,   19,  -26, -14,
    -1,  -6,  8,   -15, 5,   20,  -5,  -2,  -6,  7,   3,   5,   -13, 8,   21,  -15, -20, 9,   7,   -8,  5,   37,  -37, 6,   8,   -2,  -11, 3,   7,   21,  -1,
    -11, 2,   -20, 16,  -3,  11,  -27, 8,   -3,  -2,  1,   -7,  -5,  -17, -15, -12, -26, -8,  -15, -45, 12,  4,   -5,  22,  2,   8,   7,   6,   -13, 6,   -5,
    9,   -3,  -3,  8,   14,  0,   -19, 6,   -9,  -20, 5,   -24, 5,   -3,  -37, -6,  -20, 19,  -32, -25, -26, 10,  7,   -8,  -13, -15, 1,   -63, 8,   1,   6,
    -8,  -10, 6,   2,   -3,  -12, 19,  1,   -12, 2,   8,   17,  -21, 8,   17,  27,  28,  22,  8,   -4,  -22, 10,  2,   -6,  -10, -36, 1,   3,   -14, 10,  9,
    3,   -6,  -1,  14,  0,   11,  -17, -3,  10,  7,   -24, 14,  22,  10,  -14, -15, 7,   -1,  4,   -7,  24,  21,  -35, 8,   17,  42,  -12, 18,  -6,  -14, 42,
    -10, -4,  7,   27,  -3,  5,   20,  6,   -2,  -25, -1,  -12, 7,   -4,  -10, 14,  0,   71,  -5,  7,   42,  -7,  31,  22,  0,   21,  23,  -20, -30, -5,  11,
    2,   -22, -8,  10,  -7,  26,  -8,  -9,  19,  -18, -22, -6,  -29, -4,  -25, -8,  2,   2,   -19, -56, 12,  24,  -33, 19,  5,   7,   9,   12,  15,  -15, 12,
    18,  2,   -12, -9,  5,   14,  13,  -21, -26, 10,  -13, -7,  9,   29,  -1,  -7,  -6,  11,  -21, 11,  -24, -4,  -7,  -34, -4,  -6,  -4,  -4,  20,  7,   -10,
    14,  2,   14,  -3,  3,   4,   33,  -9,  1,   10,  -8,  2,   0,   -8,  -5,  -8,  20,  -7,  10,  -8,  13,  44,  -14, 34,  24,  7,   14,  0,   -5,  1,   10,
    -3,  7,   -5,  2,   12,  -1,  4,   26,  14,  0,   -1,  13,  10,  -3,  -9,  -4,  -4,  9,   -11, 3,   -9,  12,  -7,  6,   7,   13,  -18, 16,  -3,  2,   -2,
    5,   -7,  1,   -10, -5,  4,   5,   -4,  8,   4,   19,  21,  18,  3,   20,  20,  9,   -21, 1,   -4,  2,   -21, 9,   14,  -5,  8,   0,   2,   22,  24,  14,
    8,   6,   -3,  8,   -3,  26,  -7,  -5,  -9,  -38, -4,  -6,  -1,  11,  -4,  -9,  -13, 24,  5,   -6,  1,   34,  -49, 33,  2,   -11, -21, -18, -12, 5,   -9,
    4,   -5,  -19, 7,   7,   6,   -15, 23,  9,   -1,  -16, 16,  -23, -18, -21, 4,   12,  -20, 12,  -16, 2,   6,   2,   1,   14,  5,   5,   22,  20,  -1,  6,
    -3,  2,   -10, -16, 2,   -2,  -16, 19,  5,   -9,  13,  -26, -16, -3,  10,  7,   7,   7,   -8,  4,   5,   -5,  5,   26,  -10, -11, -3,  41,  -2,  -2,  -6,
    -38, -30, 31,  18,  -5,  10,  21,  -13, -9,  -4,  8,   -15, 21,  -7,  1,   -19, 7,   6,   25,  18,  5,   4,   21,  -24, -16, 10,  -4,  40,  15,  -29, -7,
    15,  8,   1,   9,   -9,  0,   -10, 6,   -13, -26, 18,  -12, -12, -34, 3,   -11, 31,  24,  5,   10,  2,   -26, -11, -36, 6,   14,  13,  5,   -2,  22,  0,
    -2,  10,  -9,  -16, 8,   -11, -47, -20, 0,   -4,  10,  -19, -11, 5,   1,   -8,  -25, 18,  0,   -17, 18,  -19, 10,  -10, -24, -14, 6,   2,   -3,  24,  -7,
    26,  -11, 30,  -7,  16,  -26, 20,  54,  1,   -36, 6,   31,  21,  35,  -8,  6,   -7,  21,  20,  15,  13,  9,   5,   0,   -15, 11,  14,  31,  1,   -6,  -7,
    -20, 24,  -33, 8,   -18, -18, -6,  13,  -40, -35, 5,   5,   13,  17,  -14, 14,  29,  42,  -3,  -16, 19,  -29, 34,  32,  3,   -8,  3,   17,  -13, -8,  -20,
    25,  7,   -47, -20, 8,   17,  5,   -9,  -9,  -8,  0,   -19, 3,   -21, 17,  -10, -1,  7,   2,   3,   -6,  10,  1,   9,   -13, 2,   6,   -3,  16,  -8,  7,
    -4,  -20, 0,   3,   14,  -10, -14, 10,  -16, -5,  -7,  -10, 0,   -7,  11,  -1,  6,   19,  1,   12,  8,   0,   -2,  -3,  -11, 9,   -6,  -21, -5,  7,   0,
    -10, 17,  7,   -3,  -5,  7,   18,  10,  -3,  17,  16,  15,  3,   9,   12,  -7,  -11, -2,  11,  11,  -7,  -7,  7,   -7,  8,   1,   -6,  -7,  0,   1,   -4,
    -9,  -3,  23,  12,  22,  10,  -21, -1,  18,  12,  -12, -1,  23,  2,   11,  11,  -5,  -1,  -12, 16,  16,  0,   9,   11,  -8,  14,  -15, 0,   10,  -9,  -16,
    13,  12,  -1,  -7,  -14, -11, 35,  5,   -2,  -5,  41,  1,   -6,  6,   18,  -20, -11, -4,  -46, 1,   11,  -10, 12,  -24, 40,  -12, 30,  -6,  -7,  35,  29,
    1,   -50, 14,  30,  21,  -15, -4,  5,   -4,  -19, -7,  8,   2,   0,   14,  -14, -15, 5,   -4,  5,   1,   2,   26,  12,  -13, 15,  2,   -25, -6,  27,  -8,
    9,   0,   28,  6,   2,   19,  -13, 1,   -4,  3,   -32, -18, 22,  -19, -21, -43, -6,  6,   -10, -8,  9,   -19, 18,  46,  13,  12,  9,   18,  16,  -18, 16,
    10,  2,   -2,  -12, -2,  -5,  43,  -11, 11,  23,  -25, 4,   9,   -3,  20,  -1,  -1,  -16, 22,  13,  10,  -13, -8,  -17, -2,  43,  -18, -8,  -6,  25,  3,
    14,  -6,  4,   -11, 10,  8,   6,   9,   48,  34,  17,  1,   -11, 33,  -10, 3,   -15, -12, -5,  -3,  -12, -8,  -9,  8,   0,   -4,  3,   -12, -9,  -16, 8,
    -10, -8,  -5,  0,   -9,  -7,  2,   4,   -1,  -1,  -8,  2,   -9,  -12, -12, 7,   2,   8,   -5,  -2,  -2,  -14, -8,  5,   2,   -12, -2,  -8,  -6,  3,   -2,
    0,   -5,  -2,  -8,  3,   0,   -12, -9,  2,   6,   9,   6,   2,   -13, -3,  -14, -11, -8,  -5,  -4,  -9,  2,   2,   -3,  3,   1,   -13, -1,  -8,  -10, -13,
    -4,  -7,  -1,  -12, 1,   -13, 2,   -11, -6,  3,   -8,  -5,  7,   -2,  -4,  -13, -1,  -5,  -8,  -1,  -11, -1,  -2,  4,   2,   7,   -10, 8,   3,   -13, 0,
    -7,  -3,  -6,  -6,  -3,  -4,  -9,  -2,  -1,  -1,  -9,  -9,  -13, -5,  2,   -8,  10,  -8,  6,   12,  17,  -40, -16, 13,  -23, -14, 17,  -16, -29, 6,   -35,
    -20, 29,  -8,  -14, 4,   -2,  -16, -12, -24, -1,  19,  -5,  8,   -7,  7,   9,   20,  -20, -30, -43, 6,   -8,  23,  39,  1,   -13, 15,  -2,  16,  -4,  33,
    4,   -4,  -7,  -9,  15,  4,   -53, -3,  23,  -3,  -4,  2,   -4,  33,  20,  17,  -15, 3,   -19, -15, -5,  -38, 17,  11,  -13, 0,   1,   8,   25,  -33, -23,
    -11, 4,   21,  19,  -6,  13,  -21, 8,   6,   -1,  2,   -1,  -27, 6,   -9,  -6,  40,  -9,  -27, 16,  13,  -7,  -25, -26, -19, -18, -16, -27, -5,  -1,  -5,
    -23, -5,  10,  19,  1,   19,  -10, 1,   7,   16,  -3,  -55, -4,  13,  12,  -21, -53, 7,   2,   12,  8,   -30, -14, -4,  8,   9,   6,   -4,  8,   -8,  20,
    -4,  -11, 10,  0,   21,  -5,  17,  -19, 5,   7,   -28, 2,   -12, 33,  -25, -6,  -11, -4,  -5,  6,   13,  -1,  10,  14,  -11, -3,  7,   -1,  11,  -1,  18,
    16,  0,   15,  -20, -2,  9,   -11, -14, -1,  -3,  18,  13,  2,   -18, 14,  -1,  7,   9,   -25, -1,  -5,  -11, 2,   3,   -18, 21,  9,   26,  12,  -7,  3,
    -26, 3,   -1,  9,   0,   2,   5,   -9,  -1,  -5,  3,   -9,  -7,  -15, -9,  10,  6,   14,  5,   -10, 3,   -9,  0,   0,   8,   3,   28,  -14, 6,   -8,  -18,
    8,   -1,  2,   4,   1,   5,   10,  1,   0,   -12, 12,  15,  -16, 6,   5,   -15, -8,  20,  -17, -7,  -4,  -4,  15,  -6,  5,   1,   -33, -17, 36,  17,  19,
    -1,  -30, -32, -3,  -36, -16, -64, -43, -10, -16, -1,  -3,  -5,  17,  -40, 25,  17,  9,   -20, -70, 28,  4,   45,  24,  14,  8,   -28, -2,  -2,  1,   30,
    -51, -16, -8,  -44, 13,  14,  -45, 4,   10,  -39, 3,   -8,  6,   -4,  9,   28,  28,  18,  -39, 6,   -39, 11,  9,   1,   -44, -15, -28, 6,   8,   28,  47,
    3,   -14, 18,  -9,  -3,  -6,  -11, 4,   -18, -27, -8,  -79, -38, -12, -21, 8,   -6,  7,   -9,  2,   -5,  -3,  3,   -49, 19,  20,  9,   -23, -19, -2,  -42,
    -59, -11, -1,  -19, -7,  -9,  -7,  -28, -25, -40, -8,  -18, -38, -33, -16, -48, 4,   -28, -5,  -51, 7,   -23, 29,  -15, -30, -38, -20, 10,  -24, -54, 4,
    -2,  -23, 12,  -10, 4,   -11, -9,  2,   -4,  -8,  -8,  6,   8,   -12, -6,  -2,  -10, 13,  4,   4,   13,  -10, -20, 15,  12,  -9,  10,  -6,  13,  -5,  2,
    6,   -13, 13,  -22, -20, 19,  13,  16,  4,   -6,  7,   -6,  -22, -6,  -12, 16,  -6,  -4,  -13, -9,  5,   -5,  18,  -11, -8,  -7,  8,   4,   12,  5,   5,
    0,   -9,  8,   -6,  -16, 1,   16,  5,   -6,  5,   7,   5,   21,  -24, -11, 11,  -23, -3,  12,  -17, 8,   4,   8,   9,   -5,  7,   18,  7,   -6,  20,  -14,
    10,  19,  -4,  -8,  -26, 26,  6,   -18, 13,  7,   -9,  -18, 9,   18,  -33, 8,   6,   -7,  4,   -2,  4,   5,   7,   6,   -4,  8,   18,  -6,  -24, 4,   13,
    15,  -2,  8,   24,  5,   7,   -8,  -14, 6,   5,   3,   -15, -19, -3,  -6,  1,   -9,  20,  -2,  0,   -6,  -8,  6,   -12, -22, -11, 9,   -16, -2,  3,   8,
    -14, -17, -21, -18, 3,   26,  -28, -6,  -14, 4,   -2,  -15, -22, 11,  4,   -12, 11,  5,   21,  -5,  -1,  14,  8,   22,  10,  -23, 6,   6,   2,   -27, 15,
    -13, 18,  -20, -5,  -24, 8,   7,   -29, -30, -1,  5,   -16, 1,   25,  6,   2,   17,  -1,  1,   -26, 8,   5,   17,  14,  10,  12,  -31, 12,  -13, 10,  13,
    -27, 8,   -12, -2,  -36, -6,  -3,  4,   8,   4,   -1,  20,  14,  -2,  10,  2,   6,   -2,  17,  -9,  14,  -3,  -7,  1,   22,  8,   -5,  2,   16,  -14, 0,
    -8,  13,  -10, -24, -3,  -18, -20, 13,  -3,  -13, -13, 5,   17,  -14, -4,  -3,  12,  -6,  -14, 22,  -20, 11,  -30, 13,  10,  -43, 18,  11,  14,  9,   31,
    12,  30,  1,   17,  1,   -6,  18,  11,  -18, 1,   -1,  -11, -27, 4,   -18, -14, 6,   -4,  -1,  7,   7,   3,   6,   10,  -4,  0,   -11, 0,   -8,  18,  11,
    -3,  2,   -3,  10,  -4,  -7,  14,  -12, 10,  -16, 2,   20,  -8,  12,  1,   -10, -6,  -9,  -1,  -5,  11,  -15, -10, 6,   14,  -17, -4,  -5,  3,   7,   5,
    -14, -4,  5,   -18, 1,   0,   4,   10,  -22, 10,  -6,  16,  3,   -25, -10, -17, 4,   -2,  -28, 13,  9,   11,  0,   9,   13,  7,   5,   -2,  1,   -6,  -2,
    2,   -2,  1,   -33, -6,  12,  -3,  19,  -4,  3,   12,  -2,  -15, -12, -3,  11,  -9,  11,  -9,  18,  14,  -15, -13, 15,  -9,  -2,  -3,  -14, 3,   17,  -24,
    3,   -7,  15,  -18, -18, 12,  4,   1,   -4,  22,  -24, -30, 4,   6,   12,  -29, 0,   23,  2,   8,   -17, 12,  10,  -5,  -8,  5,   -17, 3,   -23, -6,  8,
    22,  -5,  10,  8,   7,   -5,  11,  -1,  6,   0,   -21, 13,  0,   -15, 15,  29,  30,  6,   -4,  -4,  -29, 7,   14,  12,  -2,  9,   -17, 1,   1,   5,   6,
    0,   7,   -3,  9,   5,   -6,  7,   20,  7,   12,  37,  -4,  20,  1,   10,  -4,  -12, 24,  -14, 3,   2,   3,   3,   -2,  16,  -13, -12, 14,  3,   -18, 7,
    2,   0,   10,  1,   -13, -18, 3,   9,   -1,  -26, -8,  31,  9,   9,   4,   -4,  6,   7,   30,  -7,  5,   9,   9,   4,   -2,  17,  13,  -36, -14, -6,  2,
    -16, -8,  -21, 28,  9,   -21, 22,  1,   5,   3,   20,  -26, -13, -13, -9,  13,  -13, 13,  7,   -21, 10,  19,  6,   -23, -3,  4,   -30, -31, 19,  -2,  -16,
    5,   -6,  15,  -8,  1,   0,   9,   -30, 14,  -28, 10,  -4,  -5,  7,   6,   -4,  13,  30,  -16, -37, -29, -26, -4,  21,  21,  12,  -2,  -9,  2,   18,  26,
    -1,  -12, 4,   22,  6,   5,   -8,  -19, -4,  2,   13,  -6,  -12, 33,  -15, -45, -1,  -30, 0,   18,  -9,  -1,  17,  1,   -8,  7,   -5,  23,  0,   1,   16,
    -6,  -20, -14, -2,  8,   9,   8,   3,   20,  23,  -18, 32,  6,   6,   -5,  15,  -43, 16,  -11, 7,   13,  8,   5,   6,   -28, 5,   3,   -20, 3,   6,   -40,
    -15, -1,  14,  -8,  0,   -7,  -28, 3,   19,  -28, 24,  -5,  15,  -5,  -2,  -15, -12, 17,  4,   12,  15,  -10, -3,  8,   24,  -43, 7,   -4,  13,  -20, -14,
    1,   -4,  18,  -22, 17,  -12, 5,   6,   3,   15,  5,   6,   -6,  -12, 10,  -23, 30,  15,  -12, 17,  -18, -22, 10,  20,  -49, -1,  3,   4,   9,   7,   16,
    13,  20,  -13, -14, -3,  -10, -14, -5,  -1,  -9,  -11, 6,   2,   -6,  2,   -42, 10,  -29, 34,  10,  -7,  9,   11,  -3,  -25, 4,   9,   19,  -7,  11,  -20,
    11,  -2,  50,  -19, -19, -7,  13,  -15, 2,   -8,  -15, -7,  26,  -22, 13,  -6,  -15, 19,  -21, -31, 18,  25,  -5,  -7,  6,   1,   26,  5,   -5,  -1,  -4,
    -34, -17, -24, -25, -19, 0,   10,  11,  -10, -24, -16, -36, -13, -54, -29, -18, -6,  -31, 27,  16,  -8,  -9,  -23, 21,  -2,  13,  17,  38,  45,  -17, -41,
    1,   -4,  -11, 25,  -6,  -9,  -40, 14,  -3,  24,  3,   14,  20,  12,  -18, -6,  30,  26,  -20, 10,  -3,  9,   -71, -24, -5,  4,   1,   19,  -29, 6,   -8,
    -13, -28, 16,  -11, -1,  -22, -26, 13,  -16, 19,  -4,  -18, 4,   2,   -8,  -11, 19,  -13, -6,  0,   9,   30,  -18, -42, 2,   -30, 19,  -42, -8,  -1,  -10,
    9,   11,  9,   -12, -16, -22, -12, 2,   21,  17,  22,  -1,  5,   -17, -16, 7,   -23, 5,   -40, 3,   10,  30,  -7,  -2,  5,   14,  25,  -30, -17, -21, -11,
    12,  -25, -8,  -27, 0,   14,  1,   -38, 14,  7,   -6,  11,  -11, 10,  -22, -14, 33,  -35, 0,   -36, 11,  3,   0,   -22, 19,  19,  -5,  -9,  19,  -30, -19,
    -29, 21,  35,  -6,  22,  9,   -6,  0,   19,  -4,  -35, -21, -11, -14, -16, -2,  -17, 8,   -17, 1,   13,  -14, -7,  -3,  25,  -18, -14, 8,   4,   -12, 0,
    -46, 13,  -19, -13, 4,   22,  -2,  6,   6,   -1,  1,   0,   -15, 9,   12,  0,   -7,  -10, -3,  10,  9,   -38, 8,   29,  -16, -11, 15,  -9,  21,  39,  9,
    -1,  -23, -7,  12,  5,   -8,  -13, 6,   -29, 16,  -24, -30, 16,  4,   -12, 1,   13,  -22, -18, 1,   8,   0,   9,   -22, -12, -22, -1,  37,  12,  -1,  -38,
    -1,  10,  -9,  -9,  -13, 5,   -21, 21,  -8,  -10, -4,  -13, -4,  1,   -9,  -15, 0,   -8,  -23, 10,  2,   18,  0,   14,  12,  2,   25,  11,  -6,  -6,  -1,
    8,   -14, 16,  -6,  3,   -12, -8,  6,   4,   1,   13,  7,   1,   7,   -11, 16,  3,   -8,  3,   -1,  0,   -6,  22,  -7,  3,   -13, -13, 17,  -5,  28,  6,
    14,  11,  6,   -4,  -13, -5,  1,   2,   -8,  14,  -4,  16,  16,  -19, -4,  10,  15,  7,   11,  17,  16,  11,  0,   -8,  -20, 0,   12,  -1,  3,   8,   15,
    -2,  -2,  -14, 0,   -17, 7,   -20, 2,   0,   -1,  12,  -3,  13,  10,  12,  1,   4,   -5,  10,  -2,  4,   -19, -9,  -17, 17,  9,   0,   14,  15,  7,   3,
    -6,  -25, 20,  21,  6,   8,   7,   -14, 4,   -8,  -39, 1,   -45, -4,  -15, -39, 2,   -60, 2,   -3,  -5,  -2,  -12, 37,  17,  12,  6,   22,  24,  -3,  4,
    -40, 18,  -7,  9,   -46, -16, -9,  -25, -5,  22,  -16, 3,   14,  -56, 0,   22,  0,   19,  12,  -3,  2,   -6,  13,  2,   26,  8,   -2,  -7,  2,   6,   -20,
    15,  -3,  48,  12,  -3,  -14, 13,  -21, -11, -26, -21, -12, -14, 11,  -13, -13, -5,  37,  22,  13,  -3,  5,   17,  8,   -58, -4,  19,  8,   -22, 5,   10,
    9,   6,   -27, -12, -1,  -9,  -16, 14,  9,   -20, 11,  -24, -1,  -10, 9,   -17, 6,   0,   -3,  -1,  0,   12,  16,  -34, 1,   -1,  -3,  -14, 0,   8,   3,
    25,  -2,  11,  9,   -6,  -29, 9,   -28, -5,  -11, 1,   0,   -5,  -19, -21, -1,  -7,  -2,  -12, -9,  -7,  -1,  -4,  -2,  -4,  -19, -4,  -15, 8,   -1,  -8,
    -5,  -18, 1,   -8,  1,   0,   -12, -12, -5,  -7,  -12, -9,  4,   -9,  -4,  -16, -6,  -2,  1,   7,   3,   -6,  -11, -23, -21, 0,   -5,  -11, 11,  12,  -12,
    -16, 3,   -20, -3,  3,   -12, -1,  -8,  -15, -2,  -11, -26, 3,   -5,  -11, -21, -11, -12, 0,   -17, 5,   -5,  2,   -5,  2,   -18, -20, -13, -15, 12,  -19,
    0,   2,   -3,  9,   -5,  4,   -16, 16,  3,   10,  -3,  -12, -17, -20, 7,   1,   0,   -19, -2,  2,   -14, -12, -2,  -17, -8,  -18, -20, -14, 1,   -18, -5,
    0,   -7,  -23, -8,  2,   -13, -16, -16, 15,  -10, 8,   9,   -26, -12, 23,  -17, -20, 12,  6,   -9,  26,  19,  -9,  7,   10,  13,  -14, 16,  -8,  -14, 0,
    -8,  0,   -2,  37,  -9,  -9,  3,   -32, -21, -12, -5,  -16, 17,  0,   6,   14,  -27, -27, 1,   -12, 25,  2,   -15, 9,   -16, 4,   -22, 0,   13,  15,  6,
    -8,  27,  -53, 0,   1,   -1,  7,   -38, 9,   21,  12,  -3,  -1,  5,   11,  12,  8,   31,  4,   12,  4,   28,  -17, -19, -17, 4,   14,  -11, -4,  5,   11,
    11,  -24, 6,   -3,  0,   -4,  -7,  6,   19,  -5,  -34, 8,   21,  9,   -11, -4,  -29, 13,  -17, -13, 0,   -29, 8,   -17, 23,  -1,  -14, -13, -64, 6,   9,
    0,   -31, 9,   10,  17,  -11, -9,  -9,  8,   -12, -8,  -11, -28, 0,   6,   -2,  14,  12,  10,  0,   -2,  -23, -1,  -1,  6,   0,   11,  15,  4,   4,   4,
    4,   2,   -4,  13,  2,   -4,  -7,  10,  -3,  -7,  -24, -27, -10, -15, 6,   -14, 22,  -1,  -6,  -6,  -1,  10,  -8,  14,  10,  14,  0,   -10, -15, 10,  13,
    0,   11,  -15, -20, 1,   9,   -20, 6,   17,  -10, 17,  5,   -2,  -23, 14,  1,   9,   3,   -9,  -3,  21,  42,  19,  -2,  7,   -9,  11,  -4,  -33, 6,   1,
    11,  -24, 3,   10,  3,   -18, 3,   9,   -4,  -2,  -3,  -9,  -2,  16,  -9,  17,  8,   9,   -12, -13, 6,   -24, 22,  0,   -2,  5,   -13, 4,   6,   -5,  3,
    17,  4,   -4,  8,   8,   -3,  -10, 6,   15,  8,   -36, -4,  12,  2,   -13, -21, -13, 10,  -7,  10,  28,  8,   15,  -21, -7,  -17, -26, -1,  30,  4,   11,
    -8,  -38, -31, -2,  -20, -12, -5,  15,  3,   4,   19,  4,   -23, 24,  -16, -22, 18,  4,   -19, -4,  -2,  5,   -27, -1,  -13, -12, -6,  -6,  28,  -5,  -10,
    14,  -34, 4,   -7,  16,  -6,  1,   18,  -12, -37, 15,  -47, 23,  -1,  15,  8,   -3,  4,   14,  -20, -8,  -15, -7,  -19, -27, 4,   10,  -8,  9,   5,   -6,
    24,  8,   -6,  -13, 16,  -37, -1,  -9,  -15, 7,   -9,  -19, 7,   -32, -19, 12,  7,   -5,  -47, -20, -5,  -5,  21,  -40, -71, -26, -16, -29, 9,   -7,  10,
    10,  14,  19,  13,  -3,  -11, -13, -2,  -14, -17, -1,  21,  -26, 14,  -27, -3,  -37, -16, -10, 11,  65,  18,  9,   14,  -7,  0,   1,   -2,  13,  -12, -22,
    21,  16,  32,  -7,  18,  -12, -1,  -1,  0,   19,  7,   11,  17,  -17, 1,   24,  -6,  -11, -21, -5,  -2,  17,  14,  -5,  4,   9,   -9,  22,  -13, 25,  0,
    4,   -1,  -4,  21,  2,   -2,  -2,  10,  -24, -12, -46, -16, 19,  -41, -5,  18,  8,   0,   -19, 8,   -26, -41, -4,  7,   10,  -5,  0,   5,   12,  29,  -2,
    -9,  -7,  -11, 20,  16,  9,   11,  -26, 4,   -35, 18,  -10, -12, -3,  2,   -4,  -15, -11, -30, 5,   5,   -7,  2,   26,  14,  -30, 12,  14,  10,  -32, 9,
    5,   4,   0,   -2,  1,   9,   6,   -14, 18,  17,  8,   23,  2,   -2,  -13, 17,  -12, -5,  -15, -9,  -20, 9,   -24, -3,  38,  39,  -3,  12,  -12, 3,   10,
    11,  19,  14,  -23, -7,  12,  -8,  -8,  10,  -14, -7,  21,  -2,  -11, -20, -2,  8,   -34, 29,  20,  -21, -10, -37, -13, 11,  5,   4,   1,   8,   17,  15,
    -6,  -28, -49, -24, -19, 4,   -9,  16,  6,   -21, -19, 0,   1,   -6,  -4,  3,   15,  11,  -38, -21, 2,   -4,  -4,  7,   1,   -9,  17,  9,   8,   6,   11,
    -4,  -39, -42, 5,   12,  4,   -4,  19,  -13, 3,   11,  -4,  8,   -31, -8,  -4,  12,  -38, 7,   -17, -9,  13,  4,   2,   0,   -3,  19,  14,  25,  13,  2,
    23,  23,  -3,  -27, 20,  -11, 6,   12,  -2,  -27, 19,  -4,  10,  4,   -5,  -4,  -12, 13,  15,  6,   3,   -37, 10,  1,   -7,  6,   -47, -21, 10,  -88, 22,
    7,   3,   -13, -3,  3,   5,   6,   -7,  2,   -32, -16, -3,  8,   -13, -28, 17,  -6,  7,   -2,  16,  5,   0,   7,   -23, -4,  5,   24,  21,  5,   16,  2,
    13,  17,  0,   20,  16,  -5,  21,  0,   -11, 9,   -22, 3,   27,  -3,  -33, -5,  5,   5,   -22, -16, 8,   13,  5,   15,  -7,  6,   0,   -11, 12,  -7,  1,
    -32, -8,  -5,  2,   -14, 8,   18,  -20, -2,  14,  22,  1,   9,   7,   5,   2,   -4,  -3,  19,  6,   -12, -1,  -5,  -10, -4,  -9,  -6,  11,  -1,  -8,  -17,
    -15, 26,  -13, -34, 6,   -19, 4,   -49, 2,   9,   7,   -7,  13,  -2,  0,   -8,  -13, -4,  -2,  -8,  -2,  11,  10,  16,  -37, 20,  -6,  2,   2,   -10, 12,
    -15, -10, 12,  15,  13,  -21, -17, -21, 6,   15,  5,   27,  36,  -42, -33, 19,  -62, -4,  -1,  -17, 5,   -24, -52, -19, 21,  1,   -1,  9,   -20, -50, 28,
    -17, 4,   -6,  17,  -20, -11, 7,   2,   -4,  -35, -24, -7,  -3,  26,  6,   -9,  -2,  -35, -21, -4,  -11, -5,  13,  -34, -25, 0,   -19, 1,   -61, 8,   31,
    18,  24,  -1,  -23, -18, -5,  -9,  -42, -6,  6,   10,  -38, 13,  -5,  -9,  12,  -29, 6,   -21, -9,  8,   -53, 12,  -7,  -4,  -42, -11, 38,  -34, -5,  -23,
    5,   7,   8,   25,  24,  14,  -11, -2,  2,   4,   4,   29,  -26, 6,   -26, 7,   -31, -6,  -32, -14, 15,  1,   -17, -20, -5,  -33, 13,  -20, -13, 5,   37,
    -7,  -31, -2,  -10, 6,   18,  -43, 9,   11,  -32, -21, 0,   1,   -3,  -10, 8,   -26, 8,   9,   -1,  -7,  -1,  18,  -11, 0,   -9,  2,   22,  -16, 28,  -14,
    15,  5,   17,  8,   -4,  14,  13,  -2,  -29, -12, 1,   8,   9,   6,   8,   2,   16,  12,  5,   -15, 1,   -32, -12, 12,  -9,  -5,  -20, -12, 24,  -2,  -31,
    8,   8,   11,  -25, 9,   5,   -13, 1,   -38, 10,  7,   3,   -7,  2,   9,   25,  19,  7,   -16, 14,  4,   -2,  3,   -11, -7,  22,  17,  0,   2,   -17, 8,
    12,  -4,  12,  3,   3,   34,  -1,  2,   12,  -10, -24, 3,   10,  -18, 0,   7,   -14, 6,   -3,  1,   -18, 6,   -8,  -5,  13,  -2,  13,  11,  -12, -9,  5,
    -2,  -3,  -10, -17, 8,   -13, -26, -8,  -1,  27,  -4,  4,   -4,  -11, -3,  -6,  6,   -7,  -2,  4,   -6,  -3,  5,   -3,  -19, 4,   7,   -26, 12,  2,   12,
    -8,  -18, 10,  -12, 14,  9,   10,  16,  8,   -26, 2,   7,   -2,  -22, 14,  -14, 5,   -14, 11,  30,  10,  25,  16,  -12, 1,   13,  4,   5,   10,  -7,  -5,
    -5,  10,  30,  -18, -10, 9,   16,  -13, -2,  -23, 15,  -7,  -17, 9,   -1,  -6,  17,  -34, -9,  -2,  -6,  7,   -9,  -8,  13,  6,   20,  -15, -11, -5,  -8,
    1,   1,   -25, -9,  7,   0,   -22, 24,  1,   22,  -5,  10,  28,  8,   10,  -2,  -6,  -12, -12, 7,   3,   4,   -7,  -5,  -3,  -17, 3,   -25, -23, 18,  14,
    -3,  -5,  -3,  3,   -7,  -15, -1,  -5,  7,   -1,  -28, 12,  -28, 37,  -30, -9,  -8,  4,   7,   11,  16,  -4,  9,   -19, 0,   -19, -8,  7,   2,   7,   -20,
    0,   -11, 1,   -1,  -8,  10,  7,   7,   -3,  11,  1,   7,   4,   1,   14,  3,   14,  -10, -14, 23,  -9,  9,   -13, -1,  1,   7,   5,   4,   -8,  14,  6,
    -6,  12,  4,   -8,  -5,  -3,  27,  6,   3,   11,  -8,  6,   6,   24,  4,   20,  -5,  9,   0,   -21, -4,  1,   12,  -16, -2,  -4,  14,  4,   3,   1,   5,
    14,  10,  1,   -1,  18,  -1,  -15, 6,   -12, 10,  -12, 4,   -4,  -6,  0,   5,   8,   4,   -15, -3,  -4,  2,   -6,  -25, -3,  2,   2,   -1,  7,   32,  8,
    6,   5,   -8,  -7,  16,  -5,  -2,  17,  7,   -1,  8,   16,  -4,  13,  11,  20,  1,   3,   -34, 3,   7,   -13, 19,  -13, -2,  14,  -9,  2,   -6,  6,   16,
    -18, 8,   4,   -4,  10,  -11, 14,  6,   -10, 1,   -5,  -1,  12,  -4,  -12, -8,  4,   -2,  2,   6,   6,   -10, 7,   17,  6,   -2,  -5,  9,   -12, 0,   -12,
    -5,  13,  19,  -1,  1,   -12, -6,  -18, -6,  9,   2,   4,   0,   9,   1,   -11, 8,   -13, 15,  21,  -17, 13,  5,   -5,  -3,  14,  31,  0,   4,   8,   -28,
    -2,  -5,  0,   5,   -37, -24, 0,   18,  11,  5,   -4,  -3,  -6,  8,   8,   -2,  2,   14,  -24, -10, 43,  -14, -22, 0,   1,   -13, -21, 3,   -15, 21,  13,
    -7,  -11, 12,  3,   25,  11,  -13, 10,  0,   1,   11,  -15, 1,   -18, 0,   4,   9,   5,   13,  -28, 7,   45,  15,  3,   -2,  -5,  -1,  19,  5,   -7,  -4,
    -9,  35,  -17, 6,   -19, 3,   7,   -1,  -38, -13, 5,   10,  -39, 8,   0,   -6,  38,  25,  -13, 8,   -10, 9,   -7,  6,   4,   13,  1,   -5,  17,  37,  -8,
    21,  -16, -26, -1,  -4,  20,  21,  -5,  -14, -19, -45, -6,  -25, -19, 4,   -1,  -25, -3,  1,   12,  -49, 3,   3,   1,   -25, -25, 22,  25,  0,   -8,  -12,
    1,   -34, 20,  -44, 6,   -6,  11,  -9,  -12, -11, 5,   9,   -1,  -46, 8,   -3,  1,   -36, -27, -2,  5,   -12, 1,   -17, 13,  -7,  -5,  50,  -14, 27,  -2,
    -14, -5,  -10, -57, -8,  -5,  5,   14,  58,  -14, -12, 18,  11,  -2,  2,   25,  1,   29,  6,   -10, -10, 7,   -6,  -18, 1,   18,  -28, -10, 23,  28,  4,
    12,  5,   -1,  10,  -16, 18,  -14, 3,   5,   2,   -9,  -17, 11,  -22, -38, -11, -23, 10,  -37, 42,  4,   0,   -3,  -4,  1,   -2,  -2,  15,  -22, -28, -5,
    12,  -3,  2,   20,  -6,  15,  3,   -10, 6,   -34, 18,  5,   5,   -12, 3,   20,  10,  -6,  -15, 4,   -38, 19,  9,   15,  4,   44,  -3,  21,  7,   -1,  -8,
    32,  10,  -15, 5,   -11, -26, -8,  -26, 9,   1,   8,   -1,  -1,  5,   2,   -17, 23,  22,  30,  6,   26,  -13, 9,   12,  -5,  -5,  -27, -21, -16, 18,  -15,
    -35, -26, 12,  -7,  21,  -14, -2,  -27, 4,   -11, 10,  14,  15,  -1,  -31, -8,  16,  -17, 24,  47,  -16, -10, 4,   -15, 5,   16,  -30, 0,   2,   -12, -22,
    -9,  16,  5,   -1,  -5,  6,   -5,  -15, -9,  0,   -12, -1,  4,   -2,  -6,  10,  -6,  3,   9,   -1,  -6,  -7,  -9,  6,   10,  2,   -5,  9,   3,   -9,  3,
    -7,  5,   -5,  3,   -3,  -5,  4,   0,   -11, -12, -12, -17, -2,  -14, -14, 6,   -1,  5,   5,   -15, 8,   8,   -5,  8,   -4,  -6,  -13, 2,   6,   2,   -2,
    -9,  -10, -8,  -6,  -11, -10, -12, -13, -4,  -1,  -6,  2,   -3,  1,   -10, 2,   -2,  -5,  9,   -14, -6,  -7,  5,   1,   -3,  7,   -1,  3,   -6,  -4,  -7,
    4,   6,   -5,  -5,  10,  -3,  4,   3,   10,  -7,  -5,  -3,  3,   -12, -6,  -6,  -3,  -3,  -3,  -10, -3,  -14, -3,  1,   -4,  -1,  -9,  -8,  -9,  8,   0,
    -3,  -7,  -7,  -8,  1,   -15, -25, 1,   -32, -18, -20, 9,   12,  -39, -30, -13, -39, -2,  2,   11,  -28, 22,  -3,  4,   -17, 26,  -3,  -20, -13, 21,  14,
    -37, -11, -20, -14, -11, 2,   31,  -15, -16, -48, -8,  -29, 5,   -7,  0,   -7,  3,   -23, -14, -10, -16, -55, 4,   -27, -55, -45, -3,  -4,  -17, 13,  -25,
    -20, -37, 1,   30,  -6,  -27, -14, -26, -5,  -25, 4,   12,  -16, -11, -1,  -11, 17,  -57, 6,   10,  2,   -24, 7,   7,   6,   -19, -36, 5,   1,   -45, -18,
    -17, 1,   34,  1,   -38, -13, 18,  10,  -14, -30, -34, 3,   -9,  18,  2,   -47, 0,   -27, -12, 17,  -25, 46,  46,  -27, -24, -5,  -29, 33,  7,   -25, -41,
    11,  13,  -33, -8,  -28, -13, -28, -13, 0,   11,  -13, -2,  22,  -3,  10,  -8,  0,   -26, 4,   -6,  1,   21,  -16, -3,  -19, -14, 0,   -4,  31,  -11, 2,
    1,   -5,  -9,  -3,  -11, -8,  4,   12,  -6,  -11, -22, -3,  8,   1,   4,   10,  -13, -12, 7,   -20, 17,  4,   -6,  3,   2,   -4,  -14, 3,   11,  25,  9,
    -46, -10, 1,   2,   -16, 8,   -17, 10,  -1,  11,  8,   5,   -18, 3,   -25, 46,  13,  11,  19,  -2,  15,  20,  -8,  -5,  5,   19,  2,   -1,  12,  13,  -8,
    1,   3,   -6,  -2,  -6,  9,   27,  -11, 12,  5,   6,   12,  -13, 11,  25,  -12, -20, 9,   -43, -16, 3,   -4,  3,   -11, 2,   -33, 24,  -17, 8,   1,   6,
    -14, 10,  22,  -13, -5,  12,  -17, -18, -7,  -13, 16,  -1,  -10, -8,  10,  5,   11,  12,  14,  32,  -8,  -10, 42,  -8,  -1,  4,   8,   -9,  -26, 7,   1,
    -4,  -10, -14, -7,  -7,  -33, -27, -32, 8,   -10, 7,   -9,  5,   1,   1,   0,   16,  -30, 16,  -22, -21, 3,   -11, -35, -11, -10, 17,  16,  -18, -36, 2,
    20,  12,  3,   35,  -21, -32, -44, -9,  -6,  23,  15,  -13, 5,   36,  21,  -6,  -30, -6,  -6,  22,  -17, -10, -4,  -20, -19, -36, 2,   -27, 17,  -19, 0,
    7,   1,   -13, 3,   14,  -25, 5,   9,   -36, -5,  3,   23,  -24, 21,  12,  12,  41,  -34, -29, 2,   29,  -17, -2,  2,   5,   -3,  -9,  19,  12,  -8,  -16,
    -24, -11, 19,  19,  -5,  7,   9,   0,   13,  -1,  11,  10,  -14, -3,  4,   37,  26,  19,  -25, 24,  -13, -5,  19,  7,   2,   24,  -18, -6,  -3,  13,  -8,
    25,  10,  -29, 10,  14,  3,   -33, -18, -21, 13,  47,  -20, 9,   18,  11,  3,   -5,  -26, -6,  -7,  39,  -10, -31, -20, -23, -20, -3,  -11, -3,  2,   -18,
    16,  21,  15,  -13, -15, 31,  -30, 30,  8,   4,   -29, 19,  22,  -23, 5,   7,   5,   -35, 9,   -11, 22,  27,  -8,  4,   -11, -28, -24, -5,  19,  -23, 3,
    9,   26,  -5,  1,   5,   12,  -3,  29,  -5,  -13, -9,  15,  -3,  9,   -45, 3,   -68, 56,  1,   21,  -25, 4,   21,  3,   3,   27,  -6,  -10, 13,  6,   17,
    9,   28,  10,  -26, 3,   42,  4,   0,   20,  -5,  2,   -22, 32,  -19, -5,  -25, -11, -6,  33,  -66, 6,   9,   8,   -3,  -9,  17,  9,   -22, -3,  -23, -10,
    -4,  -1,  15,  3,   -2,  13,  -11, 12,  10,  -19, 13,  14,  -16, -2,  -15, 5,   -15, -35, 20,  2,   10,  11,  -6,  11,  -1,  -11, -16, -2,  7,   -10, -2,
    7,   -15, 20,  13,  -19, 18,  -9,  -5,  -4,  -11, 16,  23,  -5,  8,   11,  -12, -8,  13,  1,   2,   3,   -17, 9,   0,   9,   -3,  42,  -1,  -10, 8,   5,
    -3,  -2,  7,   0,   -2,  -12, 1,   -5,  -23, 20,  15,  5,   -1,  8,   14,  11,  0,   2,   20,  9,   3,   -2,  -6,  -6,  15,  12,  -3,  -23, 24,  -6,  -3,
    3,   7,   -16, -7,  16,  12,  -11, -7,  -5,  -2,  -5,  -7,  0,   1,   -12, 9,   -14, -6,  29,  -2,  -15, 4,   27,  -3,  13,  5,   18,  -40, 2,   26,  -23,
    -4,  -5,  8,   -8,  8,   -7,  16,  12,  4,   10,  -12, 5,   -3,  8,   -4,  -46, -34, 18,  -18, -3,  10,  -8,  11,  0,   11,  -25, 14,  -7,  9,   -22, -5,
    23,  15,  10,  21,  -14, -12, -49, 4,   7,   -3,  31,  29,  -7,  9,   -49, 34,  11,  29,  -57, -4,  -21, 8,   -25, 14,  -6,  8,   15,  -9,  -3,  -20, 14,
    -23, 15,  16,  -30, -4,  2,   15,  -40, -26, 3,   24,  -13, -14, 25,  -4,  -14, 5,   -20, -11, -15, -7,  2,   -7,  -37, 14,  -5,  -21, 2,   6,   3,   -20,
    26,  9,   19,  7,   4,   7,   16,  15,  -6,  13,  -10, 6,   -32, -7,  2,   5,   -16, 9,   -35, 38,  19,  -29, -45, 21,  21,  -22, 19,  1,   -18, -5,  -7,
    2,   9,   -4,  -59, 5,   4,   26,  -25, -13, 14,  -4,  -11, 13,  27,  -5,  22,  -14, -69, -31, -1,  18,  -2,  -38, -10, 16,  -38, -4,  -3,  -16, 3,   21,
    -22, 8,   -25, -54, -12, -17, 17,  20,  -15, -11, -16, 39,  -34, 12,  -10, -14, -36, -5,  -17, -13, 10,  -18, -4,  -19, -22, -50, 4,   28,  18,  -8,  25,
    3,   -13, -32, 19,  -32, 25,  -19, 11,  26,  6,   21,  7,   16,  -13, 47,  11,  9,   -27, -49, 42,  6,   24,  -29, -31, 8,   14,  -11, 6,   -13, 36,  11,
    -34, 11,  11,  18,  -3,  15,  -8,  -24, 1,   12,  20,  -21, 8,   -43, 31,  0,   -16, -26, 9,   -63, -19, 14,  -18, 22,  -19, 25,  -28, 40,  39,  -10, -22,
    23,  -13, -28, 7,   1,   11,  -6,  -2,  14,  6,   10,  2,   -15, 6,   -18, -17, 11,  -2,  -14, 6,   7,   6,   4,   -1,  22,  -13, -4,  -17, 12,  -15, -12,
    -17, 5,   -11, 0,   -6,  -11, 5,   -4,  6,   2,   5,   -15, 21,  11,  10,  12,  -1,  -12, -9,  7,   0,   7,   -11, 0,   14,  3,   -19, 6,   21,  -7,  -14,
    -11, -16, 4,   3,   20,  -14, -5,  -24, 2,   14,  2,   6,   -11, 5,   -10, -6,  -9,  -9,  19,  18,  2,   0,   9,   8,   13,  10,  5,   4,   -1,  -8,  -12,
    6,   5,   -26, 13,  20,  -23, -9,  14,  1,   -14, -12, 11,  -9,  4,   -12, 0,   -17, 25,  10,  16,  -9,  -22, 9,   -1,  9,   4,   7,   -7,  5,   -11, 8,
    -5,  22,  -7,  -33, -12, 17,  9,   3,   1,   -12, 4,   10,  9,   0,   -20, -1,  4,   5,   -15, 17,  6,   1,   1,   6,   3,   7,   -6,  10,  -2,  -3,  5,
    14,  6,   -5,  -4,  10,  10,  11,  14,  6,   -10, 19,  -10, -15, 21,  2,   6,   -4,  12,  -2,  2,   -16, 12,  -2,  -14, 5,   4,   2,   -10, -10, -19, -4,
    14,  -1,  24,  -3,  -11, 1,   0,   10,  -5,  6,   14,  19,  12,  -21, -6,  -2,  0,   6,   7,   14,  15,  -12, -11, -9,  -29, 4,   -11, 8,   7,   0,   -4,
    -4,  8,   14,  31,  15,  5,   4,   -15, 20,  29,  23,  16,  -27, 32,  0,   1,   8,   8,   -1,  -8,  15,  -20, -30, 16,  4,   10,  0,   -2,  -18, -11, -12,
    -12, 19,  3,   -6,  11,  -5,  9,   44,  24,  -26, -1,  0,   -26, -32, -1,  -57, -15, 11,  -34, -19, -2,  29,  -21, 3,   0,   0,   -1,  -1,  -35, 21,  22,
    0,   -5,  43,  28,  25,  -31, -44, -29, -32, 12,  -4,  18,  -24, -52, 9,   -18, -16, -28, -49, -13, -17, 8,   -16, -26, 6,   28,  -26, 5,   16,  -26, 26,
    5,   -24, -24, 18,  16,  -43, 8,   -7,  -59, -4,  -39, 7,   13,  -33, -7,  -17, 15,  3,   -37, -27, -9,  -18, -14, 22,  39,  -25, -1,  -63, 0,   8,   15,
    -16, -6,  18,  -25, -1,  -17, -9,  9,   -36, 13,  18,  -47, -65, -38, 24,  1,   -26, -6,  6,   -5,  -50, 0,   -38, 32,  15,  -50, -14, -10, 6,   6,   -29,
    -40, -33, -2,  11,  -20, -11, -28, -74, -15, 32,  -21, 24,  -15, -18, -2,  0,   15,  -7,  -3,  1,   -27, -3,  6,   5,   -28, 18,  21,  -42, -20, -28, 7,
    27,  9,   -26, -19, 34,  32,  -29, -27, -16, -41, 19,  -18, -2,  8,   18,  -6,  -14, -10, -16, -27, -24, -11, -9,  -17, -29, -21, -7,  -70, 10,  20,  1,
    12,  22,  -5,  -20, 14,  -13, -19, -35, 4,   -20, -10, 33,  -11, 52,  39,  -3,  17,  -19, -23, 21,  -38, -4,  43,  10,  -16, -5,  3,   29,  31,  -9,  -1,
    -15, -33, -7,  -13, 7,   0,   -24, 8,   -19, -11, -13, -11, -2,  -35, 21,  -35, -46, 16,  16,  7,   0,   8,   -40, 12,  26,  2,   11,  -50, 1,   13,  -4,
    -11, -16, -4,  -13, -37, 29,  -51, 31,  -16, -6,  -20, -41, -34, 17,  -34, 11,  10,  24,  -5,  -33, 18,  16,  12,  -9,  2,   -23, -5,  0,   -20, -11, -21,
    -40, 17,  22,  -20, -2,  3,   -5,  18,  8,   -23, -11, -25, -47, 12,  20,  -1,  -31, -15, -6,  -31, 0,   -12, 21,  12,  4,   -10, 1,   -7,  8,   11,  14,
    22,  18,  -39, -23, -13, 9,   8,   0,   -9,  11,  14,  -6,  8,   38,  -11, 7,   -5,  -19, -59, -11, -10, -14, -11, -49, -9,  -6,  -9,  -1,  -12, 6,   6,
    -31, -21, 0,   -18, -33, 26,  8,   28,  -19, 7,   22,  -13, -15, -15, -10, 9,   -40, -16, -35, 14,  -65, -26, -36, -16, 1,   9,   14,  -39, 3,   -2,  1,
    -5,  1,   45,  -6,  -4,  4,   12,  -3,  -7,  2,   -28, -19, -6,  -27, -3,  10,  3,   36,  14,  16,  -35, 20,  25,  -3,  4,   9,   -2,  4,   -11, 27,  5,
    -50, 5,   -8,  16,  17,  10,  0,   -2,  4,   4,   -11, -46, 4,   31,  10,  3,   25,  -3,  -16, 5,   14,  -3,  8,   21,  -6,  -58, -20, 15,  -23, -42, 20,
    -47, -10, -39, -1,  5,   -31, -17, 4,   6,   11,  5,   -27, 3,   6,   9,   -13, -4,  -23, -20, -2,  -4,  21,  21,  -48, 7,   12,  -17, 65,  31,  -24, -1,
    7,   28,  -19, -28, 29,  13,  32,  -52, 11,  8,   -20, -4,  12,  7,   -30, 23,  -28, -48, 6,   36,  -90, 1,   -22, 20,  33,  -8,  -49, 11,  19,  6,   -5,
    2,   0,   6,   -2,  -8,  -1,  17,  -21, 7,   -11, 14,  -31, 9,   -17, -39, -52, 25,  10,  11,  -10, -4,  -16, 17,  -20, -11, 4,   -14, 4,   13,  -25, 8,
    3,   -12, 3,   11,  -35, -25, 8,   4,   -7,  11,  6,   -13, 8,   -8,  -6,  -8,  3,   25,  -11, 12,  -4,  17,  5,   -3,  -3,  -10, 11,  -7,  -13, 5,   -2,
    -16, 18,  -6,  14,  -2,  4,   0,   14,  17,  -10, 8,   -24, 6,   -9,  16,  0,   2,   -7,  15,  5,   10,  15,  -19, -13, -19, -1,  12,  21,  -1,  -18, -24,
    11,  4,   -22, -16, 11,  9,   2,   9,   -9,  -20, -7,  13,  -14, 22,  12,  -1,  -17, -8,  -10, 3,   9,   7,   8,   -9,  4,   6,   -44, -6,  -18, -33, 3,
    8,   14,  -13, -2,  16,  7,   10,  -5,  11,  -4,  15,  -17, 1,   0,   -2,  -5,  12,  30,  9,   8,   9,   -2,  -3,  5,   -13, 12,  17,  -29, -9,  0,   11,
    -17, 3,   -1,  -1,  4,   -13, -3,  -5,  7,   13,  15,  -9,  4,   16,  22,  18,  -3,  3,   11,  -17, -3,  -30, -8,  -2,  5,   -26, -10, 4,   20,  3,   7,
    6,   -4,  -6,  2,   -3,  -18, 15,  -6,  -13, -4,  -19, 0,   -13, 9,   -8,  24,  9,   11,  9,   12,  -4,  -17, 6,   2,   13,  11,  -9,  -14, -19, 2,   19,
    7,   12,  8,   -3,  11,  17,  16,  -1,  -8,  15,  -2,  4,   -14, 8,   -18, 16,  -7,  -4,  16,  -23, -11, 2,   3,   7,   -11, -6,  11,  9,   6,   -8,  18,
    9,   27,  3,   -9,  25,  -17, 18,  7,   -25, 15,  7,   10,  -25, -19, 4,   1,   17,  0,   -9,  -2,  -6,  8,   -7,  5,   -4,  -11, 11,  6,   10,  -23, -3,
    -9,  -8,  7,   8,   -13, -9,  -12, -1,  -9,  16,  -34, -2,  -7,  -17, -18, 0,   -16, -22, -6,  11,  -32, -11, -19, 7,   9,   -1,  15,  26,  -7,  6,   13,
    -17, 2,   -20, 12,  -8,  0,   17,  9,   -6,  -5,  8,   4,   -26, 0,   -7,  -9,  15,  -16, -5,  -33, 11,  -11, -3,  4,   5,   -7,  19,  -9,  3,   -2,  -2,
    11,  -17, 31,  -11, 19,  3,   15,  -11, 16,  -4,  -2,  -13, -5,  11,  14,  14,  0,   10,  5,   9,   30,  -7,  20,  -15, 8,   11,  10,  -6,  5,   10,  7,
    -6,  2,   -14, 5,   0,   13,  -17, -2,  24,  15,  -5,  5,   -4,  -6,  -2,  1,   6,   6,   29,  3,   15,  6,   20,  -5,  16,  -11, 0,   11,  9,   10,  8,
    22,  -2,  13,  -8,  -36, -11, 0,   -10, -24, -35, 2,   11,  -13, -42, 4,   -26, 14,  15,  -5,  -6,  18,  -3,  -20, 19,  -39, -13, -7,  12,  23,  -24, -24,
    -26, 19,  0,   -11, -24, -2,  -6,  10,  9,   -9,  -4,  -33, 6,   -50, 1,   -1,  6,   -12, 0,   -4,  -6,  -20, -33, 18,  0,   6,   4,   20,  -29, -15, -40,
    1,   8,   8,   -6,  10,  36,  -24, 31,  -22, 0,   -20, 7,   35,  5,   27,  9,   2,   -22, 23,  5,   17,  -7,  -12, -22, -30, -3,  -2,  -4,  -16, 8,   21,
    14,  -13, -25, -4,  29,  -3,  -21, 17,  -13, 28,  27,  12,  -15, 20,  -2,  2,   8,   12,  8,   14,  -19, 4,   1,   -30, -22, 14,  -21, -4,  7,   11,  -9,
    -14, -20, -6,  9,   -11, -1,  -20, -16, 15,  23,  18,  -36, -14, -36, 22,  -58, -20, 11,  -33, -22, 25,  10,  -8,  -16, 22,  16,  -16, -14, -16, -9,  28,
    9,   4,   -7,  -8,  20,  -23, 5,   -1,  9,   17,  -6,  19,  8,   -37, -38, 10,  -17, 14,  -16, -4,  -1,  -5,  10,  -4,  -6,  -9,  -4,  -40, -23, 32,  6,
    5,   -19, 8,   9,   -19, 10,  7,   -38, 28,  26,  -38, -23, -41, -30, 9,   -8,  -28, -4,  -6,  -18, -2,  -20, -30, 11,  24,  -4,  6,   13,  12,  27,  -19,
    6,   19,  -2,  -20, -9,  15,  -15, 5,   16,  -25, -15, 18,  3,   9,   -27, 7,   9,   1,   -3,  -2,  -19, -20, -6,  25,  14,  -33, -29, -42, -26, 15,  23,
    -20, -11, 2,   -17, -20, 0,   21,  -26, -27, -5,  -6,  5,   19,  2,   -4,  -4,  13,  2,   -8,  -7,  -9,  -4,  -6,  -17, -22, -1,  -6,  1,   -1,  -1,  8,
    -12, 6,   7,   -3,  -20, 2,   20,  3,   -12, 12,  -6,  -1,  -1,  21,  12,  9,   14,  1,   13,  5,   22,  -19, 13,  9,   -4,  -14, 14,  1,   7,   9,   4,
    -5,  1,   -41, 20,  9,   -21, 5,   11,  -6,  -8,  -3,  -4,  18,  -11, -5,  -18, 2,   -11, 26,  14,  -10, -26, -1,  13,  1,   -10, 1,   1,   13,  2,   8,
    13,  14,  -2,  -8,  -30, -10, 2,   -29, 8,   -6,  -7,  11,  -1,  9,   9,   7,   -3,  0,   8,   -5,  26,  3,   -2,  -2,  7,   -4,  -6,  5,   1,   -18, 12,
    13,  9,   5,   18,  -9,  -8,  0,   -22, 20,  -8,  -11, 9,   5,   -8,  -9,  26,  -23, -23, -12, 18,  25,  36,  6,   22,  -3,  20,  32,  -24, 11,  -7,  3,
    12,  -8,  -41, -2,  19,  -5,  4,   -6,  -15, -39, -5,  6,   -32, -26, -7,  -22, -6,  -4,  22,  14,  12,  -4,  23,  -20, 15,  -4,  -33, 2,   19,  2,   24,
    -3,  -21, -7,  -30, -18, -12, -3,  -18, -6,  28,  1,   -7,  -9,  5,   28,  -24, -12, -2,  14,  9,   28,  7,   28,  -9,  -20, -2,  10,  4,   -7,  10,  -34,
    -11, -18, -8,  10,  26,  26,  -21, -7,  -35, 36,  12,  -14, -10, 22,  10,  10,  -10, 0,   23,  -32, 2,   17,  33,  -23, -22, 16,  13,  19,  1,   0,   -3,
    14,  -25, 2,   14,  28,  2,   -11, 1,   -1,  12,  40,  7,   -21, -19, 7,   4,   11,  7,   16,  -4,  1,   16,  32,  13,  10,  7,   -2,  19,  12,  -20, 5,
    -17, 9,   11,  -4,  -12, -8,  -13, 17,  -7,  -9,  11,  3,   7,   5,   0,   -35, 0,   20,  -9,  -5,  -15, -5,  1,   -25, 11,  -5,  -17, -17, -5,  -12, -7,
    14,  0,   14,  -4,  -10, -20, 20,  9,   17,  -12, -30, -8,  17,  11,  -6,  -19, -5,  -10, 0,   11,  13,  -18, -5,  20,  -11, 18,  13,  -7,  -8,  11,  8,
    6,   -9,  -10, 14,  5,   6,   9,   -23, 30,  1,   4,   -7,  5,   -17, 9,   5,   16,  0,   0,   4,   34,  -11, -5,  -13, 2,   -8,  6,   10,  3,   -7,  -8,
    -10, 5,   3,   9,   2,   -20, 11,  23,  20,  0,   15,  4,   0,   6,   1,   3,   3,   -3,  -7,  2,   7,   2,   -14, 18,  -13, 8,   -17, 2,   8,   -9,  -9,
    -10, -16, -6,  -14, 1,   8,   1,   -16, 1,   -1,  -42, -6,  -35, 13,  0,   -11, -8,  -5,  10,  13,  6,   3,   -5,  -3,  -20, -5,  -9,  -18, -5,  10,  17,
    13,  -3,  -14, -14, -23, -2,  -9,  -13, -11, -1,  6,   0,   -8,  17,  -7,  12,  -4,  -24, -11, 7,   16,  8,   -1,  -4,  20,  -2,  13,  0,   10,  8,   19,
    15,  2,   -22, 12,  13,  0,   4,   7,   8,   -11, 11,  -5,  0,   -4,  -5,  6,   -20, -6,  7,   2,   14,  19,  5,   12,  -7,  4,   2,   6,   27,  21,  -5,
    -8,  -11, -10, -18, -1,  21,  8,   13,  0,   16,  -9,  12,  -13, -15, 9,   7,   -4,  20,  2,   -15, -24, 24,  2,   4,   12,  -25, 19,  -14, 11,  4,   -17,
    9,   -11, -16, 0,   13,  10,  -5,  23,  -29, 29,  -15, 4,   -20, 15,  -17, 0,   0,   21,  -2,  6,   -17, 35,  -10, -12, -12, -40, 22,  -9,  -21, -26, -20,
    -14, 0,   -14, -42, -11, -2,  2,   9,   6,   14,  -1,  20,  -3,  3,   25,  14,  -19, -1,  38,  -12, 2,   -35, -13, -15, 11,  14,  -45, 33,  -4,  -10, -24,
    17,  5,   8,   -30, 31,  50,  39,  -30, -21, 13,  2,   -13, -8,  3,   -20, 4,   -45, -15, 12,  17,  -1,  -12, -8,  -25, -2,  9,   -1,  3,   45,  23,  33,
    29,  7,   20,  23,  24,  10,  4,   36,  -4,  -17, -4,  10,  15,  14,  4,   -7,  -18, 14,  -27, 1,   2,   -33, 0,   -6,  -7,  -1,  20,  13,  0,   -2,  -39,
    2,   37,  -4,  -13, 14,  10,  -22, -31, -2,  -11, -19, -29, 12,  3,   -22, 14,  6,   -14, 6,   -2,  -19, -45, -26, -12, 21,  11,  7,   -25, 29,  -2,  1,
    26,  -12, 28,  -28, 6,   -1,  10,  36,  -13, -5,  -19, -22, -3,  -13, -10, -10, -19, 22,  -3,  9,   14,  -43, -6,  -7,  -9,  28,  21,  -20, -27, -5,  -21,
    -12, 16,  9,   -25, -5,  -4,  9,   29,  -19, 18,  -15, 2,   -2,  12,  -21, 6,   -31, -1,  12,  -8,  -39, -10, 4,   -6,  5,   -19, -4,  -18, -19, 20,  0,
    -6,  8,   -22, -4,  5,   -34, -19, -11, 10,  25,  -6,  -4,  -8,  25,  15,  3,   10,  -39, 8,   -8,  -60, -12, -25, -15, -14, -2,  3,   0,   -2,  19,  9,
    -26, -2,  26,  -10, -1,  -1,  -10, -30, -36, -19, -37, -25, 34,  -13, -3,  -3,  32,  -10, -4,  9,   -14, -20, -16, -14, -16, -15, -8,  -25, -16, 28,  33,
    -22, -33, 16,  48,  15,  7,   -26, 18,  -26, 11,  -17, -16, -13, -24, 10,  11,  -12, 5,   -23, -46, -29, -2,  9,   -8,  8,   -12, 8,   -53, -6,  -20, 5,
    -59, -19, -23, 34,  15,  -5,  -9,  -15, 6,   3,   -16, -3,  7,   16,  3,   -1,  -22, -6,  19,  -13, -18, 0,   -5,  10,  11,  -34, 30,  -3,  1,   -17, 7,
    -9,  -15, 5,   -15, -1,  1,   -13, 16,  -4,  -15, -26, 13,  2,   16,  26,  19,  7,   -17, -23, 1,   -30, -10, -30, -8,  12,  -27, -27, 2,   1,   -31, 10,
    -3,  -43, 3,   19,  14,  -28, 7,   -13, 8,   10,  -18, 10,  20,  -8,  -7,  -16, -5,  20,  -15, 7,   5,   13,  -3,  4,   -13, -15, -15, -7,  -1,  13,  4,
    -2,  9,   -22, -5,  17,  -13, 12,  -4,  -26, -14, 6,   -2,  -31, -17, 7,   6,   12,  -13, 12,  -20, 0,   21,  13,  14,  1,   -18, 7,   14,  4,   -14, 11,
    -19, 13,  11,  -15, 4,   -13, 7,   -22, 8,   -7,  -12, -1,  -14, 6,   -6,  -12, -18, 25,  -16, 2,   -4,  22,  -20, -4,  6,   25,  11,  19,  7,   -13, 10,
    11,  -1,  -22, 5,   -15, -17, 11,  -12, 6,   -8,  9,   10,  13,  -6,  -11, -7,  24,  8,   -7,  7,   10,  4,   5,   -5,  -12, -1,  18,  -16, -31, -6,  -4,
    18,  17,  -28, -14, 9,   6,   -22, -6,  7,   -21, 4,   21,  -8,  15,  -1,  -6,  8,   -9,  1,   4,   13,  17,  -24, 8,   3,   -2,  13,  3,   5,   -5,  11,
    -13, 1,   -4,  12,  6,   -22, 13,  -9,  22,  -3,  6,   4,   -18, -2,  14,  8,   -2,  -1,  4,   -14, -14, 15,  9,   17,  7,   -16, -13, 1,   -5,  -27, -2,
    16,  8,   2,   -4,  9,   13,  18,  -34, -29, 8,   5,   2,   -5,  2,   -12, -22, -17, 3,   14,  -26, 11,  4,   30,  5,   6,   9,   9,   7,   -8,  -7,  14,
    -16, -7,  0,   0,   -2,  -13, 11,  3,   -12, 9,   -4,  0,   -4,  15,  4,   -1,  -25, -20, -1,  -32, 8,   -4,  2,   1,   -11, 1,   9,   -7,  2,   -14, 7,
    -10, -11, -2,  8,   0,   -12, -14, 5,   -2,  3,   -14, 4,   -4,  0,   10,  -31, 7,   19,  21,  -15, 20,  26,  19,  -17, -18, 10,  20,  15,  7,   4,   7,
    -15, -7,  -10, 12,  -1,  -13, -12, 1,   3,   -48, 23,  2,   15,  -33, 24,  0,   -6,  -24, -3,  -17, -1,  11,  5,   -15, -10, -26, -7,  14,  8,   0,   -3,
    12,  5,   -5,  1,   18,  8,   5,   -10, -14, 9,   -39, -12, 4,   5,   10,  -11, 10,  10,  -8,  2,   -9,  16,  3,   -2,  8,   -11, 31,  8,   -4,  5,   11,
    15,  4,   3,   17,  3,   -22, -19, -10, 2,   -7,  -18, -8,  -13, 5,   3,   -14, 9,   48,  2,   -31, 4,   -9,  -3,  -18, -17, -4,  -1,  30,  5,   11,  3,
    10,  -8,  -8,  -15, 26,  18,  7,   1,   -21, 9,   -15, 2,   -6,  -17, -28, 2,   -24, 6,   4,   -6,  7,   46,  -7,  -11, -17, -6,  25,  8,   -5,  -5,  1,
    0,   0,   -4,  8,   3,   -20, 10,  -9,  13,  -25, 14,  -3,  2,   6,   -4,  36,  -33, 0,   14,  1,   -18, -13, 4,   11,  -1,  -9,  -5,  -16, 8,   20,  2,
    12,  -16, 14,  3,   0,   2,   8,   19,  -22, 24,  1,   1,   -8,  -14, -16, 8,   5,   -12, -4,  8,   17,  4,   5,   -16, -34, 9,   -20, -11, 1,   0,   -17,
    19,  9,   -10, -7,  0,   10,  5,   -5,  -8,  18,  7,   7,   4,   -18, 12,  10,  -22, 20,  -16, -9,  9,   -34, -26, 29,  -25, 13,  -11, -2,  7,   24,  13,
    -7,  5,   15,  -10, 9,   -4,  10,  16,  32,  -29, -3,  -3,  16,  -15, -13, -13, -13, -20, 6,   1,   -6,  -3,  11,  28,  16,  27,  -12, 23,  -9,  -21, -7,
    -15, 6,   5,   -34, 6,   -6,  -33, 0,   20,  15,  -12, -2,  -6,  -8,  -23, 5,   -14, -1,  -12, -2,  33,  -13, 17,  -23, -18, 1,   -21, -3,  -4,  2,   2,
    -2,  -9,  -13, -7,  -16, 12,  14,  5,   -18, 5,   -7,  27,  -5,  2,   4,   -3,  15,  18,  4,   16,  23,  3,   -8,  -17, 2,   4,   -19, 31,  17,  -6,  12,
    15,  -4,  6,   -38, 4,   3,   23,  -14, 8,   1,   -17, 14,  0,   -14, -8,  11,  -7,  0,   -27, -6,  -3,  1,   14,  -16, 6,   22,  -11, 21,  26,  -8,  32,
    1,   -1,  -34, -8,  17,  16,  -3,  13,  5,   3,   -10, 4,   8,   37,  8,   -18, 23,  -14, 5,   25,  0,   -10, -3,  -1,  15,  -11, 1,   -14, 6,   13,  7,
    5,   -38, 19,  22,  11,  15,  -10, -3,  -5,  -1,  -34, -8,  -18, 3,   12,  23,  -12, -27, -34, 14,  2,   4,   4,   -9,  -15, 24,  13,  33,  -25, -15, -7,
    -10, 16,  30,  -8,  -13, 9,   -15, -15, 19,  13,  8,   10,  -22, 3,   -12, -21, -8,  -13, -17, -16, -8,  -9,  18,  13,  0,   -16, 6,   5,   5,   12,  3,
    33,  5,   -14, 11,  14,  -2,  -2,  -4,  2,   22,  4,   -2,  -21, -2,  -2,  13,  24,  1,   4,   20,  8,   11,  -10, -8,  -2,  5,   -19, 1,   12,  13,  -40,
    23,  -5,  3,   13,  0,   -20, -2,  0,   -32, 25,  -1,  -22, 13,  -11, -22, 11,  38,  6,   16,  10,  -22, 6,   -15, -9,  -5,  7,   19,  32,  -6,  22,  3,
    18,  -1,  8,   9,   16,  -11, -10, -33, -12, -11, -32, 15,  -8,  -34, -8,  -17, 9,   -27, -10, -3,  -16, -24, 13,  3,   5,   3,   15,  22,  -7,  14,  5,
    -36, 7,   -1,  10,  6,   -10, -18, 7,   15,  -5,  -9,  4,   -5,  -5,  -18, 12,  -16, -17, -10, -10, 24,  -43, -5,  -2,  -22, 12,  -3,  -19, 4,   9,   -7,
    -28, -5,  42,  56,  16,  -16, 23,  -7,  -21, 7,   -19, 0,   -4,  10,  -7,  -34, -6,  6,   28,  -11, -35, 3,   -44, 1,   -10, 25,  -27, -1,  3,   -9,  4,
    -9,  -3,  -2,  -2,  -24, 9,   9,   -7,  -22, -15, 36,  -15, 1,   1,   -3,  4,   3,   -2,  4,   8,   5,   -10, 19,  -1,  29,  -6,  -7,  -3,  22,  -1,  5,
    -28, 10,  -27, 0,   3,   8,   4,   -3,  4,   15,  7,   -12, -14, 4,   1,   -4,  18,  -4,  1,   7,   -10, 6,   -11, 2,   5,   -21, 20,  -8,  -6,  22,  3,
    29,  13,  -14, 3,   -5,  -34, 18,  -37, -9,  17,  -20, -6,  29,  8,   -37, -23, 11,  11,  -20, -14, -4,  25,  -29, 13,  -22, -11, 13,  -5,  -22, 5,   -19,
    -11, 13,  5,   6,   -4,  -32, 10,  -1,  20,  22,  10,  -20, 1,   -16, 29,  -14, -15, -9,  1,   14,  -1,  21,  -37, -36, 0,   -11, -18, -15, -9,  -25, -16,
    -8,  -11, -20, 5,   2,   7,   -11, -22, -17, 16,  20,  15,  -12, -10, 39,  2,   -11, -18, -31, 18,  -8,  -7,  3,   15,  8,   -10, -8,  2,   20,  0,   -12,
    -16, -21, 7,   -2,  14,  -13, 2,   -9,  0,   -12, -48, 28,  22,  16,  -1,  -3,  15,  -12, -6,  4,   12,  -4,  4,   -18, 5,   5,   37,  4,   -26, 7,   -21,
    3,   14,  13,  -12, -8,  12,  7,   -55, -23, -37, 11,  -57, 19,  -11, -35, -14, -21, 1,   15,  9,   -12, 2,   36,  -17, -9,  18,  -5,  22,  -9,  -25, -40,
    2,   -18, 17,  45,  31,  24,  -40, -9,  18,  0,   0,   -25, -20, -17, -42, -6,  36,  -29, -7,  -35, -10, -9,  1,   3,   -27, 3,   -23, 4,   4,   -20, 17,
    18,  -18, 10,  11,  11,  2,   -68, 24,  -25, 4,   -9,  2,   -25, 24,  -25, 4,   25,  -31, 15,  16,  -30, 22,  -16, -10, -8,  1,   -86, -17, -11, 46,  -5,
    -35, -7,  4,   20,  -14, 3,   7,   18,  0,   19,  24,  -1,  18,  -13, -6,  -34, 4,   2,   -13, -16, -1,  9,   -18, -11, 15,  -9,  31,  12,  6,   7,   18,
    -14, -22, 12,  -10, -13, 12,  -11, 11,  -5,  -12, -2,  -20, -5,  -18, -10, 9,   9,   -6,  -20, -12, -8,  9,   -1,  1,   -5,  6,   5,   10,  3,   12,  2,
    -4,  9,   4,   -1,  1,   -19, 0,   10,  -23, 5,   14,  14,  -13, 13,  -18, -4,  18,  -3,  10,  4,   -33, 17,  6,   36,  9,   4,   -4,  -11, -6,  -9,  -4,
    9,   -9,  3,   -3,  11,  10,  7,   -18, 2,   2,   -1,  5,   -6,  4,   23,  6,   14,  18,  -7,  -6,  18,  13,  -22, -11, 17,  -1,  1,   12,  -7,  9,   -20,
    -13, -1,  -5,  8,   -3,  -13, 12,  -5,  6,   -5,  12,  4,   -4,  8,   -26, -2,  -23, 1,   50,  1,   -11, -7,  -7,  -7,  -16, -8,  6,   5,   -4,  -3,  -17,
    6,   -9,  -5,  -5,  17,  -5,  -16, -4,  -16, -5,  -13, 7,   -5,  -6,  2,   -2,  6,   0,   -3,  -3,  -4,  -5,  -14, 4,   -14, -6,  -8,  31,  -15, -23, 0,
    -2,  2,   -8,  -2,  -5,  -12, -7,  -11, -12, -3,  -15, 1,   -11, 1,   -5,  -16, -5,  -3,  3,   -12, 1,   -9,  12,  -16, -2,  -23, -4,  1,   -3,  -32, 0,
    9,   -3,  -5,  -4,  -13, -10, -7,  -2,  -8,  -8,  2,   -13, 6,   -13, -10, -10, 2,   9,   0,   -7,  4,   3,   2,   2,   -10, 0,   13,  -15, -15, -14, -4,
    -13, -8,  -2,  1,   -1,  17,  -8,  -1,  -5,  9,   -7,  -3,  4,   -8,  -3,  -3,  7,   -9,  -13, -5,  -3,  -12, -10, -1,  -2,  -8,  -8,  -5,  2,   13,  6,
    17,  -6,  31,  -36, -3,  -16, 8,   5,   -25, 22,  6,   -5,  10,  -3,  1,   -9,  16,  -21, 15,  -1,  8,   30,  7,   -2,  2,   6,   1,   -12, 17,  10,  15,
    -4,  5,   6,   -15, -5,  0,   -8,  -9,  6,   -3,  2,   -2,  -1,  14,  -8,  -30, 17,  12,  7,   -1,  4,   -2,  22,  -5,  7,   2,   -9,  -17, -7,  -13, -9,
    3,   8,   5,   -7,  7,   -7,  5,   10,  5,   6,   1,   7,   5,   -8,  -12, 8,   6,   -6,  12,  -3,  -8,  12,  -10, 6,   1,   -10, 0,   -7,  -17, -12, -3,
    -11, -44, 0,   10,  7,   -4,  -5,  4,   16,  14,  -3,  -6,  13,  -3,  4,   11,  -16, 7,   3,   6,   0,   -10, -13, 1,   19,  2,   -32, -3,  -6,  23,  15,
    14,  -12, 7,   -16, -7,  -8,  8,   -4,  -11, 4,   6,   2,   3,   6,   -16, 3,   -5,  -9,  -9,  0,   25,  -10, 2,   10,  -4,  12,  -9,  12,  24,  -3,  4,
    -22, -6,  9,   3,   -21, -2,  0,   -1,  1,   6,   8,   -1,  -12, -1,  10,  -8,  -8,  32,  1,   -30, 12,  4,   6,   4,   12,  -25, -3,  2,   5,   -8,  -2,
    -11, 5,   15,  -15, -26, -7,  6,   -13, -2,  12,  -6,  -1,  5,   1,   8,   20,  1,   3,   13,  3,   -3,  13,  10,  -24, 5,   7,   -9,  -22, 7,   -39, 2,
    26,  -9,  -4,  33,  -13, -27, -4,  7,   -9,  7,   8,   4,   16,  16,  -7,  -1,  11,  4,   2,   7,   28,  -5,  14,  -22, -1,  -14, 0,   14,  -10, 3,   3,
    -20, 6,   20,  10,  -12, -6,  9,   -19, -3,  -16, 2,   20,  13,  14,  -42, -12, -23, -11, 1,   -16, 7,   14,  -9,  -19, -33, -2,  9,   -4,  16,  10,  5,
    6,   3,   -7,  11,  -12, -31, 4,   -14, 52,  17,  14,  10,  3,   16,  8,   4,   1,   -24, -44, 11,  33,  15,  22,  -8,  -8,  -21, -2,  10,  -8,  -17, 25,
    17,  -1,  -2,  -6,  -25, -22, 11,  -1,  5,   -23, -3,  -10, 3,   -58, -2,  28,  -6,  4,   -20, 0,   -19, -6,  -5,  -11, 4,   -42, -14, -14, 21,  14,  -34,
    15,  10,  -1,  11,  -31, -2,  8,   -12, 20,  17,  -10, -50, -12, 6,   22,  21,  -20, -14, -56, 12,  -25, 25,  -42, 2,   -4,  -30, -42, -49, 16,  -25, -10,
    -24, -8,  8,   20,  20,  5,   35,  7,   22,  19,  11,  16,  -5,  17,  -17, 8,   -5,  9,   -8,  28,  -4,  -31, 18,  19,  -1,  -5,  -18, -4,  -9,  38,  -5,
    -24, 0,   -5,  7,   -1,  6,   15,  -9,  -7,  -4,  -11, 10,  -29, -7,  -13, 7,   1,   2,   19,  6,   1,   17,  -24, 9,   25,  -19, 16,  23,  -27, -15, 8,
    7,   5,   0,   4,   16,  12,  -3,  6,   5,   -17, 16,  13,  26,  -30, -9,  -1,  10,  -10, -14, -13, -1,  23,  -7,  -11, -14, 15,  -2,  16,  2,   -30, 10,
    1,   2,   -2,  -4,  19,  -20, 22,  3,   45,  -1,  1,   5,   1,   2,   -1,  7,   -6,  14,  15,  -6,  27,  -7,  -14, -6,  -14, -23, -11, -1,  -5,  -20, 5,
    -15, -27, -5,  -15, 11,  -15, -15, 36,  6,   -19, 6,   40,  -7,  5,   19,  45,  14,  -9,  -30, 12,  15,  16,  10,  -12, 2,   27,  -4,  3,   3,   33,  10,
    15,  -4,  -7,  -10, -22, -33, -11, -1,  9,   0,   -17, -7,  -13, -18, -4,  -13, -35, -27, 6,   -19, -7,  7,   -19, 8,   -18, -3,  -55, 12,  15,  14,  -34,
    -11, 13,  58,  -2,  -2,  -30, 27,  -52, -24, -55, -52, -21, 15,  16,  2,   42,  -9,  20,  -46, 17,  29,  -17, -15, -1,  2,   -24, -35, 8,   12,  0,   10,
    6,   3,   -43, -12, 2,   26,  -3,  46,  0,   58,  -31, 24,  -12, -8,  7,   11,  -17, -31, -7,  54,  -42, -13, -17, 36,  9,   -11, -23, 22,  -2,  -41, 5,
    19,  -27, -19, 8,   -10, -30, 15,  23,  -14, 3,   -21, -92, -56, -23, -19, 5,   6,   2,   18,  25,  16,  12,  52,  -12, 2,   -12, 9,   -2,  -23, 14,  -30,
    10,  5,   -53, 19,  18,  23,  -28, 6,   -18, 44,  -28, -17, -36, -13, 5,   -19, -48, -11, 20,  -39, 48,  -6,  0,   -26, -1,  -23, -22, 13,  -25, 12,  -50,
    -10, 10,  -8,  20,  12,  -25, -28, -14, 4,   15,  -51, 2,   6,   6,   41,  3,   9,   -11, -2,  18,  -1,  -13, -58, 0,   44,  -13, -30, -42, 1,   10,  2,
    -4,  -31, -10, 2,   -11, 3,   0,   -7,  -21, -24, 8,   16,  2,   -33, 31,  -36, 25,  -13, -29, 19,  -2,  -29, -14, -39, -23, -4,  -42, -7,  -5,  -24, 20,
    -30, 17,  32,  -21, -20, 24,  18,  -7,  13,  6,   10,  -61, -29, -15, -18, -2,  -2,  8,   -11, -6,  -8,  3,   0,   -4,  10,  3,   32,  12,  -16, -13, 0,
    -5,  -1,  19,  -11, 8,   6,   -9,  10,  -9,  22,  -8,  15,  0,   2,   -4,  14,  12,  -6,  -3,  17,  7,   8,   13,  -5,  -17, 2,   -1,  -13, -14, 3,   18,
    13,  19,  -2,  -7,  11,  -26, 9,   -12, -5,  -15, 18,  4,   -10, -4,  5,   -19, 2,   15,  2,   -3,  -16, 9,   -7,  8,   2,   7,   7,   3,   16,  11,  2,
    16,  -12, 1,   7,   -1,  0,   7,   -10, -1,  6,   -17, 19,  -6,  5,   0,   -10, 13,  -11, -13, -8,  20,  21,  -1,  7,   1,   -20, 7,   -5,  0,   4,   18,
    15,  -29, 13,  -17, -20, -2,  8,   12,  20,  0,   -13, -6,  -10, -1,  6,   13,  -18, -31, 3,   -36, -15, -14, 16,  12,  -8,  3,   -31, 11,  7,   -3,  10,
    -18, -5,  -2,  7,   -11, -9,  13,  -27, 13,  8,   -17, 21,  -18, -11, 0,   17,  -2,  -11, -4,  2,   12,  -1,  15,  9,   -1,  14,  1,   7,   -38, -5,  10,
    -2,  15,  -7,  3,   25,  -7,  -12, 14,  13,  -22, 19,  8,   12,  -8,  -5,  14,  -8,  -4,  -11, 9,   -8,  6,   14,  5,   2,   4,   5,   -28, -14, 21,  -5,
    17,  3,   4,   -12, -10, 3,   14,  -23, 12,  -6,  -7,  8,   -9,  -2,  -8,  -8,  5,   -9,  2,   -29, -1,  10,  13,  2,   0,   12,  3,   34,  -7,  -41, -6,
    -13, 14,  9,   -13, 18,  9,   5,   2,   1,   -3,  -1,  -4,  5,   0,   -17, -11, -9,  -18, -7,  5,   -15, 9,   4,   -28, -31, 12,  2,   16,  8,   1,   1,
    -4,  20,  -8,  15,  -5,  2,   -3,  -6,  21,  -16, 15,  -19, 6,   5,   -18, 3,   -4,  15,  -24, 15,  -13, -10, 9,   9,   -6,  -5,  4,   13,  15,  14,  -6,
    15,  0,   -1,  20,  -2,  -4,  -12, -2,  22,  29,  -12, 10,  -7,  -22, 14,  9,   -36, 18,  12,  5,   -6,  11,  -2,  -6,  0,   -17, -13, -9,  -20, -1,  4,
    -13, -26, 4,   1,   -17, 9,   0,   -4,  10,  2,   -2,  -16, 13,  6,   12,  -4,  -14, 1,   11,  -6,  6,   0,   8,   -17, 0,   -10, -34, -5,  -5,  -21, 2,
    10,  12,  -1,  4,   -10, 14,  -1,  -5,  9,   11,  14,  2,   4,   0,   -2,  2,   8,   7,   -4,  0,   -28, -15, 10,  -7,  5,   31,  4,   -45, -7,  -10, 24,
    -12, 0,   -4,  -6,  18,  -20, -9,  -10, 0,   18,  13,  -3,  16,  5,   -2,  -7,  -11, 9,   12,  -2,  -29, -17, -6,  3,   12,  -13, -38, 10,  -12, -2,  -10,
    -10, -2,  -12, -2,  6,   -12, -6,  -11, 5,   -8,  -8,  3,   4,   -26, 14,  -2,  -12, 11,  -6,  10,  -11, -21, 6,   -14, 5,   -10, 6,   2,   4,   -18, -5,
    -10, 15,  -2,  -4,  41,  8,   14,  -12, 10,  -5,  7,   9,   0,   8,   6,   -1,  -2,  8,   11,  -13, -6,  3,   6,   -15, 6,   4,   5,   48,  7,   -10, 23,
    -6,  -11, 2,   22,  22,  -2,  5,   14,  0,   11,  -11, 2,   -25, -1,  10,  0,   11,  -14, 10,  -7,  5,   -22, 15,  14,  -18, -17, 5,   0,   14,  -2,  0,
    21,  4,   -15, 17,  26,  -16, -22, 16,  -14, -12, -4,  4,   4,   13,  -14, 11,  14,  -12, -17, 12,  -5,  11,  -13, 24,  7,   18,  -28, 5,   29,  20,  -13,
    -38, -18, -28, 9,   4,   -2,  16,  11,  -29, 7,   8,   -26, 6,   10,  2,   10,  20,  19,  11,  -11, -3,  0,   -31, 3,   -21, -19, 1,   -16, 17,  -7,  -14,
    28,  9,   -31, -8,  -7,  5,   -29, 16,  -46, -27, 25,  14,  -3,  -24, 10,  -2,  6,   36,  3,   3,   22,  18,  31,  -32, 10,  3,   -2,  12,  -22, -8,  -8,
    37,  -7,  -29, 12,  18,  15,  38,  9,   15,  22,  44,  -2,  44,  7,   -1,  -29, 1,   7,   26,  -29, -23, -35, 8,   15,  13,  3,   7,   -13, -14, 27,  -4,
    -9,  -16, 5,   0,   37,  19,  -14, 22,  3,   17,  4,   -22, -15, 1,   4,   5,   -8,  -25, -8,  2,   2,   -21, -12, 0,   -11, -11, 1,   3,   21,  -5,  0,
    9,   -9,  1,   11,  -11, -9,  -8,  13,  -12, -5,  5,   6,   4,   -8,  27,  5,   0,   7,   15,  0,   -2,  -9,  -10, 3,   5,   -4,  12,  0,   -3,  -3,  14,
    -12, -2,  -3,  1,   -6,  -1,  31,  4,   9,   -6,  -16, -22, -32, -4,  -9,  -12, -5,  26,  1,   -2,  6,   -5,  6,   21,  14,  8,   4,   0,   15,  -6,  7,
    15,  14,  15,  12,  -3,  -16, 16,  -8,  2,   -8,  -6,  -3,  -40, 8,   -18, -8,  8,   16,  13,  7,   -10, 0,   7,   11,  14,  1,   -5,  -5,  11,  12,  11,
    -9,  15,  3,   -23, 15,  14,  48,  1,   -29, -8,  9,   -26, -12, -11, -13, -7,  -1,  -10, -2,  -5,  -11, 0,   -8,  -14, -4,  0,   -1,  -5,  -8,  -6,  2,
    -11, -9,  -7,  4,   10,  -5,  -5,  0,   -6,  -11, -5,  0,   8,   -4,  2,   6,   -5,  2,   -1,  -1,  -7,  -2,  -15, 0,   -14, -8,  2,   -1,  -2,  -1,  -10,
    -6,  5,   5,   -8,  -10, -12, 7,   1,   -3,  -6,  -5,  -7,  2,   0,   6,   -12, 4,   10,  1,   1,   -3,  13,  -8,  7,   -15, -1,  -17, -9,  1,   -9,  -6,
    -10, 3,   -6,  0,   -6,  -9,  3,   -1,  9,   0,   10,  5,   2,   1,   -4,  -2,  -4,  -4,  -7,  -9,  -5,  -5,  3,   -10, -3,  -1,  2,   9,   7,   -12, 1,
    -4,  3,   -2,  -2,  -11, -5,  -4,  -9,  -7,  6,   -6,  -10, 2,   -3,  -6,  -8,  19,  22,  7,   7,   24,  31,  -1,  -12, 18,  15,  25,  19,  16,  20,  -16,
    -7,  9,   -8,  2,   -5,  -22, 25,  30,  -3,  -14, 10,  14,  30,  33,  13,  8,   -7,  6,   7,   -33, 4,   10,  -24, 19,  -16, 8,   -37, -31, 27,  26,  -9,
    -5,  -2,  29,  -19, 6,   -14, -3,  -4,  -33, -3,  3,   2,   -7,  -33, -13, -31, -12, -10, 9,   0,   -1,  -1,  -28, -11, 17,  38,  1,   25,  -30, -31, -27,
    23,  -22, 27,  -24, -16, 0,   -7,  -13, -2,  6,   -70, 2,   -30, 25,  -35, 24,  5,   13,  -18, 5,   6,   -42, -20, 12,  2,   11,  10,  16,  -1,  -14, -5,
    -8,  -11, -48, -15, -8,  -9,  8,   12,  -3,  -2,  38,  6,   -23, -8,  1,   3,   -26, -22, 10,  10,  15,  7,   -5,  11,  1,   14,  -15, -4,  -6,  1,   1,
    -22, 18,  -29, 10,  -14, 4,   3,   -25, 3,   -11, 8,   3,   -3,  -8,  2,   31,  2,   -20, -5,  4,   -1,  -9,  17,  2,   -5,  1,   -14, 5,   -16, 6,   14,
    1,   9,   11,  -8,  9,   -9,  -47, 16,  11,  -3,  -1,  2,   -3,  10,  -4,  -14, 30,  -18, 17,  12,  6,   -21, -6,  19,  29,  -39, -8,  12,  -12, 2,   3,
    -24, -15, 8,   0,   0,   -2,  2,   8,   -10, -2,  -12, -8,  5,   -3,  3,   -6,  -14, -25, 28,  -17, -10, -25, -17, 4,   4,   15,  2,   -18, 2,   -13, 8,
    -17, 3,   20,  -9,  -5,  0,   15,  9,   -11, 9,   25,  6,   2,   -12, -3,  10,  0,   8,   46,  3,   5,   -9,  -7,  20,  -4,  -25, 22,  19,  16,  4,   21,
    16,  -16, -13, -16, -32, -9,  40,  15,  9,   -6,  -3,  -20, -1,  5,   8,   10,  1,   10,  -9,  -29, 10,  9,   -8,  0,   2,   3,   -4,  7,   -40, 24,  20,
    -1,  -23, 3,   5,   2,   10,  -1,  12,  -2,  -13, -17, -26, 3,   47,  -4,  8,   -11, 10,  -12, -11, -29, -28, -2,  -8,  18,  -10, -12, -3,  -53, -55, -9,
    5,   18,  1,   6,   15,  14,  1,   11,  6,   11,  -31, -18, 10,  0,   7,   -2,  22,  -9,  -9,  0,   2,   -51, -9,  -17, -4,  27,  10,  11,  4,   -16, 9,
    -2,  10,  -5,  -5,  -27, -7,  -18, -16, -38, -10, 1,   9,   -11, -17, 5,   -4,  18,  -8,  -1,  3,   -29, -44, 21,  -13, 3,   23,  0,   3,   31,  -27, -36,
    10,  27,  12,  -37, -4,  -22, -22, -23, 14,  -6,  -29, 16,  -46, 3,   -8,  -35, -11, -39, -41, 35,  -38, 1,   -1,  -8,  -31, -32, 27,  -31, 1,   -37, -44,
    -2,  -5,  -4,  13,  -11, -8,  -25, -11, -5,  -3,  -17, -1,  -16, -38, -46, -27, 31,  -6,  -33, 20,  12,  5,   5,   10,  -10, 23,  29,  34,  25,  -14, -6,
    -20, 15,  8,   -48, -31, 14,  52,  -4,  -25, 1,   29,  -1,  -68, -13, 6,   -2,  -1,  -25, 0,   -6,  -31, -33, -9,  -9,  6,   14,  -19, 7,   -1,  31,  -22,
    8,   7,   -3,  -44, 1,   -9,  29,  5,   -9,  -1,  -16, -5,  6,   -14, -6,  36,  17,  32,  12,  -9,  -26, -31, -11, 4,   -20, 6,   26,  -22, -15, -21, -4,
    9,   -4,  -24, -5,  -7,  -6,  2,   2,   -17, -14, 6,   10,  -5,  1,   10,  13,  5,   -5,  3,   2,   -11, -4,  20,  -2,  18,  5,   0,   15,  -8,  -7,  -4,
    6,   17,  6,   1,   -8,  11,  -35, 17,  1,   20,  1,   18,  -23, 8,   2,   7,   6,   -6,  15,  -4,  9,   11,  15,  16,  5,   -5,  3,   13,  -16, -13, 18,
    2,   -2,  15,  21,  -19, 8,   -7,  -24, 2,   -5,  20,  12,  18,  9,   -14, 5,   11,  16,  17,  5,   23,  -2,  0,   -1,  4,   -1,  9,   0,   26,  11,  32,
    -8,  13,  14,  37,  -5,  16,  9,   2,   2,   -2,  11,  -5,  11,  -5,  -4,  -1,  2,   -1,  12,  4,   -10, 12,  17,  6,   20,  9,   20,  -6,  -8,  -5,  5,
    26,  -32, -4,  10,  6,   -10, -33, 9,   -10, -31, 8,   5,   -27, -25, -23, -2,  -6,  -35, -9,  35,  15,  -11, 10,  -12, -4,  20,  -23, -53, 32,  54,  -7,
    12,  -26, 3,   56,  -15, 38,  -42, -4,  -18, -25, -26, -4,  -18, -25, 11,  -19, -13, 26,  -24, 0,   16,  -11, 37,  -6,  30,  34,  -25, -64, -52, -14, 15,
    -72, 19,  2,   -30, 14,  -15, -6,  0,   4,   -43, -4,  58,  -12, -5,  17,  -7,  -10, -46, 13,  40,  -70, -6,  -11, 20,  3,   -45, 8,   -4,  -19, 9,   0,
    8,   3,   20,  -22, 18,  3,   8,   -17, -37, 11,  -48, 22,  -40, 9,   35,  3,   -9,  47,  19,  0,   23,  -44, -37, -22, -16, -35, -41, 6,   9,   10,  -74,
    -33, 20,  -35, 0,   -8,  0,   24,  -33, -46, 7,   4,   -3,  6,   -3,  -17, 7,   -44, -3,  7,   17,  -40, 7,   13,  -39, -15, 5,   -7,  1,   -4,  1,   16,
    9,   -22, 4,   8,   3,   26,  2,   -6,  -7,  15,  17,  -17, -7,  -12, -5,  7,   -1,  18,  -2,  -14, 5,   19,  -35, -14, -24, -3,  10,  2,   -7,  5,   -13,
    -6,  7,   11,  -29, 10,  -35, 23,  1,   -11, -5,  17,  11,  -4,  -4,  -18, -13, -5,  2,   -6,  17,  4,   24,  10,  17,  -9,  -20, -25, 17,  16,  -28, -7,
    6,   7,   6,   9,   -3,  34,  -24, -7,  4,   -24, 4,   -13, 3,   10,  6,   1,   6,   5,   -10, 8,   7,   -30, 20,  0,   -29, 8,   -8,  -2,  -5,  9,   4,
    -23, 10,  -27, -1,  3,   -6,  -21, -20, 23,  30,  -6,  -41, -1,  17,  -8,  10,  16,  3,   4,   -3,  6,   7,   -2,  2,   -13, -3,  15,  -14, 15,  -8,  -14,
    13,  -4,  -3,  27,  -21, 13,  0,   21,  4,   -8,  8,   -11, 9,   16,  -9,  5,   6,   -6,  -9,  17,  -17, 2,   17,  14,  12,  1,   11,  3,   16,  6,   -1,
    -1,  -5,  -18, 8,   7,   14,  -18, 14,  4,   -24, -14, -25, -12, -1,  -13, -1,  -28, -15, 3,   12,  -1,  -12, 4,   1,   -7,  11,  -19, -14, 10,  -8,  14,
    12,  1,   -5,  -3,  0,   11,  14,  2,   1,   -10, 9,   26,  -39, 12,  -18, -14, -4,  -13, 2,   -10, -25, 18,  1,   0,   -6,  10,  -16, 20,  -3,  15,  -17,
    5,   -1,  1,   9,   7,   -1,  -6,  -4,  -13, -26, 1,   26,  12,  -31, -6,  27,  4,   6,   14,  -1,  6,   0,   -10, -6,  1,   -11, 3,   5,   -11, -8,  -19,
    -17, 11,  -10, 5,   -8,  18,  13,  16,  -5,  0,   6,   18,  -12, 14,  2,   13,  -2,  -1,  12,  -12, 6,   15,  -20, 10,  -16, -1,  -15, -3,  15,  -27, -10,
    2,   -18, 10,  -3,  -4,  14,  -3,  -6,  -8,  11,  9,   0,   10,  -2,  19,  10,  -2,  6,   -2,  8,   19,  3,   32,  0,   12,  5,   -18, -8,  11,  -1,  17,
    -21, -9,  3,   9,   -9,  2,   -17, -6,  2,   -12, 6,   -5,  -12, -16, -9,  9,   14,  -16, -1,  30,  -4,  -7,  3,   8,   10,  1,   2,   0,   -2,  -21, 0,
    -31, -13, 8,   3,   -15, -13, -8,  -9,  -12, 16,  12,  -6,  16,  -14, 8,   -4,  3,   20,  9,   4,   -1,  -9,  -19, -8,  -11, 2,   -4,  0,   -6,  16,  -20,
    -20, 9,   -8,  2,   -15, -6,  5,   -14, -4,  3,   -19, 6,   -9,  -7,  -8,  2,   -6,  -5,  -20, 10,  -14, -6,  -13, 2,   2,   4,   -4,  2,   -22, -11, 0,
    -27, -11, 3,   -12, 10,  4,   8,   1,   4,   -7,  -1,  -2,  -5,  -17, -4,  -8,  -1,  -18, 8,   -9,  -3,  -7,  4,   26,  0,   -11, -5,  -9,  0,   -4,  -4,
    -7,  4,   -6,  2,   -3,  -7,  5,   -12, -10, -2,  0,   8,   -20, -19, -14, 3,   10,  3,   -11, -14, 6,   -4,  -7,  -4,  -14, -25, -11, -11, -3,  -10, -12,
    -10, -8,  -10, -11, -6,  -9,  3,   -14, -17, 2,   -6,  -26, 1,   -12, -5,  -11, 9,   -3,  -13, -10, 0,   5,   -7,  -42, 2,   -16, 14,  14,  17,  -5,  9,
    -7,  -4,  -3,  8,   9,   29,  18,  14,  -6,  5,   -9,  -10, 4,   15,  16,  -12, -20, 18,  5,   10,  12,  -8,  2,   -28, 13,  7,   -11, 8,   -4,  -1,  20,
    22,  0,   -10, -30, -2,  -23, 23,  23,  -23, -9,  -12, 6,   14,  10,  21,  -3,  9,   8,   7,   -14, 14,  25,  25,  -14, -8,  -9,  -14, -7,  12,  -20, 8,
    -5,  4,   -20, -17, -1,  -22, 13,  -39, 10,  26,  1,   -5,  1,   19,  12,  -5,  18,  -24, -22, 9,   5,   -9,  2,   -23, 22,  0,   -14, 40,  -41, -22, 21,
    13,  -10, -1,  -6,  11,  1,   -2,  39,  12,  8,   22,  -3,  4,   -1,  -20, 8,   -8,  -5,  17,  6,   -27, 8,   18,  35,  9,   -39, 20,  20,  19,  7,   -7,
    7,   23,  -4,  5,   18,  -2,  14,  10,  -4,  0,   -6,  28,  -17, 9,   -4,  2,   8,   6,   17,  -11, 7,   -5,  -5,  -1,  -8,  -2,  12,  9,   0,   -32, 0,
    -6,  -18, 1,   14,  -13, 17,  9,   15,  -8,  8,   2,   -2,  24,  3,   -5,  -19, -12, 23,  26,  -40, 6,   -1,  -4,  -8,  -1,  7,   -17, 12,  -5,  13,  -7,
    -14, 13,  -6,  32,  0,   12,  -11, -6,  0,   10,  9,   8,   -10, 6,   -5,  11,  -1,  4,   11,  11,  16,  1,   -8,  -8,  -1,  24,  -26, -1,  -11, -9,  -22,
    13,  5,   3,   -32, 6,   10,  9,   0,   -4,  -35, 12,  0,   19,  16,  -19, 8,   -8,  7,   -9,  -14, -14, 19,  -1,  -4,  8,   2,   2,   -27, -25, 23,  36,
    -7,  -13, 11,  6,   -8,  20,  -15, -3,  -6,  -17, -12, -16, -2,  4,   -1,  -79, -6,  1,   -7,  -50, 13,  0,   -22, 6,   -9,  -18, 16,  13,  8,   -7,  -25,
    -12, -13, -55, -13, -12, 3,   15,  7,   -26, -20, 18,  -26, 6,   -25, 16,  8,   23,  25,  -21, 15,  15,  -6,  6,   -33, -11, -5,  4,   -2,  7,   -2,  0,
    8,   3,   -19, -6,  -34, -2,  -48, 31,  2,   -2,  6,   -30, -33, -25, 12,  -11, -56, -44, -30, -6,  -30, -72, 27,  -5,  6,   15,  -16, -9,  -3,  -16, 21,
    -6,  -21, -41, 0,   -2,  32,  28,  34,  -8,  -39, -4,  -17, 4,   -40, -6,  6,   -29, -6,  16,  -33, -14, -40, 8,   -41, 9,   4,   21,  -16, 13,  -5,  23,
    3,   18,  -5,  -35, 10,  -24, 23,  27,  3,   2,   -8,  -8,  11,  -12, 14,  12,  10,  11,  13,  9,   -10, 10,  25,  -14, 16,  -11, -14, 10,  9,   -14, -24,
    6,   26,  -50, -11, 34,  11,  -1,  2,   -30, -3,  16,  -44, -10, -10, -1,  -57, -5,  -27, -16, 23,  5,   0,   1,   16,  12,  17,  13,  7,   -39, 0,   4,
    -35, -6,  -3,  -27, 1,   37,  -19, 4,   -4,  11,  -70, 19,  -19, -27, 12,  1,   -10, -13, 24,  4,   28,  -32, -33, -8,  31,  -22, 11,  -5,  -26, 7,   -10,
    -15, -1,  9,   6,   -29, 30,  -10, -7,  -12, 9,   -5,  23,  6,   -28, -26, -24, -5,  -13, 1,   4,   16,  -8,  -7,  -26, -4,  -4,  -14, 20,  -8,  1,   -8,
    -5,  -33, 14,  12,  15,  -8,  -30, 0,   -12, -30, -15, 0,   -16, 21,  0,   3,   5,   1,   8,   -27, 17,  9,   -9,  15,  -5,  23,  -45, -11, 11,  -13, 8,
    10,  -8,  0,   8,   21,  -13, -25, -4,  -6,  -7,  -31, 7,   4,   -29, 6,   21,  10,  13,  19,  2,   11,  -24, -12, -19, -10, 16,  -6,  3,   16,  -12, -6,
    22,  9,   -12, -8,  -4,  -6,  -25, -12, -12, 25,  -35, -8,  -24, -8,  -19, -23, -1,  -8,  7,   -16, 15,  -29, -29, -15, 13,  -38, 13,  -4,  -6,  0,   1,
    -6,  5,   -9,  14,  18,  2,   -8,  -30, 13,  -9,  25,  -17, -1,  -4,  18,  -12, -2,  7,   38,  -23, -11, -4,  -23, -10, 18,  8,   19,  1,   -44, 9,   11,
    5,   12,  -3,  7,   -10, -13, 23,  14,  7,   -4,  -30, -12, 7,   -32, -16, 6,   0,   24,  12,  14,  -25, 2,   13,  12,  -16, 13,  -19, -10, -9,  -23, 6,
    -5,  13,  1,   12,  14,  -10, 0,   -19, 10,  -25, 4,   0,   -1,  4,   12,  9,   9,   14,  -12, 10,  21,  -2,  -2,  1,   17,  -19, -23, -1,  -14, 6,   16,
    -15, -13, -6,  15,  -18, -9,  0,   6,   4,   -8,  14,  31,  -5,  6,   5,   -1,  -36, 10,  26,  -1,  -24, 3,   -12, -16, 18,  -24, -17, -5,  -22, 18,  -36,
    -7,  15,  8,   3,   12,  -1,  -3,  21,  3,   6,   6,   24,  -10, -21, 5,   19,  43,  3,   -7,  -4,  10,  5,   -26, 15,  -6,  29,  -28, -6,  -39, -26, -5,
    -12, 9,   -2,  20,  -10, -5,  13,  -18, 22,  12,  -14, 11,  -20, 15,  10,  17,  -3,  4,   -9,  7,   -27, -5,  -2,  10,  -7,  -31, -25, 4,   -4,  -16, 19,
    5,   12,  -15, 25,  12,  9,   1,   8,   -16, 13,  -2,  -11, -31, -26, -16, 19,  24,  21,  -39, 10,  -28, 18,  -27, -5,  -7,  -8,  6,   -8,  13,  -11, -13,
    -27, 5,   16,  10,  -3,  -27, 11,  -26, -12, 21,  -2,  -10, -17, 7,   7,   24,  7,   0,   13,  -5,  -5,  -4,  46,  31,  -12, 0,   8,   -7,  -2,  14,  -4,
    12,  4,   -4,  -19, -11, 13,  3,   14,  -5,  14,  -29, 6,   -2,  10,  45,  16,  19,  -7,  -6,  8,   14,  10,  7,   -6,  32,  -13, -36, -3,  5,   5,   -10,
    26,  -30, -9,  6,   6,   -2,  10,  -8,  1,   -8,  4,   -13, -9,  -4,  1,   -15, -24, 4,   56,  -15, 23,  -21, -12, 24,  0,   0,   5,   18,  -6,  0,   -12,
    7,   -1,  2,   -6,  -9,  20,  0,   4,   14,  -37, 2,   12,  29,  -14, 13,  5,   -29, -9,  6,   -9,  -34, 9,   13,  0,   -1,  -1,  -2,  -18, 1,   -30, -7,
    4,   13,  1,   2,   17,  0,   -10, -6,  10,  -1,  7,   4,   -7,  11,  -5,  -5,  -9,  -6,  -10, 11,  -18, 2,   13,  -21, -6,  -11, 9,   -10, 10,  7,   -12,
    9,   -6,  -4,  2,   5,   11,  13,  -7,  -16, 5,   11,  -20, 6,   10,  12,  11,  -3,  -5,  -9,  -21, 5,   -17, -5,  2,   27,  -7,  -10, 26,  -27, -11, -20,
    3,   10,  8,   16,  7,   17,  -14, -3,  -7,  9,   0,   -8,  -4,  1,   3,   -15, 16,  -8,  14,  -22, 30,  -4,  -31, -30, 7,   8,   22,  -10, -17, 13,  -13,
    10,  23,  19,  -33, 2,   -3,  -18, -19, -29, -18, -20, 12,  9,   12,  7,   -18, -19, 35,  -27, -21, -12, -6,  5,   -9,  -33, -1,  14,  0,   -19, 14,  -8,
    -4,  -10, 8,   -24, -19, -12, -24, 1,   -19, -9,  -2,  -6,  -2,  -8,  -17, -16, -9,  -38, -16, 22,  -17, 0,   -7,  -16, -55, -27, 0,   4,   -4,  -21, 7,
    4,   -8,  -25, 2,   -24, 17,  2,   -8,  -67, -12, 31,  0,   -14, 6,   13,  -24, 4,   1,   -20, -13, 27,  4,   21,  -19, 3,   -15, 5,   26,  -13, -7,  -10,
    12,  -35, -5,  14,  -35, 0,   -18, 2,   2,   4,   37,  -6,  3,   -26, 19,  36,  9,   9,   -42, 7,   23,  -23, 29,  -20, 5,   -27, 32,  30,  11,  8,   9,
    16,  -12, 0,   -15, 16,  0,   30,  1,   -1,  -17, 1,   1,   -28, 9,   -10, 11,  9,   -3,  12,  -15, 2,   -6,  11,  10,  10,  13,  6,   -13, 0,   5,   16,
    -10, 16,  5,   14,  5,   6,   -7,  1,   -21, 2,   -4,  14,  9,   -1,  0,   -9,  5,   8,   -14, -6,  3,   2,   1,   10,  16,  -5,  10,  -8,  -1,  14,  6,
    -9,  13,  -8,  -1,  -32, -6,  10,  -13, 10,  18,  5,   8,   -6,  1,   7,   -12, -1,  5,   -5,  8,   7,   13,  4,   -8,  4,   8,   -10, -6,  -4,  -12, -4,
    15,  9,   -7,  21,  23,  7,   -9,  12,  7,   -6,  -10, 0,   14,  5,   29,  -5,  12,  15,  4,   17,  11,  -11, 0,   -7,  0,   -6,  3,   -12, -13, 3,   -14,
    -24, 7,   -3,  -20, -1,  8,   -47, -21, -19, -36, -2,  -4,  16,  6,   1,   -21, -12, 24,  -27, 14,  -7,  -15, -13, 13,  5,   12,  12,  20,  1,   16,  10,
    6,   16,  -12, -2,  -10, 18,  8,   -13, -42, -9,  19,  -11, -15, -4,  -6,  23,  22,  9,   8,   0,   -38, 6,   23,  -30, 12,  -13, 25,  -11, 1,   -33, 23,
    -4,  -6,  -24, 3,   -28, 28,  8,   50,  -6,  -15, 2,   -8,  9,   13,  9,   6,   -3,  17,  -16, -6,  1,   -15, -22, 20,  4,   -1,  -8,  3,   -7,  -8,  37,
    0,   24,  16,  2,   -2,  37,  -10, -12, -12, -30, 5,   8,   21,  8,   7,   -24, -7,  2,   4,   -20, 3,   13,  -6,  -1,  8,   -37, 3,   7,   4,   -2,  12,
    10,  15,  -5,  -4,  -42, -31, 13,  -6,  -19, 1,   -9,  -2,  5,   -28, -10, 4,   22,  -22, -23, 8,   25,  -17, 8,   -23, 7,   21,  -10, -49, -9,  -3,  39,
    -14, 9,   12,  20,  7,   -11, 3,   33,  -32, -42, 5,   -23, 28,  15,  -16, 12,  -5,  -1,  -8,  -36, -21, -32, -7,  -17, 11,  -9,  31,  -28, 3,   -19, -14,
    15,  -6,  1,   23,  28,  -9,  -43, -11, -26, 20,  -2,  -14, -16, 2,   -32, 14,  -5,  -38, 0,   10,  -13, -28, 10,  -13, 8,   -4,  -14, 4,   -2,  1,   -6,
    -7,  -22, -10, -38, 18,  11,  21,  15,  34,  -39, 4,   2,   -8,  20,  -10, -26, -1,  -9,  35,  8,   -16, 5,   1,   -3,  -35, 26,  -5,  -2,  28,  -27, -39,
    2,   27,  -37, -5,  -42, -31, -12, -42, -15, -2,  -7,  4,   11,  -9,  6,   6,   0,   7,   1,   12,  16,  -6,  -21, -7,  20,  -16, -11, -4,  1,   11,  8,
    -1,  11,  18,  6,   6,   3,   -1,  18,  -6,  -7,  -4,  -1,  -8,  -6,  19,  -7,  9,   11,  -4,  -9,  9,   -5,  1,   14,  -39, -1,  -8,  -2,  -16, -24, 11,
    14,  1,   4,   7,   1,   -5,  -4,  7,   10,  -4,  15,  -3,  -15, 7,   10,  23,  21,  -1,  4,   16,  -10, 22,  0,   11,  27,  -10, -3,  -15, -16, 0,   9,
    -17, 0,   8,   3,   -8,  20,  -1,  30,  -8,  -9,  38,  9,   21,  -16, 21,  28,  -18, 12,  3,   -2,  3,   7,   2,   -12, -7,  9,   -38, -10, 7,   -6,  4,
    -4,  -13, 4,   -1,  -11, -9,  16,  -9,  -3,  5,   6,   38,  10,  -15, 2,   -3,  9,   0,   -2,  3,   9,   4,   5,   -9,  -15, -16, -7,  -9,  1,   -5,  -7,
    -6,  -11, -9,  -7,  -7,  0,   6,   -1,  -6,  -16, -1,  -3,  4,   -2,  1,   -10, -5,  2,   -10, 2,   -5,  3,   -6,  -12, -13, -8,  -1,  1,   0,   -7,  4,
    -7,  -5,  -1,  -11, 7,   3,   0,   -1,  -6,  -17, 0,   8,   -14, 4,   -5,  2,   -34, 0,   -10, -2,  5,   -11, -7,  1,   -9,  -12, -9,  -4,  5,   -8,  5,
    -17, -11, 3,   -10, -7,  -8,  5,   -8,  -13, -8,  6,   1,   -20, 2,   1,   3,   -3,  4,   -6,  -3,  -3,  -5,  2,   -8,  -1,  6,   0,   -6,  -13, -23, 2,
    3,   -10, 0,   -12, 2,   -17, -20, -3,  -7,  -2,  4,   0,   -9,  10,  1,   0,   -12, -9,  -12, 0,   26,  -12, 27,  -19, -3,  -18, -3,  -13, -12, -8,  1,
    -9,  -8,  -34, 4,   8,   1,   -48, -28, -57, -27, -3,  21,  19,  -6,  15,  1,   15,  -20, -8,  -42, -28, -14, -44, -23, 1,   -5,  -18, -4,  -17, -5,  -30,
    21,  35,  -9,  -9,  5,   -12, -12, 35,  -10, 9,   12,  12,  -1,  6,   -19, -38, 16,  16,  -16, -5,  -2,  18,  8,   31,  -7,  -2,  -17, -15, 30,  11,  -15,
    -7,  0,   11,  7,   -2,  -2,  -25, -4,  -4,  -4,  0,   -53, -33, 19,  -2,  -1,  20,  -65, 19,  -8,  33,  -12, -19, -32, 3,   58,  -25, -3,  -15, 3,   -31,
    4,   38,  -9,  -10, -7,  45,  5,   -9,  -19, -24, 24,  -14, 3,   24,  -13, -47, -15, -2,  13,  34,  17,  -5,  -13, -14, 0,   -30, -27, 12,  -12, 3,   7,
    -6,  16,  -7,  20,  7,   12,  8,   -3,  0,   5,   12,  -4,  15,  -27, 8,   -11, 27,  -10, 15,  -2,  1,   7,   -11, 9,   6,   -29, 19,  7,   12,  -49, -2,
    1,   7,   -24, -5,  -3,  -19, -4,  2,   -6,  10,  14,  2,   9,   9,   2,   8,   -12, -6,  16,  1,   -32, -7,  4,   3,   -6,  13,  6,   50,  45,  0,   5,
    -12, -7,  -31, 12,  -2,  24,  -28, 14,  -16, -27, -8,  16,  -2,  2,   19,  -7,  3,   -6,  7,   39,  2,   -6,  14,  -3,  11,  4,   13,  -24, 9,   -17, 2,
    3,   5,   4,   4,   -32, -5,  -12, -36, 8,   2,   -17, -7,  20,  -24, 17,  -16, -4,  18,  -27, 0,   -7,  -25, -4,  33,  16,  -2,  10,  4,   -6,  15,  19,
    7,   -3,  -12, -26, -4,  -1,  0,   -15, 9,   -7,  -9,  15,  5,   -8,  -10, 7,   23,  15,  30,  -12, -3,  -6,  -25, 7,   -8,  13,  -7,  6,   12,  -7,  -9,
    6,   11,  24,  -2,  4,   -1,  -22, 4,   -17, 9,   8,   -10, -3,  8,   11,  22,  -1,  -12, 8,   7,   13,  -10, 3,   -12, -4,  3,   -5,  3,   7,   -23, 6,
    11,  16,  -2,  11,  6,   -21, 8,   10,  -13, 10,  -4,  -8,  -20, -13, 15,  0,   -11, -7,  -13, 8,   7,   24,  1,   25,  6,   -14, 12,  -2,  13,  18,  -6,
    -19, -12, 21,  -3,  7,   -3,  1,   -10, -9,  7,   -1,  -30, -3,  9,   17,  -7,  4,   -4,  -13, 6,   -12, -21, 0,   -3,  5,   -32, -13, 28,  14,  12,  -28,
    22,  9,   -7,  11,  11,  4,   -7,  -40, 14,  -2,  -16, 6,   1,   -7,  22,  -6,  -8,  11,  -27, 9,   14,  -9,  -9,  6,   11,  6,   16,  4,   1,   9,   17,
    -14, -9,  3,   13,  2,   1,   18,  22,  11,  15,  12,  16,  -13, 17,  6,   14,  -14, -21, 1,   20,  5,   5,   21,  0,   -10, -9,  6,   -9,  9,   19,  9,
    -9,  -12, 3,   -15, -10, -3,  -35, 6,   21,  -8,  4,   0,   -15, 22,  38,  18,  -25, -14, 5,   -39, 0,   10,  -10, -2,  -35, -17, -5,  24,  11,  -24, -5,
    -22, -10, 14,  20,  22,  -8,  -12, -13, 11,  -12, -8,  1,   -11, -2,  12,  -24, 3,   2,   24,  -8,  30,  5,   6,   -20, 4,   26,  -18, -10, 7,   -10, -8,
    15,  -17, -10, 1,   4,   -10, -18, 10,  15,  -15, -18, 4,   -31, 5,   10,  -28, 7,   11,  -22, -9,  -6,  4,   -25, -27, 24,  16,  22,  -5,  -3,  -4,  -7,
    -28, 25,  10,  -8,  5,   17,  9,   -8,  -3,  -2,  17,  0,   9,   -2,  -4,  -21, 1,   -6,  3,   12,  -3,  -9,  -11, -2,  9,   -5,  24,  0,   -4,  14,  -2,
    -13, -42, -24, -14, 12,  -1,  -18, 2,   -15, -15, 21,  12,  6,   -14, 0,   0,   0,   8,   -14, 0,   -12, 9,   18,  -1,  -37, -5,  0,   2,   2,   -7,  4,
    36,  -32, -3,  -27, -20, -19, -5,  -3,  0,   -2,  35,  -7,  -11, -14, -11, -30, -13, 15,  -10, -3,  3,   14,  20,  13,  5,   -19, -28, -3,  3,   8,   1,
    -8,  -22, 2,   -16, -5,  18,  -11, -32, -1,  -16, 8,   19,  7,   5,   9,   -11, 12,  1,   -7,  22,  11,  20,  -28, 7,   11,  2,   -19, 0,   29,  16,  11,
    -27, -4,  -5,  -17, -17, 9,   -9,  -13, -22, 7,   0,   4,   -14, 35,  0,   2,   21,  14,  -19, -3,  13,  8,   13,  3,   -8,  -7,  -7,  12,  6,   -23, 28,
    8,   16,  -13, 5,   -5,  -16, -17, -13, 17,  -3,  -32, 12,  -10, 34,  -3,  18,  -15, -12, 5,   4,   15,  6,   -16, -14, 3,   23,  2,   -5,  -21, -5,  1,
    2,   2,   -2,  3,   18,  -2,  -14, -2,  -12, 16,  11,  -7,  -1,  -3,  23,  7,   12,  -5,  7,   -13, 1,   5,   -4,  -8,  -13, 3,   20,  3,   -20, 5,   -22,
    -3,  -25, 3,   7,   -9,  -24, -2,  -23, -1,  17,  11,  -24, -5,  17,  2,   6,   -8,  15,  2,   -25, 11,  5,   -3,  0,   1,   -13, -8,  7,   7,   3,   -9,
    7,   -10, 6,   -12, -2,  1,   21,  16,  3,   0,   4,   11,  2,   -11, -2,  2,   7,   -24, 12,  -16, 7,   6,   5,   -6,  20,  -21, 0,   2,   1,   -11, 0,
    -2,  8,   19,  15,  0,   -3,  3,   12,  10,  -5,  17,  -13, 9,   -3,  4,   -9,  -14, -15, 0,   -3,  -3,  -3,  -3,  5,   -7,  0,   18,  15,  19,  3,   8,
    -8,  4,   -5,  7,   5,   -11, 4,   5,   -6,  1,   -8,  -6,  -1,  -21, -5,  -18, -3,  -12, -40, -12, 5,   -4,  10,  -10, 9,   9,   1,   -14, 13,  12,  15,
    8,   -1,  -3,  -3,  12,  0,   21,  -19, 12,  0,   5,   14,  -3,  -6,  -8,  -23};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer2_bias_tensor[128] = {
    -7543464,  1065008,   -11406688, -4790480,  3349224,  -3415016, 6205008,   3116896,  -9872912,  2292440,   8873696,  -6642936, -5732128,
    1198648,   13569600,  9297232,   2391128,   1060896,  6768352,  -2251320,  8343248,  -13014480, 596240,    -4422456, 6005576,  9519280,
    2991480,   -3684352,  -15019080, -6410608,  10423920, 10257384, 15744848,  -6955448, -738104,   -8008120,  -509888,  -1685920, -16147824,
    -8976496,  6188560,   5740352,   7853920,   3532208,  -937536,  10249160,  8715384,  -8036904,  -1206872,  -5666336, -7115816, 15596816,
    1930584,   -12969248, -17128536, 2668688,   7193944,  -1620128, -10979040, 567456,   3606224,   5964456,   -7485896, 11106512, 14005472,
    -15874376, 7054136,   2864008,   -1317896,  12590944, 3439688,  248776,    -4924120, 2171136,   2321224,   2676912,  7767568,  -14480408,
    150088,    -3357448,  11231928,  -21333056, 4907672,  -4194240, -4019480,  9805064,  -6499016,  -1443312,  -625024,  -709320,  15894936,
    2234872,   -244664,   -359800,   20374960,  -4019480, -8756504, -21705192, 8807904,  -8968272,  -11945360, 12537488, -8645480, -158312,
    5325040,   74016,     -9946928,  7200112,   -6416776, 12202360, -1700312,  -649696,  -14266584, 9354800,   12950744, 803896,   4132560,
    4319656,   -7025352,  -6531912,  2300664,   -7165160, 6071368,  2909240,   8678376,  2950360,   1297336,   8433712};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer2_mul_tensor[128] = {
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056};

static PI_L2 int8_t AnomalyDetection_heavy_6_layer3_weight_tensor[16384] = {
    33,  -26, -16, 7,   -3,  -16, 3,   30,  -21, 30,  -15, -1,  -5,  -11, 16,  -13, 2,   0,   4,   9,   -5,  7,   17,  6,   -25, -9,  10,  -22, 15,  -19, 22,
    -12, -15, -6,  10,  -8,  -20, 17,  18,  -2,  9,   4,   20,  -2,  12,  -30, 2,   -9,  -26, 12,  13,  -6,  9,   -25, 9,   -2,  -2,  27,  37,  0,   -10, -2,
    4,   -10, 36,  17,  -21, 9,   17,  11,  25,  -16, -1,  11,  -6,  4,   -11, 10,  -7,  24,  -2,  -2,  12,  -3,  13,  4,   38,  17,  10,  -25, -4,  0,   5,
    12,  0,   -5,  -4,  -29, -11, 10,  50,  6,   -45, 11,  -13, 7,   -15, 9,   12,  -46, 30,  -16, 18,  10,  0,   -18, 11,  -7,  36,  -5,  0,   33,  4,   11,
    12,  4,   6,   3,   -30, 21,  10,  -19, 3,   24,  11,  -23, 21,  -29, 11,  22,  -2,  41,  10,  14,  3,   -4,  -5,  -12, 33,  -14, -24, -10, 8,   3,   -10,
    9,   -20, 24,  -5,  21,  48,  -8,  4,   2,   21,  -8,  9,   2,   1,   -6,  -10, 4,   10,  24,  36,  -6,  26,  -12, -44, 11,  8,   11,  -14, -1,  11,  -34,
    -32, 18,  18,  7,   -13, 22,  -8,  -20, 2,   3,   -3,  10,  -9,  1,   1,   -19, -14, -5,  20,  -39, 5,   -28, 14,  -39, -13, 11,  7,   8,   -25, -29, -22,
    22,  9,   3,   -5,  -10, 20,  7,   -8,  -22, 10,  -12, -18, 11,  16,  -22, 16,  6,   3,   26,  -8,  9,   -30, 13,  -24, 0,   9,   36,  -5,  3,   -29, 13,
    1,   -16, -7,  -10, 19,  -11, -2,  10,  -7,  16,  12,  2,   -16, -26, -6,  15,  5,   -13, 10,  12,  -28, -5,  10,  0,   8,   -18, -32, 2,   -5,  -8,  8,
    -15, -8,  38,  20,  -11, -16, 4,   -3,  14,  -4,  -9,  10,  -42, 33,  21,  -8,  3,   0,   -23, -27, -26, 17,  -1,  15,  -1,  -1,  -36, -6,  -6,  -32, -2,
    3,   12,  6,   -12, -25, -25, 6,   -1,  -15, -8,  23,  21,  -28, -8,  -8,  32,  -11, -47, -11, 3,   5,   -9,  7,   3,   2,   -5,  52,  -36, -21, -8,  -21,
    -14, 2,   12,  -7,  22,  7,   -7,  -19, 4,   9,   24,  -7,  0,   16,  0,   0,   4,   -17, -8,  -13, -16, 4,   14,  -15, 17,  -10, 9,   -3,  7,   -17, 44,
    8,   -19, -15, 1,   -5,  5,   22,  12,  -6,  -4,  -21, 0,   9,   10,  9,   -11, -18, 19,  -8,  -37, -6,  -34, -5,  -7,  -10, -7,  5,   -10, -17, 6,   9,
    3,   65,  -17, -15, -36, -28, 15,  -13, 2,   -14, 10,  -9,  -23, 9,   2,   -35, -14, 9,   -20, -6,  -5,  8,   -27, 12,  -6,  -2,  -9,  4,   -10, 1,   19,
    -12, 20,  6,   -28, -3,  11,  13,  -10, 1,   -14, -12, -9,  -23, -29, -21, -4,  -29, -5,  17,  9,   19,  -48, -1,  -2,  -16, 17,  14,  -16, -20, -27, 15,
    8,   -21, -2,  -12, -11, -3,  -12, 25,  -23, 9,   16,  -6,  -33, -4,  9,   7,   -20, 9,   5,   -14, 18,  -24, -36, 11,  -18, 6,   22,  -6,  9,   23,  -1,
    -4,  -17, 21,  14,  -6,  -35, -26, 18,  -1,  -24, 9,   16,  18,  9,   -7,  -5,  -43, 2,   -8,  16,  24,  -23, -13, -26, 8,   20,  14,  11,  -39, 2,   23,
    13,  10,  2,   -16, 8,   41,  -33, -9,  -18, -7,  12,  -3,  12,  -38, 42,  -21, 33,  -12, 12,  10,  -32, 20,  8,   -32, 31,  -47, 22,  -7,  -1,  24,  15,
    41,  -6,  -9,  -11, -39, 12,  -27, 22,  48,  -45, 9,   4,   -2,  -6,  36,  8,   -55, 33,  -20, -35, 3,   -49, -18, 18,  -6,  3,   -23, -20, 18,  4,   -19,
    -37, -3,  -27, 23,  -39, 30,  7,   15,  -9,  -57, -4,  7,   5,   -25, 0,   -22, -29, 18,  -11, 5,   -29, -10, 10,  -22, 5,   30,  -36, -13, 16,  8,   24,
    -11, 50,  -21, -12, -11, 13,  -21, -6,  6,   -16, -13, 0,   3,   6,   -8,  -4,  0,   4,   -6,  23,  18,  -37, 3,   14,  -5,  -11, -26, -5,  20,  -23, -20,
    -29, -2,  10,  -13, 5,   -51, -21, -13, -7,  -21, 1,   34,  9,   14,  -11, -21, 29,  38,  47,  16,  1,   -2,  -11, 25,  13,  -9,  0,   -9,  -45, -60, 3,
    -3,  -2,  -7,  -23, -29, 1,   2,   -35, 26,  -31, 1,   32,  -61, -20, 10,  -24, 71,  -14, -8,  -6,  2,   -42, -28, 16,  23,  1,   21,  -16, 11,  -3,  -10,
    -9,  56,  -37, 29,  -16, -4,  -30, 17,  -6,  -22, -1,  14,  -42, -6,  20,  -62, 19,  -16, -10, -7,  -19, 2,   -23, 12,  2,   -59, 30,  13,  6,   68,  -28,
    1,   -18, -10, 23,  31,  -29, 50,  25,  23,  10,  0,   20,  -31, -33, -63, -14, -8,  -22, -28, 10,  -23, 23,  19,  -5,  4,   -37, 11,  4,   2,   -25, 6,
    24,  -1,  56,  -6,  0,   -42, -21, -7,  -30, -22, -8,  -7,  8,   -1,  25,  -1,  26,  -20, 6,   -13, -42, -24, -19, 7,   -19, 8,   3,   5,   -26, 7,   -5,
    -9,  13,  13,  -43, -25, -14, 6,   -61, -27, -13, -15, -21, -49, -22, -36, -3,  -17, -4,  -7,  -5,  3,   -4,  -19, -24, 20,  4,   -16, -8,  -3,  -17, -14,
    -13, 12,  2,   -14, -3,  22,  2,   18,  -28, -22, -41, 2,   0,   6,   -6,  33,  18,  3,   -17, 9,   -14, 12,  16,  8,   6,   -32, -33, 8,   -21, 28,  28,
    -10, 20,  -8,  -30, 9,   10,  9,   5,   -9,  -14, -4,  -5,  -3,  6,   23,  -21, 8,   -18, -24, -18, 7,   11,  13,  9,   10,  -12, -22, 6,   36,  -6,  -22,
    10,  4,   -28, -16, 18,  6,   36,  3,   -15, -3,  -22, 15,  -1,  11,  6,   5,   -7,  -18, 0,   44,  10,  -1,  3,   16,  -12, 28,  -19, 20,  -3,  -19, 7,
    13,  11,  -16, 7,   22,  -9,  -11, 11,  2,   13,  -6,  -2,  -13, 0,   -24, 8,   15,  8,   7,   -15, 19,  7,   12,  45,  22,  -15, -17, 4,   12,  -10, 26,
    28,  -2,  7,   11,  -20, 11,  0,   -15, -5,  -9,  9,   -22, 11,  2,   27,  -17, 26,  5,   4,   7,   -11, 27,  15,  -1,  -20, -7,  -12, -4,  3,   14,  -7,
    -2,  9,   -7,  20,  52,  -4,  3,   15,  -10, 4,   -4,  8,   8,   -2,  27,  -24, 29,  15,  7,   -7,  -4,  -20, 40,  9,   19,  27,  11,  3,   -9,  24,  8,
    -2,  -11, 10,  -12, -37, 14,  -18, 4,   0,   10,  60,  -18, -8,  -2,  9,   6,   1,   3,   -24, 8,   14,  6,   -10, 8,   26,  -19, -2,  0,   5,   42,  -18,
    39,  -36, 0,   10,  34,  6,   -19, 0,   -13, -4,  -22, 6,   -1,  -5,  -3,  -2,  6,   2,   12,  27,  10,  -13, 10,  -8,  -6,  -1,  16,  28,  -11, -6,  8,
    -51, -26, -8,  14,  -30, 9,   -22, 17,  44,  -5,  -2,  12,  16,  9,   24,  -15, 21,  -32, -11, -10, -2,  -25, 1,   1,   -2,  2,   25,  -1,  -29, 20,  -3,
    8,   10,  -30, -8,  12,  -26, 3,   -3,  -16, 0,   37,  24,  -16, -17, 11,  -26, -27, 2,   24,  21,  -51, -15, -33, -5,  6,   -5,  -25, 5,   2,   -62, -15,
    -14, -32, 13,  5,   0,   -21, 32,  20,  -2,  14,  1,   16,  -9,  26,  -10, 13,  48,  1,   16,  11,  17,  9,   4,   5,   8,   36,  -6,  -24, -5,  17,  -1,
    -18, 9,   -19, 14,  -8,  1,   24,  -10, -5,  -3,  27,  12,  -2,  7,   16,  -1,  -7,  1,   1,   19,  29,  -7,  30,  -17, -30, 7,   2,   27,  -1,  12,  -1,
    -31, -31, 15,  26,  5,   -5,  19,  -15, -16, 11,  11,  -12, 17,  -11, -21, 17,  -21, -10, -3,  4,   -19, -2,  -17, 23,  -16, 2,   3,   5,   -1,  -41, -28,
    -14, 43,  10,  1,   -2,  2,   9,   16,  3,   -13, 3,   -9,  -38, 14,  -1,  -20, 14,  4,   13,  11,  -17, 33,  -26, 6,   -18, -14, 4,   27,  6,   13,  -26,
    5,   1,   -25, -20, -16, 7,   -15, -16, 12,  25,  -17, -4,  6,   5,   -26, -4,  19,  -11, 15,  0,   -31, 4,   -29, 9,   -12, 9,   9,   -6,  -4,  -16, 6,
    26,  6,   -3,  5,   11,  -5,  32,  -8,  18,  -4,  -26, 0,   18,  3,   -4,  11,  14,  9,   -9,  9,   8,   -9,  8,   0,   -15, -3,  -23, 6,   1,   -5,  8,
    -13, 7,   -1,  14,  37,  32,  -14, -5,  -3,  5,   -2,  15,  32,  4,   4,   1,   -6,  14,  -8,  -24, 7,   -10, 6,   -17, 9,   7,   25,  -17, 6,   8,   8,
    -7,  -7,  33,  19,  1,   -29, -3,  2,   -5,  10,  4,   0,   -1,  2,   -15, 20,  36,  -13, 3,   11,  -4,  3,   -11, 9,   9,   -15, 33,  -11, 25,  17,  -2,
    -28, 5,   -1,  27,  8,   3,   14,  17,  9,   1,   12,  8,   -8,  0,   5,   -1,  -1,  -7,  2,   -3,  -5,  -5,  6,   -2,  7,   4,   -7,  -18, -2,  4,   -11,
    6,   11,  -9,  -15, -6,  -4,  0,   3,   2,   -13, -5,  -1,  0,   -6,  0,   3,   7,   -6,  -3,  -14, 4,   -3,  -9,  -16, -3,  -11, -16, -13, -10, 10,  -3,
    -10, 3,   4,   -7,  8,   -12, -9,  -2,  -12, -12, 3,   6,   -13, -9,  2,   -11, 4,   -5,  -10, -11, -11, -17, 1,   -4,  1,   -3,  -1,  4,   -12, -12, -4,
    1,   -1,  -7,  -11, 1,   6,   8,   -14, 5,   4,   -6,  -8,  -7,  -12, -17, -3,  5,   -4,  -7,  5,   -10, -13, -8,  -1,  -15, -9,  -3,  -3,  -4,  -6,  0,
    -4,  0,   -5,  -8,  6,   4,   8,   -11, -10, 9,   -18, -2,  -3,  -2,  -10, -6,  -12, -22, 25,  20,  -13, 10,  21,  17,  -11, 7,   -34, 19,  30,  -17, 31,
    -3,  11,  5,   2,   14,  -11, 30,  10,  -19, -7,  13,  13,  -15, 26,  -25, 16,  -10, 12,  30,  -3,  -9,  -5,  33,  -7,  -13, -6,  9,   -15, -20, 3,   1,
    13,  28,  2,   22,  8,   -1,  6,   -9,  26,  -13, 14,  -5,  -31, -36, 15,  31,  8,   -10, 21,  -18, -23, 10,  11,  -9,  10,  8,   1,   20,  6,   -1,  4,
    17,  -5,  6,   -12, 32,  -20, 9,   15,  7,   12,  -38, -17, -4,  42,  4,   3,   4,   11,  -3,  3,   -3,  15,  3,   -17, -39, 3,   -7,  -8,  17,  2,   -4,
    15,  -9,  16,  -23, 15,  -14, -14, -1,  34,  -10, 3,   -46, -1,  5,   -7,  -3,  3,   5,   -5,  -9,  5,   -11, 48,  31,  -33, 2,   40,  7,   -31, 20,  -25,
    -19, 54,  -19, 38,  -6,  -4,  24,  7,   0,   -4,  11,  15,  -31, 11,  6,   -7,  -16, 18,  16,  -23, -13, -12, 22,  6,   -32, 22,  0,   -4,  -45, -13, 62,
    -15, -23, -14, -14, -8,  -19, -2,  20,  33,  12,  -55, -12, 46,  -5,  23,  -3,  -8,  -9,  7,   25,  -23, 28,  -11, 1,   -28, 48,  35,  8,   -16, 20,  -17,
    58,  7,   9,   -24, 26,  37,  -8,  -29, -13, 6,   21,  6,   -13, -6,  -42, -19, 62,  78,  14,  18,  -3,  10,  -15, 9,   -1,  17,  -22, -26, -31, -4,  -40,
    -23, 41,  3,   -3,  -14, -13, 27,  -7,  25,  18,  -42, -57, 56,  -34, -3,  -44, -25, 8,   6,   -8,  -13, 14,  -4,  -55, 1,   -8,  8,   5,   -14, 7,   -4,
    -4,  -8,  -5,  -7,  -14, 3,   -2,  -10, -9,  0,   -10, -6,  3,   -6,  6,   4,   -12, -2,  -10, 1,   -7,  -4,  0,   -8,  2,   -4,  -10, -3,  -9,  -1,  -11,
    -11, -10, 3,   5,   3,   8,   -5,  -2,  0,   -4,  3,   9,   -7,  -5,  5,   -7,  2,   -6,  -8,  -2,  -3,  5,   -4,  6,   -9,  7,   -6,  3,   13,  -8,  -3,
    -7,  -13, -6,  -10, -12, 6,   6,   6,   -7,  -7,  6,   -8,  1,   -4,  5,   8,   -11, -2,  5,   -12, -7,  -2,  -13, 3,   1,   -3,  -2,  -6,  -6,  -2,  -11,
    6,   0,   -7,  -10, -10, -9,  -2,  -11, -5,  2,   -1,  -3,  -12, -14, -1,  -13, -13, -7,  -10, -4,  7,   5,   -7,  -9,  -13, -12, 4,   -6,  0,   8,   52,
    26,  -33, 15,  -40, 26,  -48, 34,  -7,  11,  69,  -10, -3,  -7,  30,  50,  -2,  -14, 4,   22,  9,   -2,  23,  38,  -10, 19,  7,   -38, -8,  -24, 25,  7,
    -5,  -63, 21,  37,  -31, -5,  -48, 74,  -11, -26, -27, -13, -21, 54,  -1,  -38, 28,  7,   -37, -32, 50,  -75, 18,  -12, -55, -25, 48,  14,  -22, 19,  5,
    13,  -46, -22, -6,  27,  -22, 32,  -15, 69,  27,  -32, -10, 11,  10,  29,  -18, 12,  11,  17,  8,   -12, 31,  -45, -33, 47,  38,  -14, -2,  -22, 13,  -45,
    12,  -1,  20,  3,   -40, -14, 12,  -10, -6,  33,  24,  -5,  7,   -14, 44,  -33, 21,  -30, -25, -23, 15,  20,  42,  -57, -11, -16, -63, -35, 13,  -19, -41,
    -44, 22,  -41, 37,  40,  -18, 12,  2,   4,   -25, 19,  -43, 17,  48,  -1,  38,  2,   25,  0,   9,   6,   3,   46,  10,  -13, 1,   25,  1,   -11, 14,  -25,
    32,  -17, 10,  43,  0,   3,   9,   31,  6,   -10, 0,   10,  -16, -17, 12,  -8,  22,  33,  -12, 34,  -11, -36, 6,   -14, 26,  -10, 16,  -5,  -41, -37, 32,
    27,  6,   3,   10,  -11, -21, 8,   1,   -7,  13,  -6,  -24, 10,  -27, -3,  -3,  15,  -10, 5,   -16, 26,  -18, 2,   0,   -4,  -2,  -24, -38, -4,  41,  12,
    12,  3,   -1,  5,   6,   7,   -6,  12,  -24, -29, 2,   0,   -11, 7,   15,  0,   10,  -15, 48,  -28, 20,  -16, -10, 4,   37,  -3,  4,   -41, -13, 6,   -8,
    -9,  -12, 21,  -25, -19, 16,  -9,  0,   4,   1,   -7,  -1,  -5,  4,   7,   -5,  -15, -4,  -9,  -10, -13, -4,  3,   -6,  -5,  -8,  3,   -16, -9,  -3,  4,
    -7,  2,   -8,  -2,  10,  -6,  -4,  -12, 5,   -3,  -15, 5,   -4,  2,   3,   8,   -16, -5,  -15, -1,  -11, 1,   8,   -7,  0,   7,   5,   3,   2,   -6,  -10,
    -6,  -1,  -1,  11,  -3,  1,   0,   -12, -4,  12,  1,   -10, -4,  -5,  -1,  -2,  -7,  -4,  -8,  -8,  2,   6,   2,   5,   -4,  3,   -6,  -3,  -10, 3,   -4,
    -7,  4,   -10, 5,   3,   -15, -4,  -8,  -6,  7,   -4,  -12, -2,  0,   -3,  6,   1,   -4,  -14, 4,   -7,  3,   -8,  10,  -5,  -1,  -8,  7,   5,   -10, -1,
    0,   -8,  3,   -4,  -6,  2,   0,   6,   -2,  1,   39,  -8,  0,   18,  -4,  -12, -10, 8,   -18, 21,  -7,  -17, 2,   -29, -5,  -13, 12,  14,  6,   14,  -9,
    9,   32,  8,   -9,  -3,  10,  -11, 8,   -7,  21,  -2,  -29, -10, 13,  -4,  -13, 10,  -1,  -2,  -3,  22,  -2,  -8,  14,  -14, -21, 3,   -18, 9,   6,   12,
    5,   -18, -8,  0,   5,   46,  26,  -12, -15, -5,  4,   -7,  22,  13,  -6,  -6,  7,   -8,  11,  -8,  -22, 8,   11,  18,  -14, 8,   2,   30,  -12, -12, 12,
    -4,  -2,  -6,  19,  10,  -4,  -6,  -14, -17, 0,   15,  15,  10,  2,   -16, 3,   23,  42,  7,   -26, 22,  -10, -5,  1,   -1,  7,   -30, 31,  -23, 7,   1,
    -14, -10, -15, 0,   15,  8,   -5,  19,  0,   12,  14,  19,  4,   -6,  68,  -26, -18, 28,  -21, -53, -11, 24,  -9,  25,  -5,  -27, 9,   4,   19,  3,   -21,
    9,   12,  13,  -57, 1,   -23, -2,  -10, -29, 27,  -23, -1,  4,   41,  -2,  -12, 5,   21,  -3,  -23, 12,  33,  7,   -9,  26,  9,   -9,  0,   11,  -21, 2,
    -38, 13,  20,  -10, 10,  -25, 8,   10,  2,   44,  12,  -20, -21, -16, 10,  -10, 36,  5,   -41, 14,  2,   -19, 10,  -3,  -20, 25,  -8,  25,  -25, 5,   -13,
    39,  -4,  -8,  19,  15,  7,   -36, 39,  15,  -26, -38, -9,  -23, -11, -1,  15,  -15, 0,   -21, -19, 28,  42,  10,  5,   20,  -16, 14,  -3,  16,  22,  -40,
    26,  -14, 30,  -1,  0,   -20, -15, -37, 47,  5,   -7,  35,  7,   -2,  25,  15,  11,  -7,  6,   -7,  -8,  -11, 3,   -5,  3,   -11, -7,  -4,  -11, -10, -14,
    4,   0,   -11, 0,   -11, -3,  -1,  5,   0,   9,   -10, 1,   -5,  -7,  -11, -7,  10,  -9,  0,   -1,  9,   -6,  5,   -8,  -9,  -4,  -1,  -12, -4,  -8,  -2,
    1,   -6,  9,   -5,  9,   2,   -10, -5,  -3,  -9,  -6,  -1,  -8,  -4,  0,   9,   -10, -10, -11, -5,  -10, -11, -6,  -6,  -2,  -14, -5,  4,   0,   -15, 2,
    5,   6,   -9,  -6,  6,   1,   -12, 6,   5,   -19, 6,   3,   -2,  0,   -1,  -8,  -3,  -1,  -9,  -3,  2,   -1,  -12, 1,   11,  -3,  -9,  0,   -12, -8,  -4,
    10,  -12, -12, 5,   -1,  4,   12,  -8,  -12, -8,  -1,  -3,  3,   -5,  0,   -6,  0,   0,   -5,  -10, -2,  -8,  -24, 26,  32,  -16, 24,  21,  14,  -39, 12,
    -36, 16,  25,  -18, 28,  14,  23,  14,  1,   -2,  -2,  42,  13,  18,  -16, -7,  9,   0,   29,  -27, 37,  -26, 16,  35,  -1,  -13, 10,  37,  -9,  -21, -6,
    13,  -5,  -20, 5,   1,   42,  28,  -5,  56,  5,   -59, 0,   9,   24,  2,   7,   -2,  -58, -31, 40,  29,  13,  -16, 28,  -7,  -57, 15,  13,  1,   10,  -16,
    -8,  5,   -24, -16, -17, 7,   -28, 9,   -12, 44,  -26, -11, 0,   -13, 20,  -43, -9,  -20, 52,  -6,  14,  -4,  -5,  0,   6,   4,   -16, 7,   -21, -34, 20,
    5,   -8,  14,  -1,  9,   16,  -1,  46,  -20, 15,  -26, -21, 13,  28,  8,   15,  -43, -3,  -2,  -10, -34, -10, 16,  -19, -7,  8,   -25, 15,  22,  -10, 10,
    -2,  0,   -31, 19,  -11, 12,  31,  -27, -1,  8,   21,  11,  0,   -14, -6,  38,  -6,  -27, 0,   1,   10,  -15, 24,  14,  23,  -23, 27,  27,  16,  -32, -33,
    -15, -1,  -41, -2,  56,  -9,  -34, -6,  1,   24,  48,  0,   -10, 0,   -28, 14,  2,   48,  -12, 13,  -11, -64, -15, 45,  14,  -8,  9,   4,   16,  -35, -11,
    13,  19,  21,  15,  -21, 58,  14,  -30, -14, 23,  14,  23,  -14, 33,  -18, 7,   3,   -6,  -8,  -52, 0,   -4,  58,  22,  3,   -1,  -6,  9,   18,  -9,  -4,
    21,  -35, -29, 10,  -34, 2,   14,  -23, 15,  21,  -14, 62,  -12, 25,  -16, -12, 29,  41,  8,   -27, -64, -3,  4,   -55, -31, 13,  7,   -47, -26, -1,  38,
    -17, -9,  24,  -6,  -36, 6,   2,   -22, 22,  -10, -33, -3,  -28, 5,   -9,  -10, 6,   20,  12,  -35, -1,  -15, 4,   -28, -30, 11,  -3,  3,   7,   32,  -24,
    -36, -10, 29,  -8,  -20, 8,   13,  -3,  -1,  22,  28,  -16, 6,   -17, -25, -9,  -34, -18, 11,  -1,  20,  -16, -8,  -13, 10,  41,  45,  -6,  -13, -3,  -5,
    -11, 24,  26,  -32, 12,  7,   -19, 22,  -12, -27, 6,   -1,  17,  -30, -16, -1,  0,   -7,  10,  8,   -11, 3,   -6,  22,  38,  0,   -20, -12, -23, -3,  -11,
    9,   -12, 0,   -16, -16, 26,  17,  17,  4,   8,   -11, 6,   -1,  5,   22,  -15, 23,  -15, 25,  21,  19,  -13, -4,  -28, 56,  -5,  9,   41,  26,  8,   17,
    13,  -1,  -3,  4,   -16, -11, 12,  3,   -15, -16, 17,  -15, -1,  -7,  -30, 10,  -16, -4,  -10, 12,  7,   7,   12,  -24, -2,  19,  -4,  -11, -2,  18,  -2,
    11,  -5,  15,  10,  -40, -9,  21,  5,   -24, -4,  11,  4,   -8,  -3,  0,   -3,  7,   -11, -12, 6,   -24, 15,  17,  7,   4,   -13, 1,   -1,  8,   26,  12,
    -12, -27, 6,   5,   -14, 7,   28,  4,   6,   12,  -4,  4,   -18, -21, 10,  -10, 12,  -11, 16,  3,   15,  -5,  -2,  9,   4,   -2,  -7,  21,  28,  -4,  -19,
    3,   7,   5,   -3,  -4,  7,   9,   1,   1,   27,  50,  4,   18,  15,  -15, 2,   -8,  -1,  13,  -11, 31,  -8,  8,   20,  -12, -16, 9,   -3,  38,  5,   4,
    32,  0,   5,   1,   19,  12,  -9,  -18, 52,  40,  -20, 9,   3,   25,  -34, 35,  -19, 7,   74,  -15, 33,  -9,  28,  24,  23,  -8,  16,  32,  1,   15,  14,
    32,  2,   -11, 24,  -7,  23,  -20, -1,  4,   7,   -55, -6,  -23, -25, -19, -17, 21,  -30, -24, -7,  2,   -4,  20,  1,   39,  0,   12,  -33, 0,   44,  -48,
    17,  -19, -46, -24, 56,  2,   -8,  9,   20,  2,   -28, 10,  40,  15,  7,   7,   16,  48,  15,  -11, -3,  33,  11,  13,  -21, 8,   -5,  0,   24,  -22, -7,
    -41, -28, 39,  44,  10,  20,  -8,  9,   4,   16,  -8,  6,   25,  -36, -25, -1,  -32, -6,  15,  33,  -3,  23,  -7,  29,  -16, 38,  -27, -21, 15,  48,  9,
    30,  -31, -2,  19,  -25, 2,   10,  -14, -30, -26, 1,   -5,  -21, -12, -3,  -6,  2,   -2,  12,  -30, 16,  -1,  -13, 6,   -14, -3,  -18, 29,  9,   2,   0,
    -17, -7,  28,  -1,  -10, -18, 16,  -25, 30,  0,   25,  -17, -41, 8,   25,  -1,  -18, 6,   0,   2,   0,   22,  9,   10,  3,   -21, -14, -4,  -28, 10,  -2,
    -2,  10,  -13, -2,  -9,  -2,  39,  19,  -25, -24, 6,   7,   6,   25,  -8,  -5,  -4,  8,   -4,  19,  -12, -14, 19,  1,   6,   -15, 12,  -4,  27,  -9,  -25,
    7,   7,   -6,  0,   21,  22,  -2,  -17, -8,  -6,  -5,  14,  13,  -6,  -6,  -19, -9,  22,  28,  6,   -28, 24,  -9,  9,   -6,  -5,  5,   -38, 13,  -2,  12,
    24,  -2,  -18, -16, -27, 22,  14,  -8,  20,  3,   16,  9,   20,  9,   -5,  -3,  5,   4,   -6,  -7,  -2,  -9,  -10, -7,  2,   -8,  5,   -11, 14,  -19, -3,
    11,  -19, -14, -5,  -9,  -19, -4,  6,   -1,  7,   0,   -1,  -3,  9,   0,   6,   -4,  -10, -6,  -12, -6,  7,   -7,  -10, 6,   6,   6,   -13, 7,   -2,  2,
    0,   -9,  -8,  -1,  -9,  -13, 8,   -1,  3,   -1,  -5,  -1,  6,   -16, -8,  -8,  6,   2,   3,   -6,  2,   -9,  5,   -12, 0,   -10, -13, 1,   1,   -4,  4,
    -13, 17,  -3,  -3,  -2,  10,  1,   -9,  -4,  -6,  -11, 4,   -11, -9,  -1,  -7,  -10, 4,   6,   -17, -5,  -2,  6,   -3,  -2,  2,   5,   -11, 10,  -15, -17,
    12,  -12, 9,   -16, -2,  -2,  -16, -4,  -7,  3,   -11, -9,  18,  -10, -3,  -13, -1,  3,   -9,  -28, 19,  11,  -10, 9,   30,  2,   -16, 4,   -17, 15,  -6,
    2,   23,  2,   23,  -14, 0,   7,   2,   20,  -5,  -19, 7,   3,   -2,  4,   4,   -27, 33,  -5,  13,  27,  10,  -5,  -8,  19,  5,   -12, 16,  13,  -10, -11,
    -1,  5,   9,   31,  -5,  48,  -16, -21, 8,   -5,  26,  -15, 7,   8,   -23, -26, 7,   20,  13,  -7,  9,   -12, -17, 19,  -3,  -12, 10,  -18, 5,   18,  6,
    14,  -9,  1,   0,   10,  -9,  25,  -21, -16, 1,   9,   1,   -33, -24, -31, 32,  8,   8,   7,   -10, 4,   -5,  1,   9,   19,  -17, -37, 6,   30,  -6,  -2,
    -10, 5,   16,  -17, 26,  -27, 0,   -15, -7,  15,  33,  3,   6,   -26, 5,   5,   -15, -17, -2,  19,  -14, -3,  12,  -37, 32,  27,  -10, 14,  16,  8,   -19,
    20,  -36, 1,   40,  -6,  28,  2,   33,  -1,  -13, 11,  -1,  35,  11,  3,   1,   11,  9,   -9,  18,  -37, 26,  -6,  14,  35,  -6,  -15, -5,  29,  6,   6,
    1,   25,  0,   -8,  13,  1,   8,   38,  4,   31,  6,   -5,  -7,  -16, 37,  -6,  -2,  1,   -34, -21, 32,  22,  15,  3,   6,   -17, -33, 5,   4,   -4,  13,
    1,   -23, 35,  -8,  -8,  -6,  4,   -15, 5,   -10, 15,  -8,  -1,  2,   12,  13,  -31, -29, -21, 50,  -1,  10,  -6,  -5,  16,  5,   10,  3,   19,  -14, -37,
    15,  -31, -13, 17,  -2,  4,   7,   -18, 20,  -26, 26,  -22, -9,  16,  34,  1,   6,   -46, -13, 4,   -15, -21, -8,  19,  -17, 2,   11,  74,  0,   -14, -4,
    -5,  -39, -22, 15,  -2,  3,   -5,  -38, -6,  14,  17,  -11, -27, 18,  -10, 3,   -25, -4,  35,  10,  -6,  2,   16,  -1,  20,  36,  56,  -8,  -22, -1,  27,
    -12, -10, 7,   13,  -30, -9,  32,  24,  -17, 14,  6,   0,   3,   -25, -9,  14,  -19, 16,  -2,  -19, 0,   5,   56,  32,  -12, -25, -20, 1,   -10, 52,  44,
    -6,  16,  -1,  -21, 26,  10,  -24, 8,   17,  19,  -53, -8,  -8,  36,  -17, 1,   11,  23,  5,   -22, 46,  27,  -55, -42, -44, -29, -23, 8,   22,  4,   -10,
    -21, -30, 14,  9,   14,  29,  12,  -21, -1,  -2,  23,  9,   -19, 67,  -7,  38,  6,   -2,  -14, -26, 15,  46,  -3,  18,  31,  4,   2,   19,  17,  10,  -13,
    -13, -4,  -13, -10, 11,  0,   8,   -13, -16, -6,  -9,  6,   -3,  -4,  -5,  -19, 26,  -13, 2,   -3,  -19, -20, 12,  -26, 5,   -2,  5,   -17, 13,  -25, -6,
    5,   -13, -1,  -4,  -4,  -5,  -4,  13,  -3,  -5,  -11, 8,   14,  16,  -15, -25, 8,   -6,  -5,  -2,  5,   -2,  -18, -2,  -15, -4,  11,  -13, 19,  -4,  -17,
    -3,  -20, -6,  -12, 11,  -20, 4,   19,  12,  -24, 4,   -1,  4,   12,  17,  -15, -10, -25, -39, -8,  -22, 3,   -4,  10,  -4,  17,  27,  -21, 7,   -20, 5,
    8,   5,   -6,  -8,  -12, 0,   -11, 2,   -10, -14, 24,  -9,  -9,  -3,  -31, -34, -9,  27,  -23, 4,   6,   31,  -4,  21,  -4,  -29, 3,   -9,  6,   7,   -5,
    -11, -13, -3,  25,  -10, -26, -14, 0,   -12, 27,  10,  -31, 7,   22,  -29, -20, -12, -5,  -19, 1,   -3,  10,  -14, -7,  18,  -31, -8,  -14, -25, -10, 0,
    -4,  43,  -38, 1,   -22, 9,   -9,  2,   1,   -48, 6,   4,   -24, 17,  -18, 6,   -10, -43, 11,  -30, -9,  4,   -41, -7,  7,   3,   14,  1,   0,   0,   -11,
    -7,  -8,  12,  16,  -3,  -10, -21, -21, 27,  -3,  -14, 12,  15,  45,  9,   8,   -14, -32, -5,  -3,  26,  -23, -21, -18, 19,  7,   -28, 20,  -18, 6,   9,
    -7,  -31, 3,   -19, 8,   -25, -10, 3,   6,   -16, -11, -14, -5,  -6,  -14, -34, -10, -2,  -40, 43,  32,  -59, 11,  14,  12,  -13, -38, -6,  30,  3,   15,
    7,   -19, 6,   19,  -13, 0,   22,  -5,  38,  -22, -25, 6,   9,   -11, -8,  2,   -34, 30,  -8,  -15, -2,  -17, 6,   -8,  20,  -1,  1,   -7,  -8,  -2,  19,
    9,   -9,  -12, 19,  -19, 25,  -22, 13,  -6,  -31, -3,  24,  -6,  -20, 2,   16,  -1,  12,  0,   16,  1,   10,  -31, -8,  7,   -21, 16,  17,  12,  1,   -16,
    13,  -4,  9,   60,  24,  0,   -18, -14, 2,   -13, 34,  29,  0,   4,   8,   0,   18,  -5,  -10, 5,   -1,  17,  2,   4,   3,   27,  -23, -5,  15,  -15, 13,
    5,   26,  28,  14,  -15, 1,   11,  -4,  14,  -4,  12,  -5,  -17, 6,   30,  41,  13,  -73, 7,   10,  10,  -13, 12,  -6,  -5,  2,   -25, 17,  17,  -5,  -33,
    14,  -13, 33,  -3,  1,   33,  14,  7,   14,  7,   13,  14,  48,  -7,  -23, -5,  1,   -42, -15, 20,  -12, 45,  -10, -14, 0,   -20, 7,   -9,  17,  -11, 14,
    10,  -38, 0,   31,  -1,  -13, -19, 0,   -3,  28,  -6,  12,  -5,  -31, 8,   19,  11,  -16, 10,  2,   -12, 0,   13,  12,  -9,  -2,  -20, -12, 12,  -35, 8,
    5,   10,  13,  -23, 10,  3,   -1,  45,  37,  -7,  -16, 9,   11,  -4,  29,  38,  -6,  11,  13,  -13, 16,  1,   -6,  14,  4,   20,  -11, 8,   7,   20,  1,
    9,   6,   2,   6,   -3,  42,  29,  1,   -30, -6,  -13, -9,  -8,  2,   12,  4,   4,   0,   23,  44,  9,   14,  24,  -10, -8,  0,   10,  2,   0,   30,  -21,
    28,  14,  -3,  -13, 10,  -2,  48,  10,  2,   18,  17,  -3,  0,   24,  9,   -1,  -29, 23,  26,  -9,  3,   15,  15,  -19, 13,  -39, -3,  44,  4,   30,  9,
    29,  -8,  -8,  17,  6,   35,  10,  -10, -9,  8,   9,   -15, 9,   -31, 22,  -5,  11,  23,  8,   -6,  9,   23,  -7,  4,   12,  0,   -7,  -7,  12,  0,   19,
    19,  3,   28,  6,   -6,  -6,  -5,  31,  -6,  10,  -7,  -36, -27, 25,  29,  10,  4,   21,  -10, -15, 8,   11,  1,   12,  2,   -14, 22,  -16, -3,  4,   3,
    -5,  6,   -18, 20,  -2,  -8,  2,   9,   4,   -30, -15, 1,   44,  6,   -8,  8,   4,   15,  5,   7,   -3,  3,   -19, -35, 1,   -28, -14, 12,  9,   16,  11,
    -7,  12,  -24, 24,  -13, -17, 15,  26,  6,   9,   -53, -13, -3,  -22, -16, -8,  8,   -13, -11, -4,  -38, 22,  14,  -6,  15,  20,  1,   -15, 9,   -22, 15,
    33,  -3,  20,  0,   25,  -9,  9,   2,   -11, 36,  4,   -32, -10, 18,  15,  -6,  11,  -26, 22,  -11, 1,   39,  8,   -3,  0,   33,  -2,  -7,  19,  11,  1,
    -10, 8,   -1,  24,  41,  -4,  29,  -4,  -13, 10,  1,   26,  -8,  4,   7,   -30, -22, 20,  41,  12,  0,   22,  -22, -18, 5,   15,  -6,  10,  -18, -12, 16,
    -21, 4,   -15, 9,   -9,  -6,  -6,  23,  -18, -17, 9,   -8,  -7,  -35, -22, -36, 35,  10,  7,   7,   -11, 19,  16,  11,  -5,  4,   -20, -36, 16,  6,   -21,
    15,  -1,  2,   2,   -20, -10, -28, 20,  -9,  -6,  3,   25,  7,   -2,  -22, 0,   -8,  -21, -7,  -22, 5,   -6,  -15, 5,   -22, -13, 14,  6,   14,  -20, -24,
    -17, -48, -1,  -14, -3,  13,  -6,  -20, -26, 46,  -1,  19,  0,   -5,  -28, 16,  -38, 7,   2,   -23, -28, 20,  40,  5,   -14, -64, 0,   -6,  -3,  -10, 3,
    7,   -13, 12,  -4,  -1,  -34, 21,  -44, 5,   -5,  -5,  -20, 14,  9,   -3,  -25, 3,   -26, -1,  -4,  37,  -15, 2,   -14, 30,  -31, 42,  -23, -13, -10, 62,
    -31, -36, -8,  8,   -12, 14,  3,   2,   4,   -5,  7,   -7,  3,   10,  6,   -4,  -12, -10, -16, -39, 9,   6,   -34, 11,  6,   22,  -11, -5,  -5,  -37, 11,
    9,   21,  -14, -45, -21, 1,   3,   -36, 54,  -11, -43, -2,  -8,  -5,  -14, -23, 5,   -14, -18, -4,  2,   31,  3,   17,  15,  -16, 3,   -2,  -19, 25,  22,
    -14, 3,   25,  18,  -10, 7,   -29, 10,  15,  0,   21,  2,   25,  -3,  -10, 15,  9,   22,  -2,  -22, -5,  11,  -4,  -5,  21,  -18, 18,  -7,  0,   25,  -1,
    -2,  -6,  9,   -8,  2,   8,   -18, -9,  3,   12,  7,   19,  26,  4,   22,  -4,  -14, 7,   -1,  20,  -9,  -7,  1,   -34, -19, 5,   11,  3,   9,   20,  -4,
    -21, 10,  13,  -4,  7,   -26, 9,   15,  -14, -2,  -9,  5,   -8,  1,   1,   29,  -7,  -12, -4,  12,  -2,  -32, -26, -29, 33,  10,  0,   -3,  10,  13,  14,
    -10, 5,   22,  -14, -27, 3,   12,  -1,  11,  8,   -4,  9,   0,   20,  -26, 6,   -5,  -10, 7,   19,  -7,  14,  -31, 0,   11,  -5,  5,   -9,  8,   -4,  -15,
    14,  58,  -37, -11, 14,  8,   -32, -14, 13,  -27, 43,  -24, -2,  2,   -45, 9,   -3,  14,  6,   15,  -5,  -43, -7,  11,  13,  -5,  -10, 16,  -33, 8,   -26,
    38,  1,   -18, -15, 22,  -3,  -29, 7,   8,   -21, 13,  22,  11,  -3,  2,   -27, 14,  -8,  -29, 13,  -4,  9,   -5,  -37, 20,  2,   9,   38,  -23, -27, -8,
    -28, 18,  -15, 35,  3,   -20, 6,   11,  -34, 19,  -7,  -21, 18,  -5,  28,  -28, 3,   12,  13,  -9,  8,   19,  5,   3,   -8,  20,  28,  12,  -25, -8,  -21,
    10,  17,  14,  8,   -7,  -41, -9,  33,  60,  4,   -65, 13,  -7,  -10, -3,  -16, 0,   -22, 20,  -12, 16,  5,   -6,  -30, 9,   -3,  12,  3,   4,   46,  5,
    2,   25,  22,  11,  15,  -28, -34, -23, 9,   3,   -49, -2,  8,   -9,  62,  -1,  -47, -12, -44, 14,  17,  16,  -11, -9,  6,   -16, 9,   43,  12,  -25, 8,
    -2,  -21, -32, -29, 24,  -31, 0,   8,   41,  5,   -76, -41, 42,  18,  -9,  7,   8,   8,   16,  -28, 15,  -4,  16,  18,  -42, -4,  9,   -32, -68, 11,  -13,
    17,  33,  25,  2,   4,   9,   -32, -24, -41, 21,  -44, -18, 16,  6,   -28, -10, -23, 21,  -9,  8,   13,  -34, -18, -17, 32,  9,   -7,  23,  30,  15,  17,
    50,  6,   22,  4,   1,   11,  -68, -27, 3,   -62, 27,  11,  -22, -1,  -1,  40,  -5,  24,  -9,  -40, -62, -15, -38, -2,  24,  16,  15,  -20, 2,   -9,  -52,
    26,  4,   -38, -21, 40,  -11, -32, -23, -6,  -9,  7,   -11, 3,   -15, 6,   2,   1,   2,   -1,  -8,  9,   -13, 21,  -6,  -7,  10,  -14, -7,  6,   20,  -15,
    1,   13,  19,  -16, -9,  -18, 14,  -3,  -9,  1,   3,   -7,  -9,  -7,  -6,  -14, -5,  -7,  3,   -11, -5,  -3,  -8,  1,   0,   -2,  2,   -10, -8,  -14, 0,
    8,   -12, -9,  -15, -7,  3,   7,   8,   -25, 2,   7,   -17, -7,  6,   -7,  -2,  10,  -3,  4,   5,   -17, -12, -10, 6,   0,   -8,  15,  -7,  0,   -20, -3,
    1,   7,   4,   -4,  6,   6,   -14, 5,   1,   -22, -4,  -16, 2,   -3,  -12, -2,  -2,  -18, -3,  1,   -13, -6,  -8,  -10, -2,  4,   -9,  -23, -3,  -12, -3,
    7,   -8,  0,   5,   -7,  9,   1,   6,   -18, 1,   -3,  4,   -3,  33,  -12, -12, -3,  -3,  -13, -13, -7,  -32, 40,  2,   -4,  -6,  -29, 0,   -6,  32,  -7,
    15,  -5,  -30, -5,  27,  1,   2,   -11, 14,  -30, 12,  -13, 11,  -2,  -30, 1,   21,  3,   -4,  8,   -2,  -13, 14,  15,  8,   -8,  0,   -30, 3,   -1,  -47,
    9,   9,   18,  12,  -28, 18,  -9,  6,   52,  -32, -14, -19, -10, 4,   -1,  31,  32,  -20, 6,   4,   -18, 21,  -9,  -14, 16,  9,   23,  -3,  7,   9,   20,
    -14, -8,  3,   1,   13,  -12, 42,  36,  16,  -22, -20, -2,  -8,  7,   5,   17,  0,   -4,  0,   25,  44,  16,  -70, 19,  3,   4,   -1,  -7,  1,   -10, 20,
    -20, 16,  7,   -12, -13, 11,  4,   38,  2,   10,  43,  18,  1,   25,  12,  15,  9,   -17, 32,  26,  -16, 1,   36,  10,  -17, 10,  -29, 1,   33,  -10, 32,
    12,  19,  -3,  0,   7,   -10, 33,  6,   -21, -12, 4,   5,   0,   13,  -34, 25,  -15, 6,   40,  -9,  3,   0,   26,  -9,  -4,  1,   5,   -7,  -7,  5,   4,
    11,  30,  3,   28,  -4,  -4,  2,   -1,  25,  -5,  13,  2,   -34, -34, 12,  22,  9,   -1,  24,  -12, -19, 13,  -3,  -10, 16,  -13, -6,  21,  -18, 8,   -13,
    20,  -13, -7,  -8,  28,  -7,  1,   15,  4,   -4,  -37, -16, -16, 34,  7,   -4,  9,   -4,  9,   7,   0,   5,   18,  -16, -37, 10,  -22, -10, 10,  5,   -8,
    10,  -4,  17,  -24, 19,  -6,  -13, 9,   37,  -1,  -5,  -34, -13, -4,  -9,  -5,  -17, 7,   -5,  -13, 6,   -25, 24,  18,  -8,  10,  27,  8,   -25, 14,  -27,
    6,   42,  -2,  34,  3,   11,  5,   -8,  10,  -12, 33,  2,   -13, 1,   12,  0,   -4,  17,  -31, 27,  -12, 0,   32,  -3,  -28, 4,   28,  -8,  -7,  -6,  13,
    -16, 0,   8,   10,  13,  30,  -5,  24,  -9,  -15, 9,   -8,  33,  -16, 9,   -9,  -38, -34, 17,  26,  7,   4,   10,  -22, -30, 11,  10,  4,   14,  2,   6,
    6,   -16, 4,   -10, 16,  -14, 9,   0,   31,  -12, -6,  0,   -10, 1,   -54, -14, -13, 31,  0,   8,   -8,  3,   -4,  12,  -5,  6,   19,  -18, -24, 16,  -2,
    -27, -2,  10,  1,   15,  -2,  26,  -30, 18,  -6,  -14, 14,  23,  5,   14,  -42, -6,  7,   -4,  -1,  -8,  -1,  -14, -1,  1,   -35, 13,  14,  3,   -1,  22,
    -4,  -15, 13,  -11, 10,  1,   -5,  33,  8,   23,  2,   -5,  3,   2,   21,  -7,  -15, -12, 12,  9,   -13, 26,  -10, 32,  2,   4,   33,  7,   -2,  -9,  25,
    12,  -13, 16,  -5,  11,  5,   11,  8,   35,  27,  9,   23,  8,   -32, 1,   1,   31,  -5,  3,   0,   -45, -21, 15,  34,  13,  -10, 6,   -15, -19, 17,  -2,
    -1,  1,   -21, 7,   -17, -18, 12,  -5,  18,  -16, -4,  -6,  21,  -23, -13, 6,   9,   4,   -33, -11, -31, 43,  14,  1,   5,   -2,  4,   4,   -4,  -12, 6,
    -22, -37, -1,  6,   9,   -4,  8,   17,  5,   -10, 42,  -22, 4,   -6,  0,   16,  24,  10,  8,   -26, -14, -1,  6,   -5,  -15, 23,  -4,  -7,  11,  -2,  4,
    -12, -10, -7,  7,   2,   -8,  -1,  6,   -8,  -11, 3,   -17, -13, -9,  -8,  -12, 0,   8,   -6,  -7,  10,  -20, -3,  -14, -9,  -14, 7,   4,   -1,  -3,  6,
    8,   4,   -11, -9,  -1,  -8,  6,   12,  -6,  -12, -2,  -2,  3,   -8,  4,   -9,  2,   1,   5,   6,   0,   -7,  -10, -11, 2,   -3,  7,   -7,  -7,  -1,  -6,
    -7,  7,   -20, 3,   -7,  -2,  -11, -7,  -3,  0,   2,   -12, 13,  -9,  2,   -10, -9,  -5,  -13, 3,   -8,  6,   -8,  -2,  -4,  12,  5,   -7,  -13, -17, 1,
    -14, -5,  -2,  -3,  -1,  6,   -14, 4,   -9,  5,   -10, 9,   5,   -7,  0,   -10, -4,  -4,  2,   4,   7,   -10, -9,  -3,  -7,  -3,  -3,  -11, -6,  -1,  -5,
    0,   -10, -17, 32,  13,  -8,  7,   28,  7,   -8,  -9,  -36, 16,  26,  4,   25,  0,   26,  -1,  9,   6,   2,   32,  -8,  -8,  -18, 15,  19,  -6,  13,  -28,
    39,  -8,  8,   32,  2,   -1,  -2,  16,  -9,  -4,  15,  1,   -11, 2,   0,   -2,  25,  35,  8,   41,  -4,  -26, 5,   11,  -16, 10,  3,   0,   -43, -29, 33,
    40,  11,  1,   24,  -6,  -35, 14,  4,   -13, 7,   -17, -27, 23,  -15, 3,   -15, 8,   -32, -3,  4,   24,  -10, -17, -2,  -4,  -4,  -36, -20, -27, 33,  -1,
    7,   8,   -13, 21,  3,   -7,  -5,  19,  -18, -23, 11,  22,  3,   19,  1,   -8,  -4,  -14, -6,  -26, 7,   -18, -4,  20,  20,  7,   11,  -42, -9,  -2,  -11,
    -34, -21, 24,  -8,  -14, 1,   -26, 32,  17,  -5,  4,   40,  7,   -15, 13,  -16, 18,  42,  4,   20,  -1,  21,  -9,  -4,  4,   1,   29,  -4,  -14, -5,  2,
    4,   -1,  26,  -19, 31,  -8,  10,  28,  5,   -17, -5,  20,  -10, -16, 0,   18,  -20, -14, 10,  7,   20,  37,  -1,  31,  4,   -25, 13,  -8,  0,   -21, 13,
    2,   -41, -31, 28,  29,  8,   10,  13,  -20, -34, 12,  -2,  1,   4,   -8,  5,   27,  -12, -6,  -10, 11,  3,   -5,  -8,  29,  -19, 7,   10,  -3,  6,   -53,
    -11, -13, 46,  -3,  9,   -2,  -1,  5,   6,   2,   10,  9,   -24, -36, 13,  0,   -14, 9,   -8,  2,   11,  -3,  42,  -19, 26,  -14, -18, 13,  31,  3,   12,
    -35, 3,   -7,  -5,  -5,  -3,  5,   -6,  -7,  -5,  78,  -14, -6,  45,  -47, -24, -36, 6,   4,   13,  2,   -17, 10,  27,  18,  16,  -18, 8,   -26, 4,   -19,
    3,   48,  -3,  -12, -21, -4,  25,  44,  -30, 24,  -13, -20, -10, 11,  -11, -22, 0,   69,  -6,  -30, -6,  33,  23,  -16, 1,   -31, 0,   -27, 33,  25,  -32,
    -20, 11,  -23, -1,  18,  24,  68,  -1,  5,   -33, -9,  2,   11,  18,  -2,  26,  -18, -11, 18,  11,  -26, -4,  15,  -13, 14,  -13, -43, 62,  26,  -7,  -10,
    6,   9,   -19, 54,  2,   -61, -31, 10,  16,  -21, -4,  15,  -22, 1,   -17, -71, 14,  -24, -1,  58,  -30, -20, 9,   4,   14,  -44, -54, 35,  -13, -22, 34,
    -9,  -40, -28, -15, 37,  -15, -10, 16,  -7,  8,   -4,  -41, -11, -24, 14,  -8,  -18, 13,  4,   -57, -12, 15,  -8,  37,  -9,  -32, 1,   -17, 13,  -13, 30,
    4,   9,   -5,  -42, -5,  35,  15,  -8,  -18, 7,   -11, 18,  -5,  27,  -3,  -32, -3,  18,  8,   -19, 1,   19,  -5,  2,   8,   9,   0,   10,  -13, -15, 11,
    -32, 10,  6,   20,  -6,  -18, 20,  -7,  10,  58,  40,  3,   -24, -3,  1,   -29, 19,  19,  -19, 6,   18,  -18, 8,   11,  -28, -3,  3,   16,  -15, -3,  16,
    9,   -15, 0,   8,   10,  -6,  -5,  28,  36,  -21, -15, -2,  8,   -9,  0,   1,   -4,  6,   12,  -6,  21,  46,  0,   16,  10,  0,   5,   2,   2,   1,   -10,
    39,  -16, 21,  18,  -5,  -17, -11, -13, 22,  4,   6,   35,  18,  6,   1,   19,  13,  13,  12,  -4,  38,  -29, -6,  10,  17,  -29, 20,  -9,  -2,  33,  1,
    26,  -8,  33,  8,   13,  3,   -6,  12,  18,  19,  2,   1,   -12, -8,  20,  -31, 4,   -36, -3,  18,  -8,  -27, 13,  -37, -8,  -5,  5,   24,  6,   -11, -5,
    -8,  14,  28,  -5,  35,  -5,  6,   2,   2,   14,  -27, 15,  -6,  -43, -36, 17,  33,  16,  10,  16,  -1,  -56, 9,   23,  -2,  12,  2,   13,  33,  -12, 21,
    14,  -31, -13, 1,   -16, 12,  -5,  7,   -11, -1,  -6,  -56, -12, 21,  50,  3,   1,   -8,  1,   -2,  -1,  -4,  7,   7,   -26, -20, 8,   -22, -5,  11,  8,
    7,   -4,  -1,  3,   -13, 30,  -42, -21, 1,   44,  1,   21,  -41, -6,  -11, -22, -5,  -19, 7,   -24, -5,  0,   -53, 43,  19,  -17, 10,  31,  15,  -31, 11,
    -12, 21,  -4,  -21, 36,  10,  39,  19,  -13, -2,  -6,  43,  -5,  -14, 4,   25,  6,   -9,  23,  -13, 29,  8,   10,  48,  10,  -11, -19, 35,  18,  3,   4,
    15,  -16, 0,   12,  -4,  39,  37,  -6,  48,  -13, -34, -9,  -14, 45,  3,   20,  -2,  -54, -44, 38,  50,  7,   -14, 29,  -7,  -24, 6,   -2,  -5,  10,  -15,
    -8,  22,  -29, -3,  -27, 18,  -6,  5,   -27, 38,  -41, 4,   1,   -2,  5,   -31, -26, -62, 45,  9,   9,   6,   2,   9,   1,   7,   -11, 7,   -46, -21, -2,
    15,  5,   -5,  3,   9,   14,  7,   67,  -30, 5,   -23, -11, 3,   33,  0,   24,  -23, -2,  -7,  -1,  5,   -6,  19,  -8,  0,   13,  -34, 26,  28,  -16, 11,
    4,   -2,  -45, 39,  -34, 0,   52,  -6,  11,  11,  27,  23,  -15, 1,   9,   33,  -4,  -25, -18, 7,   2,   1,   10,  14,  21,  -29, 18,  54,  17,  -6,  -10,
    31,  15,  2,   2,   25,  -6,  -14, 1,   -15, 23,  34,  3,   26,  2,   -34, 3,   -28, 56,  -30, 5,   5,   -33, -25, 40,  23,  19,  -16, 13,  4,   -12, 0,
    1,   14,  2,   6,   -22, 36,  -15, -6,  -22, 8,   2,   8,   -13, 28,  -23, -4,  9,   -2,  20,  -35, -16, -16, 33,  8,   2,   2,   -18, 14,  16,  1,   -20,
    26,  -21, -16, 1,   -33, -15, 0,   2,   7,   23,  -8,  22,  -22, 32,  -17, -23, -2,  42,  -5,  -17, -36, -2,  -4,  -14, -3,  3,   21,  -29, -9,  15,  -18,
    1,   29,  -11, -29, -36, -14, 5,   -1,  46,  -64, -13, -5,  -8,  8,   -7,  -7,  1,   -59, -8,  -55, -21, 60,  -3,  -7,  -15, -20, -19, -19, 26,  22,  -13,
    0,   -9,  -11, -16, -13, -6,  -53, 2,   -4,  -11, -19, -1,  -33, -1,  -1,  -4,  13,  -44, -26, 8,   8,   33,  27,  6,   20,  -16, 25,  25,  1,   -48, 22,
    -15, 34,  15,  -37, 36,  -29, -9,  5,   24,  -40, 6,   -12, -19, -21, -55, 25,  -29, -13, 34,  14,  -4,  10,  5,   -19, 6,   0,   -8,  -46, -4,  -26, -27,
    30,  5,   8,   -35, -13, 13,  -12, -2,  -28, 17,  -48, -1,  -3,  20,  41,  -8,  -2,  11,  -72, -9,  -8,  -26, -11, -31, 26,  4,   5,   -24, 25,  -18, -3,
    -16, -42, -36, 31,  -23, -23, 8,   4,   -31, 7,   27,  -10, 24,  6,   -21, 9,   -10, 15,  -9,  7,   -2,  9,   -7,  -24, -7,  26,  9,   -3,  -2,  12,  -20,
    22,  -19, 24,  5,   -33, 2,   9,   3,   -19, 6,   17,  -3,  2,   10,  3,   -7,  12,  -3,  -10, 10,  -26, 14,  18,  13,  8,   -15, 4,   -9,  -1,  45,  20,
    -1,  -3,  1,   2,   -7,  25,  18,  -6,  13,  11,  3,   8,   -14, -9,  14,  7,   18,  -13, 8,   12,  12,  -8,  22,  12,  3,   -2,  0,   43,  15,  8,   -16,
    13,  -2,  9,   3,   6,   6,   -8,  8,   -17, 17,  39,  -2,  -14, 2,   -12, 6,   6,   9,   13,  -16, 19,  -17, 8,   15,  -3,  -9,  -3,  -1,  39,  3,   -10,
    18,  15,  14,  -3,  12,  12,  2,   42,  -1,  -39, -22, -24, -15, -43, 30,  -15, 16,  -6,  -10, 20,  -32, 24,  28,  39,  -15, 1,   1,   15,  -1,  5,   17,
    23,  -7,  6,   -43, 22,  36,  46,  -26, -3,  3,   -14, 17,  -23, -2,  41,  2,   5,   24,  -20, 22,  -6,  22,  4,   7,   -38, 39,  18,  -12, 12,  -26, 22,
    -12, -27, 69,  -24, -8,  -1,  1,   8,   -21, 56,  -5,  -20, 10,  14,  -8,  9,   46,  14,  2,   -12, 0,   -13, 31,  21,  13,  -37, -22, -1,  5,   -13, -30,
    60,  20,  -4,  -14, 20,  24,  0,   -8,  -9,  18,  -2,  9,   4,   1,   72,  -14, -56, 2,   7,   8,   -7,  34,  0,   -3,  12,  -21, 44,  -20, -23, 19,  0,
    -3,  60,  28,  15,  60,  -12, 20,  9,   -11, 16,  -16, -22, 22,  25,  -4,  8,   35,  5,   -26, 10,  -13, 3,   -1,  -7,  34,  11,  26,  -7,  13,  4,   5,
    19,  9,   -19, -8,  10,  -5,  -17, 8,   -6,  12,  -3,  3,   35,  -6,  1,   -11, 17,  11,  -7,  14,  -7,  -6,  -4,  -5,  -8,  25,  30,  -3,  36,  -21, -27,
    5,   9,   17,  -9,  10,  10,  -31, -42, 5,   26,  17,  -4,  4,   -9,  -16, 9,   4,   0,   9,   -28, 13,  20,  -11, -8,  -23, 13,  3,   4,   -27, 27,  -27,
    -17, -2,  -8,  12,  -21, -18, -44, 23,  15,  7,   13,  2,   -2,  -2,  -10, -7,  5,   -16, -31, 5,   20,  -9,  10,  2,   -1,  8,   -2,  17,  -17, 0,   -17,
    3,   18,  20,  5,   17,  -32, -5,  -8,  -6,  -1,  -10, 6,   -18, -1,  -1,  22,  -15, -17, -8,  -3,  -10, -12, 9,   -8,  7,   -5,  -32, -3,  -11, 6,   -7,
    21,  1,   -16, 4,   -30, -10, -2,  17,  -22, -16, 2,   -15, 9,   33,  19,  -14, -28, -7,  25,  -5,  -15, 11,  33,  -9,  -20, 11,  17,  -3,  -4,  -2,  -16,
    4,   -41, 16,  12,  5,   10,  -11, -24, -1,  7,   58,  33,  -18, -17, -2,  18,  11,  18,  37,  -14, 4,   -13, -20, 12,  -3,  -10, 6,   13,  29,  -9,  2,
    -21, 33,  10,  -39, 14,  -2,  2,   -31, 48,  32,  -20, -21, -15, 2,   -25, -4,  11,  -1,  -5,  13,  4,   29,  33,  15,  19,  14,  7,   5,   -2,  15,  -2,
    -4,  33,  -7,  54,  17,  7,   -21, -22, -1,  50,  15,  -4,  29,  15,  10,  -14, -5,  -1,  -25, -26, 15,  22,  -16, 1,   -5,  11,  -26, 17,  -30, 22,  24,
    -21, 25,  6,   27,  -3,  5,   7,   -13, 21,  -15, -40, -15, -12, 7,   3,   2,   -20, 23,  7,   7,   44,  -1,  11,  -23, 20,  -27, -9,  8,   8,   14,  -11,
    2,   8,   43,  44,  7,   26,  0,   -36, 10,  -13, 22,  -14, 12,  5,   -26, -16, -2,  5,   20,  -21, 29,  4,   -37, -1,  -8,  -29, 19,  -22, -9,  9,   -12,
    -13, -19, 6,   -26, -1,  -24, 29,  -18, -10, -2,  8,   -1,  -24, -29, -17, 12,  5,   11,  8,   -26, 17,  11,  -5,  -11, 18,  -28, -32, -1,  31,  -9,  9,
    -22, 4,   28,  12,  18,  -15, -21, 5,   1,   15,  3,   1,   13,  -30, 6,   -3,  -11, -25, -5,  9,   -1,  11,  15,  -43, 35,  16,  -10, 5,   27,  1,   -20,
    34,  -10, 17,  29,  7,   27,  -2,  17,  6,   -9,  3,   -3,  35,  -7,  -34, -6,  17,  9,   -2,  26,  -23, 31,  -23, 10,  23,  13,  -12, 7,   12,  7,   3,
    8,   21,  1,   -15, 3,   -1,  26,  23,  1,   29,  -11, -29, -4,  3,   33,  -7,  -1,  -10, -40, -20, 23,  20,  1,   8,   20,  -12, -14, 17,  11,  -3,  5,
    3,   -23, 35,  -7,  -10, -10, 7,   -27, 7,   -15, 28,  -20, -9,  2,   2,   10,  -42, -24, -21, 48,  15,  2,   3,   -3,  11,  12,  3,   -11, 19,  -3,  -39,
    16,  -28, -6,  2,   -3,  11,  7,   -19, 27,  -15, 23,  -20, -17, 19,  35,  5,   9,   -57, -11, -3,  -19, -19, -1,  11,  -24, 1,   10,  -19, 27,  33,  -10,
    1,   33,  -2,  -12, 21,  -24, 13,  18,  -4,  38,  0,   23,  0,   0,   10,  2,   14,  -5,  -28, 4,   3,   5,   -5,  10,  -21, 35,  -11, 11,  26,  -10, -2,
    6,   20,  5,   -2,  -11, 20,  -3,  -15, 1,   -4,  13,  31,  -1,  35,  -5,  -2,  -4,  1,   22,  -29, 15,  10,  -47, -25, 16,  27,  9,   6,   -2,  -9,  -22,
    6,   11,  8,   7,   -9,  -43, 48,  4,   -8,  -6,  9,   -9,  9,   1,   9,   -3,  4,   -10, 5,   3,   -29, -27, -6,  50,  10,  11,  -6,  9,   5,   -2,  -5,
    9,   21,  -17, -37, 2,   -16, -6,  18,  -13, 0,   3,   -13, 27,  -12, 14,  -7,  -17, 5,   36,  4,   17,  -30, -3,  -7,  -5,  -18, -7,  20,  -22, -12, 11,
    -5,  11,  -47, -9,  6,   -51, 10,  -21, 33,  -8,  -16, 52,  -12, -9,  22,  23,  2,   16,  -14, 9,   -36, 9,   -8,  -32, -7,  -8,  -23, 39,  -12, -3,  -44,
    27,  -13, 8,   -40, 19,  44,  7,   -37, -13, 22,  13,  35,  -17, -8,  -50, 36,  0,   -25, 11,  -12, -1,  -23, 54,  -71, 16,  6,   5,   11,  -25, -35, 9,
    19,  14,  33,  -56, 16,  16,  -25, -31, 12,  54,  17,  7,   9,   -12, -15, 34,  -67, -39, 6,   -15, 17,  7,   -1,  16,  -15, 28,  -6,  -42, -41, 18,  4,
    -30, -37, -4,  -3,  1,   -47, -9,  -8,  -10, -31, -24, -11, -25, 11,  -32, 31,  -1,  -21, 30,  -28, -18, 6,   -3,  -5,  8,   -26, -40, 9,   -23, -12, 12,
    -20, 27,  31,  -11, 54,  -21, 5,   18,  -13, -53, -25, 28,  2,   6,   9,   -54, -47, -29, 4,   3,   -13, 10,  2,   5,   -1,  -9,  3,   -12, -26, 0,   4,
    17,  8,   -2,  33,  -11, -22, -14, 33,  -1,  -29, -26, -11, -53, -50, 33,  28,  23,  -20, -2,  -36, 0,   -34, 26,  12,  -35, 15,  -7,  -45, -8,  26,  33,
    47,  15,  -25, -7,  -22, -6,  -12, -10, -15, 23,  -13, -14, 8,   5,   -42, -20, 35,  -15, -9,  -36, -36, -6,  10,  -22, 7,   -4,  -1,  -14, 40,  20,  -53,
    -27, -26, -19, -27, -1,  24,  -45, 1,   5,   -43, -16, -36, 5,   36,  9,   -18, 17,  3,   16,  4,   -31, 26,  -2,  -6,  32,  1,   1,   -26, 1,   -9,  -62,
    7,   8,   25,  4,   -12, 25,  -1,  -12, 2,   -8,  -10, 3,   4,   -23, -16, 6,   -14, 10,  -5,  0,   8,   -7,  0,   -6,  1,   -9,  1,   8,   -30, -1,  11,
    9,   -5,  -19, 19,  -44, 24,  9,   27,  -17, -27, 7,   25,  7,   -38, 9,   10,  -15, -4,  20,  15,  -9,  -4,  -22, 4,   1,   -23, 13,  -7,  8,   11,  -23,
    23,  -17, 5,   27,  -25, -29, -11, 0,   11,  -1,  24,  16,  -3,  1,   11,  -3,  25,  -22, -9,  23,  -8,  6,   -15, 10,  16,  25,  -13, 19,  15,  -3,  -20,
    -19, 22,  22,  -9,  -6,  -19, -15, 9,   8,   17,  -10, 8,   -20, -12, 20,  50,  14,  -55, 23,  -24, 3,   -7,  -1,  6,   -10, -10, -16, 32,  10,  -1,  -12,
    -22, -30, 33,  -1,  10,  25,  13,  18,  28,  6,   12,  6,   -11, 5,   -6,  0,   -22, 9,   -27, 20,  -3,  -7,  8,   14,  29,  -26, -16, -16, -7,  -27, 0,
    4,   -16, 9,   4,   -10, 15,  -23, -13, 1,   10,  9,   33,  -26, -11, 5,   5,   16,  -5,  10,  11,  -20, 16,  25,  2,   -30, -1,  -11, 9,   -12, -12, -26,
    -8,  -11, -8,  -10, -14, -13, -3,  5,   -10, 17,  9,   15,  -8,  -2,  14,  -29, -13, -21, 7,   -6,  -18, -10, 3,   -26, -7,  15,  -2,  -18, 13,  2,   -16,
    -7,  -25, -8,  5,   0,   1,   -3,  -2,  9,   -19, -15, 4,   -1,  -13, -1,  -3,  -16, -6,  16,  -3,  9,   1,   -19, 3,   11,  4,   6,   14,  0,   -10, 20,
    -20, -16, 10,  -5,  -12, 16,  -7,  -2,  -1,  16,  -6,  7,   -9,  8,   2,   4,   -40, 32,  23,  -9,  17,  10,  14,  -29, 26,  -26, -4,  37,  -15, 28,  9,
    15,  -1,  2,   9,   5,   35,  12,  -1,  -6,  3,   6,   -12, 6,   -21, 19,  -26, 0,   22,  -4,  6,   -7,  29,  11,  -4,  11,  9,   7,   -20, 11,  -7,  23,
    27,  2,   44,  3,   -9,  -7,  -6,  37,  -8,  14,  2,   -44, -26, 29,  30,  13,  3,   21,  -15, -28, 11,  7,   -6,  6,   2,   -23, 33,  -8,  -8,  -11, 4,
    -13, 11,  -12, 27,  -11, 7,   -4,  0,   18,  -27, -16, -8,  40,  3,   -4,  -8,  -6,  8,   4,   -9,  3,   18,  -21, -36, 15,  -16, -21, 11,  -9,  14,  13,
    -10, 29,  -26, 22,  -8,  -4,  -3,  26,  7,   9,   -40, -14, -5,  -14, -21, -14, 21,  -10, 2,   0,   -30, 34,  27,  -8,  19,  8,   -6,  -35, 2,   -37, 12,
    6,   3,   31,  5,   40,  -2,  -14, 4,   0,   31,  -10, -14, 4,   10,  11,  -6,  4,   -22, 30,  -13, 4,   52,  2,   -6,  0,   22,  3,   6,   13,  19,  0,
    -15, 14,  -12, 11,  28,  -5,  50,  -6,  -27, -6,  -21, 36,  -4,  8,   1,   -49, -42, 27,  35,  23,  1,   20,  -21, -23, 7,   5,   -10, 16,  -10, -14, 4,
    -20, -9,  -17, 29,  -20, 10,  -21, 20,  -35, -13, 5,   -9,  9,   -31, -27, -40, 39,  20,  -7,  5,   -14, 9,   14,  -8,  -4,  24,  -31, -29, 17,  18,  -3,
    2,   -8,  1,   17,  -10, 36,  -27, 2,   -7,  -16, 9,   37,  -3,  2,   -36, -8,  -1,  -18, -14, -19, 32,  -19, -2,  7,   64,  18,  -14, 9,   -18, -49, -24,
    27,  12,  29,  17,  -39, 18,  22,  7,   12,  -11, -15, -21, 2,   -12, 0,   38,  -11, -16, 11,  11,  4,   26,  11,  46,  0,   1,   8,   4,   4,   -18, -21,
    80,  8,   -34, -8,  26,  22,  -18, 9,   -16, 8,   -52, -19, 32,  3,   1,   -6,  11,  -3,  -2,  56,  40,  -4,  -20, -16, -6,  1,   25,  50,  -8,  28,  -25,
    -60, 19,  36,  -15, -21, -13, -19, -13, 14,  19,  26,  -3,  13,  -1,  -10, -3,  -19, 47,  37,  -58, -26, -4,  4,   -6,  -29, 0,   1,   11,  30,  -39, 45,
    20,  19,  35,  30,  -17, 5,   3,   27,  18,  4,   22,  -24, 39,  11,  8,   -26, 9,   -7,  48,  6,   6,   27,  27,  9,   -28, -29, 15,  -7,  36,  13,  8,
    1,   11,  -27, -40, 11,  7,   40,  -26, -27, 12,  11,  -9,  -11, -49, 15,  -21, 6,   -27, -8,  33,  -4,  3,   9,   -5,  0,   25,  12,  40,  -14, -17, 9,
    36,  -5,  -11, -40, -24, -10, -41, 35,  31,  -10, -20, -17, -4,  -1,  -3,  -15, 39,  -25, -5,  9,   -17, 8,   14,  -3,  25,  1,   12,  -31, 6,   -7,  40,
    15,  -36, 15,  -12, 15,  13,  10,  -21, 26,  17,  45,  -35, -30, -27, -25, -27, -13, 12,  11,  10,  -36, 66,  10,  -25, -18, -43, -46, -48, -3,  35,  -42,
    -8,  -25, -4,  28,  0,   18,  34,  22,  -57, 4,   13,  13,  -9,  -44, 31,  13,  -23, 38,  21,  -33, -21, 22,  45,  -10, -8,  62,  31,  4,   -15, 16,  -23,
    -37, 24,  -17, 0,   -1,  9,   -33, 9,   8,   -20, 15,  -7,  -23, -2,  -25, 10,  -7,  14,  0,   12,  0,   -20, -1,  10,  1,   7,   -10, 19,  -20, 11,  8,
    5,   9,   -44, -7,  15,  -5,  -8,  4,   -13, -21, -3,  27,  15,  -13, 0,   -13, -29, -3,  -18, 15,  16,  8,   6,   -33, -13, -15, 11,  48,  9,   -4,  -18,
    -3,  -5,  1,   27,  46,  -5,  13,  2,   -2,  19,  -33, -13, 13,  13,  2,   -8,  -1,  4,   28,  -25, -14, 11,  -12, -20, -1,  12,  21,  -10, -24, -1,  -15,
    -13, 3,   14,  -10, -4,  0,   1,   27,  48,  -6,  -46, 12,  -14, -1,  -11, 3,   6,   -14, 30,  -5,  26,  12,  -9,  -4,  -11, -3,  35,  -2,  4,   47,  2,
    5,   27,  16,  12,  -1,  27,  -24, -7,  19,  2,   -24, -11, 8,   -10, 5,   -2,  -30, 7,   -12, -3,  -23, 21,  6,   -9,  6,   -33, -3,  32,  12,  -11, -12,
    16,  -19, 19,  -14, 17,  0,   -27, -9,  15,  6,   -11, 11,  3,   -1,  2,   16,  9,   11,  12,  -4,  -3,  2,   -22, 8,   7,   10,  4,   -13, 23,  -9,  15,
    52,  27,  1,   -16, -2,  6,   -19, 29,  24,  3,   15,  0,   2,   8,   -19, -23, 9,   8,   7,   -11, 12,  14,  11,  -11, 2,   14,  6,   8,   -2,  30,  17,
    -3,  -32, -12, -5,  -7,  1,   11,  9,   5,   -14, -15, 21,  28,  -1,  -20, 24,  -15, -4,  3,   5,   0,   -23, 26,  -13, 13,  7,   -16, -20, -15, -17, 23,
    15,  1,   18,  20,  10,  4,   6,   18,  -11, 47,  -19, -10, 1,   0,   -28, 5,   15,  -29, 47,  3,   -27, 10,  -18, 2,   2,   29,  1,   18,  0,   -14, -12,
    29,  -4,  9,   -6,  11,  -36, 18,  -30, 10,  11,  -34, -6,  22,  -10, -16, 10,  -2,  -24, 8,   6,   11,  6,   15,  13,  -20, -4,  -44, 16,  22,  2,   2,
    -29, 0,   -5,  7,   54,  -4,  -8,  -21, 1,   5,   0,   29,  1,   -10, 7,   15,  11,  24,  -14, -6,  0,   1,   19,  -15, 7,   2,   8,   -7,  23,  0,   -15,
    -8,  8,   8,   25,  26,  -26, -4,  -11, 8,   3,   -7,  19,  -5,  1,   -7,  24,  44,  -5,  -43, 28,  -5,  -4,  -11, 10,  -8,  0,   4,   -6,  3,   23,  -19,
    -19, -2,  -28, 18,  12,  13,  28,  16,  10,  -7,  20,  20,  3,   -58, -61, 4,   17,  -19, 48,  13,  -7,  1,   -4,  9,   -21, 4,   9,   -25, 0,   -49, -18,
    -6,  -9,  60,  -4,  62,  -11, 16,  -19, -21, 27,  9,   -2,  -18, -6,  -20, -1,  28,  -4,  -5,  3,   -7,  0,   -13, -2,  -18, -4,  -1,  -2,  34,  -5,  13,
    14,  -61, -29, -8,  -2,  -77, -17, -2,  60,  -32, 0,   0,   -65, 3,   8,   35,  11,  36,  -68, 19,  40,  -13, -37, -28, -3,  -3,  25,  33,  2,   -24, 3,
    -6,  -21, -18, -10, 6,   12,  -2,  -9,  45,  -7,  37,  2,   2,   -7,  -28, -48, -4,  -27, -1,  -34, -17, 14,  18,  9,   -28, -1,  10,  -7,  -56, 13,  -15,
    -3,  -35, 25,  27,  14,  27,  22,  38,  7,   5,   -51, 4,   14,  -2,  -16, -1,  2,   -7,  2,   -31, 9,   -32, 5,   2,   -4,  22,  31,  11,  -60, -6,  -46,
    -15, -14, 4,   12,  13,  12,  6,   11,  54,  -2,  39,  -17, 20,  -12, 36,  -15, 37,  15,  19,  2,   -4,  7,   -52, 5,   26,  34,  19,  2,   5,   -17, 13,
    2,   17,  10,  -15, -1,  -35, -33, 31,  -25, -9,  -36, -24, 16,  11,  -9,  -31, 38,  -11, -19, 37,  6,   -48, -5,  4,   -22, -9,  11,  0,   -22, -20, 14,
    -21, 15,  29,  -34, -4,  24,  -12, 18,  -32, 24,  -31, 13,  -24, 5,   24,  -27, -23, -40, -8,  -1,  9,   27,  -15, 10,  14,  2,   24,  -4,  -77, 17,  -19,
    10,  -30, 1,   -28, 0,   -33, -9,  -17, 8,   -6,  -69, 123, 27,  -7,  14,  -18, 1,   -4,  25,  18,  -68, -46, 16,  19,  -19, -2,  31,  -18, -27, 29,  -22,
    18,  11,  -37, 1,   23,  15,  22,  7,   10,  13,  44,  3,   -9,  -24, -3,  16,  -4,  33,  -26, 20,  -20, 15,  30,  13,  -6,  -14, 28,  7,   -16, -7,  44,
    -17, -10, 14,  16,  35,  48,  1,   58,  -13, -42, -6,  -17, 48,  -1,  19,  -3,  -58, -34, 27,  45,  19,  -20, 8,   12,  -53, -2,  -6,  -3,  30,  -5,  -41,
    41,  -19, -20, -21, 14,  -8,  18,  -17, 29,  -19, 4,   15,  -7,  0,   -41, -22, -29, 52,  20,  6,   4,   -11, 14,  13,  -2,  -12, 26,  -7,  -19, -7,  0,
    7,   14,  -34, -20, 9,   -15, 35,  -19, 35,  -16, 4,   4,   54,  5,   -34, -39, -1,  -19, -3,  -2,  -9,  35,  -54, -10, 7,   -22, 34,  34,  -18, 11,  37,
    11,  -16, 10,  -29, 20,  23,  -8,  28,  11,  17,  -8,  9,   8,   -11, 26,  -1,  -24, 1,   7,   10,  -12, 10,  -17, 30,  -6,  9,   35,  -6,  3,   -8,  18,
    -9,  -6,  9,   17,  -5,  -20, 8,   1,   8,   18,  -1,  43,  -10, -20, 0,   -5,  32,  -1,  5,   -7,  -40, -34, 21,  33,  12,  11,  10,  -21, -16, 2,   8,
    -6,  8,   2,   -39, 30,  5,   -10, -16, 19,  -6,  -5,  -4,  21,  -9,  -12, 9,   -1,  7,   -43, -33, -12, 36,  6,   -1,  -4,  12,  -4,  2,   9,   13,  26,
    -8,  -44, 8,   -11, -4,  20,  6,   0,   15,  -15, 23,  -33, 10,  -12, -5,  16,  24,  5,   6,   -32, 1,   -3,  -13, -4,  -8,  6,   -15, -5,  7,   40,  -10,
    -10, 2,   -9,  -29, 12,  9,   -10, 26,  -13, 6,   8,   -20, 11,  -2,  10,  7,   10,  0,   -9,  0,   20,  15,  -5,  -2,  10,  -30, 31,  -6,  11,  1,   -18,
    -3,  25,  13,  -19, 1,   19,  13,  21,  16,  14,  6,   -7,  -23, -1,  16,  -28, 8,   -3,  3,   3,   -20, 1,   -7,  -7,  60,  11,  -9,  -8,  -14, -6,  -8,
    35,  25,  -9,  10,  12,  -2,  11,  18,  -1,  10,  4,   2,   8,   1,   1,   30,  -19, 10,  5,   -4,  21,  -6,  28,  29,  7,   -16, -2,  -1,  3,   11,  4,
    16,  1,   -17, 4,   21,  47,  -5,  -53, 7,   -5,  7,   -5,  -3,  7,   2,   8,   -27, 17,  22,  0,   -30, 13,  -10, 32,  15,  9,   32,  23,  4,   6,   2,
    11,  12,  -25, 21,  13,  9,   9,   22,  0,   -20, 21,  -21, -2,  18,  1,   26,  -1,  20,  -15, 11,  7,   -4,  26,  -13, -15, -3,  4,   3,   0,   12,  -17,
    48,  -5,  11,  31,  7,   -10, -10, 34,  12,  -10, 9,   -10, 0,   -14, -1,  -7,  26,  31,  9,   21,  -5,  -22, 7,   15,  14,  -10, -3,  7,   -50, -35, 14,
    22,  9,   4,   14,  -4,  -11, 15,  -3,  -7,  0,   1,   12,  7,   -10, 3,   -16, 19,  -3,  -3,  -11, 10,  -22, -15, -3,  3,   13,  -31, -10, -15, 18,  1,
    13,  -2,  -9,  -2,  13,  -4,  3,   4,   -6,  -35, 11,  1,   -1,  16,  -6,  8,   0,   -11, 23,  -18, 1,   -20, 1,   16,  37,  -4,  10,  -33, -8,  11,  -23,
    -17, -1,  18,  -18, 3,   13,  -30, -4,  33,  -23, 10,  24,  2,   -31, 26,  -19, 9,   52,  -8,  31,  20,  11,  17,  -7,  -4,  12,  41,  9,   -27, -10, 9,
    3,   -13, 24,  34,  10,  -1,  13,  42,  11,  -25, -3,  29,  -26, -19, -8,  52,  -31, -8,  13,  -3,  -12, 38,  -3,  46,  13,  -3,  4,   -29, 34,  -28, 15,
    -27, -45, -46, 54,  6,   10,  14,  24,  -10, -33, -37, 11,  26,  11,  5,   -45, 58,  -8,  -10, -21, 25,  10,  25,  3,   0,   -13, 2,   10,  -12, 15,  -43,
    -13, -5,  54,  11,  -8,  -2,  3,   18,  13,  -2,  -3,  11,  -39, -9,  10,  -31, -4,  16,  -18, 5,   13,  -19, 44,  -25, 24,  -34, -35, 18,  25,  -15, -61,
    -65, -5,  5,   -11, -17, -6,  8,   -38, -22, 10,  43,  -10, -6,  17,  -22, -55, -10, 24,  8,   17,  7,   -49, 3,   0,   1,   5,   19,  -5,  -29, -13, -41,
    -1,  69,  -8,  -11, -14, 15,  4,   1,   23,  39,  16,  -4,  4,   40,  10,  -44, -13, 33,  23,  -38, 4,   21,  4,   -3,  5,   -29, 5,   -39, -10, 13,  4,
    9,   -3,  6,   4,   2,   58,  46,  -6,  -29, 13,  1,   -15, 26,  52,  -16, 23,  -22, -40, 24,  5,   -40, -18, -1,  1,   -23, 5,   -6,  5,   -14, -14, -5,
    -2,  -12, -34, 65,  43,  -54, -39, 11,  7,   -32, -20, -12, -8,  0,   28,  -14, 30,  46,  15,  19,  22,  5,   -2,  10,  22,  -9,  -11, 44,  -1,  70,  33,
    18,  -16, -2,  -17, 48,  11,  2,   16,  34,  9,   -35, 21,  6,   -1,  29,  -19, -19, 0,   -5,  -20, 2,   13,  -14, 34,  -8,  -18, 3,   -9,  13,  -18, 12,
    -6,  9,   -2,  -21, 3,   30,  9,   -19, -13, 15,  -17, 31,  -24, 28,  -1,  -21, -2,  12,  2,   -27, 4,   15,  0,   14,  17,  12,  1,   13,  -25, -14, -10,
    -12, 12,  15,  6,   12,  -24, 6,   2,   -4,  31,  -15, -16, -38, 6,   6,   -10, 22,  22,  8,   -3,  12,  10,  18,  -9,  -4,  0,   -1,  18,  -7,  5,   6,
    8,   -27, 18,  -2,  9,   -8,  9,   32,  22,  16,  -18, 1,   -6,  4,   10,  -7,  15,  -10, -25, -5,  10,  50,  -7,  -22, 19,  -5,  13,  -8,  4,   12,  -12,
    17,  -10, 22,  20,  5,   -16, 10,  -15, 48,  16,  -10, 22,  4,   7,   13,  10,  10,  14,  -3,  -3,  -2,  -1,  1,   -10, -12, -11, -1,  -6,  -11, -6,  3,
    -3,  -8,  -7,  -5,  -11, -13, 10,  -10, 2,   2,   -14, -2,  -6,  3,   6,   -3,  -7,  -13, 4,   4,   2,   9,   -9,  -3,  -10, -1,  -7,  3,   -8,  -12, -12,
    7,   -3,  2,   -8,  1,   -5,  1,   -9,  -11, -16, 0,   -12, -10, 2,   8,   -11, -4,  4,   -3,  -1,  -16, -11, 12,  1,   7,   -6,  -6,  -1,  8,   8,   -11,
    -14, -14, 2,   4,   2,   5,   7,   -9,  -6,  -11, 0,   -5,  -6,  5,   0,   7,   -2,  -7,  0,   -10, 3,   -1,  -9,  3,   1,   5,   -7,  4,   0,   -6,  -8,
    -9,  -7,  -11, 6,   -4,  -1,  -2,  1,   -4,  4,   -7,  -11, -4,  0,   -4,  9,   2,   7,   -1,  -8,  -1,  2,   -15, 13,  16,  -9,  13,  39,  13,  1,   14,
    -26, 17,  21,  -1,  35,  8,   27,  -7,  7,   1,   -8,  19,  -3,  -9,  4,   24,  6,   -1,  8,   -17, 46,  -3,  15,  39,  12,  8,   -9,  30,  -8,  -3,  -2,
    10,  0,   -11, 2,   6,   26,  36,  12,  39,  -20, -42, 1,   8,   15,  -5,  5,   -8,  -59, -31, 12,  21,  8,   -5,  5,   -12, -36, -4,  -1,  -1,  14,  -18,
    5,   5,   -19, -8,  -17, 11,  -21, 5,   3,   32,  -37, -20, 14,  5,   10,  -21, -9,  -29, 23,  -6,  -3,  7,   -8,  14,  8,   11,  -13, 8,   -9,  -24, 5,
    22,  1,   3,   -3,  -10, 15,  -8,  14,  -26, 1,   -31, 9,   21,  19,  10,  13,  -22, 6,   1,   -5,  -2,  -17, 16,  -17, -8,  4,   69,  -14, -13, 0,   -20,
    -8,  -8,  13,  -12, 0,   -6,  -7,  -1,  -8,  11,  -4,  -4,  15,  7,   12,  -24, 5,   7,   3,   -29, -22, 18,  -14, 28,  11,  23,  -2,  -10, 4,   15,  -8,
    -29, -1,  35,  20,  -6,  13,  25,  -9,  -8,  17,  -6,  -9,  -28, 4,   -17, -27, 18,  -19, -21, -7,  4,   34,  4,   -18, -5,  5,   10,  17,  32,  30,  -45,
    3,   14,  -20, 18,  18,  6,   -9,  -1,  16,  -19, 16,  10,  10,  -2,  15,  1,   11,  6,   -7,  -10, 2,   -22, -22, -33, -12, -11, 5,   9,   0,   2,   -13,
    -45, 30,  -15, 15,  -25, 28,  -34, 9,   5,   14,  0,   -29, -1,  -24, 4,   29,  -6,  -25, 11,  -14, 87,  -9,  2,   38,  16,  9,   17,  21,  4,   -14, 54,
    0,   -12, 10,  -14, -39, -10, 6,   -5,  19,  9,   -50, -11, 11,  20,  -4,  -14, 5,   -4,  4,   -25, -10, 17,  -4,  -14, -32, 6,   -1,  -2,  20,  34,  -17,
    -5,  1,   28,  -12, -17, 4,   17,  -31, -11, 25,  32,  7,   -9,  -2,  -4,  -2,  -26, 5,   25,  -17, 18,  -4,  -22, -12, 18,  71,  38,  -3,  -6,  5,   -2,
    12,  10,  38,  -43, 16,  0,   -28, 17,  1,   -26, -6,  13,  -3,  -28, -37, -7,  -7,  9,   -21, 10,  19,  -10, -19, 45,  39,  -16, -24, -36, -30, -14, -16,
    9,   -21, -3,  15,  -34, 16,  -1,  2,   10,  28,  -10, 7,   -7,  4,   -5,  -1,  32,  3,   47,  26,  -8,  -6,  -18, -22, 48,  -8,  16,  48,  17,  21,  8,
    18,  10,  -1,  -29, 12,  27,  1,   13,  32,  17,  -6,  5,   -17, 7,   28,  -8,  16,  5,   29,  8,   -5,  -2,  -3,  21,  9,   -27, -2,  20,  4,   -16, 20,
    -17, 23,  -13, 7,   27,  -7,  -19, 7,   21,  8,   -17, 11,  7,   2,   -6,  6,   8,   2,   13,  -1,  19,  -5,  -14, 9,   -4,  37,  -9,  3,   -4,  -40, -20,
    3,   30,  1,   7,   10,  -13, -20, 16,  9,   -8,  14,  0,   -6,  24,  7,   -2,  -9,  0,   -9,  6,   -17, 13,  -4,  -10, 11,  -4,  1,   -39, -31, 3,   31,
    -2,  6,   9,   0,   -1,  6,   1,   8,   -3,  -22, -49, 13,  -2,  -5,  17,  0,   3,   -6,  -13, 17,  -13, 25,  -23, -10, 12,  33,  8,   9,   -20, 2,   11,
    -25, -13, 4,   8,   -16, -8,  0,   -28, 30,  9,   -10, 1,   30,  7,   -8,  13,  -8,  11,  -11, -10, 27,  7,   34,  2,   -10, 5,   -4,  13,  -27, -6,  2,
    9,   11,  -7,  17,  -25, 43,  2,   22,  33,  -11, 11,  -24, 20,  6,   2,   7,   13,  0,   -10, -1,  3,   41,  27,  -9,  35,  -27, -46, 10,  4,   43,  -10,
    1,   2,   -45, -32, 22,  39,  13,  -11, 23,  -5,  -16, 14,  8,   -9,  6,   -13, 10,  16,  -12, -20, -7,  17,  -20, 14,  -2,  25,  -49, -15, -6,  -3,  6,
    -23, -11, -50, 22,  17,  3,   8,   -11, 17,  -3,  -4,  -29, 8,   -32, -16, 6,   28,  -6,  1,   -9,  -3,  18,  9,   37,  -31, -4,  -19, -7,  -1,  23,  4,
    25,  -30, -1,  3,   -9,  10,  2,   -1,  -11, -6,  0,   -31, 19,  22,  -5,  10,  27,  1,   -5,  -1,  -28, 18,  19,  -5,  29,  -4,  23,  -16, -1,  11,  5,
    30,  2,   -38, -3,  21,  10,  -6,  4,   -20, 25,  -1,  15,  46,  0,   -10, 0,   -5,  -3,  4,   3,   -11, -11, -13, 14,  12,  12,  27,  -9,  41,  -1,  -22,
    -5,  8,   21,  2,   13,  13,  -49, -37, 23,  11,  7,   -8,  18,  -13, -14, 8,   15,  -16, 12,  -22, -38, 22,  -17, 10,  -8,  16,  -20, 2,   -6,  27,  -14,
    -10, 8,   1,   5,   -31, -15, -8,  30,  3,   -4,  -3,  -5,  14,  -5,  5,   13,  8,   -15, -34, 1,   31,  -14, 19,  -3,  -4,  14,  -12, 12,  -14, 1,   -15,
    0,   13,  39,  -2,  -4,  -32, -12, -8,  -18, -15, -2,  10,  -17, -7,  14,  -34, 16,  27,  -5,  9,   28,  7,   -17, 6,   -2,  29,  -14, -29, -9,  25,  31,
    17,  -5,  -7,  -3,  35,  -28, -42, -36, 14,  25,  -3,  20,  13,  33,  -18, 29,  44,  4,   -17, -48, 38,  -33, -58, 4,   48,  -4,  -2,  11,  26,  35,  45,
    6,   24,  -31, -35, 17,  3,   27,  5,   2,   -11, -52, -57, 56,  60,  17,  -16, 11,  12,  -46, -16, 4,   4,   39,  -29, -40, 24,  -13, -25, -5,  34,  -19,
    21,  -24, 35,  -33, -11, 13,  11,  -17, -60, -12, -57, 45,  7,   11,  -7,  -3,  20,  -6,  9,   -38, 43,  -35, -3,  12,  9,   16,  8,   -37, -13, 9,   -5,
    48,  -29, 30,  -24, 2,   13,  29,  8,   -68, -54, 3,   -10, -21, -30, 8,   1,   -76, -12, 5,   -35, 35,  11,  -8,  9,   26,  0,   -20, 26,  -27, 25,  30,
    0,   33,  13,  13,  13,  -4,  1,   -9,  23,  -10, -5,  -7,  13,  9,   -1,  8,   -14, 22,  -19, 11,  39,  5,   -6,  -7,  24,  -13, 3,   15,  1,   -8,  -2,
    -5,  6,   23,  24,  3,   37,  -6,  -17, 12,  10,  44,  0,   8,   -5,  -48, -34, 25,  -3,  -4,  -9,  11,  -13, -24, -5,  -6,  -10, 13,  -22, 2,   26,  -17,
    6,   -4,  3,   -24, 11,  9,   20,  -20, 0,   2,   1,   9,   -21, -25, -18, 33,  -8,  4,   0,   -9,  16,  -6,  10,  -25, 18,  -6,  -19, 9,   8,   -10, 16,
    -1,  10,  14,  5,   39,  -35, 24,  -24, -5,  5,   31,  10,  21,  -37, 4,   4,   -24, -4,  -6,  -5,  -8,  -13, 8,   -7,  -8,  0,   -11, -3,  -9,  -5,  13,
    -11, 2,   1,   5,   6,   -6,  -3,  0,   -1,  1,   -4,  7,   11,  -17, 10,  -4,  1,   0,   -4,  5,   1,   -10, 5,   -11, 0,   2,   -11, 5,   -6,  -11, -3,
    0,   -10, -6,  -8,  -5,  -11, 8,   3,   -11, 8,   -13, 0,   -3,  -10, -8,  3,   3,   -6,  -1,  -1,  -6,  3,   -2,  -6,  -7,  -5,  -11, -15, 5,   -5,  9,
    1,   6,   3,   -1,  -9,  -9,  -9,  -7,  -3,  6,   -8,  -10, -15, 3,   -2,  -14, 7,   4,   -8,  -1,  -4,  -10, -5,  3,   -16, 3,   -9,  -10, 4,   -4,  11,
    -18, -7,  -10, -13, -7,  11,  -3,  -4,  -7,  6,   -7,  -7,  -9,  -12, -1,  -2,  -5,  5,   4,   9,   11,  -8,  3,   5,   8,   -2,  -1,  -29, 37,  27,  -27,
    14,  5,   11,  -9,  20,  -31, 2,   40,  5,   20,  0,   17,  -5,  -4,  8,   -3,  37,  12,  -4,  -4,  5,   -2,  6,   11,  -35, 39,  -22, 9,   43,  10,  11,
    4,   20,  -6,  11,  2,   12,  -15, -12, 4,   16,  11,  25,  -9,  41,  4,   -6,  1,   6,   25,  -9,  -1,  -5,  -49, -33, 41,  24,  19,  -1,  18,  -17, -23,
    -6,  4,   -15, 14,  -5,  -35, 54,  -10, 7,   2,   -2,  -29, 4,   -6,  24,  -12, 3,   6,   1,   4,   -6,  1,   -3,  13,  7,   0,   6,   5,   13,  -5,  -2,
    -1,  14,  -19, -37, 12,  -17, -14, 18,  13,  4,   18,  -14, 26,  3,   19,  -4,  -25, 0,   28,  -10, 7,   -27, -10, -3,  -8,  0,   -12, 10,  -7,  -1,  -2,
    -46, 20,  16,  4,   7,   4,   -3,  -23, 13,  -13, 13,  22,  -44, -11, 15,  35,  30,  -22, 6,   2,   44,  -16, -3,  -24, 20,  17,  11,  15,  28,  39,  -6,
    16,  36,  11,  -11, -34, 17,  -18, -8,  -2,  35,  -3,  -14, 8,   -8,  42,  50,  -3,  34,  -18, -31, 1,   -22, 48,  -6,  19,  -8,  -54, -50, 33,  54,  7,
    -35, 12,  6,   -40, 17,  8,   -1,  11,  -8,  -1,  39,  -30, -12, -5,  16,  -15, 15,  -33, 32,  -55, -8,  12,  -5,  7,   -39, -22, -41, 50,  25,  2,   -1,
    3,   11,  10,  0,   -27, 27,  -30, -2,  4,   12,  16,  4,   7,   -16, 16,  5,   48,  -25, 22,  -12, -8,  -5,  34,  8,   -14, -42, 0,   -6,  -26, 2,   -8,
    28,  -60, 0,   14,  -11, -21, -6,  -7,  3,   -11, -3,  21,  -20, 35,  -12, -24, 4,   -25, 13,  -23, 21,  10,  0,   9,   -8,  9,   41,  10,  -10, -3,  13,
    -11, 25,  -33, 24,  2,   -45, -4,  12,  4,   -18, 4,   7,   -10, -7,  20,  4,   -7,  6,   -28, -22, -5,  -13, 14,  -1,  9,   -2,  -9,  -22, -6,  10,  50,
    26,  -6,  -24, -1,  11,  -5,  35,  18,  -2,  -6,  10,  -10, 16,  -29, -28, -3,  2,   8,   -11, -2,  1,   10,  -22, -18, 13,  3,   -8,  -4,  42,  28,  6,
    -11, -1,  2,   11,  -1,  15,  11,  12,  -8,  2,   28,  50,  -7,  -48, 27,  -17, 10,  -5,  3,   -1,  -19, 24,  -19, 6,   15,  -6,  -9,  -12, -11, 36,  -2,
    4,   16,  18,  16,  14,  6,   5,   -7,  -19, -34, -18, 5,   -6,  -60, -9,  20,  -27, 39,  -11, -30, 7,   -15, 5,   -12, 3,   5,   -4,  13,  -37, -1,  40,
    8,   -6,  -7,  19,  -7,  40,  -12, 26,  -14, -28, -10, 30,  -5,  -16, 0,   15,  -1,  -11, 15,  11,  5,   14,  -48, -18, -5,  -17, 8,   14,  9,   6,   -26,
    -5,  -10, 0,   50,  43,  -8,  -22, 8,   -5,  -11, 35,  35,  -11, 3,   13,  4,   15,  -17, -31, 7,   8,   20,  6,   5,   -9,  32,  -7,  0,   1,   -5,  9,
    -1,  50,  37,  -16, -28, -13, 3,   -8,  9,   -5,  -8,  1,   -8,  -9,  22,  27,  7,   21,  12,  -11, 12,  9,   -1,  9,   -16, 42,  -9,  40,  3,   -30, -12,
    -5,  -23, 36,  8,   8,   36,  19,  21,  7,   13,  4,   -1,  -21, 6,   -1,  -29, -4,  -9,  10,  5,   15,  2,   29,  7,   -20, 38,  23,  15,  27,  -5,  -6,
    -13, 62,  -12, -28, -36, 17,  -3,  6,   -7,  26,  0,   -15, 4,   11,  -5,  4,   -18, 4,   -28, -6,  -20, 43,  -20, 4,   -3,  2,   4,   56,  -10, 10,  -11,
    -6,  13,  -2,  -30, -29, -4,  3,   -26, -33, 1,   -31, -19, -5,  20,  19,  -22, -26, 23,  2,   5,   -15, 2,   13,  -12, -30, 8,   32,  -12, 3,   -57, 23,
    -27, 0,   -3,  -13, -17, -63, -27, -9,  -5,  16,  -37, -19, 7,   22,  7,   7,   -14, 28,  3,   6,   14,  32,  -3,  15,  5,   9,   9,   -29, 21,  -12, 15,
    -7,  -14, 9,   9,   18,  -57, -40, -8,  2,   -17, -21, 1,   -6,  -21, -22, 4,   -50, -29, -16, 6,   19,  -43, 10,  -30, -9,  -52, 5,   -41, 15,  -14, -8,
    -36, 10,  -42, 14,  -8,  -2,  -28, -72, -23, 16,  -14, -61, 13,  -27, -9,  -18, 0,   -40, 7,   16,  29,  -49, -88, -18, -36, -50, -20, -25, -1,  -4,  3,
    -38, 4,   -5,  -11, -23, 20,  1,   -38, -49, 19,  -11, -6,  64,  -39, -27, 43,  9,   6,   8,   -15, 1,   11,  5,   -7,  -33, 33,  -40, -65, 20,  36,  -8,
    22,  22,  -44, 4,   24,  -21, -15, -16, 41,  -2,  16,  48,  -16, 10,  0,   3,   -19, -7,  24,  7,   -3,  -28, 35,  -82, 8,   27,  22,  19,  -32, 2,   -43,
    -57, 67,  31,  21,  -52, -4,  12,  -16, 28,  22,  4,   -5,  -2,  28,  -46, 10,  -33, -6,  -27, 15,  -19, 26,  32,  3,   16,  37,  11,  -20, 19,  -15, 13,
    33,  -8,  34,  -7,  19,  -3,  0,   7,   -7,  27,  -2,  -16, -4,  21,  -4,  -4,  3,   -12, 28,  -12, -6,  38,  6,   -12, -2,  25,  -5,  -16, 2,   10,  -2,
    6,   3,   -6,  9,   38,  -3,  47,  -3,  2,   -5,  12,  39,  3,   2,   -6,  -30, -40, 10,  33,  15,  5,   9,   -17, -31, 3,   8,   1,   -1,  -15, 4,   24,
    9,   10,  -6,  12,  -8,  1,   -20, 20,  -22, 1,   0,   2,   10,  -31, -28, 2,   35,  -4,  7,   -4,  10,  13,  5,   7,   14,  10,  -30, -38, 8,   2,   -8,
    18,  11,  3,   11,  7,   25,  -26, 6,   -4,  -9,  14,  25,  5,   18,  -50, -1,  -7,  -24, -2,  4,   6,   -18, 0,   3,   -40, 12,  36,  -7,  23,  38,  16,
    -8,  28,  -34, 5,   41,  -4,  27,  9,   24,  4,   -4,  11,  -8,  29,  5,   -39, 7,   19,  1,   -17, 20,  -25, 27,  -6,  12,  26,  6,   -13, 6,   28,  -8,
    -7,  4,   21,  -2,  3,   13,  2,   11,  21,  -8,  36,  -5,  -4,  -11, -1,  17,  -24, 6,   -8,  -44, -36, 20,  22,  14,  13,  16,  -15, -9,  15,  13,  0,
    3,   -9,  -29, 37,  -6,  -4,  1,   21,  -2,  7,   -18, 19,  -21, -4,  10,  -1,  -3,  -29, -19, -12, 35,  -2,  8,   10,  -8,  1,   -5,  12,  0,   10,  -9,
    -29, 4,   -16, -6,  15,  -11, 2,   1,   -18, 30,  -10, 6,   -12, -3,  0,   27,  0,   9,   -45, 0,   11,  -16, -22, -13, 0,   -16, -10, 10,  3,   -7,  16,
    -50, -1,  11,  -23, -10, 22,  12,  -6,  46,  -41, 20,  -12, -9,  46,  9,   -35, 1,   38,  9,   6,   -10, -4,  14,  3,   21,  -19, 15,  -27, 3,   1,   -8,
    -27, 13,  -7,  -16, -3,  -12, 75,  -12, -14, -13, -1,  28,  -3,  -10, 56,  39,  4,   -40, -4,  39,  -82, 4,   -34, 15,  -9,  24,  -25, -23, 21,  8,   14,
    -27, 23,  14,  -4,  -11, 40,  13,  62,  35,  -17, 0,   -13, 25,  -2,  -36, -13, 16,  16,  -3,  -34, -52, -50, -13, 52,  5,   15,  54,  -90, 24,  -14, 9,
    -3,  14,  1,   -7,  -37, 8,   -17, -21, 31,  3,   24,  -31, 1,   13,  -14, 18,  24,  -41, -28, 35,  14,  23,  -35, -36, 2,   -94, -16, 12,  15,  -22, -49,
    1,   -28, 34,  16,  -9,  16,  32,  -4,  -21, 20,  -29, 3,   -36, -11, 25,  3,   26,  3,   5,   -1,  -7,  41,  -12, -14, -16, 5,   23,  7,   7,   -27, 48,
    -8,  21,  27,  3,   5,   -8,  21,  -4,  7,   9,   27,  -10, -15, 6,   1,   23,  23,  11,  50,  0,   -45, -4,  -14, 22,  19,  1,   5,   -46, -19, 24,  15,
    13,  -28, 21,  -4,  -31, 20,  -8,  8,   10,  -3,  -18, 20,  -8,  -7,  -26, 10,  -20, 18,  -9,  29,  -21, -15, 0,   -5,  18,  -30, -34, -28, 46,  -6,  -1,
    5,   -22, 3,   12,  -4,  -25, 22,  -39, -35, 16,  15,  -11, 14,  -18, 14,  26,  -5,  54,  -21, 1,   2,   0,   9,   31,  -4,  9,   -18, -4,  7,   -10, -1,
    12,  26,  -4,  8,   17,  40,  -19, -7,  5,   -9,  -24, -3,  18,  -19, 39,  -9,  2,   10,  -13, 15,  -14, 6,   12,  6,   2,   -33, -8,  9,   18,  -34, -17,
    13,  -34, 13,  -3,  31,  -15, -11, -13, 18,  -1,  -29, 18,  24,  -23, 25,  11,  14,  -10, -5,  -12, 6,   12,  -30, 9,   17,  -7,  4,   -31, 27,  -6,  6,
    52,  2,   -15, -10, -6,  18,  9,   47,  -34, -20, 13,  9,   -22, 20,  0,   -20, 16,  -3,  36,  -12, 1,   2,   24,  -5,  5,   10,  3,   8,   -5,  38,  24,
    -1,  -34, -16, -11, 8,   17,  20,  -3,  -7,  -54, -4,  23,  33,  9,   -65, 17,  4,   -7,  2,   -1,  -4,  -68, 16,  -29, 36,  24,  -12, -27, -5,  -16, 25,
    7,   -4,  25,  12,  3,   12,  19,  7,   -1,  24,  -7,  -12, 13,  -2,  -56, -1,  17,  -8,  29,  -6,  -26, -9,  -20, 3,   -4,  9,   -14, 1,   0,   -38, 6,
    17,  3,   6,   -3,  7,   3,   14,  -11, 16,  7,   -33, 1,   15,  3,   -12, 5,   17,  -12, -8,  -6,  1,   13,  3,   1,   -17, -2,  -21, 10,  15,  14,  -4,
    -26, 12,  -10, 9,   54,  16,  2,   -22, -2,  -2,  -11, 30,  26,  -7,  11,  13,  -11, 10,  8,   -15, -6,  1,   23,  6,   7,   4,   13,  -9,  17,  -9,  -12,
    16,  4,   35,  21,  11,  -20, 3,   -2,  -3,  -6,  0,   -5,  6,   -1,  -9,  20,  39,  12,  17,  10,  2,   7,   -9,  0,   9,   2,   27,  -7,  13,  4,   -13,
    -22, 18,  0,   39,  13,  -10, 16,  11,  12,  -13, 13,  11,  -5,  11,  -5,  -6,  1,   -3,  -12, -10, -3,  2,   9,   -10, 3,   -12, 3,   -13, -6,  -5,  0,
    -13, -4,  -12, -10, 12,  0,   1,   -11, -4,  3,   4,   0,   3,   1,   2,   -2,  1,   -4,  1,   4,   -11, -12, -11, -3,  1,   -2,  -11, 10,  -4,  -2,  -10,
    -16, -7,  -3,  9,   -2,  -7,  -5,  -4,  3,   2,   10,  4,   -5,  -16, -5,  10,  -6,  -9,  2,   -5,  -1,  -7,  -3,  7,   -8,  -14, 9,   -3,  -5,  -12, 10,
    0,   4,   -8,  2,   -14, 3,   -8,  -2,  -10, -6,  -8,  -9,  -10, 3,   -10, -12, -2,  6,   -3,  3,   -8,  -11, -1,  0,   -5,  -12, 11,  -12, -8,  -7,  -2,
    -6,  -2,  1,   -9,  -9,  0,   -9,  -3,  -12, -7,  10,  0,   -8,  3,   10,  -9,  -13, -32, 26,  9,   -6,  13,  27,  3,   -15, 19,  -32, 9,   38,  -5,  32,
    6,   15,  -4,  7,   10,  1,   9,   11,  -25, 2,   6,   14,  -6,  24,  -33, 36,  -3,  -4,  30,  -9,  -16, -11, 27,  2,   -12, 8,   24,  -6,  -20, 7,   0,
    24,  39,  -5,  48,  -3,  -32, 9,   11,  31,  -6,  14,  9,   -48, -31, 23,  30,  0,   -6,  22,  -16, -24, 19,  10,  -13, 17,  -12, -8,  31,  0,   -4,  -4,
    8,   -9,  -2,  -23, 35,  -31, 5,   2,   -5,  -3,  -46, -6,  -5,  40,  -6,  0,   9,   5,   -3,  15,  7,   13,  0,   -19, -46, 9,   -5,  -18, 21,  6,   7,
    19,  -10, 15,  -14, 21,  -4,  -6,  1,   30,  4,   -1,  -38, 2,   -4,  -10, -22, -16, 15,  -5,  -6,  -2,  13,  -21, -12, 27,  -5,  -21, -10, -1,  -15, 18,
    -17, -17, 2,   -9,  3,   -5,  3,   13,  -1,  -5,  -56, 2,   0,   6,   -32, -11, 22,  -3,  7,   16,  33,  -21, -34, -2,  25,  9,   -30, 9,   23,  5,   -1,
    12,  3,   9,   11,  -31, -5,  1,   -19, 9,   11,  13,  -3,  -13, 11,  -10, 6,   44,  42,  -16, -21, -13, 5,   -9,  20,  2,   -2,  10,  -5,  -25, 3,   0,
    -19, 18,  11,  7,   -7,  0,   -6,  30,  -3,  -5,  15,  8,   -7,  -15, 6,   21,  -10, -35, -11, -3,  -16, 13,  8,   9,   3,   -23, 0,   26,  36,  -3,  17,
    17,  -6,  9,   -4,  -2,  11,  -5,  28,  -5,  22,  23,  5,   -24, -18, -28, 38,  12,  -7,  16,  14,  13,  8,   13,  12,  4,   29,  41,  -21, -18, 9,   37,
    -4,  -30, 48,  -10, -18, 0,   -8,  5,   25,  -12, -23, -23, -33, -11, 20,  -45, -4,  14,  4,   0,   24,  26,  7,   -26, 17,  -1,  -52, -7,  -30, -4,  42,
    -14, -27, 1,   -1,  22,  -32, -30, -20, 32,  -28, 5,   -2,  -52, -14, 8,   2,   5,   -25, -1,  -35, -6,  2,   10,  9,   19,  21,  -1,  -36, 27,  -26, -19,
    33,  -11, -25, -36, 14,  16,  -22, 8,   9,   38,  -8,  -38, -7,  5,   -29, 2,   3,   -6,  -9,  -18, -19, 13,  36,  24,  7,   13,  -2,  1,   -5,  -47, -1,
    -2,  10,  19,  -19, 8,   -26, -24, -9,  -20, -43, -22, -13, -33, -6,  -3,  4,   -44, -5,  4,   -31, 19,  0,   22,  20,  2,   -16, 16,  -56, -18, 40,  -6,
    -14, 9,   7,   -41, 10,  22,  -2,  28,  1,   -33, -6,  -31, 10,  -23, 1,   7,   -6,  -2,  -19, 1,   29,  -1,  -20, 0,   1,   0,   29,  -14, 13,  10,  -21,
    10,  7,   6,   -27, -2,  21,  -8,  -2,  9,   4,   -2,  -1,  -19, -20, -5,  -40, 5,   9,   9,   -10, -18, 22,  8,   12,  56,  38,  -5,  -20, -6,  6,   5,
    22,  33,  -1,  10,  -1,  9,   4,   5,   -14, 11,  3,   6,   2,   -5,  17,  24,  2,   23,  -8,  3,   1,   9,   43,  36,  -1,  -41, -2,  -7,  10,  -6,  -5,
    0,   6,   4,   -3,  21,  46,  11,  17,  21,  -6,  9,   -4,  9,   11,  -13, 31,  -14, 5,   -1,  6,   -20, 6,   -10, 33,  5,   15,  34,  8,   12,  -9,  11,
    19,  -3,  18,  -4,  -6,  0,   -5,  -36, -8,  15,  -19, 34,  -2,  -38, 0,   -17, -1,  -12, 0,   11,  -6,  2,   -28, 5,   45,  6,   -12, -10, 19,  -2,  25,
    -11, 29,  -10, -24, 8,   9,   1,   -3,  15,  7,   1,   -9,  11,  12,  2,   -1,  -29, -12, 1,   -15, 16,  14,  5,   3,   -14, -3,  -12, 4,   48,  36,  -15,
    -31, 5,   7,   -1,  10,  39,  0,   6,   8,   -12, 11,  -17, -9,  9,   8,   18,  4,   3,   4,   17,  -3,  3,   2,   -8,  1,   -7,  10,  33,  -27, -16, -18,
    3,   3,   4,   6,   7,   -1,  -18, -9,  27,  35,  -6,  11,  14,  1,   12,  0,   0,   15,  -8,  23,  -3,  20,  8,   -14, -11, -12, -14, 33,  16,  5,   20,
    12,  9,   7,   6,   10,  -4,  -7,  -24, 9,   15,  31,  -84, 22,  6,   -24, 18,  35,  32,  -5,  -29, -10, -33, 2,   7,   -20, -1,  -9,  -25, 14,  -5,  35,
    14,  -32, -20, -17, -1,  -8,  7,   14,  10,  15,  -8,  -29, -33, -11, -18, 26,  -9,  4,   -4,  -37, -4,  -17, 3,   18,  -10, 18,  24,  -25, 1,   12,  -29,
    -19, -32, -11, -17, -57, -12, 17,  3,   13,  -39, 42,  -40, 44,  14,  -21, -36, -37, -9,  29,  12,  19,  26,  -23, 44,  -15, 31,  -53, -7,  10,  26,  19,
    -73, 34,  7,   36,  -15, 6,   -24, -29, -5,  12,  6,   24,  -51, 15,  -4,  24,  -7,  11,  -18, 5,   -24, -53, 25,  44,  -36, -38, -7,  8,   4,   33,  -63,
    -20, -37, -3,  52,  3,   0,   -38, -10, -1,  20,  -7,  0,   3,   -6,  -11, -9,  -9,  2,   -12, -2,  -4,  -8,  0,   -8,  -5,  -1,  -2,  -7,  0,   4,   -9,
    -3,  -8,  -8,  8,   -6,  2,   0,   -1,  4,   -12, -13, 1,   -5,  3,   -14, -10, 3,   4,   1,   -9,  7,   4,   -9,  -3,  -16, -5,  -5,  -10, -5,  3,   -8,
    -19, 3,   4,   -8,  -4,  -7,  3,   5,   -10, -2,  4,   -1,  0,   -5,  2,   -12, -8,  4,   4,   -5,  -1,  -1,  0,   0,   0,   6,   0,   -1,  0,   3,   -13,
    -6,  -5,  -12, 1,   4,   -13, -19, 1,   4,   -11, -5,  -11, 2,   -7,  6,   -12, -1,  -6,  -10, -17, 4,   -6,  -16, 5,   -1,  -3,  14,  2,   0,   -9,  -5,
    4,   3,   5,   1,   -8,  -10, -6,  -4,  -7,  -12, -10, 1,   7,   2,   32,  -1,  -17, 10,  0,   -37, -4,  17,  -4,  31,  4,   -36, -5,  -22, 1,   -7,  18,
    4,   0,   -5,  -24, 10,  29,  -1,  -8,  7,   17,  2,   23,  -19, 21,  -5,  -34, 1,   22,  1,   -18, -1,  4,   -6,  -2,  -2,  15,  9,   14,  -12, -18, -2,
    -33, 4,   6,   8,   -10, -19, 12,  9,   -4,  40,  28,  -1,  -11, -5,  -3,  -10, 19,  28,  7,   -1,  3,   0,   16,  -16, -27, -3,  2,   5,   -13, 7,   -2,
    11,  -8,  18,  2,   -2,  -3,  10,  36,  21,  11,  -33, 1,   -2,  13,  4,   -1,  -4,  -1,  6,   8,   25,  34,  -3,  18,  15,  7,   -6,  -6,  0,   -4,  -18,
    23,  -25, 28,  14,  7,   -20, 6,   -18, 41,  16,  2,   20,  20,  16,  -5,  6,   13,  -1,  27,  -23, -12, 6,   -6,  -25, -1,  12,  -19, 40,  -9,  -33, 10,
    -2,  -2,  -5,  9,   6,   -4,  -6,  -11, 9,   19,  -3,  4,   -7,  13,  -9,  32,  -14, 23,  -20, -30, 10,  21,  -8,  -15, 11,  3,   -10, -21, 4,   6,   10,
    7,   -22, -31, 8,   -25, 22,  21,  -7,  -1,  -16, -29, -4,  5,   18,  44,  -29, -40, 6,   -6,  3,   9,   37,  -1,  -5,  7,   -14, 19,  -15, -23, 8,   13,
    8,   -8,  11,  0,   36,  3,   -8,  16,  7,   0,   -8,  32,  36,  -9,  -14, -1,  -17, 11,  17,  9,   -4,  -3,  -27, -14, 19,  41,  -6,  21,  25,  -14, 11,
    -12, 1,   0,   -37, 26,  -2,  30,  14,  -15, -3,  -5,  -14, 34,  7,   6,   22,  -5,  17,  18,  18,  18,  -3,  44,  -16, -28, 6,   -12, -26, 5,   16,  -6,
    27,  -9,  -8,  7,   -22, -4,  -5,  5,   4,   5,   -5,  -13, -3,  25,  -4,  2,   -7,  19,  -30, 30,  -32, 7,   14,  -27, -5,  7,   5,   -21, 14,  10,  -2,
    20,  7,   2,   -1,  12,  0,   -1,  9,   -26, -1,  16,  8,   15,  -24, 9,   -7,  5,   42,  -36, -6,  -19, 7,   4,   -6,  33,  28,  -14, 4,   14,  -3,  16,
    -12, -12, 3,   -15, 6,   -7,  19,  14,  18,  -16, 18,  -2,  5,   10,  4,   35,  12,  18,  -17, 0,   -3,  -2,  9,   4,   -4,  4,   -6,  -9,  11,  50,  10,
    -48, 11,  -5,  4,   -6,  -7,  1,   -8,  11,  -10, 7,   15,  -12, -11, 11,  -8,  38,  1,   -4,  31,  9,   -5,  6,   20,  10,  13,  50,  -14, -24, 6,   -13,
    -42, 0,   26,  -15, 41,  2,   -44, 3,   1,   9,   -29, 8,   1,   14,  13,  -35, -7,  -1,  10,  -38, -42, 17,  -1,  9,   0,   35,  -20, -19, -11, 10,  -13,
    -20, -2,  43,  4,   -23, 21,  23,  -2,  13,  1,   -32, 8,   -34, 1,   -8,  -9,  3,   -28, -13, 6,   6,   52,  47,  -12, -22, 24,  5,   -16, 15,  7,   -32,
    14,  0,   -5,  22,  26,  -56, 6,   3,   24,  -32, -8,  -16, 3,   20,  3,   17,  5,   -4,  -5,  60,  43,  -2,  -25, -11, -28, -11, 5,   21,  8,   8,   -1,
    -18, 21,  66,  7,   25,  37,  -1,  15,  13,  7,   -7,  -26, 33,  -8,  36,  10,  1,   -27, 10,  -37, 40,  5,   -2,  32,  20,  22,  -3,  16,  16,  2,   -23,
    24,  24,  -7,  17,  37,  14,  -18, 17,  -15, 1,   41,  -2,  32,  15,  26,  8,   6,   13,  0,   32,  2,   8,   0,   10,  1,   4,   20,  -8,  40,  -8,  15,
    23,  10,  -28, 2,   1,   -7,  -8,  2,   7,   -5,  -9,  -1,  8,   7,   37,  3,   30,  -6,  -2,  -5,  3,   24,  4,   3,   4,   -55, -29, 5,   26,  19,  7,
    13,  3,   -34, 9,   -1,  -6,  -2,  2,   -16, 38,  16,  12,  -7,  17,  -6,  -1,  -16, 24,  -9,  -10, 7,   10,  5,   -41, -19, 13,  41,  -5,  -2,  -2,  -3,
    2,   11,  1,   7,   19,  -29, -53, -2,  -27, -7,  11,  4,   2,   -4,  -2,  14,  -22, 27,  -7,  -11, 6,   38,  6,   -9,  -37, -18, -17, -24, -19, 2,   18,
    -8,  -16, -4,  64,  -21, -13, -13, 3,   -21, -22, 9,   -26, 50,  -27, 0,   14,  -27, 14,  -7,  -8,  21,  9,   -3,  -28, 0,   13,  0,   -31, -9,  7,   -36,
    -28, 12,  47,  -7,  -24, -17, 30,  3,   -38, 20,  -12, -22, 8,   35,  24,  -22, -4,  -15, 18,  0,   -22, 3,   11,  -15, 18,  -17, 8,   -17, 23,  37,  -21,
    -27, -4,  -37, 18,  22,  43,  58,  -35, -1,  7,   -43, 26,  3,   -27, 10,  4,   36,  -31, 10,  10,  34,  -57, -12, 24,  5,   -1,  -20, 56,  17,  -8,  -27,
    -46, -50, -4,  20,  21,  -6,  1,   -57, -13, 12,  41,  16,  -67, 9,   -12, 7,   -3,  -8,  2,   -45, 13,  -17, 24,  15,  -13, -23, -4,  -16, 76,  14,  10,
    50,  4,   14,  26,  15,  16,  7,   28,  -26, -17, 17,  2,   -41, -1,  18,  -21, 52,  1,   -25, 18,  -12, 9,   -16, 29,  -2,  -7,  4,   -30, -7,  43,  15,
    -15, -7,  20,  -11, 18,  -22, 32,  -13, -27, 6,   31,  7,   -36, 17,  16,  -11, -4,  13,  3,   -9,  -1,  -17, -10, 14,  -40, 9,   22,  13,  -4,  -24, 6,
    -11, 13,  42,  41,  7,   -30, -4,  -2,  -14, 30,  35,  -18, 3,   4,   -22, 22,  8,   -28, 5,   13,  28,  -6,  12,  9,   39,  16,  -11, 8,   4,   0,   -12,
    47,  28,  -30, -33, -11, -20, -4,  -5,  -1,  -12, -2,  12,  -25, 40,  42,  9,   23,  22,  5,   14,  6,   13,  7,   -8,  44,  -17, 42,  29,  -21, -32, 2,
    -31, 41,  15,  9,   24,  13,  12,  13,  25,  20,  12,  42,  -12, -23, -2,  -8,  -35, -6,  13,  -16, 40,  5,   -17, 8,   -20, 15,  -7,  19,  -3,  4,   6,
    -22, 9,   23,  4,   -4,  4,   16,  -18, 19,  -20, 11,  6,   -35, 2,   9,   0,   -12, 6,   17,  -8,  9,   14,  10,  -1,  14,  2,   -9,  -1,  -21, 8,   7,
    9,   17,  -14, 4,   2,   7,   56,  31,  -6,  -19, -4,  7,   0,   29,  26,  -6,  5,   19,  -8,  10,  -5,  -8,  -2,  -9,  11,  -16, 15,  4,   25,  -17, 19,
    -2,  -8,  -14, 0,   31,  29,  -5,  -27, -3,  -3,  4,   2,   -1,  15,  -5,  1,   -17, 10,  58,  -12, -53, 21,  -12, -2,  -4,  10,  16,  3,   22,  -18, 26,
    20,  -7,  -20, 5,   -22, 30,  6,   6,   17,  16,  16,  11,  15,  6,   0,   46,  -7,  -5,  5,   5,   -27, -9,  29,  -18, 37,  -14, 0,   1,   0,   5,   3,
    22,  14,  5,   7,   -16, 4,   12,  10,  -23, -9,  8,   -26, 6,   3,   13,  -10, -12, 9,   20,  -6,  -36, 2,   26,  -8,  15,  20,  18,  19,  -7,  -18, -6,
    6,   -28, 24,  18,  11,  4,   -20, 39,  11,  13,  64,  47,  -1,  -29, -4,  4,   -2,  28,  42,  -13, 15,  -2,  -29, 22,  -7,  -14, 24,  4,   20,  -12, 19,
    5,   36,  5,   -42, 17,  2,   -9,  -18, 64,  26,  -43, -25, -10, -14, -5,  8,   12,  5,   -9,  -5,  -9,  45,  54,  -6,  -53, 22,  -1,  2,   3,   12,  -2,
    0,   17,  -29, 54,  3,   7,   -5,  -16, -1,  40,  13,  1,   21,  13,  0,   11,  27,  8,   -5,  27,  -9,  -28, -4,  5,   -21, -8,  14,  -24, 23,  3,   -9,
    6,   -23, 16,  -12, 31,  -5,  8,   10,  -29, 8,   24,  4,   -5,  -4,  -1,  -27, 31,  -29, 22,  0,   -40, 3,   17,  0,   -11, 4,   11,  -5,  1,   3,   1,
    14,  11,  -24, -11, 8,   -32, 5,   16,  5,   -4,  -24, -8,  2,   -6,  56,  50,  -15, -23, 0,   7,   0,   33,  35,  7,   2,   15,  -7,  24,  7,   -4,  -1,
    -6,  10,  -1,  12,  6,   20,  -12, -14, 11,  -9,  -5,  10,  22,  31,  -1,  -17, -3,  5,   10,  9,   5,   4,   -9,  -20, 0,   12,  33,  -7,  -50, 14,  4,
    9,   -1,  -5,  1,   -6,  24,  -9,  25,  15,  9,   0,   1,   -9,  35,  5,   -9,  25,  10,  16,  4,   15,  1,   16,  23,  -25, -20, -5,  -6,  -47, -4,  16,
    -9,  31,  4,   -18, 6,   -16, 3,   -15, 17,  5,   5,   10,  -30, -3,  19,  15,  4,   2,   4,   -11, 15,  -5,  27,  4,   -35, 6,   24,  -6,  -17, 10,  4,
    -5,  -2,  8,   10,  5,   13,  -5,  -17, 0,   -20, 1,   18,  9,   8,   -25, 1,   -9,  12,  39,  23,  -9,  -27, 10,  -2,  -6,  23,  34,  -13, -3,  4,   -4,
    9,   -6,  -21, 13,  -1,  20,  -11, 16,  10,  24,  -12, 8,   2,   0,   0,   -9,  40,  19,  0,   -14, 3,   2,   3,   6,   -3,  3,   -3,  -9,  -17, 23,  56,
    10,  11,  11,  -15, 13,  -16, 3,   9,   -5,  26,  -8,  9,   3,   -19, -14, -8,  -11, 35,  12,  -4,  31,  16,  16,  8,   12,  6,   -1,  38,  -24, -19, 9,
    -5,  -43, -2,  5,   -4,  16,  -1,  -4,  5,   -11, 4,   -6,  3,   -2,  16,  -4,  -40, 9,   -13, 14,  -14, -24, 15,  -30, 40,  2,   21,  4,   -23, 0,   11,
    -3,  -8,  19,  4,   -8,  12,  16,  10,  -13, 7,   -9,  -8,  1,   -29, 23,  -1,  -5,  9,   -43, 10,  0,   0,   36,  -17, -22, -28, -9,  -1,  14,  31,  -4,
    -9,  6,   18,  -4,  20,  -13, 4,   15,  -6,  5,   -1,  4,   4,   10,  0,   15,  1,   -5,  -7,  -21, 7,   30,  -32, -21, -1,  -12, -1,  18,  12,  8,   -6,
    -20, -12, 31,  48,  -1,  -58, 11,  -7,  -2,  -4,  0,   17,  -22, 26,  -20, 28,  17,  -19, -25, 0,   -11, -1,  13,  5,   33,  11,  5,   24,  25,  17,  -3,
    -41, 31,  11,  -47, -7,  21,  1,   -37, 30,  2,   1,   79,  -24, 29,  5,   -3,  26,  13,  -27, 14,  -1,  17,  -30, 4,   6,   -7,  -2,  23,  -27, -28, -43,
    -1,  -11, -6,  -29, 27,  -9,  -34, -34, -21, 60,  -21, -16, 8,   -12, -4,  -3,  -4,  52,  20,  17,  -52, -28, 52,  -21, 20,  -32, -2,  -20, 11,  3,   -17,
    28,  15,  -2,  -25, 14,  22,  31,  9,   27,  -15, 65,  29,  -2,  14,  1,   29,  4,   -41, -32, 8,   28,  -7,  -9,  -10, -52, -26, 46,  71,  5,   3,   -1,
    7,   -16, 11,  11,  14,  -23, -42, -41, 6,   -60, -41, 47,  -3,  11,  -13, -23, 13,  -16, 33,  15,  -39, -10, 46,  -8,  -8,  -29, -20, 5,   3,   -10, -13,
    22,  0,   -45, 15,  68,  -10, -21, -9,  -7,  -31, -37, 12,  -12, -29, -5,  -42, -10, 5,   16,  -3,  -28, 17,  -4,  11,  -13, -13, 25,  -2,  -15, -38, 14,
    1,   6,   40,  28,  -2,  -17, -6,  24,  -6,  4,   -1,  13,  -27, -24, 27,  26,  1,   -10, -5,  -21, 9,   -14, 6,   18,  -21, 30,  5,   -33, 1,   21,  48,
    50,  -20, -13, -5,  6,   5,   25,  23,  -5,  16,  -3,  -23, 19,  -3,  -35, 8,   4,   12,  -27, -15, -20, 26,  16,  -5,  13,  0,   -10, -39, 32,  24,  -25,
    -25, -48, -40, -13, 14,  24,  -32, -6,  -11, -29, 11,  2,   12,  34,  31,  -25, -2,  1,   5,   -4,  -16, 32,  -2,  24,  24,  -12, -1,  -28, 0,   58,  4,
    -7,  39,  8,   0,   14,  4,   -1,  -1,  70,  -32, -33, -1,  -18, -11, 10,  31,  -36, 42,  0,   -29, 11,  13,  19,  -6,  11,  -21, 22,  1,   -6,  -34, -29,
    7,   -31, -29, -13, 2,   -28, -28, 12,  -37, 8,   11,  3,   -11, -12, 8,   -2,  -17, 32,  -3,  32,  5,   -13, -36, 17,  -2,  -28, -7,  -12, 15,  -1,  -1,
    25,  9,   12,  -10, 38,  10,  -15, -7,  0,   14,  22,  -21, -27, 3,   23,  -73, -50, 41,  -4,  10,  -25, 4,   -20, 17,  31,  23,  2,   18,  -17, 10,  9,
    3,   18,  16,  16,  -32, -36, -9,  30,  21,  2,   15,  -6,  -24, -6,  23,  -29, 12,  -49, 3,   2,   13,  5,   0,   16,  2,   -57, -9,  13,  8,   -12, -26,
    13,  -6,  -82, 19,  -4,  -7,  -15, -22, -5,  16,  18,  9,   15,  28,  -24, 14,  -5,  -6,  -7,  -37, -2,  -18, -3,  39,  -8,  62,  10,  -43, -1,  37,  19,
    -2,  -40, -12, 56,  22,  8,   -1,  -7,  -11, -1,  24,  1,   15,  -53, 5,   -17, -1,  -23, -38, -42, 14,  38,  14,  23,  -38, -14, -34, -73, 0,   15,  -24,
    -23, -17, -12, -55, 4,   -19, 2,   6,   6,   3,   -37, -9,  -3,  3,   -17, -31, 11,  21,  -35, -4,  8,   48,  -12, -4,  -3,  11,  -2,  -14, 5,   -44, 22,
    -18, 7,   8,   -19, -23, 28,  -40, 52,  -27, 6,   2,   4,   -57, 5,   -20, 10,  -18, 14,  -60, -39, 5,   25,  -22, -19, 32,  12,  25,  6,   25,  -22, -9,
    -52, -37, -6,  4,   19,  26,  8,   21,  -1,  9,   -7,  -7,  23,  -4,  -42, -4};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer3_bias_tensor[128] = {
    588016,   11289496, -1690032, -355688,  1566672,  986880,   6188560,  -6462008, 2798216,   6482568,  -2668688, -10105240, 4700016,  -995104,  -2664576,
    -5201680, 9926368,  -3291656, 6100152,  11974144, 680536,   10415696, 5251024,  11852840,  3170352,  -4798704, 6817696,   -8579688, 10012720, 2247208,
    5594376,  559232,   5980904,  -4480024, -3480808, 1443312,  10767272, -6462008, 5645776,   2966808,  8694824,  -9669368,  -3556880, 3283432,  8616696,
    5894552,  -3663792, 12181800, 9955152,  8596136,  1562560,  5283920,  16172496, 10261496,  -2942136, -5201680, -17017512, 11351176, 10522608, 11758264,
    7430384,  3937240,  -571568,  17229280, 7294688,  -6141272, 840904,   13133728, -11310056, 17253952, 4878888,  4251808,   -3920792, -9048456, 11367624,
    9747496,  8335024,  -4181904, 7021240,  5902776,  -5777360, -803896,  -5014584, 8421376,   7372816,  17270400, 3688464,   10236824, 8139704,  14618160,
    11766488, -9130696, 123360,   16042968, 3501368,  6963672,  1864792,  2991480,  3904344,   7446832,  -7132264, 5619048,   7319360,  -966320,  -4091440,
    2432248,  9844128,  -5251024, -5863712, 4385448,  3063440,  -5769136, -8224,    8008120,   -6155664, 11069504, 509888,    16201280, -1509104, -4998136,
    -4899448, -6394160, -3215584, 986880,   -3003816, 13676512, -7481784, -8478944};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer3_mul_tensor[128] = {
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056,
    2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056, 2056};

static PI_L2 int8_t AnomalyDetection_heavy_6_layer4_weight_tensor[256] = {
    -18, 40,  33,  23,  86,  -78, -64, -11, -53, 15,  -16, 0,   24,  76,  5,   110, 31,  8,   -20, -53, 6,   50,  52,  -44, -10, 60,  -27, 10,  16,
    13,  -64, -31, 41,  -25, -18, 21,  24,  -54, 18,  -47, -84, 6,   -22, 13,  8,   25,  4,   28,  20,  -75, -23, 23,  46,  48,  -69, -20, -76, 24,
    -37, 41,  13,  25,  87,  -68, -38, -17, 26,  45,  -78, -80, -34, -17, -17, -64, -86, 54,  9,   -23, 21,  45,  -64, -20, 6,   25,  -44, -37, 20,
    37,  24,  62,  24,  5,   26,  75,  -13, -25, 69,  -86, 11,  16,  92,  44,  -24, -22, 1,   28,  -39, 66,  -13, -22, -94, 4,   -20, -27, -16, -39,
    24,  -70, -41, -13, -42, -23, -11, -29, 71,  -67, -84, 65,  20,  -35, -36, -40, -86, 66,  66,  18,  60,  -26, 21,  -11, -9,  -60, -4,  -97, -22,
    9,   23,  42,  7,   -32, -37, 38,  13,  -60, 31,  3,   -23, -27, 54,  22,  -53, 13,  29,  -15, -26, 58,  -20, 40,  87,  1,   34,  -21, -21, -14,
    -5,  -35, -27, 72,  19,  -21, -43, -51, 68,  17,  71,  -30, 40,  -45, -32, -28, -79, 80,  42,  33,  -20, -33, 69,  85,  29,  16,  26,  50,  87,
    -52, -24, 10,  -24, -46, 50,  18,  -1,  -29, 42,  52,  -21, -43, -26, -79, -30, -3,  -13, -72, 25,  20,  -65, 74,  -20, -18, -90, -41, 40,  4,
    2,   -14, 36,  -82, 22,  21,  85,  0,   7,   29,  7,   39,  -18, 65,  37,  21,  50,  20,  20,  34,  -59, 62,  80,  -70};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer4_bias_tensor[2] = {1512, -3395};

static PI_L2 int32_t AnomalyDetection_heavy_6_layer4_mul_tensor[2] = {1, 1};

static PI_L1 uint8_t AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[2] = {0, 2};

static PI_L1 uint8_t AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_numTiles[2] = {0, 1};

static PI_L1 uint8_t AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_numTiles[2] = {0, 1};

static PI_L1 uint8_t AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_numTiles[2] = {0, 1};

static PI_L1 uint8_t AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_numTiles[2] = {0, 1};

void *AnomalyDetection_heavy_6_inputs[1];
void *AnomalyDetection_heavy_6_outputs[1];

// ===== GLOBALS from AnomalyDetection_heavy_6 END =====


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

void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_0_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_1_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_2_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_3_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_4_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_5_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_6_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_7_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_8_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2__MERGE_CONVRQ_PASS_9_buffer;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_avgpoolGlobalAveragePool_output_0_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_flattenFlatten_output_0_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref;
void *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_out_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer0_out_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_weight_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_bias_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_mul_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_out_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer1_out_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_weight_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_bias_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_mul_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_out_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer2_out_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_weight_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_bias_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_mul_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_out_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer3_out_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_weight_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_bias_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_mul_tensor_ref;
void *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_output_0_ref;

typedef struct {
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
  uint8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
  uint8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr;
  uint8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr;
  uint8_t *ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr;

  uint8_t *ImageClassification_small_2_flattenFlatten_output_0_tensor;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;

  uint8_t *AnomalyDetection_heavy_6_layer0_out_tensor;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
  uint8_t *AnomalyDetection_heavy_6_layer1_out_tensor;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr;
  uint8_t *AnomalyDetection_heavy_6_layer2_out_tensor;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr;
  uint8_t *AnomalyDetection_heavy_6_layer3_out_tensor;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr;

} Combined_closure_2_6_args_t;

static void Combined_closure_2_6(Combined_closure_2_6_args_t* closure_args) {
    Combined_closure_2_6_args_t* args = (Combined_closure_2_6_args_t*) closure_args;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed = args->ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
  uint8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      args->ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
  uint8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      args->ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr;
  uint8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr;
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
      args->ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr;
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr;
  uint8_t *ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor = args->ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr;

  uint8_t *ImageClassification_small_2_flattenFlatten_output_0_tensor = args->ImageClassification_small_2_flattenFlatten_output_0_tensor;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      args->ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;

  uint8_t *AnomalyDetection_heavy_6_layer0_out_tensor = args->AnomalyDetection_heavy_6_layer0_out_tensor;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      args->AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
  uint8_t *AnomalyDetection_heavy_6_layer1_out_tensor = args->AnomalyDetection_heavy_6_layer1_out_tensor;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr =
      args->AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr;
  uint8_t *AnomalyDetection_heavy_6_layer2_out_tensor = args->AnomalyDetection_heavy_6_layer2_out_tensor;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr =
      args->AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr;
  uint8_t *AnomalyDetection_heavy_6_layer3_out_tensor = args->AnomalyDetection_heavy_6_layer3_out_tensor;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr =
      args->AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr =
      args->AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr;



    int core_id = pi_core_id();
    if (core_id == 0) {
        subgroup_barrier_init(2, 0, 1);
        subgroup_barrier_init(3, 2, 7);
    }
    subgroup_barrier_init(4, 0, NUM_CORES - 1);
    pi_cl_team_barrier();
    eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);


        uint32_t ImageClassification_small_2_channel_input = (uint32_t)-1;
        uint32_t ImageClassification_small_2_channel_output = (uint32_t)-1;

        if (0 <= core_id && core_id <= 1) {
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 3072);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref=
      (void *)((char *)ImageClassification_small_2_input_0 + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref=
      (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1444864, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // Transpose [1, 3, 32, 32] -> [1, 32, 32, 3] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 3;

  uint16_t dimLen_2 = 32;

  uint16_t dimLen_3 = 32;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / 2;
    const uint32_t leftover = dimLen_2 - baseChunk * 2;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          ((int8_t (*)[dimLen_2][dimLen_3][dimLen_1])
               ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref)[i_0][i_2][i_3][i_1] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                   ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1313792, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_y_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_y + 0);
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_y_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_y + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_left_ref =
      (uint8_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_left + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_right_ref =
      (uint8_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_right + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 12288);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 14784);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15648);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15712);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref=
      (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref=
      (void *)((char *)ImageClassification_small_2_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref=
      (void *)((char *)ImageClassification_small_2_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_0_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15216);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[TILING_I],
                                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref,
                                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size_1d[TILING_I], 96);

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
        (void *)((char *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref) + 69);

    mchan_transfer_1d(1442224, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref);
    mchan_transfer_1d(1441856, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref);
    mchan_transfer_1d(1441856, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_pilot_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_right_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_right_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_right[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_left_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_left_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_left[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_y_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_y_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_y[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_y_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_y_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_y[TILING_I];

            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_i8_u8_i8(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_0_buffer,
                        NULL, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, 1, 19,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_y_ref, 32, 3,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_y_ref, 32, 16, 3, 3, 1, 1,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_left_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_x_right_ref, 1, 1, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[TILING_I],
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size_1d[TILING_I], 512);

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref) +
                384);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x + 0);
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_y_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_y + 0);
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x + 0);
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_y_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_y + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15916);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15948);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15956);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 13920);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1_ref=
          (void *)((char *)
                       ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1_ref=
          (void *)((char *)
                       ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_1_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15660);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cmd[TILING_I],
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_size_1d[TILING_I], 512);

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        1441824, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref);

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        1441800, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1_ref);

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_1_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        1441800, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1_ref);

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_1_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_relativeOffset[TILING_I]);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_y_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_y_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_y[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_y_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_y_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_y[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x[TILING_I];

            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_1_buffer,
                        NULL, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref, 1, 19,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_y_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_ref, 16,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_y_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_ref, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    for (uint32_t i_0 = 0; i_0 < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_end_0[TILING_I]; i_0++) {
      const uint32_t ext_offset = i_0 * 128;
      const uint32_t loc_offset = i_0 * ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_stride_0[TILING_I];
      void *const local_buffer_offsetted = (void *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref + loc_offset);
      void *const external_buffer_offsetted =
          (void
               *)((char *)
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref +
                  ext_offset);
      mchan_transfer_2d_ext_strided(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cmd[TILING_I], local_buffer_offsetted,
                                    external_buffer_offsetted, 2, 4);
    }

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_x_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_x + 0);
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_y_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_y + 0);
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_x_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_x + 0);
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_y_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_y + 0);
  uint16_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_ch_im_out_ref =
      (uint16_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_ch_im_out + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_top_ref =
      (uint8_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_top + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_bottom_ref =
      (uint8_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_bottom + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_left_ref =
      (uint8_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_left + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_right_ref =
      (uint8_t *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_right + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15539);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15971);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 15983);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 13712);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2_ref=
          (void *)((char *)
                       ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2_ref=
          (void *)((char *)ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2 + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)
                       ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_2_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 11408);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_cmd[TILING_I],
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_size_1d[TILING_I], 512);

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_cmd[TILING_I],
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref);

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_cmd[TILING_I],
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2_ref);

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_2_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_relativeOffset[TILING_I]);

    mchan_transfer_1d(
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_cmd[TILING_I],
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2_ref);

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_2_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_relativeOffset[TILING_I]);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_right_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_right_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_right[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_left_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_left_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_left[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_bottom_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_bottom_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_bottom[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_top_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_top_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_top[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_ch_im_out_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_ch_im_out_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_ch_im_out[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_y_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_y_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_y[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_x_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_x_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_x[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_y_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_y_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_y[TILING_I];

    // UPDATE VARIABLE ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_x_ref
    *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_x_ref =
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_x[TILING_I];

            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_2_buffer,
                        NULL, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
                        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref, 1, 19,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_y_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_in_x_ref, 16,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_y_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_dim_im_out_x_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_ch_im_out_ref, 3, 3,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_top_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_y_bottom_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_left_ref,
                        *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_padding_x_right_ref, 1, 1, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    for (uint32_t i_0 = 0; i_0 < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_end_0[TILING_I]; i_0++) {
      const uint32_t ext_offset = i_0 * 128;
      const uint32_t loc_offset = i_0 * ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_stride_0[TILING_I];
      void *const local_buffer_offsetted = (void *)((char *)ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref + loc_offset);
      void *const external_buffer_offsetted =
          (void
               *)((char *)
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref +
                  ext_offset);
      mchan_transfer_2d_ext_strided(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_cmd[TILING_I], local_buffer_offsetted,
                                    external_buffer_offsetted, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_size_1d[TILING_I],
                                    4);
    }

    // UPDATE VARIABLE
    // ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref
    ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref =
        (void
             *)((char
                     *)(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref) +
                ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 8768);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 8912);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 8928);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 4096);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)
                       ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3_ref=
          (void *)((char *)
                       ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3_ref=
          (void *)((char *)ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3 + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_3_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 8192);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1445888, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1441936, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_3_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_3_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_3_buffer, NULL,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref, 1, 19, 32, 32, 4, 32, 32, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1314816, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 4096);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 8192);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1445888, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1445888, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN RQADD
  pulp_nn_add_i8_i8_u8(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
                       ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
                       ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 1, 0, 0, 1, 0, 0, 1048576, 524288, 19, 1, 4096, 1, 1,
                       2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1314816, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 5184);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 5200);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 5216);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 4096);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5_ref=
          (void *)((char *)
                       ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5_ref=
          (void *)((char *)
                       ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_4_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 5120);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1445888, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features10downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_6weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_5_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_5_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_4_buffer, NULL,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref, 1, 19, 32, 32, 4, 16, 16, 4, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311744, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 5696);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 5840);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 5856);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 4096);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6_ref=
          (void *)((char *)
                       ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6_ref=
          (void *)((char *)ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6 + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)
                       ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_5_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 5120);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1445888, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1441936, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_6_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_features00_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_DUPLICATE_FOR_RequantShift_6_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_5_buffer, NULL,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_weight_ref, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_mul_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_add_ref, 1, 19, 32, 32, 4, 16, 16, 4, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311744, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2624);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2768);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2784);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1024);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)
                       ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7_ref=
          (void *)((char *)
                       ImageClassification_small_2_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7_ref=
          (void *)((char *)ImageClassification_small_2_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7 + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_6_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2048);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442816, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1441936, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features10_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_5weight_tensor_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1add_tensor_DUPLICATE_FOR_RequantShift_7_ref);
    mchan_transfer_1d(
        1441808, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_features00_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0mul_tensor_DUPLICATE_FOR_RequantShift_7_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_6_buffer, NULL,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_weight_ref, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_mul_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_add_ref, 1, 19, 16, 16, 4, 16, 16, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311744, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1024);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2048);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split + 0);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442816, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1442816, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN RQADD
  pulp_nn_add_i8_i8_u8(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_1_ref,
                       ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_in_2_ref,
                       ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref, 1, 0, 0, 1, 0, 0, 1048576, 524288, 19, 1, 1024, 1, 1,
                       2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311744, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1600);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1632);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1664);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1024);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9_ref=
          (void *)((char *)
                       ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_7_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1536);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442816, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1441824, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_9weight_tensor_ref);
    mchan_transfer_1d(
        1441824, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_9_ref);
    mchan_transfer_1d(
        1441824, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_7_buffer, NULL,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_weight_ref, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_mul_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_add_ref, 1, 19, 16, 16, 4, 8, 8, 8, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2112);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2400);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2432);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1600);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10_ref=
          (void *)((char *)
                       ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)
                       ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_8_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1024);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442816, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split_ref);
    mchan_transfer_1d(
        1442080, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_7weight_tensor_ref);
    mchan_transfer_1d(
        1441824, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_10_ref);
    mchan_transfer_1d(
        1441824, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_features20_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_8_buffer, NULL,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_weight_ref, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_mul_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_add_ref, 1, 19, 16, 16, 4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1728);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1152);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2752);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2784);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 2240);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)
                       ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11_ref=
          (void *)((char *)
                       ImageClassification_small_2_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11 +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor_ref=
      (void *)((char *)ImageClassification_small_2_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2__MERGE_CONVRQ_PASS_9_buffer= (void *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442304, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1442368, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_8weight_tensor_ref);
    mchan_transfer_1d(
        1441824, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20downsample_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5add_tensor_DUPLICATE_FOR_RequantShift_11_ref);
    mchan_transfer_1d(
        1441824, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_features20_QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN CONV

  pulp_nn_conv_u8_i8_i8(
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_in_ref, ImageClassification_small_2__MERGE_CONVRQ_PASS_9_buffer, NULL,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_weight_ref, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_mul_ref,
      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_add_ref, 1, 19, 8, 8, 8, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 512);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 1024);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref=
          (void *)((char *)ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed +
                   0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp + 0);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442304, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(
        1442304, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN RQADD
  pulp_nn_add_i8_i8_u8(ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_1_ref,
                       ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_in_2_ref,
                       ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref, 1, 0, 0, 1, 0, 0, 1048576, 524288, 19, 1, 512, 1, 1,
                       2);

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 512);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor + 0);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442304, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // Transpose [1, 8, 8, 8] -> [1, 8, 8, 8] (Name: _MERGE_ADDRQ_PASS_2_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / 2;
    const uint32_t leftover = dimLen_3 - baseChunk * 2;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((uint8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref)[i_0][i_3][i_1][i_2] =
              ((uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                   ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311232, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_data_out_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 512);
ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_avgpoolGlobalAveragePool_output_0_tensor_ref=
      (void *)((char *)ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor + 0);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442304, ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_in_ref,
        ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }

        }
        subgroup_barrier_wait(2, core_id);

  // ReduceMean (Name: avgpoolGlobalAveragePool, Op: ReduceMean)
  BEGIN_SINGLE_CORE
  int32_t ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator = 0;

  uint8_t *dummy_ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref =
      ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref;

  for (uint32_t i_0 = 0; i_0 < 1; i_0++) {
    for (uint32_t i_1 = 0; i_1 < 8; i_1++) {
      ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator = 128 * 64;
      for (uint32_t i_2 = 0; i_2 < 8; i_2++) {
        for (uint32_t i_3 = 0; i_3 < 8; i_3++) {
        }
      }
      int8_t ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_sgn = 0;
      ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_sgn =
          -(ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator < 0) +
          (ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator >= 0);
      *dummy_ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref++ =
          (uint8_t)((ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_accumulator +
                     ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref_sgn * (64 >> 1)) /
                        64 +
                    0);
    }
  }
  END_SINGLE_CORE

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310728, ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_data_out_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_avgpoolGlobalAveragePool_output_0_tensor_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 170);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 10);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 130);
  int32_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref = (int32_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 90);
  int8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L1 + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_flattenFlatten_output_0_tensor_ref=
      (void *)((char *)ImageClassification_small_2_flattenFlatten_output_0_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref=
      (void *)((char *)ImageClassification_small_2__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref=
      (void *)((char *)ImageClassification_small_2_RequantShift_final_mul_tensor + 0);
ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref= (void *)((char *)ImageClassification_small_2_output_0 + 0);
  // TILING LOOP
  for (int TILING_I = ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles
           [*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr];
       TILING_I < ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles
                      [(*ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 0) {
    // Transfer input tiles
    ImageClassification_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441800, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_flattenFlatten_output_0_tensor_ref);
    mchan_transfer_1d(1441872, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref);
    mchan_transfer_1d(1441832, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref);
    mchan_transfer_1d(1441832, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref);

    // Wait for input tiles

    if (ImageClassification_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_input);
      mchan_channel_free(ImageClassification_small_2_channel_input);
    }


            }
        subgroup_barrier_wait(2, core_id);

  // PULP NN GEMM
  int8_t *
      ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref =
          ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *
      ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref =
          ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int8_t *
      ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
          ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
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
          ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
          NULL,
          ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref,
          ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
          ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
          1, 0, 8, 10, 1, 1, 2);
    }
  }

        subgroup_barrier_wait(2, core_id);
        if (core_id == 0) {
    // Transfer output tiles
    ImageClassification_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310730, ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref,
                      ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref);

    // Wait for output tiles

    if (ImageClassification_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(ImageClassification_small_2_channel_output);
      mchan_channel_free(ImageClassification_small_2_channel_output);
    }

        }
        subgroup_barrier_wait(2, core_id);
      }
    }

        uint32_t AnomalyDetection_heavy_6_channel_input = (uint32_t)-1;
        uint32_t AnomalyDetection_heavy_6_channel_output = (uint32_t)-1;

        if (2 <= core_id && core_id <= 7) {
  int8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 40960);
  int8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 0);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 41856);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 41600);
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 42112);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref= (void *)((char *)AnomalyDetection_heavy_6_input_0 + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref=
      (void *)((char *)AnomalyDetection_heavy_6_layer0_weight_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer0_bias_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer0_mul_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_out_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer0_out_tensor + 0);
  // TILING LOOP
  for (int TILING_I =
           AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr];
       TILING_I <
       AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[(*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 2) {
    // Transfer input tiles
    AnomalyDetection_heavy_6_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442432, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref);
    mchan_transfer_1d(1482752, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref);

    // UPDATE VARIABLE AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref
    AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref =
        (void *)((char *)(AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref) + 40960);

    mchan_transfer_1d(1442048, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref);

    // UPDATE VARIABLE AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref
    AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref =
        (void *)((char *)(AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref) + 256);

    mchan_transfer_1d(1442048, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref);

    // UPDATE VARIABLE AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref
    AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref =
        (void *)((char *)(AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref) + 256);

    // Wait for input tiles

    if (AnomalyDetection_heavy_6_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_input);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_input);
    }


            }
        subgroup_barrier_wait(3, core_id);

  // PULP NN GEMM
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int8_t
      *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;
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
      pulp_nn_linear_i8_u8_i8(
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
          NULL,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref, 1, 18, 640,
          64, 1, 1, 6);
    }
  }

        subgroup_barrier_wait(3, core_id);
        if (core_id == 2) {
    // Transfer output tiles
    AnomalyDetection_heavy_6_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_out_tensor_ref);

    // UPDATE VARIABLE AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_out_tensor_ref
    AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_out_tensor_ref =
        (void *)((char *)(AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_out_tensor_ref) + 64);

    // Wait for output tiles

    if (AnomalyDetection_heavy_6_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_output);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_output);
    }

        }
        subgroup_barrier_wait(3, core_id);
      }
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_A_ref = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 17408);
  int8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_B_ref = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 0);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_mul_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 16896);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_C_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 16384);
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 17536);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer0_out_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer0_out_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_weight_tensor_ref=
      (void *)((char *)AnomalyDetection_heavy_6_layer1_weight_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_bias_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer1_bias_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_mul_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer1_mul_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_out_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer1_out_tensor + 0);
  // TILING LOOP
  for (int TILING_I =
           AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_numTiles[*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr];
       TILING_I <
       AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_numTiles[(*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 2) {
    // Transfer input tiles
    AnomalyDetection_heavy_6_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_A_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer0_out_tensor_ref);
    mchan_transfer_1d(1458176, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_B_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_weight_tensor_ref);
    mchan_transfer_1d(1442304, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_C_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_bias_tensor_ref);
    mchan_transfer_1d(1442304, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_mul_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_mul_tensor_ref);

    // Wait for input tiles

    if (AnomalyDetection_heavy_6_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_input);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_input);
    }


            }
        subgroup_barrier_wait(3, core_id);

  // PULP NN GEMM
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_A_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_A_ref;
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_B_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_B_ref;
  int8_t
      *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref =
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref;
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
      pulp_nn_linear_u8_u8_i8(
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_A_ref,
          NULL,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_B_ref,
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_mul_ref, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_C_ref, 1, 18, 128,
          128, 1, 1, 6);
    }
  }

        subgroup_barrier_wait(3, core_id);
        if (core_id == 2) {
    // Transfer output tiles
    AnomalyDetection_heavy_6_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_data_out_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_layer1_out_tensor_ref);

    // Wait for output tiles

    if (AnomalyDetection_heavy_6_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_output);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_output);
    }

        }
        subgroup_barrier_wait(3, core_id);
      }
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_A_ref = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 17408);
  int8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_B_ref = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 0);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_mul_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 16896);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_C_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 16384);
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 17536);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer1_out_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer1_out_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_weight_tensor_ref=
      (void *)((char *)AnomalyDetection_heavy_6_layer2_weight_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_bias_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer2_bias_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_mul_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer2_mul_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_out_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer2_out_tensor + 0);
  // TILING LOOP
  for (int TILING_I =
           AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_numTiles[*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr];
       TILING_I <
       AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_numTiles[(*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 2) {
    // Transfer input tiles
    AnomalyDetection_heavy_6_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_A_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer1_out_tensor_ref);
    mchan_transfer_1d(1458176, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_B_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_weight_tensor_ref);
    mchan_transfer_1d(1442304, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_C_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_bias_tensor_ref);
    mchan_transfer_1d(1442304, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_mul_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_mul_tensor_ref);

    // Wait for input tiles

    if (AnomalyDetection_heavy_6_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_input);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_input);
    }


            }
        subgroup_barrier_wait(3, core_id);

  // PULP NN GEMM
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_A_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_A_ref;
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_B_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_B_ref;
  int8_t
      *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref =
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref;
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
      pulp_nn_linear_u8_u8_i8(
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_A_ref,
          NULL,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_B_ref,
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_mul_ref, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_C_ref, 1, 18, 128,
          128, 1, 1, 6);
    }
  }

        subgroup_barrier_wait(3, core_id);
        if (core_id == 2) {
    // Transfer output tiles
    AnomalyDetection_heavy_6_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_data_out_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_layer2_out_tensor_ref);

    // Wait for output tiles

    if (AnomalyDetection_heavy_6_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_output);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_output);
    }

        }
        subgroup_barrier_wait(3, core_id);
      }
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_A_ref = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 17408);
  int8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_B_ref = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 0);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_mul_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 16896);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_C_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 16384);
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 17536);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer2_out_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer2_out_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_weight_tensor_ref=
      (void *)((char *)AnomalyDetection_heavy_6_layer3_weight_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_bias_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer3_bias_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_mul_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer3_mul_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_out_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer3_out_tensor + 0);
  // TILING LOOP
  for (int TILING_I =
           AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_numTiles[*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr];
       TILING_I <
       AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_numTiles[(*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 2) {
    // Transfer input tiles
    AnomalyDetection_heavy_6_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_A_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer2_out_tensor_ref);
    mchan_transfer_1d(1458176, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_B_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_weight_tensor_ref);
    mchan_transfer_1d(1442304, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_C_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_bias_tensor_ref);
    mchan_transfer_1d(1442304, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_mul_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_mul_tensor_ref);

    // Wait for input tiles

    if (AnomalyDetection_heavy_6_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_input);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_input);
    }


            }
        subgroup_barrier_wait(3, core_id);

  // PULP NN GEMM
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_A_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_A_ref;
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_B_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_B_ref;
  int8_t
      *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref =
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref;
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
      pulp_nn_linear_u8_u8_i8(
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_A_ref,
          NULL,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_B_ref,
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_mul_ref, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_C_ref, 1, 18, 128,
          128, 1, 1, 6);
    }
  }

        subgroup_barrier_wait(3, core_id);
        if (core_id == 2) {
    // Transfer output tiles
    AnomalyDetection_heavy_6_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_data_out_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_layer3_out_tensor_ref);

    // Wait for output tiles

    if (AnomalyDetection_heavy_6_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_output);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_output);
    }

        }
        subgroup_barrier_wait(3, core_id);
      }
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_A_ref = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 258);
  int8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_B_ref = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 2);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_mul_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 394);
  int32_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_C_ref = (int32_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 386);
  int8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L1 + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer3_out_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer3_out_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_weight_tensor_ref=
      (void *)((char *)AnomalyDetection_heavy_6_layer4_weight_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_bias_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer4_bias_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_mul_tensor_ref= (void *)((char *)AnomalyDetection_heavy_6_layer4_mul_tensor + 0);
AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_output_0_ref= (void *)((char *)AnomalyDetection_heavy_6_output_0 + 0);
  // TILING LOOP
  for (int TILING_I =
           AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_numTiles[*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr];
       TILING_I <
       AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_numTiles[(*AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr) + 1];
       TILING_I++) {
        if (core_id == 2) {
    // Transfer input tiles
    AnomalyDetection_heavy_6_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_A_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer3_out_tensor_ref);
    mchan_transfer_1d(1442048, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_B_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_weight_tensor_ref);
    mchan_transfer_1d(1441800, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_C_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_bias_tensor_ref);
    mchan_transfer_1d(1441800, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_mul_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_layer4_mul_tensor_ref);

    // Wait for input tiles

    if (AnomalyDetection_heavy_6_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_input);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_input);
    }


            }
        subgroup_barrier_wait(3, core_id);

  // PULP NN GEMM
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_A_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_A_ref;
  int8_t *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_B_ref =
      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_B_ref;
  int8_t
      *ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref =
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref;
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
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_A_ref,
          NULL,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref,
          ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_B_ref,
          AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_mul_ref, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_C_ref, 1, 12, 128,
          2, 1, 1, 6);
    }
  }

        subgroup_barrier_wait(3, core_id);
        if (core_id == 2) {
    // Transfer output tiles
    AnomalyDetection_heavy_6_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310722, AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_data_out_ref,
                      AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_output_0_ref);

    // Wait for output tiles

    if (AnomalyDetection_heavy_6_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_heavy_6_channel_output);
      mchan_channel_free(AnomalyDetection_heavy_6_channel_output);
    }

        }
        subgroup_barrier_wait(3, core_id);
      }
    }
    subgroup_barrier_wait(4, core_id);
}


void RunNetwork() {
  int8_t *ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed;                                      // L2_init_template
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split; // L2_init_template
  int8_t
      *ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *
      ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed; // L2_init_template
  int8_t
      *ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split;                              // L2_init_template
  int8_t
      *ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *
      ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed; // L2_init_template
  int8_t
      *ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split;                              // L2_init_template
  int8_t
      *ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *
      ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed; // L2_init_template
  int8_t
      *ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp;                                 // L2_init_template
  uint8_t *ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;                                    // L2_init_template
  uint8_t *ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor;                                                            // L2_init_template
  uint8_t *ImageClassification_small_2_flattenFlatten_output_0_tensor;                                                                      // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_0_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_1_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_2_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_3_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_4_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_5_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_6_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_7_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_8_buffer;                                                                            // L2_init_template
  void *ImageClassification_small_2__MERGE_CONVRQ_PASS_9_buffer;                                                                            // L2_init_template

  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr;
  uint8_t bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      &bu_ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;

  uint8_t *AnomalyDetection_heavy_6_layer0_out_tensor; // L2_init_template
  uint8_t *AnomalyDetection_heavy_6_layer1_out_tensor; // L2_init_template
  uint8_t *AnomalyDetection_heavy_6_layer2_out_tensor; // L2_init_template
  uint8_t *AnomalyDetection_heavy_6_layer3_out_tensor; // L2_init_template

  uint8_t bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr = 0;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      &bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
  uint8_t bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr = 0;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr =
      &bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr;
  uint8_t bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr = 0;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr =
      &bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr;
  uint8_t bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr = 0;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr =
      &bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr;
  uint8_t bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr = 0;
  uint8_t *AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr =
      &bu_AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr;


  ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 0);

  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 4096);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 0);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 20480);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 8192);

  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 4096);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 0);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 1024);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 2048);

  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 1024);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 0);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 512);

  ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 1032);

  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 520);

  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 8);

  ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor = (uint8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 0);

  ImageClassification_small_2_flattenFlatten_output_0_tensor = ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor;

  AnomalyDetection_heavy_6_layer0_out_tensor = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L2 + 0);

  AnomalyDetection_heavy_6_layer1_out_tensor = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L2 + 128);

  AnomalyDetection_heavy_6_layer2_out_tensor = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L2 + 0);

  AnomalyDetection_heavy_6_layer3_out_tensor = (uint8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L2 + 128);


Combined_closure_2_6_args_t Combined_closure_args = (Combined_closure_2_6_args_t) {

              .ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed = ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr,

              .ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed = ImageClassification_small_2__MERGE_CONVRQ_PASS_0_input_0_transposed,
              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr,

              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr,

              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr,

              .ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr,

              .ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr,

              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr,

              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_split,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_5_tileIdxPtr,

              .ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_5__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_6_tileIdxPtr,

              .ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_6__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_1_tileIdxPtr,

              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_7_tileIdxPtr,

              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_split,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_8_tileIdxPtr,

              .ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_8__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_9_tileIdxPtr,

              .ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_7__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_5Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
                  ImageClassification_small_2__MERGE_CONVRQ_PASS_9__QL_REPLACED__INTEGERIZE_BN2D_SIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_tileIdxPtr,

              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_tp,
              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_2_transpose_tileIdxPtr,

              .ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  ImageClassification_small_2__QL_REPLACED__INTEGERIZE_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor = ImageClassification_small_2_avgpoolGlobalAveragePool_output_0_tensor,
              .ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1_avgpoolGlobalAveragePool_tileIdxPtr,

              .ImageClassification_small_2_flattenFlatten_output_0_tensor = ImageClassification_small_2_flattenFlatten_output_0_tensor,
              .ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
                  ImageClassification_small_2_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr,
.AnomalyDetection_heavy_6_layer0_out_tensor = AnomalyDetection_heavy_6_layer0_out_tensor,
                                                                     .AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
                                                                         AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr,
.AnomalyDetection_heavy_6_layer0_out_tensor = AnomalyDetection_heavy_6_layer0_out_tensor,
                                                                     .AnomalyDetection_heavy_6_layer1_out_tensor = AnomalyDetection_heavy_6_layer1_out_tensor,
                                                                     .AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr =
                                                                         AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_1_tileIdxPtr,
.AnomalyDetection_heavy_6_layer1_out_tensor = AnomalyDetection_heavy_6_layer1_out_tensor,
                                                                     .AnomalyDetection_heavy_6_layer2_out_tensor = AnomalyDetection_heavy_6_layer2_out_tensor,
                                                                     .AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr =
                                                                         AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_2_tileIdxPtr,
.AnomalyDetection_heavy_6_layer2_out_tensor = AnomalyDetection_heavy_6_layer2_out_tensor,
                                                                     .AnomalyDetection_heavy_6_layer3_out_tensor = AnomalyDetection_heavy_6_layer3_out_tensor,
                                                                     .AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr =
                                                                         AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_3_tileIdxPtr,
.AnomalyDetection_heavy_6_layer3_out_tensor = AnomalyDetection_heavy_6_layer3_out_tensor,
                                                                     .AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr =
                                                                         AnomalyDetection_heavy_6_TILING_CODEGEN_L1_RequantizedGemm_4_tileIdxPtr,
};
    pi_cl_team_fork(NUM_CORES, (void *)Combined_closure_2_6, &Combined_closure_args);

}

void InitNetwork() {

  ImageClassification_small_2_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 15995);

  ImageClassification_small_2_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 24576);

  ImageClassification_small_2_input_0 = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 3072);
  ImageClassification_small_2_output_0 = (int8_t *)((char *)ImageClassification_small_2_MEMORYARENA_L2 + 8);
  ImageClassification_small_2_inputs[0] = (void *)ImageClassification_small_2_input_0;
  ImageClassification_small_2_outputs[0] = (void *)ImageClassification_small_2_output_0;

  AnomalyDetection_heavy_6_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 42176);

  AnomalyDetection_heavy_6_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 768);

  AnomalyDetection_heavy_6_input_0 = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L2 + 128);
  AnomalyDetection_heavy_6_output_0 = (int8_t *)((char *)AnomalyDetection_heavy_6_MEMORYARENA_L2 + 0);
  AnomalyDetection_heavy_6_inputs[0] = (void *)AnomalyDetection_heavy_6_input_0;
  AnomalyDetection_heavy_6_outputs[0] = (void *)AnomalyDetection_heavy_6_output_0;
}
