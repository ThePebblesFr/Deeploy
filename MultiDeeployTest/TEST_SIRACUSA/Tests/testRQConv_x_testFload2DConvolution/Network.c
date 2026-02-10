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

int8_t *testRQConv_MEMORYARENA_L1;
int8_t *testRQConv_MEMORYARENA_L2;
int8_t *testRQConv_input_0;
int8_t *testRQConv_output_0;

static PI_L2 int8_t testRQConv_weight_tensor[512] = {
    -63,  41,   -81,  -108, -104, 63,   -83,  2,    -8,   -46,  -86,  -94,  -115, 48,   -109, 75,   102,  -64,  25,   92,   7,    -73,  17,   -1,   119,  -44,
    -6,   -34,  26,   -77,  -53,  -14,  8,    41,   7,    26,   -36,  72,   -81,  -51,  -67,  -42,  -125, -88,  52,   112,  0,    33,   45,   -113, 23,   89,
    49,   -3,   -57,  108,  -123, 44,   -93,  -39,  91,   67,   41,   83,   115,  16,   93,   89,   124,  -20,  45,   -72,  -56,  -103, 35,   125,  -52,  50,
    -9,   -103, -87,  39,   114,  -35,  -122, 84,   74,   -8,   20,   -33,  4,    104,  20,   14,   -126, -71,  21,   52,   -78,  -62,  -47,  -21,  -60,  94,
    121,  -121, -79,  19,   -68,  -60,  -51,  -49,  -27,  93,   -82,  38,   9,    -128, 2,    47,   119,  120,  -64,  23,   108,  109,  109,  34,   -87,  -122,
    69,   59,   95,   104,  35,   -115, -128, -20,  -100, -37,  87,   97,   63,   -37,  -8,   21,   61,   122,  3,    -8,   -54,  1,    41,   63,   -68,  121,
    96,   72,   -29,  -15,  13,   -115, -74,  -85,  123,  -59,  95,   40,   88,   -75,  94,   72,   -44,  -109, -45,  101,  110,  -95,  112,  -112, -102, 90,
    34,   -50,  18,   -102, 108,  -49,  113,  90,   -119, 75,   29,   -18,  -78,  96,   81,   -88,  39,   -18,  107,  -49,  6,    125,  7,    91,   -58,  67,
    122,  76,   101,  -74,  15,   83,   35,   -19,  -90,  -33,  32,   53,   74,   -127, 41,   -79,  83,   25,   -81,  44,   60,   -63,  46,   23,   114,  118,
    -58,  -39,  -56,  -75,  55,   11,   64,   -14,  -81,  -16,  -89,  47,   84,   -9,   -77,  81,   61,   86,   -102, -61,  53,   -88,  -72,  30,   126,  123,
    77,   -7,   33,   53,   -76,  -126, 43,   -60,  -11,  84,   -62,  -123, -43,  -68,  0,    115,  15,   -9,   59,   93,   -18,  -13,  -65,  -119, -21,  20,
    64,   23,   -121, 47,   71,   112,  -62,  36,   8,    -106, -63,  61,   103,  -23,  -109, -103, -86,  -16,  113,  -121, -45,  -1,   55,   -114, -24,  74,
    -107, 117,  -68,  48,   50,   38,   -63,  19,   -117, 14,   -82,  -128, -78,  -17,  -126, 2,    -1,   26,   -122, 4,    -125, -42,  -77,  107,  -85,  92,
    110,  -73,  -52,  90,   -53,  -50,  15,   -88,  -31,  17,   -65,  44,   3,    -116, -19,  18,   100,  -23,  -115, -112, -110, 126,  47,   12,   -6,   99,
    -37,  -5,   -88,  -51,  10,   28,   -46,  -12,  -78,  -11,  -72,  -59,  56,   -52,  -94,  -6,   -55,  -3,   17,   35,   -28,  -73,  -37,  33,   -52,  -19,
    -52,  92,   57,   84,   -41,  -3,   -70,  107,  -11,  67,   119,  80,   -112, 112,  71,   60,   66,   -125, -38,  -94,  23,   -15,  -10,  40,   -114, 39,
    84,   -25,  107,  103,  21,   24,   43,   -93,  36,   95,   53,   118,  -48,  121,  -55,  46,   126,  95,   53,   -109, -49,  117,  -125, 125,  9,    8,
    -61,  96,   42,   21,   3,    -2,   -109, -121, 109,  -7,   -121, 45,   -103, 74,   -93,  -78,  90,   -121, 40,   19,   0,    -91,  112,  -73,  -46,  -48,
    -82,  44,   63,   -87,  122,  -8,   112,  -20,  -34,  71,   54,   39,   -4,   -88,  43,   59,   -92,  -25,  106,  85,   -78,  105,  -101, -32,  111,  28,
    65,   8,    37,   -49,  -41,  94,   77,   49,   70,   102,  93,   -62,  30,   -21,  -65,  44,   -44,  -46};

static PI_L2 int32_t testRQConv_RQS1mul_tensor[4] = {71, 74, 58, 80};

static PI_L2 int32_t testRQConv_RQS1add_tensor[4] = {-997157, 852762, 832762, -677151};

static PI_L1 uint8_t testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 1};

static PI_L1 uint8_t testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles[2] = {0, 1};

void *testRQConv_inputs[1];
void *testRQConv_outputs[1];
extern struct pi_device cluster_dev;


int8_t *testFloat2DConvolution_MEMORYARENA_L1;
int8_t *testFloat2DConvolution_MEMORYARENA_L2;
float32_t *testFloat2DConvolution_input_0;
float32_t *testFloat2DConvolution_output_0;

static PI_L2 float32_t testFloat2DConvolution_weight_tensor[216] = {
    -1.0852513313293457f,  0.24468602240085602f,  1.8281809091567993f,   2.165799617767334f,    -0.7324926257133484f,  -1.216133713722229f,
    0.47602373361587524f,  0.4408690333366394f,   1.1992653608322144f,   -0.3298961818218231f,  -0.7782944440841675f,  -0.33141931891441345f,
    1.264784336090088f,    -0.5344719886779785f,  0.9204198122024536f,   1.7285990715026855f,   -0.17462031543254852f, 0.5488663911819458f,
    1.0755356550216675f,   -0.1385715752840042f,  -0.8251299858093262f,  0.5807065963745117f,   -0.7114549875259399f,  -1.0680992603302002f,
    -0.8022473454475403f,  -3.332423448562622f,   -0.8776105046272278f,  0.42320889234542847f,  -0.8942937254905701f,  -0.6278658509254456f,
    -0.18125490844249725f, 0.6646248698234558f,   -1.0466642379760742f,  -1.2581813335418701f,  0.4515478014945984f,   0.12087330967187881f,
    -0.18066009879112244f, 0.3569435477256775f,   1.2747101783752441f,   0.8254817724227905f,   0.06241195276379585f,  1.072937250137329f,
    -0.9285781979560852f,  -0.1827138364315033f,  -1.2845587730407715f,  -0.02092898264527321f, -0.38856685161590576f, 1.123566746711731f,
    -1.0814458131790161f,  0.13279174268245697f,  -0.6836380362510681f,  -0.2037459760904312f,  -0.6988926529884338f,  0.29769566655158997f,
    -0.6558053493499756f,  1.1346372365951538f,   -0.397361695766449f,   0.09144381433725357f,  -0.10261519998311996f, -0.4375804662704468f,
    0.6110832095146179f,   -0.9499455690383911f,  2.2455999851226807f,   -0.17472803592681885f, 0.5258661508560181f,   0.689538836479187f,
    -1.1488627195358276f,  -1.246726393699646f,   1.3300529718399048f,   0.8233240246772766f,   0.3619127869606018f,   1.152435064315796f,
    0.23302890360355377f,  -0.5600001811981201f,  -0.3537014126777649f,  0.26630252599716187f,  1.2891511917114258f,   1.4224637746810913f,
    -1.5122689008712769f,  0.905912458896637f,    0.1882474571466446f,   0.4949946999549866f,   0.20684567093849182f,  -0.9390848278999329f,
    0.9778059720993042f,   -1.07807457447052f,    0.24307450652122498f,  0.1851540058851242f,   -1.5438510179519653f,  1.4623011350631714f,
    -1.726426124572754f,   0.20763295888900757f,  1.4272626638412476f,   1.8927271366119385f,   0.7872235178947449f,   1.1649609804153442f,
    1.303973913192749f,    -0.9531353712081909f,  0.8871524333953857f,   -0.42157018184661865f, -1.6961714029312134f,  -0.10038387775421143f,
    -0.7835149765014648f,  0.985711932182312f,    0.680275559425354f,    0.10049467533826828f,  -0.6330665349960327f,  -1.6111278533935547f,
    -0.36971327662467957f, 1.7027642726898193f,   0.48674291372299194f,  -1.7477635145187378f,  -0.9736038446426392f,  -1.0236759185791016f,
    -0.9604427218437195f,  1.6260309219360352f,   -0.2646729350090027f,  -0.06998144090175629f, -0.5662126541137695f,  0.06153548136353493f,
    -1.15035879611969f,    0.14886286854743958f,  -0.7257568836212158f,  0.7011738419532776f,   0.9810847043991089f,   -1.3560447692871094f,
    2.199481964111328f,    -0.941659152507782f,   1.3886810541152954f,   -1.3091299533843994f,  -0.27297496795654297f, 0.6450515389442444f,
    -0.6523182392120361f,  -2.32395601272583f,    0.624085009098053f,    -2.6606662273406982f,  0.32558679580688477f,  -1.6914440393447876f,
    0.07682589441537857f,  -0.48351573944091797f, 0.21477602422237396f,  2.181380033493042f,    -0.21440501511096954f, -0.02702215686440468f,
    0.4320322871208191f,   0.585159420967102f,    1.6504380702972412f,   0.2567078471183777f,   -1.333866834640503f,   -1.1506596803665161f,
    -0.37438201904296875f, 0.5734798312187195f,   0.9724025726318359f,   0.9112952947616577f,   0.1148839071393013f,   -0.2217870056629181f,
    1.2046163082122803f,   -0.8799312114715576f,  -1.7861531972885132f,  -1.5293465852737427f,  -0.22991129755973816f, 0.7209985852241516f,
    1.1877285242080688f,   0.9728639125823975f,   0.47664663195610046f,  0.9216881990432739f,   0.942643940448761f,    -0.10735060274600983f,
    1.1415163278579712f,   0.7017327547073364f,   -0.22250185906887054f, 0.4620393216609955f,   -0.3321303129196167f,  0.956382155418396f,
    0.02270658314228058f,  -0.43670356273651123f, -0.42516589164733887f, -2.969311237335205f,   -1.1430106163024902f,  0.0037341772112995386f,
    0.28394678235054016f,  1.9582486152648926f,   1.381427526473999f,    -0.8693037629127502f,  0.7909730076789856f,   -0.3729180097579956f,
    -0.18382248282432556f, 0.675275981426239f,    0.9512938857078552f,   -1.2002058029174805f,  0.5305636525154114f,   -1.4628145694732666f,
    0.853242814540863f,    1.3799662590026855f,   0.4879334568977356f,   -0.6486525535583496f,  -0.6195771098136902f,  0.48352164030075073f,
    0.2570694088935852f,   0.7058799266815186f,   -1.3206454515457153f,  0.3897208273410797f,   -1.07869291305542f,    0.5385149717330933f,
    0.713161051273346f,    -0.07883252948522568f, -0.2887186110019684f,  0.46043094992637634f,  -1.8079217672348022f,  -0.796956479549408f,
    -0.19777126610279083f, -0.0566643662750721f,  -0.2915743887424469f,  1.2845852375030518f,   -0.9116997718811035f,  -0.7419087290763855f};

static PI_L1 uint8_t testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_numTiles[2] = {0, 1};

static PI_L1 uint8_t testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_numTiles[2] = {0, 1};

void *testFloat2DConvolution_inputs[1];
void *testFloat2DConvolution_outputs[1];

int cores_map[2][2] = {{0, 3}, {4, 7}};
int nb_dedicated_cores[2] = {4, 4};

// ===== FUSED tiling_closure – Scheduling Round 0 =====
typedef struct {
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  int core_id = pi_core_id();
  if (cores_map[0][0] <= core_id && core_id < cores_map[0][1]) {

  // Transpose [1, 2, 64, 32] -> [1, 64, 32, 2] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 2;

  uint16_t dimLen_2 = 64;

  uint16_t dimLen_3 = 32;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / 4;
    const uint32_t leftover = dimLen_2 - baseChunk * 4;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          ((int8_t (*)[dimLen_2][dimLen_3][dimLen_1])testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref)[i_0][i_2][i_3][i_1] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  }
  if (cores_map[1][0] <= core_id && core_id < cores_map[1][1]) {

  // Transpose [1, 3, 16, 16] -> [1, 16, 16, 3] (Name: conv_node_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 3;

  uint16_t dimLen_2 = 16;

  uint16_t dimLen_3 = 16;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / 4;
    const uint32_t leftover = dimLen_2 - baseChunk * 4;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          ((float32_t(*)[dimLen_2][dimLen_3][dimLen_1])testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref)[i_0][i_2][i_3][i_1] =
              ((float32_t(*)[dimLen_1][dimLen_2][dimLen_3])
                   testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  }
}

// ===== FUSED cluster_fork - Scheduling Round 0 =====
typedef struct {
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork(void *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t){
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
.testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                             testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                         .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                             testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref,
.testFloat2DConvolution_conv_node_input_0_transposed =
                                                                              testFloat2DConvolution_conv_node_input_0_transposed,
                                                                          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
                                                                              testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr,
};

  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

}

// ===== FUSED closure – Scheduling Round 0 =====
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 0);
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 4096);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
      (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 0);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
      (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 3072);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref = (void *)((char *)testRQConv_input_0 + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);

  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_input_0_ref = (void *)((char *)testFloat2DConvolution_input_0 + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_conv_node_input_0_transposed_ref =
      (void *)((char *)testFloat2DConvolution_conv_node_input_0_transposed + 0);
  const static char _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_input_0_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();
    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445888, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1444864, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    
    testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args =
        (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t){
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
.testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                             testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                         .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                             testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr,
                .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
                    testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref,
                .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
                    testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref,
.testFloat2DConvolution_conv_node_input_0_transposed =
                                                                              testFloat2DConvolution_conv_node_input_0_transposed,
                                                                          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
                                                                              testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr};    pi_cl_team_fork(NUM_CORES, (void *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork,


                    &testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();
    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314816, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1313792, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_conv_node_input_0_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();
}
  *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

    *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr > 0)
                              ? testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[(
                                    *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                         [*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
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
}

// ===== FUSED closure_L3 - Scheduling Round 0 =====
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(void *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t){
.testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                             testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                         .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                             testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr,
.testFloat2DConvolution_conv_node_input_0_transposed =
                                                                              testFloat2DConvolution_conv_node_input_0_transposed,
                                                                          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
                                                                              testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr,
};

  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args);

}

// ===== FUSED tiling_closure – Scheduling Round 1 =====
typedef struct {
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  void *testFloat2DConvolution_conv_node_buffer;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args;
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer = args->testRQConv__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  void *testFloat2DConvolution_conv_node_buffer = args->testFloat2DConvolution_conv_node_buffer;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed = args->testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_conv_node_output_0_pre_transposed_ref =
      (void *)((char *)testFloat2DConvolution_conv_node_output_0_pre_transposed + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_output_0_ref = (void *)((char *)testFloat2DConvolution_output_0 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  int core_id = pi_core_id();
  if (cores_map[0][0] <= core_id && core_id < cores_map[0][1]) {

  // PULP NN CONV

  pulp_nn_conv_i8_i8_i8(testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, testRQConv__MERGE_CONVRQ_PASS_0_buffer, NULL,
                        testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                        testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, 1, 16, 32, 64, 2,
                        27, 59, 4, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 4);

  }
  if (cores_map[1][0] <= core_id && core_id < cores_map[1][1]) {

  // 2D FP Conv HWC with Im2Col and ChannelOout parallelism (Name: conv_node, Op: Conv)

  float32_t *ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref =
      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref =
      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;

  for (uint32_t n = 0; n < 1; ++n) {
    PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
        ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref, 16, 16, 3,
        testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref, 8, 3, 3, 2, 2, NULL, false,
        ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref, 1, 1, 1, 1,
        testFloat2DConvolution_conv_node_buffer, 4);

    ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref += 3 * 16 * 16;
    ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref += 8 * 8 * 8;
  }

    if (core_id == cores_map[1][0]) {
    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_output_0_ref);

  *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

      // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_conv_node_output_0_pre_transposed_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }


        }

  // Transpose [1, 8, 8, 8] -> [1, 8, 8, 8] (Name: conv_node_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / 4;
    const uint32_t leftover = dimLen_3 - baseChunk * 4;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((float32_t(*)[dimLen_3][dimLen_1][dimLen_2])
               testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref)[i_0][i_3][i_1][i_2] =
              ((float32_t(*)[dimLen_1][dimLen_2][dimLen_3])
                   testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  }
}

// ===== FUSED cluster_fork - Scheduling Round 1 =====
typedef struct {
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  void *testFloat2DConvolution_conv_node_buffer;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork(void *testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args;
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer = args->testRQConv__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  void *testFloat2DConvolution_conv_node_buffer = args->testFloat2DConvolution_conv_node_buffer;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed = args->testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
  testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
          .testRQConv__MERGE_CONVRQ_PASS_0_buffer = testRQConv__MERGE_CONVRQ_PASS_0_buffer,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
      .testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
      .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
      .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr,
          .testFloat2DConvolution_conv_node_buffer = testFloat2DConvolution_conv_node_buffer,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref,
          .testFloat2DConvolution_conv_node_input_0_transposed = testFloat2DConvolution_conv_node_input_0_transposed,
          .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref,
              .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr,
};

  testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

}

// ===== FUSED closure – Scheduling Round 1 =====
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_closure_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t *args = (testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_closure_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed = args->testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 6372);
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 12516);
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)testRQConv_MEMORYARENA_L1 + 13028);
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)testRQConv_MEMORYARENA_L1 + 13044);
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 0);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 0);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 5120);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 3072);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
      (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 2048);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
      (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_tensor_ref = (void *)((char *)testRQConv_weight_tensor + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_RQS1add_tensor_ref = (void *)((char *)testRQConv_RQS1add_tensor + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_RQS1mul_tensor_ref = (void *)((char *)testRQConv_RQS1mul_tensor + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed + 0);
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer = (void *)((char *)testRQConv_MEMORYARENA_L1 + 10468);

  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_conv_node_input_0_transposed_ref =
      (void *)((char *)testFloat2DConvolution_conv_node_input_0_transposed + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_tensor_ref = (void *)((char *)testFloat2DConvolution_weight_tensor + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_conv_node_output_0_pre_transposed_ref =
      (void *)((char *)testFloat2DConvolution_conv_node_output_0_pre_transposed + 0);
  void *testFloat2DConvolution_conv_node_buffer = (void *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 5984);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_conv_node_output_0_pre_transposed_ref =
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_output_0_ref = (void *)((char *)testFloat2DConvolution_output_0 + 0);
  const static char _MERGE_CONVRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_L2][SB][1636011 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1]; TILING_I++) {

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();
    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445888, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);
    mchan_transfer_1d(1442304, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_tensor_ref);
    mchan_transfer_1d(1441808, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_RQS1add_tensor_ref);
    mchan_transfer_1d(1441808, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_RQS1mul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1444864, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_conv_node_input_0_transposed_ref);
    mchan_transfer_1d(1442656, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    
    testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args =
        (testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t){
            .testRQConv__MERGE_CONVRQ_PASS_0_buffer = testRQConv__MERGE_CONVRQ_PASS_0_buffer,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
      .testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
      .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
      .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr,
            .testFloat2DConvolution_conv_node_buffer = testFloat2DConvolution_conv_node_buffer,
            .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref,
            .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref,
            .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref,
          .testFloat2DConvolution_conv_node_input_0_transposed = testFloat2DConvolution_conv_node_input_0_transposed,
          .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr,
                .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
                    testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref,
                .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
                    testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref,
              .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr};    pi_cl_team_fork(NUM_CORES, (void *)testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork, &testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args);



    _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();
    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317092, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();
}
  *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

    *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr += 1;

  // Deinitialize DMA futures

    *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr > 0)
                              ? testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]; PROFILING_I++) {

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
}

// ===== FUSED closure_L3 - Scheduling Round 1 =====
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_closure_L3(void *testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed = args->testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
  testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_closure_args = (testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t){
      .testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
      .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
      .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr,
          .testFloat2DConvolution_conv_node_input_0_transposed = testFloat2DConvolution_conv_node_input_0_transposed,
          .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr,
              .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr,
};

  testRQConv__MERGE_CONVRQ_PASS_0_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_closure_args);

}

// ===== FUSED tiling_closure – Scheduling Round 2 =====
typedef struct {
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t;
static void
testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t *)
          testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  int core_id = pi_core_id();
  if (cores_map[0][0] <= core_id && core_id < cores_map[0][1]) {

  // Transpose [1, 59, 27, 4] -> [1, 4, 59, 27] (Name: _MERGE_CONVRQ_PASS_0_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 59;

  uint16_t dimLen_2 = 27;

  uint16_t dimLen_3 = 4;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

      const uint32_t baseChunk = dimLen_1 / 4;
      const uint32_t leftover = dimLen_1 - baseChunk * 4;
      const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
      const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
      for (uint32_t i_1 = offset; i_1 < offset + chunk; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((int8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref)[i_0][i_3][i_1][i_2] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                   testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  }
}

// ===== FUSED cluster_fork - Scheduling Round 2 =====
typedef struct {
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t;
static void
testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork(void *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t
      testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args =
          (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t){
              .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
                  testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
              .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
                  testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref,
          .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr,
};

  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

}

// ===== FUSED closure – Scheduling Round 2 =====
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;

  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 6372);
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref = (void *)((char *)testRQConv_output_0 + 0);

  const static char _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
           [*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr];
       TILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
                      [(*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();
    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448164, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t
        testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args =
            (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t){
                .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
                    testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
                .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
                    testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref,
          .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr};    pi_cl_team_fork(NUM_CORES, (void *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork,


                    &testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();
    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317092, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();
}
  *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr > 0)
                              ? testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles[(
                                    *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
                         [*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_suffix);
  }
  StartTimer();
}

// ===== FUSED closure_L3 - Scheduling Round 2 =====
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t;
static void testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3(void *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args) {
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;

  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t){
          .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr,
};

  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args);

}

void RunNetwork() {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer;

  uint8_t bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      &bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  uint8_t bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr = 0;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      &bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  void *testFloat2DConvolution_conv_node_buffer;

  uint8_t bu_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr = 0;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
      &bu_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
  uint8_t bu_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = 0;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = &bu_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
  uint8_t bu_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr = 0;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
      &bu_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;


  // ===== Scheduling Round 0 =====
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)testRQConv_MEMORYARENA_L2 + 6372);
  testFloat2DConvolution_conv_node_input_0_transposed = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L2 + 0);
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t){.testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                                testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                                testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr
,              .testFloat2DConvolution_conv_node_input_0_transposed = testFloat2DConvolution_conv_node_input_0_transposed,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr
};

  // testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3 CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args);

  // ===== Scheduling Round 1 =====
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = (int8_t *)((char *)testRQConv_MEMORYARENA_L2 + 0);
  testFloat2DConvolution_conv_node_output_0_pre_transposed = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L2 + 3072);
  testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
          .testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
          .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr
,          .testFloat2DConvolution_conv_node_input_0_transposed = testFloat2DConvolution_conv_node_input_0_transposed,
          .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr
,              .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr
};

  // testRQConv__MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_closure_L3(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args);

  // ===== Scheduling Round 2 =====
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t){
          .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr
};

  // testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3 CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args);

}

void InitNetwork() {

  testRQConv_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 13060);

  testRQConv_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 12744);

  testRQConv_input_0 = (int8_t *)((char *)testRQConv_MEMORYARENA_L2 + 0);
  testRQConv_output_0 = (int8_t *)((char *)testRQConv_MEMORYARENA_L2 + 6372);
  testRQConv_inputs[0] = (void *)testRQConv_input_0;
  testRQConv_outputs[0] = (void *)testRQConv_output_0;

  testFloat2DConvolution_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 6416);

  testFloat2DConvolution_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 6144);

  testFloat2DConvolution_input_0 = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L2 + 3072);
  testFloat2DConvolution_output_0 = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L2 + 0);
  testFloat2DConvolution_inputs[0] = (void *)testFloat2DConvolution_input_0;
  testFloat2DConvolution_outputs[0] = (void *)testFloat2DConvolution_output_0;
}
