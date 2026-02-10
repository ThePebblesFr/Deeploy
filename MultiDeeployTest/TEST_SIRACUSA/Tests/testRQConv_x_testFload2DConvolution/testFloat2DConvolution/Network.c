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
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref;
} testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure(void *testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args_t *args =
      (testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args_t *)testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 3, 16, 16] -> [1, 16, 16, 3] (Name: conv_node_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 3;

  uint16_t dimLen_2 = 16;

  uint16_t dimLen_3 = 16;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
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

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref;
} testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork(void *testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args_t *args =
      (testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args_t *)testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args_t
      testFloat2DConvolution_testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args =
          (testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args_t){
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref};

  // testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure CLOSURE CALL
  testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure(
      &testFloat2DConvolution_testFloat2DConvolution_conv_node_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
} testFloat2DConvolution_conv_node_input_0_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_input_0_transpose_closure(void *testFloat2DConvolution_conv_node_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_input_0_transpose_closure_args_t *args =
      (testFloat2DConvolution_conv_node_input_0_transpose_closure_args_t *)testFloat2DConvolution_conv_node_input_0_transpose_closure_args;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
      (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 0);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
      (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 3072);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_input_0_ref = (void *)((char *)testFloat2DConvolution_input_0 + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_conv_node_input_0_transposed_ref =
      (void *)((char *)testFloat2DConvolution_conv_node_input_0_transposed + 0);

  const static char conv_node_input_0_transpose_L2_suffix[] = " cycles \n";

  const static char conv_node_input_0_transpose_L2_prefix[] = "[conv_node_input_0_transpose_L2][SB][0 ops][Tile ";

  uint32_t conv_node_input_0_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t conv_node_input_0_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t conv_node_input_0_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t conv_node_input_0_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t conv_node_input_0_transpose_L2_kernel_end_measurements[1];

  uint32_t conv_node_input_0_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_numTiles
           [*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr];
       TILING_I < testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_numTiles
                      [(*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    conv_node_input_0_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1444864, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    conv_node_input_0_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    conv_node_input_0_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args_t
        testFloat2DConvolution_testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args =
            (testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args_t){
                .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref =
                    testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_in_ref,
                .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref =
                    testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork,
                    &testFloat2DConvolution_testFloat2DConvolution_conv_node_input_0_transpose_cluster_fork_args);

    conv_node_input_0_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    conv_node_input_0_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1313792, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_data_out_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_conv_node_input_0_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    conv_node_input_0_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr > 0)
                              ? testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_numTiles[(
                                    *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_numTiles
                         [*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", conv_node_input_0_transpose_L2_prefix, PROFILING_I, "Input DMA took ",
           conv_node_input_0_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               conv_node_input_0_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           conv_node_input_0_transpose_L2_suffix);

    printf("%s%u] %s%u%s", conv_node_input_0_transpose_L2_prefix, PROFILING_I, "Kernel took ",
           conv_node_input_0_transpose_L2_kernel_end_measurements[PROFILING_I] - conv_node_input_0_transpose_L2_kernel_start_measurements[PROFILING_I],
           conv_node_input_0_transpose_L2_suffix);

    printf("%s%u] %s%u%s", conv_node_input_0_transpose_L2_prefix, PROFILING_I, "Output DMA took ",
           conv_node_input_0_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               conv_node_input_0_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I],
           conv_node_input_0_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;
} testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_input_0_transpose_closure_L3(void *testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args_t *args =
      (testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args_t *)testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  testFloat2DConvolution_conv_node_input_0_transpose_closure_args_t testFloat2DConvolution_testFloat2DConvolution_conv_node_input_0_transpose_closure_args =
      (testFloat2DConvolution_conv_node_input_0_transpose_closure_args_t){.testFloat2DConvolution_conv_node_input_0_transposed =
                                                                              testFloat2DConvolution_conv_node_input_0_transposed,
                                                                          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
                                                                              testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr};

  // testFloat2DConvolution_conv_node_input_0_transpose_closure CLOSURE CALL
  testFloat2DConvolution_conv_node_input_0_transpose_closure(&testFloat2DConvolution_testFloat2DConvolution_conv_node_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *testFloat2DConvolution_conv_node_buffer;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;
} testFloat2DConvolution_conv_node_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_tiling_closure(void *testFloat2DConvolution_conv_node_tiling_closure_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_tiling_closure_args_t *args =
      (testFloat2DConvolution_conv_node_tiling_closure_args_t *)testFloat2DConvolution_conv_node_tiling_closure_args;
  void *testFloat2DConvolution_conv_node_buffer = args->testFloat2DConvolution_conv_node_buffer;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;

  // CLOSURE FUNCTION CALL

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
        testFloat2DConvolution_conv_node_buffer, NUM_CORES);

    ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref += 3 * 16 * 16;
    ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref_testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref += 8 * 8 * 8;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *testFloat2DConvolution_conv_node_buffer;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;
} testFloat2DConvolution_conv_node_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_cluster_fork(void *testFloat2DConvolution_conv_node_cluster_fork_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_cluster_fork_args_t *args =
      (testFloat2DConvolution_conv_node_cluster_fork_args_t *)testFloat2DConvolution_conv_node_cluster_fork_args;
  void *testFloat2DConvolution_conv_node_buffer = args->testFloat2DConvolution_conv_node_buffer;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref;

  // CLOSURE FUNCTION CALL
  testFloat2DConvolution_conv_node_tiling_closure_args_t testFloat2DConvolution_testFloat2DConvolution_conv_node_tiling_closure_args =
      (testFloat2DConvolution_conv_node_tiling_closure_args_t){
          .testFloat2DConvolution_conv_node_buffer = testFloat2DConvolution_conv_node_buffer,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref};

  // testFloat2DConvolution_conv_node_tiling_closure CLOSURE CALL
  testFloat2DConvolution_conv_node_tiling_closure(&testFloat2DConvolution_testFloat2DConvolution_conv_node_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
} testFloat2DConvolution_conv_node_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_closure(void *testFloat2DConvolution_conv_node_closure_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_closure_args_t *args = (testFloat2DConvolution_conv_node_closure_args_t *)testFloat2DConvolution_conv_node_closure_args;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed = args->testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 0);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 5120);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 3072);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_conv_node_input_0_transposed_ref =
      (void *)((char *)testFloat2DConvolution_conv_node_input_0_transposed + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_tensor_ref = (void *)((char *)testFloat2DConvolution_weight_tensor + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_conv_node_output_0_pre_transposed_ref =
      (void *)((char *)testFloat2DConvolution_conv_node_output_0_pre_transposed + 0);
  void *testFloat2DConvolution_conv_node_buffer = (void *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 5984);

  const static char conv_node_L2_suffix[] = " cycles \n";

  const static char conv_node_L2_prefix[] = "[conv_node_L2][SB][27648 ops][Tile ";

  uint32_t conv_node_L2_egress_dma_wait_end_measurements[1];

  uint32_t conv_node_L2_egress_dma_wait_start_measurements[1];

  uint32_t conv_node_L2_ingress_dma_wait_end_measurements[1];

  uint32_t conv_node_L2_ingress_dma_wait_start_measurements[1];

  uint32_t conv_node_L2_kernel_end_measurements[1];

  uint32_t conv_node_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_numTiles[*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr];
       TILING_I < testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_numTiles[(*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr) + 1];
       TILING_I++) {

    conv_node_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

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

    conv_node_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    conv_node_L2_kernel_start_measurements[TILING_I] = getCycles();

    testFloat2DConvolution_conv_node_cluster_fork_args_t testFloat2DConvolution_testFloat2DConvolution_conv_node_cluster_fork_args =
        (testFloat2DConvolution_conv_node_cluster_fork_args_t){
            .testFloat2DConvolution_conv_node_buffer = testFloat2DConvolution_conv_node_buffer,
            .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_in_ref,
            .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_weight_ref,
            .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)testFloat2DConvolution_conv_node_cluster_fork,
                    &testFloat2DConvolution_testFloat2DConvolution_conv_node_cluster_fork_args);

    conv_node_L2_kernel_end_measurements[TILING_I] = getCycles();

    conv_node_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_data_out_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_conv_node_output_0_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    conv_node_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr > 0)
                ? testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_numTiles[(*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_numTiles[*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", conv_node_L2_prefix, PROFILING_I, "Input DMA took ",
           conv_node_L2_ingress_dma_wait_end_measurements[PROFILING_I] - conv_node_L2_ingress_dma_wait_start_measurements[PROFILING_I], conv_node_L2_suffix);

    printf("%s%u] %s%u%s", conv_node_L2_prefix, PROFILING_I, "Kernel took ",
           conv_node_L2_kernel_end_measurements[PROFILING_I] - conv_node_L2_kernel_start_measurements[PROFILING_I], conv_node_L2_suffix);

    printf("%s%u] %s%u%s", conv_node_L2_prefix, PROFILING_I, "Output DMA took ",
           conv_node_L2_egress_dma_wait_end_measurements[PROFILING_I] - conv_node_L2_egress_dma_wait_start_measurements[PROFILING_I], conv_node_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;
} testFloat2DConvolution_conv_node_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_closure_L3(void *testFloat2DConvolution_conv_node_closure_L3_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_closure_L3_args_t *args =
      (testFloat2DConvolution_conv_node_closure_L3_args_t *)testFloat2DConvolution_conv_node_closure_L3_args;
  float32_t *testFloat2DConvolution_conv_node_input_0_transposed = args->testFloat2DConvolution_conv_node_input_0_transposed;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed = args->testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  testFloat2DConvolution_conv_node_closure_args_t testFloat2DConvolution_testFloat2DConvolution_conv_node_closure_args =
      (testFloat2DConvolution_conv_node_closure_args_t){
          .testFloat2DConvolution_conv_node_input_0_transposed = testFloat2DConvolution_conv_node_input_0_transposed,
          .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr};

  // testFloat2DConvolution_conv_node_closure CLOSURE CALL
  testFloat2DConvolution_conv_node_closure(&testFloat2DConvolution_testFloat2DConvolution_conv_node_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref;
} testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void
testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure(void *testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args_t *args =
      (testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args_t *)
          testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 8, 8] -> [1, 8, 8, 8] (Name: conv_node_output_0_pre_transpose, Op: Transpose)

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

          ((float32_t(*)[dimLen_3][dimLen_1][dimLen_2])
               testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref)[i_0][i_3][i_1][i_2] =
              ((float32_t(*)[dimLen_1][dimLen_2][dimLen_3])
                   testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref;
} testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void
testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork(void *testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args_t *args =
      (testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args_t *)testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref;
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args_t
      testFloat2DConvolution_testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args =
          (testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args_t){
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref};

  // testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure CLOSURE CALL
  testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure(
      &testFloat2DConvolution_testFloat2DConvolution_conv_node_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
} testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_output_0_pre_transpose_closure(void *testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args_t *args =
      (testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args_t *)testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed = args->testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
      (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 2048);
  float32_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
      (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L1 + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_conv_node_output_0_pre_transposed_ref =
      (void *)((char *)testFloat2DConvolution_conv_node_output_0_pre_transposed + 0);
  void *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_output_0_ref = (void *)((char *)testFloat2DConvolution_output_0 + 0);

  const static char conv_node_output_0_pre_transpose_L2_suffix[] = " cycles \n";

  const static char conv_node_output_0_pre_transpose_L2_prefix[] = "[conv_node_output_0_pre_transpose_L2][SB][0 ops][Tile ";

  uint32_t conv_node_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t conv_node_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t conv_node_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t conv_node_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t conv_node_output_0_pre_transpose_L2_kernel_end_measurements[1];

  uint32_t conv_node_output_0_pre_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_numTiles
           [*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr];
       TILING_I < testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_numTiles
                      [(*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    conv_node_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_conv_node_output_0_pre_transposed_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    conv_node_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    conv_node_output_0_pre_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args_t
        testFloat2DConvolution_testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args =
            (testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args_t){
                .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref =
                    testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_in_ref,
                .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref =
                    testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork,
                    &testFloat2DConvolution_testFloat2DConvolution_conv_node_output_0_pre_transpose_cluster_fork_args);

    conv_node_output_0_pre_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    conv_node_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_data_out_ref,
                      testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    conv_node_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr > 0)
                              ? testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_numTiles[(
                                    *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_numTiles
                         [*testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", conv_node_output_0_pre_transpose_L2_prefix, PROFILING_I, "Input DMA took ",
           conv_node_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               conv_node_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           conv_node_output_0_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", conv_node_output_0_pre_transpose_L2_prefix, PROFILING_I, "Kernel took ",
           conv_node_output_0_pre_transpose_L2_kernel_end_measurements[PROFILING_I] -
               conv_node_output_0_pre_transpose_L2_kernel_start_measurements[PROFILING_I],
           conv_node_output_0_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", conv_node_output_0_pre_transpose_L2_prefix, PROFILING_I, "Output DMA took ",
           conv_node_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               conv_node_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I],
           conv_node_output_0_pre_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;
} testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3(void *testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args_t *args =
      (testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args_t *)testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args;
  float32_t *testFloat2DConvolution_conv_node_output_0_pre_transposed = args->testFloat2DConvolution_conv_node_output_0_pre_transposed;
  uint8_t *testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
      args->testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args_t
      testFloat2DConvolution_testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args =
          (testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args_t){
              .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr};

  // testFloat2DConvolution_conv_node_output_0_pre_transpose_closure CLOSURE CALL
  testFloat2DConvolution_conv_node_output_0_pre_transpose_closure(&testFloat2DConvolution_testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {
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
  testFloat2DConvolution_conv_node_input_0_transposed = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L2 + 0);
  testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args_t
      testFloat2DConvolution_testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args =
          (testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args_t){
              .testFloat2DConvolution_conv_node_input_0_transposed = testFloat2DConvolution_conv_node_input_0_transposed,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_input_0_transpose_tileIdxPtr};

  // testFloat2DConvolution_conv_node_input_0_transpose_closure_L3 CLOSURE CALL
  testFloat2DConvolution_conv_node_input_0_transpose_closure_L3(&testFloat2DConvolution_testFloat2DConvolution_conv_node_input_0_transpose_closure_L3_args);

  testFloat2DConvolution_conv_node_output_0_pre_transposed = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L2 + 3072);
  testFloat2DConvolution_conv_node_closure_L3_args_t testFloat2DConvolution_testFloat2DConvolution_conv_node_closure_L3_args =
      (testFloat2DConvolution_conv_node_closure_L3_args_t){
          .testFloat2DConvolution_conv_node_input_0_transposed = testFloat2DConvolution_conv_node_input_0_transposed,
          .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
          .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr = testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_tileIdxPtr};

  // testFloat2DConvolution_conv_node_closure_L3 CLOSURE CALL
  testFloat2DConvolution_conv_node_closure_L3(&testFloat2DConvolution_testFloat2DConvolution_conv_node_closure_L3_args);

  testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args_t
      testFloat2DConvolution_testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args =
          (testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args_t){
              .testFloat2DConvolution_conv_node_output_0_pre_transposed = testFloat2DConvolution_conv_node_output_0_pre_transposed,
              .testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr =
                  testFloat2DConvolution_TILING_CODEGEN_L1_conv_node_output_0_pre_transpose_tileIdxPtr};

  // testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3 CLOSURE CALL
  testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3(
      &testFloat2DConvolution_testFloat2DConvolution_conv_node_output_0_pre_transpose_closure_L3_args);
}

void InitNetwork() {

  testFloat2DConvolution_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 6416);

  testFloat2DConvolution_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 6144);

  testFloat2DConvolution_input_0 = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L2 + 3072);
  testFloat2DConvolution_output_0 = (float32_t *)((char *)testFloat2DConvolution_MEMORYARENA_L2 + 0);
  testFloat2DConvolution_inputs[0] = (void *)testFloat2DConvolution_input_0;
  testFloat2DConvolution_outputs[0] = (void *)testFloat2DConvolution_output_0;
}
