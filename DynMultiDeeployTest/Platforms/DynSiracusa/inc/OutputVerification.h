#include "dory_mem.h"
#include "pmsis.h"

#define MAINSTACKSIZE 8000
#define SLAVESTACKSIZE 3800

extern struct pi_device cluster_dev;

typedef struct {
  void *expected;
  void *actual;
  uint32_t num_elements;
  uint32_t output_buf_index;
  uint32_t *err_count;
} FloatCompareArgs;

typedef struct {
  const char *model_name;
  uint32_t num_outputs;
  void **outputs;
  uint32_t *outputs_bytes;
  void **testOutputVector;
  uint32_t output_elem_size; // sizeof(MODEL_X_OUTPUTTYPE), e.g. 1, 2 or 4
  int is_output_float;       // MODEL_X_ISOUTPUTFLOAT
} OutputVerificationArgs;

void CompareFloatOnCluster(void *args);
uint32_t VerifyModelOutput(OutputVerificationArgs *args, uint32_t *tot_tested);