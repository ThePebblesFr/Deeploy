# DynMultiDeeployTest README.md

## 1. Create a test for Model_1 and Model_2

- Create a test folder and yank the original models files:
```bash
cd /app/Deeploy/DynMultiDeeployTest/
mkdir -p Tests/Model_1_x_Model_2/
cp -r ../DeeployTest/Tests/Model_1 Tests/Model_1_x_Model_2/
cp -r ../DeeployTest/Tests/Model_2 Tests/Model_1_x_Model_2/
```

- Launch dynamic 2 models code generation:
```bash
python3 test_dynamic_2_models.py -t Tests/Model_1_x_Model_2 --models Model_1 Model_2
```

## 2. Dynamic code generation breakdown

### 2.1. MultiDeeployTests generation
For each configuration needed throughout the scenarios, a MultiDeeployTest is generated. Here's the tree of the generated code:
```bash
cd /app/Deeploy/DynMultiDeeployTest/
tree -L 3 TEST_SIRACUSA/Tests
```
Here's the output:
```bash
TEST_SIRACUSA/Tests/Model_1_x_Model_2/
	Model_1_2_x_Model_2_6/
		Model_1_2/
			Network.c 			# DeeployTest for Model_1 on 2 cores
			Network.h
			testinputs.h
			testoutputs.h
		Model_2_6/
			Network.c			# DeeployTest for Model_2 on 6 cores
			Network.h
			testinputs.h
			testoutputs.h
		Network.c			# MultiDeeployTest for (Model_1, Model_2) on (2, 6) cores
		Network.h
		testinputs.h
		testoutputs.h
	Model_1_4_x_Model_2_4/
		Model_1_4/
			Network.c			# DeeployTest for Model_1 on 4 cores
			Network.h
			testinputs.h
			testoutputs.h
		Model_2_4/
			Network.c			# DeeployTest for Model_2 on 4 cores
			Network.h
			testinputs.h
			testoutputs.h
		Network.c			# MultiDeeployTest for (Model_1, Model_2) on (4, 4) cores
		Network.h
		testinputs.h
		testoutputs.h
	Model_1_6_x_Model_2_2/
		Model_1_6/
			Network.c			# DeeployTest for Model_1 on 6 cores
			Network.h
			testinputs.h
			testoutputs.h
		Model_2_2/
			Network.c			# DeeployTest for Model_2 on 2 cores
			Network.h
			testinputs.h
			testoutputs.h
		Network.c			# MultiDeeployTest for (Model_1, Model_2) on (6, 2) cores
		Network.h
		testinputs.h
		testoutputs.h
```

### 2.2. Code fusion

First, all 3 `Network.h` are fused, everything is kept. Note that we remove `RunNetwork()` and `InitNetwork()` prototype and add:
```c
void RunNetworks_2_6();
void RunNetworks_4_4();
void RunNetworks_6_2();
void InitNetworks_2_6();
void InitNetworks_4_4();
void InitNetworks_6_2();
```
Then, we use one `testinputs.h` and `testoutputs.h`.

Finally, let's breakdown `Network.c` fusion:

- includes are fused without duplicates
- weights are kept only once
- global variables are all kept
- scheduling round functions are all kept
- `RunNetwork()` and `InitNetwork()` functions must be renamed according to the hardware configuration. For that matter, we use the names defined by the prototypes in `Network.h`.

### 2.3. Weights variable renaming

Because we kept only once the weights (for L2 size purposes), we have to rename all utilization of these weight. Indeed, the way we generated the base code for MultiDeeploy execution, all weights for Model_1 (resp. Model_2) are suffixed with Model_1_2, Model_1_4 or Model_1_6 (resp. Model_2_2, Model_2_4 or Model_2_6). We must scan the whole fused `Network.c` and strip  the suffix from any occurence of a weight variable in the code.

### 2.4. Main

(static input/output, just PoC here)
We have to create a switch case statement for all possible configurations:
```c
int main() {
	// Initializations

	// Network inference
	switch(operation_mode) {
		case LOW_ENERGY:
			// send InitNetworks_4_4() cluster task
			// send RunNetworks_4_4() cluster task
			break;
		case CRITICAL_1:
			// send InitNetworks_6_2() cluster task
			// send RunNetworks_6_2() cluster task
			break;
		case CRITICAL_2:
			// send InitNetworks_2_6() cluster task
			// send RunNetworks_2_6() cluster task
			break;
		default:
			break;
	}

	// Outputs verifications

	// operation_mode state update
}
```