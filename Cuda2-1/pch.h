#pragma once

#include "Common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h> 
#include "cuda.h"
#include <cstdio>
#include <random>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

clock_t tm;

// typedefs
typedef float value_t;
typedef vector<value_t> state_t;
typedef vector<unsigned> ustate_t;
typedef vector<int> intstate_t;
typedef vector<vector<value_t>> array_t;

// random
default_random_engine rng;
uniform_real_distribution<value_t> unifdistribution(0.0, 1.0);
normal_distribution<value_t> normaldistribution(0.0, 1.0);

// enums
enum InteractionModel {randomModel, gridModel};
