#include "Common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h> 
#include "cuda.h"
#include <cstdio>
#include <random>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

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

double seconds()
{
	tm = clock();
	return((float)tm / CLOCKS_PER_SEC);
}

unsigned seq()
{
	static int i;
	return i++;
}
value_t unifRandom()
{
	return unifdistribution(rng);
}

struct Node
{
	state_t X;
	state_t Y;
	state_t FX;
	state_t FY;
	state_t FX1;
	state_t FY1;
	ustate_t ID;
	ustate_t CTR;
	state_t DX;
	state_t DY;
};

state_t makeNodeArray(Node &node, unsigned &NodeLength) {
	state_t NodeArray;
	state_t NodeVector;
	for (unsigned i = 0; i < node.ID.size(); i++) {
		NodeVector = {(value_t) node.ID[i], node.X[i], node.Y[i], 
			node.FX[i], node.FY[i], node.FX1[i], node.FY1[i],(value_t) node.CTR[i], node.DX[i], node.DY[i]};
		NodeArray.insert(NodeArray.end(), NodeVector.begin(), NodeVector.end());
	}
	NodeLength = NodeVector.size();
	return NodeArray;
}

int printNode(ofstream ofs, Node &node, value_t t)
{

	for (unsigned i = 0; i < node.ID.size(); i++) {
		ofs << t << " " << node.ID[i] << " " << node.X[i] << " " << node.Y[i] << endl;
	}
	
	return(0);
}

void InitNode(Node &node, unsigned NumNode, value_t Sigma, value_t BoxWidth) {
	node.X.resize(NumNode, 0.0);
	node.Y.resize(NumNode, 0.0);
	node.FX.resize(NumNode, 0.0);
	node.FY.resize(NumNode, 0.0);
	node.FX1.resize(NumNode, 0.0);
	node.FY1.resize(NumNode, 0.0);
	node.ID.resize(NumNode, 0);
	node.CTR.resize(NumNode, 0);
	node.DX.resize(NumNode, 0.0);
	node.DY.resize(NumNode, 0.0);
	for (unsigned i = 0; i < NumNode; i++) {
		node.X[i] = BoxWidth * unifdistribution(rng);
		node.Y[i] = BoxWidth * unifdistribution(rng);
		node.ID[i] = i;
	}
}

__global__ void NodeStep_GPU(value_t* NodeData, unsigned NodeLength,  
value_t RandomWalkSigma, value_t BoxWidth, value_t dt, curandState *states) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Constants
	value_t minPos = 0.0;
	value_t maxPos = BoxWidth;

	// Unpack
	unsigned idx = tid * NodeLength;
	value_t ID = (unsigned)NodeData[idx];
	value_t X = NodeData[idx + 1];
	value_t Y = NodeData[idx + 2];
	value_t FX = NodeData[idx + 3];
	value_t FY = NodeData[idx + 4];
	value_t FX1 = NodeData[idx + 5];
	value_t FY1 = NodeData[idx + 6];
	unsigned CTR = (unsigned)NodeData[idx + 7];

	// Random Number Generator
	CTR++;
	curand_init(tid, CTR, 0, &states[tid]);

	// Delta State
	value_t dX;
	value_t dY;

	dX = FX * dt;
	dY = FY * dt;

	dX += RandomWalkSigma * curand_normal(&states[tid]);
	dY += RandomWalkSigma * curand_normal(&states[tid]);

	// the dynamical equation with boundary checking
	X = X + dX;
	Y = Y + dY;
	if (X < minPos) { X = minPos; }
	if (X > maxPos) { X = maxPos; }
	if (Y < minPos) { Y = minPos; }
	if (Y > maxPos) { Y = maxPos; }

	// Pack
	NodeData[idx + 1] = X;
	NodeData[idx + 2] = Y;
	NodeData[idx + 3] = 0.0;
	NodeData[idx + 4] = 0.0;
	NodeData[idx + 5] = FX;
	NodeData[idx + 6] = FY;
	NodeData[idx + 7] = CTR;
	NodeData[idx + 8] = dX;
	NodeData[idx + 9] = dY;

	//__syncthreads();

}
void NodeStep_CPU(value_t* NodeData, unsigned NumNode, unsigned NodeLength, 
	value_t RandomWalkSigma, value_t BoxWidth, value_t dt) {
	// Constants
	value_t minPos = 0.0;
	value_t maxPos = BoxWidth;

	for (int tid = 0; tid < NumNode; tid++) {

		// Unpack
		unsigned idx = tid * NodeLength;
		value_t ID = (unsigned)NodeData[idx];
		value_t X = NodeData[idx + 1];
		value_t Y = NodeData[idx + 2];
		value_t FX = NodeData[idx + 3];
		value_t FY = NodeData[idx + 4];
		unsigned CTR = (unsigned)NodeData[idx + 5];

		// Random Number Generator
		CTR++;

		// Delta State
		value_t dX = FX * dt;
		value_t dY = FY * dt;
		dX += RandomWalkSigma * normaldistribution(rng);
		dY += RandomWalkSigma * normaldistribution(rng);

		// the dynamical equation with boundary checking
		X = X + dX;
		Y = Y + dY;
		if (X < minPos) { X = minPos; }
		if (X > maxPos) { X = maxPos; }
		if (Y < minPos) { Y = minPos; }
		if (Y > maxPos) { Y = maxPos; }

		// Pack
		NodeData[idx + 1] = X;
		NodeData[idx + 2] = Y;
		NodeData[idx + 3] = 0.0;
		NodeData[idx + 4] = 0.0;
		NodeData[idx + 5] = FX;
		NodeData[idx + 6] = FY;
		NodeData[idx + 7] = CTR;
		NodeData[idx + 8] = dX;
		NodeData[idx + 9] = dY;
	}
}

struct Arc
{
	intstate_t ATTR;
	ustate_t PRED;
	ustate_t SUCC;
	state_t ALPH;
	ustate_t ID;
};

state_t makeArcArray(Arc &arc, unsigned &ArcLength) {
	state_t ArcArray;
	state_t ArcVector;
	for (unsigned i = 0; i < arc.ID.size(); i++) {
		ArcVector = { (value_t)arc.ID[i], (value_t)arc.ATTR[i], (value_t)arc.PRED[i], (value_t)arc.SUCC[i], arc.ALPH[i]};
		ArcArray.insert(ArcArray.end(), ArcVector.begin(), ArcVector.end());
	}
	ArcLength = ArcVector.size();
	return ArcArray;
}

int printArc(Arc &arc)
{
	for (unsigned i = 0; i < arc.ID.size(); i++) {
		std::printf("ID %u:  PRED %u SUCC %u ATTR %d\n", arc.ID[i], arc.PRED[i], arc.SUCC[i], arc.ATTR[i]);
	}
	return(0);
}

void InitArc(Arc &arc, unsigned NumNode, value_t ProbArc, value_t ProbAttractive, value_t ForceAlpha) {
	unsigned RandomArc, RandomAttractive;
	unsigned ID = 0;
	for (unsigned i = 0; i < NumNode; i++) {
		for (unsigned j = 0; j < NumNode; j++) {
			if (i < j) {
				RandomArc = (unifdistribution(rng) < ProbArc);
				if (RandomArc > 0) {
					RandomAttractive = (unifdistribution(rng) < ProbAttractive);
					// Initialize each arc
					arc.ID.push_back(ID); // ID
					arc.ATTR.push_back(2 * RandomAttractive - 1);  // Arc Type
					arc.PRED.push_back(i);  // Pred Node
					arc.SUCC.push_back(j);  // Succ Node
					arc.ALPH.push_back(ForceAlpha);  // Force Weight
					ID++;
				}
			}
		}
	}
}

__global__ void ArcStep_GPU(value_t* ArcData, value_t* NodeData, unsigned ArcLength, unsigned NodeLength) {
	
	// Constants
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Unpack
	unsigned idx = tid * ArcLength;
	unsigned ID = (unsigned)ArcData[idx];
	int ATTR = (int)ArcData[idx + 1];
	unsigned PRED = (unsigned)ArcData[idx + 2];
	unsigned SUCC = (unsigned)ArcData[idx + 3];
	value_t ALPH = ArcData[idx + 4];
	value_t Xpred = NodeData[PRED * NodeLength + 1];
	value_t Ypred = NodeData[PRED * NodeLength + 2];
	value_t Xsucc = NodeData[SUCC * NodeLength + 1];
	value_t Ysucc = NodeData[SUCC * NodeLength + 2];

	// Dynamics
	value_t DFXpred =  ATTR * ALPH * (Xpred - Xsucc);
	value_t DFYpred =  ATTR * ALPH * (Ypred - Ysucc);
	value_t DFXsucc = -ATTR * ALPH * (Xpred - Xsucc);
	value_t DFYsucc = -ATTR * ALPH * (Ypred - Ysucc);

	// Pack
	/*NodeData[PRED * NodeLength + 3] = NodeData[PRED * NodeLength + 3] + DFXpred;
	NodeData[PRED * NodeLength + 4] = NodeData[PRED * NodeLength + 4] + DFYpred;
	NodeData[SUCC * NodeLength + 3] = NodeData[SUCC * NodeLength + 3] + DFXsucc;
	NodeData[SUCC * NodeLength + 3] = NodeData[SUCC * NodeLength + 3] + DFYsucc;*/
	atomicAdd(&(NodeData[PRED * NodeLength + 3]), DFXpred);
	atomicAdd(&(NodeData[PRED * NodeLength + 4]), DFYpred);
	atomicAdd(&(NodeData[SUCC * NodeLength + 3]), DFXsucc);
	atomicAdd(&(NodeData[SUCC * NodeLength + 4]), DFYsucc);

	//__syncthreads();
}


void ArcStep_CPU(value_t* ArcData, value_t* NodeData, unsigned NumArc, unsigned ArcLength, unsigned NodeLength) {
	// Constants


	for (unsigned tid = 0; tid < NumArc; tid++) {

		// Unpack
		unsigned idx = tid * ArcLength;
		unsigned ID = (unsigned)ArcData[idx];
		int ATTR = (int)ArcData[idx + 1];
		unsigned PRED = (unsigned) ArcData[idx + 2];
		unsigned SUCC = (unsigned) ArcData[idx + 3];
		value_t ALPH = ArcData[idx + 4];
		value_t Xpred = NodeData[PRED * NodeLength + 1];
		value_t Ypred = NodeData[PRED * NodeLength + 2];
		value_t Xsucc = NodeData[SUCC * NodeLength + 1];
		value_t Ysucc = NodeData[SUCC * NodeLength + 2];

		// Dynamics
		value_t DFXpred =  ATTR * ALPH * (Xpred - Xsucc);
		value_t DFYpred =  ATTR * ALPH * (Ypred - Ysucc);
		value_t DFXsucc = -ATTR * ALPH * (Xpred - Xsucc);
		value_t DFYsucc = -ATTR * ALPH * (Ypred - Ysucc);

		// Pack

		NodeData[PRED * NodeLength + 3] = NodeData[PRED * NodeLength + 3] + DFXpred;
		NodeData[PRED * NodeLength + 4] = NodeData[PRED * NodeLength + 4] + DFYpred;
		NodeData[SUCC * NodeLength + 3] = NodeData[SUCC * NodeLength + 3] + DFXsucc;
		NodeData[SUCC * NodeLength + 4] = NodeData[SUCC * NodeLength + 4] + DFYsucc;
	}
	
}

// test for array of struct
int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s test struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// Graph Parameters
	const unsigned NumNode = 128;		// Number of Particles
	value_t ProbArc = 0.1;				// Probability that two particles interact
	value_t ProbAttractive = 0.5;		// Probability that the interaction is attractive

	// Node Parameters
	value_t RandomWalkSigma = 10.0;		// Random Walk Sigma

	// Arc Parameters
	value_t ForceAlpha = 0.01;			// Coefficient in force model
	value_t MaxDelta = 5.0;				// Max Distance for Force

	// Simulation Parameters
	value_t dt = 1.0;					// Simulation Step Size
	unsigned NumSteps = 1000;				// Number of Steps in Run
	value_t BoxWidth = 1000.0;			// Box Width for Particles

	// Local Variables
	unsigned idx, ID;
	value_t X, Y, FX, FY, DX, DY;

	// Output File Parameters
	const string logFile("A:\\Projects\\Dynamics\\state.csv");
	unsigned LogPeriod = 10;			// Steps between outputs


	// Open Log File
	ofstream ofs(logFile, ofstream::out);
	int logFile_OK = ofs.is_open();
	bool log_output;

	// Timer
	clock_t tm;

	// Create Nodes
	Node node;
	unsigned NodeLength;
	InitNode(node, NumNode, RandomWalkSigma, BoxWidth);     // Structure of vectors
	state_t nodeArray = makeNodeArray(node, NodeLength);    // 1D vector
	auto nodeData_Host = nodeArray.data();					// 1D array

	
	// Create Arcs
	Arc arc;
	unsigned ArcLength;
	InitArc(arc, NumNode, ProbArc, ProbAttractive, ForceAlpha);	// Structure of vectors
	unsigned NumArc = arc.ID.size();
	state_t arcArray = makeArcArray(arc, ArcLength);			// 1D vector
	auto arcData_Host = arcArray.data();						// 1D array

	// RandomNumber Generator
	curandState *devStates;

	//// allocate device memory
	unsigned nBytesNode = NumNode * NodeLength * sizeof(value_t);
	unsigned nBytesArc = NumArc * ArcLength * sizeof(value_t);
	value_t *nodeData_Dev;
	value_t *arcData_Dev;
	CHECK( cudaMalloc(&nodeData_Dev, nBytesNode) );
	CHECK( cudaMalloc(&arcData_Dev, nBytesArc) );
	CHECK( cudaMalloc((void **)&devStates, NumNode * sizeof(curandState)) );

	//// copy data from host to device
	CHECK(cudaMemcpy(nodeData_Dev, nodeData_Host, nBytesNode, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(arcData_Dev, arcData_Host, nBytesArc, cudaMemcpyHostToDevice));

	//// execution configuration
	int nElemNode = NumNode;
	int blocksizeNode = NumNode;
	dim3 blockNode(blocksizeNode, 1);
	dim3 gridNode((nElemNode + blockNode.x - 1) / blockNode.x, 1);

	int nElemArc = NumArc;
	int blocksizeArc = NumArc;
	dim3 blockArc(blocksizeArc, 1);
	dim3 gridArc((nElemArc + blockArc.x - 1) / blockArc.x, 1);

	// data header
	bool GPU = true;
	bool CPU = !GPU;
	time_t now = time(0);
	char* date_time = ctime(&now);
	if (GPU) ofs << "Device = GPU" << "  NumNodes = " << NumNode << "   " << date_time << endl;
	if (CPU) ofs << "Device = CPU" << "  NumNodes = " << NumNode << "   " << date_time << endl;

	// csv file data column labels
	ofs << "t, ID, x, y, fx, fy, dx, dy" << '\n';

	//// Simulation over time
	tm = clock();
	for (unsigned t = 0; t < NumSteps; t++) {
		cout << "t:  " << t << '\n';

		if (GPU) {
			ArcStep_GPU << <gridArc, blockArc >> > (arcData_Dev, nodeData_Dev, ArcLength, NodeLength);
			NodeStep_GPU << <gridNode, blockNode >> > (nodeData_Dev, NodeLength, 
				RandomWalkSigma, BoxWidth, dt, devStates);
		}
		if (CPU) {
			ArcStep_CPU(arcData_Host, nodeData_Host, NumArc, ArcLength, NodeLength);
			NodeStep_CPU(nodeData_Host, NumNode, NodeLength, RandomWalkSigma, BoxWidth, dt);
		}

		log_output = (t%LogPeriod == 0);
		if (log_output) {
			if(GPU) CHECK(cudaMemcpy(nodeData_Host, nodeData_Dev, nBytesNode, cudaMemcpyDeviceToHost));
			for (unsigned i = 0; i < node.ID.size(); i++) {
				idx = i * NodeLength;
				ID = (unsigned) nodeData_Host[idx];
				X = nodeData_Host[idx + 1];
				Y = nodeData_Host[idx + 2];
				FX = nodeData_Host[idx + 5];
				FY = nodeData_Host[idx + 6];
				DX = nodeData_Host[idx + 8];
				DY = nodeData_Host[idx + 9];
				ofs << t << ',' << ID << ',' << X << ',' << Y << ',' << FX << ',' << FY << ',' << DX << ',' << DY << endl;
			}
		}

	}
	tm = clock() - tm;
	cout << "Elapsed Time (sec):  " << (float)tm / CLOCKS_PER_SEC << endl;

	ofs.close();

	//// Free storage
	CHECK(cudaFree(nodeData_Dev));
	CHECK(cudaFree(arcData_Dev));
	CHECK(cudaFree(devStates));

	//// reset device
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}