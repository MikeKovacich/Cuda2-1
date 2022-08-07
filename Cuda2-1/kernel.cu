#include "pch.h"
#include "BaseNode.h"
#include "BaseArc.h"
#include "Node.h"
#include "Arc.h"

// system kernel with time loop
__global__ void systemRun_GPU(unsigned NumSteps, unsigned LogPeriod) {

	// RandomNumber Generator
	curandState *devStates;

	//// allocate device memory
	unsigned NodeLength = node.NodeLength;
	unsigned NumArc = arc.NumArc;
	unsigned ArcLength = arc.ArcLength;
	unsigned nBytesNode = NumNode * NodeLength * sizeof(value_t);
	unsigned nBytesArc = NumArc * ArcLength * sizeof(value_t);
	value_t *nodeData_Dev;
	value_t *arcData_Dev;
	//// copy data from host to device
	// node
	CHECK(cudaMalloc(&nodeData_Dev, nBytesNode));
	CHECK(cudaMalloc(&arcData_Dev, nBytesArc));
	CHECK(cudaMalloc((void **)&devStates, NumNode * sizeof(curandState)));
	// arc
	CHECK(cudaMemcpy(nodeData_Dev, nodeData_Host, nBytesNode, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(arcData_Dev, arcData_Host, nBytesArc, cudaMemcpyHostToDevice));

	//// execution configuration
	int nElemNode = NumNode;
	int blocksizeNode = min(128, (int)NumNode);
	dim3 blockNode(blocksizeNode);
	dim3 gridNode((nElemNode + blockNode.x - 1) / blockNode.x, 1);

	int nElemArc = NumArc;
	int blocksizeArc = min(128, (int)NumArc);
	dim3 blockArc(blocksizeArc);
	dim3 gridArc((nElemArc + blockArc.x - 1) / blockArc.x, 1);

	//// Simulation over time
	value_t driftValue = node.Drift;
	value_t diffusionValue = node.Diffusion;
	value_t boxwidthValue = node.BoxWidth;
	value_t dtValue = node.dt;
	tm = clock();




	for (unsigned t = 0; t < NumSteps; t++) {
		printf("t:  %f \n", t);;

		// RW1D
		/*BaseArcStep_GPU << <gridArc, blockArc >> > (arcData_Dev, nodeData_Dev, ArcLength,
			NodeLength, arc.ForceAlpha);
		BaseNodeStep_GPU << <gridNode, blockNode >> > (nodeData_Dev, NodeLength,
			node.Drift, node.Diffusion, node.BoxWidth, node.dt, devStates); */

			// RW2D
		ArcStep_GPU << <gridArc, blockArc >> > (arcData_Dev, nodeData_Dev, ArcLength,
			NodeLength, t);
		CHECK(cudaGetLastError());
		NodeStep_GPU << <gridNode, blockNode >> > (nodeData_Dev, NodeLength,
			node.Drift, node.Diffusion, node.BoxWidth, node.dt, t, devStates);
		CHECK(cudaGetLastError());
		/*if (t%LogPeriod == 0) {
			CHECK(cudaMemcpy(nodeData_Host, nodeData_Dev, nBytesNode, cudaMemcpyDeviceToHost));
			node.print(dataHeader, nodeData_Host, ofs, t, false);
		}*/
	}
}




enum system {RW1D, RW2D};
int main(int argc, char **argv)
{
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s test struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// Run on GPU or CPU
	bool GPU = true;
	bool CPU = !GPU;

	// Graph Parameters
	const unsigned NumNode = 128;		// Number of Nodes
	value_t ProbArc = 0.1;				// Probability that two particles interact
	value_t ProbAttractive = 0.5;		// Probability that the interaction is attractive
	unsigned interactionModel = randomModel;

	// Node Parameters
	value_t Diffusion = 5.0;		// Random Walk Sigma
	value_t Drift = 10.0;

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

	// Create System


	int sys = RW1D;
	//BaseNode node(NumNode, Diffusion, Drift, BoxWidth, dt);
	//BaseArc arc(node.NodeLength, ProbArc, ProbAttractive, ForceAlpha);

	sys = RW2D;

	Node node(NumNode, interactionModel, Diffusion, Drift, BoxWidth, dt);
	Arc arc(node.NodeLength, interactionModel, ProbArc, ProbAttractive, ForceAlpha);

	// Create System Arrays
	node.Init();									// Structure of vectors
	state_t nodeArray = node.makeArray();			// 1D vectpr
	value_t* nodeData_Host = nodeArray.data();		// 1D array
	arc.Init(NumNode);								// Structure of vectors
	state_t  arcArray = arc.makeArray();			// 1D vector
	value_t* arcData_Host = arcArray.data();		// 1D array

	// csv file data header
	string dataHeader = node.name +' ' + arc.name + ' ';
	if(GPU) dataHeader = dataHeader + "Device = GPU";
	if(CPU) dataHeader = dataHeader + "Device = CPU";
	time_t now = time(0);
	char* date_time = ctime(&now);
	dataHeader = dataHeader + "  NumNodes = " + to_string(NumNode) + "   " + date_time;

	// csv file data column labels
	node.print(dataHeader, nodeData_Host, ofs, 0.0, true);

	if (GPU) {

		// RandomNumber Generator
		curandState *devStates;

		//// allocate device memory
		unsigned NodeLength = node.NodeLength;
		unsigned NumArc = arc.NumArc;
		unsigned ArcLength = arc.ArcLength;
		unsigned nBytesNode = NumNode * NodeLength * sizeof(value_t);
		unsigned nBytesArc = NumArc * ArcLength * sizeof(value_t);
		value_t *nodeData_Dev;
		value_t *arcData_Dev;
		//// copy data from host to device
		// node
		CHECK(cudaMalloc(&nodeData_Dev, nBytesNode));
		CHECK(cudaMalloc(&arcData_Dev, nBytesArc));
		CHECK(cudaMalloc((void **)&devStates, NumNode * sizeof(curandState)));
		// arc
		CHECK(cudaMemcpy(nodeData_Dev, nodeData_Host, nBytesNode, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(arcData_Dev, arcData_Host, nBytesArc, cudaMemcpyHostToDevice));

		//// execution configuration
		int nElemNode = NumNode;
		int blocksizeNode = min(128, (int)NumNode);
		dim3 blockNode(blocksizeNode);
		dim3 gridNode((nElemNode + blockNode.x - 1) / blockNode.x, 1);

		int nElemArc = NumArc;
		int blocksizeArc = min(128, (int)NumArc);
		dim3 blockArc(blocksizeArc);
		dim3 gridArc((nElemArc + blockArc.x - 1) / blockArc.x, 1);

		//// Simulation over time
		value_t driftValue = node.Drift;
		value_t diffusionValue = node.Diffusion;
		value_t boxwidthValue = node.BoxWidth;
		value_t dtValue = node.dt;
		tm = clock();

		systemRun_GPU(NumSteps, LogPeriod);


		//for (unsigned t = 0; t < NumSteps; t++) {
		//	cout << "t:  " << t << '\n';

		//	// RW1D
		//	/*BaseArcStep_GPU << <gridArc, blockArc >> > (arcData_Dev, nodeData_Dev, ArcLength, 
		//		NodeLength, arc.ForceAlpha);
		//	BaseNodeStep_GPU << <gridNode, blockNode >> > (nodeData_Dev, NodeLength,
		//		node.Drift, node.Diffusion, node.BoxWidth, node.dt, devStates); */

		//	// RW2D
		//	ArcStep_GPU << <gridArc, blockArc >> > (arcData_Dev, nodeData_Dev, ArcLength,
		//		NodeLength, t);
		//	CHECK(cudaGetLastError());
		//	NodeStep_GPU << <gridNode, blockNode >> > (nodeData_Dev, NodeLength,
		//		node.Drift, node.Diffusion, node.BoxWidth, node.dt, t, devStates);
		//	CHECK(cudaGetLastError());
		//	if (t%LogPeriod == 0) {
		//		CHECK(cudaMemcpy(nodeData_Host, nodeData_Dev, nBytesNode, cudaMemcpyDeviceToHost));
		//		node.print(dataHeader, nodeData_Host, ofs, t, false);
		//	}
		//}
		//// Free storage
		CHECK(cudaFree(nodeData_Dev));
		CHECK(cudaFree(arcData_Dev));
		CHECK(cudaFree(devStates));

		//// reset device
		CHECK(cudaDeviceReset());
	}

	if (CPU){
		//// Simulation over time
		tm = clock();
		for (unsigned t = 0; t < NumSteps; t++) {
			std::cout << "t:  " << t << '\n';

			arc.Step_CPU(arcData_Host, nodeData_Host, t);
			node.Step_CPU(nodeData_Host, t);

			if (t%LogPeriod == 0) {
				node.print(dataHeader, nodeData_Host, ofs, t, false);
			}
		}
	}
	
	tm = clock() - tm;
	std::cout << "Elapsed Time (sec):  " << (float)tm / CLOCKS_PER_SEC << endl;

	ofs.close();


	return EXIT_SUCCESS;
}