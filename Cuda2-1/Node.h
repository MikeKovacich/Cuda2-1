#pragma once
#include "pch.h"
#include "BaseNode.h"

struct Node : public BaseNode
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
	state_t M;

	// Parameters
	string name;
	unsigned interactionModel;

	// Default ctor
	Node(unsigned NumNode_, unsigned interactionModel_): BaseNode(NumNode_),
	interactionModel(interactionModel_)
	{
		NodeLength = 11;
		name = "RW2D";
	}

	// ctor
	Node(unsigned NumNode_, unsigned interactionModel_, value_t Diffusion_, value_t Drift_, value_t BoxWidth_, value_t dt_):
		BaseNode(NumNode_, Diffusion_, Drift_, BoxWidth_, dt_),
		interactionModel(interactionModel_)
	{
		NodeLength = 11;
		name = "RW2D";
	}

	void Init();
	void Step_CPU(value_t* NodeData, value_t t);

	state_t makeArray();
	void print(string dataHeader, value_t* data, ofstream& ofs, value_t t, bool hdr);
};

state_t Node::makeArray() {
	state_t NodeArray;
	state_t NodeVector;
	for (unsigned i = 0; i < NumNode; i++) {
		NodeVector = { (value_t)ID[i], X[i], Y[i],
			FX[i], FY[i], FX1[i], FY1[i],(value_t)CTR[i], DX[i], DY[i], M[i] };
		NodeArray.insert(NodeArray.end(), NodeVector.begin(), NodeVector.end());
	}
	return NodeArray;
}

void Node::print(string dataHeader, value_t* data, ofstream& ofs, value_t t, bool hdr)
{
	if (hdr) {
		ofs << dataHeader;
		ofs << "T, ID, X, Y, FX, FY, FX1, FY1, CTR, DX, DY, M \n";
	}
	else {
		unsigned idx, ID, CTR;
		value_t X, Y, FX, FY, FX1, FY1, DX, DY, M;
		for (unsigned i = 0; i < NumNode; i++) {
			idx = i * NodeLength;
			ID = (unsigned)data[idx];
			X = data[idx + 1];
			Y = data[idx + 2];
			FX = data[idx + 3];
			FY = data[idx + 4];
			FX1 = data[idx + 5];
			FY1 = data[idx + 6];
			CTR = (unsigned)data[idx + 7];
			DX = data[idx + 8];
			DY = data[idx + 9];
			M = data[idx + 10];

			ofs << t << "," << ID << "," << X << "," << Y << "," << FX << "," << 
				FY << "," << FX1 << "," << FY1 << "," << CTR << "," << DX << "," << DY << "," << M << "\n";
		}
	}
}

void Node::Init() {

	X.resize(NumNode, 0.0);
	Y.resize(NumNode, 0.0);
	FX.resize(NumNode, 0.0);
	FY.resize(NumNode, 0.0);
	FX1.resize(NumNode, 0.0);
	FY1.resize(NumNode, 0.0);
	ID.resize(NumNode, 0);
	CTR.resize(NumNode, 0);
	DX.resize(NumNode, 0.0);
	DY.resize(NumNode, 0.0);
	M.resize(NumNode, 1.0);

	// Inital placement
	if (interactionModel == randomModel) {
		for (unsigned i = 0; i < NumNode; i++) {
			X[i] = BoxWidth * unifdistribution(rng);
			Y[i] = BoxWidth * unifdistribution(rng);
			ID[i] = i;
		}
	}
	if (interactionModel == gridModel) {
		unsigned NumNodeSide = sqrt(NumNode);
		if (NumNodeSide * NumNodeSide != NumNode) {
			cout << "Need a perfect square number of nodes for this model" << endl;
			exit;
		}
		value_t delta = BoxWidth / (NumNodeSide + 1);
		for (unsigned i = 0; i < NumNodeSide; i++) {
			for (unsigned j = 0; j < NumNodeSide; j++) {
				X[j * NumNodeSide + i] = i * delta + delta;
				Y[j * NumNodeSide + i] = j * delta + delta;
			}
		}
		M[0] = 1000.0;
		M[NumNodeSide - 1] = 1000.0;
		M[NumNode - NumNodeSide] = 1000.0;
		M[NumNode - 1] = 1000.0;
	}

}

__device__ void NodeStep_GPU(value_t* NodeData, unsigned NodeLength,
	value_t Drift, value_t Diffusion, value_t BoxWidth, value_t dt, value_t t, curandState *states) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Constants
	value_t minPos = 0.0;
	value_t maxPos = BoxWidth;

	// Unpack
	unsigned idx = tid * NodeLength;
	unsigned ID = (unsigned)NodeData[idx];
	value_t X = NodeData[idx + 1];
	value_t Y = NodeData[idx + 2];
	value_t FX = NodeData[idx + 3];
	value_t FY = NodeData[idx + 4];
	value_t FX1 = NodeData[idx + 5];
	value_t FY1 = NodeData[idx + 6];
	unsigned CTR = (unsigned)NodeData[idx + 7];
	value_t M = NodeData[idx + 10];

	// Random Number Generator
	CTR++;
	curand_init(tid, CTR, 0, &states[tid]);

	// Delta State
	value_t dX;
	value_t dY;

	dX = FX * dt;
	dY = FY * dt;

	dX += Diffusion * sqrt(dt) * curand_normal(&states[tid]);
	dY += Diffusion * sqrt(dt) * curand_normal(&states[tid]);

	dX = dX / M;
	dY = dY / M;

	// the dynamical equation with boundary checking
	X = X + dX;
	Y = Y + dY;
	if (X < minPos) { X = minPos; }
	if (X > maxPos) { X = maxPos; }
	if (Y < minPos) { Y = minPos; }
	if (Y > maxPos) { Y = maxPos; }



	if (abs(NodeData[idx + 1] - X) < .001) {
		printf("!!!!!!!!!Node ID:  %d idx:  %d t:  %f \n", ID, idx, t);
	}
	printf("Node ID:  %d idx:  %d tid:  %d t:  %f \n", ID, idx, tid, t);

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



	__syncthreads();

}
void Node::Step_CPU(value_t* NodeData, value_t) {
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
		value_t M = NodeData[idx + 10];


		// Random Number Generator
		CTR++;

		// Delta State
		value_t dX = FX * dt;
		value_t dY = FY * dt;
		dX += Diffusion * sqrt(dt) * normaldistribution(rng);
		dY += Diffusion * sqrt(dt) * normaldistribution(rng);

		dX = dX / M;
		dY = dY / M;

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
