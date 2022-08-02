#pragma once

#pragma once
#include "pch.h"

// Single state with drift and diffusion
struct BaseNode
{
	// State
	ustate_t ID;
	state_t X;
	state_t FX;
	ustate_t CTR;
	state_t M;

	// Parameters
	string name;
	value_t dt;
	unsigned NumNode;
	unsigned NodeLength;
	value_t Diffusion;
	value_t Drift;
	value_t BoxWidth;
	value_t minPos;
	value_t maxPos;

	// Default ctor
	BaseNode(unsigned NumNode_): NumNode(NumNode_) {
		NodeLength = 5;
		Drift = 0.0;
		Diffusion = 1.0;
		BoxWidth = 1000.0;
		minPos = 0.0;
		maxPos = BoxWidth;
		dt = 1.0;
		ID.resize(NumNode, 0);
		X.resize(NumNode, BoxWidth / 2.0);
		FX.resize(NumNode, 0.0);
		CTR.resize(NumNode, 0);
		M.resize(NumNode, 1.0);
	}

	// ctor
	BaseNode(unsigned NumNode_, value_t Diffusion_, value_t Drift_, value_t BoxWidth_, value_t dt_):
		NumNode(NumNode_), 
		Drift(Drift_),
		Diffusion(Diffusion_),
		BoxWidth(BoxWidth_),
		dt(dt_) {
		NodeLength = 5;
		minPos = 0.0;
		maxPos = BoxWidth;
		ID.resize(NumNode, 0);
		X.resize(NumNode, BoxWidth / 2.0);
		FX.resize(NumNode, 0.0);
		CTR.resize(NumNode, 0);
		M.resize(NumNode, 1.0);
	}

	void Init();
	void Step_CPU(value_t* NodeData);

	state_t makeArray();
	void print(string dataHeaader, value_t* data, ofstream& ofs, value_t t, bool hdr);
};

state_t BaseNode::makeArray() {
	state_t NodeArray;
	state_t NodeVector;
	for (unsigned i = 0; i < NumNode; i++) {
		NodeVector = { (value_t)ID[i], X[i], FX[i], (value_t)CTR[i], M[i]};
		NodeArray.insert(NodeArray.end(), NodeVector.begin(), NodeVector.end());
	}
	return NodeArray;
}

void BaseNode::print(string dataHeader, value_t* data, ofstream& ofs, value_t t, bool hdr)
{
	if (hdr) {
		ofs << dataHeader;
		ofs << "T, ID, X, FX, CTR, M \n"; 
	}
	else {
		unsigned idx, ID, CTR;
		value_t X, FX, M;
		for (unsigned i = 0; i < NumNode; i++) {
			idx = i * NodeLength;
			ID = (unsigned)data[idx];
			X = data[idx + 1];
			FX = data[idx + 2];
			CTR = (unsigned)data[idx + 3];
			M = data[idx + 4];
			ofs << t << "," << ID << "," << X << "," << FX << "," << CTR << "," << M << "\n";
		}
	}
}

void BaseNode::Init() {

	name = "BaseNode";
	for (unsigned i = 0; i < NumNode; i++) {
		ID[i] = i;
	}
}

__global__ void BaseNodeStep_GPU(value_t* NodeData, unsigned NodeLength,
	value_t Drift, value_t Diffusion, value_t BoxWidth, value_t dt, curandState *states) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Constants
	value_t minPos = 0.0;
	value_t maxPos = BoxWidth;
	// Unpack
	unsigned idx = tid * NodeLength;
	unsigned ID = (unsigned)NodeData[idx];
	value_t X = NodeData[idx + 1];
	value_t FX = NodeData[idx + 2];
	unsigned CTR = (unsigned)NodeData[idx + 3];
	value_t M = NodeData[idx + 4];

	// Random Number Generator
	CTR++;
	curand_init(tid, CTR, 0, &states[tid]);

	// Delta State
	value_t dX;

	dX =  Drift * dt + FX * dt;

	dX += Diffusion * sqrt(dt) * curand_normal(&states[tid]);

	dX = dX / M;

	// the dynamical equation 
	X = X + dX;
	if (X < minPos) { X = minPos; }
	if (X > maxPos) { X = maxPos; }

	// Pack
	NodeData[idx + 1] = X;
	NodeData[idx + 2] = 0.0;
	NodeData[idx + 3] = CTR;

	//__syncthreads();

}
void BaseNode::Step_CPU(value_t* NodeData) {
	// Constants

	for (int tid = 0; tid < NumNode; tid++) {

		// Unpack
		unsigned idx = tid * NodeLength;
		unsigned ID = (unsigned)NodeData[idx];
		value_t X = NodeData[idx + 1];
		value_t FX = NodeData[idx + 2];
		unsigned CTR = (unsigned)NodeData[idx + 3];
		value_t M = NodeData[idx + 4];

		// Delta State
		value_t dX = Drift * dt + FX * dt;
		dX += Diffusion * sqrt(dt) * normaldistribution(rng);
		dX = dX / M;

		// the dynamical equation with boundary checking
		X = X + dX;
		if (X < minPos) { X = minPos; }
		if (X > maxPos) { X = maxPos; }

		// Pack
		NodeData[idx + 1] = X;
		NodeData[idx + 2] = 0.0;
		NodeData[idx + 3] = ++CTR;
	}
}

