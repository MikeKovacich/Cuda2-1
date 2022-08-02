#pragma once

#pragma once

#include "pch.h"


struct BaseArc
{
	ustate_t ID;
	intstate_t ATTR;
	ustate_t PRED;
	ustate_t SUCC;

	// Parameters
	string name;
	unsigned NumArc;
	unsigned ArcLength;
	unsigned NodeLength;
	value_t ProbArc;
	value_t ProbAttractive;
	value_t ForceAlpha;


	// Default ctor
	BaseArc(unsigned NodeLength_): NodeLength(NodeLength_) {
		ArcLength = 4;
		ProbArc = 0.5;
		ProbAttractive = 0.5;
		ForceAlpha = 1.0;
	}
	BaseArc(unsigned NodeLength_, value_t ProbArc_, value_t ProbAttractive_, value_t ForceAlpha_):
		NodeLength(NodeLength_),
		ProbArc(ProbArc_),
		ProbAttractive(ProbAttractive_),
		ForceAlpha(ForceAlpha_)
	{
		ArcLength = 4;
	}

	virtual void Init(unsigned NumNode);
	virtual void Step_CPU(value_t* ArcData, value_t* NodeData);

	virtual state_t  makeArray();
	virtual void print(string dataHeader, ofstream ofs, value_t t, bool hdr);
};

state_t BaseArc::makeArray() {
	state_t ArcArray;
	state_t ArcVector;
	for (unsigned i = 0; i < NumArc; i++) {
		ArcVector = { (value_t)ID[i], (value_t)ATTR[i], (value_t)PRED[i], (value_t)SUCC[i] };
		ArcArray.insert(ArcArray.end(), ArcVector.begin(), ArcVector.end());
	}
	return ArcArray;
}

void BaseArc::print(string dataHeader, ofstream ofs, value_t t, bool hdr)
{
	if (hdr) {
		ofs << dataHeader;
		ofs << "T, ID, ATTR, PRED, SUCC \n"; }
	else {
		for (unsigned i = 0; i < NumArc; i++) {
			ofs << t << " " << ID[i] << " "  << ATTR[i] << " " << PRED[i] << " " << SUCC[i] << "\n";
		}
	}
}

void BaseArc::Init(unsigned NumNode) {
	name = "BaseArc";
	unsigned RandomArc, RandomAttractive;
	unsigned idx = 0;
	for (unsigned i = 0; i < NumNode; i++) {
		for (unsigned j = 0; j < NumNode; j++) {
			if (i < j) {
				RandomArc = (unifdistribution(rng) < ProbArc);
				if (RandomArc > 0) {
					RandomAttractive = (unifdistribution(rng) < ProbAttractive);
					// Initialize each arc
					ID.push_back(idx); // ID
					ATTR.push_back(2 * RandomAttractive - 1);  // Arc Type
					PRED.push_back(i);  // Pred Node
					SUCC.push_back(j);  // Succ Node
					idx++;
				}
			}
		}
	}
	NumArc = idx;
}

__global__ void BaseArcStep_GPU(value_t* ArcData, value_t* NodeData, unsigned ArcLength, unsigned NodeLength,
	value_t ForceAlpha) {

	// Constants
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Unpack
	unsigned idx = tid * ArcLength;
	unsigned ID = (unsigned)ArcData[idx];
	int ATTR = (int)ArcData[idx + 1];
	unsigned PRED = (unsigned)ArcData[idx + 2];
	unsigned SUCC = (unsigned)ArcData[idx + 3];
	value_t ALPH = ForceAlpha;
	value_t Xpred = NodeData[PRED * NodeLength + 1];
	value_t Xsucc = NodeData[SUCC * NodeLength + 1];

	// Dynamics
	value_t DFXpred = ATTR * ALPH * (Xpred - Xsucc);
	value_t DFXsucc = -ATTR * ALPH * (Xpred - Xsucc);

	// Pack
	atomicAdd(&(NodeData[PRED * NodeLength + 3]), DFXpred);
	atomicAdd(&(NodeData[SUCC * NodeLength + 3]), DFXsucc);

	//__syncthreads();
}


void BaseArc::Step_CPU(value_t* ArcData, value_t* NodeData) {
	// Constants

	for (unsigned tid = 0; tid < NumArc; tid++) {

		// Unpack
		unsigned idx = tid * ArcLength;
		unsigned ID = (unsigned)ArcData[idx];
		int ATTR = (int)ArcData[idx + 1];
		unsigned PRED = (unsigned)ArcData[idx + 2];
		unsigned SUCC = (unsigned)ArcData[idx + 3];
		value_t ALPH = ForceAlpha;
		value_t Xpred = NodeData[PRED * NodeLength + 1];
		value_t Xsucc = NodeData[SUCC * NodeLength + 1];

		// Dynamics
		value_t DFXpred = ATTR * ALPH * (Xpred - Xsucc);
		value_t DFXsucc = -ATTR * ALPH * (Xpred - Xsucc);

		// Pack

		NodeData[PRED * NodeLength + 3] = NodeData[PRED * NodeLength + 3] + DFXpred;
		NodeData[SUCC * NodeLength + 3] = NodeData[SUCC * NodeLength + 3] + DFXsucc;
	}

}


