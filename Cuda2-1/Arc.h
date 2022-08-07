#pragma once

#include "pch.h"


struct Arc: public BaseArc
{
	ustate_t ID;
	intstate_t ATTR;
	ustate_t PRED;
	ustate_t SUCC;
	state_t ALPH;

	// Parameters
	unsigned interactionModel;

	// Default ctor
	Arc(unsigned NodeLength_, unsigned interactionModel_) : BaseArc(NodeLength_),
	interactionModel(interactionModel_)
	{
		ArcLength = 5;
	}

	// ctor
	Arc(unsigned NodeLength_, unsigned interactionModel_, value_t ProbArc_, value_t ProbAttractive_, value_t ForceAlpha_) :
		BaseArc(NodeLength_, ProbArc_, ProbAttractive_, ForceAlpha_),
		interactionModel(interactionModel_)
	{
		ArcLength = 5;
	}
	void Init(unsigned NumNode);
	void Step_CPU(value_t* ArcData, value_t* NodeData, value_t t);

	state_t makeArray();
	void print(string dataHeader, ofstream ofs, value_t t, bool hdr);
};

state_t Arc::makeArray() {
	state_t ArcArray;
	state_t ArcVector;
	for (unsigned i = 0; i < NumArc; i++) {
		ArcVector = { (value_t)ID[i], (value_t)ATTR[i], (value_t)PRED[i], (value_t)SUCC[i], ALPH[i] };
		ArcArray.insert(ArcArray.end(), ArcVector.begin(), ArcVector.end());
	}
	return ArcArray;
}

void Arc::print(string dataHeader, ofstream ofs, value_t t, bool hdr)
{
	if (hdr) {
		ofs << dataHeader;
		ofs << "T, ID, ATTR, PRED, SUCC, ALPH \n";
	}
	else {
		for (unsigned i = 0; i < NumArc; i++) {
			ofs << t << " " << ID[i] << " " << ATTR[i] << " " << PRED[i] 
				<< " " << SUCC[i] << " " << ALPH[i] << "\n";
		}
	}

}

void Arc::Init(unsigned NumNode) {
	name = "2D Spring Arc";
	unsigned RandomArc, RandomAttractive;
	unsigned idx = 0;
	if (interactionModel == randomModel) {
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
						ALPH.push_back(ForceAlpha);  // Force Weight
						idx++;
					}
				}
			}
		}
		NumArc = idx;
	}

	if (interactionModel == gridModel) {
		unsigned NumNodeSide = sqrt(NumNode);
		unsigned kPRED, kSUCC;

		for (unsigned i = 0; i < NumNodeSide; i++) {
			for (unsigned j = 0; j < NumNodeSide; j++) {
				kPRED = j * NumNodeSide + i;
				// horizontal arc
				if (i < NumNodeSide - 1) {
					kSUCC = j * NumNodeSide + i + 1;
					RandomAttractive = (unifdistribution(rng) < ProbAttractive);
					ID.push_back(idx); // ID
					ATTR.push_back(2 * RandomAttractive - 1);  // Arc Type
					PRED.push_back(kPRED);  // Pred Node
					SUCC.push_back(kSUCC);  // Succ Node
					ALPH.push_back(ForceAlpha);  // Force Weight
					idx++;
				}
				
				// vertical arc
				if (j < NumNodeSide - 1) {
					kSUCC = (j + 1) * NumNodeSide + i;
					RandomAttractive = (unifdistribution(rng) < ProbAttractive);
					ID.push_back(idx); // ID
					ATTR.push_back(2 * RandomAttractive - 1);  // Arc Type
					PRED.push_back(kPRED);  // Pred Node
					SUCC.push_back(kSUCC);  // Succ Node
					ALPH.push_back(ForceAlpha);  // Force Weight
					idx++;
				}
			}
		}
		NumArc = idx;
	}

}

__device__ void ArcStep_GPU(value_t* ArcData, value_t* NodeData, unsigned ArcLength, unsigned NodeLength,
	value_t t) {

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
	value_t DFXpred = -ATTR * ALPH * (Xpred - Xsucc);
	value_t DFYpred = -ATTR * ALPH * (Ypred - Ysucc);
	value_t DFXsucc =  ATTR * ALPH * (Xpred - Xsucc);
	value_t DFYsucc =  ATTR * ALPH * (Ypred - Ysucc);

	// Pack
	/*NodeData[PRED * NodeLength + 3] = NodeData[PRED * NodeLength + 3] + DFXpred;
	NodeData[PRED * NodeLength + 4] = NodeData[PRED * NodeLength + 4] + DFYpred;
	NodeData[SUCC * NodeLength + 3] = NodeData[SUCC * NodeLength + 3] + DFXsucc;
	NodeData[SUCC * NodeLength + 3] = NodeData[SUCC * NodeLength + 3] + DFYsucc;*/
	atomicAdd(&(NodeData[PRED * NodeLength + 3]), DFXpred);
	atomicAdd(&(NodeData[PRED * NodeLength + 4]), DFYpred);
	atomicAdd(&(NodeData[SUCC * NodeLength + 3]), DFXsucc);
	atomicAdd(&(NodeData[SUCC * NodeLength + 4]), DFYsucc);

	printf("Arc ID:  %d idx:  %d  t:  %f \n", ID, idx, t);

	__syncthreads();
}


void Arc::Step_CPU(value_t* ArcData, value_t* NodeData, value_t) {
	// Constants


	for (unsigned tid = 0; tid < NumArc; tid++) {

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
		value_t DFXpred = -ATTR * ALPH * (Xpred - Xsucc);
		value_t DFYpred = -ATTR * ALPH * (Ypred - Ysucc);
		value_t DFXsucc =  ATTR * ALPH * (Xpred - Xsucc);
		value_t DFYsucc =  ATTR * ALPH * (Ypred - Ysucc);

		// Pack

		NodeData[PRED * NodeLength + 3] = NodeData[PRED * NodeLength + 3] + DFXpred;
		NodeData[PRED * NodeLength + 4] = NodeData[PRED * NodeLength + 4] + DFYpred;
		NodeData[SUCC * NodeLength + 3] = NodeData[SUCC * NodeLength + 3] + DFXsucc;
		NodeData[SUCC * NodeLength + 4] = NodeData[SUCC * NodeLength + 4] + DFYsucc;
	}

}

