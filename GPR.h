#pragma once
#include "NumCpp/NdArray.hpp"
#include "RBF_kernel.h"

class GPR
{
	RBF_kernel kernel;
	double length_scale;
	double output_scale;
	double alpha; // noise in train set
	nc::NdArray<double> X_train;
	nc::NdArray<double> Y_train;

public:
	nc::NdArray<double> K;
	nc::NdArray<double> K_inv;

	GPR();
	GPR(double& length_scale, double& output_scale, double& alpha);
	void fit(nc::NdArray<double>& X_train, nc::NdArray<double>& Y_train);
	nc::NdArray<double> predict(nc::NdArray<double>& X_s);
};

