#pragma once
#include "NumCpp/NdArray.hpp"

class RBF_kernel {
	double length_scale;
	double output_scale; // std
public:
	RBF_kernel();
	RBF_kernel(double length_scale, double output_scale);
	nc::NdArray<double> estimate(nc::NdArray<double>& x1, nc::NdArray<double>& x2);
	//nc::NdArray<double> cov_matrix(nc::NdArray<double>& X1, nc::NdArray<double>& X2);

};