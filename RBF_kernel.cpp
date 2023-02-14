#pragma once

#include "RBF_kernel.h"
#include "NumCpp.hpp"

using namespace std;

RBF_kernel::RBF_kernel()
{
	this->length_scale = 1;
	this->output_scale = 1;
}

RBF_kernel::RBF_kernel(double length_scale, double output_scale)
{
	this->length_scale = length_scale;
	this->output_scale = output_scale;
}

nc::NdArray<double> RBF_kernel::estimate(nc::NdArray<double>& X1, nc::NdArray<double>& X2) {
	///============================================================================
	// Method Description:
	/// Returns 
	///
	/// @param X1
	/// @param X2
	/// 
	/// @return bool
	///		Anisotropic squared exponential kernel.
	/// Args:
	///		X1: Array of m points (m x d).
	///		X2 : Array of n points (n x d).
	/// Returns :
	///		(m x n) matrix.
	///============================================================================
	
	int m = X1.shape().rows;
	int n = X2.shape().rows;

	// 1 / ( 2*length_scale^2 )
	double gamma = 0.5 / nc::square(this->length_scale);

	nc::NdArray<double> X1_sq = nc::sum(nc::square(X1), nc::Axis::COL).reshape(-1,1); // m x 1
	nc::NdArray<double> X1_stack = X1_sq;
	for (int i = 1; i < n; i++)
	{
		X1_stack = nc::hstack({ X1_stack, X1_sq }); // m x n
	}

	nc::NdArray<double> X2_sq = nc::sum(nc::square(X2), nc::Axis::COL); // 1 x n
	nc::NdArray<double> X2_stack = X2_sq;
	for (int i = 1; i < m; i++)
	{
		X2_stack = nc::vstack({ X2_stack, X2_sq }); // m x n
	}

	nc::NdArray<double> sq_dist = X1_stack + X2_stack - 2.0 * nc::dot(X1, nc::transpose(X2));


	return nc::square(this->output_scale) * nc::exp(-gamma * sq_dist);
}