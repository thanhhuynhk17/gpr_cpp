#include "GPR.h"
#include "NumCpp.hpp"

using namespace std;


GPR::GPR()
{
	this->length_scale = 1;
	this->output_scale = 1;
	this->kernel = RBF_kernel(this->length_scale, this->output_scale);
	this->alpha = 0;
}

GPR::GPR(double& length_scale, double& output_scale, double& alpha)
{
	this->length_scale = length_scale;
	this->output_scale = output_scale;
	this->kernel = RBF_kernel(this->length_scale, this->output_scale);
	this->alpha = alpha;
}

void GPR::fit(nc::NdArray<double>& X_train, nc::NdArray<double>& Y_train)
{
	///============================================================================
	// Method Description:
	/// Returns 
	///
	/// @param X_train
	/// @param Y_train
	/// 
	/// @return 
	///		
	/// Args:
	///		X_train: Array of m points (m x d).
	///		Y_train : Array of n points (m x dY).
	/// Returns :
	///		(m x n) matrix.
	///============================================================================
	
	this->X_train = X_train;
	this->Y_train = Y_train;

	nc::NdArray<double> noise = nc::square(this->alpha) * nc::eye<double>(X_train.shape().rows);
	this->K = this->kernel.estimate(X_train, X_train) + noise;
	this->K_inv = nc::linalg::inv(this->K); // m x m
}

nc::NdArray<double> GPR::predict(nc::NdArray<double>& X_s)
{
	///============================================================================
	/// Method Description:
	///		X_s: New input locations			(n x d)
	///		X_train : Training datas			(m x d)
	///		Y_train : Training locations		(m x dY)
	/// Returns :
	///		mu_s & stds
	///============================================================================

	nc::NdArray<double> K_s = this->kernel.estimate(X_train, X_s);	// m x n
	nc::NdArray<double> K_ss = this->kernel.estimate(X_s, X_s);		// n x n

	nc::NdArray<double> mu_s = nc::transpose(K_s).dot(this->K_inv).dot(this->Y_train); // (n x m)*( m x m)*(m x dY) = n * dY
	
	nc::NdArray<double> cov_s = K_ss - nc::transpose(K_s).dot(this->K_inv).dot(K_s); // (n x n)- (n x m)*(m x m)*(m x n) = n x n
	nc::NdArray<double> std_s = nc::sqrt(nc::transpose(nc::diag(cov_s))); // n x 1
	nc::NdArray<double> std_s_stack = std_s; // n x 1
	for (int i = 1; i < Y_train.shape().cols; i++)
	{
		std_s_stack = nc::hstack({ std_s_stack, std_s }); // n x dy
	}
	
	return nc::vstack({ mu_s, std_s_stack }); // 2n*dy
}