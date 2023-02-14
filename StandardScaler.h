#pragma once

#include"NumCpp.hpp"

class StandardScaler
{
	nc::NdArray<double> X;
	nc::NdArray<double> mu;
	nc::NdArray<double> std;
	nc::NdArray<double> Z;
public:
	StandardScaler();
	StandardScaler fit(nc::NdArray<double>& X);
	nc::NdArray<double> transform(nc::NdArray<double>& X);
	nc::NdArray<double> inverse_transform(nc::NdArray<double>& Z);
	nc::NdArray<double> inverse_transform_std(nc::NdArray<double>& Z_std);
};

