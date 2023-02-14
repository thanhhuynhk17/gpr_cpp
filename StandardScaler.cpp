#include "StandardScaler.h"

using namespace std;

StandardScaler::StandardScaler()
{
}

StandardScaler StandardScaler::fit(nc::NdArray<double>& X)
{
    this->X = X; // m x d
    this->mu = nc::mean(X, nc::Axis::ROW); // 1 x d
    this->std = nc::stdev(X, nc::Axis::ROW); // 1 x d

    cout << "mean scaler: " << this->mu;
    cout << "std scaler: " << this->std << endl;
    cout << "===================" << endl;
    return *this;
}

nc::NdArray<double> StandardScaler::transform(nc::NdArray<double>& X)
{
    int m = X.shape().rows;
    int d = X.shape().cols;

    nc::NdArray<double> X_mean_stack = this->mu; // 1 x d
    for (int i = 1; i < m; i++)
    {
        X_mean_stack = nc::vstack({ X_mean_stack, this->mu }); // m x d
    }

    nc::NdArray<double> X_std_inv = nc::linalg::inv(nc::diag(this->std)); // d x d

    return (X - X_mean_stack).dot(X_std_inv);
}

nc::NdArray<double> StandardScaler::inverse_transform(nc::NdArray<double>& Z)
{
    // SCALAR: mu & std
    if (this->mu.shape().cols==1)
    {
        return Z * this->std.at(0) + this->mu.at(0);
    }
    // MULTI: mu & std
    int n = Z.shape().rows;
    int d = Z.shape().cols;

    nc::NdArray<double> X_mean_stack = this->mu; // 1 x d
    for (int i = 1; i < n; i++)
    {
        X_mean_stack = nc::vstack({ X_mean_stack, this->mu }); // n x d
    }
    nc::NdArray<double> X_std_diag = nc::diag(this->std); // d x d

    return Z.dot(X_std_diag)+X_mean_stack;
}

nc::NdArray<double> StandardScaler::inverse_transform_std(nc::NdArray<double>& Z_std)
{
    // SCALAR: mu & std
    if (this->std.shape().cols == 1)
    {
        return Z_std * this->std.at(0);
    }

    // MULTI: mu & std
    nc::NdArray<double> X_std_diag = nc::diag(this->std); // d x d
    return Z_std.dot(X_std_diag); // (n x d) * (d x d) = n x d
}

