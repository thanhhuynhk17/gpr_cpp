#include <iostream>
#include <vector>

// 3rd party libraries
#include "NumCpp.hpp"	
#include "json/json.h"

#include "GPR.h"
#include "RBF_kernel.h"
#include "StandardScaler.h"
#include "Train_Beacons.h"

using namespace std;

int main() {
	// [BEGIN]	====================== PREPROCESS DATA ======================
	// Load json data
	Json::Value rssi_data_json;
	ifstream rssi_data_json_file("training-vbd-floor3.json", std::ifstream::binary);
	rssi_data_json_file >> rssi_data_json;

	Json::Value rssi_data = rssi_data_json["data"]["beacons"];

	// Convert to Train_Beacons
	vector<Train_Beacons> train_beacons;
	for (int i = 0; i < rssi_data.size(); i++) {
		double X = rssi_data[i]["X"].asDouble();
		double Y = rssi_data[i]["Y"].asDouble();
		double Beacon00001 = rssi_data[i]["Beacon00001"].asDouble(), 
			   Beacon00002 = rssi_data[i]["Beacon00002"].asDouble(), 
			   Beacon00003 = rssi_data[i]["Beacon00003"].asDouble(), 
			   Beacon00004 = rssi_data[i]["Beacon00004"].asDouble(), 
			   Beacon00005 = rssi_data[i]["Beacon00005"].asDouble(), 
			   Beacon00006 = rssi_data[i]["Beacon00006"].asDouble();

		Train_Beacons row_data = Train_Beacons(X,Y,Beacon00001,Beacon00002,Beacon00003,Beacon00004,Beacon00005,Beacon00006,"",1);
		train_beacons.push_back(row_data);
	}

	// CONVERT TO MATRIX
	nc::NdArray<double> X_train;
	nc::NdArray<double> Y_train;
	for (int i = 0; i < train_beacons.size(); i++)
	{
		/// X_train
		nc::NdArray<double> row_X = {	train_beacons[i].Beacon00001, 
										train_beacons[i].Beacon00002,
										train_beacons[i].Beacon00003,
										train_beacons[i].Beacon00004,
										train_beacons[i].Beacon00005,
										train_beacons[i].Beacon00006,
																		}; // 1 x dimX
		/// Y_train
		nc::NdArray<double> row_Y = { train_beacons[i].X, train_beacons[i].Y}; // 1 x dimY

		// append row data
		if (i == 0) {
			X_train = row_X; // 1 x dimX
			Y_train = row_Y; // 1 x dimY
		}
		else {
			X_train = nc::append(X_train, row_X, nc::Axis::ROW); // m x dimX
			Y_train = nc::append(Y_train, row_Y, nc::Axis::ROW); // m x dimY
		}
	}
	cout << "X_train: " << X_train.shape();
	cout << "Y_train: " << Y_train.shape();
	// [END]	====================== PREPROCESS DATA ======================

	// [BEGIN]	====================== TRAIN GPR MODEL: run once ======================
	// Standardize data
	nc::Shape X_train_shape = X_train.shape();
	nc::Shape Y_train_shape = Y_train.shape();
	
	StandardScaler X_scaler = StandardScaler().fit(X_train.reshape(-1,1));
	nc::NdArray<double> X_scaled = X_scaler.transform(X_train.reshape(-1, 1)).reshape(X_train_shape);

	StandardScaler Y_scaler = StandardScaler().fit(Y_train.reshape(-1, 1));
	nc::NdArray<double> Y_scaled = Y_scaler.transform(Y_train.reshape(-1, 1)).reshape(Y_train_shape);

	// Train GPR model
	// hyper params have been optimized
	double length_scale = 3.01;
	double output_scale = 0.954;
	double alpha = 0.329; // noise
	GPR gpr = GPR(length_scale, output_scale, alpha);
	// fit data 
	gpr.fit(X_scaled, Y_scaled);
	// [END]	====================== TRAIN GPR MODEL: run once ======================

	// [BEGIN]	====================== PREDICTION ======================
    // Standardize fingerprints input
	vector<double> arr = { -66, -71, -88, -74, -77, -81 };
	nc::NdArray<double> X_s = arr; // 1 x dimX
	cout << "X_s shape: " << X_s.shape() << X_s << endl;

	nc::Shape X_s_shape = X_s.shape();
    nc::NdArray<double> X_s_scaled = X_scaler.transform(X_s.reshape(-1,1)).reshape(X_s_shape);
	
    // Make prediction
	int n_samples = X_s_scaled.shape().rows;
	nc::NdArray<double> Y_pred = gpr.predict(X_s_scaled);
	nc::NdArray<double> Y_pred_mean = Y_pred({ 0, n_samples }, Y_pred.cSlice());
	//nc::NdArray<double> Y_pred_std = Y_pred({ n_samples, n_samples * 2 }, Y_pred.cSlice());
	
	// Inverse to origin scale
	nc::NdArray<double> Y_s  = Y_scaler.inverse_transform(Y_pred_mean);
	//nc::NdArray<double> Y_std_s = Y_scaler.inverse_transform_std(Y_pred_std);

	cout << "Y_s shape:\n" << Y_s.shape() << endl;
	Y_s.print();
	// [END]	====================== PREDICTION ======================
	cout << Y_s[0] << "," << Y_s[1] << endl;
	system("pause");
	return 1;
}
