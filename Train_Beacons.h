#pragma once
#include <string>

class Train_Beacons
{
public:
    int id;
    double X, Y;
    double Beacon00001 = -200, Beacon00002 = -200, Beacon00003 = -200, Beacon00004 = -200, Beacon00005 = -200, Beacon00006 = -200;

    std::string modified;
    double generalVal;

    Train_Beacons() {};
    Train_Beacons(double _X, double _Y, double _Beacon00001, double _Beacon00002, double _Beacon00003, double _Beacon00004, double _Beacon00005, double _Beacon00006, std::string _modified, int _id);
    Train_Beacons(int _id, double _X, double _Y, std::string _modified, double _generalVal);
    ~Train_Beacons() {};

};
