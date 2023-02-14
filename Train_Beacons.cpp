#include "Train_Beacons.h"
#include <string>

Train_Beacons::Train_Beacons(double _X, double _Y, double _Beacon00001, double _Beacon00002, double _Beacon00003, double _Beacon00004, double _Beacon00005, double _Beacon00006, std::string _modified = "", int _id = 1)
{
    X = _X;
    Y = _Y;
    Beacon00001 = _Beacon00001;
    Beacon00002 = _Beacon00002;
    Beacon00003 = _Beacon00003;
    Beacon00004 = _Beacon00004;
    Beacon00005 = _Beacon00005;
    Beacon00006 = _Beacon00006;
    modified = _modified;
    id = _id;
}
Train_Beacons::Train_Beacons(int _id, double _X, double _Y, std::string _modified, double _generalVal)
{
    id = _id;
    X = _X;
    Y = _Y;
    modified = _modified;
    generalVal = _generalVal;
}