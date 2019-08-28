//
// Created by marco on 19-8-13.
//

#ifndef PROJECT_SIAMESERPN_H
#define PROJECT_SIAMESERPN_H

#include "Processor.h"
#include "SiamTracker.h"

class SiameseRPNProcessor : public IProcessor
{
public:
    SiameseRPNProcessor():siamtracker_(){};
    ~SiameseRPNProcessor(){};

    void Init(IDataStorage::Ptr dataStorage);
    void Process(IDataStorage::Ptr dataStorage);

private:
    SiamTracker siamtracker_;
};

#endif //PROJECT_SIAMESERPN_H
