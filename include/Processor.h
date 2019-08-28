//
// Created by marco on 19-8-13.
//

#ifndef PROJECT_PROCESSOR_H
#define PROJECT_PROCESSOR_H

#include "Interface.h"
#include "Storage.h"
#include <memory>

class IProcessor : public IInterface
{
public:
    typedef std::shared_ptr<IProcessor> Ptr;

    virtual void Process(IDataStorage::Ptr dataStorage) = 0;
    virtual void Init(IDataStorage::Ptr dataStorage) {};
    virtual void Cleanup() {};
};

#endif //PROJECT_PROCESSOR_H
