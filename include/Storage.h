//
// Created by marco on 19-8-13.
//

#ifndef PROJECT_STORAGE_H
#define PROJECT_STORAGE_H

#include "Interface.h"
#include "Exception.h"
#include <memory>
#include "opencv2/opencv.hpp"

#define FLAG_USING_KCF 0x00
#define FLAG_USING_SIAMESE 0x01

class IDataObject : public IInterface
{
public:
    typedef std::shared_ptr<IDataObject> Ptr;
};

class Matimage : public IDataObject
{
public:
    typedef std::shared_ptr<Matimage> Ptr;

    Matimage(const cv::Mat &image)
    {
        matimage_ = std::make_shared<cv::Mat>(image);
    }

//    Matimage(bool create)
//    {
//        if (create)
//        {
//            matimage_ = new cv::Mat();
//        }
//    }

    std::shared_ptr<cv::Mat> matimage_;
};

class InputInfo : public IDataObject {
public:
    typedef std::shared_ptr<InputInfo> Ptr;

    cv::Rect2f bbox_;
    float conf_;
    int frame_num_;
    int width_;
    int height_;
    bool flag_;
    int module_;
};

class IDataStorage : public IInterface
{
public:
    typedef std::shared_ptr<IDataStorage> Ptr;

    /**
     \fn	virtual void IDataStorage::Set(std::string key, IDataObject::Ptr object) = 0;

     \brief	Add object into storage with key

     \author	Sergey
     \date	8/11/2015

     \param	key   	Key
     \param	object	Object to add. The object should be contained in IDataObject
     */

    void Set(std::string key, IDataObject::Ptr object){
        m_storage_[key] = object;
    }

    IDataObject::Ptr Get(std::string key){
        return m_storage_[key];
    }

    void Remove(std::string key){
        auto it = m_storage_.find(key);
        if (it != m_storage_.end())
        {
            m_storage_.erase(it);
        }
    };

    template<class T>
    std::shared_ptr<T> GetAndCast(std::string key)
    {
        IDataObject::Ptr obj = Get(key);
        return std::dynamic_pointer_cast<T>(obj);
    }

    template<class T>
    std::shared_ptr<T> GetAndCastNotNull(std::string key, std::string message = "Got nullptr from storage")
    {
        std::shared_ptr<T> ptr = GetAndCast<T>(key);
        if (ptr.get() == nullptr)
        {
            throw HPEException(message);
        }
        return ptr;
    }

private:
    std::map<std::string, IDataObject::Ptr> m_storage_;

};

#endif //PROJECT_STORAGE_H
