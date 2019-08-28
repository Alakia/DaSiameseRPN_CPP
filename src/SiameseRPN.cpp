//
// Created by marco on 19-8-27.
//

#include "SiameseRPN.h"
#include <string>

void SiameseRPNProcessor::Init(IDataStorage::Ptr dataStorage) {

    InputInfo::Ptr pinfo = std::dynamic_pointer_cast<InputInfo>(dataStorage->Get("info")) ;
    Matimage::Ptr pmat = std::dynamic_pointer_cast<Matimage>(dataStorage->Get("init_image"));

    cv::Size initsize(pinfo->bbox_.width, pinfo->bbox_.height);
    cv::Point2f initpos(pinfo->bbox_.x + pinfo->bbox_.width / 2, pinfo->bbox_.y + pinfo->bbox_.height / 2);

    pinfo->flag_ = siamtracker_.Init(*pmat->matimage_, initpos, initsize);

    return;
}

void SiameseRPNProcessor::Process(IDataStorage::Ptr dataStorage) {

    InputInfo::Ptr pinfo = std::dynamic_pointer_cast<InputInfo>(dataStorage->Get("info"));
    Matimage::Ptr pframe = std::dynamic_pointer_cast<Matimage>(dataStorage->Get("frame"));
    pinfo->flag_ = siamtracker_.detect(*pframe->matimage_);

    pinfo->bbox_.x = siamtracker_.target_pos_.x - siamtracker_.target_sz_.width / 2;
    pinfo->bbox_.y = siamtracker_.target_pos_.y - siamtracker_.target_sz_.height / 2;
    pinfo->bbox_.width = siamtracker_.target_sz_.width;
    pinfo->bbox_.height = siamtracker_.target_sz_.height;

    return;
}