//
// Created by marco on 19-7-12.
//

#ifndef PROJECT_UTILS_H
#define PROJECT_UTILS_H

#include "opencv2/opencv.hpp"
#include <vector>
//#include <torch/torch.h>
//#include <ATen/ATen.h>
//#include <torch/script.h>

#define T_CONF_WIN_COS 0x01

class TrackerConfig{
public:
    TrackerConfig(){};

    int windowing_;
    int exemplar_size_;
    int instance_size_;
    int total_stride_;
    int score_size_;
    float context_amount_;
    std::vector<float> ratios_;
    std::vector<float> scales_;
    int anchor_num_;
    Eigen::MatrixXf anchor_; // need to change to anchor type.
    float penalty_k_;
    float window_influence_;
    float lr_;
    bool adaptive_;

    int Init(){
        windowing_ = T_CONF_WIN_COS;
        exemplar_size_ = 127;
        instance_size_ = 271;
        total_stride_ = 8;
        score_size_ = (instance_size_ - exemplar_size_) / total_stride_ + 1;
        context_amount_ = 0.5;
        ratios_ = {0.33, 0.5, 1, 2, 3};
        scales_ = {8};
        anchor_num_ = ratios_.size() * scales_.size();
        penalty_k_ = 0.055;
        window_influence_ = 0.42;
        lr_ = 0.295;
        adaptive_ = false;
        return 0;
    };

};

#endif //PROJECT_UTILS_H
