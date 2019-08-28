//
// Created by marco on 19-7-12.
//

#ifndef PROJECT_SIAMTRACKER_H
#define PROJECT_SIAMTRACKER_H

#include "opencv2/opencv.hpp"
#include <vector>
#include <Eigen/Core>

#include <iostream>

#include "utils.h"

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>

class SiamTracker{
public:
    SiamTracker(){
        nett_ = torch::jit::load("/home/marco/PycharmProjects/DaSiamRPN-modify/code/DaSiamRPN_Temple_gpu.pt");
        netf_ = torch::jit::load("/home/marco/PycharmProjects/DaSiamRPN-modify/code/DaSiamRPN_Forward_gpu.pt");
        device_type_ = torch::kCUDA;
    };

    bool Init(cv::Mat &img, cv::Point2f &pos, cv::Size &size);

    bool detect(cv::Mat &img);

public:
    int im_w_;
    int im_h_;
    TrackerConfig tconf_;
    // add the net instance here.
    std::shared_ptr<torch::jit::script::Module> nett_;
    std::shared_ptr<torch::jit::script::Module> netf_;
    torch::Tensor r1_kernel_;
    torch::Tensor init_r1_kernel_;
    torch::Tensor cls1_kernel_;
    torch::Tensor init_cls1_kernel_;
    torch::Tensor delta_;
    torch::Tensor score_;

    float best_score_;

    std::vector<float> avg_chans_;
    Eigen::MatrixXf window_;
    cv::Point2f target_pos_;
    cv::Size2f target_sz_;

    torch::DeviceType device_type_;

private:
    Eigen::MatrixXf generateAnchor();

    cv::Mat get_subwindow_tracking(const cv::Mat &img, const cv::Point2f &pos, const int original_sz, const bool init_flag);

    Eigen::MatrixXf HanningWindow(const int size);

    int tracker_eval(cv::Mat &x_crop, float scale_z);

    bool img2torch(const cv::Mat &img, at::Tensor &tensor_img);
};

#endif //PROJECT_SIAMTRACKER_H
