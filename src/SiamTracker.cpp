//
// Created by marco on 19-7-12.
//

#include "SiamTracker.h"

bool SiamTracker::Init(cv::Mat &img, cv::Point2f &pos, cv::Size &size) {
    target_pos_ = pos;
    target_sz_ = size;
    tconf_.Init();
    im_w_ = img.cols;
    im_h_ = img.rows;
    if(tconf_.adaptive_){
        if((target_sz_.width * target_sz_.height) / float(im_w_ * im_h_) < 4e-3){
            tconf_.instance_size_ = 287;
        }
        else{
            tconf_.instance_size_ = 271;
        }
        tconf_.score_size_ = (tconf_.instance_size_ - tconf_.score_size_) / tconf_.total_stride_ + 1;
    }

    tconf_.anchor_ = generateAnchor();
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    cv::Mat meanMat, stdMat;
    for(int i = 0; i < 3; i++){
        cv::meanStdDev(channels[i], meanMat, stdMat);
        float m = (float)(meanMat.at<double>(0, 0));
        avg_chans_.push_back(m);
    }

    float wc_z = target_sz_.width + tconf_.context_amount_ * (target_sz_.width + target_sz_.height);
    float hc_z = target_sz_.height + tconf_.context_amount_ * (target_sz_.width + target_sz_.height);
    int s_z = (int)(sqrt(wc_z * hc_z));
    cv::Mat z_crop = get_subwindow_tracking(img, pos, s_z, true);
    torch::Tensor proc_z_crop;

    torch::Device device(device_type_, 0);

    img2torch(z_crop, proc_z_crop);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(proc_z_crop.to(device));

    auto output = nett_->forward(inputs).toTuple();
    r1_kernel_ = output->elements()[0].toTensor();
    cls1_kernel_ = output->elements()[1].toTensor();

    init_r1_kernel_ = r1_kernel_;
    init_cls1_kernel_ = cls1_kernel_;

    r1_kernel_ = r1_kernel_.to(device);
    cls1_kernel_ = cls1_kernel_.to(device);

    window_ = HanningWindow(tconf_.score_size_);

    return true;
}

Eigen::MatrixXf SiamTracker::generateAnchor() {
    int anchor_nums = (int)tconf_.ratios_.size() * (int)tconf_.scales_.size();
    Eigen::Matrix<float, Eigen::Dynamic, 4> anchor(anchor_nums, 4);
    int size = tconf_.total_stride_ * tconf_.total_stride_;
    int count = 0;
    for(auto &ratio : tconf_.ratios_){
        float ws = (int)(sqrt((float)size / ratio));
        float hs = (int)(ws * ratio);
        for(auto &scale : tconf_.scales_){
            float wws = ws * scale;
            float hhs = hs * scale;
            anchor(count, 0) = 0;
            anchor(count, 1) = 0;
            anchor(count, 2) = wws;
            anchor(count, 3) = hhs;
            count++;
        }
    }

    int ssz = tconf_.score_size_;
    int ssz2 = ssz * ssz;
    int anchor_num_out = anchor_nums * ssz2;
    Eigen::Matrix<float, Eigen::Dynamic, 4> anchor_out(anchor_num_out, 4);
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < ssz2; j++){
            anchor_out.row(i * ssz2 + j) = anchor.row(i);
        }
    }

    float ori = - ((float)tconf_.score_size_ / 2) * (float)tconf_.total_stride_;
    Eigen::MatrixXf xx(tconf_.score_size_, tconf_.score_size_);
    Eigen::MatrixXf yy(tconf_.score_size_, tconf_.score_size_);
    for(int i = 0; i < tconf_.score_size_; i++){
        xx(0, i) = ori + i * tconf_.total_stride_;
    }
    for(int j = 1; j < tconf_.score_size_; j++){
        xx.row(j) = xx.row(0);
    }
    yy = xx.transpose();

    Eigen::VectorXf xv = xx.row(0).transpose();
    int extend_size = tconf_.score_size_ * tconf_.score_size_ * anchor_nums;
    Eigen::VectorXf xxout(extend_size, 1);
    for(int i = 0; i < tconf_.score_size_ * anchor_nums; i++){
        for(int j = 0; j < tconf_.score_size_; j++){
            xxout(tconf_.score_size_ * i + j, 0) = xv(j, 0);
        }
    }

    Eigen::VectorXf yyout(extend_size, 1);
    Eigen::VectorXf yv = yy.col(0);
    for(int i = 0; i < anchor_nums; i++){
        for(int j = 0; j < tconf_.score_size_; j++){
            for(int k = 0; k < tconf_.score_size_; k++){
                yyout(i * ssz2 + tconf_.score_size_ * j + k, 0) = yv(j, 0);
            }
        }
    }

    anchor_out.col(0) = xxout;
    anchor_out.col(1) = yyout;

    return anchor_out;
}

cv::Mat SiamTracker::get_subwindow_tracking(const cv::Mat &img, const cv::Point2f &pos, const int original_sz, const bool init_flag) {
    int sz = original_sz;
    int r = img.rows;
    int c = img.cols;
    int k = img.channels();
    int cc = (original_sz + 1) / 2;
    int context_xmin = (int)(pos.x - cc);
    int context_xmax = context_xmin + sz - 1;
    int context_ymin = (int)(pos.y -cc);
    int context_ymax = context_ymin + sz - 1;
    int left_pad = (int)(std::max(0, -context_xmin));
    int top_pad = (int)(std::max(0, -context_ymin));
    int right_pad = (int)(std::max(0, context_xmax - c + 1));
    int bottom_pad = (int)(std::max(0, context_ymax - r + 1));

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;

    cv::Mat im_out;
    if(left_pad != 0 || top_pad != 0 || right_pad != 0 || bottom_pad != 0){//todo: add the padding property.
        std::cout << "property needs to be added." << std::endl;
        return im_out;
    }
    else{
        im_out = img(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 2, context_ymax - context_ymin + 2)); // pitfall here. x and y index mixed in python cv2.
    }

        if(init_flag)
            cv::resize(im_out, im_out, cv::Size(tconf_.exemplar_size_, tconf_.exemplar_size_));
        else
            cv::resize(im_out, im_out, cv::Size(tconf_.instance_size_, tconf_.instance_size_));

    return im_out;
}

Eigen::MatrixXf SiamTracker::HanningWindow(const int size) {
    Eigen::MatrixXf xhan(size, 1);
    Eigen::MatrixXf yhan(size, 1);
    for(int i = 0; i < size; i++){
        xhan(i, 0) = 0.5 - 0.5 * cos(2.0 * 3.1415926535 * i / (size - 1));
        yhan(i, 0) = 0.5 - 0.5 * cos(2.0 * 3.1415926535 * i / (size - 1));
    }
    Eigen::MatrixXf output(size, size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            output(i, j) = xhan(i, 0) * yhan(j, 0);
        }
    }

    Eigen::Matrix<float, 1, Eigen::Dynamic> output_flat(1, size * size * 5);
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < size; j++){
            for(int k = 0; k < size; k++){
                output_flat(0, i * size * size + j * size + k) = output(j, k);
            }
        }
    }

    return output_flat;
}

bool SiamTracker::detect(cv::Mat &img) {
    float wc_z = target_sz_.height + tconf_.context_amount_ * (target_sz_.height + target_sz_.width);
    float hc_z = target_sz_.width + tconf_.context_amount_ * (target_sz_.height + target_sz_.width);// carefully check this.

    float s_z = sqrt(wc_z * hc_z);
    float scale_z = tconf_.exemplar_size_ / s_z;
    float d_search = (tconf_.instance_size_ - tconf_.exemplar_size_) / 2;
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;

    // extract scaled crops for search region x at previous target position
    cv::Mat x_crop = get_subwindow_tracking(img, target_pos_, (int)(s_x), false);

    int flag = tracker_eval(x_crop, scale_z);

    if(flag == 1){
        r1_kernel_ = init_r1_kernel_;
        cls1_kernel_ = init_cls1_kernel_;
        return false;
    }
    else {
        float minposx = std::fminf((float) im_w_, target_pos_.x);
        float minposy = std::fminf((float) im_h_, target_pos_.y);
        target_pos_.x = std::fmaxf(minposx, 0.0);
        target_pos_.y = std::fmaxf(minposy, 0.0);
        target_sz_.width = std::fmaxf(10.0, std::fminf((float) im_w_, target_sz_.width));
        target_sz_.height = std::fmaxf(10.0, std::fminf((float) im_h_, target_sz_.height));

        return true;
    }
}

int SiamTracker::tracker_eval(cv::Mat &x_crop, float scale_z) {
    cv::Size2f scaled_target_size;
    scaled_target_size.width = target_sz_.width * scale_z;
    scaled_target_size.height = target_sz_.height * scale_z;

    torch::Tensor proc_x_crop;
    img2torch(x_crop, proc_x_crop);
    std::vector<torch::jit::IValue> inputs;
    torch::Device device(device_type_, 0);

    inputs.emplace_back(proc_x_crop.to(device));
    inputs.emplace_back(r1_kernel_);

    inputs.emplace_back(cls1_kernel_);

    auto output = netf_->forward(inputs).toTuple();

    delta_ = output->elements()[0].toTensor();
    score_ = output->elements()[1].toTensor();

    auto permute_delta = delta_.permute({1, 2, 3, 0}).contiguous().view({4, -1});
    auto permute_score = score_.permute({1, 2, 3, 0}).contiguous().view({2, -1});
    auto softmax_score = torch::softmax(permute_score, 0);

    auto delta_shape_0 = permute_delta.size(0);
    auto delta_shape_1 = permute_delta.size(1);
    auto score_shape_0 = permute_score.size(0);
    auto score_shape_1 = permute_score.size(1);
    auto softmax_shape_0 = softmax_score.size(0);
    auto softmax_shape_1 = softmax_score.size(1);

    permute_delta = permute_delta.to(torch::kCPU);
    softmax_score = softmax_score.to(torch::kCPU);

    Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor> eigen_delta(4, delta_shape_1);
    Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor> eigen_score(2, score_shape_1);

    std::memcpy(eigen_delta.data(), permute_delta.data<float>(), permute_delta.numel() * sizeof(float));

    std::memcpy(eigen_score.data(), softmax_score.data<float>(), softmax_score.numel() * sizeof(float));

    for(int i = 0; i < delta_shape_1; i++){
        eigen_delta(0, i) = eigen_delta(0, i) * tconf_.anchor_(i, 2) + tconf_.anchor_(i, 0);
        eigen_delta(1, i) = eigen_delta(1, i) * tconf_.anchor_(i, 3) + tconf_.anchor_(i, 1);
    }

    Eigen::VectorXf exp_delta(delta_shape_1);
    for(int i = 0; i < delta_shape_1; i++){
        eigen_delta(2, i) = std::exp(eigen_delta(2, i)) * tconf_.anchor_(i, 2);
    }

    for(int i = 0; i < delta_shape_1; i++){
        eigen_delta(3, i) = std::exp(eigen_delta(3, i)) * tconf_.anchor_(i, 3);
    }

    Eigen::Matrix<float, 1, Eigen::Dynamic> s_c(1, delta_shape_1);
    Eigen::Matrix<float, 1, Eigen::Dynamic> r_c(1, delta_shape_1);
    for(int i = 0; i < delta_shape_1; i++){
        float sz1 = sqrt(( eigen_delta(2, i) + ( eigen_delta(2, i) + eigen_delta(3, i)) * 0.5 )  * ( eigen_delta(3, i) + ( eigen_delta(2, i) + eigen_delta(3, i)) * 0.5 ));
        float sz2 = sqrt(( scaled_target_size.width + ( scaled_target_size.width + scaled_target_size.height) * 0.5  ) * ( scaled_target_size.height + ( scaled_target_size.width + scaled_target_size.height) * 0.5 ));
        float sz_ratio = sz1 / sz2;
        float sz_out = sz_ratio > (1 / sz_ratio) ? sz_ratio : (1 / sz_ratio);
        s_c(0, i) = sz_out;
    }
    for(int i = 0; i < delta_shape_1; i++){
        float cm1 = (float)scaled_target_size.width / (float)scaled_target_size.height;
        float cm2 = eigen_delta(2, i)/ eigen_delta(3, i);
        float cm_ratio = cm1 / cm2;
        float cm_out = cm_ratio > (1 / cm_ratio) ? cm_ratio : (1 / cm_ratio);
        r_c(0, i) = cm_out;
    }

    Eigen::Matrix<float, 1, Eigen::Dynamic> penalty(1, delta_shape_1);
    Eigen::Matrix<float, 1, Eigen::Dynamic> pscore(1, delta_shape_1);
    for(int i = 0; i < delta_shape_1; i++) {
        penalty(0, i) = std::exp( -(r_c(0, i) * s_c(0, i) - 1) * tconf_.penalty_k_ );
        pscore(0, i) = penalty(0, i) * eigen_score(1, i);
        pscore(0, i) = pscore(0, i) * (1 - tconf_.window_influence_) + window_(0, i) * tconf_.window_influence_;
    }

    int maxind = -1;
    pscore.maxCoeff(&maxind);

    Eigen::Vector4f target;
    target[0] = eigen_delta(0, maxind) / scale_z;
    target[1] = eigen_delta(1, maxind) / scale_z;
    target[2] = eigen_delta(2, maxind) / scale_z;
    target[3] = eigen_delta(3, maxind) / scale_z;

    scaled_target_size.width /= scale_z;
    scaled_target_size.height /= scale_z;
    float lr = penalty(0, maxind) * eigen_score(1, maxind) * tconf_.lr_;

    float res_x = target[0] + target_pos_.x;
    float res_y = target[1] + target_pos_.y;

    float res_w = scaled_target_size.width * (1 - lr) + target[2] * lr;
    float res_h = scaled_target_size.height * (1 - lr) + target[3] * lr;

    if(pscore(0, maxind) <= 0.83){
        std::cout << "track failed." << std::endl;
        std::cout << "pscore score = " << pscore(0, maxind) << std::endl;
        return 1;
    }
    else {
        target_pos_.x = res_x;
        target_pos_.y = res_y;
        target_sz_.width = res_w;
        target_sz_.height = res_h;
        best_score_ = eigen_score(1, maxind);
        std::cout << "best score = " << best_score_ << std::endl;
        std::cout << "pscore score = " << pscore(0, maxind) << std::endl;

        return 0;
    }
}

bool SiamTracker::img2torch(const cv::Mat &img, at::Tensor &tensor_img) {
    if(img.empty()){
        return false;
    }

    tensor_img = torch::from_blob(img.data, {img.rows, img.cols, 3}, at::kByte).clone();

    tensor_img = tensor_img.permute({2, 0, 1}); //todo: check the dimension correctness.
    tensor_img = tensor_img.unsqueeze(0);

    tensor_img = tensor_img.to(at::kFloat);

    return true;
}
