/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef INSTANCE_SEGMENT_SOLO_H
#define INSTANCE_SEGMENT_SOLO_H

#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torchvision/vision.h>

#include "parameters.h"
#include "utils.h"

class Solov2 {
public:
    using Ptr=std::shared_ptr<Solov2>;
    Solov2(){
        size_trans_=torch::from_blob(kSoloNumGrids.data(), {int(kSoloNumGrids.size())}, torch::kFloat).clone();
        size_trans_=size_trans_.pow(2).cumsum(0);
    }
    static torch::Tensor MatrixNMS(torch::Tensor &seg_masks,torch::Tensor &cate_labels,torch::Tensor &cate_scores,torch::Tensor &sum_mask);
    void GetSegTensor(std::vector<torch::Tensor> &outputs, ImageInfo& img_info, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts);
private:
    torch::Tensor size_trans_;
};


#endif
