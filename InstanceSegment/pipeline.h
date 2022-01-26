/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef INSTANCE_SEGMENT_PIPELINE_H
#define INSTANCE_SEGMENT_PIPELINE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torchvision/vision.h>

#include "parameters.h"
#include "utils.h"

class Pipeline {
public:
    using Ptr=std::shared_ptr<Pipeline>;
    Pipeline(){

    }

    template<typename ImageType>
    std::tuple<float,float> GetXYWHS(const ImageType &img);
    void* SetInputTensorCuda(cv::Mat &img);
    void SetBufferWithNorm(const cv::Mat &img, float *buffer);
    cv::Mat ProcessPad(cv::Mat &img);
    cv::Mat ProcessMask(cv::Mat &mask, std::vector<InstInfo> &insts);

    ImageInfo img_info;
    torch::Tensor input_tensor;
private:
};


#endif //
