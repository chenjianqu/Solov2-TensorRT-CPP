/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef INSTANCE_SEGMENT_INFER_H
#define INSTANCE_SEGMENT_INFER_H

#include <optional>
#include <memory>
#include <NvInfer.h>
#include "TensorRtSample/common.h"
#include "pipeline.h"
#include "solo.h"
#include "buffer.h"


struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const{
        if (obj)
            obj->destroy();
    }
};

class Infer {
public:
    using Ptr = std::shared_ptr<Infer>;
    Infer();
    void Forward(cv::Mat &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts);
    void VisualizeResult(cv::Mat &input,cv::Mat &mask,std::vector<InstInfo> &insts);
private:
    MyBuffer::Ptr buffer_;
    Pipeline::Ptr pipeline_;
    Solov2::Ptr solo_;
    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext, InferDeleter> context_;
    double infer_time_{0};
};


#endif
