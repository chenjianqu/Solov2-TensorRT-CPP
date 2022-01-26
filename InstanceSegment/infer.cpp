/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/



#include <iostream>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

#include "infer.h"
#include "parameters.h"
#include "utils.h"


Infer::Infer()
{
    ///注册预定义的和自定义的插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");
    InfoLog("Read model param");
    std::string model_str;
    if(std::ifstream ifs(Config::kDetectorSerializePath);ifs.is_open()){
        while(ifs.peek() != EOF){
            std::stringstream ss;
            ss<<ifs.rdbuf();
            model_str.append(ss.str());
        }
        ifs.close();
    }
    else{
        auto msg=fmt::format("Can not open the DETECTOR_SERIALIZE_PATH:{}",Config::kDetectorSerializePath);
        throw std::runtime_error(msg);
    }
    InfoLog("createInferRuntime");

    ///创建runtime
    runtime_=std::unique_ptr<nvinfer1::IRuntime,InferDeleter>(
            nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    InfoLog("deserializeCudaEngine");

    ///反序列化模型
    engine_=std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(model_str.data(), model_str.size()) , InferDeleter());
    InfoLog("createExecutionContext");

    ///创建执行上下文
    context_=std::unique_ptr<nvinfer1::IExecutionContext,InferDeleter>(
            engine_->createExecutionContext());
    if(!context_){
        throw std::runtime_error("can not create context");
    }

    ///创建输入输出的内存
    buffer_ = std::make_shared<MyBuffer>(*engine_);

    Config::input_h=buffer_->dims[0].d[2];
    Config::input_w=buffer_->dims[0].d[3];
    Config::input_c=buffer_->dims[0].d[1];

    pipeline_=std::make_shared<Pipeline>();
    solo_ = std::make_shared<Solov2>();

    //cv::Mat warn_up_input(cv::Size(1226,370),CV_8UC3,cv::Scalar(128));
    cv::Mat warn_up_input = cv::imread(Config::kWarnUpImagePath);

    if(warn_up_input.empty()){
        ErrorLog("Can not open warn up image:{}", Config::kWarnUpImagePath);
        return;
    }

    cv::resize(warn_up_input,warn_up_input,cv::Size(Config::kImageWidth, Config::kImageHeight));

    WarnLog("warn up model,path:{}",Config::kWarnUpImagePath);

    //[[maybe_unused]] auto result = forward(warn_up_input);

    [[maybe_unused]] torch::Tensor mask_tensor;
    [[maybe_unused]] std::vector<InstInfo> insts_info;
    Forward(warn_up_input, mask_tensor, insts_info);

    //if(insts_info.empty())throw std::runtime_error("model not init");

    InfoLog("infer init finished");
}



void Infer::Forward(cv::Mat &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts)
{
    TicToc t_all,tt;
    ///将图片数据复制到输入buffer,同时实现了图像的归一化
    buffer_->gpu_buffer[0] = pipeline_->SetInputTensorCuda(img);
    InfoLog("Forward prepare:{} ms", tt.TocThenTic());
    ///推断
    context_->enqueue(kBatchSize, buffer_->gpu_buffer, buffer_->stream, nullptr);
    InfoLog("Forward enqueue:{} ms", tt.TocThenTic());
    ///将输出数据构建为张量
    std::vector<torch::Tensor> outputs;
    buffer_->CudaToTensor(outputs);
    InfoLog("Forward CudaToTensor:{} ms", tt.TocThenTic());
    ///后处理
    solo_->GetSegTensor(outputs, pipeline_->img_info, mask_tensor, insts);
    InfoLog("Forward GetSegTensor:{} ms", tt.TocThenTic());
    InfoLog("Forward inst number:{}",insts.size());

    infer_time_ = t_all.Toc();
}

void Infer::VisualizeResult(cv::Mat &input,cv::Mat &mask,std::vector<InstInfo> &insts)
{
    if(mask.empty()){
        cv::imshow("test",input);
        cv::waitKey(1);
    }
    else{
        mask = pipeline_->ProcessMask(mask, insts);

        cv::Mat image_test;
        cv::add(input,mask,image_test);
        for(auto &inst : insts){
            if(inst.prob < 0.2)
                continue;
            inst.name = cfg::CocoLabelVector[inst.label_id];
            cv::Point2i center = (inst.min_pt + inst.max_pt)/2;
            std::string show_text = fmt::format("{} {:.2f}",inst.name,inst.prob);
            cv::putText(image_test,show_text,center,CV_FONT_HERSHEY_SIMPLEX,0.8,
                        cv::Scalar(255,0,0),2);
            cv::rectangle(image_test, inst.min_pt, inst.max_pt, cv::Scalar(255, 0, 0), 2);
        }
        cv::putText(image_test, fmt::format("{:.2f} ms", infer_time_),
                    cv::Point2i(20, 20), CV_FONT_HERSHEY_SIMPLEX, 2,
                    cv::Scalar(0, 255, 255));
        cv::imshow("test",image_test);
        cv::waitKey(1);
    }
}
