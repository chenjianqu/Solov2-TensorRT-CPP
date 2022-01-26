/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>
#include <opencv2/video.hpp>
#include "InstanceSegment/infer.h"
#include "InstanceSegment/parameters.h"
#include "InstanceSegment/utils.h"

cv::Mat DrawSegment(cv::Mat &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts)
{
    cv::Mat img_show = img.clone();
    if(!insts.empty()){
        auto mask_size=cv::Size(img_show.cols, img_show.rows);
        mask_tensor = mask_tensor.to(torch::kInt8).abs().clamp(0,1);
        ///计算合并的mask
        auto merge_tensor = (mask_tensor.sum(0).clamp(0,1)*255).to(torch::kUInt8).to(torch::kCPU);
        auto mask = cv::Mat(mask_size,CV_8UC1,merge_tensor.data_ptr()).clone();
        cv::cvtColor(mask,mask,CV_GRAY2BGR);
        cv::scaleAdd(mask, 0.5, img_show, img_show);

        for(auto &inst: insts){
            auto color = GetRandomColor();
            DrawText(img_show, fmt::format("{}:{:.2f}", Config::CocoLabelVector[inst.label_id], inst.prob),
                     color, inst.rect.tl());
            cv::rectangle(img_show, inst.min_pt, inst.max_pt, color, 1);
        }
    }
    return img_show;
}

int main(int argc, char **argv)
{
    if(argc != 2){
        cerr<<"please input: [config file]"<<endl;
        return 1;
    }
    string config_file = argv[1];
    fmt::print("config_file:{}\n",argv[1]);

    Infer::Ptr infer;

    try{
        Config cfg(config_file);
        infer.reset(new Infer);
    }
    catch(std::runtime_error &e){
        cerr<<e.what()<<endl;
        return -1;
    }

    ///主线程，图像分割
    TicToc ticToc;

    fmt::print("Read Image:{}\n",Config::kWarnUpImagePath);
    cv::Mat img0=cv::imread(Config::kWarnUpImagePath);
    if(img0.empty()){
        cerr << "Read:" << Config::kWarnUpImagePath << " failure" << endl;
        return -1;
    }
    ticToc.Tic();
    torch::Tensor mask_tensor;
    std::vector<InstInfo> insts_info;
    infer->Forward(img0, mask_tensor, insts_info);

    fmt::print("insts_info.size():{}\n",insts_info.size());
    fmt::print("infer time:{} ms\n", ticToc.Toc());

    cv::Mat img_show=DrawSegment(img0,mask_tensor,insts_info);

    cv::imshow("raw", img_show);
    cv::waitKey(0);

    return 0;
}
