/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pipeline.h"

#include <iostream>
#include <opencv2/cudaimgproc.hpp>


using namespace torch::indexing;
using InterpolateFuncOptions=torch::nn::functional::InterpolateFuncOptions;


template<typename ImageType>
std::tuple<float,float> Pipeline::GetXYWHS(const ImageType &img)
{
    img_info.origin_h = img.rows;
    img_info.origin_w = img.cols;

    int w, h, x, y;
    float r_w = Config::input_w / (img.cols * 1.0f);
    float r_h = Config::input_h / (img.rows * 1.0f);
    if (r_h > r_w) {
        w = Config::input_w;
        h = r_w * img.rows;
        if(h%2==1)h++;//这里确保h为偶数，便于后面的使用
        x = 0;
        y = (Config::input_h - h) / 2;
    } else {
        w = r_h* img.cols;
        if(w%2==1)w++;
        h = Config::input_h;
        x = (Config::input_w - w) / 2;
        y = 0;
    }

    img_info.rect_x = x;
    img_info.rect_y = y;
    img_info.rect_w = w;
    img_info.rect_h = h;

    return {r_h,r_w};
}


void* Pipeline::SetInputTensorCuda(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = GetXYWHS(img);

    cv::Mat img_float;
    img.convertTo(img_float,CV_32FC3);
    sgLogger->debug("SetInputTensorCuda convertTo: {} ms", tt.TocThenTic());
    input_tensor = torch::from_blob(img_float.data, {img_info.origin_h, img_info.origin_w , 3 }, torch::kFloat32).to(torch::kCUDA);


    sgLogger->debug("SetInputTensorCuda from_blob:{} {} ms", Dims2Str(input_tensor.sizes()), tt.TocThenTic());

    ///bgr->rgb
    input_tensor = torch::cat({
        input_tensor.index({"...",2}).unsqueeze(2),
        input_tensor.index({"...",1}).unsqueeze(2),
        input_tensor.index({"...",0}).unsqueeze(2)
        },2);
    sgLogger->debug("SetInputTensorCuda bgr->rgb:{} {} ms", Dims2Str(input_tensor.sizes()), tt.TocThenTic());

    ///hwc->chw
    input_tensor = input_tensor.permute({2,0,1});
    sgLogger->debug("SetInputTensorCuda hwc->chw:{} {} ms", Dims2Str(input_tensor.sizes()), tt.TocThenTic());

    ///norm
    static torch::Tensor mean_t=torch::from_blob(kSoloImageMean, {3, 1, 1}, torch::kFloat32).to(torch::kCUDA).
            expand({3, img_info.origin_h, img_info.origin_w});
    static torch::Tensor std_t=torch::from_blob(kSoloImageStd, {3, 1, 1}, torch::kFloat32).to(torch::kCUDA).
            expand({3, img_info.origin_h, img_info.origin_w});
    input_tensor = ((input_tensor-mean_t)/std_t);
    sgLogger->debug("SetInputTensorCuda norm:{} {} ms", Dims2Str(input_tensor.sizes()), tt.TocThenTic());

    ///resize
    static auto options=InterpolateFuncOptions().mode(torch::kBilinear).align_corners(true);
    options=options.size(std::vector<int64_t>({img_info.rect_h, img_info.rect_w}));
    input_tensor = torch::nn::functional::interpolate(input_tensor.unsqueeze(0),options).squeeze(0);
    sgLogger->debug("SetInputTensorCuda resize:{} {} ms", Dims2Str(input_tensor.sizes()), tt.TocThenTic());

    ///拼接图像边缘
    static auto op = torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32);
    static cv::Scalar mag_color(kSoloImageMean[2], kSoloImageMean[1], kSoloImageMean[0]);
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_w = Config::input_w;
        int cat_h = (Config::input_h - img_info.rect_h) / 2;
        torch::Tensor cat_t = torch::zeros({3,cat_h,cat_w},op);
        input_tensor = torch::cat({cat_t,input_tensor,cat_t},1);
    } else {
        int cat_w= (Config::input_w - img_info.rect_w) / 2;
        int cat_h=Config::input_h;
        torch::Tensor cat_t = torch::zeros({3,cat_h,cat_w},op);
        input_tensor = torch::cat({cat_t,input_tensor,cat_t},2);
    }
    sgLogger->debug("SetInputTensorCuda cat:{} {} ms", Dims2Str(input_tensor.sizes()), tt.TocThenTic());

    input_tensor = input_tensor.contiguous();
    sgLogger->debug("SetInputTensorCuda contiguous:{} {} ms", Dims2Str(input_tensor.sizes()), tt.TocThenTic());

    return input_tensor.data_ptr();
}


cv::Mat Pipeline::ProcessPad(cv::Mat &img)
{
    TicToc tt;

    GetXYWHS(img);

    //将img resize为(INPUT_W,INPUT_H)
    cv::Mat re;
    cv::resize(img, re, cv::Size(img_info.rect_w, img_info.rect_h) , 0, 0, cv::INTER_LINEAR);

    sgLogger->debug("ProcessPad resize:{} ms", tt.TocThenTic());

    //将图片复制到out中
    static cv::Scalar mag_color(kSoloImageMean[2], kSoloImageMean[1], kSoloImageMean[0]);
    cv::Mat out(Config::input_h, Config::input_w, CV_8UC3, mag_color);
    re.copyTo(out(cv::Rect(img_info.rect_x, img_info.rect_y, re.cols, re.rows)));

    sgLogger->debug("ProcessPad copyTo out:{} ms", tt.TocThenTic());

    return out;
}

void Pipeline::SetBufferWithNorm(const cv::Mat &img, float *buffer)
{
    //assert(Config::inputH==img.rows);
    //assert(Config::inputW==img.cols);
    int i = 0,b_cnt=0;
    auto rows = std::min(img.rows,Config::input_h);
    auto cols = std::min(img.cols,Config::input_w);
    for (int row = 0; row < rows; ++row) {
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < cols; ++col) {
            buffer[b_cnt * 3 * Config::input_h * Config::input_w + i] = (uc_pixel[2] - kSoloImageMean[0]) / kSoloImageStd[0];
            buffer[b_cnt * 3 * Config::input_h * Config::input_w + i + Config::input_h * Config::input_w] =
                    (uc_pixel[1] - kSoloImageMean[1]) / kSoloImageStd[1];
            buffer[b_cnt * 3 * Config::input_h * Config::input_w + i + 2 * Config::input_h * Config::input_w] =
                    (uc_pixel[0] - kSoloImageMean[2]) / kSoloImageStd[2];
            uc_pixel += 3;
            ++i;
        }
    }

}


cv::Mat Pipeline::ProcessMask(cv::Mat &mask, std::vector<InstInfo> &insts)
{
    cv::Mat rect_img = mask(cv::Rect(img_info.rect_x, img_info.rect_y, img_info.rect_w, img_info.rect_h));
    cv::Mat out;
    cv::resize(rect_img, out, cv::Size(img_info.origin_w, img_info.origin_h), 0, 0, cv::INTER_LINEAR);

    ///调整包围框
    float factor_x = out.cols *1.f / rect_img.cols;
    float factor_y = out.rows *1.f / rect_img.rows;
    for(auto &inst : insts){
        inst.min_pt.x -= img_info.rect_x;
        inst.min_pt.y -= img_info.rect_y;
        inst.max_pt.x -= img_info.rect_x;
        inst.max_pt.y -= img_info.rect_y;

        inst.min_pt.x *= factor_x;
        inst.min_pt.y *= factor_y;
        inst.max_pt.x *= factor_x;
        inst.max_pt.y *= factor_y;
    }


    return out;
}


