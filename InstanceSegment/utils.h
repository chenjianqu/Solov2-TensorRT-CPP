/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef INSTANCE_SEGMENT_UTILS_H
#define INSTANCE_SEGMENT_UTILS_H

#include <string>
#include <vector>
#include <chrono>
#include <random>

#include <spdlog/logger.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <NvInfer.h>

#include "parameters.h"


class TicToc{
public:
    TicToc(){
        Tic();
    }

    void Tic(){
        start_ = std::chrono::system_clock::now();
    }

    double Toc(){
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_ - start_;
        return elapsed_seconds.count() * 1000;
    }

    double TocThenTic(){
        auto t= Toc();
        Tic();
        return t;
    }

    void TocPrintTic(const char* str){
        cout << str << ":" << Toc() << " ms" << endl;
        Tic();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
};


struct ImageInfo{
    int origin_h,origin_w;
    ///图像的裁切信息
    int rect_x, rect_y, rect_w, rect_h;
};


struct InstInfo{
    std::string name;
    int label_id;
    int id;
    int track_id;
    cv::Point2f min_pt,max_pt;
    cv::Rect2f rect;
    float prob;
    cv::Point2f mask_center;
    cv::Mat mask_cv;
    torch::Tensor mask_tensor;
};



template <typename T>
static std::string Dims2Str(torch::ArrayRef<T> list){
    int i = 0;
    std::string text= "[";
    for(auto e : list) {
        if (i++ > 0) text+= ", ";
        text += std::to_string(e);
    }
    text += "]";
    return text;
}

static std::string Dims2Str(nvinfer1::Dims list){
    std::string text= "[";
    for(int i=0;i<list.nbDims;++i){
        if (i > 0) text+= ", ";
        text += std::to_string(list.d[i]);
    }
    text += "]";
    return text;
}


inline cv::Point2f operator*(const cv::Point2f &lp,const cv::Point2f &rp)
{
    return {lp.x * rp.x,lp.y * rp.y};
}

template<typename MatrixType>
inline std::string Eigen2Str(const MatrixType &m){
    std::string text;
    for(int i=0;i<m.rows();++i){
        for(int j=0;j<m.cols();++j){
            text+=fmt::format("{:.2f} ",m(i,j));
        }
        if(m.rows()>1)
            text+="\n";
    }
    return text;
}


template<typename T>
inline std::string Vec2Str(const Eigen::Matrix<T,3,1> &vec){
    return Eigen2Str(vec.transpose());
}


inline cv::Scalar_<unsigned int> GetRandomColor(){
    static std::default_random_engine rde;
    static std::uniform_int_distribution<unsigned int> color_rd(0,255);
    return {color_rd(rde),color_rd(rde),color_rd(rde)};
}


void DrawText(cv::Mat &img, const std::string &str, const cv::Scalar &color, const cv::Point& pos, float scale= 1.f, int thickness= 1, bool reverse = false);

void DrawBbox(cv::Mat &img, const cv::Rect2f& bbox, const std::string &label = "", const cv::Scalar &color = {0, 0, 0});


 float GetBoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
                 const cv::Point2f &box2_minPt, const cv::Point2f &box2_maxPt);

 float GetBoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt);

 cv::Scalar ColorMap(int64_t n);


#endif
