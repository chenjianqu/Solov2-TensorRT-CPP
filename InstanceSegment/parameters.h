/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef INSTANCE_SEGMENT_PARAMETER_H
#define INSTANCE_SEGMENT_PARAMETER_H

#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <exception>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::pair;
using std::vector;

using namespace std::chrono_literals;
namespace fs=std::filesystem;

//图像归一化参数，注意是以RGB的顺序排序
inline float kSoloImageMean[3]={123.675, 116.28, 103.53};
inline float kSoloImageStd[3]={58.395, 57.12, 57.375};
constexpr int kBatchSize=1;
constexpr int kSoloTensorChannel=128;//张量的输出通道数应该是128

inline std::vector<float> kSoloNumGrids={40, 36, 24, 16, 12};//各个层级划分的网格数
inline std::vector<float> kSoloStrides={8, 8, 16, 32, 32};//各个层级的预测结果的stride

inline std::vector<std::vector<int>> kTensorQueueShape{
        {1, 128, 12, 12},
        {1, 128, 16, 16},
        {1, 128, 24, 24},
        {1, 128, 36, 36},
        {1, 128, 40, 40},
        {1, 80, 12, 12},
        {1, 80, 16, 16},
        {1, 80, 24, 24},
        {1, 80, 36, 36},
        {1, 80, 40, 40},
        {1, 128, 96, 288}
};


inline std::shared_ptr<spdlog::logger> sgLogger;

template <typename Arg1, typename... Args>
inline void DebugLog(const char* fmt, const Arg1 &arg1, const Args&... args){ sgLogger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void DebugLog(const T& msg){sgLogger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void InfoLog(const char* fmt, const Arg1 &arg1, const Args&... args){sgLogger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void InfoLog(const T& msg){sgLogger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void WarnLog(const char* fmt, const Arg1 &arg1, const Args&... args){sgLogger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void WarnLog(const T& msg){sgLogger->log(spdlog::level::warn, msg);}
template <typename Arg1, typename... Args>
inline void ErrorLog(const char* fmt, const Arg1 &arg1, const Args&... args){sgLogger->log(spdlog::level::err, fmt, arg1, args...);}
template<typename T>
inline void ErrorLog(const T& msg){sgLogger->log(spdlog::level::err, msg);}
template <typename Arg1, typename... Args>
inline void CriticalLog(const char* fmt, const Arg1 &arg1, const Args&... args){sgLogger->log(spdlog::level::critical, fmt, arg1, args...);}
template<typename T>
inline void CriticalLog(const T& msg){sgLogger->log(spdlog::level::critical, msg);}



class Config {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr=std::shared_ptr<Config>;

    explicit Config(const std::string &file_name);

    inline static std::string kDetectorOnnxPath;
    inline static std::string kDetectorSerializePath;

    inline static int kImageHeight,kImageWidth;
    inline static std::vector<std::string> CocoLabelVector;

    inline static std::string kLogPath;
    inline static std::string kLogLevel;
    inline static std::string kLogFlush;

    inline static int kSoloNmsPre;
    inline static int kSoloMaxPerImg;
    inline static std::string kSoloNmsKernel;
    inline static float kSoloNmsSigma;
    inline static float kSoloScoreThr;
    inline static float kSoloMaskThr;
    inline static float kSoloUpdateThr;

    inline static string kDatasetPath;
    inline static string kWarnUpImagePath;

    inline static std::atomic_bool ok{true};

    inline static int input_h,input_w,input_c;
};

using cfg=Config;

#endif

