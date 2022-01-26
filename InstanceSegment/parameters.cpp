/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"
#include <filesystem>

void InitLogger()
{
    auto reset_log_file=[](const std::string &path){
        if(!fs::exists(path)){
            std::ifstream file(path);//创建文件
            file.close();
        }
        else{
            std::ofstream file(path,std::ios::trunc);//清空文件
            file.close();
        }
    };

    auto get_log_level=[](const std::string &level_str){
        if(level_str=="debug")
            return spdlog::level::debug;
        else if(level_str=="info")
            return spdlog::level::info;
        else if(level_str=="warn")
            return spdlog::level::warn;
        else if(level_str=="error" || level_str=="err")
            return spdlog::level::err;
        else if(level_str=="critical")
            return spdlog::level::critical;
        else{
            cerr<<"log level not right, set default warn"<<endl;
            return spdlog::level::warn;
        }
    };

    reset_log_file(Config::kLogPath);
    sgLogger = spdlog::basic_logger_mt("segmentor_log",Config::kLogPath);
    sgLogger->set_level(get_log_level(Config::kLogLevel));
    sgLogger->flush_on(get_log_level(Config::kLogFlush));
}



Config::Config(const std::string &file_name)
{
    cv::FileStorage fs(file_name, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(fmt::format("ERROR: Wrong path to settings:{}\n",file_name));
    }

    fs["IMAGE_HEIGHT"] >> kImageHeight;
    fs["IMAGE_WIDTH"] >> kImageWidth;

    fs["LOG_PATH"] >> kLogPath;
    fs["LOG_LEVEL"] >> kLogLevel;
    fs["LOG_FLUSH"] >> kLogFlush;

    fs["ONNX_PATH"] >> kDetectorOnnxPath;
    fs["SERIALIZE_PATH"] >> kDetectorSerializePath;

    fs["SOLO_NMS_PRE"] >> kSoloNmsPre;
    fs["SOLO_MAX_PER_IMG"] >> kSoloMaxPerImg;
    fs["SOLO_NMS_KERNEL"] >> kSoloNmsKernel;
    fs["SOLO_NMS_SIGMA"] >> kSoloNmsSigma;
    fs["SOLO_SCORE_THR"] >> kSoloScoreThr;
    fs["SOLO_MASK_THR"] >> kSoloMaskThr;
    fs["SOLO_UPDATE_THR"] >> kSoloUpdateThr;

    fs["DATASET_DIR"] >> kDatasetPath;
    fs["WARN_UP_IMAGE_PATH"] >> kWarnUpImagePath;
    fs.release();

    std::map<int,std::string> CocoLabelMap={
            {1, "person"}, {2, "bicycle"}, {3, "car"}, {4, "motorcycle"}, {5, "airplane"},
            {6, "bus"}, {7, "train"}, {8, "truck"}, {9, "boat"}, {10, "traffic light"},
            {11, "fire hydrant"}, {13, "stop sign"}, {14, "parking meter"}, {15, "bench"},
            {16, "bird"}, {17, "cat"}, {18, "dog"}, {19, "horse"}, {20, "sheep"}, {21, "cow"},
            {22, "elephant"}, {23, "bear"}, {24, "zebra"}, {25, "giraffe"}, {27, "backpack"},
            {28, "umbrella"}, {31, "handbag"}, {32, "tie"}, {33, "suitcase"}, {34, "frisbee"},
            {35, "skis"}, {36, "snowboard"}, {37, "sports ball"}, {38, "kite"}, {39, "baseball bat"},
            {40, "baseball glove"}, {41, "skateboard"}, {42, "surfboard"}, {43, "tennis racket"},
            {44, "bottle"}, {46, "wine glass"}, {47, "cup"}, {48, "fork"}, {49, "knife"}, {50, "spoon"},
            {51, "bowl"}, {52, "banana"}, {53, "apple"}, {54, "sandwich"}, {55, "orange"},
            {56, "broccoli"}, {57, "carrot"}, {58, "hot dog"}, {59, "pizza"}, {60, "donut"},
            {61, "cake"}, {62, "chair"}, {63, "couch"}, {64, "potted plant"}, {65, "bed"}, {67, "dining table"},
            {70, "toilet"}, {72, "tv"}, {73, "laptop"}, {74, "mouse"}, {75, "remote"}, {76, "keyboard"},
            {77, "cell phone"}, {78, "microwave"}, {79, "oven"}, {80, "toaster"},{ 81, "sink"},
            {82, "refrigerator"}, {84, "book"}, {85, "clock"},{ 86, "vase"}, {87, "scissors"},
            {88, "teddy bear"}, {89, "hair drier"}, {90, "toothbrush"}
    };
    CocoLabelVector.reserve(CocoLabelMap.size());
    for(auto &pair : CocoLabelMap){
        CocoLabelVector.push_back(pair.second);
    }

    InitLogger();
}




