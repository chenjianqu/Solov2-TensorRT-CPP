//
// Created by chen on 2021/8/10.
//

#ifndef DETECTOR_DATALOADER_H
#define DETECTOR_DATALOADER_H

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "Frame.h"

class Dataloader{
public:
    typedef std::shared_ptr<Dataloader> Ptr;
    Dataloader();

    void Run();
    void wait_for_frame(cv::Mat &color, cv::Mat &depth,std::chrono::steady_clock::time_point &timestamp);
    Frame::Ptr wait_for_frame();
    void push_back(cv::Mat color,cv::Mat depth);

     void SetRunningFlag(bool flag){
         std::lock_guard<std::mutex> lk(flagMutex);
         running_flag=flag;
     }

     bool GetRunningFlag(){
         std::lock_guard<std::mutex> lk(flagMutex);
         return running_flag;
     }

private:
    std::queue<Frame::Ptr> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCond;

    bool running_flag=true;
    std::mutex flagMutex;

    std::shared_ptr<rs2::pipeline> pipe;
    std::shared_ptr<rs2::align> aligner;

    int color_width,color_height,depth_width,depth_height;
};




#endif //DETECTOR_DATALOADER_H
