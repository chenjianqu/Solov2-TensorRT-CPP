//
// Created by chen on 2021/8/24.
//

#ifndef DETECTOR_FRAME_H
#define DETECTOR_FRAME_H

#include <opencv2/opencv.hpp>

#include <condition_variable>
#include <chrono>

using TimeType= std::chrono::steady_clock::time_point;



class Frame{
public:
    using Ptr = std::shared_ptr<Frame>;

    Frame()=default;
    Frame(cv::Mat &color_,cv::Mat &depth_,TimeType time):
    color(color_),depth(depth_),timestamp(time){

    }


    void clone(Frame::Ptr &frame){//深度拷贝函数，代价很大
        frame.reset(new Frame);
        frame->color=this->color.clone();
        frame->depth=this->depth.clone();
    }

    cv::Mat color,depth;
    TimeType timestamp;

private:

};


class DetectFrame : public Frame{

};





#endif //DETECTOR_FRAME_H
