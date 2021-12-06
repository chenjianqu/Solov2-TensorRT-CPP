//
// Created by chen on 2021/8/10.
//

#include "Dataloader.h"
#include <librealsense2/rs.hpp>

Dataloader::Dataloader()
{
    //用于深度图像和color图像的对齐
    aligner = std::make_shared<rs2::align>(RS2_STREAM_COLOR);

    //用于剔除深度值
    rs2::threshold_filter depthThresholdFilter;
    depthThresholdFilter.set_option(RS2_OPTION_MIN_DISTANCE,RS::depth_filter_threshold_min);
    depthThresholdFilter.set_option(RS2_OPTION_MAX_DISTANCE,RS::depth_filter_threshold_max);

    //相机的配置
    rs2::config rscfg;
    rscfg.enable_stream(RS2_STREAM_COLOR,RS::camera_width,RS::camera_height,RS2_FORMAT_BGR8,RS::camera_fps);
    rscfg.enable_stream(RS2_STREAM_DEPTH,RS2_FORMAT_Z16,RS::camera_fps);//因为后面要对齐，所以深度数据流的大小为默认

    //启动pipeline
    pipe = std::make_shared<rs2::pipeline>();
    pipe->start(rscfg);


    //获得相机的重要参数，并输出
    auto frames=pipe->wait_for_frames();
    frames=aligner->process(frames);//对齐

     color_width=frames.get_color_frame().get_width();
     color_height=frames.get_color_frame().get_height();
     depth_width=frames.get_depth_frame().get_width();
     depth_height=frames.get_depth_frame().get_height();

    cout<<"Realsense2的Color大小："<<color_width<<" "<<color_height<<endl;
    cout<<"Realsense2的Depth大小："<<depth_width<<" "<<depth_height<<endl;
    if (frames.get_color_frame().get_profile().format() == RS2_FORMAT_BGR8)
        cout<<"Realsense2的Color数据格式：BGR8"<<endl;
    else if (frames.get_color_frame().get_profile().format() == RS2_FORMAT_RGB8)
        cout<<"Realsense2的Color数据格式：RGB8"<<endl;
    else
        cout<<"Realsense2的Color数据格式:其它 "<<frames.get_color_frame().get_profile().format()<<endl;


    ///获取深度相机内参
    rs2::stream_profile depth_profile =  frames.get_depth_frame().get_profile();

    rs2::video_stream_profile depth_vsprofile(depth_profile);
    rs2_intrinsics depth_intrin =  depth_vsprofile.get_intrinsics();
    RS::cx = depth_intrin.ppx;
    RS::cy = depth_intrin.ppy;
    RS::fx = depth_intrin.fx;
    RS::fy = depth_intrin.fy;
    RS::depth_height = depth_intrin.height;
    RS::depth_width = depth_intrin.width;

    std::cout<<"深度相机内参:"<<endl;
    std::cout<<"cx:"<<RS::cx<<" cy:"<<RS::cy<<std::endl;
    std::cout<<"fx:"<<RS::fx<<" fy:"<<RS::fy<<std::endl;
    std::cout<<"dw:"<<RS::depth_width<<" dh:"<<RS::depth_height<<std::endl;
}



void Dataloader::wait_for_frame(cv::Mat &color, cv::Mat &depth,std::chrono::steady_clock::time_point &timestamp) {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCond.wait(lock,[&]{return !frameQueue.empty();});
    color=frameQueue.front()->color;
    depth=frameQueue.front()->depth;
    timestamp=frameQueue.front()->timestamp;
    frameQueue.pop();
}

Frame::Ptr Dataloader::wait_for_frame() {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCond.wait(lock,[&]{return !frameQueue.empty();});
    Frame::Ptr frame=frameQueue.front();
    //frameQueue.front()->clone(frame);
    frameQueue.pop();
    return frame;
}


void Dataloader::push_back(cv::Mat color,cv::Mat depth)
{
    std::unique_lock<std::mutex> lock(queueMutex);
    if(frameQueue.size()< DATALOADER_QUEUE_SIZE){ //使得队列长度保持
        frameQueue.emplace(new Frame(color,depth,std::chrono::steady_clock::now()));
    }
    queueCond.notify_one();
}


void Dataloader::Run() {


    //开始采集数据，并放入队列中
    while(GetRunningFlag()){
        auto frames=pipe->wait_for_frames();//获得数据
        auto time_now=std::chrono::steady_clock::now();
        frames=aligner->process(frames);//对齐
        cv::Mat color(cv::Size(color_width,color_height),CV_8UC3,(void*)frames.get_color_frame().get_data());
        cv::Mat depth(cv::Size(depth_width,depth_height),CV_16UC1,(void*)frames.get_depth_frame().get_data());

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if(frameQueue.size() < DATALOADER_QUEUE_SIZE){ //使得队列长度保持
                frameQueue.emplace(new Frame(color,depth,time_now));
            }
            queueCond.notify_one();
        }
    }
    cout<<"Dataloader线程结束"<<endl;



}



