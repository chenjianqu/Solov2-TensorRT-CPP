//
// Created by chen on 2021/11/9.
//
#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "infer.h"
#include "Config.h"


Infer::Ptr infer;
ros::Publisher pub_image_track;


void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr= cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img= ptr->image.clone();

    TicToc ticToc;

    /*auto [masks,insts_info] = infer->forward(img);
    if(!masks.empty()){
        //cv::imshow("mask",masks[0]);
        cv::cvtColor(masks[0],masks[0],CV_GRAY2BGR);
        cv::add(img,masks[0],img);
    }*/

    torch::Tensor mask_tensor;
    std::vector<InstInfo> insts_info;
    infer->forward_tensor(img,mask_tensor,insts_info);

    ticToc.toc_print_tic("infer time");

    if(!insts_info.empty()){
        auto merger_tensor = mask_tensor.sum(0).to(torch::kInt8) * 255;
            merger_tensor = merger_tensor.to(torch::kCPU);
        //merger_tensor =merger_tensor.clone();
        cout<<merger_tensor.sizes()<<endl;

        auto semantic_mask = cv::Mat(cv::Size(merger_tensor.sizes()[1],merger_tensor.sizes()[0]), CV_8UC1, merger_tensor.data_ptr()).clone();
        cv::cvtColor(semantic_mask,semantic_mask,CV_GRAY2BGR);
        cv::add(img,semantic_mask,img);
    }

    //cv::imshow("img",img);
    //cv::waitKey(1);
    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(ptr->header, "bgr8", img).toImageMsg();
    pub_image_track.publish(imgTrackMsg);
}


int main(int argc,char** argv)
{
    ros::init(argc, argv, "dynamic_vins_seg_test");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2){
        printf("please intput: rosrun vins vins_node [config file] \n");
        return 1;
    }
    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    Config cfg(config_file);

    infer= std::make_shared<Infer>();

    ros::Subscriber sub_img0 = n.subscribe(Config::IMAGE0_TOPIC, 100, img0_callback);
    pub_image_track = n.advertise<sensor_msgs::Image>("image_seg", 1000);


    cout<<"wating images:"<<endl;
    ros::spin();

    return 0;

}


