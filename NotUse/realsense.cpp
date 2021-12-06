#include <iostream>

#include <opencv2/video.hpp>

#include "InstanceSegment/NotUse/Dataloader.h"
#include "InstanceSegment/infer.h"
#include "InstanceSegment/parameters.h"



int main(int argc, char **argv)
{
    if(argc != 2){
        cerr<<"please input: [config file]"<<endl;
        return 1;
    }
    string config_file = argv[1];
    fmt::print("config_file:{}",argv[1]);

    try{
        Config cfg(config_file);
    }
    catch(std::runtime_error &e){
        sgLogger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    Infer infer;

    ///相机读取线程
    sgLogger->info("初始化 Dataloader");
    Dataloader dataloader;

    std::thread dlt(&Dataloader::Run,&dataloader);

    int cnt=0;
    cout<<"循环"<<endl;


    ///主线程，图像分割
    while(cnt++ < 1500)
    {
        Frame::Ptr frame = dataloader.wait_for_frame();
        cv::Mat img=frame->color;

        TicToc ticToc;

        torch::Tensor mask_tensor;
        std::vector<InstInfo> insts_info;
        infer.forward_tensor(img,mask_tensor,insts_info);

        ticToc.toc_print_tic("infer time");

        //cloud->clear();
        cv::Mat draw_img=img;

        if(!insts_info.empty()){
            auto merger_tensor = mask_tensor.sum(0).to(torch::kInt8) * 255;
            merger_tensor = merger_tensor.to(torch::kCPU);
            //merger_tensor =merger_tensor.clone();
            cout<<merger_tensor.sizes()<<endl;

            auto semantic_mask = cv::Mat(cv::Size(merger_tensor.sizes()[1],merger_tensor.sizes()[0]), CV_8UC1, merger_tensor.data_ptr()).clone();
            cv::cvtColor(semantic_mask,semantic_mask,cv::COLOR_GRAY2BGR);
            cv::add(draw_img,semantic_mask*0.5,draw_img);

            for(auto &inst : insts_info){
                //PointCloud::Ptr pc(new PointCloud);
                //BuildPointCloud(img,frame->color,inst.min_pt.y,inst.max_pt.y,inst.min_pt.x,inst.max_pt.x,pc);
                //*cloud += (*pc);
                inst.name = CocoLabelVector[inst.label_id];
                printf("min(%d,%d) max(%d,%d)\n", inst.min_pt.x, inst.min_pt.y, inst.max_pt.x, inst.max_pt.y);
                cv::Point2i center = (inst.min_pt + inst.max_pt)/2;
                char prob_text[50];
                std::string category;
                sprintf(prob_text,"%.2f",inst.prob);
                for(auto& pair : objectClass){
                    auto &key=pair.first;
                    auto &word_set=pair.second;
                    for(auto &c : word_set){
                        if(c == inst.name){
                            category=key;
                            goto out_loop;
                        }
                    }
                }
                out_loop:
                std::string show_text =category+std::string(":")+ inst.name + prob_text;
                cv::putText(draw_img,show_text,center,cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(255,0,0),2);
                cv::rectangle(draw_img, inst.min_pt, inst.max_pt, cv::Scalar(255, 0, 0), 2);
            }

            char text[50];
            sprintf(text,"%.2lf ms",infer.infer_time);
            cv::putText(img,text,cv::Point2i(20,20),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,0,255));
        }
        //viewer.showCloud(cloud,"cloud");

        //videoWriter<<img;

        cv::imshow("img",img);
        cv::waitKey(1);
    }








/*    cv::Mat img=cv::imread("/home/chen/图片/20211110192412.jpg");

    TicToc ticToc;

    torch::Tensor mask_tensor;
    std::vector<InstInfo> insts_info;
    infer.forward_tensor(img,mask_tensor,insts_info);

    ticToc.toc_print_tic("infer time");

    if(!insts_info.empty()){
        auto merger_tensor = mask_tensor.sum(0).to(torch::kInt8) * 255;
        merger_tensor = merger_tensor.to(torch::kCPU);
        //merger_tensor =merger_tensor.clone();
        cout<<merger_tensor.sizes()<<endl;

        auto semantic_mask = cv::Mat(cv::Size(merger_tensor.sizes()[1],merger_tensor.sizes()[0]), CV_8UC1, merger_tensor.data_ptr()).clone();
        cv::cvtColor(semantic_mask,semantic_mask,cv::COLOR_GRAY2BGR);
        cv::add(img,semantic_mask*0.5,img);

        for(auto &inst : insts_info){
            inst.name = CocoLabelVector[inst.label_id];
            printf("min(%d,%d) max(%d,%d)\n", inst.min_pt.x, inst.min_pt.y, inst.max_pt.x, inst.max_pt.y);
            cv::Point2i center = (inst.min_pt + inst.max_pt)/2;
            char prob_text[50];
            std::string category;
            sprintf(prob_text,"%.2f",inst.prob);
            for(auto& [key,word_set] : objectClass){
                for(auto &c : word_set){
                    if(c == inst.name){
                        category=key;
                        goto out_loop;
                    }
                }
            }
            out_loop:
            std::string show_text =category+std::string(":")+ inst.name + prob_text;
            cv::putText(img,show_text,center,cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(255,0,0),2);
            cv::rectangle(img, inst.min_pt, inst.max_pt, cv::Scalar(255, 0, 0), 2);
        }

        char text[50];
        sprintf(text,"%.2lf ms",infer.infer_time);
        cv::putText(img,text,cv::Point2i(20,20),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,0,255));
    }

    //videoWriter<<img;

    cv::imshow("img",img);
    cv::waitKey(0);
    cv::imwrite("test.png",img);*/







    //videoWriter.release();
    cv::destroyAllWindows();


    std::cout << "Hello, World!" << std::endl;
    return 0;
}
