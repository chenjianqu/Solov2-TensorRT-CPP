/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "solo.h"

using namespace std;
using Slice=torch::indexing::Slice;
using InterpolateFuncOptions=torch::nn::functional::InterpolateFuncOptions;


torch::Tensor Solov2::MatrixNMS(torch::Tensor &seg_masks,torch::Tensor &cate_labels,torch::Tensor &cate_scores,torch::Tensor &sum_mask)
{
    int n_samples=cate_labels.sizes()[0];

    //seg_masks.shape [n,h,w] -> [n,h*w]
    seg_masks = seg_masks.reshape({n_samples,-1}).to(torch::kFloat);

    ///计算两个实例之间的内积，即相交的像素数
    auto inter_matrix=torch::mm(seg_masks,seg_masks.transpose(1,0));
    auto sum_mask_x=sum_mask.expand({n_samples,n_samples});

    ///两个两两实例之间的IOU
    auto iou_matrix = (inter_matrix / (sum_mask_x + sum_mask_x.transpose(1,0) - inter_matrix ) ).triu(1);
    auto cate_label_x = cate_labels.expand({n_samples,n_samples});

    auto label_matrix= (cate_label_x==cate_label_x.transpose(1,0)).to(torch::kFloat).triu(1);

    ///计算IoU补偿
    auto compensate_iou = std::get<0>( (iou_matrix * label_matrix).max(0) );//max()返回两个张量(最大值和最大值索引)组成的tuple
    compensate_iou = compensate_iou.expand({n_samples,n_samples}).transpose(1,0);
    auto decay_iou = iou_matrix * label_matrix;

    ///计算实例置信度的衰减系数
    torch::Tensor decay_coefficient;
    if(Config::kSoloNmsKernel == "gaussian"){
        auto decay_matrix = torch::exp(-1 * Config::kSoloNmsSigma * (decay_iou.pow(2)));
        auto compensate_matrix= torch::exp(-1 * Config::kSoloNmsSigma * (compensate_iou.pow(2)));
        decay_coefficient = std::get<0>( (decay_matrix / compensate_matrix).min(0) );
    }
    else if(Config::kSoloNmsKernel == "linear"){
        auto decay_matrix = (1-decay_iou) / (1-compensate_iou) ;
        decay_coefficient = std::get<0>( (decay_matrix).min(0) );
    }
    else{
        throw;
    }
    ///更新置信度
    auto cate_scores_update = cate_scores * decay_coefficient;
    return  cate_scores_update;
}



void Solov2::GetSegTensor(std::vector<torch::Tensor> &outputs, ImageInfo& img_info, torch::Tensor &mask_tensor,
                          std::vector<InstInfo> &insts)
{
    torch::Device device = outputs[0].device();

    constexpr int batch_index=0;
    const int kNumStage=kSoloNumGrids.size();//FPN共输出5个层级

    auto kernel_tensor=outputs[0][batch_index].view({kSoloTensorChannel, -1}).permute({1, 0});
    for(int i=1; i < kNumStage; ++i){
        auto kt=outputs[i][batch_index].view({kSoloTensorChannel, -1}); //kt的维度是(128,h*w)
        kernel_tensor = torch::cat({kernel_tensor,kt.permute({1,0})},0);
    }
    constexpr int kChannel=80;
    auto cate_tensor=outputs[kNumStage][batch_index].view({kChannel, -1}).permute({1, 0});
    for(int i= kNumStage + 1; i < 2 * kNumStage; ++i){
        auto ct=outputs[i][batch_index].view({kChannel, -1}); //ct的维度是(h*w, 80)
        cate_tensor = torch::cat({cate_tensor,ct.permute({1,0})},0);
    }
    auto feat_tensor=outputs[2 * kNumStage][batch_index];

    const int kFeatH=(int)feat_tensor.sizes()[1];
    const int kFeatW=(int)feat_tensor.sizes()[2];
    const int kPredNum=(int)cate_tensor.sizes()[0];//所有的实例数量(3872)

    ///过滤掉低于0.1置信度的实例
    auto inds= cate_tensor > Config::kSoloScoreThr;
     if(inds.sum(torch::IntArrayRef({0,1})).item().toInt() == 0){
         WarnLog("inds.sum(dims) == 0");
        return;
    }
    cate_tensor=cate_tensor.masked_select(inds);

    ///获得所有满足阈值的，得到的inds中的元素inds[i,j]表示第i个实例是属于j类
    inds=inds.nonzero();
    ///获得每个实例的类别
    auto cate_labels=inds.index({"...",1});
    ///获得满足阈值的kernel预测
    auto pred_index=inds.index({"...",0});
    auto kernel_preds=kernel_tensor.index({pred_index});

    DebugLog("After kSoloScoreThr: {}",cate_labels.sizes()[0]);
    /*for(int i=0;i<cate_labels.sizes()[0];++i){
        DebugLog("id:{},cls:{},prob:{}", i,Config::CocoLabelVector[cate_labels[i].item().toInt()],
                 cate_labels[i].item().toFloat());
    }*/

    ///计算每个实例的stride
    auto strides=torch::ones({kPredNum}, device);
    //计算各个层级上的实例的strides
    int index0=size_trans_[0].item().toInt();
    strides.index_put_({torch::indexing::Slice(torch::indexing::None,index0)}, kSoloStrides[0]);
    for(int i=1; i < kNumStage; ++i){
        int index_start=size_trans_[i - 1].item().toInt();
        int index_end=size_trans_[i].item().toInt();
        strides.index_put_({torch::indexing::Slice(index_start,index_end)}, kSoloStrides[i]);
    }
    //保留满足阈值的实例的strides
    strides=strides.index({pred_index});

    ///将mask_feat和kernel进行卷积
    auto seg_preds=feat_tensor.unsqueeze(0);
    //首先将kernel改变为1x1卷积核的形状
    kernel_preds=kernel_preds.view({kernel_preds.sizes()[0],kernel_preds.sizes()[1],1,1});
    //然后进行卷积
    seg_preds=torch::conv2d(seg_preds,kernel_preds,{},1);
    seg_preds=torch::squeeze(seg_preds,0).sigmoid();

    ///计算mask
    auto seg_masks=seg_preds > Config::kSoloMaskThr;
    auto sum_masks=seg_masks.sum({1,2}).to(torch::kFloat);

    ///根据strides过滤掉像素点太少的实例
    auto keep=sum_masks > strides;
    if(keep.sum(0).item().toInt()==0){
        cerr<<"keep.sum(0) == 0"<<endl;
        return ;
    }
    seg_masks = seg_masks.index({keep,"..."});
    seg_preds = seg_preds.index({keep,"..."});
    sum_masks = sum_masks.index({keep});
    cate_tensor = cate_tensor.index({keep});
    cate_labels = cate_labels.index({keep});

    DebugLog("After strides filtering: {}",cate_labels.sizes()[0]);
    /*for(int i=0;i<cate_labels.sizes()[0];++i){
        DebugLog("id:{},cls:{},prob:{}", i,Config::CocoLabelVector[cate_labels[i].item().toInt()],
                 cate_labels[i].item().toFloat());
    }*/

    ///根据mask预测设置实例的置信度
    auto seg_scores=(seg_preds * seg_masks.to(torch::kFloat)).sum({1,2}) / sum_masks;
    cate_tensor *= seg_scores;

    ///根据cate_score进行排序，用于NMS
    auto sort_inds = torch::argsort(cate_tensor,-1,true);
    if(sort_inds.sizes()[0] >  Config::kSoloNmsPre){
        sort_inds=sort_inds.index({torch::indexing::Slice(torch::indexing::None,Config::kSoloNmsPre)});
    }
    seg_masks=seg_masks.index({sort_inds,"..."});
    seg_preds=seg_preds.index({sort_inds,"..."});
    sum_masks=sum_masks.index({sort_inds});
    cate_tensor=cate_tensor.index({sort_inds});
    cate_labels=cate_labels.index({sort_inds});

    ///执行Matrix NMS
    auto cate_scores = MatrixNMS(seg_masks,cate_labels,cate_tensor,sum_masks);

    ///根据新的置信度过滤结果
    keep = cate_scores >= Config::kSoloUpdateThr;
    if(keep.sum(0).item().toInt() == 0){
        cout<<"keep.sum(0) == 0"<<endl;
        return ;
    }
    seg_preds = seg_preds.index({keep,"..."});
    cate_scores = cate_scores.index({keep});
    cate_labels = cate_labels.index({keep});
    sum_masks = sum_masks.index({keep});

    DebugLog("After Matrix NMS: {}",cate_labels.sizes()[0]);
    /*for(int i=0;i<cate_scores.sizes()[0];++i){
        DebugLog("id:{},cls:{},prob:{}", i,Config::CocoLabelVector[cate_labels[i].item().toInt()],
                 cate_scores[i].item().toFloat());
    }*/

    ///再次根据置信度进行排序
    sort_inds = torch::argsort(cate_scores,-1,true);
    if(sort_inds.sizes()[0] >  Config::kSoloMaxPerImg){
        sort_inds=sort_inds.index({torch::indexing::Slice(torch::indexing::None,Config::kSoloMaxPerImg)});
    }
    seg_preds=seg_preds.index({sort_inds,"..."});
    cate_scores=cate_scores.index({sort_inds});
    cate_labels=cate_labels.index({sort_inds});
    sum_masks = sum_masks.index({sort_inds});

    DebugLog("seg_preds.dims:{}", Dims2Str(seg_preds.sizes()));

    ///对mask进行双线性上采样,
    static auto options=InterpolateFuncOptions().mode(torch::kBilinear).align_corners(true);
    auto op1=options.size(std::vector<int64_t>({kFeatH * 4, kFeatW * 4}));
    seg_preds = torch::nn::functional::interpolate(seg_preds.unsqueeze(0),op1);

    ///对mask进行裁切、缩放，得到原始图片大小的mask
    seg_preds =seg_preds.index({"...",Slice(img_info.rect_y,img_info.rect_y+img_info.rect_h),
                                Slice(img_info.rect_x,img_info.rect_x+img_info.rect_w)});

    auto op2=options.size(std::vector<int64_t>({img_info.origin_h, img_info.origin_w}));
    seg_preds = torch::nn::functional::interpolate(seg_preds,op2);
    seg_preds=seg_preds.squeeze(0);
    ///阈值化
    mask_tensor = seg_preds > Config::kSoloMaskThr;

    ///根据mask计算包围框
    for(int i=0;i<mask_tensor.sizes()[0];++i){
        auto nz=mask_tensor[i].nonzero();
        auto max_xy =std::get<0>( torch::max(nz,0) );
        auto min_xy =std::get<0>( torch::min(nz,0) );

        InstInfo inst;
        inst.id = i;
        inst.label_id =cate_labels[i].item().toInt();
        inst.name = Config::CocoLabelVector[inst.label_id];
        inst.max_pt.x = max_xy[1].item().toInt();
        inst.max_pt.y = max_xy[0].item().toInt();
        inst.min_pt.x = min_xy[1].item().toInt();
        inst.min_pt.y = min_xy[0].item().toInt();
        inst.rect = cv::Rect2f(inst.min_pt,inst.max_pt);
        inst.prob = cate_scores[i].item().toFloat();
        insts.push_back(inst);
    }
}
