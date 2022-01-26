/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Solov2-TensorRT-CPP.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "buffer.h"
#include <spdlog/logger.h>
#include <cuda_runtime_api.h>
#include <optional>
#include "parameters.h"
#include "utils.h"


MyBuffer::MyBuffer(nvinfer1::ICudaEngine& engine){
    ///申请输出buffer
    binding_num=engine.getNbBindings();
    for(int i=0;i<binding_num;++i){
        auto dim=engine.getBindingDimensions(i);
        dims[i]=dim;
        int buffer_size=dim.d[0]*dim.d[1]*dim.d[2]*dim.d[3]*sizeof(float);
        size[i]=buffer_size;
        cpu_buffer[i]=(float *)malloc(buffer_size);
        if(auto s=cudaMalloc(&gpu_buffer[i], buffer_size);s!=cudaSuccess)
            throw std::runtime_error(fmt::format("cudaMalloc failed, status:{}",s));
        names[i]=engine.getBindingName(i);
    }
    if(auto s=cudaStreamCreate(&stream);s!=cudaSuccess)
        throw std::runtime_error(fmt::format("cudaStreamCreate failed, status:{}",s));
}


MyBuffer::~MyBuffer(){
    cudaStreamDestroy(stream);
    for(int i=0;i<binding_num;++i){
        if(auto s=cudaFree(gpu_buffer[i]);s!=cudaSuccess)
            ErrorLog("cudaFree failed, status:{}",s);
        delete cpu_buffer[i];
    }
    delete[] cpu_buffer;
}


void MyBuffer::CpyInputToGPU(){
    if(auto status = cudaMemcpyAsync(gpu_buffer[0], cpu_buffer[0], size[0], cudaMemcpyHostToDevice, stream);
        status != cudaSuccess)
        throw std::runtime_error(fmt::format("cudaMemcpyAsync failed, status:{}",status));
}

void MyBuffer::CpyOutputToCPU(){
    for(int i=1;i<binding_num;++i){
        if(auto status = cudaMemcpyAsync(cpu_buffer[i],gpu_buffer[i], size[i], cudaMemcpyDeviceToHost, stream);
        status != cudaSuccess)
            throw std::runtime_error(fmt::format("cudaMemcpyAsync failed, status:{}",status));
    }
    if(auto status=cudaStreamSynchronize(stream);status != cudaSuccess)
        throw std::runtime_error(fmt::format("cudaStreamSynchronize failed, status:{}",status));
}

/**
 * 采用这种方式来保证 输出张量的顺序与kTensorQueueShape一致，而不是根据名字来确定顺序
 * @param c
 * @param h
 * @param w
 * @return
 */
std::optional<int> GetQueueShapeIndex(int c, int h, int w)
{
    int index=-1;
    for(int i=0;i< (int)kTensorQueueShape.size();++i){
        if(c==kTensorQueueShape[i][1] && h==kTensorQueueShape[i][2] && w==kTensorQueueShape[i][3]){
            index=i;
            break;
        }
    }
    if(index==-1)
        return std::nullopt;
    else
        return index;
}

void MyBuffer::CudaToTensor(std::vector<torch::Tensor> &inst)
{
    inst.resize(kTensorQueueShape.size());
    auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    for(int i=1; i < binding_num; ++i){
        torch::Tensor tensor=torch::from_blob(
                gpu_buffer[i],
                {dims[i].d[0], dims[i].d[1], dims[i].d[2], dims[i].d[3]},
                opt);
        if(std::optional<int> index = GetQueueShapeIndex(
                    dims[i].d[1], dims[i].d[2], dims[i].d[3]);index){
            inst[*index] = tensor.to(torch::kCUDA);
        }
        else{
            throw std::runtime_error(fmt::format("GetQueueShapeIndex failed:{}",
                                                 Dims2Str(dims[i])));
        }
    }
}