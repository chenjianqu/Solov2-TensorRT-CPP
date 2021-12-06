# Solov2-TensorRT-CPP
in this repo, we  deployed SOLOv2 to TensorRT with C++.

## Requirements
* Ubuntu 16.04/18.04/20.04
* Cuda10.2
* Cudnn8
* TensorRT8.0.1
* OpenCV 3.4 complie with CUDA
* Libtorch 1.8.2
* CMake 3.20

## Acknowledge
[SOLO](https://github.com/wxinlong/solo/)
[SOLOv2.tensorRT](https://github.com/zhangjinsong3/SOLOv2.tensorRT)


## Getting Started

**1. Install Solov2 from [SOLO](https://github.com/wxinlong/solo/)**
download,and run it successful

**2. Export the ONNX model fron original model**
before export , You have to modify some parts of the original SOLOv2 first:  

2.1. modify `SOLO-master/mmdet/models/anchor_heads/solov2_head.py:154:0`：
```
#Modify for onnx export, frozen the input size = 800x800, batch size = 1
size = {0: 100, 1: 100, 2: 50, 3: 25, 4: 25}
feat_h, feat_w = ins_kernel_feat.shape[-2], ins_kernel_feat.shape[-1]
feat_h, feat_w = int(feat_h.cpu().numpy() if isinstance(feat_h, torch.Tensor) else feat_h), int(feat_w.cpu().numpy() if isinstance(feat_w, torch.Tensor) else feat_w)
x_range = torch.linspace(-1, 1, feat_w, device=ins_kernel_feat.device)
y_range = torch.linspace(-1, 1, feat_h, device=ins_kernel_feat.device)
y, x = torch.meshgrid(y_range, x_range)
y = y.expand([1, 1, -1, -1])
x = x.expand([1, 1, -1, -1])

# Origin from SOLO
# x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
# y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
# y, x = torch.meshgrid(y_range, x_range)
# y = y.expand([ins_feat.shape[0], 1, -1, -1])
# x = x.expand([ins_feat.shape[0], 1, -1, -1])
```

2.2 `single_stage_ins.py`
in the function of forward_dummy(), add the forward_dummy of mask.

2.3 export onnx model
#kitti
python onnx_exporter.py ../configs/solov2/solov2_light_448_r34_fpn_8gpu_3x.py ../weights/SOLOv2_light_R34.onnx --checkpoint ../checkpoints/SOLOv2_LIGHT_448_R34_3x.pth --shape 384 1152



**3. build the tensorrt model**

first edit the config file:`config.yaml`
modify the item:
```
image_width: 1226
image_height: 370


# the position of exported ONNX model
onnx_path: ""  

# save  tensorrt model to:
serialize_path: "/home/chen/ws/dynamic_ws/src/dynamic_vins/weights/tensorrt_model_1152x384.bin"

# solo parameters
SOLO_NMS_PRE: 500
SOLO_MAX_PER_IMG: 100
SOLO_NMS_KERNEL: "gaussian"
#SOLO_NMS_SIGMA=2.0
SOLO_NMS_SIGMA: 2.0
SOLO_SCORE_THR: 0.1
SOLO_MASK_THR: 0.5
SOLO_UPDATE_THR: 0.2

segmentor_log_path: "/home/chen/CLionProjects/InstanceSegment/log/segmentor_log.txt"
segmentor_log_level: "debug"
segmentor_log_flush: "debug"

# test img dir
DATASET_DIR: "/media/chen/EC4A17F64A17BBF0/datasets/kitti/odometry/colors/07/image_2/"
WARN_UP_IMAGE_PATH: "/home/chen/CLionProjects/InstanceSegment/config/kitti.png"
```
and then,compile the CMake project, run
```
<BUILD_DIR>/build_model <PROJECT_MODEL>/config/config.yaml
```



**4. run the demo**
```
<BUILD_DIR>/InstanceSegment <PROJECT_MODEL>/config/config.yaml
```

