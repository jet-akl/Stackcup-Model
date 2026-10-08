# Stackcup Model

![baxter stackcup](./1.jpg "baxter stackcup")

## 1. 项目概述

本项目旨在基于单张 RGB 图像实现透明物体的机器人抓取。系统结合了以下关键技术：

- Segment Anything 2 (SAM2)：用于目标分割与掩码生成
- Depth Anything V2 (DAM)：用于单目深度估计
- Diffusion Policy (DP)：用于动作序列预测与控制

整体流程如下：

1. 使用 Baxter 机器人采集 RGB 图像与关节位姿数据；
2. 利用 YOLOv8、SAM 和 DAM 对采集数据进行批量预处理；
3. 生成透明物体的掩码深度先验，并转换为 zarr 数据格式；
4. 配置 Diffusion Policy 训练参数，训练抓取模型；
5. 在 Baxter 平台上进行实时推理与执行抓取任务。


## 2. Pipeline 介绍

我们的工作尝试利用 Segment Anything 2 和 Depth-Anything-V2，从单张 RGB 图像中获取透明物体的深度先验，然后使用 Diffusion-Policy 模型完成机器人抓取实验。实际实验基于 Baxter 平台进行，Baxter 既用于数据采集，也用于实时推理执行。

### 2.1 数据采集与预处理

- 使用 Baxter 采集 RGB 图像与关节位姿数据；
- 借助 YOLOv8、SAM 和 DAM 进行批量数据预处理；
- 将 RGB 图像转换为掩码深度图，并打包成 zarr 格式。

### 2.2 模型训练

- 配置 Diffusion-Policy 的训练参数，使其适配当前任务的数据流；
- 对处理后的数据进行训练，得到抓取策略模型。

### 2.3 推理执行

- 使用双线程机制进行连续推理；
- 采集端实时采集 Baxter 的 RGB 图像和位姿信息；
- 推理端对数据进行预处理，并输出候选动作序列，执行机器人动作。


## 3. 配置与运行

### 3.1 创建并激活 Python 环境

测试验证可使用 Python 3.9 环境，命令如下：

```bash
conda create -n dp python=3.9
```

如果已有环境，也可以直接使用现有环境。

### 3.2 安装 Diffusion-Policy

在项目根目录执行：

```bash
cd ./
pip install -r requirements.txt
pip install .
cd -
```

说明：部分依赖库在安装过程中可能因缺失源而失败；这些依赖已放置在 `deps/` 目录中，可手动安装。

### 3.3 安装 Depth-Anything-V2

```bash
cd ./Depth-Anything-V2
pip install -r requirements.txt
pip install .
cd -
```

### 3.4 安装 SAM2

```bash
cd ./sam2-main
pip install .
cd -
```

### 3.5 配置 Baxter 环境

```bash
cd ./ws_baxter
catkin_make
catkin_make install
cd -
```

注意事项：

- 编译 Baxter 相关包之前，需要确认系统中已安装 ROS 1；
- 如果编译失败，可能是当前主流 ROS 1 版本不再兼容较老的 MoveIt 包，导致缺少大量旧版 ROS 依赖；
- 如果当前任务不强依赖 MoveIt，可以删除 `moveit_robots` 包，再重新编译即可。

编译完成后，需要配置 `baxter.sh`，将其中的 `your_ip` 替换为当前主机的 IP 地址，确保 Baxter 能正确连接。

安装完成后，在 Conda 环境中使用 Baxter 功能包时，需要先启动 Baxter 环境，再激活 Conda 环境：

```bash
. baxter.sh
conda activate [your env]
```


## 4. 数据采集

基于 Baxter 的数据采集方案共有三种：

- `keyboard_sampler_step.py`：键盘步进采集
- `xbox_sampler_step.py`：Xbox 手柄按键步进采集
- `xbox_sampler_continue.py`：Xbox 摇杆连续伺服采集

启动采集程序：

```bash
python xbox_sampler_continue.py
```


## 5. 数据预处理

对采集到的 RGB 图像进行批量处理，使用 SAM 和 DAM 生成掩码深度图：

```bash
python depth_boxSD.py
```

随后将数据集打包为 zarr 格式：

```bash
python data2zarr.py
```


## 6. Diffusion-Policy 模型训练

配置训练参数：修改 `stackcup_cnn_xbox.yaml` 文件。

启动训练：

```bash
python train.py --config-dir=. --config-name=stackcup_cnn_xbox.yaml training.device=cuda:0 hydra.run.dir='outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}'
```

该命令将训练设备设置为 CUDA，并将训练输出保存在 `outputs/` 目录下，按时间戳创建结果文件夹。


## 7. Baxter 实机推理运行

配置推理参数：修改 `stackcup_maskdepth.yaml` 文件。

Baxter 实机推理支持三种执行方式：

- `forward_step.py`：步进运行
- `forward_continue.py`：伺服连续运行

运行示例：

```bash
cd ./ws_baxter
. baxter.sh
cd -
python forward_step.py
```


## 8. 代码框架

```text
Stackcup_Model/
├── checkpoints/                   # 存放模型文件
├── data/                          # 存放采集数据
├── deps/                          # 部分第三方依赖包
├── outputs/                       # 存放 DP 模型训练结果
├── Depth-Anything-V2/             # Depth Anything V2 模型代码
├── sam2-main/                     # SAM2 模型代码
├── diffusion_policy/              # Diffusion Policy 源代码
├── ws_baxter/                     # Baxter 功能包工作空间
├── dp_space/                      # 存放推理和训练配置，以及部分关键 DP 文件
├── model/                         # Stackcup 主体模型代码
│   ├── datasetmanager/            # 数据集管理类，主要用于使用 SAM 和 DAM 批量处理 RGB 文件
│   ├── baxter/                   # Baxter 运动类，整合 Baxter 功能包调用接口
│   ├── runner/                   # 推理类，负责 Baxter 实机推理，包含抓取和放置
│   └── sampler/                  # 采样类，负责 Baxter 实机采样
├── scripts/                       # 存放运行脚本
│   ├── forward/                  # Baxter 实机推理执行脚本
│   ├── process/                  # 使用 SAM 和 DAM 对 RGB 进行批量预处理并打包为 zarr
│   ├── sampler/                  # 使用 Baxter 进行采样的脚本
│   └── train/                    # DP 模型训练脚本
├── README.md                     # 原始说明文档
├── train.py                      # 训练入口脚本
├── depth_boxSD.py                # RGB 深度预处理脚本
├── data2zarr.py                  # 数据打包脚本
├── forward_step.py               # 步进推理脚本
├── forward_continue.py           # 连续推理脚本
├── keyboard_sampler_step.py      # 键盘采样脚本
├── xbox_sampler_step.py          # 手柄步进采样脚本
├── xbox_sampler_continue.py      # 手柄连续采样脚本
├── stackcup_cnn_xbox.yaml        # 训练配置文件
├── stackcup_mask_transformer.yaml # 训练配置文件
├── stackcup_mask.yaml            # 训练配置文件
├── stackcup_maskdepth.yaml       # 推理配置文件
└── ...
```


## 9. 相关项目链接

- Diffusion Policy：https://github.com/real-stanford/diffusion_policy.git
- Depth Anything V2：https://github.com/DepthAnything/Depth-Anything-V2.git
- Segment Anything 2：https://github.com/facebookresearch/sam2.git
