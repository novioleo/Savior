# Savior

**save your time.**

**只在Ubuntu18.04下完成全部测试，其他平台暂时未测试。**

**目前项目还处于早期开发阶段，如有任何问题，欢迎添加微信nsnovio，备注部署，进群交流。**


## 背景

`savior`是一个能够进行快速集成算法模块并支持高性能部署的轻量开发框架。能够帮助将团队进行快速想法验证（PoC），避免重复的去github上找模型然后复现模型；能够帮助团队将功能进行流程拆解，很方便的提高分布式执行效率；能够有效减少代码冗余，减少不必要负担。

> workflow的已经做好的轮子很多，例如[perfect](https://github.com/PrefectHQ/prefect)、 [polyaxon](https://github.com/polyaxon/polyaxon)、 [dagster](https://github.com/dagster-io/dagster)等。 之所以开发一个新的，主要原因是那些框架都太heavy了，对于大部分用户来说没法直接白嫖。

这个项目的核心目的就是能够减少大家的重复性开发，能够把绝大部分能够直接白嫖的东西放在框架里面，然后大家专注于自己的业务属性上，提升大家的工作效率。

## 特性

1. 弹性伸缩：用户可以根据目前的请求量，自定义配置机器数。方便项目上量，并且保证服务器资源吃满（支持K8S）。
2. 流程DAG：用户通过自定义自己的流程，框架支持DAG，保证流程的最高的并行度。
3. 容灾能力强：集群中所有节点都是相同作用，不会因为部分节点挂掉而服务崩溃。
4. 可扩展性强：框架主要是实现了一种设计模式，开发者只需要按照当前设计模式，扩展性无上限。
5. 部署便捷：部署到上线不会超过5分钟（不考虑网速）。

## 框架中已集成的模型

> 更多开源模型欢迎在issue中补充，也十分欢迎您的PR。

### 人脸相关

- [x] [UltraFaceDetect](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
- [x] [RetinaFaceDetect](https://github.com/deepinsight/insightface)
- [x] [FaceParsing](https://github.com/zllrunning/face-parsing.PyTorch)
- [x] [Landmark2D](https://github.com/deepinsight/insightface)
- [x] [FaceEmbedding](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)

### OCR相关
- [x] [DB](https://github.com/WenmuZhou/PytorchOCR)
- [x] [CRNN](https://github.com/WenmuZhou/PytorchOCR)

### 图像搜索

官方已适配模型下载地址（不定时更新）：

- [百度网盘](https://pan.baidu.com/s/1DvSQMM76gGAltPLma6w1wQ)  密码: sg11

> 根据自己的需要下载模型，不用全部下载。

## 简单使用教程

1. 克隆项目`git clone https://github.com/novioleo/Savior.git`到本地。或者下载release下面的source包。
2. 启动[rabbitmq](https://hub.docker.com/_/rabbitmq)，推荐使用docker启动：`docker run --restart=always -d --hostname celery-broker --name celery-broker -p5672:5672 -p15672:15672 -e RABBITMQ_DEFAULT_USER=guest -e RABBITMQ_DEFAULT_PASS=guest rabbitmq:3-management`
3. 启动[triton](https://github.com/triton-inference-server/server)，推荐使用docker（需要安装[nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)）启动：`docker run --gpus=all --name=triton-server -p8000:8000 -p8001:8001 -v/path/to/your/model/repo/path:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models`，其中`/path/to/your/model/repo/path`是网盘中triton文件夹下载的所在文件夹。
4. 修改项目配置，进入Savior文件夹中，进入Deployment包中，复制`server_config.py.template`并重命名为`server_config.py`，修改里面triton、rabbitmq的配置。
5. 配置python与安装依赖，通过控制台进入Savior文件夹中，创建环境：`conda create -n SaviorEnv python=3.8`，激活环境`source activate SaviorEnv`，安装依赖：`python -m pip install nvidia-pyindex==1.0.6 && python -m pip install -r requirements.txt`
6. 启动ConsumerWorker，通过控制台进入Savior文件夹中，启动worker：`celery -A Deployment.ConsumerWorker worker --loglevel=INFO`，如果一切配置正确会显示已经成功加载Task。
7. 启动DispatchServer，通过控制台进入Savior文件夹中，启动server：`python Deployment/DispathServer.py`，启动成功会看到端口信息等。
8. 测试接口服务，推荐使用[apifox](https://www.apifox.cn/)进行接口调用测试，可以通过post请求测试`ocr_interface/general_ocr`接口，传入参数`image_url`，发送请求（第一次运行需要等待，模型需要预热，五次之后基本上时间会稳定），会得到一个OSS的路径，如果OSS使用的是Dummy（默认），则找到`/tmp/DummyOSS-temp-directory/{bucket_name}/{path}`对应的文件。

[生产级使用教程点我](./Docs/AdvancedTutorial.md)

## 接口结果预览

### OCR相关

[自然场景下OCR](./Docs/ResultImageGallery/general_ocr接口结果展示图.webp)

## 如何在自有项目下开发？

移步至：[DevelopTutorial](./Docs/DevelopTutorial.md)

## 感谢

感谢各位开源项目大佬的无私奉献。

