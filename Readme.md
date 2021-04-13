# Savior

**save your time.**

**只在Ubuntu18.04，Mac Big Sur下完成全部测试，其他平台暂时未测试。**

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

## 依赖的第三方组件

- rabbitmq：用于celery进行分布式的任务分发

- triton：用于gpu端的模型服务的集中部署

- milvus：用于特征向量搜索，存储【推荐有搜索需求的用户自行配置】

    > 如果觉得milvus太大，用户可以根据自己的自身情况直接使用faiss或者nmslib。并且自己实现对应helper。

## 框架中已集成的算法

> 更多开源模型欢迎在issue中补充，也十分欢迎您的PR。

### 人脸相关

- [x] [UltraFaceDetect](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) 人脸检测
- [x] [RetinaFaceDetect](https://github.com/deepinsight/insightface) 人脸检测
- [x] [FaceParsing](https://github.com/zllrunning/face-parsing.PyTorch) 人脸语义分区
- [x] [Landmark2D](https://github.com/deepinsight/insightface)  人脸landmark检测
- [x] [FaceEmbedding](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) 人脸特征向量提取
- [x] [SilentFaceAntiSpoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)  静默人脸活体检测
- [x] [Fair](https://github.com/dchen236/FairFace) 人脸年龄、性别、种族识别
- [x] HumanMattingWithUNet 人体抠图 

### OCR相关
- [x] [DB](https://github.com/WenmuZhou/PytorchOCR) 文本检测
- [x] [CRNN](https://github.com/WenmuZhou/PytorchOCR)   文本识别
- [ ] [网页、杂志等数据版式分析](https://github.com/Layout-Parser/layout-parser) 版式分析
- [ ] [文章数据版式分析](https://github.com/Layout-Parser/layout-parser) 版式分析
- [x] 文档图像方向矫正(deskew)
- [x] [文本行方向回归](https://github.com/WenmuZhou/PytorchOCR) 文本行方向检测

### 图像搜索

### 通用
- [x] NRIQA

官方已适配模型下载地址（不定时更新）：

- [百度网盘](https://pan.baidu.com/s/1DvSQMM76gGAltPLma6w1wQ)  密码: sg11

> 根据自己的需要下载模型，不用全部下载。
>
> 所有模型都是基于gpu进行转换的。其中部分模型（例如CRNN）带有RNN模块，triton在cpu的状态下没法在gpu转换的模型中正常推理，会报错。配合上专用的cpu版本即可。
> 但是考虑到绝大部分使用此框架的人，不会使用cpu的推理版本，所以这里就不放出来了。如果有需要，可以提issue。
>

## 文档

[快速上手](./Docs/QuickStart.md)

## 感谢

感谢各位开源项目大佬的无私奉献。

