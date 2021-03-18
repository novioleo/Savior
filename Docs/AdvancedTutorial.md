# 进阶版教程
## 框架思路

### 逻辑框架

![逻辑框架](./logic_architecture.png)

#### 应用层

所有HTTP接口将会注册到负载均衡与服务注册上去，可以通过负载均衡的相应接口直接调用后侧服务。

#### 接口层

每个接口可能有多个stage的service完成，不同service之间可能存在并行串行的关系。例如一个人证核验的接口，如下图所示：

![人证核验](./example_人证核验.png)

>  上图中中所有绿色的block均为service。

service能够根据实际业务的情况进行复用。

#### 操作原子层

每个service可以由多个operator组成，例如人脸的皱纹检测，包括法令纹检测的operator，鱼尾纹检测的operator，抬头纹的operator等。当然也可以将每个operator放到多个单独的service，然后组成多stage的接口。下面是基于不同侧重点的比较：

> :star: 越多说明优势约明显

| 不同模式比较 | 多service多operator            | 单service多operator            |
| ------------ | ------------------------------ | ------------------------------ |
| 开发便捷性   | :star: :star::star:            | :star::star::star::star::star: |
| 可扩展性     | :star::star::star::star::star: | :star::star:                   |
| 运行效率     | :star::star::star::star::star: | :star::star::star:             |

#### 网络层

每个operator中可能会包含多个网络，例如`face parsing` ，需要包含`face detect`用于人脸检测，需要`face landmark`用于alignment，以及后面的parsing的结果的矫正，还需要`face parsing`的结果。这里跟操作原子层一样，可以将多个网络放到多个operator中。

> :star: 越多说明优势约明显

| 不同模式比较 | 多operator多network            | 单operator多network            |
| ------------ | ------------------------------ | ------------------------------ |
| 开发便捷性   | :star: :star::star:            | :star::star::star::star::star: |
| 可扩展性     | :star::star::star::star::star: | :star::star:                   |
| 运行效率     | :star::star::star:             | :star::star::star::star::star: |
| 数据交换效率 | :star::star:                   | :star::star::star::star::star: |

> **数据交换效率**：在计算过程中涉及到很多冗余计算，例如利用保存一张图在一个backbone中的输出，并作为多个分类head的输入。如果直接将backbone的输出的结果利用oss或者文件进行存储，效率势必会降低，所以完全在内存里面才是最快的。

#### 推理层

每个网络可以实现ncnn、triton的推理。

针对于不同的设备制定不同的推理框架。针对于没有nvidia显卡的机器，可以使用ncnn进行推理，通过vulkan利用其他品牌的独立显卡或者集显甚至于CPU。

针对有nvidia显卡的设备，推荐使用triton进行推理。