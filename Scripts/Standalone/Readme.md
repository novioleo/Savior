# 单机部署Savior

## 适用场景

在只有一台服务器的情况下可以使用单机部署方案。**单机部署方案不具有任何容灾能力**，且无法体现Savior的优势。但是作为测试或者小规模使用没有问题。

## 准备工作

### 系统环境

- 系统： ubuntu （16、18、20均可）
- nvidia驱动： 455及以上
- python版本： 3.8.5
- docker版本：>=19.03

### 配置环境

#### 安装docker-compose

```shell script
python -m pip install docker-compose>=1.28.5
```
docker-compose用于编排一系列的docker容器。

#### 安装nvidia-docker以及nvidia-docker-runtime

[nvidia-docker仓库配置官方教程](https://nvidia.github.io/nvidia-docker/)

> 注意！！！
> 由于国内DNS被污染以及墙的原因，导致很多人没办法正常访问https://nvidia.github.io，这里可以通过翻墙解决，也可以通过更新host解决。
>
> 更新host可以访问http://ipaddress.com，然后输入https://nvidia.github.io，然后得到几个ip，但是并不是所有ip都是有效的，所以需要对每个ip进行telnet其443端口，如果有效再放入host文件。

参考[nvidia container toolkit安装教程](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)完成toolkit安装。

安装完成nvidia-docker-toolkit之后，使用`sudo apt-get install nvidia-docker-runtime`安装nvidia-docker-runtime，用于docker-compose可以使用nvidia docker。

#### 配置nvidia-docker-runtime

创建或打开docker的daemon文件`sudo vi /etc/docker/daemon.json`，如果当前文件为空，可以初始化为：

```json
{
    "default-runtime": "runc",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

如果已经存在其他的key-value，则增加上面的`default-runtime`和`runtimes`这两个key和对应的value即可。

配置完成后使用`sudo systemctl restart docker`重启docker。

## 开始部署

### 修改Savior启动配置文件

打开`Deployment`包下面的`server_config.py`，修改其中的Triton、RabbitMQ、Minio的host为triton、rabbitmq、minio（这个相当于是将ip指代为host了），如果想自定义名称，可以在docker-compose的yaml文件中自行修改。

其他相关参数配置请参考[快速上手教程](../../Docs/QuickStart.md)。

> 如果目前环境下已经有了相关的组件（Triton、RabbitMQ、Minio），可以不需要修改，地址该是多少就是多少。

### 修改Docker-compose配置文件

docker-compose的配置文件为`SaviorStandalone.env`，其中包含了组件的镜像的版本，以及用户名密码等。具体参考里面注释。

### 启动Docker-compose

通过命令行进入当前文件夹，运行`docker-compose --env-file ./SaviorStandalone.env -f SaviorStandalone.yaml up`启动所有关联容器。当所有关联容器都启动完全启动后会出现DispatchServer相关信息界面。参考[快速上手教程](../../Docs/QuickStart.md)的接口请求，对本地的接口进行测试。

> 注意！
> 出来的结果中minio的地址，是不可访问的，如果要访问，需要单独部署minio。或者直接查看minio的本地存储路径。
