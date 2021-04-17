# 如何应用于自己的项目

## 框架结构

为了更好基于Savior进行开发，需要了解Savior的设计模式以及运行

### 设计模式

Savior分为是一个自顶向下的开发框架，由顶至下分别为：

- API层：业务层接口定制，通过对Service的组装完成特定的输入输出
- Service层：任务定制，通过对Operator的组装完成特定功能，例如上传下载、目标检测、识别等
- Operator层：操作原子层，编码实现具体功能，例如视频拆分为图像、检测、识别等
- Infrastructure层：框架会用到的公共组件，例如RabbitMQ、Triton等

### 角色

Savior有两种角色，分别是DispatchServer和ConsumerWorker：

- DispatchServer：通过编写API层来实现接收请求相应请求，并且将任务分解分发给ConsumerWorker进行处理
- ConsumerWorker：通过编写Service层和Operator，处理具体应用型请求

### 分布式

DispatchServer利用[traefik](https://github.com/traefik/traefik)轻松的构建集群，用于应对高并发的情况

ConsumerWorker通过[celery](https://github.com/celery/celery)轻松的构建计算集群，用于处理不断分发下来的任务

对于其他基建组件，为了保证高可用以及高性能，也可以部署为对应的分布式，例如RabbitMQ、MySQL、Skywalking等。对于没有既定的分布式方案的，可以利用traefik搭建分布式版本，例如Triton等

**为了能更好的做扩展，当集群的数量大于3台服务器的时候，强烈建议基于K8S进行。**

## 接口开发

Savior的web框架采用[FastAPI](https://fastapi.tiangolo.com/)。Savior推荐用户编写自己的接口的时候将不同业务的接口编写在不同的文件中，利用router将所有接口串在一起。

开发属于自己的接口需要按照以下步骤进行：

1. 进入`Deployment/DispatchInterfaces`，参考与自己业务相似的接口代码，复制并修改名字

2. 定义接口名称与相对路径以及请求参数

   ```python
   @router.post('/dummy_interface_1')
   async def dummy_interface(
           dummy_input_1: str = Form(...),
           dummy_input_2: int = Form(...),
           dummy_input_3: float = Form(...),
   ):
     pass
   ```

3. 定义DAG，并撰写伪代码与逻辑，以文本识别举例：

   ```python
   dag = DAG()
   # 下载url图像
   download_task = ...
   # 文本区域检测
   text_detect_task = ...
   # 遍历所有检测得到的文本区域，并进行识别
   text_recognize_task = ...
   for m_region in text_detect_result:
     ...
   # 将所有检测区域和识别信息标注在图中并上传图片到oss并获得链接
   text_recognize_result_visualization_task = ...
   ```

   > 在设计service的时候尽可能避免单节点耗时过长，占用大量的资源。秉持stateless的原则进行服务的设计。而且为了提高并行度，把多个算子都会使用的公共信息（例如图像、中间变量数组）存储在OSS上，API调用service的时候不要传输大量数据（例如图像），避免点节点会存在严重的网络IO瓶颈，可以传递对应OSS的位置。

4. 在`Deployment/ConsumerServices`中创建、编写对应的ServiceTask，并配置Mock结果

   ```python
   class ImageDownloadServiceTask(ServiceTask):
       service_version = 'v1.0'
       service_name = 'image_download_service'
       mock_result = {
           'image_info':{
                     'bucket_name': 'testbucketname',
         						'path': '/fake/path',
           }
       }
       require_field = {
           "image_url",
       }
       binding_service = None
   ```

   继承ServiceTask类，其中的字段指定

   - service_version：定义service的版本，方便后续进行问题排查。
   - service_name：定义service的名称，方便后续进行问题排查。
   - mock_result：mock状态下返回的数据样式，一般为dict。
   - require_field：当前task如果需要请求需要传入的字段参数，一般为set。

   当前ServiceTask为一个下载Service，下载传入的URL至OSS并返回在OSS中的bucket name以及路径。

   其他ServiceTask如法炮制。

5. 使用编写完成的TaskService重新编写Interface

   ```python
   @router.post('/general_ocr')
   async def general_ocr(
           image_url: str = Form(...),
   ):
       dag = DAG()
       download_image_task = ImageDownloadServiceTask(_dag=dag,_is_mock=True)
       download_image_task.add_dependency_from_value('_image_url', image_url)
       text_detect_task = TextDetectServiceTask(_dag=dag,_is_mock=True)
       text_detect_task.add_dependency_from_task('_image_info', download_image_task, 'image_info')
       # 由于下载到检测是串行任务，所以这里可以直接await，但是就是需要自己跟上获取detail字段，
       detect_result = (await text_detect_task)['detail']
       recognize_task = []
       for i in range(detect_result['box_count']):
           m_recognize_task = TextRecognizeServiceTask(_task_name=f'No. {i} recognize',_dag=dag,_is_mock=True)
           m_recognize_task.add_dependency_from_task('_image_info', download_image_task, 'image_info')
           m_recognize_task.add_dependency_from_value('_box_info', detect_result['box_info'][i])
           recognize_task.append(m_recognize_task)
       recognize_result = await wait_and_compose_all_task_result(*recognize_task)
       # 上传结果图片，这个环节可以直接在interface中实现，但是为了保证服务的并发量，所以依然是将任务分发出去
       visualization_task = OCRResultVisualizationServiceTask(_dag=dag,_is_mock=True)
       visualization_task.add_dependency_from_task('_image_info', download_image_task, 'image_info')
       visualization_task.add_dependency_from_value('_box_info_list', detect_result['box_info'])
       visualization_task.add_dependency_from_value('_text_list', [m_detail for _, m_detail in recognize_result.items()])
       visualization_task_result = (await visualization_task)['detail']
       to_return_result = dict()
       to_return_result['bucket_name'] = visualization_task_result['bucket_name']
       to_return_result['path'] = visualization_task_result['path']
       to_return_result['url'] = visualization_task_result['url']
       return ORJSONResponse(to_return_result)
   ```

   这里每一个TaskService的入参可以有两个来源，一个是来自于原数据（例如请求传过来的`image_url`），另一个是来自于上游的任务的结果（例如text_detect_task的入参为图像信息，这个信息来源于download_image_task的结果的`image_info`字段）。

   用户在编写task的时候不需要去调度那个任务先执行，那个任务后执行，只需要配置好当前task启动需要的相应依赖即可。只要所有的依赖满足，任务就会立刻执行。可以理解为当前interface的编码过程其实是一个动态图的设计过程。

   以上为完成的interface的设计，其中所有的ServiceTask均为Mock状态，故会直接返回预先设定好的结果。

6. Mock版服务测试：`python Deployment/DispatchServer.py`启动server，并用apifox进行测试。如果得到的结果与预期结果一致则说明API层编写完成。

## 服务开发

通过完成**接口开发**，接下来只需要根据定义好的Task实现对应的service，这个过程中如果涉及到初始化代价很高的Operator（例如数据库、需要加载资源文件等），可以将其在全局变量状态下进行变量声明，对于一些代价小的Operator可以放在全局进行声明也可以放到service内进行声明。

以下以文本识别作为例子：

```python
@celery_worker_app.task(name="ConsumerServices.OCRService.text_recognize")
def text_recognize(_image_info, _box_info):
    """
    文本识别

    Args:
        _image_info:    待识别的完整图像
        _box_info:      图像中文本区域的位置

    Returns:    文本区域位置的识别结果

    """
    to_return_result = {'text': ''}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _box_info)
    get_image_rotation = text_orientation_op.execute(cropped_image)
    if get_image_rotation['orientation'] == TextImageOrientation.ORIENTATION_90:
        rotated_image, _ = rotate_degree_img(cropped_image, 90)
    elif get_image_rotation['orientation'] == TextImageOrientation.ORIENTATION_180:
        rotated_image, _ = rotate_degree_img(cropped_image, 180)
    elif get_image_rotation['orientation'] == TextImageOrientation.ORIENTATION_270:
        rotated_image, _ = rotate_degree_img(cropped_image, 270)
    else:
        rotated_image = cropped_image
    recognize_result = text_recognize_op.execute(rotated_image)
    to_return_result['text'] = recognize_result['text']
    return to_return_result

class TextRecognizeServiceTask(ServiceTask):
    service_version = 'v1.0.20210315'
    service_name = 'text_recognize'
    mock_result = {
        'text': '',
    }
    require_field = {
        "_image_info",
        "_box_info",
    }
    binding_service = text_recognize
```

下方的`TextRecognizeServiceTask`为在编写接口层的时候定义完成的。具体service的内容在`text_recognize`函数中。

首先是定义函数，这里需要使用celery的装饰器，其中`celery_worker_app`为在`ConsumerWorker.py`中定义的celery的app。name为当前函数在celery中的名称，用于唯一标识一个函数。注意保留`ConsumerServices`，为了避免与其他使用celery的应用冲突，这里加了这个前缀（可以在`ConsumerWorker.py`中自行修改）。当前函数的入参有两个：`_image_info`和`_box_info`，分别表示图像在oss中的信息和需要识别的区域box的信息。

```python
@celery_worker_app.task(name="ConsumerServices.OCRService.text_recognize")
def text_recognize(_image_info, _box_info):
```

定义`oss_helper`，用于下载oss中的图像文件，下载的图像为bgr的numpy格式，并存储为img变量。

```python
oss_handler = get_oss_handler()
img = oss_handler.download_image_file(
  _image_info['bucket_name'],
  _image_info['path']
)
```

抠取当前图像中的待识别部分，

```python
cropped_image = get_rotated_box_roi_from_image(img, _box_info)
```

利用文本行分类算子，判断是否需要进行翻转图像，

```python
get_image_rotation = text_orientation_op.execute(cropped_image)
if get_image_rotation['orientation'] == TextImageOrientation.ORIENTATION_90:
  rotated_image, _ = rotate_degree_img(cropped_image, 90)
elif get_image_rotation['orientation'] == TextImageOrientation.ORIENTATION_180:
  rotated_image, _ = rotate_degree_img(cropped_image, 180)
elif get_image_rotation['orientation'] == TextImageOrientation.ORIENTATION_270:
  rotated_image, _ = rotate_degree_img(cropped_image, 270)
else:
  rotated_image = cropped_image
```

将调整方向后的文本送入识别算子，

```python
recognize_result = text_recognize_op.execute(rotated_image)
```

一个service中可能需要用多个Operator才能实现一个功能，例如当前演示service使用了文本行方向检测op以及文本识别op。

## Op开发

### 算法Op开发

所有的算法Op都需要继承`DummyAlgorithm`类。对于需要进行模型推理的则需要继承`DummyAlgorithmWithModel`。

在常见的算法算子的开发过程中会经常出现大量重复的情况。例如有很多目标检测的算子，但是都是用的`YOLO`，这个时候可以先实现或者先找到`YOLO`的公共部分并抽象成基类，然后再实现对应算法的子类，尽可能避免重复开发。

例如可以像下面一样：

```python
class YOLO(DummyAlgorithmWithModel):
    def __init__(self):
        pass
      
class PedestrianDetectWithYOLO(YOLO):
    pass
  
class SurveillancePedestrianDetectWithYOLO(PedestriainDetectWithYOLO):
    pass
  
class GeneralPedestrianDetectWithYOLO(PedestriainDetectWithYOLO):
    pass
```

可以在不同的基类中增加相应的公共函数，或者公共对象，减少重复性开发。更新详细案例可以参考所有Example算子的明细。

### 非算法Op开发

所有非算法Op都需要继承`DummyOperator`类，常见的Op有数据库操作Op，数据下载与上传Op，数据预处理Op（例如将视频提取关键帧）。对于会有TCP长连接的Op，能复用就尽可能复用，减少资源消耗。

## 异常排查

### Op异常排查

每个Op在实现的时候，都需要进行基本的“单元测试”，即在Op中编写主函数，能够完成运行demo。故，如果Op发生了错误，可以直接在主函数中传入相应的值，用于复现Bug，然后通过Debug模式进行问题排查及解决问题。

> 有些内存泄漏的问题，在单个Op中无法复现，在Celery中运行的时候会出现，例如在没有安装opencv-python-contritbute-headless的情况下，运行cvtColor多次后会出现内存泄漏，安装后基本解决此问题。
>
> 如遇这种情况，请在Issue中提出，并附上详细说明。

### Service异常排查

Service在实现的时候会实现两个东西，一个是TaskService的类，另一个就是Celery的task即真实的service。可以直接在Service的代码中添加主函数，然后调用函数进行测试，这样就不会经过Celery启动了。如果有些问题很难通过编写主函数的方式排查，那么就需要参考[Celery Debug](https://docs.celeryproject.org/en/stable/userguide/debugging.html)官方教程进行调试错误。

> 当使用pdb启动之后，并且获取了对应的端口之后，可以使用IDE自带的Debug工具进行远程调试。如果启动的worker是在本地的话，也可以直接通过attach process的方式进行调试。

### Dispatch异常排查

DispatchServer的生产环境的启动模式是基于gunicorn进行启动，如果要进行问题排查，可以直接使用DispatchServer中的主函数，通过uvicorn以Debug模式启动，进行断点排查。