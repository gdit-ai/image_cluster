# 深度学习图像分类开源项目
| 星期        | 车次           | 时间  |
| ------------- |:-------------:| -----:|
| 星期一      |G1008 | 4:30 |
|  星期二  | G1006      |  14:55 |
|  星期三   | G1007    |   18:30 |
## 作者信息  
广东科学技术职业学院 计算机工程技术学院（人工智能学院）AI应用实验室  
主要从事深度学习算法、计算机视觉、智能算法应用与部署，基于人工智能，计算机视觉算法web前后端应用开发，android开发，微信小程序开发，可以承接相关的项目开发业务。QQ群：971601256

团队成员（按姓名首字母）：
刘炳辉，张越，邹嘉龙，曾熙麒

指导老师：胡建华，余君

随着人工智能热潮的发展，图像识别已经成为了其中非常重要的一部分  
图像识别是指计算机对图像进行处理、分析，以识别其中所含目标的类别及其位置(即目标检测和分类)的技术  
其中图像分类是图像识别的一个类，是给定一幅测试图像，利用训练好的分类器判定它所属的类别

运行环境：Ubuntu16.04LTS、Python3  
所需要的库：CUDA、CUDNN、Numpy、TensorFlow-GPU2.0、Keras、OpenCV、Flask、OS等等依赖库  
编辑器：Pycharm、HBuilder X  
前端框架：Vue、Element、axios

本文分为三部分：
- 第一部分：系统驱动的安装与环境的搭建
- 第二部分：利用VGG16网络训练模型与预测
- 第三部分：通过前端与后端的交接，利用页面实现模型预测的可视化

**第一部分包括：**
安装显卡驱动、安装并配置CUDA、安装CUDNN加速库

**第二部分包括：**
数据集的获取、数据集的清洗及读取、数据增强、模型训练、预测展示、绘制训练准确率折线图

**第三部分包括：**
项目结构、前端页面、服务器端、页面展示

附项目中所用到的数据集及模型：[百度网盘](https://pan.baidu.com/s/1kFfU5y_IZGhLvspIJzkaeQ)

## 一、深度学习环境搭建
建议下载好需要用到的显卡驱动、CUDA、CUDNN、Anaconda3等安装包文件，放在用户的目录下，记住该目录路径。本文的显卡是RTX 2080 super、CUDNN7+CUDA10.1

**N卡显卡驱动下载地址：[https://www.nvidia.cn/Download/index.aspx?lang=cn](https://www.nvidia.cn/Download/index.aspx?lang=cn)  
CUDA下载地址：[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)  
CUDNN下载地址(需要账号)：[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)  
Anconda3下载地址：[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)**
### 安装显卡驱动(本文以2080 super显卡为例)
1. **删除原有驱动  (这里假设有安装驱动）**
	
   打开终端（按住Ctrl + Alt + T）
   
a)  禁用nouveau  (安装NVIDIA需要把系统自带的驱动禁用)  
	打开文件:  `sudo vim /etc/modprobe.d/blacklist.conf`  
文本末尾追加：

	blacklist nouveau
	options nouveau modeset=0

然后保存退出(:wq)，打开终端:`lsudo update-initramfs  -u`  
重启Ubuntu系统:`reboot`


b)  验证nouveau是否已禁用(不禁用nouveau安装显卡会报错)  
打开终端输入：`lsmod | grep nouveau`
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218114819650.jpg)  
如果没有任何显示，则表示禁用正常  
如果有显示驱动，则回去重复前面的步骤，重新禁用显卡

2. 进入命令行界面,按Ctrl + Alt + F1

关闭图形界面:   `sudo service lightdm stop`

给驱动文件加权限:   `sudo  chmod a+x  ./驱动文件名`

执行驱动文件：     `sudo ./驱动文件名  -no-opengl-files`

  **(安装过程中，所有选择使用默认选项)**
  
3.  挂载显卡:            `modprobe nvidia`

查看显卡信息：          `nvidia-smi`
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/2019121821242521.jpg)  
如果显示上图信息，则成功安装显卡驱动，没有则安装失败。  
**安装失败的话，不要继续下面的步骤，重复前面卸载显卡和安装显卡的步骤**

4.  开启图形界面:`sudo service lighdm start`

驱动安装参考：

[https://blog.csdn.net/xunan003/article/details/81665835](https://blog.csdn.net/xunan003/article/details/81665835)

### 安装并配置CUDA
     
给cuda文件加权限    `sudo chmod a+x  cuda文件名`

安装cuda `sudo sh cuda文件名`

安装过程中的选项  
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218212924953.png)  
2. 测试CUDA

我用的是cuda10.0，所以我的安装路径是：**/usr/local/cuda-10.0**

`cd  /usr/local/cuda-10.0/samples/1_Utilities/deviceQuery`  
`sudo  make`  
`sudo  ./deviceQuery`

3. 出现如下信息，则安装成功  
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218213416178.png)  
4. 安装完成后，设置环境变量      

5. 打开终端，输入：  `sudo vim  ~/.bashrc`

然后在文档最后面加入下面几句命令：（路径为自己安装的CUDA目录）

如我的目录为：usr/local/cuda-10.0/，则输入：

	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64  
	export PATH=$PATH:/usr/local/cuda-10.0/bin  
	export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0  
	
然后保存退出，在终端运行：`source ~/.bashrc`

6. 检查是否配置成功：新开一个终端，输入`nvcc --version`，如果显示下面的文子就说明安装成功了
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218213657735.png)

**下面是官方给的在线下载，安装CUDA10.1的方法，仅供参考**
```bash
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
```
```bash
sudo sh cuda_10.1.243_418.87.00_linux.run
```

### 安装CUDNN加速库(这里以CUDNN7 + CUDA10.1为例)
1. 解压安装cudnn  `sudo dpkg -i cudnn文件名`

​//后面两个步骤是最后验证安装时需要的（不一定只有这一种方式验证）  
`sudo dpkg -i libcudnn7-dev_7.6.4.38-1+cuda10.1_amd64.deb`  
`sudo dpkg -i libcudnn7-doc_7.6.4.38-1+cuda10.1_amd64.deb`  

验证安装  

 `sudo cp -r /usr/src/cudnn_samples_v7/    /home/wdong/`  
`cd  /home/wdong/mnistCUDNN`  
`sudo  make`  
`./mnistCUDNN`
## 二、VGG16网络模型训练
### 数据集的获取
本文的数据集中包含五个类，分别是**Animal（动物）、Architecture（建筑）、people（人）、plane（飞机）、Scenery（风景）**。由于身边条件的欠缺，数据集的获取途径主要通过爬取百度图片的方式获取，并将相同类的图片放在同一文件夹（代码：```Reptile_img.py```）。
爬取一个类示例：
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218140829865.png)  
**注意：保存的图片采用英文格式，不然后续无法利用OpenCV读入。**  

### 数据集的清洗及读取
由于获取的图片数据集较为杂乱，所以我们需要人为清洗数据，以提高数据集的准确性。而后，数据集中可能会存在无法利用OpenCV读取的图片，我们采取遍历所有文件，并将其读入，将无法读入显示的图片删除，保证数据集的正确性（代码：```open_img.py```）。
将所有图片读取示例：
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218140913593.png)  
**出现下方错误证明是爬取的图片无法读入，找到相对应的图片删除就好。**  
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218140930666.png)
### 数据增强
清洗处理以后，我们五个类的图片集数量分别为：  
|类别    | 数量  |  
|:--------: | :-----: |  
|Animal  | 500张  |  
|Architecture  | 500张  |  
|people  | 700张  |  
|plane  | 500张  |  
|Scenery  | 700张  |  
总体来说，数据集的数量太少，得出的模型可能存在模型泛化能力不强的情况。所以利用数据增强，对图片集进行随即旋转、平移变换、缩放变换、剪切变换、水平翻转等等操作，使得每张数据集得到50张处理后的图片，并保存在原本的文件夹下，整个数据集得到扩大（代码：```data_augmentation.py```）。
代码示例：
```python
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
dir = './data/'
path = os.path.abspath(dir)
all_file = os.listdir(dir)
all_file.sort()
i = 0
for file1 in all_file:
    # print(file1)
    img_file = os.listdir(os.path.join(path, file1))
    img_file.sort()
    for file2 in img_file:
        img = cv2.imread(os.path.join(path, file1, file2))
        x = img.reshape((1,) + img.shape) # datagen.flow要求rank为4 (1, x.shape)
        datagen.fit(x)
        prefix = file2.split('.')[0] # 以 . 分离字符串，去掉后缀
        counter = 0
        for batch in datagen.flow(x, batch_size=32, save_to_dir=dir + all_file[i], save_prefix=prefix, save_format='jpg'):
            counter += 1
            print(file2, '======', counter)
            if counter > 30:
                break  # 否则生成器会退出循环
    i += 1
```
**rotation_range=20 图片随机旋转20°  
width_shift_range=0.15  图片宽度随机水平偏移0.15  
height_shift_range=0.15 图片高度随机竖直偏移0.15  
zoom_range=0.15 随机缩放幅度为0.15  
shear_range=0.2 剪切强度为0.15  
horizontal_flip=True 随机对图片进行水平翻转**  
### 模型训练
利用处理好的数据集导入Keras自带VGG16网络结构，进行模型训练，并将模型保存。为了方便后续的调参，将每次不同batch、epoch、size的测试loss、accuracy写入txt文本，最后通过不断调参画出折线图，得出最佳的一个模型（代码：```TrainAndTest_VGG.py```）。  
**将五个类的文件夹按ASCII大小排序，并输出相对应的标签**
```python
def CountFiles(path):
    files = []
    labels = []
    path = os.path.abspath(path)
    subdirs = os.listdir(path)
    # print(subdirs)
    subdirs.sort()
    for index in range(len(subdirs)):
        subdir = os.path.join(path, subdirs[index])
        sys.stdout.flush()
        print("label --> dir : {} --> {}".format(index, subdir))
        for image_path in glob.glob("{}/*.jpg".format(subdir)):
            files.append(image_path)
            labels.append(index)
    # 将标签与数值对应输出
    return files, labels, len(subdirs)
```
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218141115236.png)  
**将数据集中每张图片的数据与对应的标签绑定，随机打乱后将标签转为One-Hot码。**
```python
files, labels, clazz = CountFiles(r"data")
c = list(zip(files, labels))
random.shuffle(c)
files, labels = zip(*c) # 数值与标签绑定，并随机打乱

labels = np.array(labels)
labels = keras.utils.to_categorical(labels, clazz) # 将标签转为One-Hot码
```
### 加载模型并预测展示
选取五个类中的一个类的图片，加载模型训练中loss较低accuracy较高的模型，将结果可视化在原有图片上，结果完全吻合实际(代码：`Predict.py`)。  
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218141225174.png)
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218141317247.png)
### 训练准确率折线图绘制
将模型batch、size参数进行多次修改，得出下图每次训练得到的最好模型准确率与损失率，并绘制相对应的折线统计图，可以发现，当batch=25，size=(64, 64)时，得到的训练准确率最高，损失率也较低，所以最终我们采用batch=25，size=64*64的模型。  
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218141344124.png)
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218141356764.png)
## 三、模型预测的页面可视化
### 项目结构
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218145155405.png)  
```|——static```	　　　#静态资源目录，此处为存放前端上传的图片  
```	|——lib```		　　　　#css、js文件存放目录  
```	|——upload	```　　　#图片上传后的保存目录  
```	|——index.html```	　#前端页面代码  
```|——mange.py```	　　#Flask启动文件（主文件）  
```|——package.txt```	　　#记录运行此项目所需要的依赖包及版本信息(使用pip freeze>package.txt)命令打包  
```|——PredictAndMove.py```　　　　　#封装后的模型预测文件  
```|——vggmodel_21-0.00-0.98.hdf5```　　#hdf5模型文件  

> Flask非常灵活，官方并没有给出一个固定的项目目录组织结构，此处的目录结构仅为个人习惯以及项目调用使用，也更加符合大型项目结构的工程化构建方法。
### 前端页面
该项目采用前后端分离的架构，前端主要采用Vue和Element，使用Axios进行前后端数据的交互。
```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>vue上传demo</title>
		<link rel="stylesheet" href="./static/lib/element.css">
		<script type="text/javascript" src="./static/lib/vue.min.js"></script>
		<script type="text/javascript" src="./static/lib/element.js"></script>
		<script type="text/javascript" src="./static/lib/axios.min.js"></script>
	</head>

	<body>
		<div id="app">
			<input type="file" style="display: none;" ref="input" @change="fileChange" multiple= "multiple"></input>
			<el-button type="primary" round @click="handleClick" v-if="upload_awesome">上传图片</el-button>
			<div v-for="(file,index) in imageData">
				<el-tag type="success" style="margin:5px 5px;display: flex;justify-content:center;">识别结果为:{{index}}</el-tag>
				<el-row>
					<el-col :span="6" v-for="img in file">
						<div class="img-box" style="width: 100%;">
							<img :src="img.path" alt="图片正在加载...">

						</div>
					</el-col>
				</el-row>
			</div>
		</div>
	</body>
```
```css
<style type="text/css">
		.img-box img 
		{
			width: 320px;
			height: 240px;
			margin: 5px 5px;
			border: #00E676 2px solid;
		}
</style>
```
为保证项目可以正常运行，建议参照官方文档下载和安装完整的Vue、Element、和Axios包文件。这里为了压缩项目文件大小，所以只引入了用到的文件，并不是很好的做法。

官方文档地址：
Vue:  [https://cn.vuejs.org/v2/guide/](https://cn.vuejs.org/v2/guide/)  
Element:  [https://element.eleme.cn/#/zh-CN/component/installation](https://element.eleme.cn/#/zh-CN/component/installation)  
Axios:  [http://www.axios-js.com/zh-cn/docs/#axios](http://www.axios-js.com/zh-cn/docs/#axios)

>1：必须HTML头部(head)使用link标签引入element的css样式文件，script标签分别引入vue.min.js、element.js、axios.min.js文件。  
2：页面主体使用element的el-button组件编写上传图片按钮，el-tag组件渲染模型的识别结果，el-col和el-row组件控制图片显示的布局风格。  
3：使用@符号绑定了事件函数，用来在JavaScript中调用这些函数进行逻辑处理。
```javascript
<script type="text/javascript">
		var vm = new Vue({
			el: '#app', // 创建Vue实例
			data: { //定义数据对象
				upload_awesome: true,
				imageData: {},
				file: '',
				retData: '',
                index:'识别中'
			},
			methods: { //事件处理器，this将自动绑定到Vue实例上面

				handleClick() {
					this.$refs['input'].click();
				},
				fileChange(e) {
					this.file = e.target.files[0];
					this.uploads()
				},
				uploads() {
					let self = this;
					let param = new FormData();
					param.append('file', this.file);
					axios.post('/upload/', param, {}).then(function(response) {
							if (response.status === 200) {
								let data = response.data;
								self.$set(self.$data, 'imageData', data)
								self.$set(self.$data, 'upload_awesome', false)
							}
						})
						.catch(function(error) {
							console.log("上传图片失败!")
						})
				},
			}
		})
	</script>
```
这里编写前端逻辑代码，包括上传和展示，下面分别介绍自定义的三个函数fileChange()、handleClick()和uploads()。
```handleClick()：```该函数主要是点击上传按钮之后，动态的去改变input属性，从而触发

```fileChange()： ```使用@change对input的值进行监听,如果监听到值的改变，就截取上传文件的name，并调用upload()函数

```uploads()：```该函数为主函数，即是通过该函数与服务器通信，以获取、交换数据。下面对该函数的核心代码进行解读

```param.append('file', this.file)```将上传的文件信息加入到param变量

```axios.post()：```axios中post请求实例方法的别名，创建该请求的基本配置选项如下axios.post(url[, data[, config]])，即请求路径(url)、数据、和配置，只有URL是必需的，详情请访问下面网址
[http://www.axios-js.com/zh-cn/docs/#axios-options-url-config-1](http://www.axios-js.com/zh-cn/docs/#axios-options-url-config-1)
then函数则接受该请求的响应信息，结构如下：
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218150359742.png)  
通过if (response.status === 200) {}
判断响应状态，并更新相应的数据，同时视图会进行重渲染，这是vue的特性。
在使用catch时，或传递拒绝回调作为then的第二个参数时，响应可以通过error对象可被使用，主要作用是处理错误。


### 服务器端（flask后端）
项目涉及到flask的内容较少，考虑到实用性的问题，此处并没有对核心代码进行路由、蓝图规划和拆分。
```python
import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
```
导入python的第三方库文件，以及flask用到的依赖库文件
```python
# 文件上传
@app.route('/upload/', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于jpg,jpeg,png,gif"})
        upload_path = os.path.join(path, secure_filename(f.filename))
        f.save(upload_path)

        # 图片展示
        files = os.listdir(path)
        fileList = {}
        for file in files:
            file_d = os.path.join(path, file)
            # 执行模型脚本
            res = os.popen("python ./PredictAndMove.py %s" % file_d)
            labels = res.read()
            label = str(labels).strip('\n')
            if label in fileList.keys():
                fileList[label].append({"filename": file, "path": file_d})
            else:
                fileList[label] = [{"filename": file, "path": file_d}]
            # 将字典形式的数据转化为字符串
    return json.dumps(fileList)
```

>  - 定义全局配置、判断后缀名函数以及路由注册   
>   - 配置信息是以键值对的格式书写，判断文件格式的函数中使用rsplit()方法通过指定分隔符对字符串进行分割并返回一个列表   
>    -  路由注册通过render_template()方法渲染前端模板（index.html）,flask会在templates文件夹内寻找模板文件
>    -  调用模型预测的文件

upload()函数为该项目后端的核心模块（详细解释请前往GitHub[下载](https://github.com/gdit-ai/image_cluster)查看源代码）后端整体逻辑为：
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191218201701550.png)
>补充：若项目运行出现这种报错，是因为路径的问题所导致，建议使用Ubuntu系统运行flask程序，或者将用到的路径全部更换为绝对路径
### 运行程序
打开终端输入`python app.py`
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191219230603679.png)  
此时服务器即为启动状态，打开浏览器，通过127.0.0.1:5000或者ip:5000的方式访问网页，上传图像，进行识别并在前端进行展示  
![加载中](https://github.com/gdit-ai/image_cluster/blob/master/image/20191219230919650.png)
