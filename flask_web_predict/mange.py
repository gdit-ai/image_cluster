import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# 创建flask实例对象
app = Flask(__name__)

# 全局配置
UPLOAD_DIR = './templates/'
ALLOWED_EXT = ('jpg', 'jpeg', 'png', 'gif')
path = './static/upload/'

# 判断文件后缀名
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT
    #  rsplit() 方法通过指定分隔符对字符串进行分割并返回一个列表


# 路由注册
@app.route('/', methods=['GET', 'POST'])
def index():
    # 将静态文件从静态文件夹发送到浏览器
    return app.send_static_file('index.html')


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


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
