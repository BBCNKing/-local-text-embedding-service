# local-text-embedding-service
本地启动text-embedding的服务，用于适配如dify等本地embedding服务

### 安装

conda create -n aigc python=3.10

conda activate aigc

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

### 启动
sh app.sh /path/to/your/model 8080
