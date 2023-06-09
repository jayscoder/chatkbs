# ChatKBS

完全离线的智能聊天知识库系统（ChatGLM-6B+Milvus）

硬件需求参照：[https://github.com/THUDM/ChatGLM-6B#%E7%A1%AC%E4%BB%B6%E9%9C%80%E6%B1%82](https://github.com/THUDM/ChatGLM-6B#%E7%A1%AC%E4%BB%B6%E9%9C%80%E6%B1%82)

## 1. Install

必须确保全部准备工作完成后才能正常启动该项目

### 1.1 创建python环境

安装Anaconda python: 3.9+ [https://www.anaconda.com/download/](https://www.anaconda.com/download/)

配置Anaconda镜像（如果网络没问题也可以不配置）[https://mirror.tuna.tsinghua.edu.cn/help/anaconda/](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)

创建虚拟环境

```shell
# 使用anaconda虚拟环境
conda create -n chatkbs python=3.9
conda activate chatkbs

# 或者使用venv虚拟环境
python -m venv myenv
# - 在windows上
venv\Scripts\activate
# - 在macOS/Linux上
source venv/bin/activate

pip install -r requirements.txt

# 如果是macos
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
# 如果是windows/Linux
pip install torch
```

### 1.2 下载ChatGLM-6B模型

请按照[https://github.com/THUDM/ChatGLM-6B#%E4%BB%8E%E6%9C%AC%E5%9C%B0%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B](https://github.com/THUDM/ChatGLM-6B#%E4%BB%8E%E6%9C%AC%E5%9C%B0%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B)
的步骤手动下载全部文件，放到THUDM/chatglm-6b目录下

### 1.3 安装Docker并启动milvus

请从官网安装Docker：[https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

安装完Docker后启动milvus

```shell
sudo docker compose up -d
```

### 1.4 运行

```shell
python main.py

# 如需指定使用第n个cuda
python main.py --cuda=1

# 获取命令行参数帮助
python main.py --help
```

## Milvus运行介绍

Start Milvus
In the same directory as the docker-compose.yml file, start up Milvus by running:

sudo docker compose up -d

If your system has Docker Compose V2 installed instead of V1, use docker compose instead of docker-compose. Check if
this is the case with $ docker compose version. Read here for more information.
Creating milvus-etcd ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

Now check if the containers are up and running.

```shell
$ sudo docker-compose ps
```

After Milvus standalone starts, there will be three docker containers running, including the Milvus standalone service
and its two dependencies.

```
      Name                     Command                  State                            Ports
--------------------------------------------------------------------------------------------------------------------
milvus-etcd         etcd -advertise-client-url ...   Up             2379/tcp, 2380/tcp
milvus-minio        /usr/bin/docker-entrypoint ...   Up (healthy)   9000/tcp
milvus-standalone   /tini -- milvus run standalone   Up             0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp
```

Stop Milvus
To stop Milvus standalone, run:

```shell
sudo docker compose down
```

To delete data after stopping Milvus, run:

```shell
sudo rm -rf  volumes
```

