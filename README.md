# ChatKBS

完全离线的智能聊天知识库系统（ChatGLM-6B+Milvus）

## Milvus
Start Milvus
In the same directory as the docker-compose.yml file, start up Milvus by running:

sudo docker-compose up -d

If your system has Docker Compose V2 installed instead of V1, use docker compose instead of docker-compose. Check if this is the case with $ docker compose version. Read here for more information.
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

Now check if the containers are up and running.
```shell
$ sudo docker-compose ps
```


After Milvus standalone starts, there will be three docker containers running, including the Milvus standalone service and its two dependencies.
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
sudo docker-compose down
```


To delete data after stopping Milvus, run:

```shell
sudo rm -rf  volumes
```

