## 日常记录

#### Linux系统相关命令

- linux清除openjdk命令

  ```bash
  sudo apt-get remove openjdk*
  ```

- Linux配置java环境变量

  ```bash
  #可能编辑不同文件 /etc/profile
  export JAVA_HOME=/root/jdk/jdk1.8.0_291
  export JRE_HOME=${JAVA_HOME}/jre
  export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
  export PATH=${JAVA_HOME}/bin:$PATH
  ```

- tomcat的开启和关闭

  ```bash
  ./startup.sh #开启tomcat
  ./shutdown.sh #关闭tomcat
  ```

- 服务器开启和关闭端口号

  ```bash
  #开启对应端口号
  firewall-cmd --zone=public --add-port=3306/tcp --permanent
  #关闭对应端口号
  firewall-cmd --zone= public --remove-port=80/tcp --permanent # 删除
  ```

#### docker相关

- docker使用

  ```bash
  docker pull 镜像名 拉取镜像
  docker images #查看镜像
  docker ps [-a] #查看容器运行
  docker run #运行镜像，可以挂载配置文件等
  docker exec -it 容器id /bin/bash #进入容器内部
  docker start/stop 容器id #开启和关闭容器
  ```

#### git相关

- git使用

  ```bash
  git branch  #一般用于分支的操作，比如创建分支，查看分支等等，
  git checkout  #一般用于文件的操作和分支的操作
  ```

#### web相关

- 过滤器（Filter）和拦截器（Interceptor）

  > 过滤器和拦截器（Filter and Interceptor） 都是AOP编程思想的体现，都能是实现权限检查、日志记录等。不同的是
  >
  > - 使用范围不同，Filter是Servlet规范归档的，只能用于web程序中。而拦截器既可以用于web程序，也可以用于Application、Swing程序中。
  > - 规范不同，Filter是再Servlet规范中定义的，是Servlet容器支持的。而拦截器是在Spring容器内的，是Spring框架支持的。
  > - 使用的资源不同，同其他代码快一样，拦截器也是一个Spring的组件，归Spring管理，配置在Spring文件中，因此能使用Spring里的任何资源、对象，例如Service对象、数据源、事务管理等，通过ioc注入到拦截器即可；而Filter则不能。
  > -  深度不同，Filter只在Servlet前后起作用。而拦截器能够深入到方法前后、异常抛出前后，因此拦截器的使用具有更大的弹性。所以在Spring构架的程序中，优先使用拦截器。