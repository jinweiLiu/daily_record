### 日常记录

#### Linux系统相关命令

- linux清除openjdk命令

  ```bash
  sudo apt-get remove openjdk*
  ```

- Linux配置java环境变量

  ```bash
  #可能编辑不同文件 /etc/profile ~/.bashrc
  #参考连接 https://blog.csdn.net/uisoul/article/details/89439575
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

- docker启动tomcat

  ```bash
  docker run -d -p 8888:8080 -v /root/tomcat/:/usr/local/tomcat/webapps/ tomcat
  ```

  > -d 后台运行
  >
  > -p 指定访问主机的`8888`端口映射到`8080`端口。
  >
  > -v 指定我们容器的`/usr/local/tomcat/webapps/`目录为`/root/tomcat/`主机目录，后续我们要对tomcat进行操作直接在主机这个目录操作即可。

- docker启动zookeeper

  ```bash
  #下载Zookeeper镜像
  docker pull zookeeper
  #启动容器并添加映射
  docker run --privileged=true -d --name zookeeper --publish 2181:2181 -d zookeeper:latest
  #查看容器是否启动
  docker ps
  
  #客户端连接zookeeper，类似redis
  ./zkCli.sh
  #查看注册的服务
  ls /services
  #退出客户端
  quit
  ```

#### nginx相关

配置两个服务器

![image-20210528214640225](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210528214640225.png)

#### git相关

- git使用

  ```bash
  git branch  #一般用于分支的操作，比如创建分支，查看分支等等，
    git  branch -a #列出本地分支和远程分支
  git checkout  #一般用于文件的操作和分支的操作
  git push origin branchName #将本地分支推送到远程仓库上
  ```

#### web相关

- 过滤器（Filter）和拦截器（Interceptor）

  > 过滤器和拦截器（Filter and Interceptor） 都是AOP编程思想的体现，都能是实现权限检查、日志记录等。不同的是
  >
  > - 使用范围不同，Filter是Servlet规范归档的，只能用于web程序中。而拦截器既可以用于web程序，也可以用于Application、Swing程序中。
  > - 规范不同，Filter是再Servlet规范中定义的，是Servlet容器支持的。而拦截器是在Spring容器内的，是Spring框架支持的。
  > - 使用的资源不同，同其他代码快一样，拦截器也是一个Spring的组件，归Spring管理，配置在Spring文件中，因此能使用Spring里的任何资源、对象，例如Service对象、数据源、事务管理等，通过ioc注入到拦截器即可；而Filter则不能。
  > -  深度不同，Filter只在Servlet前后起作用。而拦截器能够深入到方法前后、异常抛出前后，因此拦截器的使用具有更大的弹性。所以在Spring构架的程序中，优先使用拦截器。

- 跨域请求(CORS)

  > CORS全称Cross Origin Resource Sharing
  >
  > 简单的来说就是一个项目去访问另外一个不同地址的项目的资源就叫跨域请求
  >
  > 之所以会跨域，是因为受到了同源策略的限制，同源策略要求源相同才能正常进行通信，即**协议、域名、端口号**都完全一致

  - 解决方案

    - 局部配置

      Controller上加一个@CrossOrgin

    - 全局配置

      添加一个配置类

      ```java
      @Bean
      public WebMvcConfigurer webMvcConfigurer(){
          return new WebMvcConfigurer() {
              @Override
              public void addCorsMappings(CorsRegistry registry) {
                  registry.addMapping("/**")//允许访问的资源路径
                      .allowedOrigins("http://lgp6.cn")//允许跨域访问的域名
                      .allowedMethods("*")//允许方法（POST GET等 *为全部）
                      . allowedHeaders("*") //允许的请求头 *为任何请求头
                      .allowCredentials(true) //是否携带cookire信息
                      .exposedHeaders(HttpHeaders.SET_COOKIE).maxAge(3600L); //maxAge(3600)表明在3600秒内，不需要再发送预检验请求，可以缓存该结果
      
              }
      
          };
      }
      ```

#### 语言相关

- 值传递和引用传递

  > Java的方法参数传递只有一种，就是"pass-by-value"，也就是值传递。
  >
  > - 如果是基本类型（byte, short, int, long, float, double, boolean, char），就是将原有的数据拷贝一份，方法内的操作对原有的数据不会有影响。
  > - 如果是对象类型，这里是容易误解的地方，因为正好规定对象的地址也叫做"**reference**", 我们将对象作为参数传递的时候实际上是将对象的地址传递进去。

- Python注册器（Register）

  注册器机制的引入是为了使工程的扩展性变得更好。当产品增加某个功能需要添加一些新函数或者类时，它可以保证我们可以复用之前的逻辑。

  使用方法：

  ```python
  register_obj = RegisterMachine("register")
  # decorate method
  @register_obj.register()
  def say_hello_with(name):
      print("Hello, {person}!".format(person=name))
      
  def say_hi_with(name):
      print("Hi, {person}!".format(person=name))
      
  register_obj.get("say_hello_with")("Peter")
  # function call method
  register_obj.register(say_hi_with)
  register_obj.get("say_hi_with")("John")
  ```

  从上面的例子我们可以看出，通过register_obj这个对象，通过传入对应的函数名来得到该函数，具体的实现如下：

  ```python
  class RegisterMachine(object):
      def __init__(self, name):
          # name of register
          self._name = name
          self._name_method_map = dict()
      
      def register(self, obj=None):
          # obj == None for function call register
          # otherwise for decorator way
          if obj != None:
              name = obj.__name__
              self._name_method_map[name] = obj
          
          else:
              def wrapper(func):
                  name = func.__name__
                  self._name_method_map[name] = func
                  return func
              return wrapper
  
      def get(self, name):
          return self._name_method_map[name]
  
  if __name__ == "__main__":
      register_obj = RegisterMachine("register")
      # decorate method
      @register_obj.register()
      def say_hello_with(name):
          print("Hello, {person}!".format(person=name))
  
      def say_hi_with(name):
          print("Hi, {person}!".format(person=name))
  
      register_obj.get("say_hello_with")("Peter")
      # function call method
      register_obj.register(say_hi_with)
      register_obj.get("say_hi_with")("John")
  ```

  运行结果

  > Hello, Peter! 
  >
  > Hi, John!

#### Spring boot相关

- 整合shiro 安全框架
- 整合elasticsearch 全文搜索
- hutool工具包，Hutool是一个Java工具包类库，对文件、流、加密解密、转码、正则、线程、XML等JDK方法进行封装，组成各种Util工具类 [Hutool](https://www.hutool.cn/)
- RestTemplate
- @Resource @@Autowired

#### Windows相关

删除服务

```bash
#删除服务的可执行文件后，该服务可能仍然会出现在注册表中。 如果发生这种情况下，请使用命令sc delete从注册表中删除服务的条目。
sc.exe delete "YourServiceName"
```

