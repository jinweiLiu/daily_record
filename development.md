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

- Linux拷贝文件

  ```
  cp filename 指定路径
  ```

- 历史命令

  ```
  history 查看历史命令
  history -c 清除历史命令
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

#### Git相关

- git使用

  ```bash
  rm -rf .git  #删除原有仓库
  git branch  #一般用于分支的操作，比如创建分支，查看分支等等，
    git  branch -a #列出本地分支和远程分支
  git checkout  #一般用于文件的操作和分支的操作
  git push origin branchName #将本地分支推送到远程仓库上
   
  git push --set-upstream origin branchName #gitee提交方式
  
  git reset --hard HEAD^ #回退到上个版本
  git reset --hard HEAD~3 #回退到前3次提交之前，以此类推，回退到n次提交之前
  git reset --hard commit_id #退到/进到 指定commit的sha码
  
  ##本地和远程不同时，会造成冲突，可以使用git stash或者放弃本地修改，直接覆盖
  git stash
  
  git reset -- hard #直接覆盖
  git pull
  
  git checkout . #放弃当前所有本地修改
  
  git commit --amend  #修改commit信息，输入命令后进入vim模式
  ```

- 设置忽略文件

  .gitignore
  
- git用户名和密码设定

  ```bash
  #设置git用户名/邮箱
  git config --global credential.helper store
  git config user.name 'github用户名'
  git config user.email '邮箱'
  
  #查看配置
  git config --list
  
  #清除掉缓存在git中的用户名和密码
  git credential-manager uninstall
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

- Cookie和Session

  https://www.cnblogs.com/l199616j/p/11195667.html
  
  **HTTP协议是无状态的协议。一旦数据交换完毕，客户端与服务器端的连接就会关闭，再次交换数据需要建立新的连接。这就意味着服务器无法从连接上跟踪会话。**为了弥补以上的不足，引入了会话跟踪技术Cookie和Session。
  
  - Cookie
  
    Cookie实际上是一小段的文本信息。客户端请求服务器，如果服务器需要记录该用户状态，就使用response向客户端浏览器颁发一个Cookie。客户端浏览器会把Cookie保存起来。当浏览器再请求该网站时，浏览器把请求的网址连同该Cookie一同提交给服务器。服务器检查该Cookie，以此来辨认用户状态。服务器还可以根据需要修改Cookie的内容。
  
  - Session
  
    Session是另一种记录客户状态的机制，不同的是Cookie保存在客户端浏览器中，而Session保存在服务器上。客户端浏览器访问服务器的时候，服务器把客户端信息以某种形式记录在服务器上。这就是Session。客户端浏览器再次访问时只需要从该Session中查找该客户的状态就可以了。
  
    　　如果说**Cookie机制是通过检查客户身上的“通行证”来确定客户身份的话，那么Session机制就是通过检查服务器上的“客户明细表”来确认客户身份。Session相当于程序在服务器上建立的一份客户档案，客户来访的时候只需要查询客户档案表就可以了。**
  
  - 区别
  
    - **cookie数据存放在客户的浏览器上，session数据放在服务器上.**
    - **cookie不是很安全，别人可以分析存放在本地的COOKIE并进行COOKIE欺骗考虑到安全应当使用session。**
    - **设置cookie时间可以使cookie过期。但是使用session-destory（），我们将会销毁会话。**
    - **session会在一定时间内保存在服务器上。当访问增多，会比较占用你服务器的性能考虑到减轻服务器性能方面，应当使用cookie。**
    - **单个cookie保存的数据不能超过4K，很多浏览器都限制一个站点最多保存20个cookie。(Session对象没有对存储的数据量的限制，其中可以保存更为复杂的数据类型)**

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

- python导入模块失败方法

  > import sys
  >
  > sys.path.append("路径")
  >
  > print(sys.path)  #查看包含导入模块的路径

- python堆

  heapq默认为小根堆

  - heapq.heappush(heap,item) 添加元素

    > import heapq
    >
    > h = []
    >
    > heapq.heappush(h,2)

  - heapq.heapify(list) 将列表转换为堆

    > list = [1,2,7,10,4]
    >
    > heapq.heapify(list)

  - heapq.heappop(heap) 删除最小值

    > heapq.heappop(list)

  - heapq.nlargest(n,heap) 查询堆中的最大元素，n表示查询元素个数

    > list = [3, 5, 5, 9, 6, 6, 8, 99]
    >
    > heapq.nlargest(3,list)
    >
    > out: [99,9,8]

  - heapq.nsmallest(n,heap)

    > list = [3, 5, 5, 9, 6, 6, 8, 99]
    >
    > heapq.nsmallest(3,list)
    >
    > out: [3,5,5]

#### Spring boot相关

- 整合shiro 安全框架

- 整合elasticsearch 全文搜索

- hutool工具包，Hutool是一个Java工具包类库，对文件、流、加密解密、转码、正则、线程、XML等JDK方法进行封装，组成各种Util工具类 [Hutool](https://www.hutool.cn/)

- RestTemplate

- @Resource  @Autowired

- Swagger

- RPC 和 HTTP

- 全局异常 @ControllerAdvice

- 数据校验 JSR-303验证框架

- @Transactional

  >@Transactional 是声明式事务管理 编程中使用的注解
  >
  >添加位置
  >
  >1）接口实现类或接口实现方法上，而不是接口类中。
  >2）访问权限：public 的方法才起作用。@Transactional 注解应该只被应用到 public 方法上，这是由 Spring AOP 的本质决定的。
  >系统设计：将标签放置在需要进行事务管理的方法上，而不是放在所有接口实现类上：只读的接口就不需要事务管理，由于配置了@Transactional就需要AOP拦截及事务的处理，可能影响系统性能。
  >
  >属性设置
  >
  >https://www.jianshu.com/p/907a895587bf

- 令牌桶

  限流是对某一时间窗口内的请求数进行限制，保持系统的可用性和稳定性，防止因流量暴增而导致的系统运行缓慢或宕机。常用的限流算法有令牌桶和和漏桶，而Google开源项目Guava中的RateLimiter使用的就是令牌桶控制算法。

  令牌桶算法的原理是系统以恒定的速率产生令牌，然后把令牌放到令牌桶中，令牌桶有一个容量，当令牌桶满了的时候，再向其中放令牌，那么多余的令牌会被丢弃；当想要处理一个请求的时候，需要从令牌桶中取出一个令牌，如果此时令牌桶中没有令牌，那么则拒绝该请求。

  <img src="C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210728111325543.png" alt="image-20210728111325543" style="zoom:67%;" />

  简单实现：

  ```java
  import java.util.concurrent.ArrayBlockingQueue;
  import java.util.concurrent.Executors;
  import java.util.concurrent.TimeUnit;
  
  public class TokenLimiter {
      private ArrayBlockingQueue<String> blockingQueue;
      private int limit;
      private TimeUnit timeUnit;
      private int period;
  
      public TokenLimiter(int limit, int period, TimeUnit timeUnit){
          this.limit = limit;
          this.period = period;
          this.timeUnit = timeUnit;
          blockingQueue = new ArrayBlockingQueue<>(limit);
          init();
          start();
      }
  
      /**
       * 初始化令牌桶
       */
      public void init(){
          for(int i = 0; i < limit; ++i){
              blockingQueue.add("1");
          }
      }
  
      /**
       * 获取令牌，如果令牌桶为空，返回false
       */
      public boolean tryAcquire(){
          return blockingQueue.poll() == null ? false : true;
      }
  
      private void addTokend(){
          blockingQueue.offer("1");
      }
  
      /**
       * 开启一个定时线程执行添加令牌
       */
      private void start(){
          Executors.newScheduledThreadPool(1).scheduleAtFixedRate(() -> addTokend(), 10, period, timeUnit);
      }
  }
  ```

- 秒杀系统学习 https://mp.weixin.qq.com/s?__biz=MzU1NTA0NTEwMg==&mid=2247484174&idx=1&sn=235af7ead49a7d33e7fab52e05d5021f&lang=zh_CN#rd

#### Windows相关

删除服务

```bash
#删除服务的可执行文件后，该服务可能仍然会出现在注册表中。 如果发生这种情况下，请使用命令sc delete从注册表中删除服务的条目。
sc.exe delete "YourServiceName"
```

host文件的位置

> C:\Windows\System32\drivers\etc
>
> 刷新DNS
>
> ipconfig /flushdns 

#### Maven相关

在用maven构建java项目时，最常用的打包命令有mvn package、mvn install、deploy，这三个命令都可完成打jar包或war（当然也可以是其它形式的包）的功能，但这三个命令还是有区别的。

- package命令完成了项目编译、单元测试、打包功能，但没有把打好的可执行jar包（war包或其它形式的包）布署到本地maven仓库和远程maven私服仓库
- install命令完成了项目编译、单元测试、打包功能，同时把打好的可执行jar包（war包或其它形式的包）布署到本地maven仓库，但没有布署到远程maven私服仓库
- deploy命令完成了项目编译、单元测试、打包功能，同时把打好的可执行jar包（war包或其它形式的包）布署到本地maven仓库和远程maven私服仓库

