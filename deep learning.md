## 深度学习记录

#### 工具使用

- 开启 tensorboard

  ```bash
  #开启
  tensorboard --logdir='tensorboard的log地址' --port=6006
  #映射到本地16006端口
  ssh -L 16006:127.0.0.1:6006 user@address #用户和服务器地址
  #本地地址
  http://127.0.0.1:16006
  ```


#### 代码相关

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
```

