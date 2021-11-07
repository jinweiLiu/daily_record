### typora数学公式

- 行间和行内公式
  - $$ $$  行间
  - $ $  行内

- 上下标
  - 上标 ^
  - 下标 _

- 换行及空格
  - 换行 \\\
  - 空格

  | 说明         | 表示方式   | 展示效果                                                     | 备注           |
  | ------------ | ---------- | ------------------------------------------------------------ | -------------- |
  | 两个quad空格 | a \qquad b | ![a \qquad b](http://upload.wikimedia.org/math/e/5/0/e505263bc9c94f673c580f3a36a7f08a.png) | 两个*m*的宽度  |
  | quad空格     | a \quad b  | ![a \quad b](http://upload.wikimedia.org/math/d/a/8/da8c1d9effa4501fd80c054e59ad917d.png) | 一个*m*的宽度  |
  | 大空格       | a\ b       | ![a\ b](http://upload.wikimedia.org/math/6/9/2/692d4bffca8e84ffb45cf9d5facf31d6.png) | 1/3*m*宽度     |
  | 中等空格     | a\;b       | ![a\;b](http://upload.wikimedia.org/math/b/5/a/b5ade5d5393fd7727bf77fa44ec8b564.png) | 2/7*m*宽度     |
  | 小空格       | a\,b       | ![a\,b](http://upload.wikimedia.org/math/7/b/e/7bea99aed60ba5e1fe8a134ab43fa85f.png) | 1/6*m*宽度     |
  | 没有空格     | ab         | ![ab\,](http://upload.wikimedia.org/math/b/6/b/b6bd9dba2ebfca24731ae6dc3913e625.png) |                |
  | 紧贴         | a\!b       | ![a\!b](http://upload.wikimedia.org/math/0/f/b/0fbcad5fadb912e8afa6d113a75c83e4.png) | 缩进1/6*m*宽度 |
  
- 开方
  - \sqrt{x^3}   $\sqrt{x^3}$
  - \sqrt[3]{x}   $\sqrt[3]{x}$

- 属于/不属于

  \in  $\in$

  \notin $\notin$

- 希腊字符

  \alpha

  \beta
  
- 大括号

  L = -ylogy'-(1-y)log(1-y') = \begin{cases}-logy',& \text{y=1}\\-log(1-y'),& \text{y=0} \end{cases}
  $$
  L = -ylogy'-(1-y)log(1-y') = \begin{cases}-logy',& \text{y=1}\\-log(1-y'),& \text{y=0} \end{cases}
  $$

- 交集和并集

   \cap $\cap$  \bigcap $\bigcap$

  \cup $\cup$  \bigcup $\bigcup$

- 上划线

   \hat{x}  $\hat{x}$

   \overline{x}  $\overline{x}$

   \widehat{x}  $\widehat{x}$

   \widetilde{x}  $\widetilde{x}$

   \dot{x}  $\dot{x}$

   \ddot{x}  $\ddot{x}$

- 乘号

   \times $\times$

### Latex

多行公式

```latex
\begin{equation}
	\begin{array}{l}
		z = x+f(x), \vspace{1.5ex} \\ %多行公式行距控制\vspace{1.5ex}
		B = z + f(z)
	\end{array}
\end{equation}
```

