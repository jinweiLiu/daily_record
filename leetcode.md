### 前缀和

#### 523、连续的子数组和

给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：

- 子数组大小 至少为 2 ，且

- 子数组元素总和为 k 的倍数。

如果存在，返回 true ；否则，返回 false 。

如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数。

示例 1：

```
输入：nums = [23,2,4,6,7], k = 6
输出：true
解释：[2,4] 是一个大小为 2 的子数组，并且和为 6 。
```

示例 2：

```
输入：nums = [23,2,6,4,7], k = 6
输出：true
解释：[23, 2, 6, 4, 7] 是大小为 5 的子数组，并且和为 42 。 
42 是 6 的倍数，因为 42 = 7 * 6 且 7 是一个整数。
```

**解题思路**：

(1) 直接前缀和，时间复杂度为O(n^2)

(2) 前缀和+哈希表

- 创建一个哈希表，keykey来储存当前前缀和的余数，valuevalue则储存对应的indexindex

- 如果哈希表中存在其对应的余数，我们则取出其pospos，看当前的下标 indexindex 到 pospos的距离是否大于2.（题目要求）如果是则返回true。不是我们则继续遍历。不要更新哈希表中的下标！(贪心的思维)

- 如果不存在则将当前余数与其对应的下标储存在哈希表中。

你问我答：为什么找到了相同的余数就相当于找到了一个连续的前缀和是 k*k* 的倍数？

![image-20210602085930529](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210602085930529.png)

**题解代码**：

```java
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        int m = nums.length;
        if (m < 2) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        map.put(0, -1);
        int remainder = 0;
        for (int i = 0; i < m; i++) {
            remainder = (remainder + nums[i]) % k;
            if (map.containsKey(remainder)) {
                int prevIndex = map.get(remainder);
                if (i - prevIndex >= 2) {
                    return true;
                }
            } else {
                map.put(remainder, i);
            }
        }
        return false;
    }
}
```

#### 525、连续数组

给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。

示例 1:

```
输入: nums = [0,1]
输出: 2
说明: [0, 1] 是具有相同数量0和1的最长连续子数组。
```

示例 2:

```
输入: nums = [0,1,0]
输出: 2
说明: [0, 1] (或 [1, 0]) 是具有相同数量0和1的最长连续子数组。
```

**解题思路**：

前缀和 + 哈希表，利用counter变量记录1和0的数目，如果为1，则加1，如果为0，则减1。遍历过程寻找map中与当前counter值相同的key，相减求取最长连续数组。

**题解代码**：

```java
class Solution {
    public int findMaxLength(int[] nums) {
        int n = nums.length;
        Map<Integer,Integer> map = new HashMap<>();
        int maxLen = 0;
        int counter = 0;
        map.put(counter,-1);
        for(int i = 0; i < n; ++i){
            int num = nums[i];
            if(num == 1){
                counter ++;
            }else{
                counter --;
            }
            if(map.containsKey(counter)){
                int pre = map.get(counter);
                maxLen = Math.max(maxLen,i - pre);
            }else{
                map.put(counter,i);
            }
        }
        return maxLen;
    }
}
```

### 动态规划

#### 1049、最后一块石头的重量II

有一堆石头，用整数数组 stones 表示。其中 stones[i] 表示第 i 块石头的重量。

每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。

 

示例 1：

```
输入：stones = [2,7,4,1,8,1]
输出：1
解释：
组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]，
组合 7 和 8，得到 1，所以数组转化为 [2,1,1,1]，
组合 2 和 1，得到 1，所以数组转化为 [1,1,1]，
组合 1 和 1，得到 0，所以数组转化为 [1]，这就是最优值。
```

示例 2：

```
输入：stones = [31,26,33,21,40]
输出：5
```

示例 3：

```
输入：stones = [1,2]
输出：1
```

**解题思路**：无论以何种顺序粉碎石头，最后一块石头的重量总是可以表示成
$$
\sum_{i=1}^{n-1}{k_i}\times stones_i, k_i\in {-1,1}
$$
记石头的总重量为sum, k<sub>i</sub> = -1的石头的重量之和为neg, 则其余的k<sub>i</sub> = 1的石头的重量之和为sum - neg。则有：
$$
\sum_{i=0}^{n-1}{k_i}\times stones_i = (sum - neg) -neg = sum -2\cdot neg
$$
要使最后一块石头的重量尽可能地小，neg需要在不超过⌊*sum*/2⌋ 的前提下尽可能地大。

定义二维布尔数组dp，其中$\textit{dp}[i+1][j]$ 表示前i个石头能否凑出重量j。特别地，$\textit{dp}[0][]$ 为不选任何石头的状态，因此除了 $\textit{dp}[0][0]$为真，其余 $\textit{dp}[0][j]$全为假。

对于第 i 个石头，考虑凑出重量 j：

- 若$ j<\textit{stones}[i]$，则不能选第 i 个石头，此时有$ \textit{dp}[i+1][j]=\textit{dp}[i][j]$；
- 若$ j\ge \textit{stones}[i]$，存在选或不选两种决策，不选时有 $\textit{dp}[i+1][j]=\textit{dp}[i][j]$，选时需要考虑能否凑出重量 $j-\textit{stones}$[i]，即 $\textit{dp}[i+1][j]=\textit{dp}[i][j-\textit{stones}[i]]$。若二者均为假则 $\textit{dp}[i+1][j]$ 为假，否则$ \textit{dp}[i+1][j]$ 为真。

因此状态转移方程如下：
$$
\textit{dp}[i+1][j]= \begin{cases} \textit{dp}[i][j],& j<\textit{stones}[i] \\ \textit{dp}[i][j] \lor \textit{dp}[i][j-\textit{stones}[i]], & j\ge \textit{stones}[i] \end{cases}
$$
其中 $\lor $表示逻辑或运算。求出$ \textit{dp}[n][] $后，所有为真的 $\textit{dp}[n][j]$ 中，最大的 $j$ 即为 $\textit{neg}$ 能取到的最大值。代入 $\textit{sum}-2\cdot\textit{neg}$ 中即得到最后一块石头的最小重量。

**题解代码**：

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for (int weight : stones) {
            sum += weight;
        }
        int n = stones.length, m = sum / 2;
        boolean[][] dp = new boolean[n + 1][m + 1];
        dp[0][0] = true;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= m; ++j) {
                if (j < stones[i]) {
                    dp[i + 1][j] = dp[i][j];
                } else {
                    dp[i + 1][j] = dp[i][j] || dp[i][j - stones[i]];
                }
            }
        }
        for (int j = m;; --j) {
            if (dp[n][j]) {
                return sum - 2 * j;
            }
        }
    }
}
```

#### 518、零钱兑换II

给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 

示例 1:

```
输入: amount = 5, coins = [1, 2, 5]
输出: 4
解释: 有四种方式可以凑成总金额:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

示例 2:

```
输入: amount = 3, coins = [2]
输出: 0
解释: 只用面额2的硬币不能凑成总金额3。
```

**解题思路**：

通过动态规划的方法计算可能的组合数。用 $\textit{dp}[x]$ 表示金额之和等于 xx 的硬币组合数，目标是求$ \textit{dp}[\textit{amount}]$。

- 初始化$ \textit{dp}[0]=1$；
- 遍历 $coins$，对于其中的每个元素 $coin$，进行如下操作：
  - 遍历 $i$ 从 $coin$ 到 $amount$，将 $dp[i−coin] $的值加到 $dp[i]$。
- 最终得到 $\textit{dp}[\textit{amount}]$ 的值即为答案。

**题解代码**：

```python
class Solution {
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }
}
```

279、完全平方数

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。 

示例 1：

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

示例 2：

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

**题解代码**：

```java
class Solution {
    public int numSquares(int n) {
        int[] f = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            int minn = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                minn = Math.min(minn, f[i - j * j]);
            }
            f[i] = minn + 1;
        }
        return f[n];
    }
}
```

