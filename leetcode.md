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

### 哈希表

#### 17.11、大餐计数

大餐 是指 恰好包含两道不同餐品 的一餐，其美味程度之和等于 2 的幂。

你可以搭配 任意 两道餐品做一顿大餐。

给你一个整数数组 deliciousness ，其中 deliciousness[i] 是第 i 道餐品的美味程度，返回你可以用数组中的餐品做出的不同 大餐 的数量。结果需要对 109 + 7 取余。

注意，只要餐品下标不同，就可以认为是不同的餐品，即便它们的美味程度相同。

示例 1：

```
输入：deliciousness = [1,3,5,7,9]
输出：4
解释：大餐的美味程度组合为 (1,3) 、(1,7) 、(3,5) 和 (7,9) 。
它们各自的美味程度之和分别为 4 、8 、8 和 16 ，都是 2 的幂。
```

**解题思路**：

两层遍历时间复杂度为O(n^2)

使用哈希表

**题解代码**：

```java
class Solution {
    public int countPairs(int[] deliciousness) {
       final int mod = 1000000007;
       int maxVal = -1;
       for(int val : deliciousness){
           maxVal = Math.max(maxVal,val);
       }
       int sum = maxVal * 2;
       int pairs = 0;
       Map<Integer,Integer> map = new HashMap<>();
       for(int i  = 0; i < deliciousness.length; ++i){
           int del = deliciousness[i];
           for(int j = 1; j <= sum; j <<= 1){
               int count = map.getOrDefault(j - del, 0);
               pairs = (pairs + count) % mod;
           }
           map.put(del,map.getOrDefault(del,0) + 1);
       }
       return pairs;
    }
}
```

### 摩尔投票

#### 17.10、主要元素

数组中占比超过一半的元素称之为主要元素。给你一个 整数 数组，找出其中的主要元素。若没有，返回 -1 。请设计时间复杂度为 O(N) 、空间复杂度为 O(1) 的解决方案。

示例 1：

```
输入：[1,2,5,9,5,9,5,5,5]
输出：5
```

示例 2：

```
输入：[3,2]
输出：-1
```

**解题思路**：

一次遍历，cur为上一步的数，count表示cur的数量，当前数如果和cur相同则count++，否则count--。当count为0时，替换cur。遍历结束后，再遍历一次，统计最终cur的数量是否大于数组长度的一半。

**题解代码**：

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        length = len(nums)
        count = 1
        cur = nums[0]
        for i in range(1,length):
            if count == 0:
                cur = nums[i]
                count = 1
                continue
            if nums[i] != cur:
                count -= 1
            else:
                count += 1
        acc = 0
        for num in nums:
            if num == cur:
                acc += 1
        return cur if acc * 2 >length else -1
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

322、零钱兑换

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

示例 1：

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

示例 2：

```
输入：coins = [2], amount = 3
输出：-1
```

解题思路：

- 回溯

- 动态规划

  $dp[i]$ 表示组成金额i所需最少的硬币数量。

  $$dp[0] = 0, \ dp[i] = min(dp[i], dp[i-coin[j]] + 1)$$

题解代码：

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin,amount+1):
                dp[x] = min(dp[x],dp[x-coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1
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

#### 377、组合总和IV

给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。

题目数据保证答案符合 32 位整数范围。

示例 1：

```
输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。
```

**解题思路**：

完全背包，和上题类似，但是本题求的是排列（组合不强调顺序）

**如果求组合数就是外层for循环遍历物品，内层for遍历背包**。

**如果求排列数就是外层for遍历背包，内层for循环遍历物品**。

**题解代码**：

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0]*(target + 1)
        dp[0] = 1
        for i in range(1,target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i-num]
        return dp[-1];
```

#### 279、完全平方数

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

### 二分查找

#### 374、猜数字大小

猜数字游戏的规则如下：

每轮游戏，我都会从 1 到 n 随机选择一个数字。 请你猜选出的是哪个数字。
如果你猜错了，我会告诉你，你猜测的数字比我选出的数字是大了还是小了。
你可以通过调用一个预先定义好的接口 int guess(int num) 来获取猜测结果，返回值一共有 3 种可能的情况（-1，1 或 0）：

-1：我选出的数字比你猜的数字小 pick < num
1：我选出的数字比你猜的数字大 pick > num
0：我选出的数字和你猜的数字一样。恭喜！你猜对了！pick == num
返回我选出的数字。 

示例 1：

```
输入：n = 10, pick = 6
输出：6
```

示例 2：

```
输入：n = 1, pick = 1
输出：1
```

**解题思路**：

二分查找，注意二分查找边界条件

- $left < right$
- $left <= right$

**题解代码**：

```java
/** 
 * Forward declaration of guess API.
 * @param  num   your guess
 * @return 	     -1 if num is lower than the guess number
 *			      1 if num is higher than the guess number
 *               otherwise return 0
 * int guess(int num);
 */

public class Solution extends GuessGame {
    public int guessNumber(int n) {
        int left = 1, right = n;
        while (left < right) { // 循环直至区间左右端点相同
            int mid = left + (right - left) / 2; // 防止计算时溢出
            if (guess(mid) <= 0) {
                right = mid; // 答案在区间 [left, mid] 中
            } else {
                left = mid + 1; // 答案在区间 [mid+1, right] 中
            }
        }
        // 此时有 left == right，区间缩为一个点，即为答案
        return left;
    }
}
```

#### 852、山脉数组的峰顶索引

符合下列属性的数组 arr 称为 山脉数组 ：
arr.length >= 3
存在 i（0 < i < arr.length - 1）使得：
arr[0] < arr[1] < ... arr[i-1] < arr[i]
arr[i] > arr[i+1] > ... > arr[arr.length - 1]
给你由整数组成的山脉数组 arr ，返回任何满足 arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1] 的下标 i 。

示例 1：

```
输入：arr = [0,1,0]
输出：1
```

示例 2：

```
输入：arr = [0,2,1,0]
输出：1
```

**解题思路**：

- 一次遍历
- 二分查找

**题解代码**：

```java
class Solution {
    public int peakIndexInMountainArray(int[] arr) {
        int n = arr.length;
        int left = 1, right = n - 2, ans = 0;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (arr[mid] > arr[mid + 1]) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }
}
```

### 数学

#### 877、石子游戏

亚历克斯和李用几堆石子在做游戏。偶数堆石子排成一行，每堆都有正整数颗石子 piles[i] 。

游戏以谁手中的石子最多来决出胜负。石子的总数是奇数，所以没有平局。

亚历克斯和李轮流进行，亚历克斯先开始。 每回合，玩家从行的开始或结束处取走整堆石头。 这种情况一直持续到没有更多的石子堆为止，此时手中石子最多的玩家获胜。

假设亚历克斯和李都发挥出最佳水平，当亚历克斯赢得比赛时返回 true ，当李赢得比赛时返回 false 。

示例：

```
输入：[5,3,4,5]
输出：true
解释：
亚历克斯先开始，只能拿前 5 颗或后 5 颗石子 。
假设他取了前 5 颗，这一行就变成了 [3,4,5] 。
如果李拿走前 3 颗，那么剩下的是 [4,5]，亚历克斯拿走后 5 颗赢得 10 分。
如果李拿走后 5 颗，那么剩下的是 [3,4]，亚历克斯拿走后 4 颗赢得 9 分。
这表明，取前 5 颗石子对亚历克斯来说是一个胜利的举动，所以我们返回 true 。
```

**解题思路**：

- 动态规划

  定义二维数组 $\textit{dp}$，其行数和列数都等于石子的堆数，$\textit{dp}[i][j]$ 表示当剩下的石子堆为下标 i 到下标 j 时，当前玩家与另一个玩家的石子数量之差的最大值，注意当前玩家不一定是先手 Alex。

  只有当 $i \le j$ 时，剩下的石子堆才有意义，因此当 $i>j$ 时，$\textit{dp}[i][j]=0$。

  当 $i=j$ 时，只剩下一堆石子，当前玩家只能取走这堆石子，因此对于所有 $0 \le i < \textit{nums}.\text{length}$，都有 $\textit{dp}[i][i]=\textit{piles}[i]$。

  当 $i<j$ 时，当前玩家可以选择取走 $\textit{piles}[i]$ 或 $\textit{piles}[j]$，然后轮到另一个玩家在剩下的石子堆中取走石子。在两种方案中，当前玩家会选择最优的方案，使得自己的石子数量最大化。因此可以得到如下状态转移方程：

  $$
  \textit{dp}[i][j]=\max(\textit{piles}[i] - \textit{dp}[i+1][j], \textit{piles}[j] - \textit{dp}[i][j-1])
  $$
  最后判断 $\textit{dp}[0][\textit{piles}.\text{length}-1]$ 的值，如果大于 0，则 Alex 的石子数量大于 Lee 的石子数量，因此 Alex 赢得比赛，否则 Lee 赢得比赛。


- 数学

  先手总是赢

**题解代码**：

```java
class Solution {
    public boolean stoneGame(int[] piles) {
        int length = piles.length;
        int[][] dp = new int[length][length];
        for (int i = 0; i < length; i++) {
            dp[i][i] = piles[i];
        }
        for (int i = length - 2; i >= 0; i--) {
            for (int j = i + 1; j < length; j++) {
                dp[i][j] = Math.max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
            }
        }
        return dp[0][length - 1] > 0;
    }
}
```

```java
class Solution {
    public boolean stoneGame(int[] piles) {
        return true;
    }
}
```

#### 65、有效数字

有效数字（按顺序）可以分成以下几个部分：

一个 小数 或者 整数
（可选）一个 'e' 或 'E' ，后面跟着一个 整数
小数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
下述格式之一：
至少一位数字，后面跟着一个点 '.'
至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
一个点 '.' ，后面跟着至少一位数字
整数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
至少一位数字
部分有效数字列举如下：

["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]
部分无效数字列举如下：

["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]
给你一个字符串 s ，如果 s 是一个 有效数字 ，请返回 true 。

示例 1：

```
输入：s = "0"
输出：true
```

**解题思路**：

确定有限状态自动机

> 初始状态
>
> 符号位
>
> 整数部分
>
> 左侧有整数的小数点
>
> 左侧无整数的小数点（根据前面的第二条额外规则，需要对左侧有无整数的两种小数点做区分）
>
> 小数部分
>
> 字符e
>
> 指数部分的符号位
>
> 指数部分的整数部分

<img src="C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210617092443450.png" alt="image-20210617092443450" style="zoom:80%;" />

**题解代码**：

```java
class Solution65 {
    public boolean isNumber(String s) {
        boolean ans = true, occur = false;
        int n = s.length();
        for (int i=0; i<n; i++) {
            char ch = s.charAt(i);
            if (ch == '+' || ch == '-') {
                if (!(i < n-1 && ((s.charAt(i+1)-'0' >= 0 && s.charAt(i+1)-'0' <= 9) || s.charAt(i+1) == '.')))
                    return false;
            }
            else if (ch == '.') {
                if (!((i > 0 && s.charAt(i-1)-'0'>=0 && s.charAt(i-1)-'0'<=9) ||
                        (i < n-1 && s.charAt(i+1)-'0'>=0 && s.charAt(i+1)-'0'<=9)) || occur)
                    return false;
                occur = true;
            }
            else if (ch == 'e' || ch == 'E') {
                if (i == 0 || i == n-1)
                    return false;
                else {
                    for (int j=i+1; j<n; j++) {
                        char c = s.charAt(j);
                        if ((c == '+' || c == '-') && !(j == i+1 && j != n-1))
                            return false;
                        if (c == '.' || (c-'a'>=0 && c-'a'<=25) || (c-'A'>=0 && c-'A'<=25))
                            return false;
                    }
                    break;
                }
            }
            else if (ch-'0' >= 0 && ch-'0'<=9) {
                if (i < n-1 && (s.charAt(i+1) == '+' || s.charAt(i+1) == '-'))
                    return false;
            }
            else
                return false;
        }
        return ans;
    }
}
```

#### 483、最小好进制

对于给定的整数 n, 如果n的k（k>=2）进制数的所有数位全为1，则称 k（k>=2）是 n 的一个好进制。

以字符串的形式给出 n, 以字符串的形式返回 n 的最小好进制。

示例 1：

```
输入："13"
输出："3"
解释：13 的 3 进制是 111。
```

示例 2：

```
输入："4681"
输出："8"
解释：4681 的 8 进制是 11111。
```

**解题思路**：

假设正整数 n 在 $k~(k \geq 2)$ 进制下的所有数位都为 1，且位数为 m + 1，那么有：

$n = k^0 + k^1 + k^2 + \dots + k^m$
首先讨论两种特殊情况：

m=0，此时 $n=1$，而题目保证 $n \geq 3$，所以本题中 $m>0$。
m=1，此时 $n=(11)_k$，即 $k=n-1\geq 2$，这保证了本题有解。

然后我们分别证明一般情况下的两个结论，以帮助解决本题。

**结论一：** $m < log_k{n}$  (等比数列求和)

**结论二：** $k <\sqrt[m]{n} < k+1$  (二项式定理)

**题解代码**：

```java
class Solution {
    public String smallestGoodBase(String n) {
        long nVal = Long.parseLong(n);
        int mMax = (int) Math.floor(Math.log(nVal) / Math.log(2));
        for (int m = mMax; m > 1; m--) {
            int k = (int) Math.pow(nVal, 1.0 / m);
            long mul = 1, sum = 1;
            for (int i = 0; i < m; i++) {
                mul *= k;
                sum += mul;
            }
            if (sum == nVal) {
                return Integer.toString(k);
            }
        }
        return Long.toString(nVal - 1);
    }
}
```

### 回溯

#### 1239、串联字符串的最长长度

给定一个字符串数组 arr，字符串 s 是将 arr 某一子序列字符串连接所得的字符串，如果 s 中的每一个字符都只出现过一次，那么它就是一个可行解。

请返回所有可行解 s 中最长长度。

示例 1：

```
输入：arr = ["un","iq","ue"]
输出：4
解释：所有可能的串联组合是 "","un","iq","ue","uniq" 和 "ique"，最大长度为 4。
```

示例 2：

```
输入：arr = ["cha","r","act","ers"]
输出：6
解释：可能的解答有 "chaers" 和 "acters"。
```

**解题思路**：

回溯 + 位运算

将字符串表示位二进制形式（mask >> ch  1 << ch）

**题解代码**：

```java
class Solution {
    int ans = 0;

    public int maxLength(List<String> arr) {
        List<Integer> masks = new ArrayList<>();
        for(String s : arr){
            int mask = 0;
            for(int i = 0; i < s.length(); ++i){
                int ch = s.charAt(i) - 'a';
                if(((mask >> ch) & 1) != 0){ // 若 mask 已有 ch，则说明 s 含有重复字母，无法构成可行解
                    mask = 0;
                    break;
                }
                mask |= 1 << ch; // 将 ch 加入 mask 中
            }
            if(mask > 0){
                masks.add(mask);
            }
        }
        backtrack(masks,0,0);
        return ans;
    }
    public void backtrack(List<Integer> masks, int pos, int mask){
        if(pos == masks.size()){
            ans = Math.max(ans,Integer.bitCount(mask));
            return;
        }
        if((mask & masks.get(pos)) == 0){ // mask 和 masks[pos] 无公共元素
            backtrack(masks,pos+1,mask | masks.get(pos));
        }
        backtrack(masks,pos+1,mask);  //不选择当前字符
    }
}
```

#### 剑指 Offer 38. 字符串的排列

输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

示例:

```
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

**解题思路**：

回溯法，需要去重，去重代码：

```java
if(vis[j] || (j>0 && !vis[j-1] && arr[j-1] == arr[j])){
    continue;
}
```

**题解代码**：

```java
class Solution {
    List<String> res;
    boolean[] vis;
    public String[] permutation(String s) {
        int n = s.length();
        res = new ArrayList<String>();
        vis = new boolean[n];
        char[] arr = s.toCharArray();
        Arrays.sort(arr);
        StringBuffer perm = new StringBuffer();
        backtrack(arr, n, perm);
        int size = res.size();
        String[] recArr = new String[size];
        for (int i = 0; i < size; i++) {
            recArr[i] = res.get(i);
        }
        return recArr;
    }
    public void backtrack(char []arr, int n, StringBuffer perm){
        if(perm.length() == n){
            res.add(perm.toString());
            return;
        }
        for(int j = 0; j < n; ++j){
            if(vis[j] || (j>0 && !vis[j-1] && arr[j-1] == arr[j])){
                continue;
            }
            vis[j] = true;
            perm.append(arr[j]);
            backtrack(arr,n,perm);
            perm.deleteCharAt(perm.length() - 1);
            vis[j] = false;
        }
    }
}
```

#### 698、划分k个相等的子集

给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。

示例 1：

```
输入： nums = [4, 3, 2, 3, 5, 2, 1], k = 4
输出： True
说明： 有可能将其分成 4 个子集（5），（1,4），（2,3），（2,3）等于总和。
```

**解题思路**：



**题解代码**：

```

```

### 贪心

#### 406、根据身高重建队列

假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。

请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

示例 1：

```
输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
```

**解题思路**：

考虑是依据哪个维度进行排序，先以身高h进行排序，若身高相同k值小的排在前面；接着按照排序的顺序依次插入，插入的位置依据维度k。

**题解代码**：

```java
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0]) {
                    return Integer.compare(o2[0],o1[0]);
                } else {
                    return Integer.compare(o1[1],o2[1]);
                }
            }
        });
        LinkedList<int[]> que = new LinkedList<>();

        for (int[] p : people) {
            que.add(p[1],p);
        }

        return que.toArray(new int[people.length][]);
    }
}
```

#### 452、用最少数量的剪引爆气球

在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。

一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。


示例 1：

```
输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
```

示例 2：

```
输入：points = [[1,2],[3,4],[5,6],[7,8]]
输出：4
```

**解题思路**：

按照数组的起始位置进行排序，然后从前向后遍历，如果气球重叠了更新重叠气球最小右边界

$points[i][1] = min(points[i - 1][1], points[i][1])$ 

否则弓箭数量加一

**题解代码**：

```java
class Solution {
    public int findMinArrowShots(int[][] points) {
        if(points.length == 0) return 0;
        Arrays.sort(points,(o1,o2) -> Integer.compare(o1[0],o2[0]));

        int count = 1;
        for(int i = 1; i < points.length; i++){
            if(points[i][0] > points[i-1][1]){
                count++;
            }else{
                points[i][1] = Math.min(points[i][1],points[i-1][1]);
            }
        }
        return count;
    }
}
```

### BFS

#### 752、打开转盘锁

你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。

列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 target 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。

示例 1:

```
输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。
```

**解题思路**：

广度优先搜索，使用队列

**题解代码**：

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        if("0000".equals(target)){
            return 0;
        }
        Set<String> dead = new HashSet<>();
        for(String deadend : deadends){
            dead.add(deadend);
        }
        if(dead.contains("0000")){
            return -1;
        }

        int step = 0;
        Queue<String> queue = new LinkedList<>();
        queue.offer("0000");
        Set<String> seen = new HashSet<>();
        seen.add("0000");

        while(!queue.isEmpty()){
            ++step;
            int size = queue.size();
            for(int i = 0; i < size; ++i){
                String status = queue.poll();
                for(String newStatus : get(status)){
                    if(!seen.contains(newStatus) && !dead.contains(newStatus)){
                        if(newStatus.equals(target)){
                            return step;
                        }
                        queue.offer(newStatus);
                        seen.add(newStatus);
                    }
                }
            }
        }
        return -1;
    }

    public char numPrev(char x){
        return x == '0' ? '9' : (char)(x - 1);
    }

    public char numSucc(char x){
        return x == '9' ? '0' : (char)(x + 1);
    }

    public List<String> get(String status){
        List<String> ret = new ArrayList<>();
        char[] array = status.toCharArray();
        for(int i = 0; i < 4; ++i){
            char num = array[i];
            array[i] = numPrev(num);
            ret.add(new String(array));
            array[i] = numSucc(num);
            ret.add(new String(array));
            array[i] = num;
        }
        return ret;
    }
}
```

#### 773、滑动谜题

在一个 2 x 3 的板上（board）有 5 块砖瓦，用数字 1~5 来表示, 以及一块空缺用 0 来表示.

一次移动定义为选择 0 与一个相邻的数字（上下左右）进行交换.

最终当板 board 的结果是 [[1,2,3],[4,5,0]] 谜板被解开。

给出一个谜板的初始状态，返回最少可以通过多少次移动解开谜板，如果不能解开谜板，则返回 -1 。

示例：

```
输入：board = [[1,2,3],[4,0,5]]
输出：1
解释：交换 0 和 5 ，1 步完成
```

```
输入：board = [[1,2,3],[5,4,0]]
输出：-1
解释：没有办法完成谜板
```

**解题思路**：

广度优先搜索，使用队列

**题解代码**：

```java
class Solution {
    int [][] neighbors = {{1,3},{0,2,4},{1,5},{0,4},{1,3,5},{2,4}};
    public int slidingPuzzle(int[][] board) {
        StringBuffer sb = new StringBuffer();
        for(int i = 0; i < 2; ++i){
            for(int j= 0; j < 3; ++j){
                sb.append(board[i][j]);
            }
        }
        String initial = sb.toString();
        if("123450".equals(initial)){
            return 0;
        }
        int step = 0;
        Queue<String> queue = new LinkedList<>();
        queue.offer(initial);
        Set<String> seen = new HashSet<>();
        seen.add(initial);
        while(!queue.isEmpty()){
            ++step;
            int size = queue.size();
            for(int i = 0; i < size; ++i){
                String status = queue.poll();
                for(String nextStatus:get(status)){
                    if(!seen.contains(nextStatus)){
                        if("123450".equals(nextStatus)){
                            return step;
                        }
                        queue.offer(nextStatus);
                        seen.add(nextStatus);
                    }
                }
            }
        }
        return -1;
    }

    public List<String> get(String status){
        List<String> ret = new ArrayList<>();
        char[] array = status.toCharArray();
        int x = status.indexOf('0');
        for(int y : neighbors[x]){
            swap(array,x,y);
            ret.add(new String(array));
            swap(array,x,y);
        }
        return ret;
    }
    public void swap(char[] array, int x, int y){
        char temp = array[x];
        array[x] = array[y];
        array[y] = temp;
    }
}
```

#### 815、公交路线

给你一个数组 routes ，表示一系列公交线路，其中每个 routes[i] 表示一条公交线路，第 i 辆公交车将会在上面循环行驶。

例如，路线 routes[0] = [1, 5, 7] 表示第 0 辆公交车会一直按序列 1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ... 这样的车站路线行驶。
现在从 source 车站出发（初始时不在公交车上），要前往 target 车站。 期间仅可乘坐公交车。

求出 最少乘坐的公交车数量 。如果不可能到达终点车站，返回 -1 。

示例 1：

```
输入：routes = [[1,2,7],[3,6,7]], source = 1, target = 6
输出：2
解释：最优策略是先乘坐第一辆公交车到达车站 7 , 然后换乘第二辆公交车到车站 6 。 
```

示例 2：

```
输入：routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]], source = 15, target = 12
输出：-1
```

**解题思路**：

BFS  建立不同路线中相同车站的映射关系

**题解代码**：

```java
class Solution {
    public int numBusesToDestination(int[][] routes, int source, int target) {
        if(source == target){
            return 0;
        }

        int n = routes.length;
        boolean[][] edge = new boolean[n][n];
        Map<Integer,List<Integer>> res = new HashMap<Integer,List<Integer>>();
        for(int i = 0; i < n; ++i){
            for(int site: routes[i]){
                List<Integer> list = res.getOrDefault(site,new ArrayList<>());
                for(int j : list){
                    edge[i][j] = edge[j][i] = true;
                }
                list.add(i);
                res.put(site,list);
            }
        }

        int[] dis = new int[n];
        Arrays.fill(dis,-1);
        Queue<Integer> que = new LinkedList<>();
        for(int bus: res.getOrDefault(source,new ArrayList<>())){
            dis[bus] = 1;
            que.offer(bus);
        }
        while(!que.isEmpty()){
            int x = que.poll();
            for(int y = 0; y < n; ++y){
                if(edge[x][y] && dis[y] == -1){
                    dis[y] = dis[x] + 1;
                    que.offer(y);
                }
            }
        }
        int ret = Integer.MAX_VALUE;
        for(int bus: res.getOrDefault(target,new ArrayList<>())){
            if(dis[bus] != -1){
                ret = Math.min(ret,dis[bus]);
            }
        }
        return ret == Integer.MAX_VALUE ? -1 : ret;
    }
}
```

### DFS

#### 剑指Offer 37. 序列化二叉树

请实现两个函数，分别用来序列化和反序列化二叉树。

你需要设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

<img src="C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210630091605259.png" alt="image-20210630091605259" style="zoom:67%;" />

**解题思路**：

先序、中序、后序遍历都可以

这里先序遍历这颗二叉树，遇到空子树的时候序列化成 None，否则继续递归序列化。那么我们如何反序列化呢？首先我们需要根据 , 把原先的序列分割开来得到先序遍历的元素列表，然后从左向右遍历这个序列：

- 如果当前的元素为 None，则当前为空树
- 否则先解析这棵树的左子树，再解析它的右子树

**题解代码**：

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {
    public String serialize(TreeNode root) {
        return rserialize(root, "");
    }
  
    public TreeNode deserialize(String data) {
        String[] dataArray = data.split(",");
        List<String> dataList = new LinkedList<String>(Arrays.asList(dataArray));
        return rdeserialize(dataList);
    }

    public String rserialize(TreeNode root, String str) {
        if (root == null) {
            str += "None,";
        } else {
            str += str.valueOf(root.val) + ",";
            str = rserialize(root.left, str);
            str = rserialize(root.right, str);
        }
        return str;
    }
  
    public TreeNode rdeserialize(List<String> dataList) {
        if (dataList.get(0).equals("None")) {
            dataList.remove(0);
            return null;
        }
  
        TreeNode root = new TreeNode(Integer.valueOf(dataList.get(0)));
        dataList.remove(0);
        root.left = rdeserialize(dataList);
        root.right = rdeserialize(dataList);
    
        return root;
    }
}
```

#### LCP 07、传递信息

小朋友 A 在和 ta 的小伙伴们玩传信息游戏，游戏规则如下：

有 n 名玩家，所有玩家编号分别为 0 ～ n-1，其中小朋友 A 的编号为 0
每个玩家都有固定的若干个可传信息的其他玩家（也可能没有）。传信息的关系是单向的（比如 A 可以向 B 传信息，但 B 不能向 A 传信息）。
每轮信息必须需要传递给另一个人，且信息可重复经过同一个人
给定总玩家数 n，以及按 [玩家编号,对应可传递玩家编号] 关系组成的二维数组 relation。返回信息从小 A (编号 0 ) 经过 k 轮传递到编号为 n-1 的小伙伴处的方案数；若不能到达，返回 0。

示例 1：

```
输入：n = 5, relation = [[0,2],[2,1],[3,4],[2,3],[1,4],[2,0],[0,4]], k = 3
输出：3
解释：信息从小 A 编号 0 处开始，经 3 轮传递，到达编号 4。共有 3 种方案，分别是 0->2->0->4， 0->2->1->4， 0->2->3->4。
```

示例 2：

```
输入：n = 3, relation = [[0,2],[2,1]], k = 2
输出：0
解释：信息不能从小 A 处经过 2 轮传递到编号 2
```

解题思路：

DFS或者BFS，将输入转换为一对多的关系形式

题解代码：

```java
class Solution {
    int ways, n, k;
    List<List<Integer>> edges;

    public int numWays(int n, int[][] relation, int k) {
        ways = 0;
        this.n = n;
        this.k = k;
        edges = new ArrayList<List<Integer>>();
        for (int i = 0; i < n; i++) {
            edges.add(new ArrayList<Integer>());
        }
        for (int[] edge : relation) {
            int src = edge[0], dst = edge[1];
            edges.get(src).add(dst);
        }
        dfs(0, 0);
        return ways;
    }

    public void dfs(int index, int steps) {
        if (steps == k) {
            if (index == n - 1) {
                ways++;
            }
            return;
        }
        List<Integer> list = edges.get(index);
        for (int nextIndex : list) {
            dfs(nextIndex, steps + 1);
        }
    }
}
```

### 动态规划

#### 01背包

$dp[i][j]$表示从下标为[0~i]的物品里任意取，放进容量为j的背包，价值总和最大是多少。

 递推公式：$dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);$

```java
public static void main(String[] args) {
    int[] weight = {1, 3, 4};
    int[] value = {15, 20, 30};
    int bagSize = 4;
    testWeightBagProblem(weight, value, bagSize);
}

public static void testWeightBagProblem(int[] weight, int[] value, int bagSize){
    int wLen = weight.length, value0 = 0;
    //定义dp数组：dp[i][j]表示背包容量为j时，前i个物品能获得的最大价值
    int[][] dp = new int[wLen + 1][bagSize + 1];
    //初始化：背包容量为0时，能获得的价值都为0
    for (int i = 0; i <= wLen; i++){
        dp[i][0] = value0;
    }
    //遍历顺序：先遍历物品，再遍历背包容量
    for (int i = 1; i <= wLen; i++){
        for (int j = 1; j <= bagSize; j++){
            if (j < weight[i - 1]){
                dp[i][j] = dp[i - 1][j];
            }else{
                dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - weight[i - 1]] + value[i - 1]);
            }
        }
    }
    //打印dp数组
    for (int i = 0; i <= wLen; i++){
        for (int j = 0; j <= bagSize; j++){
            System.out.print(dp[i][j] + " ");
        }
        System.out.print("\n");
    }
}
```

#### 343、整数拆分

给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

示例 1:

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
```

示例 2:

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
```

**解题思路**：

$dp[i]$表示拆分数字 $i$ 得到的最大乘积​

$dp[i] = max\{dp[i],j*(i-j),j*dp[i-j]\}$

**题解代码**：

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[n]
```

#### 96、不同的二叉搜索树

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

<img src="C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210630093131859.png" alt="image-20210630093131859" style="zoom:67%;" />

**解题思路**：

$dp[ \ ]$表示1到i为节点组成的二叉搜索树的个数，例如dp[3]，就是 元素1为头结点搜索树的数量 + 元素2为头结点搜索树的数量 + 元素3为头结点搜索树的数量

 $ dp[i] += dp[$以j为头结点左子树节点数量$] * dp[$以j为头结点右子树节点数量$] $

所以递推公式：$dp[i] += dp[j - 1] * dp[i - j]$ ，j-1 为j为头结点左子树节点数量，i-j 为以j为头结点右子树节点数量

**题解代码**：

```java
class Solution {
    public int numTrees(int n) {
        int []dp = new int[n+1];
        dp[0] = 1;
        for(int i= 1; i < n+1; ++i){
            for(int j = 1; j <= i; ++j){
                dp[i] += dp[j-1] * dp[i-j];
            }
        }
        return dp[n];
    }
}
```

#### 416、分割等和子集

类似题：[1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/)

给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

示例 1：

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```

示例 2：

```
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集
```

**解题思路**：

DFS 或者 动态规划

**题解代码**：

动态规划

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        if (n < 2) {
            return false;
        }
        int sum = 0, maxNum = 0;
        for (int num : nums) {
            sum += num;
            maxNum = Math.max(maxNum, num);
        }
        if (sum % 2 != 0) {
            return false;
        }
        int target = sum / 2;
        if (maxNum > target) {
            return false;
        }
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            for (int j = target; j >= num; --j) {
                dp[j] |= dp[j - num];
            }
        }
        return dp[target];
    }
}
```

DFS

```python
class Solution(object):
    def canPartition(self, nums):
        if not nums: return True
        total = sum(nums)
        if total & 1:  # 和为奇数
            return False
        total = total >> 1  # 除2
        nums.sort(reverse=True)  # 逆排序
        if total < nums[0]:  # 当数组最大值超过总和的一半
            return False
        return self.dfs(nums, total)

    def dfs(self, nums, total):
        if total == 0:
            return True
        if total < 0:
            return False
        for i in range(len(nums)):
            if self.dfs(nums[i+1:], total - nums[i]):  # 除去i及其之前，保证每个数只用一次
                return True
        return False
```

#### 474、一和零

给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。

如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。

示例 1：

```
输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
输出：4
解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
```

示例 2：

```
输入：strs = ["10", "0", "1"], m = 1, n = 1
输出：2
解释：最大的子集是 {"0", "1"} ，所以答案是 2 。
```

**解题思路**：

本题其实是01背包问题！

这不过这个背包有两个维度，一个是m 一个是n，而不同长度的字符串就是不同大小的待装物品。

开始动规五部曲：

1. 确定dp数组（dp table）以及下标的含义

三维定义：定义三维数组 dp，其中 dp\[i]\[j][k] 表示在前 i 个字符串中，使用 j 个 0 和 k 个 1 的情况下最多可以得到的字符串数量。假设数组 str 的长度为 l，则最终答案为 dp\[l]\[m][n]。

二维定义：**dp\[i][j]：最多有i个0和j个1的strs的最大子集的大小为dp\[i][j]**。

2. 确定递推公式

dp\[i][j] 可以由前一个strs里的字符串推导出来，strs里的字符串有zeroNum个0，oneNum个1。

dp\[i][j] 就可以是 dp\[i - zeroNum][j - oneNum] + 1。

然后我们在遍历的过程中，取dp\[i][j]的最大值。

所以递推公式：dp\[i][j] = max(dp\[i][j], dp\[i - zeroNum][j - oneNum] + 1);

此时大家可以回想一下01背包的递推公式：dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);

对比一下就会发现，字符串的zeroNum和oneNum相当于物品的重量（weight[i]），字符串本身的个数相当于物品的价值（value[i]）。

**这就是一个典型的01背包！** 只不过物品的重量有了两个维度而已。


3. dp数组如何初始化

01背包的dp数组初始化为0就可以。因为物品价值不会是负数，初始为0，保证递推的时候dp\[i][j]不会被初始值覆盖。

4. 确定遍历顺序

01背包一定是外层for循环遍历物品，内层for循环遍历背包容量且从后向前遍历！

那么本题也是，物品就是strs里的字符串，背包容量就是题目描述中的m和n。

代码如下：

```java
for (String str : strs) { // 遍历物品
    int oneNum = 0, zeroNum = 0;
    for (char c : str) {
        if (c == '0') zeroNum++;
        else oneNum++;
    }
    for (int i = m; i >= zeroNum; i--) { // 遍历背包容量且从后向前遍历！
        for (int j = n; j >= oneNum; j--) {
            dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
        }
    }
}
```

**题解代码**：

```java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        //dp[i][j]表示i个0和j个1时的最大子集
        int[][] dp = new int[m + 1][n + 1];
        int oneNum, zeroNum;
        for (String str : strs) {
            oneNum = 0;
            zeroNum = 0;
            for (char ch : str.toCharArray()) {
                if (ch == '0') {
                    zeroNum++;
                } else {
                    oneNum++;
                }
            }
            //倒序遍历
            for (int i = m; i >= zeroNum; i--) {
                for (int j = n; j >= oneNum; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
                }
            }
        }
        return dp[m][n];
    }
}
```

#### 139、单词拆分

给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
示例 1：

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```

**解题思路**：

- 回溯

使用记忆化函数，保存出现过的 backtrack(s)，避免重复计算。

定义回溯函数 backtrack(s)

​        若 s 长度为 0，则返回 True，表示已经使用 wordDict 中的单词分割完。

​        初试化当前字符串是否可以被分割 res=False

​        遍历结束索引 i，遍历区间 [1,n+1)：

​                若 s[0,⋯,i−1] 在 wordDict 中：res=backtrack(s[i,⋯,n−1]) or res。解释：保存遍历结束索引中，可以使字符串切割完成的情况。

​        返回 res

返回 backtrack(s)

- 动态规划

单词就是物品，字符串s就是背包，单词能否组成字符串s，就是问物品能不能把背包装满。

拆分时可以重复使用字典中的单词，说明就是一个完全背包！

动规五部曲分析如下：

1. 确定dp数组以及下标的含义

**dp[i] : 字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词**。

2. 确定递推公式

如果确定dp[j] 是true，且 [j, i] 这个区间的子串出现在字典里，那么dp[i]一定是true。（j < i ）。

所以递推公式是 if([j, i] 这个区间的子串出现在字典里 && dp[j]是true) 那么 dp[i] = true。

3. dp数组如何初始化

从递归公式中可以看出，dp[i] 的状态依靠 dp[j]是否为true，那么dp[0]就是递归的根基，dp[0]一定要为true，否则递归下去后面都都是false了。

那么dp[0]有没有意义呢？

dp[0]表示如果字符串为空的话，说明出现在字典里。

但题目中说了“给定一个非空字符串 s” 所以测试数据中不会出现i为0的情况，那么dp[0]初始为true完全就是为了推导公式。

下标非0的dp[i]初始化为false，只要没有被覆盖说明都是不可拆分为一个或多个在字典中出现的单词。

**题解代码**：

```java
//动态规划
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] valid = new boolean[s.length() + 1];
        valid[0] = true;
        for(int i = 1; i <= s.length(); ++i){
            for(int j = 0; j < i; ++j){
                if(wordDict.contains(s.substring(j,i)) && valid[j]){
                    valid[i] = true;
                }
            }
        }
        return valid[s.length()];
    }
}
```

```java
//记忆化回溯
class Solution {
    Set<String> memory = new HashSet<>();
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>();
        for(String str : wordDict){
            set.add(str);
        }
        return DFS(s,set);
    }
    public boolean DFS(String s,Set<String> set){
        if(s.length()==0) return true;
        if(memory.contains(s)) return false;//如果记忆中存在此字符串，返回false，结束递归。
        StringBuilder strb = new StringBuilder();
        for(int i=0;i<s.length();i++){
            strb.append(s.charAt(i));
            if(set.contains(strb.toString()) && !memory.contains(s.substring(i+1))){
                if(DFS(s.substring(i+1),set)){
                    return true;
                }else{
                    memory.add(s.substring(i+1));//对子串失败的情况进行记忆
                }
            }
        }
        memory.add(s);//对s失败的情况进行记忆
        return false;
    }
}
```

