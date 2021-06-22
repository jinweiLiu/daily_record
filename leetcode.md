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

