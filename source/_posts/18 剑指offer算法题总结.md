---
title: 剑指offer算法题总结
date: 2022-07-16 19:11:58
---

>总结剑指offer的不同类型的题

## 一、二叉树部分

#### 剑指 Offer 07. 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。
假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

```js
var buildTree = function(preorder, inorder) {
    if (!preorder.length) {
        return null
    }
    let root = new TreeNode(preorder[0]),
        i = inorder.indexOf(preorder[0]);
    root.left = buildTree(preorder.slice(1, i + 1), inorder.slice(0, i))
    root.right = buildTree(preorder.slice(i + 1, ), inorder.slice(i+1, ))
    return root
};
```

#### ！ 剑指 Offer 26. 树的子结构

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

<img src="/Users/onlycat/Library/Application Support/typora-user-images/image-20220718093822440.png" alt="image-20220718093822440" style="zoom:50%;" />

```js
// B是A的子结构：B在A中，B的左子树和右子树是A的子结构
// 首先搜索到B节点，再
const recur = function(A, B) {
    if (!B) return true
    if (!A || A.val !== B.val) return false
    return recur(A.left, B.left) && recur(A.right, B.right)
}
var isSubStructure = function(A, B) {
  	// 这里的Boolean别忘了，简单的说，B肯定要有，A自然也要有
    return Boolean(A && B) && (recur(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B))
};

```



#### 剑指 Offer 27. 二叉树的镜像

`思路，简单一点想，就是左子树和右子树互换，但是换之前，要把左右子树的先做好镜像，所以先对左右子树执行镜像操作，退出条件就是null`

```js
var mirrorTree = function(root) {
    if (root === null) {
        return null
    }
    const left = mirrorTree(root.left)
    const right = mirrorTree(root.right)
    root.left = right
    root.right = left
    return root
};
```



#### 剑指 Offer 28. 对称的二叉树

<img src="/Users/onlycat/Library/Application Support/typora-user-images/image-20220718094538854.png" alt="image-20220718094538854" style="zoom:50%;" />

```js
const check = function(p, q) {
    // 边界条件，不谢的花，后面会报错
    if (!p && !q) return true
    if (!p || !q) return false
    return p.val === q.val && check(p.left, q.right) && check(p.right, q.left)
}
var isSymmetric = function(root) {
    return check(root, root)
};

```



#### 剑指 Offer 32 -1.从上到下打印.

```js
var levelOrder = function(root) {
    if (!root) return []
    let res = [], queue = []
    queue.push(root)
    while (queue.length) {
        let node = queue.shift()
        res.push(node.val)
        if (node.left) queue.push(node.left)
        if (node.right) queue.push(node.right)
    }
    return res
};
```



#### 剑指 Offer 32- ||. 从上到下打印..

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

```js
var levelOrder = function(root) {
    let ans = [], queue = []
    if (root === null) return ans
    queue.push(root)
    while (queue.length > 0) {
        let tmp = []
        for (let i = queue.length; i > 0; i--) {
            node  = queue.shift()
            tmp.push(node.val)
            if (node.left !== null) queue.push(node.left)
            if (node.right !== null) queue.push(node.right)
        }
        ans.push([...tmp])
    }
    return ans
};

```



#### 剑指 Offer 32- |||. 从上到下打印.

实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

```js
var levelOrder = function(root) {
    let res = [], queue = []
    if (!root) return res
    queue.push(root)
    while (queue.length) {
        let tmp = []
        for (let i = queue.length; i > 0; i--) {
            node = queue.shift()
            if (res.length % 2) {
              	// 为奇数，此层为偶数层，则反方向
                tmp.unshift(node.val)
            } else {
                tmp.push(node.val)
            }
            // tmp.push(node.val)
            if (node.left) queue.push(node.left)
            if (node.right) queue.push(node.right)
        }
        res.push(tmp)
    }
    return res
};

```



#### 剑指 Offer 33.二叉搜索树的后序遍历序列…

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

`左子树上的节点都比root小，右子树的节点都比root大，因此只要递归判断该数组是否满足前提即可，判出：数组只有一个元素或者没有元素`

```js
var verifyPostorder = function(postorder) {
    const recur = function(i, j) {
        if (i >= j) return true
        let p = i
        while (postorder[p] < postorder[j]) p++
        const m = p
        while (postorder[p] > postorder[j]) p++
        return p === j && recur(i, m - 1) && recur(m, j - 1)
    }
    return recur(0, postorder.length - 1)
};
```



#### 剑指 Offer 34. 二叉树中和为某一给定值的路径

```js
var pathSum = function(root, target) {
    const res = [], path = [];
    const recur = function(node, tar) {
        if (!node) {
            return 
        }
        path.push(node.val)
        tar -= node.val
        if (tar === 0 && !node.left && !node.right) {
            res.push([...path])
        }
        recur(node.left, tar)
        recur(node.right, tar)
        path.pop()
    }
    recur(root, target)
    return res
```



#### 剑指 Offer 36. 二叉搜索树与双向.

https://leetcode.cn/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

```js
var treeToDoublyList = function(root) {
    function dfs(cur) {
        if (!cur) return 
        dfs(cur.left)
        if (pre) {
            pre.right = cur
            cur.left = pre
            // [pre.right, cur.left] = cur, pre
        } else {
            head = cur
        }
        pre = cur
        dfs(cur.right)
    }
    if (!root) return 
    let pre = null, head = null
    dfs(root)
    head.left = pre
    pre.right = head
    return head
};

```



#### 剑指 Offer 54. 二叉搜索树的第k大节点

https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/

```js
var kthLargest = function(root, k) {
    let res = null
    const dfs = root => {
        if (root) {
            dfs(root.right)
            k--
            if (0 === k) {
                res = root.val
                return
            }
            dfs(root.left)
        }
    }
    dfs(root)
    return res
};
```



#### 剑指 Offer 55-1. 二叉树的深度

```js
var maxDepth = function(root) {
    if (!root) return 0
    let queue = [root], res = 0
    while (queue.length > 0) {
        let tmp = []
        for (const node of queue) {
            if (node.left) tmp.push(node.left)
            if (node.right) tmp.push(node.right)
        }
        queue = tmp
        res++
    }
    return res
};
var maxDepth = function(root) {
    const dfs = (node) => {
        if (node === null) {
            return 0
        }
        return Math.max(dfs(node.left), dfs(node.right)) + 1
    }
    return dfs(root)
};
```



#### 剑指 Offer 55- 1l. 平衡二叉树

如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

```js
var isBalanced = function(root) {
    const depth = function(root) {
      if (!root) return 0
      return Math.max(depth(root.left), depth(root.right)) + 1
		}
    if (!root) return true
    return Math.abs(depth(root.left) - depth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right)
};
```



#### 剑指 Offer 68 -1. 二叉搜索树的最近公共祖先

`如果都大于node.val说明都在node右子树，如果都小于，则说明node左子树，否则返回该节点`

```js
var lowestCommonAncestor = function(root, p, q) {
    let ancestor = root
    while (true) {
        if (p.val < ancestor.val && q.val < ancestor.val) {
            ancestor = ancestor.left
        } else if (p.val > ancestor.val && q.val > ancestor.val) {
            ancestor = ancestor.right
        } else return ancestor
    }
};

```



#### ! 剑指 Offer 68 - II. 二叉树的最近公共祖先

https://leetcode.cn/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/

```js
var lowestCommonAncestor = function(root, p, q) {
    let ans;
    const dfs = (root, p, q) => {
        if (root === null) return false;
        const lson = dfs(root.left, p, q);
        const rson = dfs(root.right, p, q);
        if ((lson && rson) || ((root.val === p.val || root.val === q.val) && (lson || rson))) {
            ans = root;
        } 
        return lson || rson || (root.val === p.val || root.val === q.val);
    }
    dfs(root, p, q);
    return ans;
};

```



### 二、动态规划部分

#### 剑指 Offer 10- I. 斐波那契数列 2430

`注意MOD，看题`

```js
var fib = function(n) {
    const MOD  = 1000000007
    if (n < 2) {
        return n
    }
    let prev = 0, cur = 1;
    for (let i = 2; i <= n; i++) {
        [cur, prev] = [(cur + prev) % MOD, cur]
    }
    return cur
};

```

#### 剑指 Offer 10-I. 青蛙跳台阶问题      1782

`注意初始点和斐波那契不一样，也就是dp[0] = 1`

```js
var numWays = function(n) {
    // dp[i] = dp[i - 1] + dp [i -2]
    const MOD = 1000000007
    if (n < 2) return 1
    let prev = 1, cur = 1
    for (let i = 2; i <= n; i++) {
        [cur, prev] = [(cur + prev) % MOD, cur]
    }
    return cur
};
```



#### 剑指 Offer 14- 1. 剪绳子      1763

https://leetcode.cn/problems/jian-sheng-zi-lcof/

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

`7.28字节一面考到`

`定义子问题也就是状态转移方程，我之前想找到的是dp[i] 和dp[i - 1] 的联系，但这里，并没有什么联系，但是，如果dp[i]这里表示的是长度为i时最大乘积，那么，在把n分成j和i两段的时候，dp[i]这里肯定更大一些`



```js
// 动态规划，
var cuttingRope = function (n) {
    let dp = new Array(n + 1).fill(1)
    // 7.29二刷写成了i < n
    for (let i = 3; i <= n; i++) {
        for (let j = 1; j < i; j++) {
            dp[i] = Math.max(dp[i], j * dp[i - j], j * (i - j))
        }
    }
    return dp[n]
};

// 数学推导
var cuttingRope = function(n) {
    if (n <= 3) {return n - 1} 
    const a = Math.floor(n / 3) 
    const b = n % 3
    if (b === 0) {
        return Math.pow(3, a)
    }
    if (b === 1) {
        return Math.pow(3, a - 1) * 4
    }
    return Math.pow(3, a) * 2
};
```



#### 剑指 Offer 14-川1. 剪绳子！      666

不推荐动态规划了，需要对结果取模

```js
var cuttingRope = function(n) {
    const MOD = 1000000007
    if (n < 4) return n - 1
    let ret = 1
    while(n > 4) {
        n = n - 3
        ret = 3 * ret % MOD
    }
    ret = ret * n % MOD
    return ret
};
```



#### 剑指 Offer 19.正则表达式匹配      597 口

#### 剑指 Offer 42. 连续子数组的最大和      1976

`输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。`

```js
var maxSubArray = function(nums) {
    let pre = 0, maxAns = nums[0]
    nums.forEach((x) => {
        pre = Math.max(pre + x, x)
        maxAns = Math.max(maxAns, pre)
    })
    return maxAns
};

// 7.29三刷
var maxSubArray = function(nums) {
    let prev = 0, cur = nums[0], length = nums.length, max = nums[0];
    for (let i = 0; i < length; i++) {
        cur = prev > 0 ? prev + nums[i] : nums[i]
        prev = cur
        max = Math.max(cur, max)
    }
    return max
};
```



#### 剑指 Offer 43.1~n 整数中1出      670 口

#### 剑指 Offer 46. 把数字翻译成字符串      1996

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

`输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"`

```js
var translateNum = function(num) {
    if(num<10){
        return 1;
    }
    let str = num + '';
    let dp = [1,1];

    for(let i = 1;i<str.length;i++){
            let tmp = parseInt(str.slice(i - 1, i+1), 10) || 0;
      			// 05不能翻译成5，所以要大于等于10
            if(tmp >=10 && tmp <=25){
                dp[i+1] = dp[i-1] + dp[i];
            } else {
                dp[i+1] = dp[i];
            }
    }
    return dp[dp.length-1]
};

// 7.29二刷（看题解）
var translateNum = function(num) {
    if (num < 10) return 1
    num = num + ''
    const dp = [1, 1]
    for (let i = 1; i < num.length; i++) {
        let temp = parseInt(num.slice(i - 1, i + 1))
        if (temp >= 10 && temp <= 25) {
            dp[i + 1] = dp[i] + dp[i - 1]
        } else {
            dp[i + 1] = dp[i]
        }
    }
    return dp[dp.length - 1]
};

```



#### 剑指 Offer 47. 礼物的最大价值      1449

```js
/**
 * @param {number[][]} grid
 * @return {number}
 */
var maxValue = function(grid) {
    // 状态定义: dp[i][j]表示走到第i行j列所能拿到的最大礼物价值
    // 状态转移方程: dp[i][j] = max(dp[i-1][j],dp[i][j-1]) + grid[i][j];
    // 初始化: dp[0][0] = grid[0][0];
    // dp右下角元素
  	const dp = new Array(grid.length).fill().map(_ => new Array(grid[0].length).fill(0))
    dp[0][0] = grid[0][0];
    for(let i = 0;i < grid.length;i ++) {
        for(let j = 0;j < grid[0].length;j ++) {
            if(i == 0 && j == 0) continue;
            if(i == 0) dp[i][j] = dp[i][j-1] + grid[i][j];
            if(j == 0) dp[i][j] = dp[i-1][j] + grid[i][j];
            if(i !== 0 && j !== 0) dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]) + grid[i][j];
        }
    }
    return dp[grid.length-1][grid[0].length-1];
};

// 7.29二刷，注意区分行和列，简单的说，m是哪一行，n是哪一列，
var maxValue = function(grid) {
    const m = grid.length, n = grid[0].length
    const dp = new Array(m).fill().map(_ => new Array(n).fill(0))
    dp[0][0] = grid[0][0]
    for (let i = 1; i < n; i++) {
        dp[0][i] = grid[0][i] + dp[0][i-1]
    }
    for (let j = 1; j < m; j++) {
        dp[j][0] = grid[j][0] + dp[j - 1][0]
    }
    for (let i = 1; i < m; i++) {
        for (let j = 1; j < n; j++) {
            dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        }
    }
    return dp[m - 1][n - 1]
};
```



#### 剑指 Offer 49. 丑数      728

```js
var nthUglyNumber = function(n) {
    const dp = new Array(n + 1).fill(0);
    dp[1] = 1;
    let p2 = 1, p3 = 1, p5 = 1;
    for (let i = 2; i <= n; i++) {
        const num2 = dp[p2] * 2, num3 = dp[p3] * 3, num5 = dp[p5] * 5;
        dp[i] = Math.min(Math.min(num2, num3), num5);
        if (dp[i] === num2) {
            p2++;
        }
        if (dp[i] === num3) {
            p3++;
        }
        if (dp[i] === num5) {
            p5++;
        }
    }
    return dp[n];
};

// 7.29 二刷，看题解
var nthUglyNumber = function(n) {
    let dp = new Array(n + 1).fill(0)
    dp[1] = 1
    let p2 = 1, p3 = 1, p5 = 1
    for (let i = 2; i <= n; i++) {
        const num2 = dp[p2] * 2, num3 = dp[p3] * 3, num5 = dp[p5] * 5;
        dp[i] = Math.min(num2, num3, num5)
        if (dp[i] === num2) p2++
        if (dp[i] === num3) p3++
        if (dp[i] === num5) p5++
    }
    return dp[n]
};

```



#### 剑指 Offer 60.n个骰子的点数      829

不是很懂，算了

```js
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        dp = [1 / 6] * 6
        for i in range(2, n + 1):
            tmp = [0] * (5 * i + 1)
            for j in range(len(dp)):
                for k in range(6):
                    tmp[j + k] += dp[j] / 6
            dp = tmp
        return dp

```



#### 剑指 Offer 63. 股票的最大利润      1378

```js
var maxProfit = function(prices) {
    let minprice = Number.MAX_VALUE;
    let maxprofit = 0;
    for (const price of prices) {
        maxprofit = Math.max(price - minprice, maxprofit);
        minprice = Math.min(price, minprice);
    }
    return maxprofit;
};

// 7.29二刷，无问题，语法不够精简
var maxProfit = function(prices) {
    let length = prices.length
    if (length === 0) return 0
    let ret = 0, 
        minVal = prices[0]
    for (let i = 1; i < length; i++) {
        ret = Math.max(ret, prices[i] - minVal)
        minVal = Math.min(minVal, prices[i])
    }
    return ret
};


附 309 最佳买卖股票时机含冷冻期    中等
var maxProfit = function(prices) {
    if(prices.length === 0){
        return 0;
    }
    let length = prices.length, 
        f0 = -prices[0], // 持有股票
        f1 = 0,  // 不持有，并处于冷冻
        f2 = 0;  // 不持有，不处于冷冻
    for (let i = 1; i < length; i++) {
        let newf0 = Math.max(f0, f2 - prices[i]),
            newf1 = f0 + prices[i],
            newf2 = Math.max(f1, f2);
        [f0, f1, f2] = [newf0, newf1, newf2]
    }
    return Math.max(f1, f2)
};
```



### 三、链表

#### 剑指 Offer 06. 从尾到头打印链表

```js
var reversePrint = function(head) {
    let ans = []
    while (head !== null) {
        ans.unshift(head.val)
        head = head.next
    }
    return ans
};
```

#### 剑指 Offer 18.删除链表的节点

```js
var deleteNode = function(head, val) {
    if (head.val === val) {
        return head.next
    }
    let prev = head
    let cur = head.next
    while(cur) {
        if (cur.val === val) {
            prev.next = cur.next
            return head
        }
        prev = cur
        cur = cur.next
    }
};
```

#### 剑指 Offer 22. 链表中倒数第k个

```js
var getKthFromEnd = function(head, k) {
    let slow = head, fast = head
    while (fast && k > 0) {
        [fast, k] = [fast.next, k-1]
    }
    while(fast) {
        [fast, slow] = [fast.next, slow.next]
    }
    return slow
    
};

```

#### 剑指 Offer 24. 反转链表

```js
var reverseList = function(head) {
    let prev = null;
    let curr = head;
    while (curr) {
        const next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
};


```

#### 剑指 Offer 25.合并两个排序的链表

```js
var mergeTwoLists = function(l1, l2) {
    // 如果l2当前节点<l1当前节点，就把l2当前节点插入到l1
    // 当l1遍历结束，就把l2当前节点接上，
    let cur = dum = new ListNode(0)
    while (l1 && l2) {
        if (l1.val < l2.val) {
            [cur.next, l1] = [l1, l1.next]
        } else {
            [cur.next, l2] = [l2, l2.next]
        }
        cur = cur.next
    }
    cur.next = l1 ? l1 : l2
    return dum.next
};

```

#### ! 剑指 Offer 35. 复杂链表的复制

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

```js
// 方法一：回溯 + 哈希表
var copyRandomList = function(head, cachedNode = new Map()) {
    if (head === null) {
        return null;
    }
    if (!cachedNode.has(head)) {
        cachedNode.set(head, {val: head.val}) 
      	// 传参的时候别忘了传cachNode
        Object.assign(cachedNode.get(head), {next: copyRandomList(head.next, cachedNode), random: copyRandomList(head.random, cachedNode)})
    }
    return cachedNode.get(head);
}
// 方法2  迭代 + 阶段拆分
var copyRandomList = function(head) {
    if (head === null) {
        return null;
    }
    for (let node = head; node !== null; node = node.next.next) {
        const nodeNew = new Node(node.val, node.next, null);
        node.next = nodeNew;
    }
    for (let node = head; node !== null; node = node.next.next) {
        const nodeNew = node.next;
        nodeNew.random = (node.random !== null) ? node.random.next : null;
    }
    const headNew = head.next;
    for (let node = head; node !== null; node = node.next) {
        const nodeNew = node.next;
        node.next = node.next.next;
        nodeNew.next = (nodeNew.next !== null) ? nodeNew.next.next : null;
    }
    return headNew;
};
```

#### 剑指 Offer 36. 二叉搜索树与双向链表

将二叉搜索树转为排序的双向链表

```js
var treeToDoublyList = function(root) {
    if (!root) return 
    let pre = null, head = null
    dfs(root)
    head.left = pre
    pre.right = head
    return head
    
    function dfs(cur) {
        if (!cur) return 
        dfs(cur.left)
        if (pre) {
            pre.right = cur
            cur.left = pre
            // [pre.right, cur.left] = cur, pre
        } else {
            head = cur
        }
        pre = cur
        dfs(cur.right)
    }
};

```

### 四、数组

#### 剑指 Offer 03.数组中重复的数字 3712 

```js
var findRepeatNumber = function(nums) {
    let mySet = new Set()
    for (let x of nums) {
        if (mySet.has(x)) {
            return x
        }
        mySet.add(x)
    }
};

const swap = function(nums, a, b) {
        [nums[a], nums[b]] = [nums[b], nums[a]]
    }
var findRepeatNumber = function(nums) {
    for (let i = 0; i < nums.length; ) {
      	// 需要注意直到下标相等时才右移
        if(nums[i] === i) {
            i++
            continue;
        }
        if (nums[nums[i]] === nums[i]) {
            return nums[i]
        }
        swap(nums, nums[i], i)
    }
    return -1
};
```



#### 剑指 Offer 04. 二维数组中的查找 2960 
```js
var findNumberIn2DArray = function(matrix, target) {
    if (matrix === null || !matrix.length || !matrix[0].length) {
        return false
    }
    const row = matrix.length, col = matrix[0].length;
    let i = 0, j = col - 1;
    // 注意这里不要用双层循环，只能用while单层的
    while(i < row && j >= 0) {
        // console.log(i, j)
        const num = matrix[i][j]
        if (target == num) {
            return true;
        } else if (target > num) {
            i++;
        } else {
            j--;
        }
    }
    return false
};

```



#### 剑指 Offer 07. 重建二叉树1867 



#### ! 剑指 Offer 11. 旋转数组的最小数字 2250 
```js
var minArray = function(numbers) {
    let low = 0,
        high = numbers.length - 1
    while (low <= high) {
        const mid = low + Math.floor((high - low) / 2)
        if (numbers[mid] > numbers[high]) {
          	// 此时mid肯定不是最小值，所以可以加一
            low = mid + 1
        } else if (numbers[mid] < numbers[high]) {
          	// 此时mid可能是最小值
            high = mid
        } else {
          	// 此时high有可能和最小值相等
            high--
        }
    }
    return numbers[low]
};

```

#### 剑指 Offer 12. 矩阵中的路径

```js
var exist = function(board, word) {
    if (board.length === 0 || board[0].length === 0) return false;
    const m = board.length, n = board[0].length
    
    const inArea = function (x, y) {
        return x >=0 && x < m && y >=0 && y < n
    }
    const dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    
    const dfs = (i, j, k) => {
        if (!inArea(i, j) || board[i][j] !== word[k]) {
            return false
        }
        if (k === word.length - 1) {
            return true
        }
        // 回溯，这个节点遍历之后就让它置空，递归结束之后再重新设置回来
        board[i][j] = ''
        let res = false
        dirs.forEach(dir => {
            if (dfs(i + dir[0], j + dir[1], k + 1)) {
                res = true
            }
        })
      	// **
        board[i][j] = word[k]
        return res
    }
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (dfs(i, j, 0)) {
                return true
            }
        }
    }
    return false
};
```



#### 剑指 Offer 17. 打印从1到最大的n..1389

```js
var printNumbers = function(n) {
    let max = Math.pow(10,n) 
    let arr=[]
    for(var i=1;i<max;i++){
        arr.push(i)
    }
    return arr
};

```



#### 剑指 Offer 21.调整数组顺序使奇数位于偶数前… 2234 
```js
// 方法1，额外数组
var exchange = function(nums) {
    const res = []
    for(const num of nums) {
        if(num % 2 === 0) {
        res.push(num)
        } else {
        res.unshift(num)
        }
    }
    return res
};
// 方法2， 快慢指针
const swap = function(nums, a, b) {
   	if(a !== b) [nums[a], nums[b]] = [nums[b], nums[a]]
}
var exchange = function(nums) {
    let slow = 0, fast = 0
    while(fast < nums.length) {
        if (nums[fast] & 1) {
            if(slow < fast) swap(nums, slow, fast)
            slow++
        }
        fast++;
    }
    return nums
};
```



#### 剑指 Offer 29. 顺时针打印矩阵1891 
```js
var spiralOrder = function(matrix) {
    if (!matrix.length || !matrix[0].length) {
        return [];
    }

    const rows = matrix.length, columns = matrix[0].length;
    const order = [];
    let left = 0, right = columns - 1, top = 0, bottom = rows - 1;
    while (left <= right && top <= bottom) {
        for (let column = left; column <= right; column++) {
            order.push(matrix[top][column]);
        }
        for (let row = top + 1; row <= bottom; row++) {
            order.push(matrix[row][right]);
        }
        // 这里一定要加判定
        if (left < right && top < bottom) {
            for (let column = right - 1; column > left; column--) {
                order.push(matrix[bottom][column]);
            }
            for (let row = bottom; row > top; row--) {
                order.push(matrix[row][left]);
            }
        }
        [left, right, top, bottom] = [left + 1, right - 1, top + 1, bottom - 1];
    }
    return order;
};

```


#### 剑指 Offer 31. 栈的压入、弹出序列 1336 
```js
// 模拟压入和弹出的操作，最后能全部弹出即可
var validateStackSequences = function(pushed, popped) {
    let stack = [], i = 0
    for (num of pushed) {
        stack.push(num)
        while (stack.length > 0 && stack[stack.length - 1] === popped[i]){
            stack.pop()
            i++
        }
    }
    return !stack.length
};

// 7.30二刷
var validateStackSequences = function(pushed, popped) {
    let stack = [], length = pushed.length, pos = 0
    for (let i = 0; i < length; i++) {
        stack.push(pushed[i])
        while(stack.length && stack[stack.length - 1] === popped[pos]) {
            console.log(stack)
            stack.pop()
            pos++
        }
    }
    return pos === length
};
```


#### 剑指 Offer 39.数组中出现次数超一半的数.1340 

```js
var majorityElement = function(nums) {
    const map = {}
    const len = nums.length
    for (let i = 0; i < len; i++) {
        if (map[nums[i]]) {
            map[nums[i]]++
        } else {
            map[nums[i]] = 1
        }
        if (map[nums[i]] > len/2) {
            return nums[i]
        }
    }
};
// 方法二，排序取中点
var majorityElement = function(nums) {
    nums.sort((a,b)=>a-b);
    return nums[Math.floor(nums.length/2)];
};
```



#### 剑指 Offer 42.连续子数组的最大和 1981
#### 剑指 Offer 47. 礼物的最大价值1451
#### 剑指 Offer 51.数组中的逆序对1236  困难
#### ！剑指 Offer 53-1. 在排序数组中查找数字出现的次数｜.2244

返回出现的次数

```js
// 二分查找
const binarySearch = (nums, target, lower) => {
    let left = 0, right = nums.length - 1, ans = nums.length;
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] > target || (lower && nums[mid] >= target)) {
            right = mid - 1;
          	// ans保留最后>或>=target的元素，
            ans = mid;
        } else {
            left = mid + 1;
        }
    }
    return ans;
}

var search = function(nums, target) {
    let ans = 0;
    const leftIdx = binarySearch(nums, target, true);
    const rightIdx = binarySearch(nums, target, false) - 1;
  	// 万一没有这个元素的情况筛选
    if (leftIdx <= rightIdx && rightIdx < nums.length && nums[leftIdx] === target && nums[rightIdx] === target) {
        ans = rightIdx - leftIdx + 1;
    } 
    return ans;
};
```



#### ！ 剑指 Offer 53 - 11.0~n-1中缺失… 2165

```js
// 下标法
var missingNumber = function(nums) {
    const n = nums.length;
    for (let i = 0; i < n; i++) {
        if (nums[i] !== i) {
            return i;
        }
    }
    return n;
};
012356
// 二分法
var missingNumber = function(nums) {
    let i = 0, j = nums.length - 1
    // 闭区间
    while(i <= j) {
        let mid = (i + j) >> 1
        // i j分别指向右子数组首位元素和左子数组的末尾元素
        if (nums[mid] === mid) i = mid + 1
        else j = mid - 1
    }
  	// 返回的是下标
    return i
};
```



#### 剑指 Offer 56 -l.数组中数字出现的次数…1358 （位运算，异或）

在一个数组 nums 中除两个数字只出现一次之外，其他数字都出现了两次。请找出那个只出现一次的数字。

```js
var singleNumbers = function (nums) {
    let m = 0, n=0;
    // 异或运算，得到两结果数字的抑或运算
    nums.forEach(num => {
        m ^= num;
    });
    // 求其中标志位
    let d = m & -m;
    // 分组计算，计算其中一半的
    nums.forEach(num => {
        num & d ? n ^= num : '';
    });
    // 
    let n1 = n ^ m;
    return [n, n1]
};
```


#### 剑指 Offer 56 -. 数组中数字出现的次数2… 944 

在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

```js
var singleNumber = function (nums) {
  let x = 0;
  // 新建一个计数数组
  let numCount = [];
  // 将计数数组清零
  for (let i = 0; i < nums.length; i++) {
    numCount[nums[i]] = 0;
  }
  // 循环遍历记录数字出现的个数
  for (let i = 0; i < nums.length; i++) {
    numCount[nums[i]]++;
  }
  // 循环遍历数组，计数为1的即为要找的数组
  for (let i = 0; i < nums.length; i++) {
    if (numCount[nums[i]] == 1) return nums[i];
  }
};

// 状态机
var singleNumber = function (nums) {
  let ones = twos = 0;
  for (let num of nums) {
      ones = ones ^ num & ~twos
      twos = twos ^ num & ~ones
  }
  return ones
};
```



#### 剑指 Offer 57.和为s的两个数字1217 
```js
var twoSum = function(nums, target) {
    let p = 0, q = nums.length - 1
    while (true) {
        if (nums[p] + nums[q] === target) {
            return [nums[p], nums[q]]
        } else if (nums[p] + nums[q] > target) {
            q--
        } else {
            p++
        }
    }
};
```

#### 剑指 Offer 61. 扑克牌中的顺子1423

joker可以为任意牌

`没有重复，并且最小值和最大值差小于5， 所以要得到最小值和最大值，可以用set判断重复，取最大最小，或者遍历之后，最后一个值减去第一个下标为joker数量，即第一个不为joker的值`

```js
var isStraight = function(nums) {
    const set = new Set()
    let max = 0, min = 14
    for (num of nums) {
        if (num === 0) continue // 跳过大王
        max = Math.max(max, num)
        min = Math.min(min, num)
        if (set.has(num)) return false
        set.add(num)
    }
    return max - min < 5
};

var isStraight = function(nums) {
    let joker = 0
    nums = nums.sort((a, b) => a - b)
    for (let i = 0; i < 5; i++) {
        if (nums[i] === 0) joker++
        else if (nums[i] === nums[i + 1]) return false
    }
    return nums[4] - nums[joker] < 5
};
```



#### 剑指 Offer 63. 股票的最大利润1380



### 五、bfs

#### 剑指 Offer 13. 机器人的运动范围      2263 （bfs）

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

```js
const digitsum = function(n) {
  let ans = 0
  while(n) {
    ans += n % 10
    n = Math.floor(n / 10)
  }
  return ans
}
 // 这里是bfs的方法，设置一个标志位记录是否
// 
var movingCount = function(m, n, k) {
  const q = new Array()
  const visited = new Array(m).fill().map(_ => new Array(n).fill(0))
  q.push([0, 0])
  let counter = 0
  while (q.length) {
    const [x, y] = q.shift()
		if (x >= m || y >= n)  continue
    // 遍历过
    if (visited[x][y]) continue
    // 设置遍历过的标识
    visited[x][y] = 1
    if (digitsum(x) + digitsum(y) <= k) {
      // 符合条件的计数
      counter++
      // 将右、下两格加入队列
      q.push([x + 1, y], [x, y + 1])
    }   
  }
  return counter
};


```



