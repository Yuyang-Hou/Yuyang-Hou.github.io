---
title: hot100三刷记录
data: 2022-07-04 19:11:58
---

> 首刷：争取三天拿下，至少面试能想出来思路
>
> 记一些易错点
>
> - forEach中, return会失效，要在循环中return的话记得用普通循环
> - forEach中使用 return 语句的作用只能跳出当前循环（实现了continue的效果），并不能跳出整个循环，没达到期待的效果，满足条件后还继续执行了循环。因为 forEach 本身无法跳出循环，必须遍历所有的数据才能结束，它传入的是一个回调函数，因此形成了一个作用域，它内部所定义的变量不会像for循环一样污染全局变量。

### 1 两数之和   简单

```js
// 暴力
let m = nums.length
    for (let i = 0; i < m; i++) {
        for (let j = i + 1; j < m; j++) {
            if(nums[i] + nums[j] === target) return [i, j]
        }
    }
// hash
var twoSum = function(nums, target) {
    let map = new Map(), length = nums.length
    for (let i = 0; i < length; i++) {
        if(map.has(target - nums[i])) {
            return [map.get(target - nums[i]), i]
        }
        map.set(nums[i], i)
    }
};
```

### ！ 2 两数相加   中等

```js
var addTwoNumbers = function(l1, l2) {
    let result, temp;
    let add = 0;
    const getVal = (l) => l ? l.val : 0;
    const getNext = (l) => l ? l.next : l;
    while (l1 || l2) {
        let now = getVal(l1) + getVal(l2) + add;
        add = now >= 10 ? 1 : 0;
        now = now >= 10 ? now - 10 : now;

        if (!result) {
            result = new ListNode(now);
            temp = result;
        } else {
            temp.next = new ListNode(now);
            temp = temp.next;
        }
        l1 = getNext(l1);
        l2 = getNext(l2);
    }
    if (add) temp.next = new ListNode(add);
    return result;
};
// 8.5二刷，看了题解，尝试使用原来的链表，想不出来
var addTwoNumbers = function(l1, l2) {
    let head, cur;
    let add = 0;
    while(l1 || l2) {
        // 要用括号括好，否则优先级会错
        let sum = (l1 && l1.val || 0) + (l2 && l2.val || 0) + add
        add = sum > 9 ? 1 : 0
        sum = sum % 10
        console.log(sum)
        if (!head) {
            result = new ListNode(sum)
            cur = result
        } else {
            cur.next = new ListNode(sum)
            cur = cur.next
        }
        l1 = l1 && l1.next || null
        l2 = l2 && l2.next || null
    }
    // 处理最后的进位
    if (add) cur.next = new ListNode(add)
    return head
};
```



### 3 无重复字符的最长子串   中等

```js
var lengthOfLongestSubstring = function(s) {
    // 哈希集合，记录每个字符是否出现过
    const occ = new Set();
    const n = s.length;
    // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
    let rk = -1, ans = 0;
    for (let i = 0; i < n; ++i) {
        if (i != 0) {
            // 左指针向右移动一格，移除一个字符
            occ.delete(s.charAt(i - 1));
        }
        while (rk + 1 < n && !occ.has(s.charAt(rk + 1))) {
            // 不断地移动右指针
            occ.add(s.charAt(rk + 1));
            ++rk;
        }
        // 第 i 到 rk 个字符是一个极长的无重复字符子串
        ans = Math.max(ans, rk - i + 1);
    }
    return ans;
    
// 8.5二刷， i做右指针，左指针用一个变量保存
var lengthOfLongestSubstring = function(s) {
    let map = new Map(), length = s.length,
        maxLength = 0,
        left = 0
    for (let i = 0; i < length; i++) {
        if (map.has(s[i])) {
            left = Math.max(left, map.get(s[i]) + 1)
        }
        map.set(s[i], i)
        maxLength = Math.max(i - left + 1, maxLength)
    }
    return maxLength
};
```



### 4寻找两个正序数组的中位数   困难

```
var findMedianSortedArrays = function(nums1, nums2) {
    // 防止最小数组为空
    if (nums1.length == 0) {
        var middle = parseInt(nums2.length / 2)
        return nums2.length % 2 ? nums2[middle] : (nums2[middle] + nums2[middle - 1])/2
    }
    // 防止两个长度都为1的时候进bug
    if (nums2.length == 1 && nums1.length == 1) {
        return (nums2[0] + nums1[0]) / 2
    }
    // 确保nums1是小数组，继续优化时间复杂度，变为O(lb min(m,n))
    if (nums1.length > nums2.length) {
        return findMedianSortedArrays(nums2, nums1)
    }
    var len = nums2.length + nums1.length
    var cut1 = 0
    var cut2 = 0
    var cutL = 0
    var cutR = nums1.length
    while (cut1 <= nums1.length) {
        // 二分法，在视频中这里在加了 cutL是错误的二分
        cut1 = parseInt((cutL + cutR)/2)
        // 因为中位数左右两边相等，所以第二组的数量是这样的
        cut2 = parseInt(len/2) - cut1
        // 要考虑边界条件
        var L1 = (cut1 == 0) ? -Infinity : nums1[cut1 - 1]
        var L2 = (cut2 == 0) ? -Infinity : nums2[cut2 - 1]
        var R1 = (cut1 == nums1.length) ? Infinity : nums1[cut1]
        var R2 = (cut2 == nums2.length) ? Infinity : nums2[cut2]
        // console.log(R1,L1,R2,L2)
        // 判断分组条件是否正确，否，则继续二分
        if (L1 <= R2 && L2 <= R1) {
            return (len % 2) ? Math.min(R1,R2) : (Math.max(L1,L2)+Math.min(R1,R2))/2 
        } else if (L1 > R2) {
            cutR = cut1 - 1
        } else if (R1 < L2) {
            cutL = cut1 + 1
        }
    }
};
```



### 5 最长回文子串   中等

```js
var longestPalindrome = function(s) {
    const expendAroudCenter = function(s, l, r) {
        while(l >= 0 && r < s.length && s[l] === s[r]) {
            l--
            r++
        }
        return [l + 1, r - 1]  // 这里忘了减了
    }
    let start = 0; end = 0
    for (let i = 0; i < s.length; i++) {
        [l1, r1] = expendAroudCenter(s, i, i);
        [l2, r2] = expendAroudCenter(s, i, i+1)
        if (r1 - l1 > end - start) {
            [start, end] = [l1, r1]
        }
        if (r2 - l2 > end - start) {
            [start, end] = [l2, r2]
        }
    }
    return s.slice(start, end + 1)
};
// 
var longestPalindrome = function(s) {
    const expendStr = function(s, i, j) {
        while(i >= 0 && j < s.length && s[i] === s[j]) {
            i--
            j++
        }
      	// 这里不要用i++
        return [++i, --j]
    }
    let l = r = 0 
    for (let i = 0; i < s.length; i++) {
        let [l1, r1] = expendStr(s, i, i)
        let [l2, r2] = expendStr(s, i, i+1)
        if (r1 - l1 > r - l) {
            [l, r] = [l1, r1]
        }
        if (r2 - l2 > r - l) {
            [l, r] = [l2, r2]
        }
    }
    return s.slice(l, r + 1)
};
```



### 10 正则表达式匹配   困难

```
var isMatch = function (s, p) {
    let sLen = s.length;
    let pLen = p.length;
    let dp = new Array(s + 1);
    for (let i = 0; i <= sLen; i++) {
        dp[i] = new Array(pLen + 1).fill(false);
    }
    dp[0][0] = true;
    for (let i = 2; i <= pLen; i += 2) {
        if (p[i - 1] === '*') dp[0][i] = dp[0][i - 2];
    }
    for (let i = 1; i <= sLen; i++) {
        for (let j = 1; j <= pLen; j++) {
            if (s[i - 1] === p[j - 1] || p[j - 1] === '.') dp[i][j] = dp[i - 1][j - 1];
            else if (p[j - 1] === '*') {
                if (p[j - 2] !== s[i - 1] && p[j - 2] !== '.') dp[i][j] = dp[i][j - 2];
                else dp[i][j] = dp[i - 1][j] || dp[i][j - 2] || dp[i][j - 1];
            }
        }
    }
    return dp[sLen][pLen];
};

```



### 11盛最多水的容器   中等

`i，j较小的那个先向内移动 如果高的指针先移动，那肯定不如当前的面积大，因为是取小值，只能希望`

```js
var maxArea = function(height) {
    let max = 0;
    for (let i = 0, j = height.length - 1; i < j;) {//双指针i，j循环height数组
      	//i，j较小的那个先向内移动 如果高的指针先移动，那肯定不如当前的面积大，因为是取小值，只能希望
        const minHeight = height[i] < height[j] ? height[i++] : height[j--];
        const area = (j - i + 1) * minHeight;//计算面积
        max = Math.max(max, area);//更新最大面积
    }
    return max;
};
```



### 15三数之和   中等

```js
var threeSum = function(nums) {
    if (!nums || nums.length < 3) return []
    nums = nums.sort((a, b) => a - b);
    let sum = 0; // 三个数的和
        result = [];
    for (let i = 0; i < nums.length; i++) {
        if (i && nums[i] === nums[i - 1]) continue; // 若当前这一项和上一项相等则跳过
        let left = i + 1;
        let right = nums.length - 1;
        while (left < right) {
            sum = nums[i] + nums[left] + nums[right];
            if (sum > 0) {
                right--;
            } else if (sum < 0) {
                left++;
            } else {
                result.push([nums[i], nums[left++], nums[right--]]);
                while (nums[left] === nums[left - 1]) { // 一直找到不相同的那个坐标
                    left++;
                }
                while (nums[right] === nums[right + 1]) {
                    right--;
                }
            }
        }
    }
    return result;
};

```



### 17 电话号码的字母组合   中等

```js
const letterCombinations = (num) => {
  if(!num) return [];
  const arr = [['a','b','c'], ['d','e','f'],['g','h','i'],['j','k','l'],['m','n','o'],['p','q','r','s'],['t','u','v'],['w','x','y','z']];
  let result = [];
  let numArr = num.toString().split('');
  for(let i = 0, len = numArr.length; i < len; i++) {
    result.push(arr[numArr[i] - 2]);
  }
  return result.reduce((prev, cur) => {
        let temp = [], m = prev.length, n = cur.length
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
              temp.push(prev[i] + cur[j])
          }
      }
    return temp
  })
}
```



### 19 删除链表的倒数第N个结点   中等

```js
var removeNthFromEnd = function (head, n) {
    let fast = slow = head, slowPre = null
    while(--n && fast) fast = fast.next
    while(fast.next) {
        slowPre = slow
        slow = slow.next
        fast = fast.next
    }
    if (slowPre) {
        slowPre.next = slow.next
    } else {
        head = head.next
    }
    return head
};
```



### 20有效的括号   简单

```js
var isValid = function(s) {
    const n = s.length;
    if (n % 2 === 1) {
        return false;
    }
    const pairs = new Map([
        [')', '('],
        [']', '['],
        ['}', '{']
    ]);
    const stk = [];
  	// 不能用forEach
    for (let ch of s){
        if (pairs.has(ch)) {
        		// 这里易忘.length的判断
            if (!stk.length || stk[stk.length - 1] !== pairs.get(ch)) {
                return false;
            }
            stk.pop();
        } 
        else {
            stk.push(ch);
        }
    };
    return !stk.length;
};
```



### 21 合并两个有序链表   简单

```
var mergeTwoLists = function(l1, l2) {
    if (l1 === null) {
        return l2;
    } else if (l2 === null) {
        return l1;
    } else if (l1.val < l2.val) {
        l1.next = mergeTwoLists(l1.next, l2);
        return l1;
    } else {
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
    }
};

var mergeTwoLists = function(l1, l2) {
    const prehead = new ListNode(-1);

    let prev = prehead;
    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            prev.next = l1;
            l1 = l1.next;
        } else {
            prev.next = l2;
            l2 = l2.next;
        }
        prev = prev.next;
    }

    // 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
    prev.next = l1 === null ? l2 : l1;

    return prehead.next;
};
```



### 22 括号生成   中等

```js
var generateParenthesis = function(n) {
    const ans = []
    const backtrack = (s, l, r) => {
        if (s.length === 2 * n) {
            ans.push(s.join(''))
            return 
        }
        if (l < n) {
            s.push('(')
          	// 这里的l，r是局部变量，不要用l++了，这里加完，下面怎么半
            backtrack(s, l + 1, r)
            s.pop()
        }
        if (l > r) {
            s.push(')')
            backtrack(s, l, r + 1)
            s.pop()
        }
    }
    backtrack([], 0, 0)
    return ans
};
```



### ！23 合并K个升序链表   困难

```js
var mergeKLists = function(lists) {
    if(lists.length <= 0){
        return null;
    }
    function mergeTwoLists(l1, l2){
        if(l1 == null){
            return l2;
        }
        if(l2  == null){
            return l1;
        }
        let head = new ListNode(null);
        let now = head;
        while(l1 != null && l2 != null){
            now.next = new ListNode(null);
            now = now.next
            if(l1.val <= l2.val){
                now.val = l1.val;
                l1 = l1.next;
            }else{
                now.val = l2.val;
                l2 = l2.next;
            }
        }
        now.next = l1 != null ? l1 : l2;
        return head.next;
    }
    let jump = 1;
    while(jump < lists.length){
        for(let i = 0; i < lists.length; i = i + 2 * jump){
            lists[i] = mergeTwoLists(lists[i], lists[i+jump]);
        }
        jump *= 2;
    }
    return lists[0];
};

```



### 31下一个排列   中等

```js
var nextPermutation = function(nums) {
    length  = nums.length
    let i = length - 2
    while(i >= 0 && nums[i] >= nums[i + 1]){
        i--
    }
  	// 注意
    if (i >= 0) {
        let j = length - 1
        while(j >= 0 && nums[i] >= nums[j]) {
            j--
        }
        [nums[i], nums[j]] = [nums[j], nums[i]]
    }
    let l = i + 1,
        r = length - 1;
    while(r > l) {
        [nums[l], nums[r]] = [nums[r], nums[l]]
        l++
        r--
    }
    return nums
};
```



### 32最长有效括号   困难

```
var longestValidParentheses2 = function(s) {
    const arrLen = s.length
    if(arrLen<2) return 0
    //创建一个等长数组用于标记当前有效长度，初始全部值为0
    let countArr = new Array(arrLen).fill(0)
    for(let k=0;k<=arrLen-1;k++) {
        if(s[k]===')'){
            if(s[k-1]==='('){
                //当出现一个有效括号时，则其有效长度为k-2位前的有效括号+当前有效括号（值2），因为括号是两位的，所以上一个有效括号应该是k-2
                countArr[k]=k-2>0?countArr[k-2]+2:2
                //当出现两个右括号时，即两个闭口，则判断其有效长度前1位是否为左括号，既开口
            }else if(s[k-1]===')' && s[k-countArr[k-1]-1]==='('){
                //若是，则代表这两个括号有效（+2）
                countArr[k]=countArr[k-1]+2
                //加上此时有效括号前的有效括号数，如()(())    ===>   ()+(())
                if((k-countArr[k-1]-2)>0) {
                    countArr[k]=countArr[k]+countArr[k-countArr[k-1]-2]
                } 
            }
        }
    }
    countArr.sort((a,b)=>a-b)
    return countArr[arrLen-1]
};
```



### 33 搜索旋转排序数组   中等

```js
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var search = function (nums, target) {
    let left = 0, right = nums.length - 1
    while(left <= right) {
        let mid = (left + right) >> 1
        if (nums[mid] === target) {
            return mid
        }
        // 这里为什么要分情况讨论是为了得到target在左右两边的哪个位置
        if (nums[mid] < nums[right]) {
            // 这里切记要有等于的情况
            if(target <= nums[right] && target > nums[mid]) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        } else {
            // =的情况
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
    }
    return -1
};
```



### ！34 在排序数组中查找元素的第一个和最后一个位置   中等

```js
const binarySearch = (nums, target, lower) => {
    // ans记录
    let left = 0, right = nums.length - 1, ans = nums.length;
    while (left <= right) {
        const mid = (left + right) >> 1
        if (nums[mid] > target || (lower && nums[mid] >= target)) {
            right = mid - 1;
            // 主要是记住这里，要保留ans
            ans = mid;
        } else {
            left = mid + 1;
        }
    }
    return ans;
}

var searchRange = function(nums, target) {
    let ans = [-1, -1];
    const leftIdx = binarySearch(nums, target, true);
    const rightIdx = binarySearch(nums, target, false) - 1;
    // 
    if (leftIdx <= rightIdx && nums[leftIdx] === target && nums[rightIdx] === target) {
        ans = [leftIdx, rightIdx];
    } 
    return ans;
};
```



### ！39组合总和   中等

```js
var combinationSum = function(candidates, target) {
    const ans = [];
    const dfs = (target, combine, idx) => {
        if (idx === candidates.length) {
            return;
        }
        if (target === 0) {
            ans.push(combine);
            return;
        }
        // 直接跳过
        dfs(target, combine, idx + 1);
        // 选择当前数
        if (target - candidates[idx] >= 0) {
            // [创建了一个副本]
            dfs(target - candidates[idx], [...combine, candidates[idx]], idx);
        }
    }
    dfs(target, [], 0);
    return ans;
};
```



### 42 接雨水   困难

``` js
var trap = function(height) {
    let ans = 0
    let left = 0, right = height.length  -1 
    let leftMax = 0, rightMax = 0
    while(left < right) {
        leftMax = Math.max(leftMax, height[left])
        rightMax = Math.max(rightMax, height[right])
        if (height[left] < height[right]) {
            ans += leftMax - height[left++]
        } else {
            ans += rightMax - height[right--]
        }
    }
    return ans
};
```



### 46 全排列   中等

```js
var permute = function(nums) {
    const res = [], path = []
    const used = new Array(nums.length).fill(false)

    const dfs = () => {
        if (path.length === nums.length) {
            res.push(path.slice())
            return 
        }
        for (let i = 0; i < nums.length; i++) {
            if (used[i]) continue
          
            path.push(nums[i])
            used[i] = true
          
            dfs()
          
            path.pop()
            used[i] = false
        }
    }
    dfs()
    return res
};
```



### 48 旋转图像   中等

```js
var rotate = function(matrix) {
    const n = matrix.length;
    for (let i = 0; i < Math.floor(n / 2); ++i) {
        for (let j = 0; j < Math.floor((n + 1) / 2); ++j) {
            const temp = matrix[i][j];
            matrix[i][j] = matrix[n - j - 1][i];
            matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
            matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
            matrix[j][n - i - 1] = temp;
        }
    }
};
var rotate = function(matrix) {
    const n = matrix.length;
    // 水平翻转
    for (let i = 0; i < Math.floor(n / 2); i++) {
        for (let j = 0; j < n; j++) {
            [matrix[i][j], matrix[n - i - 1][j]] = [matrix[n - i - 1][j], matrix[i][j]];
        }
    }
    // 主对角线翻转
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < i; j++) {
            [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
        }
    }
};

```



### ! 49字母异位词分组   中等

`定义一个获取该字符串排序后的样子的函数，遍历，把相同的就放到一起,map转数组一定要用扩展运算符`

```js
var groupAnagrams = function (strs) {
    const map = new Map();

    const getKey = (a) => {
        // 将字母排序后用作 map key，如 'eve' => 'eev'， 'vee' => 'eev'
        return a.split('').sort().join('')
    }

    for (let item of strs) {
        // 获取 item 的 key，保证相同字母组合的 key 相同。
        const key = getKey(item);
        if (map.get(key)) {
            map.get(key).push(item);
        } else {
            map.set(key, [item]);
        }
    }
  	// ！这里必须要要用。。。
    return [...map.values()];
};

```



### 53 最大子数组和   简单

```js
var maxSubArray = function(nums) {
  if(nums.length === 0)return 0
	let reMax = nums[0]
	// dep[n] 代表的是第0 到 第n项之间nums所有的子项组合中的最大值
	let dep = []
	dep[0] = nums[0]
	for(let i = 1 ; i<nums.length; i++){
		// 根据排列规律 我们可以发现dep[i]其实就是
		// 当dep[i-1]>0时为dep[i-1]+nums[i]
		// 当dep[i-1]<0时就是nums[i]本身
		// 之所以是这样因为第n项的所有子项的排列规律
		// 为第n-1项的每一项加上nums[i] 再加上[nums[i]]
		// 递推规律
		// 第一个子组合是以第一个数字结尾的连续序列，也就是 [-2]，最大值-2
		// 第二个子组合是以第二个数字结尾的连续序列，也就是 [-2,1], [1]，最大值1
		// 第三个子组合是以第三个数字结尾的连续序列，也就是 [-2,1,3], [1,3], [3]，最大值4
		if(dep[i-1]>0){
			dep[i] = dep[i-1] + nums[i]
		}else{
			dep[i] = nums[i]
		}
		reMax = Math.max(reMax,dep[i])
	}
	return reMax
	
};

```



### 55 跳跃游戏   中等

```js
var canJump = function(nums) {
  let len = nums.length;
  let pos = undefined;
  for (let i = len - 2; i >= 0; i--) {
    if (nums[i] === 0 && pos === undefined)
      pos = i;
    if (pos !== undefined && i + nums[i] > pos)
      pos = undefined
  }
  return pos === undefined
};

```



### 56合并区间   中等

```js
// 按区间起始点升序排序；
// 初始结果数组res，第一个元素赋值为区间的第一个元素res = FintervalstOy;
// 从1开始遍历剩下的区间，与res最后一个元素进行重合判断；
// 如果重合，更新res最后一个元素；
// 如果不重合，追加到res。
var merge = function(intervals) {
    if (intervals.length === 0) return [];
    intervals.sort((a, b) => a[0] - b[0]);
    let res = [intervals[0]];
    for (let i = 1; i < intervals.length; i++) {
        let temp = check(res[res.length - 1], intervals[i]);
        if (temp) {
            res[res.length - 1] = temp;
        } else {
            res.push(intervals[i]);
        }
    }
    return res;
    // 判断重合（不重合：最小的比最大的还大，最大的比最小的还小）
    function check(arr1, arr2) {
        if (arr1[0] > arr2[1] || arr1[1] < arr2[0]) {
            return null;
        }
        return [Math.min(arr1[0], arr2[0]), Math.max(arr1[1], arr2[1])];
    }
};

```



### 62不同路径   中等

```js
var uniquePaths = function(m, n) {
    const dp = new Array(m).fill(new Array(n).fill(0))
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if(i === 0 || j === 0) {
                dp[i][j] = 1
            } else {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
            }
        }
    }
    return dp[m - 1][n - 1]
};
```



### 64 最小路径和   中等

```js
var minPathSum = function(dp) {
    let m = dp.length, n = dp[0].length;
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if(!i && !j) continue
            if (i === 0) {
                dp[i][j] = dp[i][j - 1] + dp[i][j]
                continue
            } 
            if (j === 0) {
                dp[i][j] = dp[i - 1][j] + dp[i][j]
                continue
            }
            dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + dp[i][j]
        }
    }
    return dp[m - 1][n - 1]
};
```



### 70爬楼梯   简单

```js
var climbStairs = function(n) {
    let p = 0, q = 0, r = 1;
    for (let i = 1; i <= n; ++i) {
        p = q;
        q = r;
        r = p + q;
    }
    return r;
};
```



### 72编辑距离   困难

### 75颜色分类   中等

`切记不能用几个if穿在一起写，因为执行完这个if置换之后，就有可能满足下面的if了`

```js
var sortColors = function (nums) {
    if (!nums.length) return []
    let [cur, left, right] = [0, 0, nums.length - 1];
    while (cur <= right) {
        if (nums[cur] === 0) {
            swap(left, cur);
            left++;
            cur++
        } else if (nums[cur] === 2) {
            swap(cur, right);
            right--;
        } else {
            cur++
        }
    }
    return nums
};

const swap = (a, b) => {
  let temp = nums[a];
  nums[a] = nums[b];
  nums[b] = temp;
};
```



### 76 最小覆盖子串   困难
### 78子集   中等 

```js
var subsets = function(nums) {
    let res = [], path = []
    const dfs = function(index) {
        if (index === nums.length) {
            res.push(path.slice())
            return 
        }
        // 选择
        path.push(nums[index])
        dfs(index + 1)
        path.pop() 
        // 不选择
        dfs(index + 1) 
    }
    dfs(0)
    return res
```



### 79单词搜索   中等

```js
var exist = function(board, word) {
    const h = board.length, w = board[0].length;
    const visited = new Array(h).fill().map(_ => new Array(w).fill(false))
    const dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    const inArea = function (x, y) {
        return x >=0 && x < h && y >=0 && y < w
    }
    const check = function(i, j, k) {
        if (board[i][j] !== word[k]) {
            return false
        } else if (k === word.length - 1) {
            return true
        }
        visited[i][j] = true
        let result = false
        for (let [dx, dy] of dirs) {
            let [newi, newj] = [i + dx, j + dy] 
            if(inArea(newi, newj) && !visited[newi][newj]) {
                if (check(newi, newj, k + 1)) {
                    result = true;
                    break;
                }
            }   
        }
        visited[i][j] = false
        return result
    }
    for (let i = 0; i < h; i++) {
        for (let j = 0; j < w; j++) {
            const flag = check(i, j, 0);
            if (flag) {
                return true;
            }
        }
    }
    return false;
};
```



### 84 柱状图中最大的矩形   困难
### 85 最大矩形   困难
### 94 二叉树的中序遍历   简单

```js
var inorderTraversal = function(root) {
    const dfs = (root) => {
        if (root === null) {
            return 
        }
        if (root.left) {
          dfs(root.left)
        }
        ans.push(root.val)
        if (root.right) {
          dfs(root.right)
        }
    }
    const ans = []
    dfs(root)
    return ans
};
```



### 96 不同的二叉搜索树   中等

`笛卡尔积`

<img src="/Users/onlycat/Library/Application Support/typora-user-images/image-20220723100317112.png" alt="image-20220723100317112" style="zoom:50%; margin-left:0" />

```js
var numTrees = function(n) {
    const G = new Array(n + 1).fill(0);
    G[0] = 1;
    G[1] = 1;

    for (let i = 2; i <= n; ++i) {
        for (let j = 1; j <= i; ++j) {
            G[i] += G[j - 1] * G[i - j];
        }
    }
    return G[n];
};

```



### 98 验证二叉搜索树   中等

```js
const helper = function (root, lower, upper) {
  
    if (root === null) {
        return true
    }
    if (root.val <= lower || root.val >= upper){
        return false
    }
    return helper(root.left, lower, root.val) && helper(root.right, root.val, upper)
}
var isValidBST = function(root) {
    return helper(root, -Infinity, Infinity)
};
```



### 101 对称二叉树   简单

```js
const func = (p, q) =>  {
    if (p === null && q === null) return true
    if (p === null || q === null) return false
    return p.val === q.val && func(p.left, q.right) && func(p.right, q.left)
}
var isSymmetric = function(root) {
    return func(root, root)
};
```



### 102 二叉树的层序遍历   中等

```js
var levelOrder = function(root) {
    const ret = [];
    if (!root) {
        return ret;
    }

    const q = [];
    q.push(root);
    while (q.length !== 0) {
        const currentLevelSize = q.length;
        ret.push([]);
        for (let i = 1; i <= currentLevelSize; ++i) {
            const node = q.shift();
            ret[ret.length - 1].push(node.val);
            if (node.left) q.push(node.left);
            if (node.right) q.push(node.right);
        }
    }

     return ret;
};
```



### 104二叉树的最大深度   简单

```js
var maxDepth = function(root) {
    let count = 0
    if (!root) {
        return count;
    }
    const q = [];
    q.push(root);
    while (q.length !== 0) {
        const currentLevelSize = q.length;
        for (let i = 1; i <= currentLevelSize; ++i) {
            let node = q.shift();
            if (node.left) q.push(node.left);
            if (node.right) q.push(node.right);
        }
        count++
    }
     return count;
};
```



### 105从前序与中序遍历序列构造二叉树   中等

```js
var buildTree = function(preorder, inorder) {
    if (preorder.length) {
        let root = new TreeNode(preorder.shift())
        let rootIndex = inorder.indexOf(root.val)
        root.left = buildTree(preorder.slice(0, rootIndex), inorder.slice(0, rootIndex))
        root.right = buildTree(preorder.slice(rootIndex, ), inorder.slice(rootIndex + 1, ))
        return root 
    } else {
        return null
    }
};
```



### 114二叉树展开为链表   中等

```js
var flatten = function(root) {
    let curr = root
    while(curr !== null) {
        if (curr.left !== null) {
            // 左节点作为右节点
            let next = curr.left
            let predecessor = next
            while (predecessor.right !== null) {
                predecessor = predecessor.right
            }
            // 右节点放到左子树中最右的节点右边
            predecessor.right = curr.right
            curr.right = next
            // 左节点制空
            curr.left = null
        }
        curr = curr.right
    }
    return root
};
```



### 121 买卖股票的最佳时机     简单

```js
var maxProfit = function(prices) {
    let min = prices[0]
    let max = 0
    for (let i = 1; i < prices.length; i++) {
        min = Math.min(min, prices[i]) 
        max = Math.max(max, prices[i] - min)
    }
    return max
};
```



### 124二叉树中的最大路径和     困难



### 128 最长连续序列     中等

```js
var longestConsecutive = function(nums: number[]): number {
    // 设置set
    let set: Set<number> = new Set(nums)
    let ret = 0
    // 元素遍历
    for (const num of set) {
        if (!set.has(num - 1)) {
            let curNum = num + 1
            let curCnt = 1
            while(set.has(curNum)) {
                curNum++
                curCnt++
            }
            ret = Math.max(ret, curCnt)
        }
    }
    return ret
};
```



### 136 只出现一次的数字     简单

```typescript
function singleNumber(nums: number[]): number {
  	// 一个数异或0不变
  	// 两个等数异或等于0
  	// 两个不等数异或等于1
    return nums.reduce((p, c) => p ^ c)
};
```



### ！139单词拆分     中等

```js
function wordBreak(s: string, wordDict: string[]): boolean {
    const n : number = s.length
    const wordDictSet : Set<string> = new Set (wordDict)
    const dp : Array<boolean> = new Array(n + 1).fill(false)
    
    dp[0] = true
    for (let i = 1; i <= n; i++) {
        for (let j = 0; j < i; j++) {
            if (dp[j] && wordDictSet.has(s.substr(j, i - j))) {
                dp[i] = true
                break
            }
        }
    }
    return dp[n]
};
```



### 141 环形链表     简单

```js
var hasCycle = function(head) {
    if (head === null || head.next === null) return false
    let slow = head, fast = head.next
    while (slow !== fast) {
        if (fast === null || fast.next === null) {
            return false
        }
        slow = slow.next
        fast = fast.next.next
    }
    return true
};
```



### 142 环形链表川     中等

```js
var detectCycle = function(head) {
    const visited = new Set()
    while(head !== null) {
        if (visited.has(head)) {
            return head
        }
        visited.add(head)
        head = head.next
    }
    return null
};
// 快慢指针
var detectCycle = function(head) {
    if (head === null) {
        return null;
    }
    let slow = head, fast = head;
    while (fast !== null) {
        slow = slow.next;
        if (fast.next !== null) {
            fast = fast.next.next;
        } else {
            return null;
        }
        if (fast === slow) {
            let ptr = head;
            while (ptr !== slow) {
                ptr = ptr.next;
                slow = slow.next;
            }
            return ptr;
        }
    }
    return null;
};
```



### 146 LRU 缓存     中等

```

```



### ！148 排序链表     中等

`递归，中分`

```js
var mergeTwoLists = function(l1, l2) {
    if (l1 === null) {
        return l2;
    } else if (l2 === null) {
        return l1;
    } else if (l1.val < l2.val) {
        l1.next = mergeTwoLists(l1.next, l2);
        return l1;
    } else {
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
    }
};


var sortList = function(head) {
  	// 判出条件
    if(head == null || head.next == null) return head
  	// 取中点
    let fast = head.next
    let slow = head
    while (fast && fast.next) {
        fast = fast.next.next
        slow = slow.next
    }
    let mid = slow.next
    slow.next = null
  	// 递归
    let left = sortList(head)
    let right = sortList(mid)
    // 合并
    return mergeTwoLists(left, right)
};
```



### 152 乘积最大子数组     中等

```js
const maxProduct = function (nums) {
    let ans = nums[0], 
        maxValue = nums[0], 
        minValue = nums[0]
    for( let i = 1; i < nums.length; i++) {
        let maxProduct = nums[i] * maxValue
        let minProduct = nums[i] * minValue
        // 更新最大值，最小值
        maxValue = Math.max(maxProduct, minProduct, nums[i])
        minValue = Math.min(maxProduct, minProduct, nums[i])
        ans = Math.max(maxValue, ans)
    }
    return ans
};
```



### 155 最小栈     中等

```js
var MinStack = function() {
    this.x_stack = [];
    this.min_stack = [Infinity];
};

MinStack.prototype.push = function(x) {
    this.x_stack.push(x);
    this.min_stack.push(Math.min(this.min_stack[this.min_stack.length - 1], x));
};

MinStack.prototype.pop = function() {
    this.x_stack.pop();
    this.min_stack.pop();
};

MinStack.prototype.top = function() {
    return this.x_stack[this.x_stack.length - 1];
};

MinStack.prototype.getMin = function() {
    return this.min_stack[this.min_stack.length - 1];
};

```



### 160 相交链表     简单

```js
var getIntersectionNode = function(headA, headB) {
    if (headA === null || headB === null) {
        return null;
    }
    let pA = headA, pB = headB;
    while (pA !== pB) {
        pA = !pA ? headB : pA.next;
        pB = !pB ? headA : pB.next;
    }
    return pA;
};

```



### 169 多数元素     简单

```js
// hash,建议用排序
var majorityElement = function(nums) {
    const n = nums.length/2
    let numsMap = new Map()
    for (let num of nums) { 
        if (!numsMap.has(num)) {
            numsMap.set(num, 1)
        }
        else {
            numsMap.set(num, numsMap.get(num) + 1)
        }
        if (numsMap.get(num) > n) {
            return num
        }
    }
};
```



### 198 打家劫舍     中等

```js
var rob = function(nums) {
  	// 特殊情况
    if (!nums.length) {
        return 0
    }
    const size = nums.length
    if (size === 1) {
        return nums[0]
    }
  	// 核心
    let p = nums[0], q = Math.max(nums[0], nums[1])
    for (let i = 2; i < size; i++) {
      	[p, q] = [q, Math.max(p + nums[i], q)]
    }
    return q
};
```



### 200岛屿数量     中等

```js
var numIslands = function(grid) {
    const m = grid.length
    if (!m) {
        return 0
    }
    const n = grid[0].length
    const d = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    // 保存岛屿数量
    let count = 0
    // 判断所选元素是否在二维平面内
    const inArea = function (x, y) {
        return x >=0 && x < m && y >=0 && y < n
    }
    
    // 核心
    const dfs = function (grid, i, j) {
        grid[i][j] = '0'
        for (let k = 0; k < 4; k++) {
            const newX = i + d[k][0]
            const newY = j + d[k][1]
            inArea(newX, newY) && grid[newX][newY] == 1 && dfs(grid, newX, newY)
        }
        return
    }

    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                count++
                dfs(grid, i, j)
            }
        }
    }
    return count
};
```



### 206反转链表     简单

```js
var reverseList = function(head) {
    let prev = null
    let curr = head
    while(curr) {
        let next = curr.next
        curr.next = prev
        prev = curr
        curr = next
    }
    return prev
};
```



### ! 207 课程表     中等

```js
/**
 * @param {number} numCourses
 * @param {number[][]} prerequisites
 * @return {boolean}
 */

var canFinish = function(numCourses, prerequisites) {                      
    let indegree = new Array(numCourses).fill(0)       // 顶点的入度
    const dirctMap = new Array(numCourses).fill().map(_ => new Array())                                  //邻接表(在dfs遍历的时候用得到)
    let result = 0                                     //正规的拓扑排序应该得到一个数组，内容是顺序表，为了节省内存用数量替代
    let ifFind = new Array(numCourses).fill(0)         //用来记录找到了哪些节点
    for(let i = 0,l = prerequisites.length;i<l;i++){    //初始化邻接表
        let item = prerequisites[i];
        indegree[item[0]]++                  //记录该节点的入度
        dirctMap[item[1]].push(item[0])      //在邻接表里记录该节点
    }
  
    for(let i = 0;i<numCourses;i++){         //循环遍历所有节点
            dfs(dirctMap,i,0)
    }
    /*
    *dirctMap : 要遍历的邻接表
    *index : 当前节点
    *status : 是否第一层
    */
    function dfs(dirctMap,index,status){
        if(status&&ifFind[index]===1){  //如果不是第一层并且该节点已经被找到过了，那么说明遇到了环路，则提前结束，返回结果
            return false
        }
        if(ifFind[index] === 0&& indegree[index]===0){//判断入度为0并且违背找到过的节点可以进入
            result++            //找到一个节点，总数量+1
            ifFind[index] = 1;  //标记该节点被找到
            dirctMap[index].forEach(i=>{    //广度遍历该节点的兄弟节点
                indegree[i]--               //由于原来的节点被‘移除了’所以兄弟节点的入度要减一
                dfs(dirctMap,i,1)
            })
        }
    }
    return result === numCourses            //如果最后找到的节点数量小于总数的话就说明有环路 
                                            //(因为有环路的话在剥离过程中入度始终不会为0，一直不满足条件，所以不会被找到)
}
```



### 208实现 Trie (前缀树）   中等  

```

```



### 215数组中的第K个最大元素  中等   
### 221 最大正方形     中等

```js
var maximalSquare = function (matrix) {
    const m = matrix.length
    if (m === 0) {
        return 0
    }
    const n = matrix[0].length
    let maxWidth = 0
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (matrix[i][j] === '1') {
                matrix[i][j] = (i == 0 || j == 0) ? 1 : Math.min(matrix[i-1][j-1],matrix[i][j-1],matrix[i-1][j]) + 1
                maxWidth = Math.max(maxWidth, matrix[i][j])
            } else {
                matrix[i][j] = 0
            }
        }
    }
    return maxWidth * maxWidth
};
```



### 226翻转二叉树     简单

```js
var invertTree = function(root) {
    if (root === null) {
        return null;
    }
    const left = invertTree(root.left);
    const right = invertTree(root.right);
    root.left = right;
    root.right = left;
    return root;
};
```



### ! 34 回文链表     简单

```js
var isPalindrome = function(head) {
    // 这句可有可无，毕竟题目中说了链表长度至少为1
    if(!head) return true;
    let slow = head, fast = head.next;
    while(fast && fast.next) {
        slow = slow.next;
        fast = fast.next.next;
    }
  	// next
    let back = reverseList(slow.next);
    while(back) {
        if(head.val !== back.val) {
            return false;
        }
        head = head.next;
        back = back.next;
    }
    return true;
};
function reverseList(head){
    let prev = null
    let curr = head
    while(curr) {
        let next = curr.next
        curr.next = prev
        prev = curr
        curr = next
    }
    return prev
}

```



### 236 二叉树的最近公共祖先  中等   

```js
var lowestCommonAncestor = function(root, p, q) {
    let ans = null
    const dfs = (root, p, q) => {
        if (root === null) return false
        const lson = dfs(root.left, p, q)
        const rson = dfs(root.right, p, q)
        if ((lson && rson) || (root.val === p.val || root.val === q.val) && (lson || rson) ) {
            ans = root
        }
        return lson || rson || root.val === p.val || root.val === q.val
    }
    dfs (root, p, q)
    return ans
};
```



### 238除自身以外数组的乘积   中等  

```js
var productExceptSelf = function(nums: number[]): number[] {
    const length = nums.length;
    const answer = new Array<number>(length);

    // answer[i] 表示索引 i 左侧所有元素的乘积
    // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
    answer[0] = 1;
    for (let i = 1; i < length; i++) {
        answer[i] = nums[i - 1] * answer[i - 1];
    }

    // R 为右侧所有元素的乘积
    // 刚开始右边没有元素，所以 R = 1
    let R = 1;
    for (let i = length - 1; i >= 0; i--) {
        // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
        answer[i] = answer[i] * R;
        // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
        R *= nums[i];
    }
    return answer;
};
```



### ! 239 滑动窗口最大值     困难

<img src="/Users/onlycat/Library/Application Support/typora-user-images/image-20220809195950924.png" alt="image-20220809195950924" style="zoom:50%;" />

```js
var maxSlidingWindow = function(nums, k) {
    if (!nums.length) return []

    let ans = [],
        windows = []

    for (let i = 0; i < nums.length; i++) {
        if (i >= k && windows[0] <= i - k) windows.shift()
        while(windows.length && nums[windows[windows.length-1]] <= nums[i]) windows.pop()
        windows.push(i)
        if (i >= k - 1) ans.push(nums[windows[0]])
    }
    
    return ans
};

```



### 240搜索二维矩阵     中等

```js
var searchMatrix = function(matrix, target) {
  if (matrix.length === 0 || matrix[0].length === 0) return false;
  
  let ans = false,
      rowLimit = matrix.length,
      colLimit = matrix[0].length;
  
  let row = rowLimit - 1, col = 0;
  
  while (true) {
    if (row < 0 || col >= colLimit) break;
    
    let curr = matrix[row][col];
    
    if (curr === target) {
      ans = true;
      break;
    }
    
    if (target > curr) col++;
    if (target < curr) row--;
  }
  
  return ans;
};
```



### 253会议室     中等 VIP

```

```



### 279 完全平方数    中等

```js
var numSquares = function(n) {
    const dp = new Array(n + 1).fill(0)
    for (let i = 1; i <= n; i++) {
        let minn = Number.MAX_VALUE
        for (let j = 1; j * j <= i; j++) {
            minn = Math.min(minn, dp[i - j*j])
        }
        dp[i] = minn + 1
    }
    return dp[n]
};
```



### 283移动零    简单

```js
var moveZeroes = function(nums) {
    const swap = function(nums, a, b) {
        [nums[a], nums[b]] = [nums[b], nums[a]]
    }
    let Index0 = -1
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] !== 0) {
            if (Index0 !== -1 ) {
                swap(nums, Index0, i)
                Index0++
            }
        } else {
            if (Index0 === -1 ) {
                Index0 = i
            }
        }
    }
    return nums
};
```



### 287 寻找重复数    中等

```js
var findDuplicate = function(nums) {
    let slow  = 0, fast = 0;
    do {
        slow = nums[slow]
        fast = nums[nums[fast]] 
    } while (slow != fast)
    slow = 0
    while (slow != fast) {
        slow = nums[slow]
        fast = nums[fast]
    }
    return slow
};
```



### 297二叉树的序列化与反序列化    困难
### 300 最长递增子序列    中等

```js
var lengthOfLIS = function(nums) {
    if(nums.length === 0){
        return 0;
    }
    let length = nums.length, max = 1
    let dp = new Array(length).fill(1)
    for (let i = 1; i < length; i++) {
        for (let j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = Math.max(dp[i], dp[j] + 1)
                max = Math.max(dp[i], max)
            }
        }
    }
    return max
};
```



### 301删除无效的括号    困难

```

```



### 309 最佳买卖股票时机含冷冻期    中等

```js
var maxProfit = function(prices) {
    let f0 = -prices[0], // 持有股票
        f1 = 0,  // 不持有，并处于冷冻
        f2 = 0;  // 不持有，不处于冷冻
    for (let i = 1; i < prices.length; i++) {
        let newf0 = Math.max(f0, f2 - prices[i]),
            newf1 = f0 + prices[i],
            newf2 = Math.max(f1, f2);
        [f0, f1, f2] = [newf0, newf1, newf2]
    }
    return Math.max(f1, f2)
};
```



### 312戳气球    困难

```

```



### 322 零钱兑换    中等

```
var coinChange = function(coins, amount) {
    const dp = new Array(amount + 1).fill(Infinity)
    dp[0] = 0
    for (let coin of coins) {
        for (let x = coin; x < amount + 1; x++) {
            dp[x] = Math.min(dp[x], dp[x - coin] + 1)
        }
    }
    return dp[amount] === Infinity ? -1 : dp[amount]
};
```



### 337 打家劫舍川    中等

```js
var rob = function(root) {
    const dfs = (node) => {
        if (node === null) {
            return [0, 0];
        }
        const l = dfs(node.left);
        const r = dfs(node.right);
        const selected = node.val + l[1] + r[1];
        const notSelected = Math.max(l[0], l[1]) + Math.max(r[0], r[1]);
        return [selected, notSelected];
    }
    
    const rootStatus = dfs(root);
    return Math.max(rootStatus[0], rootStatus[1]);
}; 
```



### 338 比特位计数    简单

```js
var countBits = function(n) {
    const bits = new Array(n + 1).fill(0);
    for (let i = 1; i <= n; i++) {
        bits[i] = bits[i & (i - 1)] + 1; // i & i-1 比i少一个1，并且比i小
    }
    return bits;
};
```



### 347 前K个高频元素    中等

```
let topKFrequent = function(nums, k) {
    // 利用Map来记录key-整数和value-频率的关系
    let map = new Map()
    nums.map((num) => {
        if (map.has(num)) map.set(num, map.get(num) + 1)
        else map.set(num, 1)
    })
    
    // 如果元素数量小于等于k -> 直接返回字典key-整数
    if(map.size <= k) {
        return [...map.keys()]
    }

    // 返回桶排序结果
    // return bucketSort(map, k)
    return [...map.keys()].sort((a, b) => 
        map.get(b) - map.get(a)
    ).slice(0, k)
};
```



### ！394字符串解码    中等

```js
var decodeString = function(s) {
    let numStack = [];              // 倍数 num 的等待栈
    let strStack = [];              // 待拼接 str 的等待栈
    let num = 0, result = '';
    for(let i = 0; i < s.length; i++) {
        const item = s[i];
        if(!isNaN(item)) {              // 判断是数字时
            num = num * 10 + parseInt(item);
        } else if(item === '[') {
            strStack.push(result);
            result = '';
            numStack.push(num);
            num = 0;
        } else if(item === ']') {
            const repeatTimes = numStack.pop();         // 从栈中获取次数
            result = strStack.pop() + result.repeat(repeatTimes);
        } else {
            result += item;
        }
    }
    return result;
};
```



### 399 除法求值    中等

```

```



### 406根据身高重建队列    中等

```js
var reconstructQueue = function(people) {
    if (!people || !people.length) return [];
    people.sort((a, b) => a[0] === b[0] ? a[1] - b[1] : b[0] - a[0]);
    
    const res = [];
    people.forEach(item => {
        res.splice(item[1], 0, item); // 插入到k对应的位置
    })
    return res;
};
```



### 416分割等和子集    中等

```js
var canPartition = function(nums) {
    const n = nums.length;
    if (n < 2) {
        return false;
    }
    let sum = 0, maxNum = 0;
    for (const num of nums) {
        sum += num;
        maxNum = maxNum > num ? maxNum : num;
    }
    if (sum & 1) {
        return false;
    }
    const target = Math.floor(sum / 2);
    if (maxNum > target) {
        return false;
    }
    const dp = new Array(target + 1).fill(false);
    dp[0] = true;
    for (const num of nums) {
        for (let j = target; j >= num; --j) {
            dp[j] |= dp[j - num];
        }
    }
    return dp[target];
};
```



### ！437 路径总和川    中等

```js
var pathSum = function(root, targetSum) {
    if (root == null) {
        return 0;
    }
    
    let ret = rootSum(root, targetSum);
    ret += pathSum(root.left, targetSum);
    ret += pathSum(root.right, targetSum);
    return ret;
};

const rootSum = (root, targetSum) => {
    let ret = 0;

    if (root == null) {
        return 0;
    }
    const val = root.val;
    if (val === targetSum) {
        ret++;
    } 

    ret += rootSum(root.left, targetSum - val);
    ret += rootSum(root.right, targetSum - val);
    return ret;
}
```



### 438 找到字符串中所有字母异位词    中等

```js
var findAnagrams = function (s, p) {
    const pLen = p.length
    const res = [] // 返回值
    const map = new Map() // 存储 p 的字符
    for (let item of p) {
        map.set(item, map.get(item) ? map.get(item) + 1 : 1)
    }
    // 存储窗口里的字符情况
    const window = new Map()
    let valid = 0 // 有效字符个数

    for (let i = 0; i < s.length; i++) {
        const right = s[i]
        // 向右扩展
        window.set(right, window.get(right) ? window.get(right) + 1 : 1)
        // 扩展的节点值是否满足有效字符
        if (window.get(right) === map.get(right)) {
            valid++
        }
        if (i >= pLen) {
            // 移动窗口 -- 超出之后，收缩回来， 这是 pLen 长度的固定窗口
            const left = s[i - pLen]
            // 原本是匹配的，现在移出去了，肯定就不匹配了
            if (window.get(left) === map.get(left)) {
                valid--
            }
            window.set(left, window.get(left) - 1)
        }
        // 如果有效字符数量和存储 p 的map 的数量一致，则当前窗口的首字符保存起来
        if (valid === map.size) {
            res.push(i - pLen+1)
        }
    }
    return res
};
```



### 448找到所有数组中消失的数字    简单

```js
var findDisappearedNumbers = function(nums) {
    const n = nums.length;
    for (const num of nums) {
        const x = (num - 1) % n;
        nums[x] += n;
    }
    const ret = [];
    for (const [i, num] of nums.entries()) {
        if (num <= n) {
            ret.push(i + 1);
        }
    }
    return ret;
};
```



### 461 汉明距离    简单

```js
var hammingDistance = function(x, y) {
    let s = x ^ y, ret = 0;
    while (s != 0) {
        ret += s & 1;
        s >>= 1;
    }
    return ret;
};
```



### ！494目标和    中等

```js
var findTargetSumWays = function(nums, target) {
    let sum = 0;
    for (const num of nums) {
        sum += num;
    }
    const diff = sum - target;
    if (diff < 0 || diff % 2 !== 0) {
        return 0;
    }
    const neg = Math.floor(diff / 2);
    const dp = new Array(neg + 1).fill(0);
    dp[0] = 1;
    for (const num of nums) {
        for (let j = neg; j >= num; j--) {
            dp[j] += dp[j - num];
        }
    }
    return dp[neg];
};
```



### 538 把二叉搜索树转换为累加树    中等

```js
var convertBST = function(root) {
    const dfs = (node) => {
        if (node === null) {
            return;
        }
        dfs(node.right);
        total += node.val
        node.val = total
        dfs(node.left);
    }
    let total = 0
    dfs(root)
    return root  
};
```



### 543二叉树的直径    简单

```js
var diameterOfBinaryTree = function(root) {
    if(root == null || (root.left == null && root.right == null)) return 0
    let res = 0
    function dfs(root) {
        if(root == null) return 0
        let left = dfs(root.left)
        let right = dfs(root.right)
        res = Math.max(res, left + right + 1)
        return Math.max(left, right) + 1
    }
    dfs(root)
    return res - 1
};
```



### 560 和为K的子数组    中等

```js
// 非最优
var subarraySum = function(nums, k) {
    let count = 0
    for (let i = 0; i < nums.length; i++) {
        let sum = 0
        for (let j = i; j >= 0; j--) {
            sum += nums[j]
            if (sum === k) {
                count++
            }
        }
    }
    return count
};
// 优化
var subarraySum = function(nums, k) {
    const mp = new Map();
    mp.set(0, 1);
    let count = 0, pre = 0;
    for (const x of nums) {
        pre += x;
        if (mp.has(pre - k)) {
            count += mp.get(pre - k);
        }
        if (mp.has(pre)) {
            mp.set(pre, mp.get(pre) + 1);
        } else {
            mp.set(pre, 1);
        }
    }
    return count;
};
```



### ！581 最短无序连续子数组    中等

```js
var findUnsortedSubarray = function(nums) {
    if (isSorted(nums)) {
        return 0;
    }
    const numsSorted = [...nums].sort((a, b) => a - b);
    let left = 0;
    while (nums[left] === numsSorted[left]) {
        left++;
    }
    let right = nums.length - 1;
    while (nums[right] == numsSorted[right]) {
        right--;
    }
    return right - left + 1;
};

const isSorted = (nums) => {
    for (let i = 1; i < nums.length; i++) {
        if (nums[i] < nums[i - 1]) {
            return false;
        }
    }
    return true;
}

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/shortest-unsorted-continuous-subarray/solution/zui-duan-wu-xu-lian-xu-zi-shu-zu-by-leet-yhlf/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
// 不是很理解这个思路
var findUnsortedSubarray = function(nums) {
    const n = nums.length;
    let maxn = -Number.MAX_VALUE, right = -1;
    let minn = Number.MAX_VALUE, left = -1;
    for (let i = 0; i < n; i++) {
        if (maxn > nums[i]) {
            right = i;
        } else {
            maxn = nums[i];
        }
        if (minn < nums[n - i - 1]) {
            left = n - i - 1;
        } else {
            minn = nums[n - i - 1];
        }
    }
    return right === -1 ? 0 : right - left + 1;
};
```



### 617 合并二叉树    简单

```js
var mergeTrees = function(t1, t2) {
    if (t1 == null) return t2
    if (t2 == null) return t1
    let merged = new TreeNode(t1.val + t2.val)
    merged.left = mergeTrees(t1.left, t2.left)
    merged.right = mergeTrees(t1.right, t2.right)
    return merged
};
```



### 621 任务调度器    中等 不懂

```
var leastInterval = function(tasks, n) {
    const freq = _.countBy(tasks);
    console.log(freq)
    // 最多的执行次数
    const maxExec = Math.max(...Object.values(freq));
    // 具有最多执行次数的任务数量
    let maxCount = 0;
    Object.values(freq).forEach(v => {
        if (v === maxExec) {
            maxCount++;
        }
    })

    return Math.max((maxExec - 1) * (n + 1) + maxCount, tasks.length);
};
```



### 647回文子串    中等

```js
var countSubstrings = function(s) {
    const n = s.length;
    let ans = 0;
    const helper = (a, b) => {
        while(a >= 0 && b < s.length && s[a] === s[b]) {
                ++ans;
                --a;
                ++b;
            }
        }
    for (let i = 0; i < n; i++) {
        if (i !== 0) {
            helper(i - 1, i)
        }
        helper(i, i)
    }
    return ans;
};
```



### 739 每日温度    中等

```js
var dailyTemperatures = function(T) {
    let res = new Array(T.length).fill(0),
        stack = [];
    for(let i = 0; i < T.length; i++){
        while(stack.length > 0 && stack[stack.length - 1][0] < T[i]){
            res[stack[stack.length - 1][1]] = i - stack[stack.length - 1][1];
            stack.pop();
        }
        stack.push([T[i], i]);
    }
    return res;
};
```

