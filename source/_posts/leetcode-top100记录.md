---
Title: leetcode top100记录
---

争取三天拿下，然后狂背，至少面试能想出来思路

### 1 两数之和   简单

```
var twoSum = function(nums, target) {
    var obj ={}
    for(var i =0;i<nums.length-1;i++){
        obj.a=i
        for(var j = i+1;j<nums.length;j++){
            obj.b=j
            if(nums[i]+nums[j]==target){
                return Object.values(obj)
            }
        }
    } 
};
```

### 2 两数相加   中等

```
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
```



### 3 无重复字符的最长子串   中等

```
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

```
var longestPalindrome = function(s) {
    const expendAroudCenter = function(s, l, r) {
        while(l >= 0 && r < s.length && s[l] === s[r]) {
            l--
            r++
        }
        return [l + 1, r - 1]
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

```
var maxArea = function(height) {
    let max = 0;
    for (let i = 0, j = height.length - 1; i < j;) {//双指针i，j循环height数组
      	//i，j较小的那个先向内移动 如果高的指针先移动，那肯定不如当前的面积大
        const minHeight = height[i] < height[j] ? height[i++] : height[j--];
        const area = (j - i + 1) * minHeight;//计算面积
        max = Math.max(max, area);//更新最大面积
    }
    return max;
};
```



### 15三数之和   中等

```
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

```
const letterCombinations = (num) => {
  if(!num) return [];
  const arr = [['a','b','c'], ['d','e','f'],['g','h','i'],['j','k','l'],['m','n','o'],['p','q','r','s'],['t','u','v'],['w','x','y','z']];
  let result = [];
  let numArr = num.toString().split('');
  for(let i = 0, len = numArr.length; i < len; i++) {
    result.push(arr[numArr[i]-2]);
  }

  // 1、使用while方法递归, 执行用时：64ms，内存占用：33.6M
  while(result.length > 1) {
    let arr1 = result[0], arr2 = result[1], temp = [];
    for(let i = 0, len1 = arr1.length; i < len1; i++) {
      for(let j = 0, len2 = arr2.length; j < len2; j++) {
        temp.push(arr1[i] + arr2[j]);
      }
    }
    result.splice(0, 2, temp);  // 生成的数组替换前两个数组
  }

  return result[0];  
}
```



### 19 删除链表的倒数第N个结点   中等

```
var removeNthFromEnd = function (head, n) {
  // 判断n是否小于或等于0，其实现实代码中，这里我会报错
  if (!head || !n) return head
  // 快慢指针
  let slow = head
  let quick = head
  let temp = null
  n = n-1
  while(quick && n) {
    quick = quick.next
    n--
  }
  // 判断n是否大于链表的长度，其实现实代码中，这里我会报错
  if (!quick) {
    return head
  }
  while(quick.next) {
    temp = slow
    quick = quick.next
    slow = slow.next
  }
  // 如果slow是第一个，就没有temptemp，这时就返回slow.next
  if (temp) {
    temp.next = slow.next
  } else {
    head = slow.next
  }
  return head
};

```



### 20有效的括号   简单

```
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
    for (let ch of s){
        if (pairs.has(ch)) {
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



### 23 合并K个升序链表   困难

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
    // 二分法
    let start = 0;
    let end = nums.length - 1;

    while (start <= end) {
        // >> 1 相当于除以2向下取整
        let mid = (start + end) >> 1;

        if (nums[mid] === target) {
            return mid;
        }

        // 如果中间数小于最右边数，则右半段是有序的
        // 如果中间数大于最右边数，则左半段是有序的
        if (nums[mid] < nums[end]) {
            // 判断target是否在(mid, end]之间
            if (nums[mid] < target && target <= nums[end]) {
                // 如果在，则中间数右移即start增大
                start = mid + 1;
            } else {
                // 如果不在，则中间数左移即end减小
                end = mid - 1;
            }
        } else {
            // [start, mid)
            if (nums[start] <= target && target < nums[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
    }

    return -1;
};

```



### 34 在排序数组中查找元素的第一个和最后一个位置   中等

```
const binarySearch = (nums, target, lower) => {
    let left = 0, right = nums.length - 1, ans = nums.length;
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] > target || (lower && nums[mid] >= target)) {
            right = mid - 1;
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
    if (leftIdx <= rightIdx && rightIdx < nums.length && nums[leftIdx] === target && nums[rightIdx] === target) {
        ans = [leftIdx, rightIdx];
    } 
    return ans;
};
```



### 39组合总和   中等

```
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
            dfs(target - candidates[idx], [...combine, candidates[idx]], idx);
        }
    }

    dfs(target, [], 0);
    return ans;
};
```



### 42 接雨水   困难

```
var trap = function(height) {
    let ans = 0;
    let left = 0, right = height.length - 1;
    let leftMax = 0, rightMax = 0;
    while (left < right) {
        leftMax = Math.max(leftMax, height[left]);
        rightMax = Math.max(rightMax, height[right]);
        if (height[left] < height[right]) {
            ans += leftMax - height[left];
            ++left;
        } else {
            ans += rightMax - height[right];
            --right;
        }
    }
    return ans;
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



### 49字母异位词分组   中等

```
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
    let arr = new Array(m);
    for(let i = 0; i < m; i++) {
        arr[i] = new Array(n);
    }
    
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (i == 0 || j == 0 ) {
                arr[i][j] = 1
            } else  {
                arr[i][j] = arr[i - 1][j] + arr[i][j - 1]
            }
        }
    }
    return arr[m-1][n-1]
};

```



### 64 最小路径和   中等

```js
var minPathSum = function(dp) {
    let row = dp.length, col = dp[0].length

    for(let i = 1; i < row; i++)//初始化第一列
        dp[i][0] += dp[i - 1][0]

    for(let j = 1; j < col; j++)//初始化第一行
        dp[0][j] += dp[0][j - 1]

    for(let i = 1; i < row; i++)
        for(let j = 1; j < col; j++)
            dp[i][j] += Math.min(dp[i - 1][j], dp[i][j - 1])//取上面和左边最小的
    
    return dp[row - 1][col - 1]
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

```js
var sortColors = function (nums) {
    if (!Array.isArray(nums)) console.log('Invaild Input');
    if (!nums.length) return []
    let [cur, left, right] = [0, 0, nums.length - 1];
    const swap = (a, b) => {
        let temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    };
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
```



### 76 最小覆盖子串   困难
### 78子集   中等
### 79单词搜索   中等
### 84 柱状图中最大的矩形   困难
### 85 最大矩形   困难
### 94 二叉树的中序遍历   简单
### 96 不同的二叉搜索树   中等
### 98 验证二叉搜索树   中等
### 101 对称二叉树   简单
### 102 二叉树的层序遍历   中等
### 104二叉树的最大深度   简单
### 105从前序与中序遍历序列构造二叉树   中等
### 114二叉树展开为链表   中等
### 121 买卖股票的最佳时机     简单
### 124二叉树中的最大路径和     困难
### 128 最长连续序列     中等
### 136 只出现一次的数字     简单
### 139单词拆分     中等
### 141 环形链表     简单