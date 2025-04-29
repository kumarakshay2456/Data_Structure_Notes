
# 1. Sliding Window and Two Pointer Problems

## Sliding Window Technique

The sliding window technique is used to perform operations on a specific window size of an array or string. It's particularly useful for solving problems involving subarrays or substrings.

### Types of Sliding Windows:

1. **Fixed-Size Window**: The window size remains constant throughout.
2. **Variable-Size Window**: The window size changes based on certain conditions.

### Fixed-Size Window Example: Maximum Sum Subarray of Size K

```python
def max_sum_subarray(arr, k):
    """
    Find the maximum sum of a subarray of size k.
    
    Args:
        arr: List of integers
        k: Size of the subarray
    Returns:
        Maximum sum of any subarray of size k
    """
    n = len(arr)
    
    # Edge case
    if n < k:
        return None
    
    # Compute sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window from left to right
    for i in range(k, n):
        # Add the incoming element and remove the outgoing element
        window_sum = window_sum + arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

**Example Usage:**

```python
arr = [1, 4, 2, 10, 2, 3, 1, 0, 20]
k = 4
result = max_sum_subarray(arr, k)  # Output: 24 (subarray [2, 10, 2, 10])
```

**Step-by-step Execution:**

```
Initial window: [1, 4, 2, 10], Sum = 17
Slide window: [4, 2, 10, 2], Sum = 17 + 2 - 1 = 18
Slide window: [2, 10, 2, 3], Sum = 18 + 3 - 4 = 17
Slide window: [10, 2, 3, 1], Sum = 17 + 1 - 2 = 16
Slide window: [2, 3, 1, 0], Sum = 16 + 0 - 10 = 6
Slide window: [3, 1, 0, 20], Sum = 6 + 20 - 2 = 24
Max sum: 24
```

### Variable-Size Window Example: Longest Substring Without Repeating Characters

```python
def length_of_longest_substring(s):
    """
    Find the length of the longest substring without repeating characters.
    
    Args:
        s: Input string
    Returns:
        Length of the longest substring without repeating characters
    """
    char_index = {}  # Store the current index of characters
    max_length = 0
    window_start = 0
    
    for window_end in range(len(s)):
        # If character is already in the current window, adjust the window start
        if s[window_end] in char_index and char_index[s[window_end]] >= window_start:
            window_start = char_index[s[window_end]] + 1
        else:
            # Update max_length if current window is larger
            max_length = max(max_length, window_end - window_start + 1)
        
        # Update character index
        char_index[s[window_end]] = window_end
    
    return max_length
```

**Example Usage:**

```python
s = "abcabcbb"
result = length_of_longest_substring(s)  # Output: 3 (substring "abc")
```

**Step-by-step Execution:**

```
Start: window_start = 0, max_length = 0, char_index = {}
i=0, char='a': window_start = 0, max_length = 1, char_index = {'a': 0}
i=1, char='b': window_start = 0, max_length = 2, char_index = {'a': 0, 'b': 1}
i=2, char='c': window_start = 0, max_length = 3, char_index = {'a': 0, 'b': 1, 'c': 2}
i=3, char='a': window_start = 1, max_length = 3, char_index = {'a': 3, 'b': 1, 'c': 2}
i=4, char='b': window_start = 2, max_length = 3, char_index = {'a': 3, 'b': 4, 'c': 2}
i=5, char='c': window_start = 3, max_length = 3, char_index = {'a': 3, 'b': 4, 'c': 5}
i=6, char='b': window_start = 5, max_length = 3, char_index = {'a': 3, 'b': 6, 'c': 5}
i=7, char='b': window_start = 7, max_length = 3, char_index = {'a': 3, 'b': 7, 'c': 5}
Final max_length: 3
```

# 1.1 Two Pointer Technique

The two pointer technique uses two pointers to solve problems in a single pass, often reducing time complexity from O(n²) to O(n).

### Types of Two Pointer Approaches:

1. **Opposite Direction Pointers**: Start from both ends and move toward the center.
2. **Same Direction Pointers**: Start from one end and move in the same direction at different speeds.

### Opposite Direction Example: Two Sum in Sorted Array

```python
def two_sum_sorted(arr, target):
    """
    Find two numbers in a sorted array that add up to the target.
    
    Args:
        arr: Sorted list of integers
        target: Target sum
    Returns:
        Indices of the two numbers (0-indexed)
    """
    left = 0
    right = len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need a larger sum, move left pointer right
        else:
            right -= 1  # Need a smaller sum, move right pointer left
    
    return []  # No solution found
```

**Example Usage:**

```python
arr = [2, 7, 11, 15]
target = 9
result = two_sum_sorted(arr, target)  # Output: [0, 1]
```

**Step-by-step Execution:**

```
Start: left = 0, right = 3
Iteration 1: arr[0] + arr[3] = 2 + 15 = 17 > 9, move right pointer: left = 0, right = 2
Iteration 2: arr[0] + arr[2] = 2 + 11 = 13 > 9, move right pointer: left = 0, right = 1
Iteration 3: arr[0] + arr[1] = 2 + 7 = 9 == 9, return [0, 1]
```

### Same Direction Example: Remove Duplicates from Sorted Array

```python
def remove_duplicates(nums):
    """
    Remove duplicates from a sorted array in-place.
    
    Args:
        nums: Sorted list of integers
    Returns:
        Length of the new array without duplicates
    """
    if not nums:
        return 0
    
    # Slow pointer points to the last unique element
    slow = 0
    
    # Fast pointer scans through the array
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1  # Length of the de-duplicated array
```

**Example Usage:**

```python
nums = [1, 1, 2, 2, 3, 4, 4, 5]
length = remove_duplicates(nums)  # Output: 5
print(nums[:length])  # Output: [1, 2, 3, 4, 5]
```

**Step-by-step Execution:**

```
Start: slow = 0, fast = 1, nums = [1, 1, 2, 2, 3, 4, 4, 5]
Iteration 1: nums[1] == nums[0], no change: slow = 0, fast = 1
Iteration 2: nums[2] != nums[0], increment slow, copy: slow = 1, nums[1] = 2
Iteration 3: nums[3] == nums[1], no change: slow = 1, fast = 3
Iteration 4: nums[4] != nums[1], increment slow, copy: slow = 2, nums[2] = 3
Iteration 5: nums[5] != nums[2], increment slow, copy: slow = 3, nums[3] = 4
Iteration 6: nums[6] == nums[3], no change: slow = 3, fast = 6
Iteration 7: nums[7] != nums[3], increment slow, copy: slow = 4, nums[4] = 5
Final array: [1, 2, 3, 4, 5, x, x, x] where x is remaining original values
Length: 5
```

## More Advanced Examples

### 1. Minimum Size Subarray Sum (Variable-Size Sliding Window)

Find the minimal length subarray where the sum is greater than or equal to a target value.

```python
def min_subarray_len(target, nums):
    """
    Find the minimum length subarray with sum >= target.
    
    Args:
        target: Target sum
        nums: List of positive integers
    Returns:
        Minimum length of subarray with sum >= target or 0 if none
    """
    n = len(nums)
    min_length = float('inf')
    window_sum = 0
    window_start = 0
    
    for window_end in range(n):
        window_sum += nums[window_end]
        
        # Shrink the window as small as possible while maintaining the sum >= target
        while window_sum >= target:
            min_length = min(min_length, window_end - window_start + 1)
            window_sum -= nums[window_start]
            window_start += 1
    
    return min_length if min_length != float('inf') else 0
```

**Example Usage:**

```python
nums = [2, 3, 1, 2, 4, 3]
target = 7
result = min_subarray_len(target, nums)  # Output: 2 (subarray [4, 3])
```

### 2. 3Sum (Two Pointers with Sorting)

Find all unique triplets in the array that sum up to a target value.

```python
def three_sum(nums):
    """
    Find all unique triplets that sum to zero.
    
    Args:
        nums: List of integers
    Returns:
        List of triplets [a, b, c] such that a + b + c = 0
    """
    result = []
    nums.sort()
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # Two-sum problem for the remaining array
        left = i + 1
        right = n - 1
        target = -nums[i]  # Looking for two elements that sum to -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result
```

**Example Usage:**

```python
nums = [-1, 0, 1, 2, -1, -4]
result = three_sum(nums)  # Output: [[-1, -1, 2], [-1, 0, 1]]
```

### 3. Maximum Consecutive Ones III (Sliding Window with Constraints)

Find the maximum number of consecutive 1's in the array if you can flip at most K 0's.

```python
def max_consecutive_ones(nums, k):
    """
    Find the longest subarray with at most k zeros.
    
    Args:
        nums: Binary array (0s and 1s)
        k: Maximum number of zeros allowed to flip
    Returns:
        Length of the longest subarray with at most k zeros
    """
    max_length = 0
    window_start = 0
    zero_count = 0
    
    for window_end in range(len(nums)):
        # If we encounter a zero, increment the count
        if nums[window_end] == 0:
            zero_count += 1
        
        # If we have too many zeros, shrink the window
        while zero_count > k:
            if nums[window_start] == 0:
                zero_count -= 1
            window_start += 1
        
        # Update the maximum length
        max_length = max(max_length, window_end - window_start + 1)
    
    return max_length
```

**Example Usage:**

```python
nums = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
k = 2
result = max_consecutive_ones(nums, k)  # Output: 6
```

## When to Use Each Technique

### Use Sliding Window When:

- You need to process subarrays or substrings of a specific size
- You're tracking a running sum, average, or other aggregate values
- You need to find the longest/shortest subarray meeting certain criteria
- Common problems: maximum sum subarray, string anagrams, longest substring

### Use Two Pointers When:

- You're working with sorted arrays
- You need to find pairs or triplets with specific properties
- You need to remove or merge elements in-place
- You're testing for palindromes or symmetry
- You need to partition arrays (like in quicksort)
- Common problems: two sum, removing duplicates, container with most water

Both techniques offer ways to optimize solutions from O(n²) or worse down to O(n) time complexity, making them essential tools for efficient array and string manipulation.

# 2. Rain Water Trapping Problem

**Explanation:** This problem asks us to calculate how much rainwater can be trapped between bars of different heights. For each position, the water that can be trapped is determined by the minimum of the maximum heights to its left and right, minus the height at that position.

**Python Solution:**

```python
def trap(height):
    """
    Calculate trapped water between bars
    
    Args:
        height: List of integers representing bar heights
    Returns:
        Total water trapped
    """
    if not height or len(height) < 3:
        return 0
    
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    
    while left < right:
        # Choose which side to process based on height comparison
        if height[left] < height[right]:
            # Water at current position depends on left_max
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            # Water at current position depends on right_max
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water
```

**Example with Visualization:** For the array `[0,1,0,2,1,0,1,3,2,1,2,1]`:

```
Bar heights:     [0,1,0,2,1,0,1,3,2,1,2,1]
Visual representation:
                     #
                     # #   #
         #     #     # # # #
Step-by-step calculation:
- At position 0: No water (no left barrier)
- At position 2: 1 unit (min(1,3)-0=1)
- At position 4: 1 unit (min(2,3)-1=1)
- At position 5: 2 units (min(2,3)-0=2)
- At position 6: 1 unit (min(2,3)-1=1)
- At position 9: 1 unit (min(3,2)-1=1)
Total trapped water: 6 units
```

# 3. LinkedList Problems

First, let's define a Node class for our LinkedList implementations:

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### 1. Reverse a LinkedList

**Explanation:** Reversing a linked list involves changing the direction of pointers by keeping track of previous, current, and next nodes as we traverse the list.

```python
def reverse_linked_list(head):
    """
    Reverse a linked list
    
    Args:
        head: Head of the linked list
    Returns:
        New head of the reversed linked list
    """
    prev = None
    current = head
    
    while current:
        # Save next node before changing current.next
        next_node = current.next
        
        # Reverse the pointer
        current.next = prev
        
        # Move prev and current one step forward
        prev = current
        current = next_node
    
    # prev is the new head
    return prev
```

**Example:**

```
Original list: 1 -> 2 -> 3 -> 4 -> 5 -> None
Process:
- Initial: prev=None, current=1
- Iteration 1: Save next=2, 1->None, prev=1, current=2
- Iteration 2: Save next=3, 2->1->None, prev=2, current=3
- Iteration 3: Save next=4, 3->2->1->None, prev=3, current=4
- Iteration 4: Save next=5, 4->3->2->1->None, prev=4, current=5
- Iteration 5: Save next=None, 5->4->3->2->1->None, prev=5, current=None
Reversed list: 5 -> 4 -> 3 -> 2 -> 1 -> None
```

### 2. Detect Cycle in LinkedList

**Explanation:** We use Floyd's Cycle-Finding Algorithm (Tortoise and Hare approach) with two pointers: slow and fast. If there's a cycle, the fast pointer will eventually catch up to the slow pointer.

```python
def has_cycle(head):
    """
    Determine if a linked list has a cycle
    
    Args:
        head: Head of the linked list
    Returns:
        True if cycle exists, False otherwise
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    # Fast moves twice as fast as slow
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        # If they meet, there's a cycle
        if slow == fast:
            return True
    
    # If fast reaches the end, no cycle
    return False
```

**Example:**

```
Cyclic list: 1 -> 2 -> 3 -> 4 -> 2 (points back to node 2)
- Slow starts at 1, fast starts at 1
- Iteration 1: slow=2, fast=3
- Iteration 2: slow=3, fast=2
- Iteration 3: slow=4, fast=4
- Iteration 4: slow=2, fast=3
- Iteration 5: slow=3, fast=2
- Iteration 6: slow=4, fast=4 (they meet) -> Cycle detected!
```

### 3. Find Middle of LinkedList

**Explanation:** Use two pointers: slow and fast. While fast moves twice as fast as slow, when fast reaches the end, slow will be at the middle.

```python
def middle_node(head):
    """
    Find the middle node of a linked list
    
    Args:
        head: Head of the linked list
    Returns:
        Middle node (if even length, return second middle node)
    """
    if not head:
        return None
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

**Example:**

```
List: 1 -> 2 -> 3 -> 4 -> 5
- slow=1, fast=1
- Iteration 1: slow=2, fast=3
- Iteration 2: slow=3, fast=5
- Iteration 3: slow=4, fast=None (end reached)
Middle node: 3

List: 1 -> 2 -> 3 -> 4 -> 5 -> 6
- slow=1, fast=1
- Iteration 1: slow=2, fast=3
- Iteration 2: slow=3, fast=5
- Iteration 3: slow=4, fast=None (end reached)
Middle node: 4 (second middle for even length)
```

# 4. Tree Traversals and Medium Questions

First, let's define a TreeNode class:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Tree Traversals:

#### 1. Inorder Traversal (Left-Root-Right)

```python
def inorder_traversal(root):
    """
    Inorder traversal of a binary tree
    
    Args:
        root: Root of the binary tree
    Returns:
        List of node values in inorder sequence
    """
    result = []
    
    def inorder_helper(node):
        if not node:
            return
        # Process left subtree
        inorder_helper(node.left)
        # Process current node
        result.append(node.val)
        # Process right subtree
        inorder_helper(node.right)
    
    inorder_helper(root)
    return result

# Iterative approach using a stack
def inorder_traversal_iterative(root):
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.val)
        
        # Go to right subtree
        current = current.right
    
    return result
```

#### 2. Preorder Traversal (Root-Left-Right)

```python
def preorder_traversal(root):
    """
    Preorder traversal of a binary tree
    
    Args:
        root: Root of the binary tree
    Returns:
        List of node values in preorder sequence
    """
    result = []
    
    def preorder_helper(node):
        if not node:
            return
        # Process current node
        result.append(node.val)
        # Process left subtree
        preorder_helper(node.left)
        # Process right subtree
        preorder_helper(node.right)
    
    preorder_helper(root)
    return result

# Iterative approach
def preorder_traversal_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first so left gets processed first (LIFO)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
```

#### 3. Postorder Traversal (Left-Right-Root)

```python
def postorder_traversal(root):
    """
    Postorder traversal of a binary tree
    
    Args:
        root: Root of the binary tree
    Returns:
        List of node values in postorder sequence
    """
    result = []
    
    def postorder_helper(node):
        if not node:
            return
        # Process left subtree
        postorder_helper(node.left)
        # Process right subtree
        postorder_helper(node.right)
        # Process current node
        result.append(node.val)
    
    postorder_helper(root)
    return result

# Iterative approach (more complex)
def postorder_traversal_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [(root, False)]  # (node, is_visited)
    
    while stack:
        node, is_visited = stack.pop()
        
        if is_visited:
            result.append(node.val)
        else:
            # Re-add current node as visited
            stack.append((node, True))
            # Add right then left (LIFO order)
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))
    
    return result
```

#### 4. Level Order Traversal (BFS)

```python
from collections import deque

def level_order_traversal(root):
    """
    Level order traversal of a binary tree
    
    Args:
        root: Root of the binary tree
    Returns:
        List of lists, each inner list contains node values at the same level
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

### Medium Tree Questions:

#### 1. Validate Binary Search Tree

```python
def is_valid_bst(root):
    """
    Determine if a binary tree is a valid Binary Search Tree
    
    Args:
        root: Root of the binary tree
    Returns:
        True if valid BST, False otherwise
    """
    def validate(node, low=float('-inf'), high=float('inf')):
        # Empty trees are valid BSTs
        if not node:
            return True
        
        # Check if node's value is within valid range
        if node.val <= low or node.val >= high:
            return False
        
        # Validate left and right subtrees with updated constraints
        return (validate(node.left, low, node.val) and 
                validate(node.right, node.val, high))
    
    return validate(root)
```

**Example:**

```
Valid BST:
    2
   / \
  1   3
- For node 2: Valid range is (-∞, ∞)
- For node 1: Valid range is (-∞, 2)
- For node 3: Valid range is (2, ∞)
All nodes are within their valid ranges -> Valid BST

Invalid BST:
    5
   / \
  1   4
     / \
    3   6
- For node 5: Valid range is (-∞, ∞)
- For node 1: Valid range is (-∞, 5)
- For node 4: Valid range is (5, ∞) - VIOLATION! 4 is not > 5
Invalid BST
```

#### 2. Maximum Depth of Binary Tree

```python
def max_depth(root):
    """
    Find the maximum depth of a binary tree
    
    Args:
        root: Root of the binary tree
    Returns:
        Maximum depth (number of nodes along the longest path)
    """
    if not root:
        return 0
    
    # Recursively find the depth of left and right subtrees
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    # Return the larger depth + 1 (for current node)
    return max(left_depth, right_depth) + 1
```

## 5. Heap Popular Questions - Kth Max or Min Element

In Python, we use the `heapq` module which implements a min-heap.

### Find Kth Largest Element:

```python
import heapq

def find_kth_largest(nums, k):
    """
    Find the kth largest element in an array
    
    Args:
        nums: List of integers
        k: Position (kth) to find
    Returns:
        The kth largest element
    """
    # Use a min-heap of size k
    min_heap = []
    
    for num in nums:
        # Add current element to heap
        heapq.heappush(min_heap, num)
        
        # If heap size exceeds k, remove the smallest element
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    # The smallest element in the heap is the kth largest overall
    return min_heap[0]  # or heapq.heappop(min_heap)
```

**Example:**

```
Array: [3, 2, 1, 5, 6, 4], k=2
- Iteration 1: heap=[3]
- Iteration 2: heap=[2,3]
- Iteration 3: heap=[1,3,2]
- Iteration 4: heap=[3,5,2] (popped 1)
- Iteration 5: heap=[4,5,6] (popped 3, then 2)
- Iteration 6: heap=[5,6,4] (popped 4, then reordered)
2nd largest element: 5
```

### Find Kth Smallest Element:

```python
import heapq

def find_kth_smallest(nums, k):
    """
    Find the kth smallest element in an array
    
    Args:
        nums: List of integers
        k: Position (kth) to find
    Returns:
        The kth smallest element
    """
    # Method 1: Using heapq's built-in function
    return heapq.nsmallest(k, nums)[-1]
    
    # Method 2: Using a max-heap of size k
    # Negate values to simulate max-heap with min-heap
    max_heap = []
    
    for num in nums:
        # Negate value to simulate max-heap
        heapq.heappush(max_heap, -num)
        
        # If heap size exceeds k, remove the largest element
        if len(max_heap) > k:
            heapq.heappop(max_heap)
    
    # The largest element in the max-heap is the kth smallest overall
    return -max_heap[0]  # Negate back to get original value
```

**Example:**

```
Array: [3, 2, 1, 5, 6, 4], k=2
Using max-heap approach:
- Iteration 1: heap=[-3]
- Iteration 2: heap=[-3,-2]
- Iteration 3: heap=[-2,-1] (popped -3)
- Iteration 4: heap=[-5,-2] (popped -1)
- Iteration 5: heap=[-6,-5] (popped -2)
- Iteration 6: heap=[-6,-5] (popped -4)
2nd smallest element: 2 (negate -2)
```

# 6. Dutch National Flag Problem

**Explanation:** The Dutch National Flag problem sorts an array of 0s, 1s, and 2s in a single pass using three pointers.

```python
def sort_colors(nums):
    """
    Sort an array with only 0s, 1s, and 2s
    
    Args:
        nums: List of integers (0s, 1s, and 2s)
    Returns:
        None (sorts array in-place)
    """
    # Initialize the three pointers
    low = 0        # for 0s (beginning of array)
    mid = 0        # for 1s (current element)
    high = len(nums) - 1  # for 2s (end of array)
    
    while mid <= high:
        if nums[mid] == 0:
            # Swap current element with element at low pointer
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            # 1 is already in the correct position
            mid += 1
        else:  # nums[mid] == 2
            # Swap current element with element at high pointer
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # Don't increment mid here since we need to check the swapped element
    
    return nums  # For clarity, though the function works in-place
```

**Example:**

```
Array: [2, 0, 2, 1, 1, 0]
- Initial: low=0, mid=0, high=5
- Iteration 1: nums[mid]=2, swap with high, high=4, array=[0,0,2,1,1,2]
- Iteration 2: nums[mid]=0, swap with low, low=1, mid=1, array=[0,0,2,1,1,2]
- Iteration 3: nums[mid]=0, swap with low, low=2, mid=2, array=[0,0,2,1,1,2]
- Iteration 4: nums[mid]=2, swap with high, high=3, array=[0,0,1,1,2,2]
- Iteration 5: nums[mid]=1, mid=3, array=[0,0,1,1,2,2]
- Iteration 6: nums[mid]=1, mid=4, array=[0,0,1,1,2,2]
- Iteration 7: mid > high, exit loop
Final sorted array: [0, 0, 1, 1, 2, 2]
```
