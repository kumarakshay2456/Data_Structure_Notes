
# 7. Merge Intervals

**Explanation:** The merge intervals problem asks us to merge all overlapping intervals in a collection and return the non-overlapping intervals.

```python
def merge_intervals(intervals):
    """
    Merge overlapping intervals
    
    Args:
        intervals: List of interval pairs [start, end]
    Returns:
        List of merged non-overlapping intervals
    """
    # Handle edge case
    if not intervals:
        return []
    
    # Sort intervals based on start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]  # Result list starting with first interval
    
    for current in intervals[1:]:
        # Get the last interval in our merged list
        previous = merged[-1]
        
        # If current interval overlaps with previous
        if current[0] <= previous[1]:
            # Merge them by updating the end time of previous interval
            previous[1] = max(previous[1], current[1])
        else:
            # No overlap, add current interval to result
            merged.append(current)
    
    return merged
```

**Example:**

```
Intervals: [[1,3], [2,6], [8,10], [15,18]]
- Sort (already sorted by start time)
- Initialize merged=[[1,3]]
- Interval [2,6]: overlaps with [1,3], merge to [1,6]
- Interval [8,10]: no overlap with [1,6], add to merged
- Interval [15,18]: no overlap with [8,10], add to merged
Result: [[1,6], [8,10], [15,18]]

Intervals: [[1,4], [4,5]]
- Initialize merged=[[1,4]]
- Interval [4,5]: 4 <= 4, merged to [1,5]
Result: [[1,5]]
```

# 8. Design LRU Cache

**Explanation:** An LRU (Least Recently Used) Cache is a data structure that maintains a fixed-size cache, evicting the least recently used items when the cache is full.

```python
class LRUCache:
    """
    LRU Cache implementation using a dictionary and doubly linked list
    """
    class Node:
        """Helper doubly linked list node class"""
        def __init__(self, key=0, value=0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity):
        """
        Initialize the LRU Cache
        
        Args:
            capacity: Maximum number of items the cache can hold
        """
        self.capacity = capacity
        self.cache = {}  # Maps key to node
        
        # Initialize dummy head and tail nodes
        self.head = self.Node()  # Most recently used
        self.tail = self.Node()  # Least recently used
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head (most recently used position)"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove a node from the linked list"""
        prev = node.prev
        next_node = node.next
        
        prev.next = next_node
        next_node.prev = prev
    
    def _move_to_head(self, node):
        """Move a node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove and return the least recently used node (before tail)"""
        node = self.tail.prev
        self._remove_node(node)
        return node
    
    def get(self, key):
        """
        Retrieve value by key and mark as recently used
        
        Args:
            key: Key to retrieve
        Returns:
            Value if key exists, -1 otherwise
        """
        if key not in self.cache:
            return -1
        
        # Update recently used status
        node = self.cache[key]
        self._move_to_head(node)
        return node.value
    
    def put(self, key, value):
        """
        Add or update key-value pair
        
        Args:
            key: Key to add/update
            value: Value to store
        """
        # If key exists, update value and move to head
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
            return
        
        # Create new node and add to cache
        new_node = self.Node(key, value)
        self.cache[key] = new_node
        self._add_node(new_node)
        
        # If over capacity, remove least recently used item
        if len(self.cache) > self.capacity:
            lru_node = self._pop_tail()
            del self.cache[lru_node.key]
```

**Example:**

```
Operations:
1. LRUCache(2)       # Initialize with capacity 2
2. put(1, 1)         # Cache: {1=1}
3. put(2, 2)         # Cache: {1=1, 2=2}
4. get(1)            # Return 1, Cache: {2=2, 1=1} (1 is now most recent)
5. put(3, 3)         # Evict key 2, Cache: {1=1, 3=3}
6. get(2)            # Return -1 (not found)
7. put(4, 4)         # Evict key 1, Cache: {3=3, 4=4}
8. get(1)            # Return -1 (not found)
9. get(3)            # Return 3, Cache: {4=4, 3=3} (3 is now most recent)
10. get(4)           # Return 4, Cache: {3=3, 4=4} (4 is now most recent)
```

# 9. HashMap Internal Working

**Explanation:** Here's how a HashMap works internally in Java/Python:

1. **Hash Function**: Converts keys to integers (hash codes)
2. **Compression Function**: Maps hash code to an index in the underlying array
3. **Collision Resolution**: Handles when multiple keys map to the same index

Let's implement a simple HashMap:

```python
class HashMap:
    """
    Simple HashMap implementation with separate chaining for collision resolution
    """
    class Entry:
        """Key-value pair entry"""
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.next = None
    
    def __init__(self, capacity=16, load_factor=0.75):
        """
        Initialize empty HashMap
        
        Args:
            capacity: Initial capacity (default: 16)
            load_factor: Load factor threshold for resizing (default: 0.75)
        """
        self.capacity = capacity
        self.size = 0
        self.load_factor = load_factor
        self.buckets = [None] * capacity
    
    def _hash(self, key):
        """Generate hash for key and compress to bucket index"""
        return hash(key) % self.capacity
    
    def put(self, key, value):
        """
        Add or update key-value pair
        
        Args:
            key: Key to add/update
            value: Value to store
        """
        # Get index from hash function
        index = self._hash(key)
        
        # If bucket is empty
        if not self.buckets[index]:
            self.buckets[index] = self.Entry(key, value)
            self.size += 1
        else:
            # Traverse chain to find key or end
            current = self.buckets[index]
            
            # Check if key already exists in chain
            while current:
                if current.key == key:
                    current.value = value  # Update existing key
                    return
                if not current.next:
                    break
                current = current.next
            
            # Key not found, add to end of chain
            current.next = self.Entry(key, value)
            self.size += 1
        
        # Check load factor and resize if necessary
        if self.size / self.capacity >= self.load_factor:
            self._resize()
    
    def get(self, key):
        """
        Retrieve value by key
        
        Args:
            key: Key to retrieve
        Returns:
            Value if key exists, None otherwise
        """
        index = self._hash(key)
        current = self.buckets[index]
        
        # Traverse chain to find key
        while current:
            if current.key == key:
                return current.value
            current = current.next
        
        return None
    
    def remove(self, key):
        """
        Remove key-value pair
        
        Args:
            key: Key to remove
        Returns:
            True if key was removed, False otherwise
        """
        index = self._hash(key)
        current = self.buckets[index]
        previous = None
        
        # Empty bucket
        if not current:
            return False
        
        # Key found at head of chain
        if current.key == key:
            self.buckets[index] = current.next
            self.size -= 1
            return True
        
        # Traverse chain
        previous = current
        current = current.next
        while current:
            if current.key == key:
                previous.next = current.next
                self.size -= 1
                return True
            previous = current
            current = current.next
        
        return False
    
    def _resize(self):
        """Double the capacity and rehash all entries"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [None] * self.capacity
        
        # Rehash all existing entries
        for bucket in old_buckets:
            current = bucket
            while current:
                self.put(current.key, current.value)
                current = current.next
```

**How a Production HashMap Works:**

1. **Initial Capacity**: Starts with a fixed size array (typically a power of 2)
2. **Hash Function**: Good hash functions distribute keys uniformly
3. **Collision Resolution**:
    - **Separate Chaining**: Each bucket contains a linked list of entries (like our example)
    - **Open Addressing**: Probing for next available slot (linear, quadratic, or double hashing)
4. **Load Factor**: When entries / capacity exceeds load factor, the map resizes
5. **Resizing**: Typically doubles in size and rehashes all elements

**Example:**

```
Operations on the HashMap:
1. put("apple", 10)  # hash("apple") % 16 = 0, buckets[0] = Entry("apple", 10)
2. put("banana", 20) # hash("banana") % 16 = 5, buckets[5] = Entry("banana", 20)
3. put("cherry", 30) # hash("cherry") % 16 = 0, buckets[0] = Entry("apple", 10) -> Entry("cherry", 30)
4. get("apple")      # Traverse buckets[0], return 10
5. get("cherry")     # Traverse buckets[0], return 30
6. remove("apple")   # Remove from buckets[0] chain, leaving only Entry("cherry", 30)
```

# 10. Stream API Questions - Find First Non-Repeating Element

**Explanation:** Using Java Stream API to find the first non-repeating element in an array.

```java
// Java implementation
public static <T> Optional<T> findFirstNonRepeating(List<T> list) {
    Map<T, Long> frequency = list.stream()
        .collect(Collectors.groupingBy(
            Function.identity(),
            Collectors.counting()
        ));
    
    return list.stream()
        .filter(item -> frequency.get(item) == 1)
        .findFirst();
}
```

Python equivalent using collections:

```python
from collections import Counter

def find_first_non_repeating(arr):
    """
    Find the first non-repeating element in an array
    
    Args:
        arr: List of elements
    Returns:
        First non-repeating element or None if not found
    """
    # Count occurrences of each element
    counter = Counter(arr)
    
    # Find first element with count 1
    for element in arr:
        if counter[element] == 1:
            return element
    
    return None
```

**Example:**

```
Array: [4, 5, 1, 2, 5, 1, 2, 3, 4, 3, 6]
Counts: {4:2, 5:2, 1:2, 2:2, 3:2, 6:1}
First non-repeating: 6

Array: [1, 2, 3, 1, 2, 3]
Counts: {1:2, 2:2, 3:2}
First non-repeating: None (all elements repeat)

Array: [7, 5, 3, 2, 1]
Counts: {7:1, 5:1, 3:1, 2:1, 1:1}
First non-repeating: 7 (first element)
```

# 11. Comparator and Comparable Difference (Java)

**Explanation:**

1. **Comparable**:
    
    - Interface implemented by the class itself
    - Natural ordering of objects
    - Has a single method: `compareTo()`
    - Objects can be sorted using `Collections.sort()` without additional parameters
2. **Comparator**:
    
    - External utility class
    - Multiple custom orderings possible
    - Has a single method: `compare()`
    - Must be provided when calling `Collections.sort(list, comparator)`

```java
// Comparable example (Java)
public class Person implements Comparable<Person> {
    private String name;
    private int age;
    
    // Constructor and getters/setters...
    
    @Override
    public int compareTo(Person other) {
        // Natural order: by age
        return this.age - other.age;
    }
}

// Usage:
List<Person> people = new ArrayList<>();
// Add people...
Collections.sort(people); // Sorts by age automatically
```

```java
// Comparator example (Java)
public class NameComparator implements Comparator<Person> {
    @Override
    public int compare(Person p1, Person p2) {
        return p1.getName().compareTo(p2.getName());
    }
}

// Usage:
List<Person> people = new ArrayList<>();
// Add people...
Collections.sort(people, new NameComparator()); // Sort by name
```

Python equivalents:

```python
# Python Comparable equivalent (using __lt__ magic method)
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __lt__(self, other):
        # Natural order: by age
        return self.age < other.age

# Usage:
people = [Person("Alice", 30), Person("Bob", 25)]
sorted_people = sorted(people)  # Sorts by age automatically
```

```python
# Python Comparator equivalent (using key function)
people = [Person("Alice", 30), Person("Bob", 25)]

# Sort by name
sorted_by_name = sorted(people, key=lambda person: person.name)

# Sort by age
sorted_by_age = sorted(people, key=lambda person: person.age)
```

**Key Differences:**

1. **Implementation**: Comparable is implemented by the class itself, Comparator is a separate class
2. **Methods**: Comparable uses `compareTo()`, Comparator uses `compare()`
3. **Flexibility**:
    - Comparable: One natural ordering per class
    - Comparator: Multiple custom orderings possible
4. **Control**: Comparable requires modifying the class, Comparator can be used when you can't modify the class
5. **Usage**: Comparable works directly with sort methods, Comparator must be explicitly provided

# 12. Implement Core Data Structures in Go

### Stack Implementation in Go

```go
package main

import (
    "errors"
    "fmt"
)

// Stack represents a simple LIFO stack
type Stack struct {
    elements []interface{}
}

// NewStack creates a new empty stack
func NewStack() *Stack {
    return &Stack{elements: make([]interface{}, 0)}
}

// Push adds an element to the top of the stack
func (s *Stack) Push(element interface{}) {
    s.elements = append(s.elements, element)
}

// Pop removes and returns the top element
func (s *Stack) Pop() (interface{}, error) {
    if s.IsEmpty() {
        return nil, errors.New("stack is empty")
    }
    
    index := len(s.elements) - 1
    element := s.elements[index]
    s.elements = s.elements[:index]
    return element, nil
}

// Peek returns the top element without removing it
func (s *Stack) Peek() (interface{}, error) {
    if s.IsEmpty() {
        return nil, errors.New("stack is empty")
    }
    
    return s.elements[len(s.elements)-1], nil
}

// IsEmpty checks if the stack is empty
func (s *Stack) IsEmpty() bool {
    return len(s.elements) == 0
}

// Size returns the number of elements in the stack
func (s *Stack) Size() int {
    return len(s.elements)
}

// Example usage:
func main() {
    stack := NewStack()
    stack.Push(1)
    stack.Push(2)
    stack.Push(3)
    
    element, _ := stack.Pop()
    fmt.Println("Popped:", element) // 3
    
    element, _ = stack.Peek()
    fmt.Println("Peeked:", element) // 2
    
    fmt.Println("Size:", stack.Size()) // 2
}
```

### LinkedList Implementation in Go

```go
package main

import "fmt"

// Node represents a node in a linked list
type Node struct {
    Value interface{}
    Next  *Node
}

// LinkedList represents a singly linked list
type LinkedList struct {
    head *Node
    tail *Node
    size int
}

// NewLinkedList creates a new empty linked list
func NewLinkedList() *LinkedList {
    return &LinkedList{head: nil, tail: nil, size: 0}
}

// AddFirst adds an element to the beginning of the list
func (ll *LinkedList) AddFirst(value interface{}) {
    newNode := &Node{Value: value, Next: ll.head}
    ll.head = newNode
    
    if ll.tail == nil {
        ll.tail = newNode
    }
    
    ll.size++
}

// AddLast adds an element to the end of the list
func (ll *LinkedList) AddLast(value interface{}) {
    newNode := &Node{Value: value, Next: nil}
    
    if ll.tail == nil {
        ll.head = newNode
        ll.tail = newNode
    } else {
        ll.tail.Next = newNode
        ll.tail = newNode
    }
    
    ll.size++
}

// RemoveFirst removes and returns the first element
func (ll *LinkedList) RemoveFirst() (interface{}, bool) {
    if ll.head == nil {
        return nil, false
    }
    
    value := ll.head.Value
    ll.head = ll.head.Next
    ll.size--
    
    if ll.head == nil {
        ll.tail = nil
    }
    
    return value, true
}

// Contains checks if a value is in the list
func (ll *LinkedList) Contains(value interface{}) bool {
    current := ll.head
    
    for current != nil {
        if current.Value == value {
            return true
        }
        current = current.Next
    }
    
    return false
}

// Size returns the number of elements in the list
func (ll *LinkedList) Size() int {
    return ll.size
}

// IsEmpty checks if the list is empty
func (ll *LinkedList) IsEmpty() bool {
    return ll.size == 0
}

// Display prints the list elements
func (ll *LinkedList) Display() {
    current := ll.head
    fmt.Print("List: ")
    
    for current != nil {
        fmt.Printf("%v ", current.Value)
        current = current.Next
    }
    
    fmt.Println()
}

// Example usage:
func main() {
    list := NewLinkedList()
    list.AddLast(1)
    list.AddLast(2)
    list.AddFirst(0)
    
    list.Display() // List: 0 1 2
    
    value, _ := list.RemoveFirst()
    fmt.Println("Removed:", value) // 0
    
    list.Display() // List: 1 2
    
    fmt.Println("Contains 2:", list.Contains(2)) // true
    fmt.Println("Size:", list.Size()) // 2
}
```

### BST (Binary Search Tree) Implementation in Go

```go
package main

import "fmt"

// TreeNode represents a node in a binary search tree
type TreeNode struct {
    Value int
    Left  *TreeNode
    Right *TreeNode
}

// BST represents a binary search tree
type BST struct {
    root *TreeNode
}

// NewBST creates a new empty binary search tree
func NewBST() *BST {
    return &BST{root: nil}
}

// Insert adds a value to the BST
func (bst *BST) Insert(value int) {
    bst.root = bst.insertRecursive(bst.root, value)
}

// insertRecursive is a helper method for Insert
func (bst *BST) insertRecursive(node *TreeNode, value int) *TreeNode {
    if node == nil {
        return &TreeNode{Value: value, Left: nil, Right: nil}
    }
    
    if value < node.Value {
        node.Left = bst.insertRecursive(node.Left, value)
    } else if value > node.Value {
        node.Right = bst.insertRecursive(node.Right, value)
    }
    
    return node
}

// Search checks if a value is in the BST
func (bst *BST) Search(value int) bool {
    return bst.searchRecursive(bst.root, value)
}

// searchRecursive is a helper method for Search
func (bst *BST) searchRecursive(node *TreeNode, value int) bool {
    if node == nil {
        return false
    }
    
    if value == node.Value {
        return true
    } else if value < node.Value {
        return bst.searchRecursive(node.Left, value)
    } else {
        return bst.searchRecursive(node.Right, value)
    }
}

// Delete removes a value from the BST
func (bst *BST) Delete(value int) {
    bst.root = bst.deleteRecursive(bst.root, value)
}

// deleteRecursive is a helper method for Delete
func (bst *BST) deleteRecursive(node *TreeNode, value int) *TreeNode {
    if node == nil {
        return nil
    }
    
    if value < node.Value {
        node.Left = bst.deleteRecursive(node.Left, value)
    } else if value > node.Value {
        node.Right = bst.deleteRecursive(node.Right, value)
    } else {
        // Case 1: Leaf node (no children)
        if node.Left == nil && node.Right == nil {
            return nil
        }
        
        // Case 2: One child
        if node.Left == nil {
            return node.Right
        }
        if node.Right == nil {
            return node.Left
        }
        
        // Case 3: Two children
        // Find inorder successor (smallest node in right subtree)
        successor := bst.findMin(node.Right)
        node.Value = successor.Value
        node.Right = bst.deleteRecursive(node.Right, successor.Value)
    }
    
    return node
}

// findMin finds the node with minimum value in a subtree
func (bst *BST) findMin(node *TreeNode) *TreeNode {
    current := node
    
    for current != nil && current.Left != nil {
        current = current.Left
    }
    
    return current
}

// InOrderTraversal traverses the BST in-order (left-root-right)
func (bst *BST) InOrderTraversal() {
    fmt.Print("In-order: ")
    bst.inOrderRecursive(bst.root)
    fmt.Println()
}

// inOrderRecursive is a helper method for InOrderTraversal
func (bst *BST) inOrderRecursive(node *TreeNode) {
    if node != nil {
        bst.inOrderRecursive(node.Left)
        fmt.Printf("%d ", node.Value)
        bst.inOrderRecursive(node.Right)
    }
}

// Example usage:
func main() {
    bst := NewBST()
    bst.Insert(50)
    bst.Insert(30)
    bst.Insert(70)
    bst.Insert(20)
    bst.Insert(40)
    
    bst.InOrderTraversal() // In-order: 20 30 40 50 70
    
    fmt.Println("Contains 30:", bst.Search(30)) // true
    fmt.Println("Contains 60:", bst.Search(60)) // false
    
    bst.Delete(30)
    bst.InOrderTraversal() // In-order: 20 40 50 70
}
```

### Queue Implementation in Go

```go
package main

import (
    "errors"
    "fmt"
)

// Queue represents a simple FIFO queue
type Queue struct {
    elements []interface{}
}

// NewQueue creates a new empty queue
func NewQueue() *Queue {
    return &Queue{elements: make([]interface{}, 0)}
}

// Enqueue adds an element to the end of the queue
func (q *Queue) Enqueue(element interface{}) {
    q.elements = append(q.elements, element)
}

// Dequeue removes and returns the first element
func (q *Queue) Dequeue() (interface{}, error) {
    if q.IsEmpty() {
        return nil, errors.New("queue is empty")
    }
    
    element := q.elements[0]
    q.elements = q.elements[1:]
    return element, nil
}

// Peek returns the first element without removing it
func (q *Queue) Peek() (interface{}, error) {
    if q.IsEmpty() {
        return nil, errors.New("queue is empty")
    }
    
    return q.elements[0], nil
}

// IsEmpty checks if the queue is empty
func (q *Queue) IsEmpty() bool {
    return len(q.elements) == 0
}

// Size returns the number of elements in the queue
func (q *Queue) Size() int {
    return len(q.elements)
}

// Example usage:
func main() {
    queue := NewQueue()
    queue.Enqueue(1)
    queue.Enqueue(2)
    queue.Enqueue(3)
    
    element, _ := queue.Dequeue()
    fmt.Println("Dequeued:", element) // 1
    
    element, _ = queue.Peek()
    fmt.Println("Peeked:", element) // 2
    
    fmt.Println("Size:", queue.Size()) // 2
}
```

# 13. Cassandra Data Modeling Concepts

**Key Concepts:**

1. **Distributed NoSQL Database**:
    
    - Designed for high availability and horizontal scalability
    - No single point of failure
    - Linear scalability by adding nodes
2. **Data Model**:
    
    - Column-family (wide-column) database
    - Tables contain rows with primary keys
    - Each row contains columns (name-value pairs)
    - No joins or foreign keys
3. **Primary Key Structure**:
    
    - **Partition Key**: Determines which node stores the data
    - **Clustering Columns**: Determine sort order within a partition
4. **Query-Driven Design**:
    
    - Model data around query patterns, not entities
    - Denormalization is common to optimize reads
    - "Write once, read many times" philosophy
5. **Consistency Levels**:
    
    - Configurable consistency (ONE, QUORUM, ALL, etc.)
    - Trade-off between consistency, availability, and partition tolerance

**Data Modeling Principles:**

1. **Know Your Queries First**:
    
    - Identify all access patterns before designing tables
    - Each query pattern may require a dedicated table
2. **Denormalize for Read Performance**:
    
    - Duplicate data to avoid joins
    - Storage is cheap, performance is critical
3. **Partition Key Selection**:
    
    - Should distribute data evenly across the cluster
    - Avoid hotspots (too many reads/writes to a single partition)
    - Common partition key choices: user_id, tenant_id, date
4. **Clustering Columns for Sorting**:
    
    - Define the order of data within a partition
    - Support range queries (e.g., BETWEEN, >, <)
5. **Avoid Large Partitions**:
    
    - Keep partitions under ~100MB
    - Use composite partition keys if needed

**Example Data Model:**

Consider a user activity tracking system:

```sql
-- Query: Find user's activities by date range
CREATE TABLE user_activities_by_day (
    user_id UUID,
    activity_date DATE,
    activity_id TIMEUUID,
    activity_type TEXT,
    details MAP<TEXT, TEXT>,
    PRIMARY KEY ((user_id, activity_date), activity_id)
) WITH CLUSTERING ORDER BY (activity_id DESC);

-- Query: Find all users who performed a specific activity
CREATE TABLE activities_by_type (
    activity_type TEXT,
    activity_date DATE,
    user_id UUID,
    activity_id TIMEUUID,
    details MAP<TEXT, TEXT>,
    PRIMARY KEY ((activity_type, activity_date), user_id, activity_id)
);
```

**Anti-Patterns to Avoid:**

1. **Using a Single Partition Key for All Data**:
    
    - Creates hotspots and poor distribution
    - Example: using a constant value as partition key
2. **Unbounded Growth of a Partition**:
    
    - Using high-cardinality clustering keys without limits
    - Example: storing all user messages in a single partition
3. **Storing Large Objects**:
    
    - Cassandra is not optimized for large BLOBs
    - Better to store references to external storage
4. **Modeling Relationships Like in RDBMS**:
    
    - No joins in Cassandra; must denormalize
    - Example: trying to normalize data into many tables
5. **Querying by Non-Primary Key Columns**:
    
    - Leads to full table scans, which are inefficient
    - Solution: Create additional tables to support each query pattern

