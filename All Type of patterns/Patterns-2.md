
**Islands (Matrix Traversal)**

**Usage:** This pattern describes all the efficient ways of traversing a matrix (or 2D array).

**DS Involved:** -  Matrix, Queue


![pattern_1.png](../Images/pattern_2.png)

**Sample Problems:**

Certainly! Here are some additional examples of problems related to matrix traversal and patterns:

Sure, here are the descriptions with solution links:

1. **Number of Islands**:
    Given a 2D grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. This problem can be solved using depth-first search (DFS) or breadth-first search (BFS) to traverse the grid and mark visited islands.

    [Solution link - LeetCode](https://leetcode.com/problems/number-of-islands/)

2. **Flood Fill**:
    Given a 2D array, a starting point, and a new color, fill the connected region of the starting point with the new color. This problem can be solved using a recursive depth-first search (DFS) or iterative breadth-first search (BFS) to traverse the connected components of the region.

    [Solution link - LeetCode](https://leetcode.com/problems/flood-fill/)

3. **Cycle in a Matrix**:
    Given a 2D array representing a grid, determine if there is a cycle in the matrix. A cycle occurs if there is a path of adjacent cells such that the first and last cell are the same, and all intermediate cells are distinct. This problem can be solved using depth-first search (DFS) or breadth-first search (BFS) to traverse the grid and detect cycles.

    [Solution link - GeeksforGeeks](https://www.geeksforgeeks.org/detect-cycle-in-a-direct-graph-using-colors/)

4. **Unique Paths**:
    Given a grid of size m x n, find the number of unique paths from the top-left corner to the bottom-right corner, where movement is restricted to moving right or down. This problem can be solved using dynamic programming to calculate the number of paths for each cell based on the number of paths from adjacent cells.

    [Solution link - LeetCode](https://leetcode.com/problems/unique-paths/)

5. **Word Search**:
    Given a 2D board of letters and a word, determine if the word exists in the board. The word can be constructed from letters of sequentially adjacent cells, where "adjacent" cells are horizontally or vertically neighboring. This problem can be solved using depth-first search (DFS) to traverse the board and backtracking to explore all possible paths.

    [Solution link - LeetCode](https://leetcode.com/problems/word-search/)

6. **Matrix Zigzag Traversal**:
    Given a matrix of integers, traverse the matrix in a zigzag pattern, starting from the top-left corner and ending at the bottom-right corner. This problem can be solved by alternating between moving diagonally up-right and diagonally down-left while traversing the matrix.

    [Solution link - LeetCode](https://leetcode.com/problems/diagonal-traverse/)

7. **Rotating the Box**:
    Given a 2D grid of characters representing a box, rotate the box 90 degrees clockwise. This problem can be solved by simulating the rotation operation by swapping elements of the grid in-place.

    [Solution link - LeetCode](https://leetcode.com/problems/rotating-the-box/)

These solution links should help you explore more about how various techniques such as DFS, BFS, dynamic programming, and simulation are applied in different scenarios.