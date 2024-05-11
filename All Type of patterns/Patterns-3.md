**Two Pointers**

**Usage:**  This technique uses two pointers to iterate input data. Generally, both pointers move in the opposite direction at a constant interval.

**DS Involved:** Array, String, LinkedList

     ![[Screenshot 2024-05-10 at 5.08.33 PM.png]]

Sample Problems:


1. **Squaring a Sorted Array**:
   Given a sorted array of integers, return a new array where each element is the square of the original array's elements in non-decreasing order. This problem can be solved using the Two Pointers technique where one pointer starts from the beginning (for negative numbers) and another pointer starts from the end (for positive numbers), allowing us to square and merge the elements in sorted order.

2. **Dutch National Flag Problem**:
   Given an array containing only 0s, 1s, and 2s, sort the array in-place. This problem can be solved using the Two Pointers technique where three pointers are used to partition the array into three regions representing the Dutch flag's colors (red, white, and blue).

3. **Minimum Window Sort**:
   Given an array of integers, return the length of the shortest subarray that, when sorted, results in the entire array being sorted in non-decreasing order. This problem can be solved using the Two Pointers technique where two pointers are used to find the boundaries of the unsorted subarray.

4. **Two Sum II - Input array is sorted**:
   Given an array of integers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. This problem can be solved using the Two Pointers technique where one pointer starts from the beginning and the other starts from the end, moving towards each other.

5. **Container With Most Water**:
   Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai), n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). Find two lines, which, together with the x-axis forms a container, such that the container contains the most water. This problem can be solved using the Two Pointers technique to optimize the area calculation by moving pointers towards each other.

6. **Trapping Rain Water**:
   Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining. This problem can be solved using the Two Pointers technique where two pointers are used to determine the maximum height of the left and right sides of the current position.

7. **3Sum**:
   Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero. This problem can be solved using the Two Pointers technique by first sorting the array and then using a combination of pointers to find the triplets.

8. **Remove Duplicates from Sorted Array**:
   Given a sorted array nums, remove the duplicates in-place such that each element appears only once and returns the new length. This problem can be solved using the Two Pointers technique where one pointer is used to iterate through the array and another pointer is used to track the position to overwrite duplicates.

9. **Longest Substring Without Repeating Characters**:
   Given a string s, find the length of the longest substring without repeating characters. This problem can be solved using the Two Pointers technique where one pointer is used to iterate through the string and another pointer is used to track the start of the current substring.

10. **Move Zeroes**:
   Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements. This problem can be solved using the Two Pointers technique where one pointer is used to iterate through the array and another pointer is used to track the position to place non-zero elements.

11. **Max Consecutive Ones III**:
   Given an array A of 0s and 1s, and an integer K, return the length of the longest contiguous subarray of 1s after flipping at most K 0s to 1s. This problem can be solved using the Two Pointers technique where one pointer tracks the start of the window and another pointer moves through the array while maintaining at most K zeros in the window.

12. **Longest Mountain in Array**:
   Given an array A of integers, return the length of the longest mountain. A mountain is defined by an array A that increases until a peak is reached, and then decreases. This problem can be solved using the Two Pointers technique where one pointer traverses upwards until the peak and another pointer traverses downwards until the end of the mountain.

13. **Longest Subarray with Ones After Replacement**:
    Given an array of 0s and 1s and an integer k, you can flip at most k 0s to 1s. Find the length of the longest contiguous subarray containing all 1s. This problem can be solved using the Two Pointers technique where one pointer tracks the start of the window and another pointer moves through the array while maintaining at most k zeros in the window.

14. **3Sum Closest**:
    Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. This problem can be solved using the Two Pointers technique combined with sorting the array first.

15. **Reverse Vowels in a String**:
    Given a string s, reverse only all the vowels of the string and return it. This problem can be solved using the Two Pointers technique where one pointer starts from the beginning of the string and another pointer starts from the end, moving towards each other to swap vowels.

16. **Valid Palindrome II**:
    Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome. This problem can be solved using the Two Pointers technique where one pointer starts from the beginning of the string and another pointer starts from the end, moving towards each other to check if characters match.

16. **Longest Substring with At Most K Distinct Characters**:
    Given a string s and an integer k, return the length of the longest substring of s that contains at most k distinct characters. This problem can be solved using the Two Pointers technique where one pointer tracks the start of the window and another pointer moves through the string, maintaining a map of characters and their counts within the window.

17. **Valid Palindrome**:
    Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases. This problem can be solved using the Two Pointers technique where one pointer starts from the beginning of the string and another pointer starts from the end, moving towards each other to check if characters match.

18. **Merge Sorted Array**:
    Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array. This problem can be solved using the Two Pointers technique where one pointer starts from the end of nums1 (to utilize the extra space) and another pointer starts from the end of nums2, moving towards the beginning of the arrays to merge them.

19. **Remove Element**:
    Given an integer array nums and an integer val, remove all occurrences of val in nums in-place and return the new length. This problem can be solved using the Two Pointers technique where one pointer is used to iterate through the array and another pointer is used to track the position to overwrite elements.

20. **Valid Parentheses**:
    Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. This problem can be solved using the Two Pointers technique where one pointer is used to iterate through the string and another pointer is used to maintain a stack of opening parentheses.

21. **Longest Repeating Character Replacement**:
    Given a string s and an integer k, you can change at most k characters of s to any other lowercase English letter. Find the length of the longest substring containing all repeating letters you can get after performing the above operations. This problem can be solved using the Two Pointers technique where one pointer tracks the start of the window and another pointer moves through the string, maintaining the count of characters within the window.

22. **Valid Mountain Array**:
    Given an array of integers arr, return true if and only if it is a valid mountain array. This problem can be solved using the Two Pointers technique where one pointer traverses upwards until the peak and another pointer traverses downwards from the peak.
