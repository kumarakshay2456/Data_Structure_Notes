1.  **Sliding Window :**
	 **Usages:** This algorithmic technique is used when we need to handle the input data in a specific window size 
	 **DS Involved:**  Array, String, HashTable
		 
 
 [[pattern_1.png]](pattern_1.png)
 


**Sample Problems :**


1. **Maximum Sum Subarray of Size K**:
   Given an array of integers and a positive integer k, find the maximum sum of any contiguous subarray of size k. This problem can be efficiently solved using a sliding window approach by maintaining the sum of elements within the window as it slides through the array.

2. **Longest Substring Without Repeating Characters**:
   Given a string, find the length of the longest substring without repeating characters. This problem can be solved using a sliding window approach where the window expands when encountering a new character and contracts when encountering a repeating character.

3. **Smallest Subarray with a Sum Greater than or Equal to a Given Value**:
   Given an array of positive integers and a target value, find the length of the smallest contiguous subarray whose sum is greater than or equal to the target value. This problem can be solved using a sliding window approach where the window expands to include more elements until the sum is greater than or equal to the target value, and then contracts to find the smallest subarray.

4. **Fruit Into Baskets**:
   Given an array of integers representing the number of fruits in a row, where each fruit is represented by a different integer, and you have two baskets, you need to put maximum fruits in each basket. The baskets can only contain two distinct types of fruits. This problem can be solved using a sliding window approach where the window expands until there are more than two distinct types of fruits, and then contracts to maintain only two distinct types of fruits in the window.

5. **Longest Subarray with Ones After Replacement**:
   Given an array of 0s and 1s and an integer k, you can flip at most k 0s to 1s. Find the length of the longest contiguous subarray containing all 1s. This problem can be solved using a sliding window approach where the window expands until the number of 0s in the window is less than or equal to k, and then contracts to maintain at most k 0s in the window.

6. **Minimum Size Subarray Sum**:
   Given an array of positive integers and a target sum, find the minimum length of a contiguous subarray of which the sum is greater than or equal to the target sum. This problem can be solved using a sliding window approach where the window expands to include more elements until the sum is greater than or equal to the target sum, and then contracts to find the minimum length subarray.

7. **Longest Mountain in Array**:
   Given an array A of integers, return the length of the longest mountain. A mountain is defined by an array A that increases until a peak is reached, and then decreases. This problem can be solved using a sliding window approach where the window expands while the mountain pattern is valid and contracts otherwise.

8. **Maximum Points You Can Obtain from Cards**:
   There are several cards arranged in a row, and each card has an associated number of points. You will be given an integer array cardPoints and an integer k. Return the maximum number of points you can obtain by choosing k cards from the beginning or the end of the array. This problem can be solved using a sliding window approach where the window size is adjusted to include exactly k cards.

9. **Longest Continuous Increasing Subsequence**:
   Given an unsorted array of integers, find the length of longest continuous increasing subsequence (subarray). This problem can be solved using a sliding window approach where the window expands while the subsequence is increasing and contracts when it's not.

10. **Find All Anagrams in a String**:
    Given a string s and a non-empty string p, find all the start indices of p's anagrams in s. An anagram of a string is another string that contains the same characters, only the order of characters can be different. This problem can be solved using a sliding window approach where the window size is fixed to the length of string p and moved through string s to check for anagrams.


11. **Subarrays with K Different Integers**:
    Given an array A of positive integers, find the number of subarrays that contain exactly K different integers. This problem can be solved using a sliding window approach where the window expands to include more elements until there are exactly K different integers, and then contracts while maintaining K different integers.

12. **Longest Repeating Character Replacement**:
    Given a string s that consists of only uppercase English letters, you can perform at most k operations on that string. Find the length of the longest substring containing the same letter you can get after performing the operations. This problem can be solved using a sliding window approach where the window expands to include more characters until the number of operations required to make all characters the same is less than or equal to k.

13. **Longest Substring with At Most Two Distinct Characters**:
    Given a string s, find the length of the longest substring T that contains at most two distinct characters. This problem can be solved using a sliding window approach where the window expands to include more characters until there are more than two distinct characters, and then contracts while maintaining at most two distinct characters.

14. **Minimum Window Substring**:
    Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n). This problem can be solved using a sliding window approach where the window expands to include more characters of S until it covers all characters of T, and then contracts to find the minimum window.

15. **Longest Substring with Equal Occurrences of Two Distinct Characters**:
    Given a string s, return the maximum length of a substring containing at most two distinct characters such that the number of occurrences of each character is the same. This problem can be solved using a sliding window approach where the window expands to include more characters until there are at most two distinct characters with equal occurrences, and then contracts while maintaining the property.

16. **Longest Subarray with Ones After Replacement**:
    Given an array of 0s and 1s and an integer k, you can flip at most k 0s to 1s. Find the length of the longest contiguous subarray containing all 1s. This problem can be solved using a sliding window approach where the window expands until the number of 0s in the window is less than or equal to k, and then contracts to maintain at most k 0s in the window.

17. **Longest Subarray with Absolute Diff Less Than or Equal to Limit**:
    Given an array of integers nums and an integer limit, return the size of the longest non-empty subarray such that the absolute difference between any two elements of this subarray is less than or equal to limit. This problem can be solved using a sliding window approach where the window maintains the maximum and minimum elements encountered so far and contracts when the absolute difference exceeds the limit.

18. **Count Number of Nice Subarrays**:
    Given an array of integers nums and an integer k, a subarray is called nice if there are k odd numbers on it. Return the number of nice sub-arrays. This problem can be solved using a sliding window approach where the window expands to include more elements until the number of odd numbers in the window is equal to k, and then contracts while maintaining k odd numbers.

19. **Minimum Operations to Reduce X to Zero**:
    You are given an integer array nums and an integer x. You can pick a subarray either from the beginning or from the end of the array to remove, and you need to remove at least one element to make the sum of the remaining elements equal to x. Return the minimum number of operations to reduce x to exactly 0 if it's possible, otherwise, return -1. This problem can be solved using a sliding window approach where the window expands to include more elements until the sum exceeds x, and then contracts to find the minimum operations.

20. **Number of Substrings Containing All Three Characters**:
    Given a string s consisting only of characters 'a', 'b', and 'c', return the number of substrings containing at least one occurrence of all these characters. This problem can be solved using a sliding window approach where the window expands until it contains at least one occurrence of all three characters, and then contracts to count the substrings.


21. **Longest Subarray of 1's After Deleting One Element**:
    Given an array of 0s and 1s, find the length of the longest subarray containing only 1s after deleting one element. This problem can be solved using a sliding window approach where the window expands to include more 1s and contracts when encountering a 0, keeping track of the longest subarray length.

22. **Longest Turbulent Subarray**:
    Given an integer array arr, return the length of a maximum size turbulent subarray of arr. A subarray is turbulent if the comparison sign alternates between each adjacent pair of elements. This problem can be solved using a sliding window approach where the window expands to include more elements while maintaining the turbulent property and contracts otherwise.

23. **Subarrays with All Elements Unique**:
    Given an array of integers nums, return the number of subarrays with all distinct elements. This problem can be solved using a sliding window approach where the window expands to include more elements until a duplicate is encountered, and then contracts while maintaining all elements distinct.

24. **Max Consecutive Ones III**:
    Given an array A of 0s and 1s, and an integer K, return the length of the longest contiguous subarray of 1s after flipping at most K 0s to 1s. This problem can be solved using a sliding window approach where the window expands until the number of 0s in the window is less than or equal to K, and then contracts to find the maximum length.

25. **Minimum Number of K Consecutive Bit Flips**:
    Given an array A of 0s and 1s, and an integer K, return the minimum number of K-bit flips required so that there is no 0 in the array. This problem can be solved using a sliding window approach where the window expands until a 0 is encountered, and then flips the bits within the window and moves to the next window.

26. **Minimum Window Subsequence**:
    Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W. This problem can be solved using a sliding window approach where the window expands to find a valid subsequence and contracts to minimize the length.

27. **Number of Longest Increasing Subsequence**:
    Given an integer array nums, return the number of longest increasing subsequences. This problem can be solved using a sliding window approach where the window expands to find increasing subsequences and contracts to maintain the longest length.

28. **Max Sum of Rectangle No Larger Than K**:
    Given a non-empty 2D matrix matrix and an integer k, find the max sum of a rectangle in the matrix such that its sum is no larger than k. This problem can be solved using a sliding window approach where the window expands to find candidate rectangles and contracts to maintain the sum within the limit.

29. **String Compression**:
    Given an array of characters, compress it in-place and return the new length of the array. This problem can be solved using a sliding window approach where the window expands to count consecutive characters and contracts to write the compressed version.

30. **Longest Chunked Palindrome Decomposition**:
    Given a string text, you should perform k operations where each operation consists of selecting a non-empty substring of text and appending it to the end of another substring. Return the lexicographically largest substring. This problem can be solved using a sliding window approach where the window expands to find chunks of the palindrome and contracts to check if the chunk can be appended.


31. **Longest Mountain in Array**:
    Given an array A of integers, return the length of the longest mountain. A mountain is defined by an array A that increases until a peak is reached, and then decreases. This problem can be solved using a sliding window approach where the window expands while the mountain pattern is valid and contracts otherwise.

32. **Minimum Size Subarray Sum**:
    Given an array of positive integers and a target sum, find the minimum length of a contiguous subarray of which the sum is greater than or equal to the target sum. This problem can be solved using a sliding window approach where the window expands to include more elements until the sum is greater than or equal to the target sum, and then contracts to find the minimum length subarray.

33. **Subarray Product Less Than K**:
    Given an array of positive integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is less than k. This problem can be solved using a sliding window approach where the window expands to include more elements until the product exceeds k, and then contracts to count the subarrays.

34. **Find K-Length Substrings With No Repeated Characters**:
    Given a string S, return the number of substrings of length K with no repeated characters. This problem can be solved using a sliding window approach where the window expands to include more characters until the substring length is K, and then contracts to count the substrings with no repeated characters.

35. **Number of Subarrays with Bounded Maximum**:
    Given an integer array nums, and integers left and right, return the number of contiguous non-empty subarrays such that the value of the maximum array element in that subarray is in the range [left, right]. This problem can be solved using a sliding window approach where the window expands to include more elements until the maximum falls within the range, and then contracts to count the subarrays.
