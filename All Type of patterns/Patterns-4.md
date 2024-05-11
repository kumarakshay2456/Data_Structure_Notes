**Fast & Slow Pointers**

**Usage:** Also known as Hare & Tortoise algorithm. This technique uses two pointers that traverse the input data at different speeds.

**DS Involved:** Array, String, LinkedList

      ![[Screenshot 2024-05-10 at 5.21.03 PM.png]]


**Sample Problems:**

1. **Middle of the Linked List**:
    Given a non-empty, singly linked list, return the middle node of the linked list. If there are two middle nodes, return the second middle node. This problem can be solved using the Fast & Slow Pointers technique where one pointer moves one step at a time while another pointer moves two steps at a time. When the fast pointer reaches the end of the list, the slow pointer will be at the middle node.

2. **Happy Number**:
    A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy. This problem can be solved using the Fast & Slow Pointers technique where one pointer calculates the next number in the sequence while another pointer checks for a cycle.

3. **Cycle in a Circular Array**:
    Given a circular array (the next element of the last element is the first element of the array), determine if there is a cycle in the array. This problem can be solved using the Fast & Slow Pointers technique where one pointer moves one step at a time while another pointer moves two steps at a time. If there's a cycle, the pointers will eventually meet.

4. **Intersection of Two Linked Lists**:
    Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null. This problem can be solved using the Fast & Slow Pointers technique where two pointers are used to traverse each linked list. If one pointer reaches the end of a list, it continues from the head of the other list. If the lists intersect, the pointers will meet at the intersection node.

5. **Linked List Cycle**:
   Given head, the head of a linked list, determine if the linked list has a cycle in it. This problem can be solved using the Fast & Slow Pointers technique where one pointer moves one step at a time while another pointer moves two steps at a time. If there's a cycle, the pointers will eventually meet.

6. **Linked List Cycle II**:
   Given a linked list, return the node where the cycle begins. If there is no cycle, return null. This problem can be solved using the Fast & Slow Pointers technique where one pointer moves one step at a time while another pointer moves two steps at a time. If there's a cycle, the pointers will meet at some point, and then moving one pointer back to the head and both pointers moving at the same pace will meet at the start of the cycle.

7. **Find the Duplicate Number**:
   Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one. This problem can be solved using the Fast & Slow Pointers technique where one pointer moves one step at a time while another pointer moves two steps at a time, thus detecting the cycle in the array.

8. **Linked List Cycle III**:
   Given a linked list, return the node where the cycle begins. If there is no cycle, return null. This problem can be solved using the Fast & Slow Pointers technique where one pointer moves one step at a time while another pointer moves two steps at a time. If there's a cycle, the pointers will meet at some point, and then moving one pointer back to the head and both pointers moving at the same pace will meet at the start of the cycle.


