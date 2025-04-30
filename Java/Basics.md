
# Complete Java Crash Course for Interview Preparation

## 0. Preparation

Before starting, you need to set up your development environment:

**Java Development Kit (JDK) Installation:**

1. Download the latest JDK from [Oracle's website](https://www.oracle.com/java/technologies/downloads/) or use OpenJDK
2. Install and set up environment variables (JAVA_HOME and add to PATH)
3. Verify installation by running `java -version` in your terminal

**IDE Setup:**

- IntelliJ IDEA (recommended for beginners): Download Community Edition from JetBrains
- VS Code: Install with Java Extension Pack
- Or use online IDEs like Replit if you prefer not to install anything

## 1. Java Basics

### Main Method

Every Java program starts execution from the main method:

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World!");
    }
}
```

- `public`: Access modifier allowing the method to be called from anywhere
- `static`: Method belongs to the class, not to objects of the class
- `void`: Return type (returns nothing)
- `main`: Method name
- `String[] args`: Command-line arguments passed as an array of strings

### Variables and Data Types

**Primitive Types:**

- `int`: Integer values (-2,147,483,648 to 2,147,483,647)
- `byte`: Small integers (-128 to 127)
- `short`: Medium integers (-32,768 to 32,767)
- `long`: Large integers (use L suffix: `long x = 123456789L;`)
- `float`: Floating-point numbers (use F suffix: `float x = 3.14F;`)
- `double`: Higher precision floating-point numbers
- `char`: Single characters (use single quotes: `char c = 'A';`)
- `boolean`: True/false values

**Variable Declaration:**

```java
int age = 25;
double salary = 50000.50;
char grade = 'A';
boolean isActive = true;
```

**Reference Types:**

- String: `String name = "John";`
- Arrays: `int[] numbers = {1, 2, 3, 4, 5};`
- Custom objects: `Person person = new Person();`

### Control Structures

**Conditional Statements:**

```java
int score = 85;

// If-else
if (score >= 90) {
    System.out.println("A");
} else if (score >= 80) {
    System.out.println("B");
} else {
    System.out.println("C");
}

// Switch
int day = 3;
switch (day) {
    case 1:
        System.out.println("Monday");
        break;
    case 2:
        System.out.println("Tuesday");
        break;
    default:
        System.out.println("Other day");
}
```

**Loops:**

```java
// For loop
for (int i = 0; i < 5; i++) {
    System.out.println(i);
}

// While loop
int i = 0;
while (i < 5) {
    System.out.println(i);
    i++;
}

// Do-while loop
int j = 0;
do {
    System.out.println(j);
    j++;
} while (j < 5);

// Enhanced for loop (for arrays and collections)
int[] numbers = {1, 2, 3, 4, 5};
for (int num : numbers) {
    System.out.println(num);
}
```

### Methods

```java
// Method declaration
public static int add(int a, int b) {
    return a + b;
}

// Method overloading (same name, different parameters)
public static double add(double a, double b) {
    return a + b;
}
```

### Practice Problems

**Factorial Program:**

```java
public static int factorial(int n) {
    if (n == 0 || n == 1) {
        return 1;
    }
    
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
```

**Palindrome Check:**

```java
public static boolean isPalindrome(String str) {
    str = str.toLowerCase().replaceAll("[^a-zA-Z0-9]", "");
    int left = 0;
    int right = str.length() - 1;
    
    while (left < right) {
        if (str.charAt(left) != str.charAt(right)) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}
```

**Fibonacci Series:**

```java
public static void printFibonacci(int n) {
    int a = 0, b = 1;
    
    System.out.print(a + " " + b + " ");
    
    for (int i = 2; i < n; i++) {
        int c = a + b;
        System.out.print(c + " ");
        a = b;
        b = c;
    }
}
```

## 2. Object-Oriented Programming

### Classes and Objects

```java
// Class definition
public class Person {
    // Instance variables (fields)
    private String name;
    private int age;
    
    // Constructor
    public Person(String name, int age) {
        this.name = name;  // 'this' refers to the current object
        this.age = age;
    }
    
    // Default constructor
    public Person() {
        this.name = "Unknown";
        this.age = 0;
    }
    
    // Methods
    public void introduce() {
        System.out.println("Hi, I'm " + name + " and I'm " + age + " years old.");
    }
    
    // Getters and Setters
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public int getAge() {
        return age;
    }
    
    public void setAge(int age) {
        if (age >= 0) {
            this.age = age;
        }
    }
}

// Usage
Person person1 = new Person("John", 30);
person1.introduce();
```

### Inheritance

```java
// Base class (parent)
public class Animal {
    protected String name;
    
    public Animal(String name) {
        this.name = name;
    }
    
    public void eat() {
        System.out.println(name + " is eating.");
    }
    
    public void sleep() {
        System.out.println(name + " is sleeping.");
    }
}

// Derived class (child)
public class Dog extends Animal {
    private String breed;
    
    public Dog(String name, String breed) {
        super(name);  // Call to parent constructor
        this.breed = breed;
    }
    
    // Method overriding
    @Override
    public void eat() {
        System.out.println(name + " the " + breed + " is eating dog food.");
    }
    
    // New method
    public void bark() {
        System.out.println(name + " is barking.");
    }
}

// Usage
Dog dog = new Dog("Max", "Golden Retriever");
dog.eat();    // Calls the overridden method
dog.sleep();  // Calls the inherited method
dog.bark();   // Calls the dog-specific method
```

### Polymorphism

```java
// Using the classes from above
Animal animal1 = new Animal("Generic Animal");
Animal animal2 = new Dog("Rex", "German Shepherd");  // Upcasting

animal1.eat();  // Calls Animal's eat method
animal2.eat();  // Calls Dog's eat method (runtime polymorphism)

// animal2.bark();  // Compile error - Animal doesn't have bark method
((Dog) animal2).bark();  // Downcasting to call Dog's method
```

### Abstract Classes and Methods

```java
public abstract class Shape {
    protected String color;
    
    public Shape(String color) {
        this.color = color;
    }
    
    // Abstract method (no implementation)
    public abstract double calculateArea();
    
    // Concrete method
    public void display() {
        System.out.println("This is a " + color + " shape.");
    }
}

public class Circle extends Shape {
    private double radius;
    
    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }
    
    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }
}

// Usage
// Shape shape = new Shape("Red");  // Error: Cannot instantiate abstract class
Circle circle = new Circle("Blue", 5.0);
System.out.println("Area: " + circle.calculateArea());
circle.display();  // Inherited method
```

### Interfaces

```java
public interface Flyable {
    void fly();  // Abstract by default
    
    // Default method (Java 8+)
    default void glide() {
        System.out.println("Gliding through the air");
    }
}

public interface Swimmable {
    void swim();
}

public class Bird implements Flyable {
    private String species;
    
    public Bird(String species) {
        this.species = species;
    }
    
    @Override
    public void fly() {
        System.out.println("The " + species + " bird is flying.");
    }
}

// Multiple interfaces
public class Duck extends Animal implements Flyable, Swimmable {
    public Duck(String name) {
        super(name);
    }
    
    @Override
    public void fly() {
        System.out.println(name + " is flying low.");
    }
    
    @Override
    public void swim() {
        System.out.println(name + " is swimming in the pond.");
    }
}
```

### Access Modifiers

- `public`: Accessible from anywhere
- `protected`: Accessible within the same package and subclasses
- `default` (no modifier): Accessible within the same package only
- `private`: Accessible only within the same class

### Other Keywords

- `final`: For constants, preventing inheritance or method overriding
- `static`: For class-level members (not instance-specific)

## 3. Java Collections Framework

### ArrayList

```java
import java.util.ArrayList;

ArrayList<String> names = new ArrayList<>();
names.add("John");          // Add element
names.add("Alice");
names.add("Bob");
names.add(1, "Charlie");    // Add at specific position

String name = names.get(0); // Access element
names.set(2, "Dave");       // Update element
names.remove(3);            // Remove by index
names.remove("Alice");      // Remove by value

int size = names.size();    // Get size
boolean contains = names.contains("Dave");  // Check if element exists

// Iterate
for (String s : names) {
    System.out.println(s);
}
```

### LinkedList

```java
import java.util.LinkedList;

LinkedList<Integer> numbers = new LinkedList<>();
numbers.add(10);
numbers.add(20);
numbers.add(30);

// Additional LinkedList operations
numbers.addFirst(5);        // Add to the beginning
numbers.addLast(40);        // Add to the end
int first = numbers.getFirst();
int last = numbers.getLast();
numbers.removeFirst();
numbers.removeLast();
```

### HashSet

```java
import java.util.HashSet;

HashSet<String> uniqueNames = new HashSet<>();
uniqueNames.add("John");
uniqueNames.add("Alice");
uniqueNames.add("John");    // Duplicate, will be ignored

boolean hasAlice = uniqueNames.contains("Alice");
uniqueNames.remove("Alice");
int size = uniqueNames.size();

// Iterate
for (String name : uniqueNames) {
    System.out.println(name);
}
```

### TreeSet

```java
import java.util.TreeSet;

TreeSet<Integer> sortedNumbers = new TreeSet<>();
sortedNumbers.add(30);
sortedNumbers.add(10);
sortedNumbers.add(50);
sortedNumbers.add(20);

// Elements are automatically sorted
for (int num : sortedNumbers) {
    System.out.println(num);  // Prints: 10, 20, 30, 50
}

// TreeSet specific operations
int first = sortedNumbers.first();
int last = sortedNumbers.last();
int higher = sortedNumbers.higher(20);  // Next higher element after 20
int lower = sortedNumbers.lower(30);    // Next lower element before 30
```

### HashMap

```java
import java.util.HashMap;

HashMap<String, Integer> studentMarks = new HashMap<>();
studentMarks.put("John", 90);
studentMarks.put("Alice", 85);
studentMarks.put("Bob", 78);

int johnMarks = studentMarks.get("John");
boolean hasAlice = studentMarks.containsKey("Alice");
studentMarks.remove("Bob");

// Iterate over keys
for (String student : studentMarks.keySet()) {
    System.out.println(student);
}

// Iterate over values
for (int marks : studentMarks.values()) {
    System.out.println(marks);
}

// Iterate over entries
for (HashMap.Entry<String, Integer> entry : studentMarks.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}
```

### TreeMap

```java
import java.util.TreeMap;

TreeMap<String, Integer> sortedMap = new TreeMap<>();
sortedMap.put("Charlie", 85);
sortedMap.put("Alice", 90);
sortedMap.put("Bob", 78);

// Keys are automatically sorted
for (String key : sortedMap.keySet()) {
    System.out.println(key + ": " + sortedMap.get(key));
}
```

### Practical Examples

**Remove Duplicates Using Set:**

```java
public static Integer[] removeDuplicates(Integer[] array) {
    HashSet<Integer> set = new HashSet<>();
    for (int num : array) {
        set.add(num);
    }
    return set.toArray(new Integer[0]);
}
```

**Student Marks Management:**

```java
import java.util.*;

public class StudentDatabase {
    private HashMap<String, Integer> studentMarks = new HashMap<>();
    
    public void addStudent(String name, int marks) {
        studentMarks.put(name, marks);
    }
    
    public int getMarks(String name) {
        return studentMarks.getOrDefault(name, -1);
    }
    
    public List<String> getTopStudents(int n) {
        // Convert to list of entries
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(studentMarks.entrySet());
        
        // Sort by marks (descending)
        Collections.sort(entries, (a, b) -> b.getValue().compareTo(a.getValue()));
        
        // Extract top n names
        List<String> topStudents = new ArrayList<>();
        for (int i = 0; i < Math.min(n, entries.size()); i++) {
            topStudents.add(entries.get(i).getKey());
        }
        
        return topStudents;
    }
}
```

## 4. Exception Handling and File I/O

### Exception Handling

```java
// Basic try-catch
try {
    int result = 10 / 0;  // ArithmeticException
    System.out.println(result);
} catch (ArithmeticException e) {
    System.out.println("Error: " + e.getMessage());
}

// Multiple catch blocks
try {
    int[] arr = new int[5];
    arr[10] = 50;  // ArrayIndexOutOfBoundsException
} catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("Array index error: " + e.getMessage());
} catch (Exception e) {
    System.out.println("General error: " + e.getMessage());
}

// try-catch-finally
try {
    // Code that might throw an exception
    String str = null;
    str.length();  // NullPointerException
} catch (NullPointerException e) {
    System.out.println("Null reference: " + e.getMessage());
} finally {
    System.out.println("This will always execute");
}

// throw keyword
public static void validateAge(int age) {
    if (age < 0) {
        throw new IllegalArgumentException("Age cannot be negative");
    }
}

// throws declaration
public static void readFile(String filename) throws IOException {
    // Code to read file
}
```

### Checked vs Unchecked Exceptions

- **Checked Exceptions**: Must be handled explicitly (with try-catch or throws)
    - Examples: IOException, SQLException, ClassNotFoundException
- **Unchecked Exceptions**: Don't need to be declared or caught
    - Examples: NullPointerException, ArrayIndexOutOfBoundsException, ArithmeticException

### File I/O

```java
import java.io.*;

// Writing to a file
public static void writeToFile(String filename, String content) {
    try (FileWriter writer = new FileWriter(filename);
         BufferedWriter bufferedWriter = new BufferedWriter(writer)) {
        
        bufferedWriter.write(content);
        System.out.println("Successfully wrote to the file");
        
    } catch (IOException e) {
        System.out.println("An error occurred while writing to the file");
        e.printStackTrace();
    }
}

// Reading from a file
public static String readFromFile(String filename) {
    StringBuilder content = new StringBuilder();
    
    try (FileReader reader = new FileReader(filename);
         BufferedReader bufferedReader = new BufferedReader(reader)) {
        
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            content.append(line).append("\n");
        }
        
    } catch (IOException e) {
        System.out.println("An error occurred while reading the file");
        e.printStackTrace();
    }
    
    return content.toString();
}

// Usage
writeToFile("sample.txt", "Hello, this is a test file.");
String fileContent = readFromFile("sample.txt");
System.out.println(fileContent);
```

## 5. Java Interview Coding Practice

### Reverse a String

```java
public static String reverseString(String str) {
    char[] chars = str.toCharArray();
    int left = 0;
    int right = chars.length - 1;
    
    while (left < right) {
        // Swap characters
        char temp = chars[left];
        chars[left] = chars[right];
        chars[right] = temp;
        
        // Move pointers
        left++;
        right--;
    }
    
    return new String(chars);
}

// Alternative using StringBuilder
public static String reverseStringBuilder(String str) {
    return new StringBuilder(str).reverse().toString();
}
```

### Check for Anagrams

```java
public static boolean areAnagrams(String str1, String str2) {
    // Remove spaces and convert to lowercase
    str1 = str1.replaceAll("\\s", "").toLowerCase();
    str2 = str2.replaceAll("\\s", "").toLowerCase();
    
    // If lengths are different, they can't be anagrams
    if (str1.length() != str2.length()) {
        return false;
    }
    
    // Convert to char arrays and sort
    char[] chars1 = str1.toCharArray();
    char[] chars2 = str2.toCharArray();
    Arrays.sort(chars1);
    Arrays.sort(chars2);
    
    // Compare sorted arrays
    return Arrays.equals(chars1, chars2);
}

// Alternative using character frequency
public static boolean areAnagramsUsingMap(String str1, String str2) {
    // Remove spaces and convert to lowercase
    str1 = str1.replaceAll("\\s", "").toLowerCase();
    str2 = str2.replaceAll("\\s", "").toLowerCase();
    
    // If lengths are different, they can't be anagrams
    if (str1.length() != str2.length()) {
        return false;
    }
    
    // Create frequency map
    HashMap<Character, Integer> freqMap = new HashMap<>();
    
    // Increment count for each character in str1
    for (char c : str1.toCharArray()) {
        freqMap.put(c, freqMap.getOrDefault(c, 0) + 1);
    }
    
    // Decrement count for each character in str2
    for (char c : str2.toCharArray()) {
        if (!freqMap.containsKey(c) || freqMap.get(c) == 0) {
            return false;
        }
        freqMap.put(c, freqMap.get(c) - 1);
    }
    
    return true;
}
```

### Find Duplicates in an Array

```java
public static List<Integer> findDuplicates(int[] nums) {
    List<Integer> duplicates = new ArrayList<>();
    HashSet<Integer> seen = new HashSet<>();
    
    for (int num : nums) {
        if (!seen.add(num)) {  // add() returns false if element already exists
            duplicates.add(num);
        }
    }
    
    return duplicates;
}
```

### Word Frequency Counter

```java
public static HashMap<String, Integer> countWordFrequency(String text) {
    // Split text into words
    String[] words = text.toLowerCase().split("\\W+");
    
    // Count frequency using HashMap
    HashMap<String, Integer> wordFrequency = new HashMap<>();
    
    for (String word : words) {
        if (!word.isEmpty()) {  // Skip empty strings
            wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
        }
    }
    
    return wordFrequency;
}

// Display results
public static void printWordFrequency(String text) {
    HashMap<String, Integer> frequency = countWordFrequency(text);
    
    // Sort by frequency (descending)
    List<Map.Entry<String, Integer>> entries = new ArrayList<>(frequency.entrySet());
    entries.sort((a, b) -> b.getValue().compareTo(a.getValue()));
    
    // Print results
    for (Map.Entry<String, Integer> entry : entries) {
        System.out.println(entry.getKey() + ": " + entry.getValue());
    }
}
```

### Implement a Stack Using ArrayList

```java
public class Stack<T> {
    private ArrayList<T> elements;
    
    public Stack() {
        elements = new ArrayList<>();
    }
    
    // Push element onto the stack
    public void push(T element) {
        elements.add(element);
    }
    
    // Pop element from the stack
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return elements.remove(elements.size() - 1);
    }
    
    // Peek at the top element without removing it
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return elements.get(elements.size() - 1);
    }
    
    // Check if stack is empty
    public boolean isEmpty() {
        return elements.isEmpty();
    }
    
    // Get the size of the stack
    public int size() {
        return elements.size();
    }
}

// Usage
Stack<Integer> stack = new Stack<>();
stack.push(10);
stack.push(20);
stack.push(30);
System.out.println(stack.pop());   // 30
System.out.println(stack.peek());  // 20
System.out.println(stack.size());  // 2
```

## 6. Common Java Interview Questions and Answers

### == vs .equals()

- `==` compares object references (memory addresses) for reference types
- `.equals()` compares the content/values of objects
- For primitives, `==` compares values

Example:

```java
String s1 = "Hello";
String s2 = "Hello";
String s3 = new String("Hello");

System.out.println(s1 == s2);          // true (same string pool reference)
System.out.println(s1 == s3);          // false (different objects)
System.out.println(s1.equals(s3));     // true (same content)
```

### ArrayList vs LinkedList

- **ArrayList**:
    
    - Backed by dynamic array
    - Fast random access (O(1) for get/set)
    - Slow insertions/deletions in the middle (O(n))
    - Good for frequent access, infrequent modifications
- **LinkedList**:
    
    - Implemented as doubly-linked list
    - Slow random access (O(n))
    - Fast insertions/deletions anywhere (O(1) if you have the position)
    - Good for frequent modifications, infrequent random access
    - Has additional methods like addFirst(), addLast()

### Why Use Interfaces?

- Define a contract for classes to implement
- Achieve abstraction by hiding implementation details
- Support multiple inheritance (a class can implement multiple interfaces)
- Enable polymorphism
- Facilitate loose coupling between components
- Allow for easy testing through mocking

### Abstract Class vs Interface

- **Abstract Class**:
    
    - Can have both abstract and concrete methods
    - Can have instance variables, constructors
    - A class can extend only one abstract class
    - Can have access modifiers for methods, properties
    - Use when related classes share code
- **Interface**:
    
    - Prior to Java 8: All methods are abstract (no implementation)
    - Java 8+: Can have default and static methods with implementation
    - Can only have constants (public static final)
    - A class can implement multiple interfaces
    - All methods are implicitly public
    - Use when unrelated classes implement the same behavior

### What is a Constructor? Can it be Overridden?

- A constructor is a special method that initializes a new object
- It has the same name as the class and no return type
- Constructors can be overloaded (multiple constructors with different parameters)
- Constructors are NOT inherited, so they CANNOT be overridden
- A subclass constructor can call the superclass constructor using `super()`

```java
public class Person {
    private String name;
    
    // Constructor
    public Person(String name) {
        this.name = name;
    }
}

public class Employee extends Person {
    private int employeeId;
    
    public Employee(String name, int employeeId) {
        super(name);  // Call parent constructor
        this.employeeId = employeeId;
    }
}
```

### What is Encapsulation? How is it Achieved in Java?

- Encapsulation is bundling data and methods that operate on that data within a single unit (class)
- It hides the internal state of objects from the outside world
- Achieved in Java through:
    1. Declaring variables as private
    2. Providing public getter and setter methods
    3. Implementing validation in setter methods

```java
public class BankAccount {
    private double balance;  // Private variable
    
    // Public getter
    public double getBalance() {
        return balance;
    }
    
    // Public setter with validation
    public void deposit(double amount) {
        if (amount > 0) {
            this.balance += amount;
        } else {
            throw new IllegalArgumentException("Deposit amount must be positive");
        }
    }
    
    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            this.balance -= amount;
        } else {
            throw new IllegalArgumentException("Invalid withdrawal amount");
        }
    }
}
```

## 7. Optional: Spring Boot Introduction

### What is Spring Boot?

Spring Boot is a framework that simplifies the development of Spring applications by providing:

- Auto-configuration
- Standalone applications with embedded servers
- Simplified dependency management
- Production-ready features (metrics, health checks)

### Key Annotations

- `@SpringBootApplication`: Combines @Configuration, @EnableAutoConfiguration, and @ComponentScan
- `@RestController`: Marks a class as a REST controller (combines @Controller and @ResponseBody)
- `@GetMapping`, `@PostMapping`, etc.: HTTP method-specific shortcuts for @RequestMapping
- `@Autowired`: For dependency injection
- `@Service`: Marks a class as a service component
- `@Repository`: Marks a class as a repository (data access component)

### Simple REST API Example

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;

@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}

@RestController
@RequestMapping("/api")
class UserController {
    
    @GetMapping("/users")
    public List<User> getAllUsers() {
        // Return list of users
        return Arrays.asList(
            new User(1, "John"),
            new User(2, "Alice")
        );
    }
    
    @GetMapping("/users/{id}")
    public User getUserById(@PathVariable int id) {
        // Return user by ID
        return new User(id, "User " + id);
    }
    
    @PostMapping("/users")
    public User createUser(@RequestBody User user) {
        // Create new user
        System.out.println("Created user: " + user.getName());
        return user;
    }
}

class User {
    private int id;
    private String name;
    
    // Constructor, getters, setters
    public User(int id, String name) {
        this.id = id;
        this.name = name;
    }
    
    public int getId() {
        return id;
    }
    
    public String getName() {
        return name;
    }
}
```

## Java Cheatsheet Summary

### OOP Principles

- **Encapsulation**: Hiding data and exposing through methods
- **Inheritance**: Creating new classes based on existing ones
- **Polymorphism**: Objects can take different forms
- **Abstraction**: Hiding complex implementation details

### Java Syntax Reference

- **Class**: `public class MyClass {}`
- **Main method**: `public static void main(String[] args) {}`
- **Variable**: `int x = 10;`
- **Array**: `int[] numbers = {1, 2, 3};`
- **Loop**: `for (int i = 0; i < 10; i++) {}`
- **Conditional**: `if (condition) {} else {}`
- **Method**: `public int add(int a, int b) { return a + b; }`

### Collections Hierarchy

- **List**: Ordered collection (ArrayList, LinkedList)
- **Set**: Unique elements (HashSet, TreeSet)
- **Map**: Key-value pairs (HashMap, TreeMap)
- **Queue**: FIFO structure (LinkedList, PriorityQueue)
- **Stack**: LIFO structure

### Exceptions

- **Try-catch**: `try {} catch (Exception e) {}`
- **Finally**: `finally {}`
- **Throw**: `throw new Exception()`
- **Throws**: `public void method() throws Exception {}`
- **Checked exceptions**: Must be handled (IOException)
- **Unchecked exceptions**: Runtime exceptions (NullPointerException)

## Conclusion

This comprehensive guide covers all the essential Java concepts needed for interview preparation. By following this roadmap and practicing these examples, you'll gain a solid foundation in Java programming. Remember that practice is key to mastering these concepts, so try to implement and experiment with the code examples provided.

For further learning, consider exploring:

1. Multithreading and concurrency
2. Java 8+ features (Streams, Lambda expressions)
3. JVM internals and memory management
4. Design patterns in Java
5. Advanced Spring Boot concepts (if pursuing backend development)