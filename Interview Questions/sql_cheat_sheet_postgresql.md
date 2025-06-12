
# ✅ SQL Cheat Sheet for Backend Interviews (PostgreSQL Focused)

---

## 🔹 1. Basic Commands

```sql
SELECT name, age FROM users;
SELECT * FROM users;
SELECT * FROM users WHERE age > 30 AND city = 'New York';
SELECT * FROM users ORDER BY age DESC LIMIT 10 OFFSET 5;
SELECT name AS full_name FROM users;
```

---

## 🔹 2. INSERT, UPDATE, DELETE

```sql
INSERT INTO users (name, age) VALUES ('Alice', 25);
UPDATE users SET age = 26 WHERE name = 'Alice';
DELETE FROM users WHERE name = 'Alice';
```

---

## 🔹 3. JOINS

```sql
-- INNER JOIN
SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN
SELECT u.name, o.amount FROM users u LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN
SELECT u.name, o.amount FROM users u RIGHT JOIN orders o ON u.id = o.user_id;

-- FULL OUTER JOIN
SELECT u.name, o.amount FROM users u FULL OUTER JOIN orders o ON u.id = o.user_id;
```

---

## 🔹 4. GROUP BY & Aggregates

```sql
SELECT department, COUNT(*) AS total_employees, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 50000;
```

---

## 🔹 5. Subqueries

```sql
SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees);
SELECT name FROM users WHERE id IN (SELECT user_id FROM orders WHERE amount > 1000);
```

---

## 🔹 6. Common Table Expressions (CTE)

```
A CTE is a temporary result set that you can reference within a SELECT, INSERT, UPDATE, or DELETE statement.

You define it using the WITH keyword.
```

```sql
WITH high_earners AS (
  SELECT * FROM employees WHERE salary > 100000
)
SELECT name FROM high_earners;
```

---

## 🔹 7. Window Functions

```sql
SELECT name, department,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
```

---

## 🔹 8. Case Statement

```sql
SELECT name,
  CASE
    WHEN score >= 90 THEN 'A'
    WHEN score >= 75 THEN 'B'
    ELSE 'C'
  END AS grade
FROM students;
```

---

## 🔹 9. Indexes

```sql
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_lower_email ON users(LOWER(email));
CREATE INDEX idx_active_users ON users(id) WHERE is_active = true;
```

---

## 🔹 10. Constraints

```sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  age INT CHECK (age >= 18),
  country VARCHAR(100) DEFAULT 'USA'
);
```

---

## 🔹 11. Transactions

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

---

## 🔹 12. PostgreSQL Specific Features

### JSONB

```sql
INSERT INTO logs (event_data) VALUES ('{"type": "signup", "source": "mobile"}');
SELECT * FROM logs WHERE event_data->>'type' = 'signup';
SELECT * FROM logs WHERE event_data @> '{"type": "signup"}';
```

### UPSERT (ON CONFLICT)

```sql
INSERT INTO users (id, name)
VALUES (1, 'Alice')
ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name;
```

### RETURNING

```sql
INSERT INTO users (name, age)
VALUES ('Bob', 30)
RETURNING id;
```

---

## 🔹 13. Set Operations

```sql
SELECT city FROM customers
UNION
SELECT city FROM suppliers;

SELECT city FROM customers
UNION ALL
SELECT city FROM suppliers;
```

---

## 🔹 14. Materialized Views

```sql
CREATE MATERIALIZED VIEW top_customers AS
SELECT user_id, SUM(amount) AS total_spent
FROM orders
GROUP BY user_id;

REFRESH MATERIALIZED VIEW top_customers;
```

---

## 🔹 15. EXPLAIN & Performance Tuning

```sql
EXPLAIN ANALYZE
SELECT * FROM orders WHERE amount > 1000;
```

---

## 🔹 16. Normalization Summary

- 1NF: No repeating groups
- 2NF: No partial dependency on PK
- 3NF: No transitive dependency

---

## 🔹 17. Date/Time Functions

```sql
SELECT NOW();
SELECT CURRENT_DATE;
SELECT AGE(NOW(), birth_date);
```

---

## 🔹 18. Miscellaneous

```sql
SELECT DISTINCT country FROM users;
SELECT * FROM users ORDER BY created_at DESC LIMIT 10 OFFSET 20;
SELECT * FROM orders WHERE shipped_date IS NULL;
```
