
# ✅ MongoDB Cheat Sheet for Backend Developers

---

## 🔹 1. Basic Commands

```js
// Show all databases
show dbs;

// Switch or create a database
use myDatabase;

// Show all collections in the current database
show collections;
```

---

## 🔹 2. CRUD Operations

### 🟢 Create

```js
db.users.insertOne({ name: "Alice", age: 25 });
db.users.insertMany([
  { name: "Bob", age: 30 },
  { name: "Charlie", age: 35 }
]);
```

### 🔵 Read

```js
db.users.find();                        // Find all
db.users.find({ age: { $gt: 25 } });   // With filter
db.users.findOne({ name: "Alice" });   // Find one
```

### 🟡 Update

```js
db.users.updateOne(
  { name: "Alice" },
  { $set: { age: 26 } }
);

db.users.updateMany(
  { age: { $lt: 30 } },
  { $inc: { age: 1 } }
);
```

### 🔴 Delete

```js
db.users.deleteOne({ name: "Alice" });
db.users.deleteMany({ age: { $gt: 30 } });
```

---

## 🔹 3. Query Operators

```js
// Comparison
$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin

// Logical
$and, $or, $not, $nor

// Element
$exists, $type

// Evaluation
$regex, $expr
```

Example:
```js
db.users.find({ age: { $gte: 25, $lte: 35 } });
db.users.find({ name: { $regex: /^A/ } });
```

---

## 🔹 4. Array Queries

```js
db.courses.insertOne({ name: "Math", topics: ["algebra", "geometry"] });

db.courses.find({ topics: "algebra" });
db.courses.find({ topics: { $all: ["algebra", "geometry"] } });
db.courses.find({ topics: { $size: 2 } });
```

---

## 🔹 5. Projections (Fields)

```js
db.users.find({}, { name: 1, age: 1, _id: 0 });
```

---

## 🔹 6. Sorting, Limiting, Skipping

```js
db.users.find().sort({ age: -1 });
db.users.find().limit(5);
db.users.find().skip(10).limit(5);
```

---

## 🔹 7. Aggregation Framework

```js
db.orders.aggregate([
  { $match: { status: "shipped" } },
  { $group: { _id: "$customerId", total: { $sum: "$amount" } } },
  { $sort: { total: -1 } }
]);
```

---

## 🔹 8. Indexes

```js
db.users.createIndex({ email: 1 });
db.users.createIndex({ age: -1, name: 1 });
```

---

## 🔹 9. Transactions (Replica Set or Sharded Cluster Required)

```js
const session = db.getMongo().startSession();
session.startTransaction();
try {
  session.getDatabase("test").users.insertOne({ name: "A" });
  session.getDatabase("test").orders.insertOne({ orderId: 1 });
  session.commitTransaction();
} catch (e) {
  session.abortTransaction();
}
session.endSession();
```

---

## 🔹 10. Schema Design Tips

- Embed if data is frequently read together
- Reference if data grows large or is shared
- Denormalization is acceptable
- Use ObjectId for unique IDs

---

## 🔹 11. Other Useful Commands

```js
db.collection.drop();              // Drop collection
db.dropDatabase();                 // Drop database
db.collection.countDocuments();   // Count
db.collection.distinct("field");  // Distinct values
```
