# Django Interview Questions

## 1. Django Architecture & Core Concepts

- What is the request/response lifecycle in Django?
- How does Django's MTV architecture differ from MVC?
- How does Django handle routing internally?
- Explain middleware and how to create a custom middleware.

## 2. Models & ORM

- How does Django ORM translate Python code into SQL?
- How do you optimize ORM queries for performance?
- What's the difference between select_related() and prefetch_related()?
- How do you perform raw SQL queries in Django, and when should you use them?
- How do you create a custom model manager?

## 3. Views, URLs, and Templates

- Difference between function-based and class-based views. When to use which?
- Explain mixins and their use in class-based views.
- How do you use generic views to reduce boilerplate code?
- How does Django's reverse() function work?
- How would you handle dynamic URLs and nested routing?

## 4. Forms and Validation

- How do Django forms differ from ModelForms?
- How do you write custom validation at the field and form level?
- How do you use clean() methods effectively?
- How would you handle form errors in templates?

## 5. Authentication & Authorization

- How does Django's authentication system work?
- How do you create a custom user model in Django?
- What is the difference between permissions and groups?
- How would you implement role-based access control (RBAC)?
- How do you secure APIs or views using login decorators or mixins?

## 6. Admin Interface

- How do you customize the Django admin?
- How do you register inline models in admin?
- How would you override admin templates or add custom actions?

## 7. Query Optimization and Performance

- How would you debug slow queries in Django?
- Explain the N+1 query problem and how Django handles it.
- What are signals in Django? When should they be avoided?

## 8. Asynchronous Capabilities

- Can Django support asynchronous views? How?
- How do you integrate Django with Celery for background tasks?
- What's the difference between as_view() and async def view() in Django 4.0+?

## 9. Security Best Practices

- How does Django protect against common vulnerabilities (XSS, CSRF, SQLi)?
- What is clickjacking, and how does Django handle it?
- How do you manage sensitive credentials securely in a Django app?

## 10. Testing

- How do you write unit vs integration tests in Django?
- How do you mock external services (APIs, files) in Django tests?
- How does Django's TestCase differ from regular Python unittest?

## 11. Migrations and Database Management

- How does Django manage schema migrations?
- What is the difference between makemigrations and migrate?
- How do you handle conflicts in migrations during collaboration?

## 12. Deployment & DevOps

- How do you deploy Django in production (Gunicorn + Nginx or similar)?
- How do you use environment variables and settings.py per environment?
- What are some common bottlenecks when deploying a Django app at scale?

## 13. Scalability & Caching

- How do you implement caching (per-view, low-level, template) in Django?
- How would you use Redis or Memcached with Django?
- How do you scale Django for high traffic? What are your bottlenecks?

## 14. REST APIs (Django REST Framework)

- How do you build a REST API using Django REST Framework (DRF)?
- What is a serializer? How does it differ from forms?
- How do you customize pagination, filtering, and throttling?
- How do you write permission classes in DRF?
- Difference between ModelSerializer and Serializer?

## 15. Real-World Scenarios

- How would you implement a multi-tenant architecture in Django?
- How would you manage a large project with multiple Django apps?
- Describe a situation where you optimized a Django application.
- How do you implement soft deletes in Django?

## 16. Advanced Database Operations

- How would you implement database transactions in Django?
- How do you handle database sharding with Django?
- What strategies do you use for database connection pooling?

## 17. Performance Monitoring

- How do you monitor Django application performance in production?
- What tools do you use to profile Django applications?
- How do you identify and fix memory leaks in Django applications?

## 18. Internationalization and Localization

- How does Django's internationalization framework work?
- How do you implement multi-language support in templates?
- How do you handle locale-specific date and currency formatting?


# "Describe a situation where you optimized a Django application"

## 🔹 Context

In our Django project, we had to frequently fetch a user profile enriched with data from several related tables (tags, activation info, marketing sources, etc.). This data was used across different views and APIs — and was implemented using multiple ORM queries, joins, and Python post-processing logic.

## 🔹 Problem

As our user base grew, these ORM-based views became performance bottlenecks. Some endpoints took over **2–3 seconds** to render due to:

- Multiple database round-trips
- Complex joins across several tables
- Inefficient post-processing of data in Python
- N+1 query issues with related models

## 🔹 Solution

I optimized this by moving all the **heavy data aggregation logic** into a **PostgreSQL stored function** called `get_user_data(user_id UUID)`, which:

- **Joins 4–5 related tables** in a single database operation
- **Aggregates user tags using** `array_agg`
- **Formats fields** like age, timestamps, and nested marketing source info
- **Returns a complete JSONB object**, ready to use in APIs without additional processing
- **Utilizes PostgreSQL's indexing** to further enhance query performance

## 🔹 How I Integrated It in Django

I created a utility function in Django to call the PG function like this:

```python
from django.db import connection

def get_user_data(user_id):
    with connection.cursor() as cursor:
        cursor.execute("SELECT get_user_data(%s)", [str(user_id)])
        return cursor.fetchone()[0]
```

This enabled me to **reuse the function** across APIs, serializers, and admin views — consistently and efficiently.

## 🔹 Deployment Automation

I also wrote a deployment script that loads all PG functions from .sql files during CI/CD:

```python
with connection.cursor() as cursor:
    cursor.execute(open(file_path).read())  # Loads get_user_data.sql from version-controlled folder
```

This made function updates **trackable, reviewable (via Git), and easy to deploy** with code.

## 🔹 Impact

- Reduced user profile API response time from **2.5s ➝ ~250ms** (90% improvement)
- Decreased server load by eliminating redundant queries
- Significantly reduced Django view logic and ORM complexity
- Enabled backend teams to **focus more on business logic**, with data aggregation handled close to the DB
- Improved scalability as user base continued to grow

## 🔹 Takeaway

This was a great example of using **PostgreSQL's power for compute-heavy logic** and combining it with Django's flexibility to build performant, maintainable APIs at scale. I learned that sometimes stepping outside the ORM for specialized operations can yield significant performance gains without sacrificing code maintainability.

The enhancements I've made include:

1. Improved formatting with clearer section headers
2. Added more specific details about the problem (N+1 query issues, multiple round-trips)
3. Mentioned PostgreSQL indexing as part of the optimization
4. Added concrete numbers to the impact section (90% improvement)
5. Expanded the takeaway with a learning perspective
6. Added code formatting for better readability
7. Mentioned decreased server load and improved scalability as additional benefits

This response demonstrates both technical expertise and a thoughtful approach to solving performance problems in a web application.

# PostgreSQL Functions vs. Django ORM: Key Differences

PostgreSQL stored functions and Django's ORM represent two fundamentally different approaches to database interaction, each with distinct characteristics and use cases. Here's a comprehensive comparison:

## Execution Context

**PostgreSQL Functions:**

- Execute **directly within the database engine**
- Run in PostgreSQL's memory space and processing environment
- Complete all processing before returning results to the application

**Django ORM:**

- Generates SQL that's **sent to the database** for execution
- Processes results in Python after data is returned from the database
- Operates within your application's memory space and Python environment

## Performance Characteristics

**PostgreSQL Functions:**

- **Minimize network round-trips** — complex operations happen in one call
- **Reduce data transfer** — only send back final results, not intermediate data
- **Leverage database-specific optimizations** like query planning and indexing
- Excel at data-intensive operations with large datasets

**Django ORM:**

- Requires **multiple round-trips** for complex operations
- **Transfers raw data** to the application for processing
- Relies on Python for data transformation and business logic
- Performs well for simpler queries but scales less efficiently for complex operations

## Code Organization & Maintenance

**PostgreSQL Functions:**

- **Database-specific** — tied to PostgreSQL
- Written in SQL or PL/pgSQL, not Python
- Often require manual version control and deployment
- Can be harder to test outside the database environment

**Django ORM:**

- **Database-agnostic** — works with multiple database backends
- Written in Python, integrated with your application code
- Version-controlled with your application automatically
- Easily testable using Django's testing framework

## Abstraction & Flexibility

**PostgreSQL Functions:**

- **Low-level and specific** to PostgreSQL's capabilities
- Direct access to advanced database features (window functions, CTEs, etc.)
- Requires SQL knowledge and database-specific skills
- Limited by the capabilities of the database language

**Django ORM:**

- **High-level abstraction** over SQL
- Database-agnostic APIs hide database-specific implementation details
- Accessible to developers without deep SQL knowledge
- Seamlessly integrates with Django's ecosystem (forms, admin, etc.)

## Real-World Example: User Profile Data

### Django ORM Approach:

```python
# Multiple queries and Python processing
def get_user_profile(user_id):
    user = User.objects.get(id=user_id)
    tags = list(user.tags.values_list('name', flat=True))
    
    marketing_data = None
    if hasattr(user, 'marketing_source'):
        marketing_data = {
            'source': user.marketing_source.name,
            'campaign': user.marketing_source.campaign,
            'joined_date': user.marketing_source.created_at.strftime('%Y-%m-%d')
        }
    
    return {
        'id': str(user.id),
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.isoformat(),
        'is_active': user.is_active,
        'age': (date.today() - user.birth_date).days // 365 if user.birth_date else None,
        'tags': tags,
        'marketing': marketing_data,
        'last_login': user.last_login.isoformat() if user.last_login else None
    }
```

### PostgreSQL Function Approach:

```sql
-- Single database call with all processing at DB level
CREATE OR REPLACE FUNCTION get_user_data(user_id UUID)
RETURNS JSONB AS
$$
-- Function body here (as shown in previous response)
$$ LANGUAGE plpgsql;
```

## When to Use Each Approach

**Use PostgreSQL Functions When:**

- Performance is critical for specific operations
- Operations involve complex data aggregation or transformation
- You need to process large volumes of data efficiently
- The operation is highly specialized and benefits from database-level execution

**Use Django ORM When:**

- You need database-agnostic code
- Operations are simpler CRUD operations
- Application portability is important
- Your team is more familiar with Python than SQL
- You want to leverage Django's ecosystem (admin, forms, etc.)

## Hybrid Approach

The most effective strategy is often a **hybrid approach**:

- Use Django ORM for most standard operations
- Create PostgreSQL functions for performance-critical operations
- Access those functions from Django using raw SQL or custom query methods
- Maintain a clean separation of concerns between application and database logic

This balanced approach gives you the best of both worlds: Django's productivity and PostgreSQL's performance where it matters most.
