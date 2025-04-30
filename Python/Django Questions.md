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