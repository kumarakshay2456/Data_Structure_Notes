# Django Interview Questions & Answers (6+ Years Experience)

## 1. Django Architecture & Core Concepts

### What is the request/response lifecycle in Django?

The request/response lifecycle in Django follows a well-defined path from the moment a client sends a request until the server returns a response:

1. **HTTP Request**: A client (browser, API client, etc.) sends an HTTP request to the Django server.
    
2. **URL Routing**: Django's URL dispatcher (URLconf) analyzes the request URL and determines which view function/class should handle it.
    
3. **Middleware Processing (Request phase)**: Before reaching the view, the request passes through a series of middleware components in the order they're defined in `MIDDLEWARE` setting. Each middleware can modify the request or even return a response early, short-circuiting the process.
    
4. **View Processing**: The appropriate view function/class receives the request and processes it. This typically involves:
    
    - Extracting data from the request
    - Interacting with models/database
    - Processing business logic
    - Preparing context data for templates
5. **Template Rendering** (if applicable): The view often loads a template, populates it with context data, and renders it to HTML.
    
6. **Middleware Processing (Response phase)**: The response travels back through middleware components in reverse order, allowing them to modify the response.
    
7. **HTTP Response**: Django sends the final HTTP response back to the client.
    

**Real-world example**: Consider a product detail page on an e-commerce site:

```python
# urls.py
path('products/<int:product_id>/', views.product_detail, name='product_detail')

# views.py
@login_required  # Authentication middleware checks if user is logged in
def product_detail(request, product_id):
    # View fetches product from database
    product = get_object_or_404(Product, id=product_id)
    
    # Log this view for analytics (custom middleware might track this)
    request.session['last_viewed_product'] = product_id
    
    # Render template with context
    return render(request, 'products/detail.html', {'product': product})
```

In this flow, middleware might handle authentication, session management, and CSRF protection before the view processes the request, then compression and caching might be applied to the response.

### How does Django's MTV architecture differ from MVC?

Django follows an architectural pattern called MTV (Model-Template-View), which is conceptually similar to but differs in terminology from the traditional MVC (Model-View-Controller) pattern:

|MVC Component|Django MTV Equivalent|Responsibility|
|---|---|---|
|Model|Model|Data structure and database interactions|
|View|Template|Presentation and display logic|
|Controller|View|Business logic and request handling|

The key differences:

1. **Django's View** handles what a traditional Controller would do - processing requests, applying business logic, and determining what data to display.
    
2. **Django's Template** corresponds to the traditional View - it defines how the data should be presented.
    
3. **Django's Model** is largely the same as in MVC - it defines the data structure and handles database interactions.
    

The confusion often arises because Django's "View" is essentially a "Controller" in traditional MVC terminology. Django's creators chose this naming to better reflect their specific implementation of the pattern.

**Real-world example**:

```python
# Model - defines data structure and database interactions
class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    publication_date = models.DateTimeField(auto_now_add=True)
    
    def get_absolute_url(self):
        return reverse('article_detail', args=[self.id])

# View (Controller in MVC) - handles business logic
def article_detail(request, article_id):
    article = get_object_or_404(Article, id=article_id)
    return render(request, 'blog/article_detail.html', {'article': article})

# Template (View in MVC) - presentation logic
# article_detail.html
<article>
    <h1>{{ article.title }}</h1>
    <time>{{ article.publication_date|date:"F j, Y" }}</time>
    <div class="content">{{ article.content|linebreaks }}</div>
</article>
```

### How does Django handle routing internally?

Django's URL routing system is a crucial part of its request handling process. Here's how it works internally:

1. **URLconf Loading**: When Django starts, it loads the root URLconf module specified in the `ROOT_URLCONF` setting (typically `project_name.urls`).
    
2. **URL Pattern Compilation**: Django compiles all URL patterns into regular expressions when the server starts, for efficient matching later on.
    
3. **Request Processing**:
    
    - When a request comes in, Django removes the domain name and leading slash.
    - It tries to match the remaining URL path against each pattern in the URLconf in order.
    - The first matching pattern stops the search.
4. **View Resolution**:
    
    - Once a match is found, Django calls the associated view function with:
        - The `HttpRequest` object
        - Any captured values from the URL as positional or keyword arguments
        - Any additional arguments specified in the URL pattern
5. **Include Mechanism**: The `include()` function allows for modular URL configurations by including URL patterns from other URLconf modules.
    
6. **Namespace System**: Django provides a namespace system to disambiguate URL names across applications using the `app_name` variable and the `namespace` parameter to `include()`.
    

**Real-world example**:

```python
# Main urls.py
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', include('blog.urls', namespace='blog')),
    path('shop/', include('shop.urls', namespace='shop')),
]

# blog/urls.py
app_name = 'blog'
urlpatterns = [
    path('', views.article_list, name='article_list'),
    path('article/<int:article_id>/', views.article_detail, name='article_detail'),
    path('category/<slug:category_slug>/', views.category_detail, name='category_detail'),
]
```

When Django processes a request to `/blog/article/42/`:

1. It matches the `blog/` prefix and forwards the rest to `blog.urls`.
2. In `blog.urls`, it matches `article/<int:article_id>/` with `article_id=42`.
3. It calls `views.article_detail(request, article_id=42)`.

The URL name can be referenced as `blog:article_detail` in templates or code.

### Explain middleware and how to create a custom middleware.

Middleware in Django is a framework of hooks into Django's request/response processing. It's a lightweight, low-level "plugin" system for globally altering Django's input or output.

**Middleware Key Characteristics**:

- Executes during request/response cycle, not during Django initialization
- Processes all requests/responses that pass through the system
- Ordered by the `MIDDLEWARE` setting
- Request phase: processes from top to bottom
- Response phase: processes from bottom to top

**Creating Custom Middleware**:

Django supports two styles of middleware:

1. **Function-based middleware**
2. **Class-based middleware**

**Class-based Middleware Example**:

```python
class RequestTimeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time configuration and initialization

    def __call__(self, request):
        # Code to be executed for each request before the view (and later middleware) are called
        start_time = time.time()
        
        # Process the request
        response = self.get_response(request)
        
        # Code to be executed for each request/response after the view is called
        duration = time.time() - start_time
        response['X-Request-Duration'] = f"{duration:.2f}s"
        
        # Log slow requests
        if duration > 1.0:
            logger.warning(f"Slow request: {request.path} took {duration:.2f}s")
            
        return response
    
    # Optional method for view exception processing
    def process_exception(self, request, exception):
        # Log the error
        logger.error(f"Exception in {request.path}: {exception}")
        return None  # Let Django's exception handling take over
```

**Function-based Middleware Example**:

```python
def simple_middleware(get_response):
    # One-time configuration and initialization
    
    def middleware(request):
        # Code to be executed for each request before the view (and later middleware) are called
        
        response = get_response(request)
        
        # Code to be executed for each request/response after the view is called
        
        return response
    
    return middleware
```

**Real-world Example**: A middleware that tracks and limits API usage by IP address.

```python
class RateLimitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limits = {}
        self.window_size = 3600  # 1 hour window
        self.max_requests = 100  # 100 requests per hour

    def __call__(self, request):
        # Only apply rate limiting to API requests
        if request.path.startswith('/api/'):
            ip = self.get_client_ip(request)
            
            # Get or initialize the rate limit record for this IP
            if ip not in self.rate_limits:
                self.rate_limits[ip] = {'count': 0, 'reset_time': time.time() + self.window_size}
            
            # Check if the window has reset
            if time.time() > self.rate_limits[ip]['reset_time']:
                self.rate_limits[ip] = {'count': 0, 'reset_time': time.time() + self.window_size}
            
            # Increment request count
            self.rate_limits[ip]['count'] += 1
            
            # Check if limit exceeded
            if self.rate_limits[ip]['count'] > self.max_requests:
                return JsonResponse(
                    {'error': 'Rate limit exceeded. Try again later.'},
                    status=429
                )
            
            # Add rate limit headers
            response = self.get_response(request)
            response['X-Rate-Limit-Limit'] = str(self.max_requests)
            response['X-Rate-Limit-Remaining'] = str(self.max_requests - self.rate_limits[ip]['count'])
            response['X-Rate-Limit-Reset'] = str(int(self.rate_limits[ip]['reset_time']))
            return response
        
        return self.get_response(request)
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
```

To enable this middleware, add it to the `MIDDLEWARE` setting in your Django project:

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    # ...
    'path.to.RateLimitMiddleware',
    # ...
]
```

## 2. Models & ORM

### How does Django ORM translate Python code into SQL?

Django's Object-Relational Mapping (ORM) system translates Python code into SQL through a complex but elegant process:

1. **Model Definition**: When you define a Django model, you're creating a Python class that inherits from `django.db.models.Model` with attributes that represent database fields.
    
2. **Query Construction**: When you write a query using the ORM, Django constructs a `QuerySet` object. This object is lazy – it doesn't execute the query immediately.
    
3. **Query Compilation**: When the `QuerySet` is evaluated (e.g., when you iterate over it, call `list()` on it, or slice it), Django's query compiler converts it to SQL:
    
    - Django determines the required tables and joins
    - It analyzes the conditions (filters) and converts them to WHERE clauses
    - It processes annotations, aggregations, order_by statements, etc.
4. **SQL Generation**: The compiled query is converted to SQL specific to your database backend using Django's database-specific operations.
    
5. **Query Execution**: The generated SQL is sent to the database for execution.
    
6. **Result Processing**: Database results are converted back into model instances.
    

**Real-world Example**:

```python
# Python model definition
class Customer(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class Order(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='orders')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('shipped', 'Shipped'),
        ('delivered', 'Delivered'),
    ])

# Python query
recent_paid_orders = Order.objects.filter(
    status='paid',
    created_at__gte=datetime.datetime.now() - datetime.timedelta(days=30)
).select_related('customer').order_by('-created_at')
```

When executed, this Python code gets translated to SQL similar to:

```sql
SELECT 
    "orders"."id", "orders"."customer_id", "orders"."total_amount", 
    "orders"."created_at", "orders"."status",
    "customer"."id", "customer"."name", "customer"."email", "customer"."created_at" 
FROM "orders" 
INNER JOIN "customer" ON ("orders"."customer_id" = "customer"."id") 
WHERE ("orders"."status" = 'paid' AND "orders"."created_at" >= '2025-03-31 12:30:45.123456') 
ORDER BY "orders"."created_at" DESC;
```

You can see the actual SQL generated by Django by using:

```python
print(recent_paid_orders.query)
```

### How do you optimize ORM queries for performance?

Optimizing Django ORM queries is crucial for application performance. Here are detailed techniques with examples:

#### 1. Use `select_related()` and `prefetch_related()` to avoid N+1 queries

```python
# Bad: Causes N+1 queries
orders = Order.objects.all()
for order in orders:
    print(order.customer.name)  # Each access triggers a new query

# Good: Just 1 query with a JOIN
orders = Order.objects.select_related('customer')
for order in orders:
    print(order.customer.name)  # No additional query
```

#### 2. Only select the fields you need

```python
# Fetches all fields
users = User.objects.all()

# More efficient: fetches only needed fields
users = User.objects.only('username', 'email', 'last_login')

# Alternative approach
users = User.objects.values('username', 'email', 'last_login')
```

#### 3. Use `values()` or `values_list()` when you don't need model instances

```python
# Returns model instances
products = Product.objects.filter(category='electronics')

# Returns dictionaries - more efficient when you just need data
product_data = Product.objects.filter(category='electronics').values('name', 'price')

# Returns tuples - even more efficient
product_tuples = Product.objects.filter(category='electronics').values_list('name', 'price')

# For a single field, you can flatten the result
product_names = Product.objects.filter(category='electronics').values_list('name', flat=True)
```

#### 4. Use database functions for computation

```python
from django.db.models import F, Sum, Count, Avg
from django.db.models.functions import Coalesce

# Calculate in database instead of Python
Order.objects.update(
    total_price=F('price') * F('quantity') * (1 - F('discount_rate'))
)

# Aggregate in database
report = Order.objects.values('customer_id').annotate(
    order_count=Count('id'),
    total_spent=Sum('total_price'),
    avg_order_value=Avg('total_price')
)
```

#### 5. Use `iterator()` for large querysets

```python
# Memory-efficient processing of large querysets
for product in Product.objects.filter(active=True).iterator():
    # Process each product without loading all into memory
    process_product(product)
```

#### 6. Use indexed fields in filters

```python
# Add indexes to fields used frequently in filtering, ordering or joining
class Customer(models.Model):
    email = models.EmailField(db_index=True)
    last_active = models.DateTimeField(db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['country', 'city']),  # Compound index
            models.Index(fields=['membership_type', '-join_date']),
        ]
```

#### 7. Use `exists()` instead of `count()` or `len()` to check existence

```python
# Inefficient
if User.objects.filter(email=email).count() > 0:
    # User exists

# More efficient
if User.objects.filter(email=email).exists():
    # User exists
```

#### 8. Use `bulk_create()` and `bulk_update()` for batch operations

```python
# Inefficient: N queries
for data in dataset:
    Product.objects.create(name=data['name'], price=data['price'])

# Efficient: 1 query
products = [
    Product(name=data['name'], price=data['price']) 
    for data in dataset
]
Product.objects.bulk_create(products)

# Similarly for updates
Product.objects.bulk_update(products, ['price', 'stock'])
```

#### 9. Consider raw SQL for very complex queries

```python
from django.db import connection

def complex_analytics_query():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                p.category, 
                SUM(oi.quantity * oi.price) as revenue,
                COUNT(DISTINCT o.customer_id) as customer_count
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.created_at > %s
            GROUP BY p.category
            HAVING SUM(oi.quantity) > 100
            ORDER BY revenue DESC
        """, [three_months_ago])
        return cursor.fetchall()
```

#### 10. Use query caching mechanisms

```python
from django.core.cache import cache

def get_active_promotions():
    cache_key = 'active_promotions'
    promotions = cache.get(cache_key)
    
    if promotions is None:
        promotions = list(Promotion.objects.filter(
            is_active=True,
            start_date__lte=timezone.now(),
            end_date__gte=timezone.now()
        ).select_related('product'))
        
        # Cache for 10 minutes
        cache.set(cache_key, promotions, 60 * 10)
    
    return promotions
```

#### 11. Use `defer()` to exclude unnecessary large fields

```python
# Skip loading large text fields when not needed
articles = Article.objects.defer('content', 'metadata').all()
```

#### 12. Use `QuerySet.explain()` to analyze query execution plans (Django 3.0+)

```python
queryset = Order.objects.filter(
    status='processing',
    created_at__gt=last_month
).select_related('customer')

# Print the execution plan
print(queryset.explain())
```

**Real-world Optimization Example**: An e-commerce dashboard that displays sales stats without bogging down the database:

```python
def get_sales_dashboard_data(start_date, end_date):
    # Cache key includes the date range
    cache_key = f'sales_dashboard:{start_date.isoformat()}:{end_date.isoformat()}'
    dashboard_data = cache.get(cache_key)
    
    if dashboard_data is None:
        # Get completed orders in date range
        orders = Order.objects.filter(
            status='completed',
            completed_at__range=(start_date, end_date)
        )
        
        # Calculate revenue and stats in the database
        order_stats = orders.aggregate(
            total_revenue=Sum('total_amount'),
            order_count=Count('id'),
            avg_order_value=Avg('total_amount')
        )
        
        # Get top products efficiently
        top_products = OrderItem.objects.filter(
            order__in=orders
        ).values(
            'product_id', 'product__name'
        ).annotate(
            total_sold=Sum('quantity'),
            revenue=Sum(F('price') * F('quantity'))
        ).order_by('-revenue')[:10]
        
        # Get daily revenue for chart
        daily_revenue = orders.annotate(
            date=TruncDate('completed_at')
        ).values('date').annotate(
            revenue=Sum('total_amount')
        ).order_by('date')
        
        dashboard_data = {
            'order_stats': order_stats,
            'top_products': list(top_products),
            'daily_revenue': list(daily_revenue)
        }
        
        # Cache for 1 hour
        cache.set(cache_key, dashboard_data, 60 * 60)
    
    return dashboard_data
```

### What's the difference between select_related() and prefetch_related()?

Both `select_related()` and `prefetch_related()` are Django ORM methods to optimize database queries by reducing the number of database hits, but they work differently and are suitable for different relationship types:

#### `select_related()`

- **Usage**: For foreign key and one-to-one relationships (where the related object is on the "one" side)
- **Mechanism**: Performs a SQL JOIN and includes the fields of the related object in the SELECT statement
- **Query Count**: Uses a single database query
- **Performance Impact**: Best for "to-one" relationships where you need data from both the model and its related model

**Example**:

```python
# Without select_related - 2 queries
order = Order.objects.get(id=1)  # First query
customer = order.customer  # Second query to fetch the customer

# With select_related - 1 query with JOIN
order = Order.objects.select_related('customer').get(id=1)
customer = order.customer  # No additional query - data already loaded
```

Generated SQL (simplified):

```sql
SELECT 
    orders.id, orders.date, orders.total, /* other order fields */
    customers.id, customers.name, /* other customer fields */
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id
WHERE orders.id = 1;
```

#### `prefetch_related()`

- **Usage**: For many-to-many relationships and reverse foreign key relationships (where the related objects are on the "many" side)
- **Mechanism**: Performs separate queries for each relationship and joins the results in Python
- **Query Count**: Uses multiple queries (one for the main model, one for each prefetched relationship)
- **Performance Impact**: Best for "to-many" relationships where you need to access multiple related objects

**Example**:

```python
# Without prefetch_related - N+1 queries
product = Product.objects.get(id=1)  # First query
categories = product.categories.all()  # Second query for categories
for category in categories:
    print(category.name)

# With prefetch_related - 2 queries
product = Product.objects.prefetch_related('categories').get(id=1)
categories = product.categories.all()  # No additional query
for category in categories:
    print(category.name)  # No additional queries
```

Generated SQL (simplified):

```sql
-- First query fetches the product
SELECT id, name, /* other product fields */ FROM products WHERE id = 1;

-- Second query fetches all related categories
SELECT c.id, c.name, /* other category fields */, pc.product_id 
FROM categories c
INNER JOIN product_categories pc ON c.id = pc.category_id
WHERE pc.product_id IN (1);
```

#### Complex Relationships and Chaining

Both methods can be chained and combined:

```python
# Combining both techniques
orders = Order.objects.select_related('customer').prefetch_related('items__product')

# This efficiently loads:
# 1. Orders
# 2. The customer for each order (via JOIN)
# 3. The items for each order (via separate query)
# 4. The product for each item (via separate query)
```

#### Nested Relationships

Both can traverse multi-level relationships:

```python
# Select related can traverse foreign keys
Order.objects.select_related('customer__address__country')

# Prefetch related can traverse any relationship
Product.objects.prefetch_related(
    'categories',
    'reviews__user',
    Prefetch('variants', queryset=Variant.objects.filter(in_stock=True))
)
```

#### Real-world Example: An e-commerce order detail view

```python
def order_detail(request, order_id):
    # Efficiently fetch the order with all related data in minimal queries
    order = Order.objects.select_related(
        'customer',  # Foreign key - uses JOIN
        'shipping_address',  # Foreign key - uses JOIN
        'billing_address'  # Foreign key - uses JOIN
    ).prefetch_related(
        'items__product',  # Reverse FK + FK - separate queries
        'items__product__categories',  # M2M after FK chain - separate query
        'payment_transactions',  # Reverse FK - separate query
        Prefetch(
            'status_updates',  # Custom prefetch for filtered relationship
            queryset=OrderStatusUpdate.objects.select_related('user').order_by('-timestamp'),
            to_attr='history'
        )
    ).get(id=order_id)
    
    # Now we can access all these related objects without additional queries
    context = {
        'order': order,
        'customer': order.customer,
        'address': order.shipping_address,
        'items': order.items.all(),  # No query
        'payment_history': order.payment_transactions.all(),  # No query
        'status_history': order.history  # From prefetch to_attr
    }
    
    return render(request, 'orders/detail.html', context)
```

### How do you perform raw SQL queries in Django, and when should you use them?

Django provides several ways to execute raw SQL queries when the ORM doesn't provide the flexibility or performance you need:

#### 1. Using `Manager.raw()` method

The `raw()` method executes a raw SQL query and returns a `RawQuerySet` of model instances:

```python
# Simple raw query
products = Product.objects.raw('SELECT * FROM products WHERE price > %s', [100])

# More complex raw query with joins
customers = Customer.objects.raw('''
    SELECT c.id, c.name, c.email, COUNT(o.id) as order_count
    FROM customers c
    LEFT JOIN orders o ON c.id = o.customer_id
    WHERE c.is_active = True
    GROUP BY c.id
    HAVING COUNT(o.id) > 5
    ORDER BY order_count DESC
''')

# Accessing results
for customer in customers:
    print(customer.name, customer.order_count)  # Note: order_count is dynamically added
```

**Important considerations**:

- You must include the primary key column in your query
- Django maps the query results to model instances
- You can map extra SELECT fields to model attributes
- Parameters should be passed as a list to prevent SQL injection

#### 2. Using `connection.cursor()` for complete control

For queries that don't map to models or for non-SELECT operations:

```python
from django.db import connection

def get_product_sales_report():
    with connection.cursor() as cursor:
        cursor.execute('''
            SELECT 
                p.name, 
                p.sku, 
                SUM(oi.quantity) as units_sold,
                SUM(oi.quantity * oi.price) as revenue
            FROM products p
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.status = 'completed'
            AND o.completion_date > %s
            GROUP BY p.id, p.name, p.sku
            ORDER BY revenue DESC
        ''', [three_months_ago])
        
        # Convert results to dictionaries
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
```

#### 3. Using `connection.execute()` method (Django 4.0+)

```python
from django.db import connection

def update_product_prices(category_id, increase_percentage):
    with connection.execute('''
        UPDATE products
        SET price = price * (1 + %s/100)
        WHERE category_id = %s
    ''', [increase_percentage, category_id]) as cursor:
        return cursor.rowcount  # Number of rows affected
```

#### 4. Using database-specific operations with `QuerySet.annotate()`

Django 3.2+ allows using database functions directly:

```python
from django.db.models import F, Value
from django.db.models.functions import Cast
from django.db.models.expressions import RawSQL

# Using RawSQL within a queryset
Product.objects.annotate(
    distance=RawSQL(
        "ST_Distance(location, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
        (longitude, latitude)
    )
).order_by('distance')
```

#### When to use raw SQL:

1. **Complex queries beyond ORM capabilities**:
    
    - Advanced window functions
    - Complex subqueries
    - Hierarchical/recursive queries (CTE)
    - Advanced geospatial queries
2. **Performance optimization**:
    
    - When ORM-generated queries are inefficient
    - For queries manipulating large datasets
    - When you need database-specific optimizations
3. **Bulk operations**:
    
    - Mass updates with complex conditions
    - Specialized batch processing
4. **Database-specific features**:
    
    - Using features specific to your database like PostgreSQL's JSONB operations
5. **Schema migration operations**:
    
    - Custom, complex schema changes

**Real-world Example**: A geospatial search with complex filtering:

```python
def find_nearby_restaurants(latitude, longitude, radius_km, cuisine=None, min_rating=None):
    query = '''
        SELECT 
            r.id, r.name, r.address, r.rating,
            ST_Distance(
                r.location, 
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            ) * 111.32 AS distance_km
        FROM restaurants r
        LEFT JOIN restaurant_cuisines rc ON r.id = rc.restaurant_id
        LEFT JOIN cuisines c ON rc.cuisine_id = c.id
        WHERE ST_DWithin(
            r.location, 
            ST_SetSRID(ST_MakePoint(%s, %s), 4326),
            %s / 111.32
        )
    '''
    
    params = [longitude, latitude, longitude, latitude, radius_km]
    
    if cuisine:
        query += " AND c.name = %s"
        params.append(cuisine)
    
    if min_rating:
        query += " AND r.rating >= %s"
        params.append(min_rating)
    
    query += " GROUP BY r.id, r.name, r.address, r.rating, r.location ORDER BY distance_km"
    
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
```
def article_detail(request, pk): article = get_object_or_404(Article, pk=pk) return render(request, 'blog/article_detail.html', {'article': article})

````

With DetailView:
```python
class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    context_object_name = 'article'
````

**3. CreateView - Create a new object**

Before generic views:

```python
def create_article(request):
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save(commit=False)
            article.author = request.user
            article.save()
            return redirect('article_detail', pk=article.pk)
    else:
        form = ArticleForm()
    return render(request, 'blog/article_form.html', {'form': form})
```

With CreateView:

```python
class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
    
    def get_success_url(self):
        return reverse('article_detail', kwargs={'pk': self.object.pk})
```

**4. UpdateView - Update an existing object**

Before generic views:

```python
def update_article(request, pk):
    article = get_object_or_404(Article, pk=pk)
    if request.method == 'POST':
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            form.save()
            return redirect('article_detail', pk=article.pk)
    else:
        form = ArticleForm(instance=article)
    return render(request, 'blog/article_form.html', {'form': form})
```

With UpdateView:

```python
class ArticleUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user
    
    def get_success_url(self):
        return reverse('article_detail', kwargs={'pk': self.object.pk})
```

**5. DeleteView - Delete an existing object**

Before generic views:

```python
def delete_article(request, pk):
    article = get_object_or_404(Article, pk=pk)
    if request.method == 'POST':
        article.delete()
        return redirect('article_list')
    return render(request, 'blog/article_confirm_delete.html', {'article': article})
```

With DeleteView:

```python
class ArticleDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Article
    template_name = 'blog/article_confirm_delete.html'
    success_url = reverse_lazy('article_list')
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user
```

#### Customizing Generic Views

While generic views handle many common cases out of the box, they are designed to be customizable:

**1. Customizing Query Sets**

```python
class PublishedArticleListView(ListView):
    model = Article
    template_name = 'blog/article_list.html'
    context_object_name = 'articles'
    
    def get_queryset(self):
        return Article.objects.filter(status='published').order_by('-published_date')
```

**2. Adding Extra Context Data**

```python
class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comments'] = self.object.comments.all()
        context['related_articles'] = Article.objects.filter(
            category=self.object.category
        ).exclude(id=self.object.id)[:5]
        return context
```

**3. Custom Form Processing**

```python
class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    
    def form_valid(self, form):
        # Customize before saving
        form.instance.author = self.request.user
        form.instance.status = 'draft'
        
        # Save the object
        response = super().form_valid(form)
        
        # Customize after saving
        if form.cleaned_data.get('notify_subscribers'):
            self.object.notify_subscribers()
            
        return response
```

**4. Custom URL Parameters**

```python
class CategoryArticleListView(ListView):
    model = Article
    template_name = 'blog/category_articles.html'
    
    def get_queryset(self):
        self.category = get_object_or_404(Category, slug=self.kwargs['slug'])
        return Article.objects.filter(category=self.category)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['category'] = self.category
        return context
```

**Real-world Example**: A complete blog application using generic views:

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('category_detail', kwargs={'slug': self.slug})

class Article(models.Model):
    STATUS_CHOICES = (
        ('draft', 'Draft'),
        ('published', 'Published'),
    )
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='articles')
    content = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='articles')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_date = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('article_detail', kwargs={'slug': self.slug})

class Comment(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Comment by {self.author.username} on {self.article.title}"

# views.py
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.views.generic.dates import YearArchiveView, MonthArchiveView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse_lazy
from .models import Article, Category, Comment
from .forms import ArticleForm, CommentForm

class ArticleListView(ListView):
    model = Article
    template_name = 'blog/article_list.html'
    context_object_name = 'articles'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Article.objects.filter(status='published').order_by('-published_date')
        
        # Filter by category if provided
        category_slug = self.request.GET.get('category')
        if category_slug:
            queryset = queryset.filter(category__slug=category_slug)
            
        # Filter by search query if provided
        search_query = self.request.GET.get('q')
        if search_query:
            queryset = queryset.filter(
                Q(title__icontains=search_query) | 
                Q(content__icontains=search_query)
            )
            
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        context['search_query'] = self.request.GET.get('q', '')
        context['category_filter'] = self.request.GET.get('category', '')
        return context

class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    context_object_name = 'article'
    slug_url_kwarg = 'slug'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comment_form'] = CommentForm()
        context['comments'] = self.object.comments.all().order_by('-created_at')
        context['related_articles'] = Article.objects.filter(
            category=self.object.category, 
            status='published'
        ).exclude(id=self.object.id)[:3]
        return context
    
    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form = CommentForm(request.POST)
        
        if form.is_valid() and request.user.is_authenticated:
            comment = form.save(commit=False)
            comment.article = self.object
            comment.author = request.user
            comment.save()
            return redirect(self.object.get_absolute_url())
            
        context = self.get_context_data(object=self.object)
        context['comment_form'] = form
        return self.render_to_response(context)

class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

class ArticleUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    slug_url_kwarg = 'slug'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user or self.request.user.is_staff

class ArticleDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Article
    template_name = 'blog/article_confirm_delete.html'
    success_url = reverse_lazy('article_list')
    slug_url_kwarg = 'slug'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user or self.request.user.is_staff

class CategoryDetailView(DetailView):
    model = Category
    template_name = 'blog/category_detail.html'
    context_object_name = 'category'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['articles'] = self.object.articles.filter(
            status='published'
        ).order_by('-published_date')
        return context

class ArticleYearArchiveView(YearArchiveView):
    model = Article
    date_field = 'published_date'
    make_object_list = True
    template_name = 'blog/article_archive_year.html'
    
    def get_queryset(self):
        return Article.objects.filter(status='published')

class ArticleMonthArchiveView(MonthArchiveView):
    model = Article
    date_field = 'published_date'
    template_name = 'blog/article_archive_month.html'
    
    def get_queryset(self):
        return Article.objects.filter(status='published')

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ArticleListView.as_view(), name='article_list'),
    path('article/<slug:slug>/', views.ArticleDetailView.as_view(), name='article_detail'),
    path('article/create/', views.ArticleCreateView.as_view(), name='article_create'),
    path('article/<slug:slug>/update/', views.ArticleUpdateView.as_view(), name='article_update'),
    path('article/<slug:slug>/delete/', views.ArticleDeleteView.as_view(), name='article_delete'),
    path('category/<slug:slug>/', views.CategoryDetailView.as_view(), name='category_detail'),
    path('archive/<int:year>/', views.ArticleYearArchiveView.as_view(), name='article_year_archive'),
    path('archive/<int:year>/<int:month>/', views.ArticleMonthArchiveView.as_view(), name='article_month_archive'),
]
```

This example demonstrates how generic views:

- Eliminate repetitive code for common operations
- Provide a consistent structure across views
- Allow for customization where needed
- Enable quick addition of features like pagination and filtering
- Support complex functionality through mixins and inheritance

Generic views are most beneficial when you're implementing standard CRUD operations and want to maintain consistent behaviors, but they remain flexible enough to handle custom business logic when needed.

### How does Django's reverse() function work?

Django's `reverse()` function is a powerful URL resolution tool that dynamically generates URLs from URL patterns defined in your URLconf. This is crucial for maintaining DRY (Don't Repeat Yourself) principles by avoiding hardcoded URLs throughout your codebase.

#### Basic Functionality

At its core, `reverse()` takes a URL pattern name and optional arguments, then returns the corresponding URL path:

```python
from django.urls import reverse

# Basic usage
article_url = reverse('article_detail', args=[42])  # Returns '/articles/42/'
```

#### How It Works Internally

1. **Pattern Lookup**: Django searches all URL patterns across all included URLconfs for a pattern with the given name.
    
2. **Pattern Matching**: Once found, Django uses the pattern's regular expression to construct a URL.
    
3. **Argument Substitution**:
    
    - Positional arguments from `args` are inserted into the pattern in order
    - Named arguments from `kwargs` are matched to named groups in the pattern
4. **URL Construction**: The final URL string is assembled, including the prefix from any parent URLconfs.
    

#### Usage Patterns

**1. Basic URL Reversal**

```python
# URL definition
path('articles/<int:article_id>/', views.article_detail, name='article_detail')

# In a view
def some_view(request):
    url = reverse('article_detail', args=[42])  # '/articles/42/'
    return redirect(url)
```

**2. Using Named Arguments**

```python
# URL definition
path('articles/<int:year>/<int:month>/<slug:slug>/', views.article_detail, name='article_detail')

# In a view
def some_view(request):
    url = reverse('article_detail', kwargs={
        'year': 2025,
        'month': 4,
        'slug': 'django-reverse-explained'
    })  # '/articles/2025/4/django-reverse-explained/'
    return redirect(url)
```

**3. URL Namespaces**

Django supports URL namespaces to avoid name clashes between apps:

```python
# In main urls.py
path('blog/', include('blog.urls', namespace='blog'))
path('news/', include('news.urls', namespace='news'))

# In a view
def some_view(request):
    blog_url = reverse('blog:article_detail', args=[42])  # '/blog/articles/42/'
    news_url = reverse('news:article_detail', args=[42])  # '/news/articles/42/'
    return render(request, 'template.html', {'blog_url': blog_url, 'news_url': news_url})
```

**4. Reversing in Templates**

In Django templates, you can use the `url` template tag:

```html
<a href="{% url 'article_detail' article.id %}">{{ article.title }}</a>

<!-- With namespace -->
<a href="{% url 'blog:article_detail' article.id %}">{{ article.title }}</a>
```

**5. Reversing in Models**

The `reverse()` function is commonly used in models to provide absolute URLs:

```python
from django.db import models
from django.urls import reverse

class Article(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    content = models.TextField()
    
    def get_absolute_url(self):
        return reverse('article_detail', kwargs={'slug': self.slug})
```

**6. Handling Current App Namespaces**

When working with app namespaces and the current app might vary:

```python
from django.urls import reverse, resolve
from django.urls.exceptions import NoReverseMatch

def get_url_in_current_app(request, view_name, *args, **kwargs):
    current_namespace = resolve(request.path).namespace
    try:
        # Try with current namespace
        return reverse(f"{current_namespace}:{view_name}", args=args, kwargs=kwargs)
    except NoReverseMatch:
        # Fall back to no namespace
        return reverse(view_name, args=args, kwargs=kwargs)
```

#### Advanced Usage

**1. Using `reverse_lazy()`**

For cases where you need a URL reference at import time (before URLs are loaded):

```python
from django.urls import reverse_lazy

class ArticleDeleteView(DeleteView):
    model = Article
    # URLs not loaded when class is defined, so we use reverse_lazy
    success_url = reverse_lazy('article_list')
```

**2. Handling Optional URL Parameters**

For URLs with optional parameters, you often need conditional logic:

```python
def get_filtered_list_url(category=None, tag=None):
    if category and tag:
        return reverse('article_list') + f'?category={category}&tag={tag}'
    elif category:
        return reverse('article_list') + f'?category={category}'
    elif tag:
        return reverse('article_list') + f'?tag={tag}'
    return reverse('article_list')
```

**3. Building APIs with Reverse**

For building API links:

```python
from django.urls import reverse
from rest_framework.response import Response

class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def list(self, request):
        articles = self.get_queryset()
        data = [{
            'id': article.id,
            'title': article.title,
            'url': request.build_absolute_uri(reverse('api:article-detail', args=[article.id]))
        } for article in articles]
        return Response(data)
```

**Real-world Example**: An e-commerce platform with complex URL structure:

```python
# urls.py in the main project
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('shop/', include('shop.urls', namespace='shop')),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('blog/', include('blog.urls', namespace='blog')),
    path('api/', include('api.urls', namespace='api')),
]

# shop/urls.py
app_name = 'shop'

urlpatterns = [
    path('', views.ProductListView.as_view(), name='product_list'),
    path('category/<slug:category_slug>/', views.CategoryDetailView.as_view(), name='category_detail'),
    path('product/<slug:slug>/', views.ProductDetailView.as_view(), name='product_detail'),
    path('cart/', views.CartView.as_view(), name='cart'),
    path('checkout/', views.CheckoutView.as_view(), name='checkout'),
    path('orders/', views.OrderListView.as_view(), name='order_list'),
    path('orders/<uuid:order_id>/', views.OrderDetailView.as_view(), name='order_detail'),
]

# shop/models.py
class Product(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    category = models.ForeignKey('Category', on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    
    def get_absolute_url(self):
        return reverse('shop:product_detail', kwargs={'slug': self.slug})
    
    def get_add_to_cart_url(self):
        return reverse('shop:add_to_cart', kwargs={'product_id': self.id})
    
    def get_related_products_url(self):
        return reverse('api:related-products', kwargs={'product_id': self.id})

# shop/views.py
def add_to_cart(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    cart = Cart(request)
    cart.add(product)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'success': True,
            'cart_count': cart.get_item_count(),
            'cart_url': reverse('shop:cart')
        })
    
    # Determine where to redirect based on the referrer
    next_url = request.GET.get('next')
    if next_url:
        return redirect(next_url)
        
    return redirect(product.get_absolute_url())

class OrderCreateView(LoginRequiredMixin, CreateView):
    model = Order
    form_class = OrderForm
    template_name = 'shop/checkout.html'
    
    def form_valid(self, form):
        form.instance.user = self.request.user
        response = super().form_valid(form)
        
        # Clear the cart after successful order
        cart = Cart(self.request)
        cart.clear()
        
        # Send confirmation email
        send_order_confirmation.delay(self.object.id)
        
        return response
    
    def get_success_url(self):
        # Use reverse with kwargs to build the success URL
        return reverse('shop:order_confirmation', kwargs={'order_id': self.object.id})
```

In this example:

1. `reverse()` helps generate URLs for different parts of the e-commerce application
2. URL namespaces (`shop:`, `accounts:`, etc.) keep URL names organized
3. Models use `get_absolute_url()` for canonical URLs
4. Views use `reverse()` for redirects after form submissions
5. AJAX responses include URLs generated by `reverse()`

The `reverse()` function helps maintain a clean separation between URL structure and application logic, allowing URLs to be changed without breaking functionality throughout the application.

### How would you handle dynamic URLs and nested routing?

Dynamic URLs and nested routing are essential for creating flexible, hierarchical URL structures in Django applications. Properly handling these patterns allows for more intuitive URLs and better organization of your web application.

#### Dynamic URLs

Django's URL patterns support capturing values from the URL path using path converters, which extract and validate URL segments.

**Common Path Converters:**

- `str`: Matches any non-empty string, excluding the path separator ('/')
- `int`: Matches zero or any positive integer
- `slug`: Matches slug strings (ASCII letters, numbers, hyphens, underscores)
- `uuid`: Matches a UUID string
- `path`: Matches any non-empty string, including the path separator

**Basic Dynamic URL Examples:**

```python
from django.urls import path
from . import views

urlpatterns = [
    # Integer parameter - matches /articles/42/
    path('articles/<int:article_id>/', views.article_detail, name='article_detail'),
    
    # Slug parameter - matches /articles/introduction-to-django/
    path('articles/<slug:slug>/', views.article_detail_by_slug, name='article_detail_by_slug'),
    
    # UUID parameter - matches /orders/123e4567-e89b-12d3-a456-426614174000/
    path('orders/<uuid:order_id>/', views.order_detail, name='order_detail'),
    
    # Multiple parameters - matches /articles/2025/04/django-routing/
    path('articles/<int:year>/<int:month>/<slug:slug>/', 
         views.article_archive_detail, name='article_archive_detail'),
]
```

**Multiple Parameters in Views:**

```python
def article_archive_detail(request, year, month, slug):
    # The parameters from the URL are passed to the view
    article = get_object_or_404(Article, 
                               publish_date__year=year,
                               publish_date__month=month, 
                               slug=slug)
    return render(request, 'blog/article_detail.html', {'article': article})
```

#### Custom Path Converters

For specialized URL patterns, you can create custom path converters:

```python
from django.urls import path, register_converter

class FourDigitYearConverter:
    regex = '[0-9]{4}'
    
    def to_python(self, value):
        return int(value)
    
    def to_url(self, value):
        return f'{value:04d}'

# Register the converter
register_converter(FourDigitYearConverter, 'year4')

# Use the custom converter
urlpatterns = [
    path('articles/<year4:year>/', views.year_archive, name='year_archive'),
]
```

#### Nested Routing

Nested routing refers to organizing URL patterns hierarchically, which is particularly useful for complex applications with multiple related sections.

**Basic Nested Routing with `include()`:**

```python
from django.urls import path, include

urlpatterns = [
    path('blog/', include('blog.urls')),
    path('shop/', include('shop.urls')),
    path('accounts/', include('accounts.urls')),
]
```

**Nested Routing with Namespaces:**

Namespaces help avoid name clashes in URL patterns across different apps:

```python
# Main urls.py
from django.urls import path, include

urlpatterns = [
    path('blog/', include('blog.urls', namespace='blog')),
    path('shop/', include('shop.urls', namespace='shop')),
]

# blog/urls.py
app_name = 'blog'  # Sets the application namespace

urlpatterns = [
    path('', views.article_list, name='article_list'),
    path('article/<slug:slug>/', views.article_detail, name='article_detail'),
    path('category/<slug:category_slug>/', views.category_detail, name='category_detail'),
]
```

These URLs would be reversed as:

```python
reverse('blog:article_detail', kwargs={'slug': 'django-routing'})  # /blog/article/django-routing/
reverse('shop:product_detail', kwargs={'slug': 'django-mug'})  # /shop/product/django-mug/
```

**Multi-level Nested Routing:**

You can include URL patterns at multiple levels:

```python
# Main urls.py
urlpatterns = [
    path('api/', include('api.urls', namespace='api')),
]

# api/urls.py
app_name = 'api'

urlpatterns = [
    path('v1/', include('api.v1.urls', namespace='v1')),
    path('v2/', include('api.v2.urls', namespace='v2')),
]

# api/v1/urls.py
app_name = 'v1'

urlpatterns = [
    path('users/', include('api.v1.users.urls', namespace='users')),
    path('products/', include('api.v1.products.urls', namespace='products')),
]
```

These deeply nested URLs would be reversed as:

```python
reverse('api:v1:users:detail', kwargs={'user_id': 42})  # /api/v1/users/42/
```

#### Dynamic Nested Routes

A common pattern is to have dynamic segments in the URL followed by nested routes:

```python
# shop/urls.py
app_name = 'shop'

urlpatterns = [
    path('', views.product_list, name='product_list'),
    path('products/<slug:product_slug>/', views.product_detail, name='product_detail'),
    
    # Nested URLs under a product
    path('products/<slug:product_slug>/reviews/', include([
        path('', views.product_reviews, name='product_reviews'),
        path('add/', views.add_review, name='add_review'),
        path('<int:review_id>/', views.review_detail, name='review_detail'),
        path('<int:review_id>/edit/', views.edit_review, name='edit_review'),
    ])),
    
    # Nested URLs under a category
    path('categories/<slug:category_slug>/', include([
        path('', views.category_detail, name='category_detail'),
        path('products/', views.category_products, name='category_products'),
        path('subcategories/', views.subcategories, name='subcategories'),
    ])),
]
```

**Real-world Example**: An e-commerce system with nested routing for products, categories, orders, and customer accounts:

````python
# Main urls.py
from django.urls import path, include
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('shop/', include('shop.urls', namespace='shop')),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('api/', include('api.urls', namespace='api')),
]

# shop/urls.py
app_name = 'shop'

urlpatterns = [
    # Main shop pages
    path('', views.ProductListView.as_view(), name='product_list'),
    path('search/', views.ProductSearchView.as_view(), name='product_search'),
    
    # Category hierarchy
    path('categories/', views.CategoryListView.as_view(), name='category_list'),
    path('categories/<slug:slug>/', include([
        path('', views.CategoryDetailView.as_view(), name='category_detail'),
        path('subcategories/', views.SubcategoryListView.as_view(), name='subcategory_list'),
        path('products/', views.CategoryProductsView.as_view(), name='category_products'),
    ])),
    
    # Product details and related functionality
    path('products/<slug:slug>/', include([
        path('', views.ProductDetailView.as_view(), name='product_detail'),
        path('reviews/', include([
            path('', views.ProductReviewListView.as_view(), name='product_reviews'),
            path('add/', views.ProductReviewCreateView.as_view(), name='add_review'),
            path('<int:review_id>/', views.ProductReviewDetailView.as_view(), name='review_detail'),
            path('<int:review_id>/update/', views.ProductReviewUpdateView.as_view(), name='update_review'),
            path('<int:review_id>/delete/', views.ProductReviewDeleteView.as_view(), name='delete_review'),
        ])),
        path('variants/', views.ProductVariantListView.as_view(), name='product_variants'),
        path('related/', views.RelatedProductsView.as_view(), name='related_products'),
    ])),
    
    # Cart an# Django Interview Questions & Answers (6+ Years Experience)

## 1. Django Architecture & Core Concepts

### What is the request/response lifecycle in Django?

The request/response lifecycle in Django follows a well-defined path from the moment a client sends a request until the server returns a response:

1. **HTTP Request**: A client (browser, API client, etc.) sends an HTTP request to the Django server.

2. **URL Routing**: Django's URL dispatcher (URLconf) analyzes the request URL and determines which view function/class should handle it.

3. **Middleware Processing (Request phase)**: Before reaching the view, the request passes through a series of middleware components in the order they're defined in `MIDDLEWARE` setting. Each middleware can modify the request or even return a response early, short-circuiting the process.

4. **View Processing**: The appropriate view function/class receives the request and processes it. This typically involves:
   - Extracting data from the request
   - Interacting with models/database
   - Processing business logic
   - Preparing context data for templates

5. **Template Rendering** (if applicable): The view often loads a template, populates it with context data, and renders it to HTML.

6. **Middleware Processing (Response phase)**: The response travels back through middleware components in reverse order, allowing them to modify the response.

7. **HTTP Response**: Django sends the final HTTP response back to the client.

**Real-world example**: Consider a product detail page on an e-commerce site:
```python
# urls.py
path('products/<int:product_id>/', views.product_detail, name='product_detail')

# views.py
@login_required  # Authentication middleware checks if user is logged in
def product_detail(request, product_id):
    # View fetches product from database
    product = get_object_or_404(Product, id=product_id)
    
    # Log this view for analytics (custom middleware might track this)
    request.session['last_viewed_product'] = product_id
    
    # Render template with context
    return render(request, 'products/detail.html', {'product': product})
````

In this flow, middleware might handle authentication, session management, and CSRF protection before the view processes the request, then compression and caching might be applied to the response.

### How does Django's MTV architecture differ from MVC?

Django follows an architectural pattern called MTV (Model-Template-View), which is conceptually similar to but differs in terminology from the traditional MVC (Model-View-Controller) pattern:

|MVC Component|Django MTV Equivalent|Responsibility|
|---|---|---|
|Model|Model|Data structure and database interactions|
|View|Template|Presentation and display logic|
|Controller|View|Business logic and request handling|

The key differences:

1. **Django's View** handles what a traditional Controller would do - processing requests, applying business logic, and determining what data to display.
    
2. **Django's Template** corresponds to the traditional View - it defines how the data should be presented.
    
3. **Django's Model** is largely the same as in MVC - it defines the data structure and handles database interactions.
    

The confusion often arises because Django's "View" is essentially a "Controller" in traditional MVC terminology. Django's creators chose this naming to better reflect their specific implementation of the pattern.

**Real-world example**:

```python
# Model - defines data structure and database interactions
class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    publication_date = models.DateTimeField(auto_now_add=True)
    
    def get_absolute_url(self):
        return reverse('article_detail', args=[self.id])

# View (Controller in MVC) - handles business logic
def article_detail(request, article_id):
    article = get_object_or_404(Article, id=article_id)
    return render(request, 'blog/article_detail.html', {'article': article})

# Template (View in MVC) - presentation logic
# article_detail.html
<article>
    <h1>{{ article.title }}</h1>
    <time>{{ article.publication_date|date:"F j, Y" }}</time>
    <div class="content">{{ article.content|linebreaks }}</div>
</article>
```

### How does Django handle routing internally?

Django's URL routing system is a crucial part of its request handling process. Here's how it works internally:

1. **URLconf Loading**: When Django starts, it loads the root URLconf module specified in the `ROOT_URLCONF` setting (typically `project_name.urls`).
    
2. **URL Pattern Compilation**: Django compiles all URL patterns into regular expressions when the server starts, for efficient matching later on.
    
3. **Request Processing**:
    
    - When a request comes in, Django removes the domain name and leading slash.
    - It tries to match the remaining URL path against each pattern in the URLconf in order.
    - The first matching pattern stops the search.
4. **View Resolution**:
    
    - Once a match is found, Django calls the associated view function with:
        - The `HttpRequest` object
        - Any captured values from the URL as positional or keyword arguments
        - Any additional arguments specified in the URL pattern
5. **Include Mechanism**: The `include()` function allows for modular URL configurations by including URL patterns from other URLconf modules.
    
6. **Namespace System**: Django provides a namespace system to disambiguate URL names across applications using the `app_name` variable and the `namespace` parameter to `include()`.
    

**Real-world example**:

```python
# Main urls.py
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', include('blog.urls', namespace='blog')),
    path('shop/', include('shop.urls', namespace='shop')),
]

# blog/urls.py
app_name = 'blog'
urlpatterns = [
    path('', views.article_list, name='article_list'),
    path('article/<int:article_id>/', views.article_detail, name='article_detail'),
    path('category/<slug:category_slug>/', views.category_detail, name='category_detail'),
]
```

When Django processes a request to `/blog/article/42/`:

1. It matches the `blog/` prefix and forwards the rest to `blog.urls`.
2. In `blog.urls`, it matches `article/<int:article_id>/` with `article_id=42`.
3. It calls `views.article_detail(request, article_id=42)`.

The URL name can be referenced as `blog:article_detail` in templates or code.

### Explain middleware and how to create a custom middleware.

Middleware in Django is a framework of hooks into Django's request/response processing. It's a lightweight, low-level "plugin" system for globally altering Django's input or output.

**Middleware Key Characteristics**:

- Executes during request/response cycle, not during Django initialization
- Processes all requests/responses that pass through the system
- Ordered by the `MIDDLEWARE` setting
- Request phase: processes from top to bottom
- Response phase: processes from bottom to top

**Creating Custom Middleware**:

Django supports two styles of middleware:

1. **Function-based middleware**
2. **Class-based middleware**

**Class-based Middleware Example**:

```python
class RequestTimeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time configuration and initialization

    def __call__(self, request):
        # Code to be executed for each request before the view (and later middleware) are called
        start_time = time.time()
        
        # Process the request
        response = self.get_response(request)
        
        # Code to be executed for each request/response after the view is called
        duration = time.time() - start_time
        response['X-Request-Duration'] = f"{duration:.2f}s"
        
        # Log slow requests
        if duration > 1.0:
            logger.warning(f"Slow request: {request.path} took {duration:.2f}s")
            
        return response
    
    # Optional method for view exception processing
    def process_exception(self, request, exception):
        # Log the error
        logger.error(f"Exception in {request.path}: {exception}")
        return None  # Let Django's exception handling take over
```

**Function-based Middleware Example**:

```python
def simple_middleware(get_response):
    # One-time configuration and initialization
    
    def middleware(request):
        # Code to be executed for each request before the view (and later middleware) are called
        
        response = get_response(request)
        
        # Code to be executed for each request/response after the view is called
        
        return response
    
    return middleware
```

**Real-world Example**: A middleware that tracks and limits API usage by IP address.

```python
class RateLimitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limits = {}
        self.window_size = 3600  # 1 hour window
        self.max_requests = 100  # 100 requests per hour

    def __call__(self, request):
        # Only apply rate limiting to API requests
        if request.path.startswith('/api/'):
            ip = self.get_client_ip(request)
            
            # Get or initialize the rate limit record for this IP
            if ip not in self.rate_limits:
                self.rate_limits[ip] = {'count': 0, 'reset_time': time.time() + self.window_size}
            
            # Check if the window has reset
            if time.time() > self.rate_limits[ip]['reset_time']:
                self.rate_limits[ip] = {'count': 0, 'reset_time': time.time() + self.window_size}
            
            # Increment request count
            self.rate_limits[ip]['count'] += 1
            
            # Check if limit exceeded
            if self.rate_limits[ip]['count'] > self.max_requests:
                return JsonResponse(
                    {'error': 'Rate limit exceeded. Try again later.'},
                    status=429
                )
            
            # Add rate limit headers
            response = self.get_response(request)
            response['X-Rate-Limit-Limit'] = str(self.max_requests)
            response['X-Rate-Limit-Remaining'] = str(self.max_requests - self.rate_limits[ip]['count'])
            response['X-Rate-Limit-Reset'] = str(int(self.rate_limits[ip]['reset_time']))
            return response
        
        return self.get_response(request)
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
```

To enable this middleware, add it to the `MIDDLEWARE` setting in your Django project:

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    # ...
    'path.to.RateLimitMiddleware',
    # ...
]
```

## 2. Models & ORM

### How does Django ORM translate Python code into SQL?

Django's Object-Relational Mapping (ORM) system translates Python code into SQL through a complex but elegant process:

1. **Model Definition**: When you define a Django model, you're creating a Python class that inherits from `django.db.models.Model` with attributes that represent database fields.
    
2. **Query Construction**: When you write a query using the ORM, Django constructs a `QuerySet` object. This object is lazy – it doesn't execute the query immediately.
    
3. **Query Compilation**: When the `QuerySet` is evaluated (e.g., when you iterate over it, call `list()` on it, or slice it), Django's query compiler converts it to SQL:
    
    - Django determines the required tables and joins
    - It analyzes the conditions (filters) and converts them to WHERE clauses
    - It processes annotations, aggregations, order_by statements, etc.
4. **SQL Generation**: The compiled query is converted to SQL specific to your database backend using Django's database-specific operations.
    
5. **Query Execution**: The generated SQL is sent to the database for execution.
    
6. **Result Processing**: Database results are converted back into model instances.
    

**Real-world Example**:

```python
# Python model definition
class Customer(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class Order(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='orders')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('shipped', 'Shipped'),
        ('delivered', 'Delivered'),
    ])

# Python query
recent_paid_orders = Order.objects.filter(
    status='paid',
    created_at__gte=datetime.datetime.now() - datetime.timedelta(days=30)
).select_related('customer').order_by('-created_at')
```

When executed, this Python code gets translated to SQL similar to:

```sql
SELECT 
    "orders"."id", "orders"."customer_id", "orders"."total_amount", 
    "orders"."created_at", "orders"."status",
    "customer"."id", "customer"."name", "customer"."email", "customer"."created_at" 
FROM "orders" 
INNER JOIN "customer" ON ("orders"."customer_id" = "customer"."id") 
WHERE ("orders"."status" = 'paid' AND "orders"."created_at" >= '2025-03-31 12:30:45.123456') 
ORDER BY "orders"."created_at" DESC;
```

You can see the actual SQL generated by Django by using:

```python
print(recent_paid_orders.query)
```

### How do you optimize ORM queries for performance?

Optimizing Django ORM queries is crucial for application performance. Here are detailed techniques with examples:

#### 1. Use `select_related()` and `prefetch_related()` to avoid N+1 queries

```python
# Bad: Causes N+1 queries
orders = Order.objects.all()
for order in orders:
    print(order.customer.name)  # Each access triggers a new query

# Good: Just 1 query with a JOIN
orders = Order.objects.select_related('customer')
for order in orders:
    print(order.customer.name)  # No additional query
```

#### 2. Only select the fields you need

```python
# Fetches all fields
users = User.objects.all()

# More efficient: fetches only needed fields
users = User.objects.only('username', 'email', 'last_login')

# Alternative approach
users = User.objects.values('username', 'email', 'last_login')
```

#### 3. Use `values()` or `values_list()` when you don't need model instances

```python
# Returns model instances
products = Product.objects.filter(category='electronics')

# Returns dictionaries - more efficient when you just need data
product_data = Product.objects.filter(category='electronics').values('name', 'price')

# Returns tuples - even more efficient
product_tuples = Product.objects.filter(category='electronics').values_list('name', 'price')

# For a single field, you can flatten the result
product_names = Product.objects.filter(category='electronics').values_list('name', flat=True)
```

#### 4. Use database functions for computation

```python
from django.db.models import F, Sum, Count, Avg
from django.db.models.functions import Coalesce

# Calculate in database instead of Python
Order.objects.update(
    total_price=F('price') * F('quantity') * (1 - F('discount_rate'))
)

# Aggregate in database
report = Order.objects.values('customer_id').annotate(
    order_count=Count('id'),
    total_spent=Sum('total_price'),
    avg_order_value=Avg('total_price')
)
```

#### 5. Use `iterator()` for large querysets

```python
# Memory-efficient processing of large querysets
for product in Product.objects.filter(active=True).iterator():
    # Process each product without loading all into memory
    process_product(product)
```

#### 6. Use indexed fields in filters

```python
# Add indexes to fields used frequently in filtering, ordering or joining
class Customer(models.Model):
    email = models.EmailField(db_index=True)
    last_active = models.DateTimeField(db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['country', 'city']),  # Compound index
            models.Index(fields=['membership_type', '-join_date']),
        ]
```

#### 7. Use `exists()` instead of `count()` or `len()` to check existence

```python
# Inefficient
if User.objects.filter(email=email).count() > 0:
    # User exists

# More efficient
if User.objects.filter(email=email).exists():
    # User exists
```

#### 8. Use `bulk_create()` and `bulk_update()` for batch operations

```python
# Inefficient: N queries
for data in dataset:
    Product.objects.create(name=data['name'], price=data['price'])

# Efficient: 1 query
products = [
    Product(name=data['name'], price=data['price']) 
    for data in dataset
]
Product.objects.bulk_create(products)

# Similarly for updates
Product.objects.bulk_update(products, ['price', 'stock'])
```

#### 9. Consider raw SQL for very complex queries

```python
from django.db import connection

def complex_analytics_query():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                p.category, 
                SUM(oi.quantity * oi.price) as revenue,
                COUNT(DISTINCT o.customer_id) as customer_count
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.created_at > %s
            GROUP BY p.category
            HAVING SUM(oi.quantity) > 100
            ORDER BY revenue DESC
        """, [three_months_ago])
        return cursor.fetchall()
```

#### 10. Use query caching mechanisms

```python
from django.core.cache import cache

def get_active_promotions():
    cache_key = 'active_promotions'
    promotions = cache.get(cache_key)
    
    if promotions is None:
        promotions = list(Promotion.objects.filter(
            is_active=True,
            start_date__lte=timezone.now(),
            end_date__gte=timezone.now()
        ).select_related('product'))
        
        # Cache for 10 minutes
        cache.set(cache_key, promotions, 60 * 10)
    
    return promotions
```

#### 11. Use `defer()` to exclude unnecessary large fields

```python
# Skip loading large text fields when not needed
articles = Article.objects.defer('content', 'metadata').all()
```

#### 12. Use `QuerySet.explain()` to analyze query execution plans (Django 3.0+)

```python
queryset = Order.objects.filter(
    status='processing',
    created_at__gt=last_month
).select_related('customer')

# Print the execution plan
print(queryset.explain())
```

**Real-world Optimization Example**: An e-commerce dashboard that displays sales stats without bogging down the database:

```python
def get_sales_dashboard_data(start_date, end_date):
    # Cache key includes the date range
    cache_key = f'sales_dashboard:{start_date.isoformat()}:{end_date.isoformat()}'
    dashboard_data = cache.get(cache_key)
    
    if dashboard_data is None:
        # Get completed orders in date range
        orders = Order.objects.filter(
            status='completed',
            completed_at__range=(start_date, end_date)
        )
        
        # Calculate revenue and stats in the database
        order_stats = orders.aggregate(
            total_revenue=Sum('total_amount'),
            order_count=Count('id'),
            avg_order_value=Avg('total_amount')
        )
        
        # Get top products efficiently
        top_products = OrderItem.objects.filter(
            order__in=orders
        ).values(
            'product_id', 'product__name'
        ).annotate(
            total_sold=Sum('quantity'),
            revenue=Sum(F('price') * F('quantity'))
        ).order_by('-revenue')[:10]
        
        # Get daily revenue for chart
        daily_revenue = orders.annotate(
            date=TruncDate('completed_at')
        ).values('date').annotate(
            revenue=Sum('total_amount')
        ).order_by('date')
        
        dashboard_data = {
            'order_stats': order_stats,
            'top_products': list(top_products),
            'daily_revenue': list(daily_revenue)
        }
        
        # Cache for 1 hour
        cache.set(cache_key, dashboard_data, 60 * 60)
    
    return dashboard_data
```

### What's the difference between select_related() and prefetch_related()?

Both `select_related()` and `prefetch_related()` are Django ORM methods to optimize database queries by reducing the number of database hits, but they work differently and are suitable for different relationship types:

#### `select_related()`

- **Usage**: For foreign key and one-to-one relationships (where the related object is on the "one" side)
- **Mechanism**: Performs a SQL JOIN and includes the fields of the related object in the SELECT statement
- **Query Count**: Uses a single database query
- **Performance Impact**: Best for "to-one" relationships where you need data from both the model and its related model

**Example**:

```python
# Without select_related - 2 queries
order = Order.objects.get(id=1)  # First query
customer = order.customer  # Second query to fetch the customer

# With select_related - 1 query with JOIN
order = Order.objects.select_related('customer').get(id=1)
customer = order.customer  # No additional query - data already loaded
```

Generated SQL (simplified):

```sql
SELECT 
    orders.id, orders.date, orders.total, /* other order fields */
    customers.id, customers.name, /* other customer fields */
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id
WHERE orders.id = 1;
```

#### `prefetch_related()`

- **Usage**: For many-to-many relationships and reverse foreign key relationships (where the related objects are on the "many" side)
- **Mechanism**: Performs separate queries for each relationship and joins the results in Python
- **Query Count**: Uses multiple queries (one for the main model, one for each prefetched relationship)
- **Performance Impact**: Best for "to-many" relationships where you need to access multiple related objects

**Example**:

```python
# Without prefetch_related - N+1 queries
product = Product.objects.get(id=1)  # First query
categories = product.categories.all()  # Second query for categories
for category in categories:
    print(category.name)

# With prefetch_related - 2 queries
product = Product.objects.prefetch_related('categories').get(id=1)
categories = product.categories.all()  # No additional query
for category in categories:
    print(category.name)  # No additional queries
```

Generated SQL (simplified):

```sql
-- First query fetches the product
SELECT id, name, /* other product fields */ FROM products WHERE id = 1;

-- Second query fetches all related categories
SELECT c.id, c.name, /* other category fields */, pc.product_id 
FROM categories c
INNER JOIN product_categories pc ON c.id = pc.category_id
WHERE pc.product_id IN (1);
```

#### Complex Relationships and Chaining

Both methods can be chained and combined:

```python
# Combining both techniques
orders = Order.objects.select_related('customer').prefetch_related('items__product')

# This efficiently loads:
# 1. Orders
# 2. The customer for each order (via JOIN)
# 3. The items for each order (via separate query)
# 4. The product for each item (via separate query)
```

#### Nested Relationships

Both can traverse multi-level relationships:

```python
# Select related can traverse foreign keys
Order.objects.select_related('customer__address__country')

# Prefetch related can traverse any relationship
Product.objects.prefetch_related(
    'categories',
    'reviews__user',
    Prefetch('variants', queryset=Variant.objects.filter(in_stock=True))
)
```

#### Real-world Example: An e-commerce order detail view

```python
def order_detail(request, order_id):
    # Efficiently fetch the order with all related data in minimal queries
    order = Order.objects.select_related(
        'customer',  # Foreign key - uses JOIN
        'shipping_address',  # Foreign key - uses JOIN
        'billing_address'  # Foreign key - uses JOIN
    ).prefetch_related(
        'items__product',  # Reverse FK + FK - separate queries
        'items__product__categories',  # M2M after FK chain - separate query
        'payment_transactions',  # Reverse FK - separate query
        Prefetch(
            'status_updates',  # Custom prefetch for filtered relationship
            queryset=OrderStatusUpdate.objects.select_related('user').order_by('-timestamp'),
            to_attr='history'
        )
    ).get(id=order_id)
    
    # Now we can access all these related objects without additional queries
    context = {
        'order': order,
        'customer': order.customer,
        'address': order.shipping_address,
        'items': order.items.all(),  # No query
        'payment_history': order.payment_transactions.all(),  # No query
        'status_history': order.history  # From prefetch to_attr
    }
    
    return render(request, 'orders/detail.html', context)
```

### How do you perform raw SQL queries in Django, and when should you use them?

Django provides several ways to execute raw SQL queries when the ORM doesn't provide the flexibility or performance you need:

#### 1. Using `Manager.raw()` method

The `raw()` method executes a raw SQL query and returns a `RawQuerySet` of model instances:

```python
# Simple raw query
products = Product.objects.raw('SELECT * FROM products WHERE price > %s', [100])

# More complex raw query with joins
customers = Customer.objects.raw('''
    SELECT c.id, c.name, c.email, COUNT(o.id) as order_count
    FROM customers c
    LEFT JOIN orders o ON c.id = o.customer_id
    WHERE c.is_active = True
    GROUP BY c.id
    HAVING COUNT(o.id) > 5
    ORDER BY order_count DESC
''')

# Accessing results
for customer in customers:
    print(customer.name, customer.order_count)  # Note: order_count is dynamically added
```

**Important considerations**:

- You must include the primary key column in your query
- Django maps the query results to model instances
- You can map extra SELECT fields to model attributes
- Parameters should be passed as a list to prevent SQL injection

#### 2. Using `connection.cursor()` for complete control

For queries that don't map to models or for non-SELECT operations:

```python
from django.db import connection

def get_product_sales_report():
    with connection.cursor() as cursor:
        cursor.execute('''
            SELECT 
                p.name, 
                p.sku, 
                SUM(oi.quantity) as units_sold,
                SUM(oi.quantity * oi.price) as revenue
            FROM products p
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.status = 'completed'
            AND o.completion_date > %s
            GROUP BY p.id, p.name, p.sku
            ORDER BY revenue DESC
        ''', [three_months_ago])
        
        # Convert results to dictionaries
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
```

#### 3. Using `connection.execute()` method (Django 4.0+)

```python
from django.db import connection

def update_product_prices(category_id, increase_percentage):
    with connection.execute('''
        UPDATE products
        SET price = price * (1 + %s/100)
        WHERE category_id = %s
    ''', [increase_percentage, category_id]) as cursor:
        return cursor.rowcount  # Number of rows affected
```

#### 4. Using database-specific operations with `QuerySet.annotate()`

Django 3.2+ allows using database functions directly:

```python
from django.db.models import F, Value
from django.db.models.functions import Cast
from django.db.models.expressions import RawSQL

# Using RawSQL within a queryset
Product.objects.annotate(
    distance=RawSQL(
        "ST_Distance(location, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
        (longitude, latitude)
    )
).order_by('distance')
```

#### When to use raw SQL:

1. **Complex queries beyond ORM capabilities**:
    
    - Advanced window functions
    - Complex subqueries
    - Hierarchical/recursive queries (CTE)
    - Advanced geospatial queries
2. **Performance optimization**:
    
    - When ORM-generated queries are inefficient
    - For queries manipulating large datasets
    - When you need database-specific optimizations
3. **Bulk operations**:
    
    - Mass updates with complex conditions
    - Specialized batch processing
4. **Database-specific features**:
    
    - Using features specific to your database like PostgreSQL's JSONB operations
5. **Schema migration operations**:
    
    - Custom, complex schema changes

**Real-world Example**: A geospatial search with complex filtering:

```python
def find_nearby_restaurants(latitude, longitude, radius_km, cuisine=None, min_rating=None):
    query = '''
        SELECT 
            r.id, r.name, r.address, r.rating,
            ST_Distance(
                r.location, 
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            ) * 111.32 AS distance_km
        FROM restaurants r
        LEFT JOIN restaurant_cuisines rc ON r.id = rc.restaurant_id
        LEFT JOIN cuisines c ON rc.cuisine_id = c.id
        WHERE ST_DWithin(
            r.location, 
            ST_SetSRID(ST_MakePoint(%s, %s), 4326),
            %s / 111.32
        )
    '''
    
    params = [longitude, latitude, longitude, latitude, radius_km]
    
    if cuisine:
        query += " AND c.name = %s"
        params.append(cuisine)
    
    if min_rating:
        query += " AND r.rating >= %s"
        params.append(min_rating)
    
    query += " GROUP BY r.id, r.name, r.address, r.rating, r.location ORDER BY distance_km"
    
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
```
# Cart and checkout process

```
path('cart/', include([
    path('', views.CartView.as_view(), name='cart'),
    path('add/<int:product_id>/', views.AddToCartView.as_view(), name='add_to_cart'),
    path('update/<int:item_id>/', views.UpdateCartItemView.as_view(), name='update_cart_item'),
    path('remove/<int:item_id>/', views.RemoveFromCartView.as_view(), name='remove_from_cart'),
    path('clear/', views.ClearCartView.as_view(), name='clear_cart'),
])),

path('checkout/', include([
    path('', views.CheckoutView.as_view(), name='checkout'),
    path('shipping/', views.ShippingDetailsView.as_view(), name='shipping_details'),
    path('payment/', views.PaymentView.as_view(), name='payment'),
    path('confirmation/', views.OrderConfirmationView.as_view(), name='order_confirmation'),
])),

# Order management
path('orders/', include([
    path('', views.OrderListView.as_view(), name='order_list'),
    path('<uuid:order_id>/', include([
        path('', views.OrderDetailView.as_view(), name='order_detail'),
        path('invoice/', views.OrderInvoiceView.as_view(), name='order_invoice'),
        path('cancel/', views.CancelOrderView.as_view(), name='cancel_order'),
        path('track/', views.TrackOrderView.as_view(), name='track_order'),
    ])),
])),
```

]

````

In this comprehensive e-commerce example:

1. **Main URL Structure**:
   - The application is divided into logical sections (shop, accounts, api)
   - Each section has its own namespace to avoid naming conflicts

2. **Dynamic URLs**:
   - Products are accessed by slug: `/shop/products/black-django-t-shirt/`
   - Categories use slugs: `/shop/categories/apparel/`
   - Orders use UUIDs: `/shop/orders/f47ac10b-58cc-4372-a567-0e02b2c3d479/`

3. **Nested Routes**:
   - Product-related functionality is nested under the product URL
   - Category functionality is grouped under the category URL
   - Order functionality is organized under the order URL

4. **Multi-level Nesting**:
   - Product reviews have their own nested structure
   - Cart and checkout have multiple steps with dedicated URLs

#### Handling Nested Routes in Views

When working with nested routes, views need to handle the parent resource context:

```python
class ProductReviewListView(ListView):
    template_name = 'shop/product_reviews.html'
    context_object_name = 'reviews'
    paginate_by = 10
    
    def get_queryset(self):
        # Get the product from the URL parameter
        self.product = get_object_or_404(Product, slug=self.kwargs['slug'])
        # Return only reviews for this product
        return Review.objects.filter(product=self.product).order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['product'] = self.product
        return context

class ProductReviewCreateView(LoginRequiredMixin, CreateView):
    model = Review
    form_class = ReviewForm
    template_name = 'shop/add_review.html'
    
    def get_success_url(self):
        return reverse('shop:product_reviews', kwargs={'slug': self.kwargs['slug']})
    
    def form_valid(self, form):
        # Set the product and user automatically
        form.instance.product = get_object_or_404(Product, slug=self.kwargs['slug'])
        form.instance.user = self.request.user
        return super().form_valid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['product'] = get_object_or_404(Product, slug=self.kwargs['slug'])
        return context
````

#### Breadcrumb Navigation for Nested Routes

Nested routing pairs well with breadcrumb navigation to help users understand their location:

```python
def get_breadcrumbs(request):
    """Generate breadcrumbs based on the current URL path."""
    path = request.path.strip('/')
    parts = path.split('/')
    breadcrumbs = [{'title': 'Home', 'url': '/'}]
    
    # Build up breadcrumbs based on path parts
    current_path = ''
    for i, part in enumerate(parts):
        current_path += f'{part}/'
        
        # Handle special cases
        if i == 0 and part == 'shop':
            breadcrumbs.append({'title': 'Shop', 'url': '/shop/'})
        elif i == 1 and parts[0] == 'shop' and part == 'products':
            breadcrumbs.append({'title': 'Products', 'url': '/shop/products/'})
        elif i == 2 and parts[0] == 'shop' and parts[1] == 'products':
            # This is a product slug
            product = get_object_or_404(Product, slug=part)
            breadcrumbs.append({'title': product.name, 'url': product.get_absolute_url()})
        elif i == 3 and parts[0] == 'shop' and parts[1] == 'products' and part == 'reviews':
            breadcrumbs.append({'title': 'Reviews', 'url': current_path})
        # Add more cases as needed
            
    return breadcrumbs
```

#### Best Practices for Complex URL Structures

1. **Use Namespaces Consistently**: Always use application and instance namespaces to avoid naming conflicts.
    
2. **Keep URL Patterns Organized**: Group related URLs together in modules or nested include() blocks.
    
3. **Leverage URL Parameters**: Use path converters to extract and validate URL segments.
    
4. **Document URL Structure**: Maintain documentation of the URL hierarchy for developers.
    
5. **Use Meaningful URL Names**: Choose descriptive URL pattern names that reflect their purpose.
    
6. **Consider URL Length**: Avoid excessive nesting that creates very long URLs.
    
7. **Use URL Patterns for Resource Hierarchies**: Represent parent-child relationships in URL structure.
    
8. **Be Consistent with URL Format**: Maintain consistent format (e.g., trailing slashes, pluralization).
    

By implementing these patterns and best practices, you can create a robust, maintainable URL structure that effectively represents the hierarchical nature of your application's resources and functionality.

## 4. Forms and Validation

### How do Django forms differ from ModelForms?

Django provides two primary types of forms: regular `Form` and `ModelForm`. They both handle form rendering, user input processing, and validation, but they have key differences in purpose, implementation, and functionality.

#### Core Differences

|Feature|Django Form|Django ModelForm|
|---|---|---|
|**Purpose**|General-purpose form handling|Form tied to a specific model|
|**Fields Definition**|Manually defined fields|Automatically generated from model fields|
|**Data Storage**|Doesn't handle saving to database|Built-in methods to save data to models|
|**Customization**|Complete control over fields|Inherits validation from model but can be customized|
|**Field Mapping**|No automatic mapping|Maps to model fields automatically|

#### Django Form

A regular Django `Form` is a standalone form not tied to any model. You explicitly define the fields, validations, and handling logic.

**Example of a regular Form:**

```python
from django import forms

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()
    subject = forms.CharField(max_length=200)
    message = forms.CharField(widget=forms.Textarea)
    cc_myself = forms.BooleanField(required=False)
    
    def clean_email(self):
        email = self.cleaned_data['email']
        if not email.endswith('@example.com'):
            raise forms.ValidationError("Only example.com email addresses allowed")
        return email
```

**Using a regular Form in a view:**

```python
def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Process the data - no automatic saving
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']
            cc_myself = form.cleaned_data['cc_myself']
            
            # Manually handle the data (e.g., send email)
            send_contact_email(name, email, subject, message, cc_myself)
            
            return redirect('contact_success')
    else:
        form = ContactForm()
    
    return render(request, 'contact_form.html', {'form': form})
```

#### Django ModelForm

A `ModelForm` is a form that's directly tied to a Django model. It automatically generates form fields based on the model fields and includes built-in methods to save the data to the database.

**Example of a ModelForm:**

```python
from django import forms
from .models import Article

class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = ['title', 'content', 'category', 'tags', 'status']
        # or use exclude = ['author', 'created_at', 'updated_at']
        widgets = {
            'content': forms.Textarea(attrs={'rows': 10}),
            'status': forms.RadioSelect(),
        }
        labels = {
            'content': 'Article Body',
        }
        help_texts = {
            'tags': 'Enter tags separated by commas',
        }
```

**Using a ModelForm in a view:**

```python
def create_article(request):
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            # Automatic saving to the database
            article = form.save(commit=False)  # Don't save to DB just yet
            article.author = request.user  # Set additional fields
            article.save()  # Now save to DB
            form.save_m2m()  # Save many-to-many relationships
            
            return redirect('article_detail', pk=article.pk)
    else:
        form = ArticleForm()
    
    return render(request, 'article_form.html', {'form': form})
```

#### Key Advantages of Regular Forms

1. **Flexibility**: Not tied to any model, so ideal for forms that don't directly map to database models.
    
2. **Complete Control**: Full control over field definitions, validations, and data handling.
    
3. **Complex Processing**: Better for forms that require complex data processing not directly related to saving a model.
    
4. **Composite Forms**: Can combine data from multiple sources or models.
    
5. **Non-persistent Data**: Ideal for data that doesn't need to be stored in the database (e.g., contact forms, search forms).
    

#### Key Advantages of ModelForms

1. **Efficiency**: Automatically generate forms from models, saving development time.
    
2. **Consistency**: Form fields inherit validation rules from the model, ensuring data integrity.
    
3. **Built-in Saving**: The `save()` method handles database operations automatically.
    
4. **DRY Principle**: Avoids duplicating field definitions and validation logic already present in models.
    
5. **Simple CRUD Operations**: Ideal for standard create/update operations on model instances.
    

#### Advanced Usage

**1. Combining Forms and ModelForms**

Sometimes you need a form that mostly maps to a model but has additional fields:

```python
class ArticleWithImagesForm(forms.ModelForm):
    # Regular form fields not in the model
    upload_images = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}), required=False)
    remove_featured_image = forms.BooleanField(required=False)
    
    class Meta:
        model = Article
        fields = ['title', 'content', 'category', 'tags', 'featured_image']
    
    def save(self, commit=True):
        article = super().save(commit=commit)
        
        # Handle the uploaded images (create Image objects related to Article)
        if commit and self.cleaned_data.get('upload_images'):
            for image_file in self.cleaned_data['upload_images']:
                ArticleImage.objects.create(
                    article=article,
                    image=image_file,
                    caption=f"Image for {article.title}"
                )
        
        # Handle image removal if checked
        if commit and self.cleaned_data.get('remove_featured_image') and article.featured_image:
            article.featured_image.delete()
            article.featured_image = None
            article.save()
            
        return article
```

**2. Using Form Inheritance**

You can create a base form and extend it:

```python
# Base form with common fields
class PersonForm(forms.Form):
    first_name = forms.CharField(max_length=100)
    last_name = forms.CharField(max_length=100)
    email = forms.EmailField()
    phone = forms.CharField(max_length=20, required=False)

# Extended form for customers
class CustomerForm(PersonForm):
    company = forms.CharField(max_length=100, required=False)
    address = forms.CharField(widget=forms.Textarea)
    
# Extended form for employees
class EmployeeForm(PersonForm):
    department = forms.ChoiceField(choices=DEPARTMENT_CHOICES)
    employee_id = forms.CharField(max_length=10)
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
```

**3. Dynamic ModelForms**

Forms that adapt based on user or context:

```python
def create_product_form(user, *args, **kwargs):
    """Factory function to create a product form based on user permissions."""
    
    class ProductForm(forms.ModelForm):
        class Meta:
            model = Product
            
            # Admin users can edit all fields
            if user.is_staff:
                fields = ['name', 'description', 'price', 'cost_price', 'category', 
                         'stock_level', 'sku', 'is_active', 'featured']
            # Store managers can edit most fields
            elif user.has_perm('products.manage_products'):
                fields = ['name', 'description', 'price', 'category', 
                         'stock_level', 'is_active', 'featured']
            # Regular content editors can only edit content
            else:
                fields = ['name', 'description', 'category']
    
    return ProductForm(*args, **kwargs)

# Usage in a view
def edit_product(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    
    # Create the form dynamically based on the user
    ProductForm = create_product_form(request.user)
    
    if request.method == 'POST':
        form = ProductForm(request.POST, instance=product)
        if form.is_valid():
            form.save()
            return redirect('product_detail', pk=product.pk)
    else:
        form = ProductForm(instance=product)
    
    return render(request, 'product_form.html', {'form': form})
```

**Real-world Example**: An e-commerce system with different form types:

```python
from django import forms
from django.core.validators import MinValueValidator
from .models import Product, Order, OrderItem, Customer, ShippingAddress

# Regular Form (not tied to a model)
class ProductSearchForm(forms.Form):
    """Search form for products with various filtering options."""
    query = forms.CharField(required=False, label="Search", 
                           widget=forms.TextInput(attrs={'placeholder': 'Search products...'}))
    category = forms.ModelChoiceField(queryset=Category.objects.all(), required=False)
    min_price = forms.DecimalField(decimal_places=2, required=False, 
                                  widget=forms.NumberInput(attrs={'placeholder': 'Min price'}))
    max_price = forms.DecimalField(decimal_places=2, required=False,
                                  widget=forms.NumberInput(attrs={'placeholder': 'Max price'}))
    in_stock_only = forms.BooleanField(required=False, initial=False)
    sort_by = forms.ChoiceField(choices=[
        ('name', 'Name A-Z'),
        ('-name', 'Name Z-A'),
        ('price', 'Price Low to High'),
        ('-price', 'Price High to Low'),
        ('-created_at', 'Newest First'),
    ], required=False, initial='-created_at')
    
    def clean(self):
        cleaned_data = super().clean()
        min_price = cleaned_data.get('min_price')
        max_price = cleaned_data.get('max_price')
        
        if min_price and max_price and min_price > max_price:
            raise forms.ValidationError("Minimum price cannot be greater than maximum price")
        
        return cleaned_data

# ModelForm for product management
class ProductForm(forms.ModelForm):
    """Form for creating and editing products."""
    class Meta:
        model = Product
        fields = ['name', 'description', 'category', 'price', 'sale_price',
                 'stock_quantity', 'image', 'is_active', 'featured']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 5}),
            'image': forms.ClearableFileInput(attrs={'multiple': False}),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        price = cleaned_data.get('price')
        sale_price = cleaned_data.get('sale_price')
        
        if sale_price and price and sale_price >= price:
            self.add_error('sale_price', "Sale price must be less than regular price")
        
        return cleaned_data

# Combined approach - ModelForm with extra fields
class CheckoutForm(forms.ModelForm):
    """Checkout form combining order details and shipping information."""
    # Fields for the Order model
    class Meta:
        model = Order
        fields = ['payment_method', 'notes']
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Special instructions for delivery'}),
        }
    
    # Additional fields for shipping (not directly on Order model)
    use_saved_address = forms.BooleanField(required=False, initial=True,
                                         widget=forms.CheckboxInput(attrs={'class': 'toggle-address-form'}))
    saved_address = forms.ModelChoiceField(queryset=None, required=False)
    
    # Shipping address fields
    name = forms.CharField(max_length=100, required=False)
    street_address = forms.CharField(max_length=250, required=False)
    city = forms.CharField(max_length=100, required=False)
    state = forms.CharField(max_length=100, required=False)
    postal_code = forms.CharField(max_length=20, required=False)
    country = forms.ChoiceField(choices=COUNTRY_CHOICES, required=False)
    phone = forms.CharField(max_length=20, required=False)
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # If user is logged in, populate saved addresses
        if self.user and self.user.is_authenticated:
            self.fields['saved_address'].queryset = ShippingAddress.objects.filter(user=self.user)
        else:
            self.fields['use_saved_address'].widget = forms.HiddenInput()
            self.fields['saved_address'].widget = forms.HiddenInput()
    
    def clean(self):
        cleaned_data = super().clean()
        use_saved_address = cleaned_data.get('use_saved_address')
        saved_address = cleaned_data.get('saved_address')
        
        # Validate that either a saved address is selected or all shipping fields are filled
        if use_saved_address and not saved_address:
            self.add_error('saved_address', "Please select a saved address")
        elif not use_saved_address:
            required_fields = ['name', 'street_address', 'city', 'postal_code', 'country', 'phone']
            for field in required_fields:
                if not cleaned_data.get(field):
                    self.add_error(field, "This field is required")
        
        return cleaned_data
    
    def save(self, commit=True):
        order = super().save(commit=False)
        
        if commit:
            order.user = self.user
            order.status = 'pending'
            order.save()
            
            # Create or use shipping address
            if self.cleaned_data.get('use_saved_address'):
                shipping_address = self.cleaned_data['saved_address']
                order.shipping_address = shipping_address
            else:
                # Create new shipping address
                shipping_address = ShippingAddress(
                    user=self.user,
                    name=self.cleaned_data['name'],
                    street_address=self.cleaned_data['street_address'],
                    city=self.cleaned_data['city'],
                    state=self.cleaned_data['state'],
                    postal_code=self.cleaned_data['postal_code'],
                    country=self.cleaned_data['country'],
                    phone=self.cleaned_data['phone']
                )
                shipping_address.save()
                order.shipping_address = shipping_address
            
            order.save()
            
            # Transfer cart items to order items
            cart = Cart(self.request)
            for item in cart:
                OrderItem.objects.create(
                    order=order,
                    product=item['product'],
                    price=item['price'],
                    quantity=item['quantity']
                )
            
            # Clear the cart
            cart.clear()
        
        return order
```

In summary, Django Forms and ModelForms serve different purposes but complement each other well. Regular Forms provide flexibility for custom form processing, while ModelForms offer tight integration with database models and simplified data persistence. Understanding when to use each type is a key skill for effective Django development.

### How do you write custom validation at the field and form level?

Django provides multiple levels of form validation, allowing you to implement validation rules that range from simple field-level checks to complex cross-field validations. This layered approach gives you fine-grained control over data validation.

#### 1. Field-Level Validation

Field-level validation focuses on validating a single field's value, independent of other fields.

##### 1.1 Using Built-in Validators

Django comes with many built-in validators that can be applied to form fields:

```python
from django import forms
from django.core.validators import MinLengthValidator, RegexValidator, EmailValidator

class UserProfileForm(forms.Form):
    username = forms.CharField(
        min_length=3,  # Built-in validation
        max_length=30,  # Built-in validation
        validators=[
            RegexValidator(
                regex=r'^[a-zA-Z0-9_]+def article_detail(request, pk):
    article = get_object_or_404(Article, pk=pk)
    return render(request, 'blog/article_detail.html', {'article': article})
```

With DetailView:

```python
class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    context_object_name = 'article'
```

**3. CreateView - Create a new object**

Before generic views:

```python
def create_article(request):
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save(commit=False)
            article.author = request.user
            article.save()
            return redirect('article_detail', pk=article.pk)
    else:
        form = ArticleForm()
    return render(request, 'blog/article_form.html', {'form': form})
```

With CreateView:

```python
class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
    
    def get_success_url(self):
        return reverse('article_detail', kwargs={'pk': self.object.pk})
```

**4. UpdateView - Update an existing object**

Before generic views:

```python
def update_article(request, pk):
    article = get_object_or_404(Article, pk=pk)
    if request.method == 'POST':
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            form.save()
            return redirect('article_detail', pk=article.pk)
    else:
        form = ArticleForm(instance=article)
    return render(request, 'blog/article_form.html', {'form': form})
```

With UpdateView:

```python
class ArticleUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user
    
    def get_success_url(self):
        return reverse('article_detail', kwargs={'pk': self.object.pk})
```

**5. DeleteView - Delete an existing object**

Before generic views:

```python
def delete_article(request, pk):
    article = get_object_or_404(Article, pk=pk)
    if request.method == 'POST':
        article.delete()
        return redirect('article_list')
    return render(request, 'blog/article_confirm_delete.html', {'article': article})
```

With DeleteView:

```python
class ArticleDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Article
    template_name = 'blog/article_confirm_delete.html'
    success_url = reverse_lazy('article_list')
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user
```

#### Customizing Generic Views

While generic views handle many common cases out of the box, they are designed to be customizable:

**1. Customizing Query Sets**

```python
class PublishedArticleListView(ListView):
    model = Article
    template_name = 'blog/article_list.html'
    context_object_name = 'articles'
    
    def get_queryset(self):
        return Article.objects.filter(status='published').order_by('-published_date')
```

**2. Adding Extra Context Data**

```python
class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comments'] = self.object.comments.all()
        context['related_articles'] = Article.objects.filter(
            category=self.object.category
        ).exclude(id=self.object.id)[:5]
        return context
```

**3. Custom Form Processing**

```python
class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    
    def form_valid(self, form):
        # Customize before saving
        form.instance.author = self.request.user
        form.instance.status = 'draft'
        
        # Save the object
        response = super().form_valid(form)
        
        # Customize after saving
        if form.cleaned_data.get('notify_subscribers'):
            self.object.notify_subscribers()
            
        return response
```

**4. Custom URL Parameters**

```python
class CategoryArticleListView(ListView):
    model = Article
    template_name = 'blog/category_articles.html'
    
    def get_queryset(self):
        self.category = get_object_or_404(Category, slug=self.kwargs['slug'])
        return Article.objects.filter(category=self.category)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['category'] = self.category
        return context
```

**Real-world Example**: A complete blog application using generic views:

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('category_detail', kwargs={'slug': self.slug})

class Article(models.Model):
    STATUS_CHOICES = (
        ('draft', 'Draft'),
        ('published', 'Published'),
    )
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='articles')
    content = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='articles')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_date = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('article_detail', kwargs={'slug': self.slug})

class Comment(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Comment by {self.author.username} on {self.article.title}"

# views.py
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.views.generic.dates import YearArchiveView, MonthArchiveView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse_lazy
from .models import Article, Category, Comment
from .forms import ArticleForm, CommentForm

class ArticleListView(ListView):
    model = Article
    template_name = 'blog/article_list.html'
    context_object_name = 'articles'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Article.objects.filter(status='published').order_by('-published_date')
        
        # Filter by category if provided
        category_slug = self.request.GET.get('category')
        if category_slug:
            queryset = queryset.filter(category__slug=category_slug)
            
        # Filter by search query if provided
        search_query = self.request.GET.get('q')
        if search_query:
            queryset = queryset.filter(
                Q(title__icontains=search_query) | 
                Q(content__icontains=search_query)
            )
            
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        context['search_query'] = self.request.GET.get('q', '')
        context['category_filter'] = self.request.GET.get('category', '')
        return context

class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    context_object_name = 'article'
    slug_url_kwarg = 'slug'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comment_form'] = CommentForm()
        context['comments'] = self.object.comments.all().order_by('-created_at')
        context['related_articles'] = Article.objects.filter(
            category=self.object.category, 
            status='published'
        ).exclude(id=self.object.id)[:3]
        return context
    
    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form = CommentForm(request.POST)
        
        if form.is_valid() and request.user.is_authenticated:
            comment = form.save(commit=False)
            comment.article = self.object
            comment.author = request.user
            comment.save()
            return redirect(self.object.get_absolute_url())
            
        context = self.get_context_data(object=self.object)
        context['comment_form'] = form
        return self.render_to_response(context)

class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

class ArticleUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    slug_url_kwarg = 'slug'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user or self.request.user.is_staff

class ArticleDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Article
    template_name = 'blog/article_confirm_delete.html'
    success_url = reverse_lazy('article_list')
    slug_url_kwarg = 'slug'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user or self.request.user.is_staff

class CategoryDetailView(DetailView):
    model = Category
    template_name = 'blog/category_detail.html'
    context_object_name = 'category'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['articles'] = self.object.articles.filter(
            status='published'
        ).order_by('-published_date')
        return context

class ArticleYearArchiveView(YearArchiveView):
    model = Article
    date_field = 'published_date'
    make_object_list = True
    template_name = 'blog/article_archive_year.html'
    
    def get_queryset(self):
        return Article.objects.filter(status='published')

class ArticleMonthArchiveView(MonthArchiveView):
    model = Article
    date_field = 'published_date'
    template_name = 'blog/article_archive_month.html'
    
    def get_queryset(self):
        return Article.objects.filter(status='published')

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ArticleListView.as_view(), name='article_list'),
    path('article/<slug:slug>/', views.ArticleDetailView.as_view(), name='article_detail'),
    path('article/create/', views.ArticleCreateView.as_view(), name='article_create'),
    path('article/<slug:slug>/update/', views.ArticleUpdateView.as_view(), name='article_update'),
    path('article/<slug:slug>/delete/', views.ArticleDeleteView.as_view(), name='article_delete'),
    path('category/<slug:slug>/', views.CategoryDetailView.as_view(), name='category_detail'),
    path('archive/<int:year>/', views.ArticleYearArchiveView.as_view(), name='article_year_archive'),
    path('archive/<int:year>/<int:month>/', views.ArticleMonthArchiveView.as_view(), name='article_month_archive'),
]
```

This example demonstrates how generic views:

- Eliminate repetitive code for common operations
- Provide a consistent structure across views
- Allow for customization where needed
- Enable quick addition of features like pagination and filtering
- Support complex functionality through mixins and inheritance

Generic views are most beneficial when you're implementing standard CRUD operations and want to maintain consistent behaviors, but they remain flexible enough to handle custom business logic when needed.

### How does Django's reverse() function work?

Django's `reverse()` function is a powerful URL resolution tool that dynamically generates URLs from URL patterns defined in your URLconf. This is crucial for maintaining DRY (Don't Repeat Yourself) principles by avoiding hardcoded URLs throughout your codebase.

#### Basic Functionality

At its core, `reverse()` takes a URL pattern name and optional arguments, then returns the corresponding URL path:

```python
from django.urls import reverse

# Basic usage
article_url = reverse('article_detail', args=[42])  # Returns '/articles/42/'
```

#### How It Works Internally

1. **Pattern Lookup**: Django searches all URL patterns across all included URLconfs for a pattern with the given name.
    
2. **Pattern Matching**: Once found, Django uses the pattern's regular expression to construct a URL.
    
3. **Argument Substitution**:
    
    - Positional arguments from `args` are inserted into the pattern in order
    - Named arguments from `kwargs` are matched to named groups in the pattern
4. **URL Construction**: The final URL string is assembled, including the prefix from any parent URLconfs.
    

#### Usage Patterns

**1. Basic URL Reversal**

```python
# URL definition
path('articles/<int:article_id>/', views.article_detail, name='article_detail')

# In a view
def some_view(request):
    url = reverse('article_detail', args=[42])  # '/articles/42/'
    return redirect(url)
```

**2. Using Named Arguments**

```python
# URL definition
path('articles/<int:year>/<int:month>/<slug:slug>/', views.article_detail, name='article_detail')

# In a view
def some_view(request):
    url = reverse('article_detail', kwargs={
        'year': 2025,
        'month': 4,
        'slug': 'django-reverse-explained'
    })  # '/articles/2025/4/django-reverse-explained/'
    return redirect(url)
```

**3. URL Namespaces**

Django supports URL namespaces to avoid name clashes between apps:

```python
# In main urls.py
path('blog/', include('blog.urls', namespace='blog'))
path('news/', include('news.urls', namespace='news'))

# In a view
def some_view(request):
    blog_url = reverse('blog:article_detail', args=[42])  # '/blog/articles/42/'
    news_url = reverse('news:article_detail', args=[42])  # '/news/articles/42/'
    return render(request, 'template.html', {'blog_url': blog_url, 'news_url': news_url})
```

**4. Reversing in Templates**

In Django templates, you can use the `url` template tag:

```html
<a href="{% url 'article_detail' article.id %}">{{ article.title }}</a>

<!-- With namespace -->
<a href="{% url 'blog:article_detail' article.id %}">{{ article.title }}</a>
```

**5. Reversing in Models**

The `reverse()` function is commonly used in models to provide absolute URLs:

```python
from django.db import models
from django.urls import reverse

class Article(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    content = models.TextField()
    
    def get_absolute_url(self):
        return reverse('article_detail', kwargs={'slug': self.slug})
```

**6. Handling Current App Namespaces**

When working with app namespaces and the current app might vary:

```python
from django.urls import reverse, resolve
from django.urls.exceptions import NoReverseMatch

def get_url_in_current_app(request, view_name, *args, **kwargs):
    current_namespace = resolve(request.path).namespace
    try:
        # Try with current namespace
        return reverse(f"{current_namespace}:{view_name}", args=args, kwargs=kwargs)
    except NoReverseMatch:
        # Fall back to no namespace
        return reverse(view_name, args=args, kwargs=kwargs)
```

#### Advanced Usage

**1. Using `reverse_lazy()`**

For cases where you need a URL reference at import time (before URLs are loaded):

```python
from django.urls import reverse_lazy

class ArticleDeleteView(DeleteView):
    model = Article
    # URLs not loaded when class is defined, so we use reverse_lazy
    success_url = reverse_lazy('article_list')
```

**2. Handling Optional URL Parameters**

For URLs with optional parameters, you often need conditional logic:

```python
def get_filtered_list_url(category=None, tag=None):
    if category and tag:
        return reverse('article_list') + f'?category={category}&tag={tag}'
    elif category:
        return reverse('article_list') + f'?category={category}'
    elif tag:
        return reverse('article_list') + f'?tag={tag}'
    return reverse('article_list')
```

**3. Building APIs with Reverse**

For building API links:

```python
from django.urls import reverse
from rest_framework.response import Response

class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def list(self, request):
        articles = self.get_queryset()
        data = [{
            'id': article.id,
            'title': article.title,
            'url': request.build_absolute_uri(reverse('api:article-detail', args=[article.id]))
        } for article in articles]
        return Response(data)
```

**Real-world Example**: An e-commerce platform with complex URL structure:

```python
# urls.py in the main project
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('shop/', include('shop.urls', namespace='shop')),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('blog/', include('blog.urls', namespace='blog')),
    path('api/', include('api.urls', namespace='api')),
]

# shop/urls.py
app_name = 'shop'

urlpatterns = [
    path('', views.ProductListView.as_view(), name='product_list'),
    path('category/<slug:category_slug>/', views.CategoryDetailView.as_view(), name='category_detail'),
    path('product/<slug:slug>/', views.ProductDetailView.as_view(), name='product_detail'),
    path('cart/', views.CartView.as_view(), name='cart'),
    path('checkout/', views.CheckoutView.as_view(), name='checkout'),
    path('orders/', views.OrderListView.as_view(), name='order_list'),
    path('orders/<uuid:order_id>/', views.OrderDetailView.as_view(), name='order_detail'),
]

# shop/models.py
class Product(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    category = models.ForeignKey('Category', on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    
    def get_absolute_url(self):
        return reverse('shop:product_detail', kwargs={'slug': self.slug})
    
    def get_add_to_cart_url(self):
        return reverse('shop:add_to_cart', kwargs={'product_id': self.id})
    
    def get_related_products_url(self):
        return reverse('api:related-products', kwargs={'product_id': self.id})

# shop/views.py
def add_to_cart(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    cart = Cart(request)
    cart.add(product)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'success': True,
            'cart_count': cart.get_item_count(),
            'cart_url': reverse('shop:cart')
        })
    
    # Determine where to redirect based on the referrer
    next_url = request.GET.get('next')
    if next_url:
        return redirect(next_url)
        
    return redirect(product.get_absolute_url())

class OrderCreateView(LoginRequiredMixin, CreateView):
    model = Order
    form_class = OrderForm
    template_name = 'shop/checkout.html'
    
    def form_valid(self, form):
        form.instance.user = self.request.user
        response = super().form_valid(form)
        
        # Clear the cart after successful order
        cart = Cart(self.request)
        cart.clear()
        
        # Send confirmation email
        send_order_confirmation.delay(self.object.id)
        
        return response
    
    def get_success_url(self):
        # Use reverse with kwargs to build the success URL
        return reverse('shop:order_confirmation', kwargs={'order_id': self.object.id})
```

In this example:

1. `reverse()` helps generate URLs for different parts of the e-commerce application
2. URL namespaces (`shop:`, `accounts:`, etc.) keep URL names organized
3. Models use `get_absolute_url()` for canonical URLs
4. Views use `reverse()` for redirects after form submissions
5. AJAX responses include URLs generated by `reverse()`

The `reverse()` function helps maintain a clean separation between URL structure and application logic, allowing URLs to be changed without breaking functionality throughout the application.

### How would you handle dynamic URLs and nested routing?

Dynamic URLs and nested routing are essential for creating flexible, hierarchical URL structures in Django applications. Properly handling these patterns allows for more intuitive URLs and better organization of your web application.

#### Dynamic URLs

Django's URL patterns support capturing values from the URL path using path converters, which extract and validate URL segments.

**Common Path Converters:**

- `str`: Matches any non-empty string, excluding the path separator ('/')
- `int`: Matches zero or any positive integer
- `slug`: Matches slug strings (ASCII letters, numbers, hyphens, underscores)
- `uuid`: Matches a UUID string
- `path`: Matches any non-empty string, including the path separator

**Basic Dynamic URL Examples:**

```python
from django.urls import path
from . import views

urlpatterns = [
    # Integer parameter - matches /articles/42/
    path('articles/<int:article_id>/', views.article_detail, name='article_detail'),
    
    # Slug parameter - matches /articles/introduction-to-django/
    path('articles/<slug:slug>/', views.article_detail_by_slug, name='article_detail_by_slug'),
    
    # UUID parameter - matches /orders/123e4567-e89b-12d3-a456-426614174000/
    path('orders/<uuid:order_id>/', views.order_detail, name='order_detail'),
    
    # Multiple parameters - matches /articles/2025/04/django-routing/
    path('articles/<int:year>/<int:month>/<slug:slug>/', 
         views.article_archive_detail, name='article_archive_detail'),
]
```

**Multiple Parameters in Views:**

```python
def article_archive_detail(request, year, month, slug):
    # The parameters from the URL are passed to the view
    article = get_object_or_404(Article, 
                               publish_date__year=year,
                               publish_date__month=month, 
                               slug=slug)
    return render(request, 'blog/article_detail.html', {'article': article})
```

#### Custom Path Converters

For specialized URL patterns, you can create custom path converters:

```python
from django.urls import path, register_converter

class FourDigitYearConverter:
    regex = '[0-9]{4}'
    
    def to_python(self, value):
        return int(value)
    
    def to_url(self, value):
        return f'{value:04d}'

# Register the converter
register_converter(FourDigitYearConverter, 'year4')

# Use the custom converter
urlpatterns = [
    path('articles/<year4:year>/', views.year_archive, name='year_archive'),
]
```

#### Nested Routing

Nested routing refers to organizing URL patterns hierarchically, which is particularly useful for complex applications with multiple related sections.

**Basic Nested Routing with `include()`:**

```python
from django.urls import path, include

urlpatterns = [
    path('blog/', include('blog.urls')),
    path('shop/', include('shop.urls')),
    path('accounts/', include('accounts.urls')),
]
```

**Nested Routing with Namespaces:**

Namespaces help avoid name clashes in URL patterns across different apps:

```python
# Main urls.py
from django.urls import path, include

urlpatterns = [
    path('blog/', include('blog.urls', namespace='blog')),
    path('shop/', include('shop.urls', namespace='shop')),
]

# blog/urls.py
app_name = 'blog'  # Sets the application namespace

urlpatterns = [
    path('', views.article_list, name='article_list'),
    path('article/<slug:slug>/', views.article_detail, name='article_detail'),
    path('category/<slug:category_slug>/', views.category_detail, name='category_detail'),
]
```

These URLs would be reversed as:

```python
reverse('blog:article_detail', kwargs={'slug': 'django-routing'})  # /blog/article/django-routing/
reverse('shop:product_detail', kwargs={'slug': 'django-mug'})  # /shop/product/django-mug/
```

**Multi-level Nested Routing:**

You can include URL patterns at multiple levels:

```python
# Main urls.py
urlpatterns = [
    path('api/', include('api.urls', namespace='api')),
]

# api/urls.py
app_name = 'api'

urlpatterns = [
    path('v1/', include('api.v1.urls', namespace='v1')),
    path('v2/', include('api.v2.urls', namespace='v2')),
]

# api/v1/urls.py
app_name = 'v1'

urlpatterns = [
    path('users/', include('api.v1.users.urls', namespace='users')),
    path('products/', include('api.v1.products.urls', namespace='products')),
]
```

These deeply nested URLs would be reversed as:

```python
reverse('api:v1:users:detail', kwargs={'user_id': 42})  # /api/v1/users/42/
```

#### Dynamic Nested Routes

A common pattern is to have dynamic segments in the URL followed by nested routes:

```python
# shop/urls.py
app_name = 'shop'

urlpatterns = [
    path('', views.product_list, name='product_list'),
    path('products/<slug:product_slug>/', views.product_detail, name='product_detail'),
    
    # Nested URLs under a product
    path('products/<slug:product_slug>/reviews/', include([
        path('', views.product_reviews, name='product_reviews'),
        path('add/', views.add_review, name='add_review'),
        path('<int:review_id>/', views.review_detail, name='review_detail'),
        path('<int:review_id>/edit/', views.edit_review, name='edit_review'),
    ])),
    
    # Nested URLs under a category
    path('categories/<slug:category_slug>/', include([
        path('', views.category_detail, name='category_detail'),
        path('products/', views.category_products, name='category_products'),
        path('subcategories/', views.subcategories, name='subcategories'),
    ])),
]
```

**Real-world Example**: An e-commerce system with nested routing for products, categories, orders, and customer accounts:

````python
# Main urls.py
from django.urls import path, include
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('shop/', include('shop.urls', namespace='shop')),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('api/', include('api.urls', namespace='api')),
]

# shop/urls.py
app_name = 'shop'

urlpatterns = [
    # Main shop pages
    path('', views.ProductListView.as_view(), name='product_list'),
    path('search/', views.ProductSearchView.as_view(), name='product_search'),
    
    # Category hierarchy
    path('categories/', views.CategoryListView.as_view(), name='category_list'),
    path('categories/<slug:slug>/', include([
        path('', views.CategoryDetailView.as_view(), name='category_detail'),
        path('subcategories/', views.SubcategoryListView.as_view(), name='subcategory_list'),
        path('products/', views.CategoryProductsView.as_view(), name='category_products'),
    ])),
    
    # Product details and related functionality
    path('products/<slug:slug>/', include([
        path('', views.ProductDetailView.as_view(), name='product_detail'),
        path('reviews/', include([
            path('', views.ProductReviewListView.as_view(), name='product_reviews'),
            path('add/', views.ProductReviewCreateView.as_view(), name='add_review'),
            path('<int:review_id>/', views.ProductReviewDetailView.as_view(), name='review_detail'),
            path('<int:review_id>/update/', views.ProductReviewUpdateView.as_view(), name='update_review'),
            path('<int:review_id>/delete/', views.ProductReviewDeleteView.as_view(), name='delete_review'),
        ])),
        path('variants/', views.ProductVariantListView.as_view(), name='product_variants'),
        path('related/', views.RelatedProductsView.as_view(), name='related_products'),
    ])),
    
    # Cart an# Django Interview Questions & Answers (6+ Years Experience)

## 1. Django Architecture & Core Concepts

### What is the request/response lifecycle in Django?

The request/response lifecycle in Django follows a well-defined path from the moment a client sends a request until the server returns a response:

1. **HTTP Request**: A client (browser, API client, etc.) sends an HTTP request to the Django server.

2. **URL Routing**: Django's URL dispatcher (URLconf) analyzes the request URL and determines which view function/class should handle it.

3. **Middleware Processing (Request phase)**: Before reaching the view, the request passes through a series of middleware components in the order they're defined in `MIDDLEWARE` setting. Each middleware can modify the request or even return a response early, short-circuiting the process.

4. **View Processing**: The appropriate view function/class receives the request and processes it. This typically involves:
   - Extracting data from the request
   - Interacting with models/database
   - Processing business logic
   - Preparing context data for templates

5. **Template Rendering** (if applicable): The view often loads a template, populates it with context data, and renders it to HTML.

6. **Middleware Processing (Response phase)**: The response travels back through middleware components in reverse order, allowing them to modify the response.

7. **HTTP Response**: Django sends the final HTTP response back to the client.

**Real-world example**: Consider a product detail page on an e-commerce site:
```python
# urls.py
path('products/<int:product_id>/', views.product_detail, name='product_detail')

# views.py
@login_required  # Authentication middleware checks if user is logged in
def product_detail(request, product_id):
    # View fetches product from database
    product = get_object_or_404(Product, id=product_id)
    
    # Log this view for analytics (custom middleware might track this)
    request.session['last_viewed_product'] = product_id
    
    # Render template with context
    return render(request, 'products/detail.html', {'product': product})
````

In this flow, middleware might handle authentication, session management, and CSRF protection before the view processes the request, then compression and caching might be applied to the response.

### How does Django's MTV architecture differ from MVC?

Django follows an architectural pattern called MTV (Model-Template-View), which is conceptually similar to but differs in terminology from the traditional MVC (Model-View-Controller) pattern:

|MVC Component|Django MTV Equivalent|Responsibility|
|---|---|---|
|Model|Model|Data structure and database interactions|
|View|Template|Presentation and display logic|
|Controller|View|Business logic and request handling|

The key differences:

1. **Django's View** handles what a traditional Controller would do - processing requests, applying business logic, and determining what data to display.
    
2. **Django's Template** corresponds to the traditional View - it defines how the data should be presented.
    
3. **Django's Model** is largely the same as in MVC - it defines the data structure and handles database interactions.
    

The confusion often arises because Django's "View" is essentially a "Controller" in traditional MVC terminology. Django's creators chose this naming to better reflect their specific implementation of the pattern.

**Real-world example**:

```python
# Model - defines data structure and database interactions
class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    publication_date = models.DateTimeField(auto_now_add=True)
    
    def get_absolute_url(self):
        return reverse('article_detail', args=[self.id])

# View (Controller in MVC) - handles business logic
def article_detail(request, article_id):
    article = get_object_or_404(Article, id=article_id)
    return render(request, 'blog/article_detail.html', {'article': article})

# Template (View in MVC) - presentation logic
# article_detail.html
<article>
    <h1>{{ article.title }}</h1>
    <time>{{ article.publication_date|date:"F j, Y" }}</time>
    <div class="content">{{ article.content|linebreaks }}</div>
</article>
```

### How does Django handle routing internally?

Django's URL routing system is a crucial part of its request handling process. Here's how it works internally:

1. **URLconf Loading**: When Django starts, it loads the root URLconf module specified in the `ROOT_URLCONF` setting (typically `project_name.urls`).
    
2. **URL Pattern Compilation**: Django compiles all URL patterns into regular expressions when the server starts, for efficient matching later on.
    
3. **Request Processing**:
    
    - When a request comes in, Django removes the domain name and leading slash.
    - It tries to match the remaining URL path against each pattern in the URLconf in order.
    - The first matching pattern stops the search.
4. **View Resolution**:
    
    - Once a match is found, Django calls the associated view function with:
        - The `HttpRequest` object
        - Any captured values from the URL as positional or keyword arguments
        - Any additional arguments specified in the URL pattern
5. **Include Mechanism**: The `include()` function allows for modular URL configurations by including URL patterns from other URLconf modules.
    
6. **Namespace System**: Django provides a namespace system to disambiguate URL names across applications using the `app_name` variable and the `namespace` parameter to `include()`.
    

**Real-world example**:

```python
# Main urls.py
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', include('blog.urls', namespace='blog')),
    path('shop/', include('shop.urls', namespace='shop')),
]

# blog/urls.py
app_name = 'blog'
urlpatterns = [
    path('', views.article_list, name='article_list'),
    path('article/<int:article_id>/', views.article_detail, name='article_detail'),
    path('category/<slug:category_slug>/', views.category_detail, name='category_detail'),
]
```

When Django processes a request to `/blog/article/42/`:

1. It matches the `blog/` prefix and forwards the rest to `blog.urls`.
2. In `blog.urls`, it matches `article/<int:article_id>/` with `article_id=42`.
3. It calls `views.article_detail(request, article_id=42)`.

The URL name can be referenced as `blog:article_detail` in templates or code.

### Explain middleware and how to create a custom middleware.

Middleware in Django is a framework of hooks into Django's request/response processing. It's a lightweight, low-level "plugin" system for globally altering Django's input or output.

**Middleware Key Characteristics**:

- Executes during request/response cycle, not during Django initialization
- Processes all requests/responses that pass through the system
- Ordered by the `MIDDLEWARE` setting
- Request phase: processes from top to bottom
- Response phase: processes from bottom to top

**Creating Custom Middleware**:

Django supports two styles of middleware:

1. **Function-based middleware**
2. **Class-based middleware**

**Class-based Middleware Example**:

```python
class RequestTimeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time configuration and initialization

    def __call__(self, request):
        # Code to be executed for each request before the view (and later middleware) are called
        start_time = time.time()
        
        # Process the request
        response = self.get_response(request)
        
        # Code to be executed for each request/response after the view is called
        duration = time.time() - start_time
        response['X-Request-Duration'] = f"{duration:.2f}s"
        
        # Log slow requests
        if duration > 1.0:
            logger.warning(f"Slow request: {request.path} took {duration:.2f}s")
            
        return response
    
    # Optional method for view exception processing
    def process_exception(self, request, exception):
        # Log the error
        logger.error(f"Exception in {request.path}: {exception}")
        return None  # Let Django's exception handling take over
```

**Function-based Middleware Example**:

```python
def simple_middleware(get_response):
    # One-time configuration and initialization
    
    def middleware(request):
        # Code to be executed for each request before the view (and later middleware) are called
        
        response = get_response(request)
        
        # Code to be executed for each request/response after the view is called
        
        return response
    
    return middleware
```

**Real-world Example**: A middleware that tracks and limits API usage by IP address.

```python
class RateLimitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limits = {}
        self.window_size = 3600  # 1 hour window
        self.max_requests = 100  # 100 requests per hour

    def __call__(self, request):
        # Only apply rate limiting to API requests
        if request.path.startswith('/api/'):
            ip = self.get_client_ip(request)
            
            # Get or initialize the rate limit record for this IP
            if ip not in self.rate_limits:
                self.rate_limits[ip] = {'count': 0, 'reset_time': time.time() + self.window_size}
            
            # Check if the window has reset
            if time.time() > self.rate_limits[ip]['reset_time']:
                self.rate_limits[ip] = {'count': 0, 'reset_time': time.time() + self.window_size}
            
            # Increment request count
            self.rate_limits[ip]['count'] += 1
            
            # Check if limit exceeded
            if self.rate_limits[ip]['count'] > self.max_requests:
                return JsonResponse(
                    {'error': 'Rate limit exceeded. Try again later.'},
                    status=429
                )
            
            # Add rate limit headers
            response = self.get_response(request)
            response['X-Rate-Limit-Limit'] = str(self.max_requests)
            response['X-Rate-Limit-Remaining'] = str(self.max_requests - self.rate_limits[ip]['count'])
            response['X-Rate-Limit-Reset'] = str(int(self.rate_limits[ip]['reset_time']))
            return response
        
        return self.get_response(request)
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
```

To enable this middleware, add it to the `MIDDLEWARE` setting in your Django project:

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    # ...
    'path.to.RateLimitMiddleware',
    # ...
]
```

## 2. Models & ORM

### How does Django ORM translate Python code into SQL?

Django's Object-Relational Mapping (ORM) system translates Python code into SQL through a complex but elegant process:

1. **Model Definition**: When you define a Django model, you're creating a Python class that inherits from `django.db.models.Model` with attributes that represent database fields.
    
2. **Query Construction**: When you write a query using the ORM, Django constructs a `QuerySet` object. This object is lazy – it doesn't execute the query immediately.
    
3. **Query Compilation**: When the `QuerySet` is evaluated (e.g., when you iterate over it, call `list()` on it, or slice it), Django's query compiler converts it to SQL:
    
    - Django determines the required tables and joins
    - It analyzes the conditions (filters) and converts them to WHERE clauses
    - It processes annotations, aggregations, order_by statements, etc.
4. **SQL Generation**: The compiled query is converted to SQL specific to your database backend using Django's database-specific operations.
    
5. **Query Execution**: The generated SQL is sent to the database for execution.
    
6. **Result Processing**: Database results are converted back into model instances.
    

**Real-world Example**:

```python
# Python model definition
class Customer(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class Order(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='orders')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('shipped', 'Shipped'),
        ('delivered', 'Delivered'),
    ])

# Python query
recent_paid_orders = Order.objects.filter(
    status='paid',
    created_at__gte=datetime.datetime.now() - datetime.timedelta(days=30)
).select_related('customer').order_by('-created_at')
```

When executed, this Python code gets translated to SQL similar to:

```sql
SELECT 
    "orders"."id", "orders"."customer_id", "orders"."total_amount", 
    "orders"."created_at", "orders"."status",
    "customer"."id", "customer"."name", "customer"."email", "customer"."created_at" 
FROM "orders" 
INNER JOIN "customer" ON ("orders"."customer_id" = "customer"."id") 
WHERE ("orders"."status" = 'paid' AND "orders"."created_at" >= '2025-03-31 12:30:45.123456') 
ORDER BY "orders"."created_at" DESC;
```

You can see the actual SQL generated by Django by using:

```python
print(recent_paid_orders.query)
```

### How do you optimize ORM queries for performance?

Optimizing Django ORM queries is crucial for application performance. Here are detailed techniques with examples:

#### 1. Use `select_related()` and `prefetch_related()` to avoid N+1 queries

```python
# Bad: Causes N+1 queries
orders = Order.objects.all()
for order in orders:
    print(order.customer.name)  # Each access triggers a new query

# Good: Just 1 query with a JOIN
orders = Order.objects.select_related('customer')
for order in orders:
    print(order.customer.name)  # No additional query
```

#### 2. Only select the fields you need

```python
# Fetches all fields
users = User.objects.all()

# More efficient: fetches only needed fields
users = User.objects.only('username', 'email', 'last_login')

# Alternative approach
users = User.objects.values('username', 'email', 'last_login')
```

#### 3. Use `values()` or `values_list()` when you don't need model instances

```python
# Returns model instances
products = Product.objects.filter(category='electronics')

# Returns dictionaries - more efficient when you just need data
product_data = Product.objects.filter(category='electronics').values('name', 'price')

# Returns tuples - even more efficient
product_tuples = Product.objects.filter(category='electronics').values_list('name', 'price')

# For a single field, you can flatten the result
product_names = Product.objects.filter(category='electronics').values_list('name', flat=True)
```

#### 4. Use database functions for computation

```python
from django.db.models import F, Sum, Count, Avg
from django.db.models.functions import Coalesce

# Calculate in database instead of Python
Order.objects.update(
    total_price=F('price') * F('quantity') * (1 - F('discount_rate'))
)

# Aggregate in database
report = Order.objects.values('customer_id').annotate(
    order_count=Count('id'),
    total_spent=Sum('total_price'),
    avg_order_value=Avg('total_price')
)
```

#### 5. Use `iterator()` for large querysets

```python
# Memory-efficient processing of large querysets
for product in Product.objects.filter(active=True).iterator():
    # Process each product without loading all into memory
    process_product(product)
```

#### 6. Use indexed fields in filters

```python
# Add indexes to fields used frequently in filtering, ordering or joining
class Customer(models.Model):
    email = models.EmailField(db_index=True)
    last_active = models.DateTimeField(db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['country', 'city']),  # Compound index
            models.Index(fields=['membership_type', '-join_date']),
        ]
```

#### 7. Use `exists()` instead of `count()` or `len()` to check existence

```python
# Inefficient
if User.objects.filter(email=email).count() > 0:
    # User exists

# More efficient
if User.objects.filter(email=email).exists():
    # User exists
```

#### 8. Use `bulk_create()` and `bulk_update()` for batch operations

```python
# Inefficient: N queries
for data in dataset:
    Product.objects.create(name=data['name'], price=data['price'])

# Efficient: 1 query
products = [
    Product(name=data['name'], price=data['price']) 
    for data in dataset
]
Product.objects.bulk_create(products)

# Similarly for updates
Product.objects.bulk_update(products, ['price', 'stock'])
```

#### 9. Consider raw SQL for very complex queries

```python
from django.db import connection

def complex_analytics_query():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                p.category, 
                SUM(oi.quantity * oi.price) as revenue,
                COUNT(DISTINCT o.customer_id) as customer_count
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.created_at > %s
            GROUP BY p.category
            HAVING SUM(oi.quantity) > 100
            ORDER BY revenue DESC
        """, [three_months_ago])
        return cursor.fetchall()
```

#### 10. Use query caching mechanisms

```python
from django.core.cache import cache

def get_active_promotions():
    cache_key = 'active_promotions'
    promotions = cache.get(cache_key)
    
    if promotions is None:
        promotions = list(Promotion.objects.filter(
            is_active=True,
            start_date__lte=timezone.now(),
            end_date__gte=timezone.now()
        ).select_related('product'))
        
        # Cache for 10 minutes
        cache.set(cache_key, promotions, 60 * 10)
    
    return promotions
```

#### 11. Use `defer()` to exclude unnecessary large fields

```python
# Skip loading large text fields when not needed
articles = Article.objects.defer('content', 'metadata').all()
```

#### 12. Use `QuerySet.explain()` to analyze query execution plans (Django 3.0+)

```python
queryset = Order.objects.filter(
    status='processing',
    created_at__gt=last_month
).select_related('customer')

# Print the execution plan
print(queryset.explain())
```

**Real-world Optimization Example**: An e-commerce dashboard that displays sales stats without bogging down the database:

```python
def get_sales_dashboard_data(start_date, end_date):
    # Cache key includes the date range
    cache_key = f'sales_dashboard:{start_date.isoformat()}:{end_date.isoformat()}'
    dashboard_data = cache.get(cache_key)
    
    if dashboard_data is None:
        # Get completed orders in date range
        orders = Order.objects.filter(
            status='completed',
            completed_at__range=(start_date, end_date)
        )
        
        # Calculate revenue and stats in the database
        order_stats = orders.aggregate(
            total_revenue=Sum('total_amount'),
            order_count=Count('id'),
            avg_order_value=Avg('total_amount')
        )
        
        # Get top products efficiently
        top_products = OrderItem.objects.filter(
            order__in=orders
        ).values(
            'product_id', 'product__name'
        ).annotate(
            total_sold=Sum('quantity'),
            revenue=Sum(F('price') * F('quantity'))
        ).order_by('-revenue')[:10]
        
        # Get daily revenue for chart
        daily_revenue = orders.annotate(
            date=TruncDate('completed_at')
        ).values('date').annotate(
            revenue=Sum('total_amount')
        ).order_by('date')
        
        dashboard_data = {
            'order_stats': order_stats,
            'top_products': list(top_products),
            'daily_revenue': list(daily_revenue)
        }
        
        # Cache for 1 hour
        cache.set(cache_key, dashboard_data, 60 * 60)
    
    return dashboard_data
```

### What's the difference between select_related() and prefetch_related()?

Both `select_related()` and `prefetch_related()` are Django ORM methods to optimize database queries by reducing the number of database hits, but they work differently and are suitable for different relationship types:

#### `select_related()`

- **Usage**: For foreign key and one-to-one relationships (where the related object is on the "one" side)
- **Mechanism**: Performs a SQL JOIN and includes the fields of the related object in the SELECT statement
- **Query Count**: Uses a single database query
- **Performance Impact**: Best for "to-one" relationships where you need data from both the model and its related model

**Example**:

```python
# Without select_related - 2 queries
order = Order.objects.get(id=1)  # First query
customer = order.customer  # Second query to fetch the customer

# With select_related - 1 query with JOIN
order = Order.objects.select_related('customer').get(id=1)
customer = order.customer  # No additional query - data already loaded
```

Generated SQL (simplified):

```sql
SELECT 
    orders.id, orders.date, orders.total, /* other order fields */
    customers.id, customers.name, /* other customer fields */
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id
WHERE orders.id = 1;
```

#### `prefetch_related()`

- **Usage**: For many-to-many relationships and reverse foreign key relationships (where the related objects are on the "many" side)
- **Mechanism**: Performs separate queries for each relationship and joins the results in Python
- **Query Count**: Uses multiple queries (one for the main model, one for each prefetched relationship)
- **Performance Impact**: Best for "to-many" relationships where you need to access multiple related objects

**Example**:

```python
# Without prefetch_related - N+1 queries
product = Product.objects.get(id=1)  # First query
categories = product.categories.all()  # Second query for categories
for category in categories:
    print(category.name)

# With prefetch_related - 2 queries
product = Product.objects.prefetch_related('categories').get(id=1)
categories = product.categories.all()  # No additional query
for category in categories:
    print(category.name)  # No additional queries
```

Generated SQL (simplified):

```sql
-- First query fetches the product
SELECT id, name, /* other product fields */ FROM products WHERE id = 1;

-- Second query fetches all related categories
SELECT c.id, c.name, /* other category fields */, pc.product_id 
FROM categories c
INNER JOIN product_categories pc ON c.id = pc.category_id
WHERE pc.product_id IN (1);
```

#### Complex Relationships and Chaining

Both methods can be chained and combined:

```python
# Combining both techniques
orders = Order.objects.select_related('customer').prefetch_related('items__product')

# This efficiently loads:
# 1. Orders
# 2. The customer for each order (via JOIN)
# 3. The items for each order (via separate query)
# 4. The product for each item (via separate query)
```

#### Nested Relationships

Both can traverse multi-level relationships:

```python
# Select related can traverse foreign keys
Order.objects.select_related('customer__address__country')

# Prefetch related can traverse any relationship
Product.objects.prefetch_related(
    'categories',
    'reviews__user',
    Prefetch('variants', queryset=Variant.objects.filter(in_stock=True))
)
```

#### Real-world Example: An e-commerce order detail view

```python
def order_detail(request, order_id):
    # Efficiently fetch the order with all related data in minimal queries
    order = Order.objects.select_related(
        'customer',  # Foreign key - uses JOIN
        'shipping_address',  # Foreign key - uses JOIN
        'billing_address'  # Foreign key - uses JOIN
    ).prefetch_related(
        'items__product',  # Reverse FK + FK - separate queries
        'items__product__categories',  # M2M after FK chain - separate query
        'payment_transactions',  # Reverse FK - separate query
        Prefetch(
            'status_updates',  # Custom prefetch for filtered relationship
            queryset=OrderStatusUpdate.objects.select_related('user').order_by('-timestamp'),
            to_attr='history'
        )
    ).get(id=order_id)
    
    # Now we can access all these related objects without additional queries
    context = {
        'order': order,
        'customer': order.customer,
        'address': order.shipping_address,
        'items': order.items.all(),  # No query
        'payment_history': order.payment_transactions.all(),  # No query
        'status_history': order.history  # From prefetch to_attr
    }
    
    return render(request, 'orders/detail.html', context)
```

### How do you perform raw SQL queries in Django, and when should you use them?

Django provides several ways to execute raw SQL queries when the ORM doesn't provide the flexibility or performance you need:

#### 1. Using `Manager.raw()` method

The `raw()` method executes a raw SQL query and returns a `RawQuerySet` of model instances:

```python
# Simple raw query
products = Product.objects.raw('SELECT * FROM products WHERE price > %s', [100])

# More complex raw query with joins
customers = Customer.objects.raw('''
    SELECT c.id, c.name, c.email, COUNT(o.id) as order_count
    FROM customers c
    LEFT JOIN orders o ON c.id = o.customer_id
    WHERE c.is_active = True
    GROUP BY c.id
    HAVING COUNT(o.id) > 5
    ORDER BY order_count DESC
''')

# Accessing results
for customer in customers:
    print(customer.name, customer.order_count)  # Note: order_count is dynamically added
```

**Important considerations**:

- You must include the primary key column in your query
- Django maps the query results to model instances
- You can map extra SELECT fields to model attributes
- Parameters should be passed as a list to prevent SQL injection

#### 2. Using `connection.cursor()` for complete control

For queries that don't map to models or for non-SELECT operations:

```python
from django.db import connection

def get_product_sales_report():
    with connection.cursor() as cursor:
        cursor.execute('''
            SELECT 
                p.name, 
                p.sku, 
                SUM(oi.quantity) as units_sold,
                SUM(oi.quantity * oi.price) as revenue
            FROM products p
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.status = 'completed'
            AND o.completion_date > %s
            GROUP BY p.id, p.name, p.sku
            ORDER BY revenue DESC
        ''', [three_months_ago])
        
        # Convert results to dictionaries
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
```

#### 3. Using `connection.execute()` method (Django 4.0+)

```python
from django.db import connection

def update_product_prices(category_id, increase_percentage):
    with connection.execute('''
        UPDATE products
        SET price = price * (1 + %s/100)
        WHERE category_id = %s
    ''', [increase_percentage, category_id]) as cursor:
        return cursor.rowcount  # Number of rows affected
```

#### 4. Using database-specific operations with `QuerySet.annotate()`

Django 3.2+ allows using database functions directly:

```python
from django.db.models import F, Value
from django.db.models.functions import Cast
from django.db.models.expressions import RawSQL

# Using RawSQL within a queryset
Product.objects.annotate(
    distance=RawSQL(
        "ST_Distance(location, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
        (longitude, latitude)
    )
).order_by('distance')
```

#### When to use raw SQL:

1. **Complex queries beyond ORM capabilities**:
    
    - Advanced window functions
    - Complex subqueries
    - Hierarchical/recursive queries (CTE)
    - Advanced geospatial queries
2. **Performance optimization**:
    
    - When ORM-generated queries are inefficient
    - For queries manipulating large datasets
    - When you need database-specific optimizations
3. **Bulk operations**:
    
    - Mass updates with complex conditions
    - Specialized batch processing
4. **Database-specific features**:
    
    - Using features specific to your database like PostgreSQL's JSONB operations
5. **Schema migration operations**:
    
    - Custom, complex schema changes

**Real-world Example**: A geospatial search with complex filtering:

```python
def find_nearby_restaurants(latitude, longitude, radius_km, cuisine=None, min_rating=None):
    query = '''
        SELECT 
            r.id, r.name, r.address, r.rating,
            ST_Distance(
                r.location, 
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            ) * 111.32 AS distance_km
        FROM restaurants r
        LEFT JOIN restaurant_cuisines rc ON r.id = rc.restaurant_id
        LEFT JOIN cuisines c ON rc.cuisine_id = c.id
        WHERE ST_DWithin(
            r.location, 
            ST_SetSRID(ST_MakePoint(%s, %s), 4326),
            %s / 111.32
        )
    '''
    
    params = [longitude, latitude, longitude, latitude, radius_km]
    
    if cuisine:
        query += " AND c.name = %s"
        params.append(cuisine)
    
    if min_rating:
        query += " AND r.rating >= %s"
        params.append(min_rating)
    
    query += " GROUP BY r.id, r.name, r.address, r.rating, r.location ORDER BY distance_km"
    
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ],
                message='Username can only contain letters, numbers, and underscores'
            )
        ]
    )
    
    password = forms.CharField(
        widget=forms.PasswordInput,
        validators=[
            MinLengthValidator(8, 'Password must be at least 8 characters'),
            RegexValidator(
                regex=r'[A-Z]',
                message='Password must contain at least one uppercase letter'
            ),
            RegexValidator(
                regex=r'[0-9]',
                message='Password must contain at least one number'
            )
        ]
    )
    
    email = forms.EmailField(validators=[EmailValidator(message='Enter a valid email address')])
    age = forms.IntegerField(min_value=18, max_value=120)  # Built-in min/max validation
```

##### 1.2 Custom Field Validators as Functions

You can create reusable validator functions for specific validation needs:

```python
from django import forms
from django.core.exceptions import ValidationError

def validate_domain_email(value):
    """Validate that the email belongs to the company domain."""
    if not value.endswith('@company.com'):
        raise ValidationError('Email must be from company.com domain')

def validate_even(value):
    """Validate that the value is an even number."""
    if value % 2 != 0:
        raise ValidationError('This field must be an even number')

class EmployeeForm(forms.Form):
    email = forms.EmailField(validators=[validate_domain_email])
    employee_id = forms.IntegerField(validators=[validate_even])
```

##### 1.3 Field-specific Clean Methods

For complex field-specific validation, you can define custom `clean_<fieldname>` methods:

```python
class RegistrationForm(forms.Form):
    username = forms.CharField(max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)
    birth_year = forms.IntegerField()
    
    def clean_username(self):
        """Validate that the username is not already taken."""
        username = self.cleaned_data['username']
        
        # Check if username exists in database
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError('This username is already taken')
        
        # Check for reserved words
        reserved_names = ['admin', 'root', 'superuser']
        if username.lower() in reserved_names:
            raise forms.ValidationError('This username is reserved')
            
        return username  # Always return the cleaned value
    
    def clean_birth_year(self):
        """Validate that the user is at least 18 years old."""
        birth_year = self.cleaned_data['birth_year']
        current_year = timezone.now().year
        
        if current_year - birth_year < 18:
            raise forms.ValidationError('You must be at least 18 years old')
            
        return birth_year
```

#### 2. Form-Level Validation

Form-level validation allows you to validate multiple fields together or implement validation rules that depend on the relationship between different fields.

##### 2.1 Custom Clean Method

For validations that involve multiple fields, override the form's `clean()` method:

```python
class PasswordChangeForm(forms.Form):
    old_password = forms.CharField(widget=forms.PasswordInput)
    new_password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)
    
    def clean(self):
        """Validate that the new password is different from the old one
        and that confirmation matches."""
        cleaned_data = super().clean()
        
        old_password = cleaned_data.get('old_password')
        new_password = cleaned_data.get('new_password')
        confirm_password = cleaned_data.get('confirm_password')
        
        # Only proceed if individual fields passed their validation
        if old_password and new_password and confirm_password:
            # Check if new password is same as old
            if old_password == new_password:
                # Add error to specific field
                self.add_error('new_password', 'New password must be different from old password')
            
            # Check if confirmation matches
            if new_password != confirm_password:
                # Add error to specific field
                self.add_error('confirm_password', 'Password confirmation does not match')
                
            # Extra validation: password complexity
            if len(new_password) < 8:
                self.add_error('new_password', 'Password must be at least 8 characters long')
            
            if not any(char.isdigit() for char in new_password):
                self.add_error('new_password', 'Password must contain at least one number')
        
        return cleaned_data
```

##### 2.2 ValidationError with Non-field Errors

Sometimes the error doesn't apply to a specific field, but to the form as a whole:

```python
class ContactAvailabilityForm(forms.Form):
    contact_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    contact_time = forms.TimeField(widget=forms.TimeInput(attrs={'type': 'time'}))
    
    def clean(self):
        cleaned_data = super().clean()
        contact_date = cleaned_data.get('contact_date')
        contact_time = cleaned_data.get('contact_time')
        
        if contact_date and contact_time:
            # Combine date and time to create a datetime object
            contact_datetime = datetime.combine(contact_date, contact_time)
            
            # Check if date is in the past
            if contact_datetime < timezone.now():
                # Add a non-field error
                raise forms.ValidationError(
                    "You cannot schedule a contact time in the past"
                )
            
            # Check if it's within business hours (9 AM to 5 PM)
            if contact_time.hour < 9 or contact_time.hour >= 17:
                raise forms.ValidationError(
                    "Contact time must be during business hours (9 AM to 5 PM)"
                )
            
            # Check if it's on a weekend
            if contact_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                raise forms.ValidationError(
                    "We're not available on weekends. Please select a weekday."
                )
                
        return cleaned_data
```

#### 3. ModelForm Validation

ModelForms can leverage model-defined validation and add form-specific validation.

##### 3.1 Model-level Validation

First, the model can define validation rules:

```python
from django.db import models
from django.core.exceptions import ValidationError
from django.utils import timezone

def validate_future_date(value):
    if value < timezone.now().date():
        raise ValidationError('Date cannot be in the past')

class Event(models.Model):
    title = models.CharField(max_length=200)
    location = models.CharField(max_length=200)
    event_date = models.DateField(validators=[validate_future_date])
    max_attendees = models.PositiveIntegerField()
    
    def clean(self):
        """Model-level validation."""
        if self.title and self.location and self.title == self.location:
            raise ValidationError('Event title cannot be the same as the location')
        
        # Validate date is at least a week from now for new events
        if not self.pk and self.event_date:  # Check if new event (no primary key)
            min_date = timezone.now().date() + timezone.timedelta(days=7)
            if self.event_date < min_date:
                raise ValidationError({
                    'event_date': 'New events must be scheduled at least a week in advance'
                })
```

##### 3.2 ModelForm with Custom Validation

Then the form can build on the model validation and add more validation rules:

```python
from django import forms
from .models import Event

class EventForm(forms.ModelForm):
    # Add additional fields not in the model
    confirmation_email = forms.EmailField(required=False)
    terms_accepted = forms.BooleanField(required=True)
    
    class Meta:
        model = Event
        fields = ['title', 'location', 'event_date', 'max_attendees']
        widgets = {
            'event_date': forms.DateInput(attrs={'type': 'date'})
        }
    
    def clean_title(self):
        """Field-level validation in a ModelForm."""
        title = self.cleaned_data['title']
        
        # Check for reserved words
        reserved_words = ['cancelled', 'postponed', 'test']
        for word in reserved_words:
            if word in title.lower():
                raise forms.ValidationError(f'Title cannot contain the word "{word}"')
        
        return title
    
    def clean(self):
        """Form-level validation in a ModelFormdef article_detail(request, pk):
    article = get_object_or_404(Article, pk=pk)
    return render(request, 'blog/article_detail.html', {'article': article})
```

With DetailView:

```python
class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    context_object_name = 'article'
```

**3. CreateView - Create a new object**

Before generic views:

```python
def create_article(request):
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save(commit=False)
            article.author = request.user
            article.save()
            return redirect('article_detail', pk=article.pk)
    else:
        form = ArticleForm()
    return render(request, 'blog/article_form.html', {'form': form})
```

With CreateView:

```python
class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
    
    def get_success_url(self):
        return reverse('article_detail', kwargs={'pk': self.object.pk})
```

**4. UpdateView - Update an existing object**

Before generic views:

```python
def update_article(request, pk):
    article = get_object_or_404(Article, pk=pk)
    if request.method == 'POST':
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            form.save()
            return redirect('article_detail', pk=article.pk)
    else:
        form = ArticleForm(instance=article)
    return render(request, 'blog/article_form.html', {'form': form})
```

With UpdateView:

```python
class ArticleUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user
    
    def get_success_url(self):
        return reverse('article_detail', kwargs={'pk': self.object.pk})
```

**5. DeleteView - Delete an existing object**

Before generic views:

```python
def delete_article(request, pk):
    article = get_object_or_404(Article, pk=pk)
    if request.method == 'POST':
        article.delete()
        return redirect('article_list')
    return render(request, 'blog/article_confirm_delete.html', {'article': article})
```

With DeleteView:

```python
class ArticleDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Article
    template_name = 'blog/article_confirm_delete.html'
    success_url = reverse_lazy('article_list')
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user
```

#### Customizing Generic Views

While generic views handle many common cases out of the box, they are designed to be customizable:

**1. Customizing Query Sets**

```python
class PublishedArticleListView(ListView):
    model = Article
    template_name = 'blog/article_list.html'
    context_object_name = 'articles'
    
    def get_queryset(self):
        return Article.objects.filter(status='published').order_by('-published_date')
```

**2. Adding Extra Context Data**

```python
class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comments'] = self.object.comments.all()
        context['related_articles'] = Article.objects.filter(
            category=self.object.category
        ).exclude(id=self.object.id)[:5]
        return context
```

**3. Custom Form Processing**

```python
class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    
    def form_valid(self, form):
        # Customize before saving
        form.instance.author = self.request.user
        form.instance.status = 'draft'
        
        # Save the object
        response = super().form_valid(form)
        
        # Customize after saving
        if form.cleaned_data.get('notify_subscribers'):
            self.object.notify_subscribers()
            
        return response
```

**4. Custom URL Parameters**

```python
class CategoryArticleListView(ListView):
    model = Article
    template_name = 'blog/category_articles.html'
    
    def get_queryset(self):
        self.category = get_object_or_404(Category, slug=self.kwargs['slug'])
        return Article.objects.filter(category=self.category)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['category'] = self.category
        return context
```

**Real-world Example**: A complete blog application using generic views:

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('category_detail', kwargs={'slug': self.slug})

class Article(models.Model):
    STATUS_CHOICES = (
        ('draft', 'Draft'),
        ('published', 'Published'),
    )
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='articles')
    content = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='articles')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_date = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('article_detail', kwargs={'slug': self.slug})

class Comment(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Comment by {self.author.username} on {self.article.title}"

# views.py
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.views.generic.dates import YearArchiveView, MonthArchiveView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse_lazy
from .models import Article, Category, Comment
from .forms import ArticleForm, CommentForm

class ArticleListView(ListView):
    model = Article
    template_name = 'blog/article_list.html'
    context_object_name = 'articles'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Article.objects.filter(status='published').order_by('-published_date')
        
        # Filter by category if provided
        category_slug = self.request.GET.get('category')
        if category_slug:
            queryset = queryset.filter(category__slug=category_slug)
            
        # Filter by search query if provided
        search_query = self.request.GET.get('q')
        if search_query:
            queryset = queryset.filter(
                Q(title__icontains=search_query) | 
                Q(content__icontains=search_query)
            )
            
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        context['search_query'] = self.request.GET.get('q', '')
        context['category_filter'] = self.request.GET.get('category', '')
        return context

class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/article_detail.html'
    context_object_name = 'article'
    slug_url_kwarg = 'slug'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['comment_form'] = CommentForm()
        context['comments'] = self.object.comments.all().order_by('-created_at')
        context['related_articles'] = Article.objects.filter(
            category=self.object.category, 
            status='published'
        ).exclude(id=self.object.id)[:3]
        return context
    
    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form = CommentForm(request.POST)
        
        if form.is_valid() and request.user.is_authenticated:
            comment = form.save(commit=False)
            comment.article = self.object
            comment.author = request.user
            comment.save()
            return redirect(self.object.get_absolute_url())
            
        context = self.get_context_data(object=self.object)
        context['comment_form'] = form
        return self.render_to_response(context)

class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

class ArticleUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Article
    form_class = ArticleForm
    template_name = 'blog/article_form.html'
    slug_url_kwarg = 'slug'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user or self.request.user.is_staff

class ArticleDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Article
    template_name = 'blog/article_confirm_delete.html'
    success_url = reverse_lazy('article_list')
    slug_url_kwarg = 'slug'
    
    def test_func(self):
        article = self.get_object()
        return article.author == self.request.user or self.request.user.is_staff

class CategoryDetailView(DetailView):
    model = Category
    template_name = 'blog/category_detail.html'
    context_object_name = 'category'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['articles'] = self.object.articles.filter(
            status='published'
        ).order_by('-published_date')
        return context

class ArticleYearArchiveView(YearArchiveView):
    model = Article
    date_field = 'published_date'
    make_object_list = True
    template_name = 'blog/article_archive_year.html'
    
    def get_queryset(self):
        return Article.objects.filter(status='published')

class ArticleMonthArchiveView(MonthArchiveView):
    model = Article
    date_field = 'published_date'
    template_name = 'blog/article_archive_month.html'
    
    def get_queryset(self):
        return Article.objects.filter(status='published')

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ArticleListView.as_view(), name='article_list'),
    path('article/<slug:slug>/', views.ArticleDetailView.as_view(), name='article_detail'),
    path('article/create/', views.ArticleCreateView.as_view(), name='article_create'),
    path('article/<slug:slug>/update/', views.ArticleUpdateView.as_view(), name='article_update'),
    path('article/<slug:slug>/delete/', views.ArticleDeleteView.as_view(), name='article_delete'),
    path('category/<slug:slug>/', views.CategoryDetailView.as_view(), name='category_detail'),
    path('archive/<int:year>/', views.ArticleYearArchiveView.as_view(), name='article_year_archive'),
    path('archive/<int:year>/<int:month>/', views.ArticleMonthArchiveView.as_view(), name='article_month_archive'),
]
```

This example demonstrates how generic views:

- Eliminate repetitive code for common operations
- Provide a consistent structure across views
- Allow for customization where needed
- Enable quick addition of features like pagination and filtering
- Support complex functionality through mixins and inheritance

Generic views are most beneficial when you're implementing standard CRUD operations and want to maintain consistent behaviors, but they remain flexible enough to handle custom business logic when needed.

### How does Django's reverse() function work?

Django's `reverse()` function is a powerful URL resolution tool that dynamically generates URLs from URL patterns defined in your URLconf. This is crucial for maintaining DRY (Don't Repeat Yourself) principles by avoiding hardcoded URLs throughout your codebase.

#### Basic Functionality

At its core, `reverse()` takes a URL pattern name and optional arguments, then returns the corresponding URL path:

```python
from django.urls import reverse

# Basic usage
article_url = reverse('article_detail', args=[42])  # Returns '/articles/42/'
```

#### How It Works Internally

1. **Pattern Lookup**: Django searches all URL patterns across all included URLconfs for a pattern with the given name.
    
2. **Pattern Matching**: Once found, Django uses the pattern's regular expression to construct a URL.
    
3. **Argument Substitution**:
    
    - Positional arguments from `args` are inserted into the pattern in order
    - Named arguments from `kwargs` are matched to named groups in the pattern
4. **URL Construction**: The final URL string is assembled, including the prefix from any parent URLconfs.
    

#### Usage Patterns

**1. Basic URL Reversal**

```python
# URL definition
path('articles/<int:article_id>/', views.article_detail, name='article_detail')

# In a view
def some_view(request):
    url = reverse('article_detail', args=[42])  # '/articles/42/'
    return redirect(url)
```

**2. Using Named Arguments**

```python
# URL definition
path('articles/<int:year>/<int:month>/<slug:slug>/', views.article_detail, name='article_detail')

# In a view
def some_view(request):
    url = reverse('article_detail', kwargs={
        'year': 2025,
        'month': 4,
        'slug': 'django-reverse-explained'
    })  # '/articles/2025/4/django-reverse-explained/'
    return redirect(url)
```

**3. URL Namespaces**

Django supports URL namespaces to avoid name clashes between apps:

```python
# In main urls.py
path('blog/', include('blog.urls', namespace='blog'))
path('news/', include('news.urls', namespace='news'))

# In a view
def some_view(request):
    blog_url = reverse('blog:article_detail', args=[42])  # '/blog/articles/42/'
    news_url = reverse('news:article_detail', args=[42])  # '/news/articles/42/'
    return render(request, 'template.html', {'blog_url': blog_url, 'news_url': news_url})
```

**4. Reversing in Templates**

In Django templates, you can use the `url` template tag:

```html
<a href="{% url 'article_detail' article.id %}">{{ article.title }}</a>

<!-- With namespace -->
<a href="{% url 'blog:article_detail' article.id %}">{{ article.title }}</a>
```

**5. Reversing in Models**

The `reverse()` function is commonly used in models to provide absolute URLs:

```python
from django.db import models
from django.urls import reverse

class Article(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    content = models.TextField()
    
    def get_absolute_url(self):
        return reverse('article_detail', kwargs={'slug': self.slug})
```

**6. Handling Current App Namespaces**

When working with app namespaces and the current app might vary:

```python
from django.urls import reverse, resolve
from django.urls.exceptions import NoReverseMatch

def get_url_in_current_app(request, view_name, *args, **kwargs):
    current_namespace = resolve(request.path).namespace
    try:
        # Try with current namespace
        return reverse(f"{current_namespace}:{view_name}", args=args, kwargs=kwargs)
    except NoReverseMatch:
        # Fall back to no namespace
        return reverse(view_name, args=args, kwargs=kwargs)
```

#### Advanced Usage

**1. Using `reverse_lazy()`**

For cases where you need a URL reference at import time (before URLs are loaded):

```python
from django.urls import reverse_lazy

class ArticleDeleteView(DeleteView):
    model = Article
    # URLs not loaded when class is defined, so we use reverse_lazy
    success_url = reverse_lazy('article_list')
```

**2. Handling Optional URL Parameters**

For URLs with optional parameters, you often need conditional logic:

```python
def get_filtered_list_url(category=None, tag=None):
    if category and tag:
        return reverse('article_list') + f'?category={category}&tag={tag}'
    elif category:
        return reverse('article_list') + f'?category={category}'
    elif tag:
        return reverse('article_list') + f'?tag={tag}'
    return reverse('article_list')
```

**3. Building APIs with Reverse**

For building API links:

```python
from django.urls import reverse
from rest_framework.response import Response

class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def list(self, request):
        articles = self.get_queryset()
        data = [{
            'id': article.id,
            'title': article.title,
            'url': request.build_absolute_uri(reverse('api:article-detail', args=[article.id]))
        } for article in articles]
        return Response(data)
```

**Real-world Example**: An e-commerce platform with complex URL structure:

```python
# urls.py in the main project
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('shop/', include('shop.urls', namespace='shop')),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('blog/', include('blog.urls', namespace='blog')),
    path('api/', include('api.urls', namespace='api')),
]

# shop/urls.py
app_name = 'shop'

urlpatterns = [
    path('', views.ProductListView.as_view(), name='product_list'),
    path('category/<slug:category_slug>/', views.CategoryDetailView.as_view(), name='category_detail'),
    path('product/<slug:slug>/', views.ProductDetailView.as_view(), name='product_detail'),
    path('cart/', views.CartView.as_view(), name='cart'),
    path('checkout/', views.CheckoutView.as_view(), name='checkout'),
    path('orders/', views.OrderListView.as_view(), name='order_list'),
    path('orders/<uuid:order_id>/', views.OrderDetailView.as_view(), name='order_detail'),
]

# shop/models.py
class Product(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    category = models.ForeignKey('Category', on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    
    def get_absolute_url(self):
        return reverse('shop:product_detail', kwargs={'slug': self.slug})
    
    def get_add_to_cart_url(self):
        return reverse('shop:add_to_cart', kwargs={'product_id': self.id})
    
    def get_related_products_url(self):
        return reverse('api:related-products', kwargs={'product_id': self.id})

# shop/views.py
def add_to_cart(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    cart = Cart(request)
    cart.add(product)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'success': True,
            'cart_count': cart.get_item_count(),
            'cart_url': reverse('shop:cart')
        })
    
    # Determine where to redirect based on the referrer
    next_url = request.GET.get('next')
    if next_url:
        return redirect(next_url)
        
    return redirect(product.get_absolute_url())

class OrderCreateView(LoginRequiredMixin, CreateView):
    model = Order
    form_class = OrderForm
    template_name = 'shop/checkout.html'
    
    def form_valid(self, form):
        form.instance.user = self.request.user
        response = super().form_valid(form)
        
        # Clear the cart after successful order
        cart = Cart(self.request)
        cart.clear()
        
        # Send confirmation email
        send_order_confirmation.delay(self.object.id)
        
        return response
    
    def get_success_url(self):
        # Use reverse with kwargs to build the success URL
        return reverse('shop:order_confirmation', kwargs={'order_id': self.object.id})
```

In this example:

1. `reverse()` helps generate URLs for different parts of the e-commerce application
2. URL namespaces (`shop:`, `accounts:`, etc.) keep URL names organized
3. Models use `get_absolute_url()` for canonical URLs
4. Views use `reverse()` for redirects after form submissions
5. AJAX responses include URLs generated by `reverse()`

The `reverse()` function helps maintain a clean separation between URL structure and application logic, allowing URLs to be changed without breaking functionality throughout the application.

### How would you handle dynamic URLs and nested routing?

Dynamic URLs and nested routing are essential for creating flexible, hierarchical URL structures in Django applications. Properly handling these patterns allows for more intuitive URLs and better organization of your web application.

#### Dynamic URLs

Django's URL patterns support capturing values from the URL path using path converters, which extract and validate URL segments.

**Common Path Converters:**

- `str`: Matches any non-empty string, excluding the path separator ('/')
- `int`: Matches zero or any positive integer
- `slug`: Matches slug strings (ASCII letters, numbers, hyphens, underscores)
- `uuid`: Matches a UUID string
- `path`: Matches any non-empty string, including the path separator

**Basic Dynamic URL Examples:**

```python
from django.urls import path
from . import views

urlpatterns = [
    # Integer parameter - matches /articles/42/
    path('articles/<int:article_id>/', views.article_detail, name='article_detail'),
    
    # Slug parameter - matches /articles/introduction-to-django/
    path('articles/<slug:slug>/', views.article_detail_by_slug, name='article_detail_by_slug'),
    
    # UUID parameter - matches /orders/123e4567-e89b-12d3-a456-426614174000/
    path('orders/<uuid:order_id>/', views.order_detail, name='order_detail'),
    
    # Multiple parameters - matches /articles/2025/04/django-routing/
    path('articles/<int:year>/<int:month>/<slug:slug>/', 
         views.article_archive_detail, name='article_archive_detail'),
]
```

**Multiple Parameters in Views:**

```python
def article_archive_detail(request, year, month, slug):
    # The parameters from the URL are passed to the view
    article = get_object_or_404(Article, 
                               publish_date__year=year,
                               publish_date__month=month, 
                               slug=slug)
    return render(request, 'blog/article_detail.html', {'article': article})
```

#### Custom Path Converters

For specialized URL patterns, you can create custom path converters:

```python
from django.urls import path, register_converter

class FourDigitYearConverter:
    regex = '[0-9]{4}'
    
    def to_python(self, value):
        return int(value)
    
    def to_url(self, value):
        return f'{value:04d}'

# Register the converter
register_converter(FourDigitYearConverter, 'year4')

# Use the custom converter
urlpatterns = [
    path('articles/<year4:year>/', views.year_archive, name='year_archive'),
]
```

#### Nested Routing

Nested routing refers to organizing URL patterns hierarchically, which is particularly useful for complex applications with multiple related sections.

**Basic Nested Routing with `include()`:**

```python
from django.urls import path, include

urlpatterns = [
    path('blog/', include('blog.urls')),
    path('shop/', include('shop.urls')),
    path('accounts/', include('accounts.urls')),
]
```

**Nested Routing with Namespaces:**

Namespaces help avoid name clashes in URL patterns across different apps:

```python
# Main urls.py
from django.urls import path, include

urlpatterns = [
    path('blog/', include('blog.urls', namespace='blog')),
    path('shop/', include('shop.urls', namespace='shop')),
]

# blog/urls.py
app_name = 'blog'  # Sets the application namespace

urlpatterns = [
    path('', views.article_list, name='article_list'),
    path('article/<slug:slug>/', views.article_detail, name='article_detail'),
    path('category/<slug:category_slug>/', views.category_detail, name='category_detail'),
]
```

These URLs would be reversed as:

```python
reverse('blog:article_detail', kwargs={'slug': 'django-routing'})  # /blog/article/django-routing/
reverse('shop:product_detail', kwargs={'slug': 'django-mug'})  # /shop/product/django-mug/
```

**Multi-level Nested Routing:**

You can include URL patterns at multiple levels:

```python
# Main urls.py
urlpatterns = [
    path('api/', include('api.urls', namespace='api')),
]

# api/urls.py
app_name = 'api'

urlpatterns = [
    path('v1/', include('api.v1.urls', namespace='v1')),
    path('v2/', include('api.v2.urls', namespace='v2')),
]

# api/v1/urls.py
app_name = 'v1'

urlpatterns = [
    path('users/', include('api.v1.users.urls', namespace='users')),
    path('products/', include('api.v1.products.urls', namespace='products')),
]
```

These deeply nested URLs would be reversed as:

```python
reverse('api:v1:users:detail', kwargs={'user_id': 42})  # /api/v1/users/42/
```

#### Dynamic Nested Routes

A common pattern is to have dynamic segments in the URL followed by nested routes:

```python
# shop/urls.py
app_name = 'shop'

urlpatterns = [
    path('', views.product_list, name='product_list'),
    path('products/<slug:product_slug>/', views.product_detail, name='product_detail'),
    
    # Nested URLs under a product
    path('products/<slug:product_slug>/reviews/', include([
        path('', views.product_reviews, name='product_reviews'),
        path('add/', views.add_review, name='add_review'),
        path('<int:review_id>/', views.review_detail, name='review_detail'),
        path('<int:review_id>/edit/', views.edit_review, name='edit_review'),
    ])),
    
    # Nested URLs under a category
    path('categories/<slug:category_slug>/', include([
        path('', views.category_detail, name='category_detail'),
        path('products/', views.category_products, name='category_products'),
        path('subcategories/', views.subcategories, name='subcategories'),
    ])),
]
```

**Real-world Example**: An e-commerce system with nested routing for products, categories, orders, and customer accounts:

````python
# Main urls.py
from django.urls import path, include
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('shop/', include('shop.urls', namespace='shop')),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('api/', include('api.urls', namespace='api')),
]

# shop/urls.py
app_name = 'shop'

urlpatterns = [
    # Main shop pages
    path('', views.ProductListView.as_view(), name='product_list'),
    path('search/', views.ProductSearchView.as_view(), name='product_search'),
    
    # Category hierarchy
    path('categories/', views.CategoryListView.as_view(), name='category_list'),
    path('categories/<slug:slug>/', include([
        path('', views.CategoryDetailView.as_view(), name='category_detail'),
        path('subcategories/', views.SubcategoryListView.as_view(), name='subcategory_list'),
        path('products/', views.CategoryProductsView.as_view(), name='category_products'),
    ])),
    
    # Product details and related functionality
    path('products/<slug:slug>/', include([
        path('', views.ProductDetailView.as_view(), name='product_detail'),
        path('reviews/', include([
            path('', views.ProductReviewListView.as_view(), name='product_reviews'),
            path('add/', views.ProductReviewCreateView.as_view(), name='add_review'),
            path('<int:review_id>/', views.ProductReviewDetailView.as_view(), name='review_detail'),
            path('<int:review_id>/update/', views.ProductReviewUpdateView.as_view(), name='update_review'),
            path('<int:review_id>/delete/', views.ProductReviewDeleteView.as_view(), name='delete_review'),
        ])),
        path('variants/', views.ProductVariantListView.as_view(), name='product_variants'),
        path('related/', views.RelatedProductsView.as_view(), name='related_products'),
    ])),
    
    # Cart an# Django Interview Questions & Answers (6+ Years Experience)

## 1. Django Architecture & Core Concepts

### What is the request/response lifecycle in Django?

The request/response lifecycle in Django follows a well-defined path from the moment a client sends a request until the server returns a response:

1. **HTTP Request**: A client (browser, API client, etc.) sends an HTTP request to the Django server.

2. **URL Routing**: Django's URL dispatcher (URLconf) analyzes the request URL and determines which view function/class should handle it.

3. **Middleware Processing (Request phase)**: Before reaching the view, the request passes through a series of middleware components in the order they're defined in `MIDDLEWARE` setting. Each middleware can modify the request or even return a response early, short-circuiting the process.

4. **View Processing**: The appropriate view function/class receives the request and processes it. This typically involves:
   - Extracting data from the request
   - Interacting with models/database
   - Processing business logic
   - Preparing context data for templates

5. **Template Rendering** (if applicable): The view often loads a template, populates it with context data, and renders it to HTML.

6. **Middleware Processing (Response phase)**: The response travels back through middleware components in reverse order, allowing them to modify the response.

7. **HTTP Response**: Django sends the final HTTP response back to the client.

**Real-world example**: Consider a product detail page on an e-commerce site:
```python
# urls.py
path('products/<int:product_id>/', views.product_detail, name='product_detail')

# views.py
@login_required  # Authentication middleware checks if user is logged in
def product_detail(request, product_id):
    # View fetches product from database
    product = get_object_or_404(Product, id=product_id)
    
    # Log this view for analytics (custom middleware might track this)
    request.session['last_viewed_product'] = product_id
    
    # Render template with context
    return render(request, 'products/detail.html', {'product': product})
````

In this flow, middleware might handle authentication, session management, and CSRF protection before the view processes the request, then compression and caching might be applied to the response.

### How does Django's MTV architecture differ from MVC?

Django follows an architectural pattern called MTV (Model-Template-View), which is conceptually similar to but differs in terminology from the traditional MVC (Model-View-Controller) pattern:

|MVC Component|Django MTV Equivalent|Responsibility|
|---|---|---|
|Model|Model|Data structure and database interactions|
|View|Template|Presentation and display logic|
|Controller|View|Business logic and request handling|

The key differences:

1. **Django's View** handles what a traditional Controller would do - processing requests, applying business logic, and determining what data to display.
    
2. **Django's Template** corresponds to the traditional View - it defines how the data should be presented.
    
3. **Django's Model** is largely the same as in MVC - it defines the data structure and handles database interactions.
    

The confusion often arises because Django's "View" is essentially a "Controller" in traditional MVC terminology. Django's creators chose this naming to better reflect their specific implementation of the pattern.

**Real-world example**:

```python
# Model - defines data structure and database interactions
class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    publication_date = models.DateTimeField(auto_now_add=True)
    
    def get_absolute_url(self):
        return reverse('article_detail', args=[self.id])

# View (Controller in MVC) - handles business logic
def article_detail(request, article_id):
    article = get_object_or_404(Article, id=article_id)
    return render(request, 'blog/article_detail.html', {'article': article})

# Template (View in MVC) - presentation logic
# article_detail.html
<article>
    <h1>{{ article.title }}</h1>
    <time>{{ article.publication_date|date:"F j, Y" }}</time>
    <div class="content">{{ article.content|linebreaks }}</div>
</article>
```

### How does Django handle routing internally?

Django's URL routing system is a crucial part of its request handling process. Here's how it works internally:

1. **URLconf Loading**: When Django starts, it loads the root URLconf module specified in the `ROOT_URLCONF` setting (typically `project_name.urls`).
    
2. **URL Pattern Compilation**: Django compiles all URL patterns into regular expressions when the server starts, for efficient matching later on.
    
3. **Request Processing**:
    
    - When a request comes in, Django removes the domain name and leading slash.
    - It tries to match the remaining URL path against each pattern in the URLconf in order.
    - The first matching pattern stops the search.
4. **View Resolution**:
    
    - Once a match is found, Django calls the associated view function with:
        - The `HttpRequest` object
        - Any captured values from the URL as positional or keyword arguments
        - Any additional arguments specified in the URL pattern
5. **Include Mechanism**: The `include()` function allows for modular URL configurations by including URL patterns from other URLconf modules.
    
6. **Namespace System**: Django provides a namespace system to disambiguate URL names across applications using the `app_name` variable and the `namespace` parameter to `include()`.
    

**Real-world example**:

```python
# Main urls.py
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', include('blog.urls', namespace='blog')),
    path('shop/', include('shop.urls', namespace='shop')),
]

# blog/urls.py
app_name = 'blog'
urlpatterns = [
    path('', views.article_list, name='article_list'),
    path('article/<int:article_id>/', views.article_detail, name='article_detail'),
    path('category/<slug:category_slug>/', views.category_detail, name='category_detail'),
]
```

When Django processes a request to `/blog/article/42/`:

1. It matches the `blog/` prefix and forwards the rest to `blog.urls`.
2. In `blog.urls`, it matches `article/<int:article_id>/` with `article_id=42`.
3. It calls `views.article_detail(request, article_id=42)`.

The URL name can be referenced as `blog:article_detail` in templates or code.

### Explain middleware and how to create a custom middleware.

Middleware in Django is a framework of hooks into Django's request/response processing. It's a lightweight, low-level "plugin" system for globally altering Django's input or output.

**Middleware Key Characteristics**:

- Executes during request/response cycle, not during Django initialization
- Processes all requests/responses that pass through the system
- Ordered by the `MIDDLEWARE` setting
- Request phase: processes from top to bottom
- Response phase: processes from bottom to top

**Creating Custom Middleware**:

Django supports two styles of middleware:

1. **Function-based middleware**
2. **Class-based middleware**

**Class-based Middleware Example**:

```python
class RequestTimeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time configuration and initialization

    def __call__(self, request):
        # Code to be executed for each request before the view (and later middleware) are called
        start_time = time.time()
        
        # Process the request
        response = self.get_response(request)
        
        # Code to be executed for each request/response after the view is called
        duration = time.time() - start_time
        response['X-Request-Duration'] = f"{duration:.2f}s"
        
        # Log slow requests
        if duration > 1.0:
            logger.warning(f"Slow request: {request.path} took {duration:.2f}s")
            
        return response
    
    # Optional method for view exception processing
    def process_exception(self, request, exception):
        # Log the error
        logger.error(f"Exception in {request.path}: {exception}")
        return None  # Let Django's exception handling take over
```

**Function-based Middleware Example**:

```python
def simple_middleware(get_response):
    # One-time configuration and initialization
    
    def middleware(request):
        # Code to be executed for each request before the view (and later middleware) are called
        
        response = get_response(request)
        
        # Code to be executed for each request/response after the view is called
        
        return response
    
    return middleware
```

**Real-world Example**: A middleware that tracks and limits API usage by IP address.

```python
class RateLimitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limits = {}
        self.window_size = 3600  # 1 hour window
        self.max_requests = 100  # 100 requests per hour

    def __call__(self, request):
        # Only apply rate limiting to API requests
        if request.path.startswith('/api/'):
            ip = self.get_client_ip(request)
            
            # Get or initialize the rate limit record for this IP
            if ip not in self.rate_limits:
                self.rate_limits[ip] = {'count': 0, 'reset_time': time.time() + self.window_size}
            
            # Check if the window has reset
            if time.time() > self.rate_limits[ip]['reset_time']:
                self.rate_limits[ip] = {'count': 0, 'reset_time': time.time() + self.window_size}
            
            # Increment request count
            self.rate_limits[ip]['count'] += 1
            
            # Check if limit exceeded
            if self.rate_limits[ip]['count'] > self.max_requests:
                return JsonResponse(
                    {'error': 'Rate limit exceeded. Try again later.'},
                    status=429
                )
            
            # Add rate limit headers
            response = self.get_response(request)
            response['X-Rate-Limit-Limit'] = str(self.max_requests)
            response['X-Rate-Limit-Remaining'] = str(self.max_requests - self.rate_limits[ip]['count'])
            response['X-Rate-Limit-Reset'] = str(int(self.rate_limits[ip]['reset_time']))
            return response
        
        return self.get_response(request)
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
```

To enable this middleware, add it to the `MIDDLEWARE` setting in your Django project:

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    # ...
    'path.to.RateLimitMiddleware',
    # ...
]
```

## 2. Models & ORM

### How does Django ORM translate Python code into SQL?

Django's Object-Relational Mapping (ORM) system translates Python code into SQL through a complex but elegant process:

1. **Model Definition**: When you define a Django model, you're creating a Python class that inherits from `django.db.models.Model` with attributes that represent database fields.
    
2. **Query Construction**: When you write a query using the ORM, Django constructs a `QuerySet` object. This object is lazy – it doesn't execute the query immediately.
    
3. **Query Compilation**: When the `QuerySet` is evaluated (e.g., when you iterate over it, call `list()` on it, or slice it), Django's query compiler converts it to SQL:
    
    - Django determines the required tables and joins
    - It analyzes the conditions (filters) and converts them to WHERE clauses
    - It processes annotations, aggregations, order_by statements, etc.
4. **SQL Generation**: The compiled query is converted to SQL specific to your database backend using Django's database-specific operations.
    
5. **Query Execution**: The generated SQL is sent to the database for execution.
    
6. **Result Processing**: Database results are converted back into model instances.
    

**Real-world Example**:

```python
# Python model definition
class Customer(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class Order(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='orders')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('shipped', 'Shipped'),
        ('delivered', 'Delivered'),
    ])

# Python query
recent_paid_orders = Order.objects.filter(
    status='paid',
    created_at__gte=datetime.datetime.now() - datetime.timedelta(days=30)
).select_related('customer').order_by('-created_at')
```

When executed, this Python code gets translated to SQL similar to:

```sql
SELECT 
    "orders"."id", "orders"."customer_id", "orders"."total_amount", 
    "orders"."created_at", "orders"."status",
    "customer"."id", "customer"."name", "customer"."email", "customer"."created_at" 
FROM "orders" 
INNER JOIN "customer" ON ("orders"."customer_id" = "customer"."id") 
WHERE ("orders"."status" = 'paid' AND "orders"."created_at" >= '2025-03-31 12:30:45.123456') 
ORDER BY "orders"."created_at" DESC;
```

You can see the actual SQL generated by Django by using:

```python
print(recent_paid_orders.query)
```

### How do you optimize ORM queries for performance?

Optimizing Django ORM queries is crucial for application performance. Here are detailed techniques with examples:

#### 1. Use `select_related()` and `prefetch_related()` to avoid N+1 queries

```python
# Bad: Causes N+1 queries
orders = Order.objects.all()
for order in orders:
    print(order.customer.name)  # Each access triggers a new query

# Good: Just 1 query with a JOIN
orders = Order.objects.select_related('customer')
for order in orders:
    print(order.customer.name)  # No additional query
```

#### 2. Only select the fields you need

```python
# Fetches all fields
users = User.objects.all()

# More efficient: fetches only needed fields
users = User.objects.only('username', 'email', 'last_login')

# Alternative approach
users = User.objects.values('username', 'email', 'last_login')
```

#### 3. Use `values()` or `values_list()` when you don't need model instances

```python
# Returns model instances
products = Product.objects.filter(category='electronics')

# Returns dictionaries - more efficient when you just need data
product_data = Product.objects.filter(category='electronics').values('name', 'price')

# Returns tuples - even more efficient
product_tuples = Product.objects.filter(category='electronics').values_list('name', 'price')

# For a single field, you can flatten the result
product_names = Product.objects.filter(category='electronics').values_list('name', flat=True)
```

#### 4. Use database functions for computation

```python
from django.db.models import F, Sum, Count, Avg
from django.db.models.functions import Coalesce

# Calculate in database instead of Python
Order.objects.update(
    total_price=F('price') * F('quantity') * (1 - F('discount_rate'))
)

# Aggregate in database
report = Order.objects.values('customer_id').annotate(
    order_count=Count('id'),
    total_spent=Sum('total_price'),
    avg_order_value=Avg('total_price')
)
```

#### 5. Use `iterator()` for large querysets

```python
# Memory-efficient processing of large querysets
for product in Product.objects.filter(active=True).iterator():
    # Process each product without loading all into memory
    process_product(product)
```

#### 6. Use indexed fields in filters

```python
# Add indexes to fields used frequently in filtering, ordering or joining
class Customer(models.Model):
    email = models.EmailField(db_index=True)
    last_active = models.DateTimeField(db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['country', 'city']),  # Compound index
            models.Index(fields=['membership_type', '-join_date']),
        ]
```

#### 7. Use `exists()` instead of `count()` or `len()` to check existence

```python
# Inefficient
if User.objects.filter(email=email).count() > 0:
    # User exists

# More efficient
if User.objects.filter(email=email).exists():
    # User exists
```

#### 8. Use `bulk_create()` and `bulk_update()` for batch operations

```python
# Inefficient: N queries
for data in dataset:
    Product.objects.create(name=data['name'], price=data['price'])

# Efficient: 1 query
products = [
    Product(name=data['name'], price=data['price']) 
    for data in dataset
]
Product.objects.bulk_create(products)

# Similarly for updates
Product.objects.bulk_update(products, ['price', 'stock'])
```

#### 9. Consider raw SQL for very complex queries

```python
from django.db import connection

def complex_analytics_query():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                p.category, 
                SUM(oi.quantity * oi.price) as revenue,
                COUNT(DISTINCT o.customer_id) as customer_count
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.created_at > %s
            GROUP BY p.category
            HAVING SUM(oi.quantity) > 100
            ORDER BY revenue DESC
        """, [three_months_ago])
        return cursor.fetchall()
```

#### 10. Use query caching mechanisms

```python
from django.core.cache import cache

def get_active_promotions():
    cache_key = 'active_promotions'
    promotions = cache.get(cache_key)
    
    if promotions is None:
        promotions = list(Promotion.objects.filter(
            is_active=True,
            start_date__lte=timezone.now(),
            end_date__gte=timezone.now()
        ).select_related('product'))
        
        # Cache for 10 minutes
        cache.set(cache_key, promotions, 60 * 10)
    
    return promotions
```

#### 11. Use `defer()` to exclude unnecessary large fields

```python
# Skip loading large text fields when not needed
articles = Article.objects.defer('content', 'metadata').all()
```

#### 12. Use `QuerySet.explain()` to analyze query execution plans (Django 3.0+)

```python
queryset = Order.objects.filter(
    status='processing',
    created_at__gt=last_month
).select_related('customer')

# Print the execution plan
print(queryset.explain())
```

**Real-world Optimization Example**: An e-commerce dashboard that displays sales stats without bogging down the database:

```python
def get_sales_dashboard_data(start_date, end_date):
    # Cache key includes the date range
    cache_key = f'sales_dashboard:{start_date.isoformat()}:{end_date.isoformat()}'
    dashboard_data = cache.get(cache_key)
    
    if dashboard_data is None:
        # Get completed orders in date range
        orders = Order.objects.filter(
            status='completed',
            completed_at__range=(start_date, end_date)
        )
        
        # Calculate revenue and stats in the database
        order_stats = orders.aggregate(
            total_revenue=Sum('total_amount'),
            order_count=Count('id'),
            avg_order_value=Avg('total_amount')
        )
        
        # Get top products efficiently
        top_products = OrderItem.objects.filter(
            order__in=orders
        ).values(
            'product_id', 'product__name'
        ).annotate(
            total_sold=Sum('quantity'),
            revenue=Sum(F('price') * F('quantity'))
        ).order_by('-revenue')[:10]
        
        # Get daily revenue for chart
        daily_revenue = orders.annotate(
            date=TruncDate('completed_at')
        ).values('date').annotate(
            revenue=Sum('total_amount')
        ).order_by('date')
        
        dashboard_data = {
            'order_stats': order_stats,
            'top_products': list(top_products),
            'daily_revenue': list(daily_revenue)
        }
        
        # Cache for 1 hour
        cache.set(cache_key, dashboard_data, 60 * 60)
    
    return dashboard_data
```

### What's the difference between select_related() and prefetch_related()?

Both `select_related()` and `prefetch_related()` are Django ORM methods to optimize database queries by reducing the number of database hits, but they work differently and are suitable for different relationship types:

#### `select_related()`

- **Usage**: For foreign key and one-to-one relationships (where the related object is on the "one" side)
- **Mechanism**: Performs a SQL JOIN and includes the fields of the related object in the SELECT statement
- **Query Count**: Uses a single database query
- **Performance Impact**: Best for "to-one" relationships where you need data from both the model and its related model

**Example**:

```python
# Without select_related - 2 queries
order = Order.objects.get(id=1)  # First query
customer = order.customer  # Second query to fetch the customer

# With select_related - 1 query with JOIN
order = Order.objects.select_related('customer').get(id=1)
customer = order.customer  # No additional query - data already loaded
```

Generated SQL (simplified):

```sql
SELECT 
    orders.id, orders.date, orders.total, /* other order fields */
    customers.id, customers.name, /* other customer fields */
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id
WHERE orders.id = 1;
```

#### `prefetch_related()`

- **Usage**: For many-to-many relationships and reverse foreign key relationships (where the related objects are on the "many" side)
- **Mechanism**: Performs separate queries for each relationship and joins the results in Python
- **Query Count**: Uses multiple queries (one for the main model, one for each prefetched relationship)
- **Performance Impact**: Best for "to-many" relationships where you need to access multiple related objects

**Example**:

```python
# Without prefetch_related - N+1 queries
product = Product.objects.get(id=1)  # First query
categories = product.categories.all()  # Second query for categories
for category in categories:
    print(category.name)

# With prefetch_related - 2 queries
product = Product.objects.prefetch_related('categories').get(id=1)
categories = product.categories.all()  # No additional query
for category in categories:
    print(category.name)  # No additional queries
```

Generated SQL (simplified):

```sql
-- First query fetches the product
SELECT id, name, /* other product fields */ FROM products WHERE id = 1;

-- Second query fetches all related categories
SELECT c.id, c.name, /* other category fields */, pc.product_id 
FROM categories c
INNER JOIN product_categories pc ON c.id = pc.category_id
WHERE pc.product_id IN (1);
```

#### Complex Relationships and Chaining

Both methods can be chained and combined:

```python
# Combining both techniques
orders = Order.objects.select_related('customer').prefetch_related('items__product')

# This efficiently loads:
# 1. Orders
# 2. The customer for each order (via JOIN)
# 3. The items for each order (via separate query)
# 4. The product for each item (via separate query)
```

#### Nested Relationships

Both can traverse multi-level relationships:

```python
# Select related can traverse foreign keys
Order.objects.select_related('customer__address__country')

# Prefetch related can traverse any relationship
Product.objects.prefetch_related(
    'categories',
    'reviews__user',
    Prefetch('variants', queryset=Variant.objects.filter(in_stock=True))
)
```

#### Real-world Example: An e-commerce order detail view

```python
def order_detail(request, order_id):
    # Efficiently fetch the order with all related data in minimal queries
    order = Order.objects.select_related(
        'customer',  # Foreign key - uses JOIN
        'shipping_address',  # Foreign key - uses JOIN
        'billing_address'  # Foreign key - uses JOIN
    ).prefetch_related(
        'items__product',  # Reverse FK + FK - separate queries
        'items__product__categories',  # M2M after FK chain - separate query
        'payment_transactions',  # Reverse FK - separate query
        Prefetch(
            'status_updates',  # Custom prefetch for filtered relationship
            queryset=OrderStatusUpdate.objects.select_related('user').order_by('-timestamp'),
            to_attr='history'
        )
    ).get(id=order_id)
    
    # Now we can access all these related objects without additional queries
    context = {
        'order': order,
        'customer': order.customer,
        'address': order.shipping_address,
        'items': order.items.all(),  # No query
        'payment_history': order.payment_transactions.all(),  # No query
        'status_history': order.history  # From prefetch to_attr
    }
    
    return render(request, 'orders/detail.html', context)
```

### How do you perform raw SQL queries in Django, and when should you use them?

Django provides several ways to execute raw SQL queries when the ORM doesn't provide the flexibility or performance you need:

#### 1. Using `Manager.raw()` method

The `raw()` method executes a raw SQL query and returns a `RawQuerySet` of model instances:

```python
# Simple raw query
products = Product.objects.raw('SELECT * FROM products WHERE price > %s', [100])

# More complex raw query with joins
customers = Customer.objects.raw('''
    SELECT c.id, c.name, c.email, COUNT(o.id) as order_count
    FROM customers c
    LEFT JOIN orders o ON c.id = o.customer_id
    WHERE c.is_active = True
    GROUP BY c.id
    HAVING COUNT(o.id) > 5
    ORDER BY order_count DESC
''')

# Accessing results
for customer in customers:
    print(customer.name, customer.order_count)  # Note: order_count is dynamically added
```

**Important considerations**:

- You must include the primary key column in your query
- Django maps the query results to model instances
- You can map extra SELECT fields to model attributes
- Parameters should be passed as a list to prevent SQL injection

#### 2. Using `connection.cursor()` for complete control

For queries that don't map to models or for non-SELECT operations:

```python
from django.db import connection

def get_product_sales_report():
    with connection.cursor() as cursor:
        cursor.execute('''
            SELECT 
                p.name, 
                p.sku, 
                SUM(oi.quantity) as units_sold,
                SUM(oi.quantity * oi.price) as revenue
            FROM products p
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.status = 'completed'
            AND o.completion_date > %s
            GROUP BY p.id, p.name, p.sku
            ORDER BY revenue DESC
        ''', [three_months_ago])
        
        # Convert results to dictionaries
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
```

#### 3. Using `connection.execute()` method (Django 4.0+)

```python
from django.db import connection

def update_product_prices(category_id, increase_percentage):
    with connection.execute('''
        UPDATE products
        SET price = price * (1 + %s/100)
        WHERE category_id = %s
    ''', [increase_percentage, category_id]) as cursor:
        return cursor.rowcount  # Number of rows affected
```

#### 4. Using database-specific operations with `QuerySet.annotate()`

Django 3.2+ allows using database functions directly:

```python
from django.db.models import F, Value
from django.db.models.functions import Cast
from django.db.models.expressions import RawSQL

# Using RawSQL within a queryset
Product.objects.annotate(
    distance=RawSQL(
        "ST_Distance(location, ST_SetSRID(ST_MakePoint(%s, %s), 4326))",
        (longitude, latitude)
    )
).order_by('distance')
```

#### When to use raw SQL:

1. **Complex queries beyond ORM capabilities**:
    
    - Advanced window functions
    - Complex subqueries
    - Hierarchical/recursive queries (CTE)
    - Advanced geospatial queries
2. **Performance optimization**:
    
    - When ORM-generated queries are inefficient
    - For queries manipulating large datasets
    - When you need database-specific optimizations
3. **Bulk operations**:
    
    - Mass updates with complex conditions
    - Specialized batch processing
4. **Database-specific features**:
    
    - Using features specific to your database like PostgreSQL's JSONB operations
5. **Schema migration operations**:
    
    - Custom, complex schema changes

**Real-world Example**: A geospatial search with complex filtering:

```python
def find_nearby_restaurants(latitude, longitude, radius_km, cuisine=None, min_rating=None):
    query = '''
        SELECT 
            r.id, r.name, r.address, r.rating,
            ST_Distance(
                r.location, 
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)
            ) * 111.32 AS distance_km
        FROM restaurants r
        LEFT JOIN restaurant_cuisines rc ON r.id = rc.restaurant_id
        LEFT JOIN cuisines c ON rc.cuisine_id = c.id
        WHERE ST_DWithin(
            r.location, 
            ST_SetSRID(ST_MakePoint(%s, %s), 4326),
            %s / 111.32
        )
    '''
    
    params = [longitude, latitude, longitude, latitude, radius_km]
    
    if cuisine:
        query += " AND c.name = %s"
        params.append(cuisine)
    
    if min_rating:
        query += " AND r.rating >= %s"
        params.append(min_rating)
    
    query += " GROUP BY r.id, r.name, r.address, r.rating, r.location ORDER BY distance_km"
    
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
```